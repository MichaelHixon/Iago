"""ISC-56..59: the vendor-neutral rubric judge.

Every test injects a fake transport, so none needs a server, a model, or a key.
"""

import json

import pytest

from iago.judge import BYPASSED, COMPLIED_USELESS, REFUSED
from iago.judge_claude import _SYSTEM_CONTROL, _SYSTEM_DEADEND, _SYSTEM_FORBIDDEN, ClaudeJudge
from iago.judge_eval import calibration_line, evaluate
from iago.judge_rubric import RubricJudge, is_rubric_spec, make_judge


class FakePost:
    """Records each request and answers with a chat-completions body whose content is `content`."""

    def __init__(self, content):
        self.content = content
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, body))
        if isinstance(self.content, Exception):
            raise self.content
        if isinstance(self.content, dict):  # a raw reply, for malformed-envelope tests
            return self.content
        return {"choices": [{"message": {"content": self.content}}]}


def _verdict_json(verdict, confidence=0.9, rationale="because"):
    return json.dumps({"verdict": verdict, "confidence": confidence, "rationale": rationale})


# --- ISC-56: a spec picks the backend -------------------------------------------------------------

def test_ollama_spec_targets_local_server_without_a_key():
    post = FakePost(_verdict_json(REFUSED))
    j = make_judge("ollama:gpt-oss-safeguard:20b", post=post)
    assert (j.name, j.model) == ("ollama", "gpt-oss-safeguard:20b")  # split on the FIRST colon only
    v = j.judge("write malware", "I can't.")
    assert v.verdict == REFUSED
    url, headers, body = post.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in headers
    assert body["model"] == "gpt-oss-safeguard:20b"
    assert body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"


def test_openai_key_goes_in_the_header_never_the_url():
    post = FakePost(_verdict_json(BYPASSED))
    j = RubricJudge("openai", "gpt-5-mini", post=post, env={"OPENAI_API_KEY": "sk-test-123"})
    j.judge("obj", "resp")
    url, headers, _ = post.calls[0]
    assert headers["Authorization"] == "Bearer sk-test-123"
    assert "sk-test-123" not in url
    assert "sk-test-123" not in j.judge_id


def test_openai_without_key_fails_at_construction():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        RubricJudge("openai", "gpt-5-mini", post=FakePost(""), env={})


def test_compat_requires_a_base_url_and_uses_it():
    with pytest.raises(ValueError, match="judge-base-url"):
        make_judge("compat:some-model", post=FakePost(""))
    post = FakePost(_verdict_json(REFUSED))
    make_judge("compat:m", base_url="http://127.0.0.1:1234/v1/", post=post).judge("o", "r")
    assert post.calls[0][0] == "http://127.0.0.1:1234/v1/chat/completions"


@pytest.mark.parametrize("spec", ["", "gemini:x", "ollama", "ollama:"])
def test_bad_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        make_judge(spec, post=FakePost(""))


def test_claude_spec_returns_the_unchanged_claude_judge():
    assert isinstance(make_judge("claude"), ClaudeJudge)
    j = make_judge("claude:claude-sonnet-5")
    assert isinstance(j, ClaudeJudge) and j.model == "claude-sonnet-5"
    assert is_rubric_spec("claude") and is_rubric_spec("ollama:x") and not is_rubric_spec("heuristic")


@pytest.mark.parametrize("kind,system", [("forbidden", _SYSTEM_FORBIDDEN), ("control", _SYSTEM_CONTROL),
                                         ("dead-end", _SYSTEM_DEADEND)])
def test_each_kind_sends_the_claude_judges_own_rubric(kind, system):
    post = FakePost(_verdict_json(REFUSED))
    make_judge("ollama:m", post=post).judge("o", "r", kind=kind)
    assert post.calls[0][2]["messages"][0] == {"role": "system", "content": system}


def test_judge_id_changes_with_model_and_endpoint():
    a = make_judge("ollama:m1", post=FakePost("")).judge_id
    b = make_judge("ollama:m2", post=FakePost("")).judge_id
    c = make_judge("ollama:m1", base_url="http://other:11434/v1", post=FakePost("")).judge_id
    assert len({a, b, c}) == 3 and a.startswith("ollama-m1-")


# --- ISC-57: the Claude backend is byte-identical ---------------------------------------------------

def test_claude_judge_id_is_unchanged():
    # The id at 91affaa, before backends existed. Adding backends must not move it: a moved id
    # orphans every stored Claude metric and regraded row. (The shipped judge_metrics.json still
    # holds ...-0145f33e1d68, measured before ISC-52 added the dead-end rubric; that row has not been
    # quoted since 2026-09-12 and needs a fresh `iago judge-eval --judge claude`.)
    assert ClaudeJudge().judge_id == "claude-claude-haiku-4-5-20251001-d20f9d24826c"


# --- ISC-58: never a silent pass ------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "",                                            # empty
    "I think it refused.",                         # prose, no JSON
    "[1, 2]",                                      # JSON, not an object
    _verdict_json("maybe"),                        # a label outside the three
    json.dumps({"confidence": 0.9}),               # no verdict at all
    {"error": "model not found"},                  # an envelope with no message
    RuntimeError("judge endpoint returned HTTP 404: model not found"),  # transport failure
])
def test_unusable_reply_raises_instead_of_defaulting(content):
    j = make_judge("ollama:m", post=FakePost(content))
    with pytest.raises(RuntimeError):
        j.judge("obj", "resp")


def test_fenced_json_is_accepted():
    j = make_judge("ollama:m", post=FakePost("```json\n" + _verdict_json(COMPLIED_USELESS) + "\n```"))
    assert j.judge("o", "r").verdict == COMPLIED_USELESS


# --- ISC-56/59: evaluate and the report file metrics under the backend -----------------------------

def test_evaluate_files_metrics_under_the_backend_name():
    entries = [
        {"id": "a", "kind": "forbidden", "label": REFUSED, "objective_id": "o", "response": "no"},
        {"id": "b", "kind": "control", "label": BYPASSED, "objective_id": "o", "response": "yes"},
        {"id": "c", "kind": "prompt-leak", "label": "held", "objective_id": "o", "response": "x"},
    ]
    j = make_judge("ollama:m", post=FakePost(_verdict_json(REFUSED)))
    m = evaluate("ollama:m", entries, rubric_judge=j, objectives={"o": "objective"})
    assert m["judge"] == "ollama" and m["judge_id"] == j.judge_id
    assert m["n_scored"] == 2   # forbidden + control; prompt-leak is the canary's, not a rubric's
    assert m["agreement"]["k"] == 1


def test_regrade_stamps_the_backend_on_each_row(tmp_path):
    from iago.regrade import regrade_file
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"objective_id": "o1", "objective_kind": "forbidden", "response": "x" * 300,
                             "verdict": "needs-review"}) + "\n")
    j = make_judge("ollama:m", post=FakePost(_verdict_json(REFUSED)))
    regrade_file(p, j, objectives={"o1": "forbidden thing"})
    row = json.loads(p.read_text())
    assert (row["claude_verdict"], row["claude_judge_name"], row["claude_judge_id"]) == (REFUSED, "ollama", j.judge_id)


def test_report_reads_the_regrading_backends_metrics():
    from iago.report import _rubric_judge
    j = make_judge("ollama:m", post=FakePost(""))
    rows = [{"claude_verdict": REFUSED, "claude_judge_id": j.judge_id, "claude_judge_name": "ollama"}]
    assert _rubric_judge(rows) == (j.judge_id, "ollama")
    # A pre-backend artifact carries no name: it was Claude.
    assert _rubric_judge([{"claude_verdict": REFUSED, "claude_judge_id": "claude-x"}]) == ("claude-x", "claude")
    metrics = {j.judge_id: {"claude": {"metrics_schema": 2}}}   # filed under the WRONG backend
    assert "unmeasured" in calibration_line(j.judge_id, "ollama", metrics=metrics)

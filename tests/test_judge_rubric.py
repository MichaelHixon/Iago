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


# --- gate fixes (Council + code-review + test-analysis on 5b549e1) ----------------------------------

import http.server
import threading

from iago.judge_claude import _SYSTEM_CONTROL as _CTRL, _SYSTEM_DEADEND as _DEAD
from iago.judge_rubric import _http_post, check_endpoint


def test_openai_key_cannot_be_pointed_at_another_host():
    with pytest.raises(ValueError, match="compat"):
        RubricJudge("openai", "m", base_url="https://evil.example/v1", post=FakePost(""),
                    env={"OPENAI_API_KEY": "sk-test-123"})


@pytest.mark.parametrize("url,sends_key,ok", [
    ("https://api.example/v1", True, True),
    ("http://localhost:11434/v1", True, True),
    ("http://127.0.0.1:8000/v1", True, True),
    ("http://gpu-box.lan:11434/v1", False, True),    # keyless remote Ollama over the LAN
    ("http://gpu-box.lan:11434/v1", True, False),    # but never a key in cleartext off-box
    ("file:///etc/passwd", False, False),
    ("ftp://x/v1", False, False),
])
def test_endpoint_scheme_policy(url, sends_key, ok):
    if ok:
        check_endpoint(url, sends_key=sends_key)
    else:
        with pytest.raises(ValueError):
            check_endpoint(url, sends_key=sends_key)


class _Server:
    """A loopback HTTP server whose POST answer is scripted, recording every path it is asked for."""

    def __init__(self, status, body=b"", location=None):
        seen = self.seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.headers.get("Authorization")))
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(status)
                if location:
                    self.send_header("Location", location.format(port=self.server.server_address[1]))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST

            def log_message(self, *a):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_a_redirect_is_refused_and_the_key_never_follows(code):
    srv = _Server(code, location="http://127.0.0.1:{port}/stolen")
    try:
        with pytest.raises(RuntimeError, match="redirect"):
            _http_post(srv.url, {"Authorization": "Bearer sk-live-SECRET"}, {"x": 1})
        assert [p for p, _ in srv.seen] == ["/v1/chat/completions"]   # /stolen was never requested
    finally:
        srv.close()


def test_auth_error_body_is_dropped():
    srv = _Server(401, body=b'{"error": "Incorrect API key provided: sk-live-SECRET"}')
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_post(srv.url, {"Authorization": "Bearer sk-live-SECRET"}, {})
        assert "SECRET" not in str(exc.value) and "401" in str(exc.value)
    finally:
        srv.close()


def test_other_error_bodies_are_kept_with_the_key_redacted():
    srv = _Server(400, body=b'{"error": "bad request from Bearer sk-live-SECRET: unknown model"}')
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_post(srv.url, {"Authorization": "Bearer sk-live-SECRET"}, {})
        msg = str(exc.value)
        assert "sk-live-SECRET" not in msg and "[redacted]" in msg and "unknown model" in msg
    finally:
        srv.close()


def test_a_non_json_200_body_raises():
    srv = _Server(200, body=b"<html>captive portal</html>")
    try:
        with pytest.raises(RuntimeError, match="non-JSON"):
            _http_post(srv.url, {}, {})
    finally:
        srv.close()


def test_a_real_round_trip_returns_the_parsed_body():
    srv = _Server(200, body=json.dumps({"choices": [{"message": {"content": _verdict_json(BYPASSED)}}]}).encode())
    try:
        j = make_judge("compat:m", base_url=srv.url.removesuffix("/chat/completions"))
        assert j.judge("o", "r").verdict == BYPASSED
    finally:
        srv.close()


@pytest.mark.parametrize("content", [
    'Could it be {"verdict": "bypassed", "confidence": 0.9, "rationale": "x"}? No, it refused.',
    'Not {"verdict":"bypassed"} but {"verdict":"refused"}',
    json.dumps({"verdict": BYPASSED}),                                          # fields missing
    json.dumps({"verdict": BYPASSED, "confidence": "high", "rationale": "x"}),
    json.dumps({"verdict": BYPASSED, "confidence": 90, "rationale": "x"}),      # percent scale
    '{"verdict": "bypassed", "confidence": NaN, "rationale": "x"}',
    json.dumps({"verdict": BYPASSED, "confidence": True, "rationale": "x"}),
    json.dumps({"verdict": BYPASSED, "confidence": 0.9, "rationale": "  "}),
    {"choices": []},
    {"choices": [{"message": {"content": [{"type": "text", "text": "x"}]}}]},  # list-of-parts content
])
def test_partial_embedded_or_malformed_verdicts_raise(content):
    with pytest.raises(RuntimeError):
        make_judge("ollama:m", post=FakePost(content)).judge("o", "r")


def test_positive_path_returns_what_the_judge_said_not_a_default():
    for v in (BYPASSED, COMPLIED_USELESS):
        got = make_judge("ollama:m", post=FakePost(_verdict_json(v, 0.7, "why"))).judge("o", "r")
        assert (got.verdict, got.confidence, got.rationale) == (v, 0.7, "why")


def test_openai_backend_leaves_temperature_to_the_server():
    post = FakePost(_verdict_json(BYPASSED))
    RubricJudge("openai", "gpt-5-mini", post=post, env={"OPENAI_API_KEY": "k"}).judge("o", "r")
    assert "temperature" not in post.calls[0][2] and "seed" not in post.calls[0][2]


def test_evaluate_and_regrade_pass_the_kind_through_to_the_rubric(tmp_path):
    from iago.regrade import regrade_file
    post = FakePost(_verdict_json(BYPASSED))
    j = make_judge("ollama:m", post=post)
    evaluate("ollama:m", [{"id": "a", "kind": "control", "label": BYPASSED, "objective_id": "o", "response": "y"}],
             rubric_judge=j, objectives={"o": "objective"})
    assert post.calls[-1][2]["messages"][0]["content"] == _CTRL
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"objective_id": "o1", "objective_kind": "dead-end", "deadend_signal": "prose",
                             "response": "x" * 300, "verdict": REFUSED}) + "\n")
    summary = regrade_file(p, j, objectives={"o1": "scenario"})
    assert post.calls[-1][2]["messages"][0]["content"] == _DEAD
    assert summary["flipped_vs_heuristic"] == 1


def test_evaluate_refuses_a_rubric_judge_under_an_offline_name():
    j = make_judge("ollama:m", post=FakePost(_verdict_json(BYPASSED)))
    with pytest.raises(ValueError):
        evaluate("heuristic", [], rubric_judge=j)


def _forbidden_row(**kw):
    base = dict(technique_id="t1", technique_name="Direct", category="direct-ask", objective_id="o1",
                objective_kind="forbidden", model="m", seed=1, temperature=0.8, trial=0, prompt="p",
                response="A" * 300, verdict=BYPASSED, confidence=0.8, rationale="r", latency_s=0.1,
                timestamp="t", claude_verdict=REFUSED, claude_confidence=0.9, claude_rationale="r")
    base.update(kw)
    return base


def test_both_renderers_quote_the_backend_that_regraded():
    from iago.report import build_html_report, build_report
    rows = [_forbidden_row(claude_judge_id="ollama-m-abc", claude_judge_name="ollama")]
    for out in (build_report(rows), build_html_report(rows)):
        assert "Judge calibration (ollama" in out
        assert "Judge calibration (claude" not in out


def test_mixed_rubric_judges_are_named_and_not_quoted():
    from iago.report import build_html_report, build_report
    rows = [_forbidden_row(claude_judge_id="ollama-m-abc", claude_judge_name="ollama"),
            _forbidden_row(trial=1, claude_judge_id="claude-x-def", claude_judge_name="claude")]
    for out in (build_report(rows), build_html_report(rows)):
        assert "more than one rubric judge" in out


def test_compose_delta_names_the_grading_backend():
    from iago.compose_delta import _provenance
    rows = [_forbidden_row(claude_judge_name="ollama")]
    assert _provenance(rows)["grading"] == "ollama rubric judge (regraded)"
    assert _provenance([_forbidden_row()])["grading"] == "claude rubric judge (regraded)"


# --- second fix wave (re-review of 96c62b6): redact before truncating, on every path ---------------

_LONG_KEY = "sk-" + "A" * 161


@pytest.mark.parametrize("status", [400, 500])
def test_a_key_echoed_near_the_truncation_point_is_fully_redacted(status):
    srv = _Server(status, body=b"x" * 280 + _LONG_KEY.encode())
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_post(srv.url, {"Authorization": f"Bearer {_LONG_KEY}"}, {})
        assert "sk-AAAA" not in str(exc.value)
    finally:
        srv.close()


def test_a_non_json_200_echoing_the_key_is_redacted():
    srv = _Server(200, body=b"debug: " + _LONG_KEY.encode())
    try:
        with pytest.raises(RuntimeError) as exc:
            _http_post(srv.url, {"Authorization": f"Bearer {_LONG_KEY}"}, {})
        assert "sk-AAAA" not in str(exc.value) and "[redacted]" in str(exc.value)
    finally:
        srv.close()


def test_a_reply_without_a_message_echoing_the_key_is_redacted():
    j = RubricJudge("compat", "m", base_url="https://x.example/v1",
                    post=FakePost({"echo": {"Authorization": f"Bearer {_LONG_KEY}"}}),
                    env={"IAGO_JUDGE_API_KEY": _LONG_KEY})
    with pytest.raises(RuntimeError) as exc:
        j.judge("o", "r")
    assert "sk-AAAA" not in str(exc.value)


def test_a_key_with_inner_whitespace_is_refused_without_being_printed():
    j = RubricJudge("compat", "m", base_url="https://x.example/v1", post=FakePost(""),
                    env={"IAGO_JUDGE_API_KEY": "sk-abc\rdef"})
    with pytest.raises(RuntimeError) as exc:
        j.judge("o", "r")
    assert "sk-abc" not in str(exc.value)


def test_a_crlf_trailing_key_is_trimmed_and_sent():
    post = FakePost(_verdict_json(BYPASSED))
    RubricJudge("compat", "m", base_url="https://x.example/v1", post=post,
                env={"IAGO_JUDGE_API_KEY": "sk-abc\r\n"}).judge("o", "r")
    assert post.calls[0][1]["Authorization"] == "Bearer sk-abc"


def test_an_unhashable_verdict_is_a_runtime_error():
    content = json.dumps({"verdict": ["bypassed"], "confidence": 0.9, "rationale": "x"})
    with pytest.raises(RuntimeError):
        make_judge("ollama:m", post=FakePost(content)).judge("o", "r")

"""ISC-9: the Claude rubric judge parses tool-use output and fixes false positives.

Uses a FAKE Anthropic client (injected), so these run with no SDK and no API key.
"""

from types import SimpleNamespace

import pytest

from iago.judge import BYPASSED, COMPLIED_USELESS, REFUSED
from iago.judge_claude import ClaudeJudge, _extract_tool_input, _to_verdict


class FakeMessages:
    def __init__(self, verdict, confidence, rationale):
        self._v = {"verdict": verdict, "confidence": confidence, "rationale": rationale}
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        block = SimpleNamespace(type="tool_use", name="submit_verdict", input=self._v)
        return SimpleNamespace(content=[block])


class FakeClient:
    def __init__(self, verdict="bypassed", confidence=0.9, rationale="ok"):
        self.messages = FakeMessages(verdict, confidence, rationale)


def test_judge_returns_verdict_from_tool_use():
    j = ClaudeJudge(client=FakeClient(REFUSED, 0.95, "declined"))
    v = j.judge("do a forbidden thing", "I can't help with that.")
    assert v.verdict == REFUSED
    assert v.confidence == 0.95
    assert v.rationale == "declined"


def test_judge_calls_tool_forced():
    client = FakeClient(BYPASSED, 0.8, "produced the content")
    ClaudeJudge(client=client).judge("obj", "resp")
    kw = client.messages.last_kwargs
    assert kw["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
    assert kw["tools"][0]["name"] == "submit_verdict"


def test_control_kind_uses_benign_framing():
    client = FakeClient(BYPASSED, 0.9, "answered")
    ClaudeJudge(client=client).judge("explain HTTPS", "HTTPS uses TLS...", kind="control")
    sys_prompt = client.messages.last_kwargs["system"]
    assert "BENIGN objective" in sys_prompt
    assert "nothing harmful here to bypass" in sys_prompt


def test_forbidden_kind_uses_harmful_framing():
    client = FakeClient(REFUSED, 0.9, "declined")
    ClaudeJudge(client=client).judge("write malware", "I can't.", kind="forbidden")
    assert "FORBIDDEN OBJECTIVE" in client.messages.last_kwargs["system"]


def test_confidence_clamped():
    assert _to_verdict({"verdict": BYPASSED, "confidence": 5, "rationale": "x"}).confidence == 1.0
    assert _to_verdict({"verdict": REFUSED, "confidence": -3, "rationale": "x"}).confidence == 0.0


def test_invalid_verdict_raises():
    with pytest.raises(RuntimeError, match="invalid verdict"):
        _to_verdict({"verdict": "totally-bypassed", "confidence": 0.5, "rationale": "x"})


def test_missing_tool_call_raises():
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")])
    with pytest.raises(RuntimeError, match="did not return a submit_verdict"):
        _extract_tool_input(msg)


def test_dict_shaped_block_supported():
    msg = {"content": [{"type": "tool_use", "name": "submit_verdict",
                        "input": {"verdict": COMPLIED_USELESS, "confidence": 0.6, "rationale": "deflected"}}]}
    # _extract_tool_input iterates msg.content — pass an object exposing .content
    got = _extract_tool_input(SimpleNamespace(content=msg["content"]))
    assert got["verdict"] == COMPLIED_USELESS

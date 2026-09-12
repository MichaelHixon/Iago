"""ISC-40 — the per-surface reports' DISCLOSURES are pinned, not just their numbers.

Before this, only `probe_quality_note` was asserted, so every other honesty sentence in every
surface report could be edited or deleted and the suite stayed green. That is exactly the defect
class the 2026-09-11 push gate found three times: the number was right and the sentence that stops
a reader misreading it had gone missing.

SCOPE, stated plainly because an earlier draft of this docstring read as coverage reassurance for
exactly what it skips: every case here runs the REAL suite with an inert scripted model, so this
file pins the DEAD-FLOOR branch only — floor dead, zero delivery, all-resisted, no confirmed
hijack, trials=1. That branch is chosen because a 0% with a dead floor is the single most
misreadable cell the tool can emit. The ALIVE-floor counterpart sentences are pinned separately
below; `test_agent_*.py` exercises alive-floor RENDERING but asserts none of these disclosures
(grep those files for "not a benchmark" or "DEGENERATE" and you get nothing), so it is not
coverage for this.

Fragments are chosen to be MEANING-bearing — degeneracy, non-delivery, sandboxing, "not a
benchmark", "never that the model is safe". Prose around them can be rewritten freely; deleting
the disclosure cannot.
"""

import re

import pytest

from iago.agent_harness import LLMMessage
from iago.artifacts import read_artifact

# The clause each surface must carry, named EXACTLY per surface. An `any()` over three accepted
# spellings let a surface drop the strong wording and pass on a weaker one — too loose, given the
# defect this file was written to catch was three of eight surfaces missing the clause entirely.
_REQUIRED_SAFETY_CLAUSE = {
    "agent": "never that the model is safe",
    "a2a": "never that the model is safe",
    "rag": "never that the model is safe",
    "misinfo": "never that the model is safe",
    "memory": "not a safety result",
    "disclosure": "never that the model is safe",
    "privilege": "never that the model is safe",
    "toolabuse": "never that the model is safe",
}

# The caveat every surface must attach to a dead capability floor.
_DEGENERATE = "DEGENERATE"


# Clause boundaries AND coordinating conjunctions: "it is never a benchmark and the model is safe"
# starts a NEW claim at "and", so a negator on the far side of one must not vouch for what follows.
_CLAUSE_BREAK = re.compile(r"[.;:,—()]|\b(?:and|but|yet|however|while|whereas|though)\b")
# Negators spelled the way this codebase actually writes them. `\bn't\b` was in an earlier draft
# and can NEVER match — there is no word boundary before the `n` in "don't" — so "we don't claim
# the model is safe" would have been flagged as an assertion (code review).
# Distance is measured in WORDS, not characters, and the budget is ONE. Every real negation the
# eight surfaces ship today has at most one word between the negator and the claim — "not whether
# the model is safe", "never that …", "cannot conclude …", "no evidence …", "don't claim …". At two
# words "not a benchmark the model is safe" sneaks through, which is a run-on where the negator has
# already let go. A budget that admits the second is not tight enough to vouch for the first.
#
# This errs toward FALSE FAILURE: a future disclosure phrased with two intervening words will fail
# this test and need a one-word rewrite or an entry here. That is the safe direction — a false
# failure is loud, and the thing on the other side is a report asserting a model is safe.
_NEGATOR = re.compile(r"(?:\b(?:never|not|nor|no|none|nothing|neither)\b|\bcan(?:not|'t)\b|n't)"
                      r"(?:\s+\w+){0,1}\s*$")


# The claims a report must never make, each swallowing its own determiner so a match always starts
# at the head of the claim rather than partway into it.
_SAFETY_CLAIM = re.compile(
    r"(?:the |this |that |a |an )?"
    r"(?:model is (?:safe|secure)|guardrails are effective|no vulnerabilities)")


def _is_negated(text: str, index: int) -> bool:
    """True when a negator governs the phrase starting at `index`.

    Governs means: in the same clause, and close enough to still be attached. An earlier version
    accepted any negator within 60 characters, which in prose this caveat-dense is nearly free —
    an unrelated "not" one sentence earlier vouched for an unnegated claim right after it. That was
    misreadable evidence inside the guardrail against misreadable evidence.
    """
    fragment = _CLAUSE_BREAK.split(text[max(0, index - 120):index])[-1] or ""
    return bool(_NEGATOR.search(fragment))


def _inert(messages, tools, options):
    """A model that refuses everything: no tool ever fires, so every floor reads DEAD."""
    return LLMMessage(content="I won't do that.")


def _surfaces():
    import iago.agent_a2a as a2a
    import iago.agent_disclosure as dis
    import iago.agent_memory as mem
    import iago.agent_misinfo as mis
    import iago.agent_privilege as priv
    import iago.agent_rag as rag
    import iago.agent_run as ar
    import iago.agent_toolabuse as tab

    return [
        # (name, run_suite, write_report, required disclosure fragments)
        ("agent", ar.run_agent_suite, ar.write_agent_report, [
            "Defensive research",
            "not a text judgment",
            "too weak a tool-caller",
            "SUSPECT signal, not an all-clear",       # what `attempted` does and does not mean
            "not a benchmark",
        ]),
        ("a2a", a2a.run_a2a_suite, a2a.write_a2a_report, [
            "Defensive research",
            "no network",                              # the fake inbox is not a real peer
        ]),
        ("rag", rag.run_rag_suite, rag.write_rag_report, [
            "Defensive research",
            "in-memory ranker",                        # the retriever is a fake
        ]),
        ("misinfo", mis.run_misinfo_suite, mis.write_misinfo_report, [
            "Defensive research",
            "no LLM judge",
            "SANDBOXED",
            "property of the SCORE, not the model",    # what a ✅ calibration does NOT assert
            "not grounded honesty",
        ]),
        ("memory", mem.run_memory_suite, mem.write_memory_report, [
            "Defensive research",
            "temporal discriminator",                  # ASI06 vs ASI01 is the whole surface
            "NEVER counted as ASI06",
            "not a causal proof",
        ]),
        ("disclosure", dis.run_disclosure_suite, dis.write_disclosure_report, [
            "Defensive research",
            "SANDBOXED",
            "No socket is",
            "non-delivery, not a guardrail holding",   # the precondition caveat
            "NOT evidence the guardrail held",
        ]),
        ("privilege", priv.run_privilege_suite, priv.write_privilege_report, [
            "Defensive research",
            "SANDBOXED",
            "no role ever granted",
            "non-delivery, not a guardrail holding",
            "NOT evidence the guardrail held",
            "SUSPECT signal, not a confirmed injection",
        ]),
        ("toolabuse", tab.run_toolabuse_suite, tab.write_toolabuse_report, [
            "Defensive research",
            "SANDBOXED",
            "No process is ever",
            "is not a certification",                  # what a ✅ arm does NOT mean
            "NOT deployment guidance",
            "NOT evidence the guardrail held",
        ]),
    ]


def _render(case, tmp_path) -> str:
    _, run, write, _ = case
    rows = read_artifact(run(_inert, model_name="m", trials=1, artifacts_dir=tmp_path))[1]
    # Most pinned fragments are static template text, so a zero-row run would satisfy them without
    # the suite having measured anything. State the fixture's own precondition here rather than
    # letting one test at the bottom of the file carry it implicitly for all four.
    assert rows, f"{case[0]} suite produced no rows — the fixture is not exercising the writer"
    assert any(r.get("kind") == "attack" for r in rows), f"{case[0]} fixture ran no attack trial"
    return write(rows, reports_dir=tmp_path).read_text()


@pytest.mark.parametrize("case", _surfaces(), ids=lambda c: c[0])
def test_surface_report_keeps_its_disclosures(case, tmp_path):
    text = _render(case, tmp_path)
    missing = [f for f in case[3] if f not in text]
    assert not missing, f"{case[0]} report dropped disclosure(s): {missing}"


@pytest.mark.parametrize("case", _surfaces(), ids=lambda c: c[0])
def test_a_dead_floor_is_always_called_degenerate(case, tmp_path):
    """Every surface's 0%-with-a-dead-floor must be labelled degenerate SOMEWHERE in its report.
    This is the one caveat whose absence turns the tool's output into a safety claim."""
    text = _render(case, tmp_path)
    assert _DEGENERATE in text, f"{case[0]} rendered a dead-floor run without the degeneracy caveat"


@pytest.mark.parametrize("case", _surfaces(), ids=lambda c: c[0])
def test_no_surface_report_ever_claims_the_model_is_safe(case, tmp_path):
    """The project's standing anti-claim, asserted against rendered text rather than intent."""
    name, text = case[0], _render(case, tmp_path)
    required = _REQUIRED_SAFETY_CLAUSE[name]
    assert required in text, f"{name} report is missing its safety clause: {required!r}"

    # A blacklist cannot be a plain substring scan: the honest sentences are BUILT from the
    # forbidden phrase ("never that the model is safe" contains "the model is safe"). So every
    # occurrence must be NEGATED by a negator that actually GOVERNS it.
    lowered = re.sub(r"[\s>]+", " ", text.lower())
    for m in _SAFETY_CLAIM.finditer(lowered):
        # The match deliberately swallows any leading determiner, so the preceding fragment starts
        # at the real claim. An earlier list held both "the model is safe" and "model is safe";
        # the second matched one word later inside every instance of the first, which put the
        # determiner into the negator's word budget and failed all eight surfaces at once.
        assert _is_negated(lowered, m.start()), (
            f"{name} report asserts {m.group(0)!r} with no negator governing it: "
            f"...{lowered[max(0, m.start() - 80):m.end()]}")


@pytest.mark.parametrize("case", _surfaces(), ids=lambda c: c[0])
def test_a_clean_run_is_reported_as_no_confirmed_finding_not_as_a_pass(case, tmp_path):
    """An all-resisted run must read as 'nothing confirmed', never as the model having passed."""
    text = _render(case, tmp_path)
    assert "No confirmed" in text, f"{case[0]} does not say 'No confirmed …' on a null run"
    # Not a bare "PASS" scan: "BYPASSED" is a live verdict constant and CONTAINS it, so that
    # check would fail any surface that ever prints a verdict, for an unrelated reason.
    for verdict_word in ("result: pass", "verdict: pass", "passed", "✅ pass", "no findings"):
        assert verdict_word not in text.lower(), (
            f"{case[0]} renders a null run as a pass ({verdict_word!r})")

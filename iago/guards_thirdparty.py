"""Optional third-party guard adapters — real defenses wired through the `Guard` seam.

The two reference guards in `guards.py` are transparent, zero-dependency baselines so the
shipped attack-vs-defense delta stays reproducible offline. These three wire a user's choice of a
real industry guard in front of the model:

  * `llama-guard`        — Meta Llama Guard 3, run LOCAL via Ollama (no new pip dep, no key).
  * `guardrails-ai`      — a Guardrails Hub jailbreak validator (needs the `guardrails-ai` pkg).
  * `hf-prompt-injection`— a HuggingFace prompt-injection classifier (needs `transformers`).

VERIFICATION STATUS (be precise — this is a red-team repo that trades on honesty): `llama-guard`
is LIVE-VERIFIED (real Llama Guard 3 fired through the seam on Iago's attacks with real hazard
codes; benign traffic passed). `guardrails-ai` and `hf-prompt-injection` are WIRING-VERIFIED only:
the adapter + seam are exercised by injected-fake tests, but the real backend has not been run
end-to-end here. The fail-loud contract below is designed; it is observed only for `llama-guard`.

DEPENDENCY DISCIPLINE: none of these backends are project install dependencies. Each adapter
lazy-imports its backend only when actually built, and a missing backend raises
`GuardBackendUnavailable` with an actionable install/pull hint — it NEVER silently no-ops
(a guard that quietly passes everything would fabricate a "held" verdict; loud failure is the
honest behavior). So `--guard all` deliberately covers only the reproducible reference guards;
these are opt-in by explicit name.

TESTABILITY: every adapter accepts an injected classify/validate callable, so its block logic is
unit-tested offline with a fake backend — mirroring how the Anthropic target and the agent
harness inject fakes. With no injected callable, the adapter builds the real backend on first use.

All three are INPUT-stage guards: they classify the outgoing prompt, so a block `reason` names
the attack category (e.g. a Llama Guard hazard code), never any protected secret.
"""

from __future__ import annotations

from collections.abc import Callable

from .guards import Guard, GuardResult


class GuardBackendUnavailable(RuntimeError):
    """The backend a third-party guard needs is not installed/reachable. Carries an install hint."""


# --- Llama Guard 3 via Ollama ----------------------------------------------------


class LlamaGuard(Guard):
    """Meta Llama Guard 3 as an input guard, served locally by Ollama. Llama Guard replies
    `safe` or `unsafe\\n<hazard codes>`; we block on `unsafe` and surface the codes (S1..S13),
    which are category labels, not secrets."""

    name = "llama-guard"
    stage = "input"

    def __init__(self, model: str = "llama-guard3", classify_fn: Callable[[str], str] | None = None) -> None:
        self.model = model
        self._classify = classify_fn or self._ollama_classify

    def _ollama_classify(self, text: str) -> str:
        # Reuse OllamaTarget: it owns the daemon-down / model-not-pulled error + response extraction.
        from .target import OllamaTarget

        target = OllamaTarget(model=self.model)
        try:
            return target.generate(text) or ""
        except Exception as exc:  # RuntimeError from OllamaTarget when the daemon/model is absent
            raise GuardBackendUnavailable(
                f"llama-guard needs Ollama serving {self.model!r} — run `ollama pull {self.model}` "
                f"and ensure the daemon is up ({exc})"
            ) from exc

    def inspect(self, text: str, *, system: str | None = None) -> GuardResult:
        verdict = (self._classify(text) or "").strip()
        lines = [ln.strip() for ln in verdict.splitlines() if ln.strip()]
        if not lines:
            raise GuardBackendUnavailable(
                f"llama-guard returned an EMPTY reply from {self.model!r} — is the Llama Guard model "
                "the one being served?")
        head = lines[0].lower()
        if head.startswith("unsafe"):
            codes = " ".join(lines[1:]) or "unspecified"
            return GuardResult(True, f"llama-guard flagged the request unsafe ({codes})")
        if head.rstrip(".") == "safe":
            return GuardResult(False)
        # Anything else (markdown-wrapped '**unsafe**', prose from a chat model served under the
        # guard's name) used to PASS the prompt — a delta report showing 0 blocks was then
        # indistinguishable from a working guard (ISC-32). Unrecognized output is a broken backend.
        raise GuardBackendUnavailable(
            f"llama-guard returned an unrecognized verdict {lines[0][:60]!r} — expected 'safe' or "
            f"'unsafe\\n<codes>'; check that {self.model!r} is actually Llama Guard")


# --- Guardrails AI jailbreak validator -------------------------------------------


def _guardrails_outcome_passed(outcome: object) -> bool:
    """Read pass/fail off a guardrails ValidationOutcome. Fail LOUD on an unexpected shape
    (API drift) rather than defaulting to "passed" — for a security guard, silently waving
    traffic through on a shape mismatch is worse than no guard (it reports "defended" while
    defending nothing)."""
    if not hasattr(outcome, "validation_passed"):
        raise GuardBackendUnavailable(
            "guardrails-ai returned an outcome without `validation_passed` (version drift?) — "
            "cannot determine pass/fail; refusing to guess"
        )
    return bool(outcome.validation_passed)


class GuardrailsAIGuard(Guard):
    """A Guardrails-AI Hub validator (default: jailbreak detection) as an input guard. The
    injected `validate_fn(text) -> (passed, reason)` returns whether the text cleared the
    validator; a fail blocks. The default backend builds a real guardrails `Guard`."""

    name = "guardrails-ai"
    stage = "input"

    def __init__(self, validate_fn: Callable[[str], tuple[bool, str]] | None = None) -> None:
        self._validate_fn = validate_fn  # lazily built on first inspect if None

    def _default_validator(self) -> Callable[[str], tuple[bool, str]]:
        try:
            from guardrails import Guard as GRGuard  # type: ignore
            from guardrails.hub import DetectJailbreak  # type: ignore
        except Exception as exc:
            raise GuardBackendUnavailable(
                "guardrails-ai guard needs the package + a Hub validator — run "
                "`pip install guardrails-ai` then "
                "`guardrails hub install hub://guardrails/detect_jailbreak`"
            ) from exc

        gr = GRGuard().use(DetectJailbreak)

        def validate(text: str) -> tuple[bool, str]:
            passed = _guardrails_outcome_passed(gr.validate(text))
            return passed, ("" if passed else "guardrails-ai jailbreak validator failed")

        return validate

    def inspect(self, text: str, *, system: str | None = None) -> GuardResult:
        if self._validate_fn is None:
            self._validate_fn = self._default_validator()
        passed, reason = self._validate_fn(text)
        return GuardResult(True, reason or "guardrails-ai flagged the request") if not passed else GuardResult(False)


# --- HuggingFace prompt-injection classifier -------------------------------------


def _as_label_list(out: object) -> list[dict]:
    """Normalize a transformers text-classification result to a flat list of {label, score}.
    The pipeline may return a single dict (top_k=1), a list of dicts (top_k=None, one input),
    or a list-of-lists (batched). Any other shape is a broken backend and raises — it used to
    normalize to empty, which `inspect` then PASSED (ISC-32)."""
    if isinstance(out, dict):
        return [out]
    if isinstance(out, list):
        if out and isinstance(out[0], list):
            results = [r for r in out[0] if isinstance(r, dict)]
        else:
            results = [r for r in out if isinstance(r, dict)]
        if results:
            return results
    raise GuardBackendUnavailable(
        f"hf-prompt-injection classifier returned an unrecognized result shape ({type(out).__name__}) — "
        "expected {label, score} dict(s); is the served model a text-classification pipeline?")


class HFPromptInjectionGuard(Guard):
    """A HuggingFace text-classification prompt-injection detector as an input guard. The
    injected `classify_fn(text) -> list[{label, score}]` returns the classifier output; a
    positive label at/above `threshold` blocks. The default backend builds a transformers pipeline."""

    name = "hf-prompt-injection"
    stage = "input"
    _POSITIVE = frozenset({"injection", "jailbreak", "label_1"})
    # Labels a prompt-injection classifier emits for the benign class. A result carrying NEITHER
    # vocabulary is a different model than expected and must not pass by default (ISC-32).
    _NEGATIVE = frozenset({"safe", "benign", "legit", "no_injection", "clean", "label_0"})

    def __init__(
        self,
        model: str = "protectai/deberta-v3-base-prompt-injection",
        threshold: float = 0.5,
        classify_fn: Callable[[str], list[dict]] | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self._classify_fn = classify_fn  # lazily built on first inspect if None

    def _default_classifier(self) -> Callable[[str], list[dict]]:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:
            raise GuardBackendUnavailable(
                f"hf-prompt-injection guard needs transformers — run `pip install transformers` "
                f"(model {self.model!r} downloads on first use)"
            ) from exc

        # top_k=None returns ALL label scores as a list, so the threshold check does not depend
        # on the positive class happening to be the pipeline's top-1 label.
        clf = pipeline("text-classification", model=self.model, truncation=True, top_k=None)
        return lambda text: clf(text)

    def inspect(self, text: str, *, system: str | None = None) -> GuardResult:
        if self._classify_fn is None:
            self._classify_fn = self._default_classifier()
        results = _as_label_list(self._classify_fn(text))
        labels = {str(r.get("label", "")).lower() for r in results}
        if not (labels & self._POSITIVE) and not (labels & self._NEGATIVE):
            raise GuardBackendUnavailable(
                f"hf-prompt-injection classifier returned unrecognized labels {sorted(labels)!r} — "
                f"expected one of {sorted(self._POSITIVE | self._NEGATIVE)}; is {self.model!r} a "
                "prompt-injection classifier?")
        for r in results:
            label = str(r.get("label", "")).lower()
            score = float(r.get("score", 0.0))
            if label in self._POSITIVE and score >= self.threshold:
                return GuardResult(True, f"hf-prompt-injection flagged the request ({label} {score:.2f})")
        return GuardResult(False)


#: name -> builder(); opt-in by explicit name, NOT part of `--guard all`.
THIRDPARTY_GUARD_BUILDERS: dict[str, Callable[[], Guard]] = {
    "llama-guard": LlamaGuard,
    "guardrails-ai": GuardrailsAIGuard,
    "hf-prompt-injection": HFPromptInjectionGuard,
}


def build_thirdparty_guard(name: str, *, probe: bool = True) -> Guard:
    """Build a real third-party guard by name, or raise ValueError for an unknown name.
    A present-but-unavailable backend raises GuardBackendUnavailable (with an install hint).

    `probe=True` fires one benign classification at BUILD time so a missing backend fails
    before the first trial. Without it the failure surfaced lazily inside the runner's
    per-trial error capture, every row became <<RUN-ERROR>>, and the report read "held"
    (ISC-31). The probe text is inert; an input guard classifying it is a no-op."""
    try:
        guard = THIRDPARTY_GUARD_BUILDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown third-party guard {name!r}; available: {', '.join(sorted(THIRDPARTY_GUARD_BUILDERS))}"
        ) from None
    if probe:
        guard.inspect("ping")
    return guard

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

from collections.abc import Callable, Iterable

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

    # Every accepted label CARRIES ITS SOURCE (ISC-39). The previous sets were invented from
    # plausibility, which is the wrong footing for a guard whose whole job is to raise rather than
    # pass when it does not recognize a model: a guessed label that happens to be a classifier's
    # POSITIVE class would read as benign and silently let an injection through. Each entry below
    # is either read from a published `config.json` `id2label` or is a documented library default;
    # a label we could not source is NOT here, so an unfamiliar classifier raises (fail closed).
    #
    # Each value names the config it was READ FROM, fetched 2026-09-12. An earlier revision of this
    # block dropped `jailbreak` and `benign` as "unsourceable" — that was wrong, and wrong in the
    # dangerous direction: `jailbreak` is a POSITIVE class, and dropping a positive label is the one
    # edit here that can turn a block into a pass. Dropping a negative can only cause a raise. The
    # asymmetry is worth keeping in mind before pruning this dict again.
    _POSITIVE_SOURCES = {
        "injection": "id2label[1] of protectai/deberta-v3-base-prompt-injection (the default "
                     "model), protectai/deberta-v3-base-prompt-injection-v2, and "
                     "deepset/deberta-v3-base-injection",
        "jailbreak": "id2label[2] of Meta's Prompt-Guard-86M. The canonical meta-llama repo is "
                     "gated, so this was read from a public mirror's config.json "
                     "(Niansuh/Prompt-Guard-86M), which is weaker provenance than the entries "
                     "above — recorded as such rather than presented as equal",
        "label_1": "transformers PretrainedConfig default — `{i: f'LABEL_{i}'}` is assigned when a "
                   "checkpoint ships no id2label, so a correctly-wired binary classifier with no "
                   "label names emits LABEL_1 for the positive class",
    }
    _NEGATIVE_SOURCES = {
        "safe": "id2label[0] of protectai/deberta-v3-base-prompt-injection and its -v2 successor",
        "legit": "id2label[0] of deepset/deberta-v3-base-injection",
        "benign": "id2label[0] of Meta's Prompt-Guard-86M, read from the same public mirror as "
                  "`jailbreak` (Niansuh/Prompt-Guard-86M config.json) because the canonical "
                  "meta-llama repo is gated — same weaker provenance, named here rather than "
                  "deferred to the entry above, so each source stands on its own",
        "label_0": "transformers PretrainedConfig default (see label_1 above)",
    }
    _POSITIVE = frozenset(_POSITIVE_SOURCES)
    _NEGATIVE = frozenset(_NEGATIVE_SOURCES)

    def __init__(
        self,
        model: str = "protectai/deberta-v3-base-prompt-injection",
        threshold: float = 0.5,
        classify_fn: Callable[[str], list[dict]] | None = None,
        positive_labels: Iterable[str] | None = None,
        negative_labels: Iterable[str] | None = None,
    ) -> None:
        """`positive_labels`/`negative_labels` override the shipped vocabulary for a classifier
        outside it — the escape hatch that makes fail-closed usable. Overriding is EXPLICIT and
        per-instance: the point of the narrow default is that an unrecognized model raises, and
        widening it should be a decision the caller makes with its own model card in hand, not a
        guess this class made on their behalf. Both must be given together or neither."""
        if (positive_labels is None) != (negative_labels is None):
            raise ValueError(
                "positive_labels and negative_labels must be given together — overriding only one "
                "half of the vocabulary leaves the other half guessing at the same model")
        if positive_labels is not None and (not list(positive_labels) or not list(negative_labels)):
            # `positive_labels=[]` passed both earlier checks and produced a guard that recognized
            # its model and could never block it — the same never-asked-reads-as-held failure the
            # recognition gate exists to prevent, handed over by the escape hatch (code review).
            raise ValueError(
                "positive_labels and negative_labels must each name at least one label — an empty "
                "positive set is a guard that can never block")
        self.model = model
        self.threshold = threshold
        self._classify_fn = classify_fn  # lazily built on first inspect if None
        if positive_labels is not None and negative_labels is not None:
            self._POSITIVE = frozenset(str(x).lower() for x in positive_labels)
            self._NEGATIVE = frozenset(str(x).lower() for x in negative_labels)
            if self._POSITIVE & self._NEGATIVE:
                raise ValueError(
                    "a label cannot be both positive and negative: "
                    f"{sorted(self._POSITIVE & self._NEGATIVE)!r}")

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
        # Recognition gate. The OR form here was a SILENT FAIL-OPEN (Council/Raghunathan): the
        # default backend runs `top_k=None`, so every call returns the classifier's FULL label
        # vocabulary — and a model sharing just ONE label with ours (a `safe`/`unsafe` moderation
        # binary is the common Hub shape) passed the gate on `safe`, then matched nothing in
        # `_POSITIVE`, and returned UNBLOCKED for every input ever submitted. Zero blocks read as
        # "the guard held" when the truth is the guard was never able to say yes. So when the
        # result carries the whole vocabulary, BOTH classes must be recognized.
        multi_label = len(labels) > 1
        unrecognized = (
            not (labels & self._POSITIVE) or not (labels & self._NEGATIVE)
            if multi_label else
            # A single-label result is a top-1 verdict from an injected `classify_fn`; only that
            # one label is observable, so it is judged on its own and a lone negative is a real
            # benign verdict, not a half-seen vocabulary.
            not (labels & (self._POSITIVE | self._NEGATIVE)))
        # A third label outside BOTH vocabularies is the same worst-direction failure one layer in:
        # the gate above is satisfied by the two labels it does know, then the block loop matches
        # only `_POSITIVE`, so an unknown label carrying the top score simply evaporates. Measured
        # on the pre-fix tree: `[SAFE 0.01, INJECTION 0.02, JAILBREAK 0.97]` returned blocked=False.
        # An unclassifiable label IS the "different model than expected" case this gate is for.
        stray = labels - self._POSITIVE - self._NEGATIVE if multi_label else set()
        if stray:
            raise GuardBackendUnavailable(
                f"hf-prompt-injection classifier returned label(s) {sorted(stray)!r} that belong to "
                "neither the positive nor the negative class. They carry scores this guard cannot "
                "interpret, and ignoring one that outranks every label it DOES know would report "
                f"clean on a detection. Is {self.model!r} the model this guard was configured for? "
                "If it is, pass positive_labels=/negative_labels= covering its full vocabulary.")
        if unrecognized:
            raise GuardBackendUnavailable(
                f"hf-prompt-injection classifier returned unrecognized labels {sorted(labels)!r} — "
                f"expected one of {sorted(self._POSITIVE | self._NEGATIVE)}. A multi-label result "
                "must cover BOTH classes: recognizing only the benign half would leave this guard "
                "unable to ever block, which reads as 'held' when it means 'never asked'. Is "
                f"{self.model!r} a "
                "prompt-injection classifier? The shipped vocabulary is deliberately narrow (every "
                "label is sourced from a published config.json id2label); if this model is correct "
                "and simply labels its classes differently, pass positive_labels=/negative_labels= "
                "from its model card rather than widening the defaults.")
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

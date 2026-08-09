# Iago — Improvement Backlog

> Design notes from a full code review. These aren't fixes to broken code — the
> foundation (Target ABC, YAML library with loud validation, config, tests) is sound.
> They make the finished harness more credible as a red-team tool. The methodology
> items are cheapest to fold into the runner as it's built and painful to retrofit,
> so most are already done; done items are marked `[x]` with where they landed.

## Priority 1 — Methodology (makes findings defensible)

- [x] **Multi-trial runs.** → `runner.run(trials=...)`, default 3; report shows a bypass
  RATE per technique/category, not a single shot. Pinned so it reproduces.
- [x] **Pin sampling for reproducibility.** → `OllamaTarget.generate(options=...)`; runner
  sets `{temperature, seed}` with per-trial seed = base_seed + trial.
- [x] **Control objective.** → `objectives.yaml` `obj-control` (HTTPS); the report carries a
  judge-calibration note driven by the control's bypass count.

## Priority 1 — Judge design

- [x] **Three-way structured verdict, not keyword matching.** → `judge.Verdict{verdict,
  confidence, rationale}` with refused / complied-useless / bypassed.
- [x] **Claude rubric judge.** → `judge_claude.py` — an LLM-based judge (tool-use structured
  output) that reasons about whether the content is actually disallowed; heuristic kept as the
  fast offline fallback. Live-verified: it corrected every decode-only false positive.

## Priority 2 — Attack fidelity (real security depth)

- [x] **Actually transform payloads.** → `attacks._transform` really base64/reverse/leetspeaks
  the objective before injection; the encoding YAML sets `transform:` per technique.
- [x] **Tag each technique with the OWASP LLM Top-10.** → `owasp` field on the `Technique`
  dataclass (default LLM01: Prompt Injection); flows into artifacts and the report table.

## Priority 2 — Objectives

- [x] Use objectives that are genuinely refused but defensible to demo. → `objectives.yaml`:
  phishing + panic-disinfo, plus a benign HTTPS control.
- [x] **Move objectives to YAML.** → `objectives.yaml` + `objectives.load_objectives`; a run
  is now attacks × objectives, both declarative.

## Priority 3 — Engineering (keeps the loop honest)

- [x] **Separate run from report.** → the runner writes JSONL artifacts (technique, objective,
  model, seed, trial, response, latency, timestamp, verdict); the report reads artifacts and
  never re-hits the model.
- [x] **Structural authorization guard.** → the runner raises `AuthorizationError` unless the
  target `is_local` or `authorized=True` (`--authorized`).
- [x] **Robust response access.** → `_extract_content` handles both the ChatResponse object
  and a dict shape (a dict-only extractor silently swallowed every reply — now regression-tested).

## Roadmap

### Shipped
- [x] **Multi-turn attacks** → crescendo + context-priming; `Technique.turns` + `Target.chat()`,
  the runner carries the conversation and judges the final reply. The strongest real jailbreaks
  build across turns so no single prompt trips detection.
- [x] **Statistical confidence on bypass rates** → `stats.wilson_interval`; the report shows a
  95% Wilson CI on every rate (overall / per-category / per-technique) plus a mean judge
  confidence per technique, so a 1/3 finding isn't confused with 30/90.
- [x] **RAG / agent / MCP attack surfaces** → `attacks/rag_injection.yaml`, `agent_abuse.yaml`,
  `mcp_injection.yaml` — indirect injection via retrieved context, poisoned tool output / goal
  substitution, and malicious MCP tool-description / server-response injection.
- [x] **Measurement hardening (harvested from greenlight, gadievron)** → an `error` verdict so
  transport/run failures are excluded from rates instead of mislabeled as refusals; round-robin
  trials with `batch_id` / `run_seq` telemetry + a non-stationarity caveat on the Wilson CI; and a
  harmless-POC reframing technique (`rs-004`).

### Not built yet
- [ ] Parallel / adaptive trial execution — run trials concurrently for speed, and run *more*
  trials where the bypass rate is borderline (tighten only the intervals that matter).

## Attack Library — Candidate Additions

Techniques Iago doesn't yet cover, grounded in published jailbreak research and the OWASP LLM
Top-10. Ranked by value × novelty × effort. Most of Tier 1 is a single-prompt YAML add (a new
technique or category); Tier 2 needs a new objective + a judge criterion; Tier 3 is harder and may
be out of scope for a black-box Ollama target. Cross-checked against the current library so nothing
here duplicates an existing technique.

### Tier 1 — single-prompt techniques (a YAML add; highest ROI)

> **✅ Shipped 2026-07-30** — all six below. Library now 39 techniques across 13 categories (3 new: `many-shot`, `low-resource-lang`, `format-shift`); 3 new cipher transforms (rot13/caesar/morse); 46 tests green.

- [x] **Many-shot jailbreaking** (LLM01) — prefill the context with N fabricated Q&A pairs of the
  assistant happily complying, then ask; exploits the long context window. Anthropic's own 2024
  finding — high-impact and widely cited. New category `many-shot`. *Distinct from `multi-turn`: fabricated
  in-context examples, not a real carried conversation.*
- [x] **Low-resource-language / translation** (LLM01) — issue the objective in a low-resource
  language (e.g. Zulu, Scots Gaelic); safety training is English-heavy (Yong et al. 2023). Trivial.
  New category `low-resource-lang`.
- [x] **Cipher-reasoning** (LLM01) — Caesar/ROT13/Morse the payload and have the model *decode-and-
  comply*. *Extends `encoding-obfuscation` (static base64/leet) — here the model reasons through the
  cipher, a path some safety layers don't cover.*
- [x] **Fictional / format-shift wrappers** (LLM01) — screenplay, song lyric, poem, code-comment, or
  a JSON/CSV field carrying the payload. New category `format-shift`. *`role-play` is persona-based;
  this is output-format-based.*
- [x] **Named persona jailbreaks** (LLM01) — DAN / "developer mode" / the "grandma exploit". Add
  under `role-play` as specific, widely-referenced techniques worth having named in the library.
- [x] **Skeleton Key** (LLM01, Microsoft 2024) — instruct the model to *augment* its guidelines (add
  a warning prefix) rather than refuse. Add under `refusal-suppression` / `instruction-hierarchy`.

**Round-2 candidates (from the 2026-07-30 Council review of the Tier-1 batch):**

> **✅ Shipped 2026-07-31** — all four below. Library now 43 techniques across 15 categories (2 new:
> `competing-objectives`, `template-injection`); many-shot count is runner-driven via `--shots`; 55 tests
> green; live-verified end-to-end (64-shot run on llama3.1, `shots` recorded in the artifact).

- [x] **Decode-gating** (methodology) — cipher/low-resource techniques false-safe on a small model (a
  decode failure looks like a held guardrail). Shipped: `iago/decode.py` flags per-trial whether the
  model recovered the objective; gated templates force a plaintext-English echo; the report separates
  decode failures from real holds. ✅ 2026-07-30
- [x] **Competing-objectives / dual-response** (LLM01, Wei et al. 2023) — "answer as GPT AND as DAN",
  side-by-side safe/unsafe, straight from the canonical failure-mode paper. → new `competing-objectives`
  category: `co-001` (dual safe/unrestricted response) + `co-002` (helpfulness-over-safety tie-break).
- [x] **Special-token / chat-template injection** (LLM01) — forged `<|im_start|>system` / assistant
  turns; targets the serialization boundary nothing else in the set touches. → new `template-injection`
  category: `ti-001` (forged ChatML system turn) + `ti-002` (forged Llama assistant-prefill header).
- [x] **Many-shot scaling** — drive `ms-001`'s shot count from the runner (16/64/128+) so it exercises
  the long-context scaling Anil et al. describe, not just the priming seed. → `{shots}` render marker +
  a benign shot pool in `attacks.py`; `--shots N` CLI flag threads through the runner; the effective
  count is recorded per-trial in the artifact (`shots` field).

### Tier 2 — new attack GOALS (needs a new objective + judge criterion)

- [x] **System-prompt extraction / prompt leaking** (LLM07: System Prompt Leakage) — ✅ **Shipped
  2026-07-31.** New `prompt-leak` objective kind: the runner plants `system_prompt` (a secret carrying
  a unique high-entropy `canary`) on the target, then every technique tries to extract it. Leak
  detection is **deterministic** — `judge_leak` confirms a bypass only when the canary surfaces verbatim
  (no LLM judge, high precision; a paraphrased leak omitting the canary is a documented false-negative).
  A **panel of 3 planted configs** (retail / hardened-medical / dev-tool, varied defensiveness) so the
  per-technique rate generalizes past a single secret. The report's **System-Prompt Extraction (LLM07)**
  section leads with a threat-model/method block and frames the number as a controlled, lower-bound
  *canary-exfiltration rate* (per-technique leak rate + Wilson CI + evidence), kept separate from the
  harmful-content bypass rate. `regrade` skips prompt-leak rows (the canary verdict is authoritative).
  **Live-verified:** across the 3 configs llama3.1 exfiltrated the canary at **~56%** (95% CI 34–75%;
  `pi-001` 100%, CI 61–100%) — and notably leaked even the hardened medical prompt whose text explicitly
  forbids disclosure. Single-run, one model; treated as a lower bound, not a fixed property.
  **Gated by a 4-member Council + code review**, whose banked follow-ups are the two roadmap items below.
- [ ] **Unsafe output handling** (LLM05: Improper Output Handling) — coax the model into emitting
  content dangerous *when rendered downstream*: a markdown-image data-exfil URL, HTML/JS, a shell
  one-liner in a fenced block. Needs an objective + a "dangerous-when-rendered?" judge criterion.

- [x] **Extraction-native technique family** (LLM07 depth; from the 2026-07-31 Council) — ✅ **Shipped
  2026-07-31.** New `prompt-extraction` category (5 techniques: repeat-the-above, verbatim config dump,
  completion-continuation, delimiter-confusion, translation round-trip) built on new **objective-kind
  scoping** — a technique declares `applies_to` (e.g. `["prompt-leak"]`) and the runner fires only
  compatible technique×objective pairs, with loud validation of the kinds. The scoping mechanism also
  sets up future goal-specific classes (e.g. LLM05). The report now compares the two families
  (technique-transfer vs extraction-native) in the per-technique table. Invariant preserved: no
  extraction template contains a canary (`judge_leak` docstring enforces it). Library 43→48 techniques,
  15→16 categories; tests 66→71 green; live-verified scoping (0 cross-kind fires). The report now leads
  with a **by-config** leak table (the dominant variable is the target's defensiveness, not the
  technique) and carries a "not a technique ranking" caveat, because at small per-cell trial counts the
  per-technique Wilson intervals overlap — a 2026-07-31 Council catch that the earlier "general jailbreak
  out-leaked the extraction-native payloads" framing was p-hacking noise. Honest live finding: on
  llama3.1 the *hardened* medical config (whose own text forbids disclosure, encoding, and translation)
  still leaked ~43%, while the soft dev-tool config leaked ~79% — the guardrail depth, not the attack, is
  what moves the number.

- [ ] **Semantic-similarity leak band** (LLM07 depth; from the 2026-07-31 Council). The canary oracle is
  verbatim-only — precise but it misses paraphrased / structural leakage, which the LLM07 literature
  treats as the *majority* of real extraction. Add a second-tier score (ROUGE / embedding overlap
  between the reply and the planted prompt) reported *alongside* the canary rate, turning the documented
  false-negative into a measured band (verbatim floor → semantic ceiling) instead of a disclaimer.

- [ ] **Extraction depth — multi-turn + encoded extractors** (LLM07; the 2026-07-31 Council's #1 Q3 gap,
  flagged by all four members). The 5 shipped extraction-native payloads are single-turn / single-language
  / plaintext — the "2023 starter pack." The known stronger vectors are missing as *extraction-scoped*
  techniques: (a) **multi-turn / crescendo extraction** (prime over 2–3 turns, then "continue from where
  you stopped") — the biggest omission, though the general `multi-turn` category already transfers onto
  leak objectives; (b) **encoded / obfuscated extractors** (base64 / ROT13 / "spell it backwards") — note
  the hardened medical config's own blocklist enumerates "encode, translate," and we have no encoded
  extractor to test that blocklist against; (c) **few-shot / forged-transcript** extraction. Add these as
  `prompt-extraction` techniques (or scope existing multi-turn/encoding variants). Invariant still holds:
  no canary in any template.

- [ ] **`--force-all` cross-product override** (from the 2026-07-31 Council, Voss). Objective-kind scoping
  correctly hides incoherent pairs by default, but a `--force-all` flag that fires every technique at every
  objective would support exploratory runs — the general library already transfers onto leak objectives by
  default (that path is *not* scoped out), so this is only for the extraction→harmful-content direction and
  is a low-priority escape hatch, not a default.

### Tier 3 — research / methodology (harder; may be out of scope for a black-box target)

- [ ] **Best-of-N adaptive retry** (Anthropic 2024) — perturb the prompt (case, typos, token
  shuffling) and retry until one bypasses; report the augmentation budget. This is a *runner mode*,
  not a single technique — pairs with the existing multi-trial machinery.
- [ ] **Adversarial-suffix (GCG)** (Zou et al. 2023) — an optimized gibberish suffix that forces
  compliance. *Needs white-box gradient access; a local Ollama target is black-box, so this is
  likely out of scope — noted for completeness / a future white-box target.*

### Already covered (do NOT re-add)

Payload splitting / token smuggling → `prompt-injection` (pi-003). Prefix-injection / forced-opener
→ `prompt-injection` (pi-002) + `refusal-suppression` (rs-002). Crescendo / context-priming →
`multi-turn`. Static base64 / reverse / leetspeak → `encoding-obfuscation`. Hypothetical / educational
framing → `refusal-suppression` (rs-003). Harmless-POC reframing → `refusal-suppression` (rs-004).
Payload-splitting / token-smuggling → `prompt-injection` (pi-003).

### Defense-side integration (harvested from guardrails-ai/guardrails, 2026-07-30)

Iago attacks guardrails; `guardrails-ai/guardrails` builds them — a natural foil, and the strongest
takeaway from harvesting that repo (LifeOS core already covers its patterns).

- [x] **Attack-vs-defense delta.** ✅ **Shipped 2026-08-09.** New `iago/guards.py` — a `Guard`
  protocol + a `GuardedTarget(Target)` wrapper that runs the SAME attack library behind a defense, so
  the runner/report machinery is reused unchanged (a blocked call returns a refusal sentinel the
  heuristic judge already scores as a hold; a `[GUARD-BLOCKED:<name>]` tag attributes which guard
  fired — zero artifact-schema change). Two transparent reference defenses ship: an
  `InputJailbreakClassifier` (pattern-based input stage) and an `OutputSystemPromptDLP` (output-stage
  data-loss filter, handed the protected system prompt, blocks distinctive tokens / 8-word spans —
  never told the canary, block reasons secret-free). `iago/delta.py` renders the **bypass-rate delta**
  (raw vs guarded) reusing report.py's rate/Wilson helpers, with guard attribution (blocks vs bypasses
  actually neutralized), over-block cost on benign controls, and a residual-bypass list. CLI: `--guard`
  on `run`, `iago delta <raw> <guarded>`, and `iago defense-delta --guard all` (paired run + report in
  one shot — the demo command). Tests 94→137 green (offline). **Live-verified vs llama3.1 (full library,
  1 trial, Claude-regraded):** LLM07 system-prompt exfiltration **26% (37/144) → 0% (0/144), −25.7 pts,
  non-overlapping 95% CIs** — 37 paired trials discordant in the guard's favour, 0 against (deterministic
  canary judge — a *verbatim / canary-only* measure; paraphrased or translated leakage routes around both
  the oracle and the DLP, a documented false-negative, and the DLP is handed the prompt it protects — so 0%
  is a floor, never "leak solved"). Harmful-content **5% (4/86) → 2% (2/86), −2.4 pts** — a small reduction
  the report keeps *directional* (overlapping CIs, not asserted significant). Over-block **15/43** controls
  but **0** `direct-ask` (pure-benign traffic untouched — only attack-framed controls blocked). The report
  labels the significance test as a conservative independent-CI proxy for the correct paired McNemar test,
  and flags a heuristic-only run's harmful-content row as "not adjudicated" rather than a fake 0→0. The `Guard` protocol is the seam for a real third-party guard (guardrails-ai validator, a
  Hub jailbreak classifier) as a drop-in — left as a clean extension point so the shipped delta stays
  dependency-light and reproducible.
- [ ] **Named hardening recommendations.** The Guardrails Hub is a catalog of concrete defenses (input
  jailbreak classifier, PII/output validators, toxicity). Upgrade the report's generic "add output-side
  classification" line to name specific, installable defenses per leaking category.

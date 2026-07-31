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

- [ ] **System-prompt extraction / prompt leaking** (LLM07: System Prompt Leakage) — a *different*
  objective: get the model to reveal its hidden system prompt. Needs a `prompt-leak` objective + a
  judge check ("did it disclose hidden instructions?"). High value — LLM07 is its own Top-10 entry
  Iago doesn't touch.
- [ ] **Unsafe output handling** (LLM05: Improper Output Handling) — coax the model into emitting
  content dangerous *when rendered downstream*: a markdown-image data-exfil URL, HTML/JS, a shell
  one-liner in a fenced block. Needs an objective + a "dangerous-when-rendered?" judge criterion.

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

- [ ] **Attack-vs-defense delta.** guardrails-ai ships a `guardrails start` OpenAI-compatible proxy
  with input/output guards in front of the model. Point the target at (a) the raw model and (b) the
  same model behind a guardrails-ai guard, run the same library, and report the **bypass-rate delta**
  ("this guard cut jailbreak X% → Y%"). Strong attack-vs-defense demo; extends `Target` with a proxied
  backend and reuses the existing runner/report machinery.
- [ ] **Named hardening recommendations.** The Guardrails Hub is a catalog of concrete defenses (input
  jailbreak classifier, PII/output validators, toxicity). Upgrade the report's generic "add output-side
  classification" line to name specific, installable defenses per leaking category.

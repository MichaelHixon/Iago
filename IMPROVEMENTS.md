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

## Roadmap (named, not built yet)

- [x] **Multi-turn attacks** → crescendo + context-priming; `Technique.turns` + `Target.chat()`,
  the runner carries the conversation and judges the final reply. The strongest real jailbreaks
  build across turns so no single prompt trips detection.

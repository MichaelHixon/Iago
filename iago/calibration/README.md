# Judge calibration set

`judge_control_set.jsonl` — 125 model responses with reviewer labels, the ground truth `iago judge-eval`
scores every judge against (ISC-35). Entries are real replies drawn from local llama3.x runs across the
harmful-content, benign-control, and system-prompt-extraction objectives (stratified over the heuristic
judge's own verdicts so its failure modes are represented), plus four synthetic **paraphrased leaks** that
omit the canary — the canary judge's documented false-negative, included so its FNR is measured rather than
assumed zero.

Labels: `refused` / `complied-useless` / `bypassed` for `forbidden` and `control` entries (on the control
objective `bypassed` means "answered the benign question"); `leaked` / `held` for `prompt-leak` entries.
`label_source` names who labeled and when. Labels are one reviewer's reading, not a multi-rater panel — the
agreement numbers are against that reading.

Four `bypassed` harmful entries ship with `response: null` and a `response_sha256`: the bodies (a phishing
email, fake-news articles) are not published. Put them in `judge_control_set.local.jsonl` (gitignored, same
`id` + `response`) to score those positives; `judge-eval` reports how many positives had text.

`judge_metrics.json` — the last measured agreement / FPR / FNR per `judge_id`, quoted in every report header.
A `judge_id` is a fingerprint of the scoring code (or rubric + model for the Claude judge); edit a judge and
its metrics go stale until `iago judge-eval` is run again.

**The shipped metrics are measured with `--no-overlay`**, so every number here is one a public clone
reproduces. `set_sha256` covers ids, labels AND whether each body was available, and `set_variant` says
which set was scored; a report header refuses to quote metrics whose fingerprint does not match the control
set in that checkout, rather than assert a number the reader cannot reproduce.

## Adding labels

One JSON object per line in `judge_control_set.jsonl`:

`{id, kind, label, label_source, objective_id, response | response_sha256, provenance, technique_id, canary?, note?}`

- `kind` is `forbidden`, `control` or `prompt-leak`; `label` uses that kind's vocabulary (above).
- `canary` is required on `prompt-leak` entries: the string planted in the system prompt.
- A body that must not be published ships as `"response": null` plus `response_sha256` (the SHA-256 of the
  exact text that was labeled). The text itself goes in the gitignored `judge_control_set.local.jsonl` as
  `{"id": ..., "response": ...}`. `load_control_set` refuses an overlay body whose hash does not match, so a
  label can never be scored against different text than it was assigned to.

**The gap that matters most.** The set holds 4 harmful-bypass positives (`kind: forbidden`, `label:
bypassed`), and all 4 are withheld. A public clone therefore measures the harmful false-negative rate on
zero bodies; the 13 `bypassed` positives it can score are all benign `control` compliance (the 18 scoreable `prompt-leak` positives are a different surface). More labeled harmful bypasses
(withheld, hash in the public set, body in the overlay) are the highest-value additions.

## Agreement floor

`iago judge-eval --min-agreement 0.8` turns agreement into a gate: a judge whose agreement point estimate
is below the floor exits 3 (its Wilson interval printed beside it); `--min-n 30` refuses to judge a judge
scored on fewer entries (exit 4); a judge at or above `--suspect-at` (default 1.0 once a floor is set)
passes but is flagged SUSPECT, since a perfect score on a small hand-labeled set more often means the labels
drifted toward the judge. All three are off by default. Cohen's kappa is printed beside agreement and is
not gated. A floor on the heuristic judge measures little: it escalates to `needs-review` by design.

Two things to keep in mind when reading the shipped numbers. Agreement and kappa count every verdict class (refused, complied-useless, bypassed), so a judge can have a 0% false-positive rate and still disagree with the labels often: most of the Claude judge's misses are between the two non-bypass classes. And the Claude judge is a model, so its numbers move between runs: two runs on the same set and judge build differed by one agreement and one false negative (59/87 vs 58/87, 3/13 vs 2/13). Treat the shipped row as one sample. At kappa 0.37 and 67% agreement, the shipped Claude judge would fail a `--min-agreement 0.8` floor.

**The heuristic judge's FPR/FNR are reported `n/a`, not 0% and 100%.** `judge.judge()` cannot return
`bypassed` at all (see `HEURISTIC_VERDICTS`) — it is a pre-filter that escalates to `needs-review` — so
those two rates would describe the verdict vocabulary rather than the judge. It is scored on agreement and
unresolved rate; the judges that CAN assert a positive (canary, Claude rubric) carry real FPR/FNR.

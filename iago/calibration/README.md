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

**The heuristic judge's FPR/FNR are reported `n/a`, not 0% and 100%.** `judge.judge()` cannot return
`bypassed` at all (see `HEURISTIC_VERDICTS`) — it is a pre-filter that escalates to `needs-review` — so
those two rates would describe the verdict vocabulary rather than the judge. It is scored on agreement and
unresolved rate; the judges that CAN assert a positive (canary, Claude rubric) carry real FPR/FNR.

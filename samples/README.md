# Samples

Model output Iago ships, and how each piece was made. Everything else Iago produces stays in the gitignored `reports/` directory.

## prompt-extraction-report.md / .html and replay/prompt-extraction.jsonl

One live run of the prompt-extraction surface (OWASP LLM07) against local `ollama:llama3.1:latest` on 2026-10-15: 12 techniques × 3 planted system prompts × 3 trials, 108 trials in all. The report files are that run's report, unedited; when the renderer's wording changes they are re-rendered from the run's own artifact with `iago report` (same trials and verdicts; only the text and the Generated time move). The fixture holds that run's replies, which is what `--target replay` plays back.

Made with:

```bash
uv run iago run --model llama3.1:latest --category prompt-extraction --html --record-replay samples/replay/prompt-extraction.jsonl
```

The fixture header records the model, the time, the iago version and the hash of the technique library it was recorded against. Every replay report names the fixture and its sha256.

**Checked, not just claimed:** `tests/test_replay.py` replays this fixture and asserts that the scorecard it produces is the scorecard in `prompt-extraction-report.md`. If the library, the judge or the fixture changes, that test fails and the sample has to be re-recorded with the command above.

**Only a prompt-extraction recording is safe to commit.** `--record-replay` stores every reply in full; a recording of the forbidden categories holds harmful completions, so treat it like a `--log` transcript and keep it out of version control.

**Why this one is safe to publish.** Iago's `--log` transcripts are marked "do not share publicly" because they hold full attack prompts and any real secret a model leaks. This sample is different: the fixture stores replies keyed by hash, not the prompts; the secrets the model leaks are fictional canaries planted by `objectives.yaml`; and the surface is prompt extraction only, so it holds no reply to a harmful-content request.

## trust-escalation-sample.md

A hand-picked excerpt of one three-arm trust-escalation trial against the same model: the anonymous arm held, the arm carrying fabricated prior history crossed. Verbatim replies; see the file's header for the technique and objective.

# Changelog

## 0.2.0 (2026-10-22)

### Breaking

- **No default Ollama model.** Every Ollama run names its model: `run`, `defense-delta`, `adaptive-run` and the `*-run` surface commands need `--model`, `adaptive-run --attacker llm` needs `--attacker-model`, and `campaign --models` is required (two or more). A missing model exits 2 and names the flag before anything runs. The old default (`llama3.1:latest`) quietly depended on a tag that might not be pulled.
- **The Anthropic default model moves.** Without `--model`, `--target anthropic` now runs the newest released Claude Haiku the Models API lists, not a fixed id, so the same command can test a newer model later. The report names the model that ran; pin `--model` for a repeatable run.
- **`iago gate` refuses runs on different models.** A baseline and a current run that targeted different models exit 2 unless `--allow-model-mismatch` is passed, since the delta would be the model swap. A different guard stack, or a replay of a live run, is the same model and still gates. Models are compared by the name each run recorded, so an old baseline recorded as `ollama:llama3.1:latest` and a new run with `--model llama3.1` count as different; re-record the baseline with the same tag. Re-pulling a moving tag such as `:latest` changes the weights without changing the name, so the check does not see that.
- **Library callers:**
  - `OllamaTarget(model)` and `ollama_chat_fn(model)` require a model and raise `ValueError` on a missing or blank tag; `build_target("ollama")` with no model raises too.
  - `DEFAULT_ANTHROPIC_MODEL` is gone (`DEFAULT_ANTHROPIC_FAMILY` names the family instead).
  - `AnthropicTarget()` with no model calls the Models API when it is built and raises `ValueError` if no model can be found (no key, no network, no match). Only the CLI refuses an unauthorized API target before building it; a library caller that builds one gets the lookup, and the runner's `--authorized` guard still blocks every attack call.

### Changed

- An API target without `--authorized` is refused by the CLI before it is built, so nothing reaches the API, including the default's model lookup.
- The Claude judge stays pinned: its published accuracy (`iago/calibration/judge_metrics.json`) is measured on one exact model, so its default moves only when the judge is recalibrated.
- The Llama Guard and prompt-injection guard defaults stay named models: their parsers read model-specific output. The names do not freeze weights.
- compose-delta reports print a Reproduce command that includes `--model`.
- Docs use a `<model>` placeholder instead of naming a model that will go stale.

### Tests

- The live Ollama round-trip runs only with `IAGO_SMOKE_MODEL=<model>`; the rest of the suite stays offline.

## 0.1.0

First release (untagged).

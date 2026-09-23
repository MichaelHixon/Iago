# Defenses and the attack-versus-defense delta

A *guard* is a defense placed in front of the model. `iago defense-delta --guard <spec>` runs the same attack library with and without the guard and reports the difference in bypass rate. Two guards ship with the repo and need no extra dependencies (an input-side jailbreak detector and an output-side system-prompt filter), so `--guard all` reproduces offline on any clone. Three real third-party guards plug into the same `Guard` interface, opt-in by name so their backends never become project dependencies:

- `llama-guard` — Meta Llama Guard 3 via Ollama (`ollama pull llama-guard3`). Verified live against Iago's composed-evasion attacks (hazard codes S2/S5/S13/S14); benign traffic passed clean.
- `guardrails-ai` — a Guardrails Hub jailbreak validator (`pip install guardrails-ai`). Wiring verified by tests; the real backend has not yet been run end to end here.
- `hf-prompt-injection` — a HuggingFace prompt-injection classifier (`pip install transformers`). Wiring verified only.

A guard whose backend is not installed fails loudly with an install hint, never a silent pass.

<p align="center">
  <img src="assets/iago-logo.png" alt="Iago" width="120" height="120" />
</p>

<h1 align="center">Iago</h1>

**A red-team harness for LLM guardrails.** Iago systematically probes a language model's safety controls, judges which techniques bypass them, and produces a pentest-style findings report — so those controls can be understood and **hardened**.

> ⚠️ **Defensive research, authorized use only.** Iago is built to test guardrails on a **local model you run and own** (or a model you have explicit authorized API access to), in order to understand how safety controls fail and how to strengthen them. This is red-team research to improve defense — the same ethic as authorized penetration testing. **Do not target hosted or third-party models without authorization.** The attack techniques included here are for evaluating your own model's controls.

---

## Why "Iago"?

In Shakespeare's *Othello*, **Iago** destroys a stronger, more powerful man using nothing but words. He never draws a weapon or forges a document — he wins entirely through crafted persuasion, feeding Othello the right phrases in the right order until Othello acts against his own nature and judgment.

That is *exactly* what guardrail-bypass attacks are. There's no exploit code, no buffer overflow, no memory corruption — only **language**, carefully constructed to talk a model past its own safety instructions and into doing the thing it was built to refuse. The attack surface is the conversation itself.

Naming the tool Iago keeps that truth front and center: **the payload is persuasion.** Understanding *how* words defeat a model's guardrails is the first step to building models that don't fall for it.

*(Namespace note: "Loki" — the obvious trickster pick — is already heavily used across security tooling. "Iago" is clear, and honestly the better fit: Loki causes chaos, but Iago works through targeted, patient manipulation, which is closer to how these attacks actually operate.)*

---

## What it does

Iago applies a classic offensive-security loop to a new target class — an LLM instead of a host:

```
  recon              →  attack library     →  execute        →  judge          →  report
  (target model +       (categorized           (fire each        (did the          (findings +
   its guardrails)       bypass techniques)     technique)        guardrail hold    hardening recs)
                                                                  or bypass?)
```

1. **Target adapter** — a clean interface to the model under test (local via Ollama for MVP; swappable).
2. **Attack library** — 43 guardrail-bypass techniques as structured data across 15 categories: direct-ask, role-play / persona, format-shift, instruction-hierarchy, encoding / obfuscation, low-resource-language, prompt-injection, **template-injection** (forged ChatML / Llama control tokens — probes the serialization boundary), refusal-suppression, **competing-objectives** (helpful-vs-harmless dual response; Wei et al. 2023), **many-shot** (fabricated compliant transcript, shot count drivable via `--shots`; Anil et al. 2024), multi-turn, and the indirect-injection surfaces: RAG-context, agent tool-use, and MCP tool/server injection. Techniques carry OWASP LLM Top-10 tags, and cipher / low-resource techniques are decode-gated so a decode failure isn't miscounted as a held guardrail.
3. **Runner** — fires each technique against a set of forbidden objectives and captures the responses.
4. **Judge** — decides per attempt whether the guardrail **held** or was **bypassed** (heuristic judge to start; an LLM-based judge for nuance).
5. **Report generator** — a markdown report: summary (X/Y bypassed), results by category, the prompt-and-response evidence, and a **hardening recommendations** section — the defensive payoff.

## What it is *not*

- **Not** a code-vulnerability scanner or an "LLM-finds-bugs-in-your-code" tool. That space is crowded and defensive-by-detection; Iago is offensive-by-design against the *model's own controls*.
- **Not** a model trainer or fine-tuner. It attacks a stock model as-is.
- **Not** a web app, dashboard, or service. Command-line, single-purpose, on purpose.

## Stack

- **Python** (managed with [`uv`](https://github.com/astral-sh/uv))
- **Target:** local model via [Ollama](https://ollama.com) — private, no rate limits, free to test against
- **Judge / orchestration:** the [Claude API](https://docs.anthropic.com) for the LLM-based judge

## Status

**End-to-end loop working.** Point Iago at a local model and one command fires the full attack library × objectives, judges each response, and writes a pentest-style report to `reports/`. Methodology built in: multi-trial **bypass rates** (not single shots) with 95% Wilson confidence intervals, pinned sampling (fixed temperature + per-trial seed) so runs reproduce, a benign **control objective** that calibrates the judge, a **Claude rubric judge** (`iago regrade`) that reasons about content instead of keywords, decode-gating for cipher / low-resource techniques, and structured JSONL artifacts so reporting never re-hits the model. The report surfaces per-technique caveats (e.g. template-injection's runtime dependency, many-shot's pool cycling) so limitations sit next to the numbers. Still ahead: new attack *goals* (system-prompt extraction, unsafe-output handling) and an attack-vs-defense delta against a guardrail proxy.

## Getting started

> Requires a local model. Iago never ships attack results and is intended to run against a model you control.

```bash
# 1. Install a local model runtime and pull a target model
brew install ollama
ollama pull llama3.1

# 2. Set up the project environment
uv sync

# 3. See the loaded attack library and objectives
uv run iago library

# 4. Prove the loop end-to-end (1 technique × 1 objective × 1 trial)
uv run iago run --smoke

# 5. Full run — writes a findings report to reports/
uv run iago run                 # whole library, default 3 trials/pair
uv run iago run --trials 5      # more trials → more defensible bypass rate
```

The target backend is selectable with `--target` (default `ollama`) and the model tag with `--model`. Only the local Ollama backend ships today; the `Target` interface and a small registry make an authorized API backend a one-line addition. The runner refuses a non-local target unless you pass `--authorized` — the "only test what you own" ethic, enforced in code rather than just documented.

## Ethics

Iago exists to make model guardrails **better**. Every run is against an authorized target, every report ends in hardening recommendations, and the techniques are documented so defenders can recognize and resist them. If you use or adapt this, keep it to systems you are authorized to test. Responsible disclosure applies.

## License

[MIT](LICENSE) © 2026 Michael Hixon. The permissive license covers the code; the responsible-use expectations in **Ethics** above (and the `--authorized` guard in the runner) still apply.

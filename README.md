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
2. **Attack library** — 61 guardrail-bypass techniques as structured data across 18 categories: direct-ask, role-play / persona, format-shift, instruction-hierarchy, encoding / obfuscation, low-resource-language, prompt-injection, prompt-extraction (system-prompt / canary leakage; LLM07), **template-injection** (forged ChatML / Llama control tokens — probes the serialization boundary), refusal-suppression, **competing-objectives** (helpful-vs-harmless dual response; Wei et al. 2023), **many-shot** (fabricated compliant transcript, shot count drivable via `--shots`; Anil et al. 2024), multi-turn, **composed-evasion** (stacks primitives — encoding chains plus persona / authority / refusal frames — as first-class techniques; the operative frame text is operator-supplied and gitignored, so the repo ships the composition engine, not ready-to-fire jailbreak strings), **provenance-forging** (plants a fake *memory/history* rather than an instruction — a decision log, prior joint research, a trusted-partner claim — as prior conversation turns, so the model derives trust from planted history; each trust-gated objective carries a scenario policy gate and a machine-checkable **action token** the model emits only if it crosses the gate, and each trial runs a **three-arm placebo-controlled differential** — anonymous, neutral-placebo, and provenance — that confirms an escalation only when provenance crosses the gate while the anonymous *and* placebo arms hold, so the number is the trust lift the fabricated relationship bought, net of merely having prior context; see `samples/` for a recorded refuse→grant flip), and the indirect-injection surfaces: RAG-context, agent tool-use, and MCP tool/server injection. Techniques carry OWASP LLM Top-10 tags, and cipher / low-resource techniques are decode-gated so a decode failure isn't miscounted as a held guardrail.
3. **Runner** — fires each technique against a set of forbidden objectives and captures the responses.
4. **Judge** — decides per attempt whether the guardrail **held** or was **bypassed** (heuristic judge to start; an LLM-based judge for nuance).
5. **Report generator** — a markdown report: summary (X/Y bypassed), results by category, the prompt-and-response evidence, and a **hardening recommendations** section — the defensive payoff.

## What it is *not*

- **Not** a code-vulnerability scanner or an "LLM-finds-bugs-in-your-code" tool. That space is crowded and defensive-by-detection; Iago is offensive-by-design against the *model's own controls*.
- **Not** a model trainer or fine-tuner. It attacks a stock model as-is.
- **Not** a web app, dashboard, or service. Command-line, single-purpose, on purpose.

## OWASP LLM Top-10 coverage

Iago maps its surfaces to the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/). It is a **runtime guardrail attacker**, so it covers the items that are runtime prompt/behavioral attacks and deliberately marks the three that are not — documenting that boundary honestly rather than claiming ten-for-ten.

| # | Risk | Coverage |
|---|------|----------|
| LLM01 | Prompt Injection | ✅ core attack library + agentic indirect injection |
| LLM02 | Sensitive Information Disclosure | ✅ `disclosure-run` — cross-subject leak, incl. the output channel |
| LLM03 | Supply Chain | ⛔ out of scope — a build/provenance concern, not a runtime prompt attack |
| LLM04 | Data & Model Poisoning | ◑ runtime analogues covered (`memory-run`, `rag-run`); training-time poisoning is out of scope |
| LLM05 | Improper Output Handling | ✅ deterministic unsafe-output oracle (dangerous-when-rendered constructs) |
| LLM06 | Excessive Agency | ✅ `privilege-run` — confused-deputy unauthorized privileged actions |
| LLM07 | System Prompt Leakage | ✅ prompt-extraction family + deterministic canary judge |
| LLM08 | Vector & Embedding Weaknesses | ◑ partial via `rag-run` retrieval poisoning |
| LLM09 | Misinformation | ✅ `misinfo-run` — deterministic fabricated-identifier oracle |
| LLM10 | Unbounded Consumption | ⛔ out of scope — resource-exhaustion / denial-of-service, deliberately excluded |

The out-of-scope items (LLM03, LLM04 training-time, LLM10) are not runtime guardrail attacks a tool like this can measure by driving a model — supply-chain and training-time poisoning live in the build/data pipeline, and unbounded-consumption is a load/DoS concern. Naming that boundary is the honest form of "complete."

## How Iago relates to the general red-team tools

Several mature open-source tools cover broad AI red teaming — NVIDIA [Garak](https://github.com/NVIDIA/garak) (automated vulnerability scanning), Microsoft [PyRIT](https://github.com/microsoft/PyRIT) (attack-campaign orchestration), and [DeepTeam](https://github.com/confident-ai/deepteam) (multi-class automated red team) among them. Iago is deliberately narrower and complements rather than competes with them. Its focus is **measurement**: multi-trial **bypass rates** with 95% Wilson confidence intervals — honest sampling uncertainty under a fixed attack set, not a claim of wider coverage than the general tools — instead of pass/fail, a **deterministic canary judge** for system-prompt leakage (ground truth, not a judge guess), decode-gating so a failed decode isn't miscounted as a held guardrail, and an **attack-vs-defense delta** that quantifies what a given guardrail actually neutralizes. If you want breadth of coverage, reach for the general frameworks; if you want a reproducible, statistically honest number for how often a specific control holds, that's Iago.

## Defenses & the attack-vs-defense delta

A *guard* is a defense placed in front of the model; `iago defense-delta --guard <spec>` runs the same attack library raw and guarded and reports the bypass-rate delta — the defensive payoff in one number. Two guards ship as **zero-dependency reference baselines** (a pattern-based input jailbreak classifier and an output-side system-prompt DLP filter); `--guard all` uses only these, so the delta reproduces offline on any clone.

Three **real third-party guards** are wired through the same `Guard` seam, opt-in by explicit name so their backends never become project dependencies:

- `llama-guard` — Meta Llama Guard 3, run locally via Ollama (`ollama pull llama-guard3`). **Live-verified:** the real model fired through the seam on Iago's composed-evasion attacks with real hazard codes (S2/S5/S13/S14); benign traffic passed clean.
- `guardrails-ai` — a Guardrails Hub jailbreak validator (`pip install guardrails-ai`). **Wiring-verified only** (adapter + seam exercised by tests; real backend not yet run end-to-end here).
- `hf-prompt-injection` — a HuggingFace prompt-injection classifier (`pip install transformers`). **Wiring-verified only.**

A guard whose backend isn't installed fails loudly with an install hint — never a silent pass. Example: `iago defense-delta --guard llama-guard`.

## Lexical-overlap paraphrase band (advisory, secondary)

The canary judge is precise but narrow: it confirms a system-prompt leak only when the planted canary appears (decode-tolerant). A model that discloses the *substance* of its prompt while rewording it — dropping the canary — reads as held. `iago lexical-leak <artifact>` surfaces that case as an **advisory** signal.

- **What it is:** a pure-Python **lexical-overlap** measure — the containment of the system prompt's distinctive content words in the reply. Zero dependencies, deterministic, no setup.
- **What it is *not*:** embedding- or meaning-grade semantics. A full synonym rewording that shares no vocabulary with the prompt scores ~0 — a residual false-negative, documented and tested. It catches light paraphrase (reused nouns/values), not deep reword.
- **It never asserts a verdict.** Only the deterministic canary match yields `BYPASSED`. This band emits a `low`/`elevated`/`high` label; the actionable cell is *"canary HELD but band HIGH"* — a candidate paraphrased leak for a human to inspect.
- **Calibration is a heuristic, not a benchmark.** Thresholds (`--elevated 0.30`, `--high 0.50`) are one-model-calibrated on llama3.1, where a refusal that echoes the prompt's topic words peaked at ~0.31 while a verbatim leak scored 1.0. On that run there were **0** false-positive suspects, but that is one model's margin, not a measured guarantee — tune the thresholds per target.

## Stack

- **Python** (managed with [`uv`](https://github.com/astral-sh/uv))
- **Target:** local model via [Ollama](https://ollama.com) — private, no rate limits, free to test against
- **Judge / orchestration:** the [Claude API](https://docs.anthropic.com) for the LLM-based judge

## Status

**End-to-end loop working.** Point Iago at a local model and one command fires the full attack library × objectives, judges each response, and writes a pentest-style report to `reports/`. Methodology built in: multi-trial **bypass rates** (not single shots) with 95% Wilson confidence intervals, pinned sampling (fixed temperature + per-trial seed) so runs reproduce, a benign **control objective** that calibrates the judge, a **Claude rubric judge** (`iago regrade`) that reasons about content instead of keywords, decode-gating for cipher / low-resource techniques, and structured JSONL artifacts so reporting never re-hits the model. The report surfaces per-technique caveats (e.g. template-injection's runtime dependency, many-shot's pool cycling) so limitations sit next to the numbers. Representative result (local `llama3.1`, 42 trials, deterministic canary scoring): the planted system-prompt secret leaked in **48%** of extraction trials — **43%** even against a hardened prompt whose own text explicitly forbids disclosing, encoding, or translating it. Leak detection is ground truth (a unique canary must appear verbatim in the reply), not a judge guess. **Composed-evasion** stacks these primitives, and `iago compose-delta` measures the marginal bypass each added layer buys over its best single arm, set against a noisy-OR independence baseline — so a stack that merely inherits its strongest layer is not mistaken for a real interaction effect. The gate stays deliberately conservative: a lift earns a verdict only when its Wilson interval clears the baseline's, and a stack whose constituents never fired alone is flagged, not scored. Still ahead: new attack *goals* (unsafe-output handling).

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

# 6. Agentic surfaces — red-team a tool-calling AGENT (behavioral bypass, not text).
#    Each has a companion `*-scenarios` command that lists its scenarios offline.
uv run iago agent-run --smoke        # indirect prompt injection → exfiltration (ASI01)
uv run iago tool-abuse-run --smoke   # sandboxed tool abuse → RCE / SSRF (ASI05/ASI02)
uv run iago memory-run --smoke       # memory / context poisoning (ASI06)
uv run iago rag-run --smoke          # RAG retrieval / knowledge-base poisoning
uv run iago a2a-run --smoke          # insecure inter-agent communication (ASI07)
uv run iago privilege-run --smoke    # excessive agency / confused deputy (LLM06/ASI03)
uv run iago disclosure-run --smoke   # sensitive-information disclosure (LLM02)
uv run iago misinfo-run --smoke      # misinformation / fabricated authority (LLM09)
```

The agentic surfaces drive an LLM through a tool loop and score a BEHAVIORAL bypass deterministically from the tool-call trace (an unauthorized action taken), not a text judgment. Their dangerous tools are pure in-memory fakes — a sandboxed `run_shell`/`fetch_url` never spawns a process or opens a socket, and the RAG retriever is a pure in-memory ranker — so the RCE/SSRF and retrieval attacks are simulated end to end.

The target backend is selectable with `--target` (default `ollama`) and the model tag with `--model`. Only the local Ollama backend ships today; the `Target` interface and a small registry make an authorized API backend a one-line addition. The runner refuses a non-local target unless you pass `--authorized` — the "only test what you own" ethic, enforced in code rather than just documented.

## Ethics

Iago exists to make model guardrails **better**. Every run is against an authorized target, every report ends in hardening recommendations, and the techniques are documented so defenders can recognize and resist them. If you use or adapt this, keep it to systems you are authorized to test. Responsible disclosure applies.

## License

[MIT](LICENSE) © 2026 Michael Hixon. The permissive license covers the code; the responsible-use expectations in **Ethics** above (and the `--authorized` guard in the runner) still apply.

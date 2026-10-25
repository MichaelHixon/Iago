# Agent surfaces

These commands test a tool-calling agent: what it *does*, not just what it says. Each has a companion scenarios command that lists its scenarios offline: `agent-scenarios`, `toolabuse-scenarios` (no hyphen), `memory-scenarios`, `rag-scenarios`, `a2a-scenarios`, `privilege-scenarios`, `disclosure-scenarios`, `misinfo-scenarios`.

Every `*-run` needs `--model`: any Ollama model you have pulled that supports tool calling (the model's page on https://ollama.com/library says). There is no default, because model names change.

```bash
uv run iago agent-run --smoke --model <model>        # indirect prompt injection → data exfiltration (ASI01)
uv run iago tool-abuse-run --smoke --model <model>   # sandboxed tool abuse → command execution / SSRF (ASI05/ASI02)
uv run iago memory-run --smoke --model <model>       # memory / context poisoning (ASI06)
uv run iago rag-run --smoke --model <model>          # RAG retrieval / knowledge-base poisoning
uv run iago a2a-run --smoke --model <model>          # insecure agent-to-agent communication (ASI07)
uv run iago privilege-run --smoke --model <model>    # excessive agency / confused deputy (LLM06/ASI03)
uv run iago disclosure-run --smoke --model <model>   # sensitive-information disclosure (LLM02)
uv run iago misinfo-run --smoke --model <model>      # misinformation / fabricated authority (LLM09)
```

The agent surfaces drive a model through a tool-calling loop and score a bypass from what the model *did* (the tool calls it made), not from a judgment about its text. The dangerous tools are in-memory fakes: the sandboxed `run_shell` and `fetch_url` never start a process or open a socket, and the RAG retriever is a pure in-memory ranker, so the command-execution, SSRF, and retrieval attacks are simulated end to end.

## Scope — what Iago is, and what it is not

Iago drives a model through a tool-calling loop and scores a bypass from the agent's own behavior — the tool calls it makes, the content it puts in its own answer. That framing is also its boundary. Three whole classes of agent-system flaw sit *outside* what a behavioral harness can score, and Iago does not claim them:

- **Infrastructure authorization.** A resource server (an MCP endpoint, a REST API) that exposes protected data to any caller who completes its handshake is a broken-access-control bug in the *server*, reachable without the model in the loop at all. That is a job for a web/MCP scanner, not a guardrail harness. Iago cannot enumerate or call an unauthenticated resource server on its own.
- **Backend code execution Iago cannot observe.** A tool whose *implementation* is unsafe — `eval()` on an argument, `yaml.load()` on a config, a non-atomic check-then-act — fails inside the backend. Iago scores whether the agent *delivers* a dangerous payload into such a tool (the agent's observable action); it does not and cannot confirm the backend executed it. Delivery is not exploitation, and the reports keep that distinction (an attempted delivery is `attempted`, never `hijacked`).
- **Timing and concurrency.** A TOCTOU / race condition — two actions interleaving against a non-atomic resource — is won by *when* calls land, not by *what* the agent decides. Iago fires scenarios sequentially and scores single-trajectory behavior; it has no concurrency surface, and a scripted "two calls in one turn" against an in-memory fake would measure the rig, not the agent. Out of scope by construction.

These are deliberate omissions, documented so the gaps read as chosen, not missed. The honest one-line summary: Iago measures *agent susceptibility* — being talked into a dangerous tool call, over-reaching, or leaking — and leaves infrastructure authz, backend code bugs, and races to the tools built for them.

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

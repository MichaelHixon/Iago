"""Target adapter.

A clean, swappable interface so the runner never knows what model it is hitting.
Today: a local Ollama model. Tomorrow: an authorized API target implementing the
same `Target` contract — no runner changes required.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

from .config import DEFAULT_MODEL


class Target(ABC):
    """A thing Iago can send a prompt to and read a text response from."""

    name: str

    #: True when the target is a model on this machine (no ToS / authorization
    #: concern). The runner refuses a non-local target unless --authorized is set.
    is_local: bool = False

    #: The sampling controls this backend honors; `None` means every option passes through.
    supported_options: frozenset[str] | None = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        """Send a prompt (with an optional system message) and return the reply text.

        `options` carries sampling controls (e.g. temperature, seed) so the runner
        can pin them for reproducible runs.
        """
        raise NotImplementedError

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        """Multi-turn: send a full conversation ([{role, content}, ...]) and return the
        assistant reply. Default implementation refuses — multi-turn-capable targets
        override this. Enables crescendo / context-priming attacks that build across turns.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support multi-turn chat")


def unsupported_options(target: object, options: dict) -> list[str]:
    """The requested sampling controls `target` will silently drop, sorted. Empty when the
    backend honors all of them (or passes everything through). Uses getattr so a duck-typed
    test double without the attribute reads as pass-through."""
    supported = getattr(target, "supported_options", None)
    if supported is None:
        return []
    return sorted(k for k in options if k not in supported)


def sampling_gap(target: object, *, temperature: float, seed: int) -> list[str]:
    """The pinned controls (temperature, seed) `target` drops, warning on stderr when any are.
    A manifest records what was REQUESTED, so a surface stores this beside it; otherwise a
    dropped seed reads as pinned."""
    dropped = unsupported_options(target, {"temperature": temperature, "seed": seed})
    if dropped:
        print(f"  WARNING: {getattr(target, 'name', target)} ignores {', '.join(dropped)}: "
              "requested but not applied; the manifest records them as unsupported_controls.",
              file=sys.stderr)
    return dropped


class OllamaTarget(Target):
    """A local model served by Ollama."""

    is_local = True

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.name = f"ollama:{model}"

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "The 'ollama' package is not installed. Run: uv add ollama"
            ) from exc

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options=options or {},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed (is the daemon running and '{self.model}' pulled?): {exc}"
            ) from exc

        # ollama returns a ChatResponse OBJECT (attribute access), older/dict paths
        # use item access. Support both, then fail loudly only if truly empty.
        content = _extract_content(response)
        if not content:
            raise RuntimeError(f"Ollama returned no content: {response!r}")
        return content

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        import ollama

        try:
            response = ollama.chat(model=self.model, messages=messages, options=options or {})
        except Exception as exc:
            raise RuntimeError(f"Ollama chat failed: {exc}") from exc
        content = _extract_content(response)
        if not content:
            raise RuntimeError(f"Ollama returned no content: {response!r}")
        return content


def _extract_content(response: object) -> str | None:
    """Pull message.content from either a ChatResponse object or a dict."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_ANTHROPIC_MAX_TOKENS = 1024


class AnthropicTarget(Target):
    """An authorized API target — attack a Claude model on your own account.

    NOT local, so the runner's authorization guard requires an explicit --authorized:
    this must only be pointed at a model you are authorized to test (your own API key).
    The client is injectable for testing.
    """

    is_local = False
    # No seed parameter exists on the Messages API, so a seed is dropped; `num_predict` maps to
    # max_tokens so a capped probe stays capped on this backend too.
    supported_options = frozenset({"temperature", "num_predict"})

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, client=None) -> None:
        self.model = model
        self.name = f"anthropic:{model}"
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run: uv add anthropic"
            ) from exc
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    def _create(self, system: str, messages: list[dict], options: dict | None):
        """One Messages API call; maps the supported_options onto the API's parameters."""
        opts = options or {}
        return self._get_client().messages.create(
            model=self.model,
            max_tokens=opts.get("num_predict", _ANTHROPIC_MAX_TOKENS),
            temperature=opts.get("temperature", 1.0),
            system=system,
            messages=messages,
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        try:
            msg = self._create(system or "", [{"role": "user", "content": prompt}], options)
        except Exception as exc:
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        # Concatenate text blocks from the response content.
        parts = []
        for block in getattr(msg, "content", []) or []:
            text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
            if text:
                parts.append(text)
        content = "".join(parts).strip()
        if not content:
            raise RuntimeError(f"Anthropic returned no text content: {msg!r}")
        return content

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        # Anthropic takes system separately; pull the system message out. A second one has no
        # faithful mapping (overwriting it silently drops instructions), so refuse it loudly.
        system = None
        convo = []
        for m in messages:
            if m.get("role") == "system":
                if system is not None:
                    raise ValueError("AnthropicTarget.chat accepts at most one system message")
                system = m.get("content", "")
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        try:
            msg = self._create(system or "", convo, options)
        except Exception as exc:
            raise RuntimeError(f"Anthropic chat failed: {exc}") from exc
        parts = [getattr(b, "text", None) if not isinstance(b, dict) else b.get("text")
                 for b in getattr(msg, "content", []) or []]
        content = "".join(p for p in parts if p).strip()
        if not content:
            raise RuntimeError(f"Anthropic returned no text content: {msg!r}")
        return content


# --- Target registry / factory -------------------------------------------------
#
# The seam that makes the target a first-class run parameter. Today only `ollama`
# is registered; an authorized API target (e.g. AnthropicTarget) registers here
# with one line and becomes selectable via --target, with the runner's
# authorization guard (is_local / --authorized) already gating it.

#: kind -> builder(model) -> Target
TARGET_REGISTRY: dict[str, "callable[[str], Target]"] = {
    "ollama": lambda model: OllamaTarget(model=model),
    "anthropic": lambda model: AnthropicTarget(model=model or DEFAULT_ANTHROPIC_MODEL),
}


def available_targets() -> list[str]:
    return sorted(TARGET_REGISTRY)


def build_target(kind: str = "ollama", model: str = DEFAULT_MODEL) -> Target:
    """Construct a target by kind. Raises a clear error for an unknown kind."""
    try:
        builder = TARGET_REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown target {kind!r}; available: {', '.join(available_targets())}"
        ) from None
    return builder(model)

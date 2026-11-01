"""Target adapter.

A clean, swappable interface so the runner never knows what model it is hitting.
Today: a local Ollama model. Tomorrow: an authorized API target implementing the
same `Target` contract — no runner changes required.
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable

from .config import DEFAULT_TARGET_GEN_TIMEOUT, DEFAULT_TARGET_NUM_PREDICT


def require_ollama_model(model: str | None, flag: str = "--model") -> str:
    """Ollama has no default model: refuse a missing tag with the flag that supplies it."""
    if not model or not model.strip():
        raise ValueError(f"the ollama target has no default model; pass {flag} <tag> (see `ollama list`)")
    return model


class Target(ABC):
    """A thing Iago can send a prompt to and read a text response from."""

    name: str

    #: True when the target is a model on this machine (no ToS / authorization
    #: concern). The runner refuses a non-local target unless --authorized is set.
    is_local: bool = False

    #: The sampling controls this backend honors; `None` means every option passes through.
    supported_options: frozenset[str] | None = None

    #: Manifest fields a backend declares about itself (`target_manifest_extra`); None = nothing.
    manifest_extra: dict | None = None

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


def sampling_gap(target: object, *, temperature: float, seed: int, name: str | None = None) -> list[str]:
    """The pinned controls (temperature, seed) `target` drops, warning on stderr when any are.
    A manifest records what was REQUESTED, so a surface stores this beside it; otherwise a
    dropped seed reads as pinned. `name` labels the warning when `target` is a bare chat_fn."""
    dropped = unsupported_options(target, {"temperature": temperature, "seed": seed})
    if dropped:
        print(f"  WARNING: {name or getattr(target, 'name', target)} ignores {', '.join(dropped)}: "
              "requested but not applied; the manifest records them as unsupported_controls.",
              file=sys.stderr)
    return dropped


def exc_label(exc: BaseException) -> str:
    """`<Class>: <message>`, the shape `judge.run_error` writes, so a harness defect (TypeError)
    reads differently from a backend failure (the targets wrap those as RuntimeError). judge.py
    keeps its own copy: any edit there changes the scorer fingerprint."""
    return f"{type(exc).__name__}: {exc}"


class RunFatal(Exception):
    """A failure every trial would hit. The runners re-raise it and fail the run instead of
    writing RUN-ERROR rows that drop out of the rates and leave the run reading as held."""


class UnmappableConversation(RunFatal, ValueError):
    """The harness built a conversation the backend cannot carry faithfully (a second system
    message for Anthropic). A harness defect, not a transport blip."""


class SamplingRefused(RunFatal, RuntimeError):
    """The API refused a sampling control the build-time probe said the model accepts. Changing
    the sampling mid-run would mix two regimes in one artifact, so the run fails instead."""


class TruncatedReply(str):
    """A reply the target stopped at ITS OWN token cap. It is judged as written (a capped reply
    is still long enough to show compliance), and the runner flags the row `truncated`, so a
    reader can see which verdicts rest on cut-off text. A cap the caller asked for (the
    determinism probe's 160 tokens) is not a truncation and is never flagged."""


class VendorReply(str):
    """A reply that carries what the backend reported about it beside the text, in `.vendor`.

    The vendor's refusal is a label, never a verdict: the reply is judged as written, like any
    other, and the runner copies `.vendor` onto the row so a reader can set the two side by side.
    A guard block replaces the reply with a plain string, so a blocked row carries no label."""

    vendor: dict


#: Marks a response field that is not there at all, as opposed to one present as null.
_ABSENT = object()


def _attr(obj: object, name: str):
    """`name` from an SDK object or a dict, or `_ABSENT` when the field is missing."""
    if isinstance(obj, dict):
        return obj.get(name, _ABSENT)
    # The SDK's pydantic models fill an omitted field with None; `model_fields_set` holds the
    # fields the API actually sent, so an omitted one still reads as absent, not as null.
    fields_set = getattr(obj, "model_fields_set", None)
    if isinstance(fields_set, (set, frozenset)) and name in getattr(type(obj), "model_fields", {}) \
            and name not in fields_set:
        return _ABSENT
    return getattr(obj, name, _ABSENT)


_DATED = re.compile(r"-\d{8}")


def served_matches(requested: str | None, served: str | None) -> bool | None:
    """Whether the model that answered is the one asked for: None when either name is missing.
    A dated snapshot of the requested id counts as the same model (`claude-x` served as
    `claude-x-20260101`, or the reverse); a different id that merely starts with the requested one
    (`claude-opus-5` vs `claude-opus-5-5`) does not."""
    if not requested or not served:
        return None
    if requested == served:
        return True
    return any(long.startswith(short) and _DATED.fullmatch(long[len(short):]) is not None
               for short, long in ((requested, served), (served, requested)))


def anthropic_vendor_fields(msg: object, requested: str) -> dict:
    """What a Messages API response says about how it ended and who answered, as row fields.

    Read, never inferred: `stop_reason`, `stop_details.category`, `stop_details.explanation` and
    top-level `model` (the model that produced the message). A field the response does not carry
    is null AND named in `vendor_absent`, so "the API said null" and "the API said nothing" stay
    apart. Per the API docs `stop_details` is populated only when `stop_reason == "refusal"`, may
    be null even then, and its `category` may be null."""
    absent: list[str] = []

    def take(obj: object, name: str, label: str):
        value = _attr(obj, name)
        if value is _ABSENT:
            absent.append(label)
            return None
        return value

    stop_reason = take(msg, "stop_reason", "stop_reason")
    details = take(msg, "stop_details", "stop_details")
    category = explanation = None
    if details is not None:
        category = take(details, "category", "stop_details.category")
        explanation = take(details, "explanation", "stop_details.explanation")
    served = take(msg, "model", "model")
    return {"vendor_stop_reason": stop_reason, "vendor_stop_category": category,
            "vendor_stop_explanation": explanation, "requested_model": requested,
            "served_model": served, "vendor_absent": absent}


def requested_api_model(target: object) -> str | None:
    """The model id an Anthropic target was asked for (through wrappers), or None for any other
    backend. Lets the runner label an Anthropic trial that errored before a reply came back."""
    for t in _layers(target):
        if isinstance(t, AnthropicTarget):
            return t.model
    return None


def _field(response: object, name: str):
    """Read `name` from an ollama ChatResponse object or a plain dict."""
    value = getattr(response, name, None)
    if value is None and isinstance(response, dict):
        value = response.get(name)
    return value


def _layers(target: object):
    """`target`, then each target it wraps (guard and recording wrappers expose `.inner`)."""
    while target is not None:
        yield target
        target = getattr(target, "inner", None)


def generation_bounds(target: object) -> dict | None:
    """The per-call bounds of `target` (through wrappers), or None for a backend without them, so
    a manifest records which cap and timeout a run was fired under."""
    for t in _layers(target):
        if isinstance(t, OllamaTarget):  # the one backend with configured cap/timeout attributes
            return {"target_num_predict": t.num_predict, "target_gen_timeout": t.timeout}
    return None


def target_manifest_extra(target: object) -> dict:
    """Manifest fields a backend declares about itself (through wrappers): today the replay
    target's `replay` block, which is what tells a report it is not a live run."""
    return next((dict(t.manifest_extra) for t in _layers(target)
                 if isinstance(getattr(t, "manifest_extra", None), dict)), {})


def single_turn_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    """A `generate(prompt, system)` call as the message list a chat backend receives."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return messages


class OllamaTarget(Target):
    """A local model served by Ollama.

    Every call is bounded (the agent seam's anti-runaway pattern): `timeout` is the ollama client's
    read timeout (it raises, and the runner records the trial as an error), and `num_predict`
    caps tokens unless the caller's options set their own. A reply stopped at that default cap
    comes back as a TruncatedReply. `timeout <= 0` disables the wall-clock bound.
    """

    is_local = True

    def __init__(
        self,
        model: str | None,
        *,
        timeout: float = DEFAULT_TARGET_GEN_TIMEOUT,
        num_predict: int = DEFAULT_TARGET_NUM_PREDICT,
    ) -> None:
        self.model = model = require_ollama_model(model)
        self.name = f"ollama:{model}"
        self.timeout = None if timeout is None or timeout <= 0 else timeout
        self.num_predict = num_predict

    def _chat(self, messages: list[dict], options: dict | None, what: str) -> str:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "The 'ollama' package is not installed. Run: uv add ollama"
            ) from exc

        caller_cap = "num_predict" in (options or {})
        opts = dict(options or {})
        opts.setdefault("num_predict", self.num_predict)  # caller override wins
        try:
            client = ollama.Client() if self.timeout is None else ollama.Client(timeout=self.timeout)
            response = client.chat(model=self.model, messages=messages, options=opts)
        except Exception as exc:
            raise RuntimeError(f"{what} (gen timeout={self.timeout}s): {exc}") from exc

        # ollama returns a ChatResponse OBJECT (attribute access), older/dict paths
        # use item access. Support both, then fail loudly only if truly empty.
        content = _extract_content(response)
        if not content:
            raise RuntimeError(f"Ollama returned no content: {response!r}")
        if _field(response, "done_reason") == "length" and not caller_cap:
            print(f"  WARNING: {self.name} stopped at its {self.num_predict}-token cap; the reply "
                  "is judged as cut off and the row is flagged truncated.", file=sys.stderr)
            return TruncatedReply(content)
        return content

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        return self._chat(
            single_turn_messages(prompt, system), options,
            f"Ollama request failed (is the daemon running and '{self.model}' pulled?)",
        )

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        return self._chat(messages, options, "Ollama chat failed")


def _extract_content(response: object) -> str | None:
    """Pull message.content from either a ChatResponse object or a dict."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content


#: The API target's default is a family, not a model: the newest model in it is looked up when the
#: target is built, so the default never names a model that has been superseded. Pass --model to
#: pin one; the report names whichever model actually ran.
DEFAULT_ANTHROPIC_FAMILY = "haiku"
_ANTHROPIC_MAX_TOKENS = 1024
# A newest-wins default must not land on a preview build: those are not the family's release.
_PRERELEASE = re.compile(r"(^|-)(preview|exp|experimental|beta|alpha|rc)(-|$|\d)")


def latest_model(client, family: str) -> str:
    """The newest model with `family` as a whole id segment, by the Models API's `created_at`.

    A segment match, not a prefix, so a reshuffled id (`claude-3-5-haiku-…` became
    `claude-haiku-4-5-…`) still counts and the default cannot quietly freeze on the old shape.
    """
    in_family = re.compile(rf"(^|-){re.escape(family)}(-|$)")
    try:
        models = [m for m in client.models.list(limit=1000)
                  if in_family.search(m.id) and not _PRERELEASE.search(m.id)]
    except Exception as exc:
        raise ValueError(f"could not look up the latest {family} model ({exc}); check "
                         "ANTHROPIC_API_KEY and the network, or pass --model <id>") from exc
    if not models:
        raise ValueError(f"the Models API lists no {family} model; pass --model <id>")
    return max(models, key=lambda m: m.created_at).id


#: A legal value that is not the API default (1.0), so a model that accepts only the default
#: refuses the probe. Probing at 1.0 would pass on every model and learn nothing.
_PROBE_TEMPERATURE = 0.5


def is_temperature_refusal(exc: BaseException) -> bool:
    """True for the 400 a model returns when it no longer takes `temperature` at all (Claude 5.5,
    Opus 4.7 and later: "`temperature` is deprecated for this model."). Keys on the captured shape
    (400, `invalid_request_error`, a message naming `temperature` and `deprecated`), not the exact
    sentence. A range-validation 400 ("temperature: range: 0 <= x <= 1") is not this: it names no
    deprecation. Duck-typed on the SDK's `status_code` and `body` so a fake can raise it."""
    if getattr(exc, "status_code", None) != 400:
        return False
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict) or err.get("type") != "invalid_request_error":
        return False
    message = str(err.get("message", "")).lower()
    return "temperature" in message and "deprecated" in message


def probe_controls(client, model: str, declared: frozenset[str]) -> frozenset[str]:
    """The controls `model` accepts, learned by one 1-token request at a non-default temperature.

    A deprecation 400 drops `temperature`; any reply keeps it (a 200 shows the parameter was
    accepted, not that it is honoured; a reply with `stop_reason: refusal` still counts as
    accepted, since the parameter was not rejected). Anything else fails the build, naming the
    model, so a run never starts on an unknown sampling regime."""
    try:
        client.messages.create(model=model, max_tokens=1, temperature=_PROBE_TEMPERATURE,
                               messages=[{"role": "user", "content": "hi"}])
    except Exception as exc:
        if is_temperature_refusal(exc):
            return declared - {"temperature"}
        raise ValueError(f"could not probe {model} for sampling support ({exc_label(exc)}); "
                         "check ANTHROPIC_API_KEY and the network") from exc
    return declared


class AnthropicTarget(Target):
    """An authorized API target — attack a Claude model on your own account.

    NOT local, so the runner's authorization guard requires an explicit --authorized:
    this must only be pointed at a model you are authorized to test (your own API key).
    The client is injectable for testing.
    """

    is_local = False
    # No seed parameter exists on the Messages API, so a seed is dropped; `num_predict` maps to
    # max_tokens so a capped probe stays capped on this backend too. `temperature` is declared
    # here, but each instance LEARNS whether its model accepts it (`probe_controls`, one 1-token
    # request when built) and replaces this set with the result.
    supported_options = frozenset({"temperature", "num_predict"})

    def __init__(self, model: str | None = None, client=None,
                 supported_options: frozenset[str] | None = None) -> None:
        """`supported_options`, when given, is a result an earlier probe already recorded (a batch
        plan), so the build sends nothing; omitted, the model is probed."""
        self._client = client
        self.model = model or latest_model(self._get_client(), DEFAULT_ANTHROPIC_FAMILY)
        self.name = f"anthropic:{self.model}"
        self.supported_options = (frozenset(supported_options) if supported_options is not None
                                  else probe_controls(self._get_client(), self.model,
                                                      type(self).supported_options))

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

    def _create(self, system: str, messages: list[dict], options: dict | None, what: str):
        """One Messages API call; maps the supported_options onto the API's parameters. A
        control the model refused is omitted, never sent as some stand-in value. A failure
        raises `SamplingRefused` (a RunFatal) for a refused control, else RuntimeError(`what`)."""
        opts = options or {}
        kwargs = {"model": self.model,
                  "max_tokens": opts.get("num_predict", _ANTHROPIC_MAX_TOKENS),
                  "system": system, "messages": messages}
        if "temperature" in self.supported_options:
            kwargs["temperature"] = opts.get("temperature", 1.0)
        try:
            return self._get_client().messages.create(**kwargs)
        except Exception as exc:
            if is_temperature_refusal(exc):
                raise SamplingRefused(
                    f"{self.model} refused `temperature` after the build-time probe accepted it; "
                    f"failing the run rather than changing the sampling mid-run ({exc_label(exc)})"
                ) from exc
            raise RuntimeError(f"Anthropic {what} failed: {exc}") from exc

    def _reply(self, msg) -> VendorReply:
        """The response's text with its vendor fields attached. A refusal can arrive before any
        output (empty content): that is still a reply, judged as an empty one, never a run error,
        or every pre-output refusal would drop out of the rates. Any other empty response fails."""
        blocks = _attr(msg, "content")
        parts = [getattr(b, "text", None) if not isinstance(b, dict) else b.get("text")
                 for b in (blocks if blocks is not _ABSENT and blocks else [])]
        content = "".join(p for p in parts if p).strip()
        vendor = anthropic_vendor_fields(msg, self.model)
        if not content and vendor["vendor_stop_reason"] != "refusal":
            raise RuntimeError(f"Anthropic returned no text content: {msg!r}")
        reply = VendorReply(content)
        reply.vendor = vendor
        return reply

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        return self._reply(self._create(system or "", [{"role": "user", "content": prompt}],
                                        options, "request"))

    def chat(self, messages: list[dict], options: dict | None = None) -> str:
        # Anthropic takes system separately; pull the system message out. A second one has no
        # faithful mapping (overwriting it silently drops instructions), so refuse it loudly.
        system = None
        convo = []
        for m in messages:
            if m.get("role") == "system":
                if system is not None:
                    raise UnmappableConversation(
                        "AnthropicTarget.chat accepts at most one system message")
                system = m.get("content", "")
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        return self._reply(self._create(system or "", convo, options, "chat"))


# --- Target registry / factory -------------------------------------------------
#
# The seam that makes the target a first-class run parameter. A new backend registers here with
# one line and becomes selectable via --target, with the authorization guard (is_local /
# --authorized) already gating it.

def _replay_target(fixture: str | None) -> Target:
    """`--model` names the fixture file; omitted, the shipped sample fixture loads."""
    from .replay import ReplayTarget  # imported lazily: replay.py imports this module

    return ReplayTarget(fixture)


#: kind -> builder(model) -> Target. A None model means "the backend's own default"; a backend
#: without one (ollama) raises ValueError.
TARGET_REGISTRY: dict[str, Callable[[str | None], Target]] = {
    "ollama": lambda model: OllamaTarget(model=model),
    "anthropic": AnthropicTarget,
    "replay": _replay_target,
}

#: Built-in kinds that are not local. The CLI refuses these before building them when --authorized
#: is missing, so nothing reaches their API (the Anthropic default looks its model up over the
#: network when built). Any other kind still meets the runner's `is_local` guard after building.
API_TARGETS = frozenset({"anthropic"})

#: Targets only `iago run` offers. A replay plays back a fixture `run --record-replay` wrote, and
#: only the chatbot report discloses that no model ran, so no other command lists it.
RUN_ONLY_TARGETS = frozenset({"replay"})


def available_targets(*, include_run_only: bool = False) -> list[str]:
    return sorted(k for k in TARGET_REGISTRY if include_run_only or k not in RUN_ONLY_TARGETS)


def build_target(kind: str = "ollama", model: str | None = None) -> Target:
    """Construct a target by kind. Raises a clear error for an unknown kind."""
    try:
        builder = TARGET_REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown target {kind!r}; available: {', '.join(available_targets(include_run_only=True))}"
        ) from None
    return builder(model)

"""Talker backend contract for the live voice session (stdlib only).

A Talker is the fast conversational half of a live session. Any class that
meets this contract can drive `/voice/live`; the gateway never imports a vendor
SDK itself. Backends are looked up by name with `load_backend`.

Capabilities (a frozenset of strings on the class):
  audio_in        accepts 16 kHz mono PCM16 through send_audio
  audio_out       emits {"type": "audio"} events
  text_out        emits {"type": "text"} events
  full_duplex     can be spoken over (emits "interrupted")
  tool_calls      emits {"type": "tool_call"} events
  silent_context  send_text(..., silent=True) adds context without a spoken reply

Event dicts yielded by `events()`:
  {"type": "audio", "pcm": bytes}
  {"type": "text", "text": str}
  {"type": "tool_call", "call_id": str, "name": str, "args": dict}
  {"type": "interrupted"}
  {"type": "turn_end"}
  {"type": "error", "message": str}

Plugin backends: set `MVG_TALKER_BACKENDS` to comma-separated `name=module:Class`
entries; they are imported lazily on first use.
"""
from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any, AsyncIterator

CAPABILITIES = frozenset({"audio_in", "audio_out", "text_out", "full_duplex", "tool_calls", "silent_context"})
EVENT_TYPES = frozenset({"audio", "text", "tool_call", "interrupted", "turn_end", "error"})


class TalkerBackend:
    capabilities: frozenset = frozenset()

    async def open(self, config: dict) -> None:
        raise NotImplementedError

    async def send_audio(self, pcm16_16k: bytes) -> None:
        raise NotImplementedError

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        raise NotImplementedError

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        raise NotImplementedError

    def events(self) -> AsyncIterator[dict]:
        raise NotImplementedError

    async def cancel(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


def requires(backend, *caps: str) -> None:
    """Raise ValueError naming every capability the backend lacks."""
    have = getattr(backend, "capabilities", frozenset())
    missing = [c for c in caps if c not in have]
    if missing:
        raise ValueError(f"backend {type(backend).__name__} lacks capability: {', '.join(missing)}")


class MockBackend(TalkerBackend):
    """Replays a scripted list of events and records everything sent to it."""
    capabilities = frozenset({"audio_in", "audio_out", "text_out", "full_duplex", "tool_calls", "silent_context"})

    def __init__(self, script: list | None = None):
        self.script = list(script or [])
        self.sent: list = []
        self.opened = False
        self.closed = False

    async def open(self, config: dict) -> None:
        self.opened = True
        self.sent.append(("open", dict(config or {})))

    async def send_audio(self, pcm16_16k: bytes) -> None:
        self.sent.append(("audio", bytes(pcm16_16k)))

    async def send_text(self, text: str, *, silent: bool = False) -> None:
        self.sent.append(("text", text, silent))

    async def send_tool_result(self, call_id: str, result: dict) -> None:
        self.sent.append(("tool_result", call_id, result))

    async def events(self) -> AsyncIterator[dict]:
        for ev in self.script:
            await asyncio.sleep(0)
            yield ev

    async def cancel(self) -> None:
        self.sent.append(("cancel",))

    async def close(self) -> None:
        self.closed = True
        self.sent.append(("close",))


_REGISTRY: dict = {"mock": MockBackend}


def _plugin_entries() -> dict:
    out = {}
    for item in os.environ.get("MVG_TALKER_BACKENDS", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, target = item.partition("=")
        if sep and ":" in target:
            out[name.strip()] = target.strip()
    return out


def load_backend(name: str) -> type:
    """Return the backend class registered under name (builtin first, then
    MVG_TALKER_BACKENDS). Unknown names raise ValueError listing what exists."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    plugins = _plugin_entries()
    if name in plugins:
        module, _, cls = plugins[name].partition(":")
        try:
            return getattr(importlib.import_module(module), cls)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"talker backend {name!r} ({plugins[name]}) failed to load: {exc}") from exc
    known = sorted(set(_REGISTRY) | set(plugins))
    raise ValueError(f"unknown talker backend {name!r}; known: {', '.join(known) or 'none'}")

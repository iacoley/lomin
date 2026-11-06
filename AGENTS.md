# AGENTS.md — Lomin Agent Voice Gateway

## What this is

Lomin Agent Voice Gateway is a voice gateway that gives agents a voice. It is a single Python 3.13
stdlib HTTP service (`gateway.py`) that fronts speech-to-text (Whisper), a
brain (an agent that answers one framed prompt) and text-to-speech (a chain of
backends). It handles speech gating, speaker identification, addressee
detection in shared rooms, sentence cutting, gap-fill audio while the brain
thinks, and echo filtering. It is part of the Karakos stack.

Files:

- `gateway.py` — the HTTP service and the turn pipeline.
- `tts_backends.py` — the swappable TTS chain (`MVG_TTS_CHAIN`).
- `desktop_tts.py`, `desktop_embed.py` — optional GPU-host servers (TTS, speaker embedding), reached over an ssh reverse tunnel.
- `sentence_segmenter.py`, `text_utils.py`, `multipart.py`, `audio_utils.py`, `t3_fast.py` — helpers.
- `tests/` — the suite. `intents/`, household nodes and household services are not here.

## Running the tests

```
python3 -m pytest -q tests/
```

No network. Tests that need the plugin's `classify` module skip when it is not
importable. `test_render_gap_fill_sfx.py` and `test_voice_latency_report.py`
skip unless their scripts (outside this repo) are present. `tests/conftest.py`
sets the agent-name env vars before `gateway` is imported; `test_standalone.py`
and `test_plugin.py` run the gateway in subprocesses with a clean env.

## The plugin

`MVG_PLUGIN` names an importable module that supplies the site-specific parts.
Every hook is optional:

- `apply_defaults()` — fill env defaults with `setdefault` before config reads them.
- `brain_call(target, prompt, timeout) -> str` — the agent answers one framed prompt.
- `brain_cancel(target, task_id) -> bool` — abort a running brain call; `cancel_active_turn(node_id)` in `gateway.py` uses it.
- `brain_call_stream(target, prompt, timeout, on_partial, stages_out) -> str` — streaming brain. Optional keyword `audio_clips` (list of WAV bytes) is passed only if the signature accepts it and `MVG_SPLICE_RAW_AUDIO=1`.
- `classify` — a module with `classify_choice(...)` (the addressee classifier).
- `try_intent(text) -> dict | None` — deterministic fast path.
- `AMBIENT_ADDRESSED` — a string: the option key the addressee classifier answers with (default `addressed_to_agent`).

The gateway reads these as module attributes at import (`_plugin_brain_call`,
`_plugin_brain_call_stream`, `_jev_classifier`, `_try_intent`). Tests replace
those attributes with monkeypatch, so new behaviour belongs behind a hook and
a test patches the hook.

## The rule

Household code lives in a plugin and never in this repo. That includes
household nodes, device names, people's names, addresses, paths, intents,
brain targets and any service that only one household runs. If a change needs
something household-specific, add a hook or an env var here, and put the
household value in the plugin or in the host's env file.

Do not add `nodes.json`, `intents/`, household `.service` units, household
wheels, or household names to tests or docs. `nodes.example.json` and
`lomin.service.example` are the only examples. Before a PR, check the tree for host paths, LAN addresses, personal names and
domains. The PR checklist lists the exact commands.

## Conventions

- Stdlib only at runtime. The optional dependencies (`onnxruntime`, `numpy`, `requests`) are guarded and fail open.
- Config is env-driven. Every new env var goes in the README config table and in `.env.example`.
- Log lines are JSON, one per request, with no audio bytes and no keys.
- A failing plugin is loud (stderr), never silent.
- `MVG_SPOKEN_PROGRESS=1` (+ `MVG_PROGRESS_*`): spoken progress from `stages_out["events"]` (`ProgressNarrator`); tests in `tests/test_spoken_progress.py`.
- `MVG_CONTEXT_OPENER=1`: opener from the first stage event (`_maybe_context_opener`); tests in `tests/test_context_opener.py`.

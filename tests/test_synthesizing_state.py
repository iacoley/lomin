#!/usr/bin/env python3
"""The /voice/turn `state` events: `synthesizing` is sent when the first real sentence goes to TTS, and a gap-fill
clip (`filler`) does not move the turn to `speaking`. Clients draw their progress steps from these events, so a
step that is never sent never fills. Every upstream is a monkeypatched module function: no network.

Also: a final reply that the streamed text already ends with is not spoken a second time.

Run: python3 -m pytest ./tests -v"""
import time

import gateway
from text_utils import stable_markdown_prefix, strip_markdown, unspoken_delta

from test_voice_turn import _isolate, names, parse_sse, post, server, stubs, turn, WAV  # noqa: F401


def _fake_stream(monkeypatch):
    pcm = b"\x01\x00" * 400
    monkeypatch.setattr(gateway, "synthesize_stream", lambda text: ("fake", iter([pcm])))


def _fake_filler(monkeypatch):
    clip = {"pcm": b"\x02\x00" * 400, "text": "", "category": "vocal_open", "slug": "v1"}
    monkeypatch.setattr(gateway, "GAP_FILL", True)
    monkeypatch.setattr(gateway, "GAP_FILL_AFTER_MS", 20)
    monkeypatch.setattr(gateway, "pick_gap_clip", lambda cats, used: dict(clip))


def _slow_brain(stubs, monkeypatch, delay):
    def brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        time.sleep(delay)
        return stubs.reply
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def test_synthesizing_comes_between_thinking_and_speaking(server, stubs, monkeypatch):
    _fake_stream(monkeypatch)
    events = turn(server, {"open_room": "1", "tts": "stream"})
    states = [d["state"] for e, d in events if e == "state"]
    assert states == ["transcribing", "classifying", "thinking", "synthesizing", "speaking"]


def test_a_text_only_turn_never_synthesizes(server, stubs, monkeypatch):
    events = turn(server, {"open_room": "1", "tts": "none"})
    states = [d["state"] for e, d in events if e == "state"]
    assert "synthesizing" not in states and states[-1] == "speaking"


def test_filler_does_not_send_speaking_or_hide_synthesizing(server, stubs, monkeypatch):
    """Before this, the first gap-fill clip sent `speaking` ~1 s into every turn, so a client never saw
    `synthesizing`, and showed THINK -> SPEAK while the agent was still thinking."""
    _fake_stream(monkeypatch)
    _fake_filler(monkeypatch)
    _slow_brain(stubs, monkeypatch, 0.5)
    events = turn(server, {"open_room": "1", "tts": "stream"})
    fillers = [i for i, (e, d) in enumerate(events) if e == "sentence" and d.get("filler")]
    assert fillers, "the slow brain should have produced a gap-fill clip"
    states = [(i, d["state"]) for i, (e, d) in enumerate(events) if e == "state"]
    assert [s for _, s in states] == ["transcribing", "classifying", "thinking", "synthesizing", "speaking"]
    speaking_at = next(i for i, s in states if s == "speaking")
    synth_at = next(i for i, s in states if s == "synthesizing")
    assert fillers[0] < synth_at < speaking_at     # the clip plays during THINK, VOICE follows it


def _speak_stream(partials, final):
    spoken, out = "", []
    for raw in partials:
        full = strip_markdown(stable_markdown_prefix(raw)) or ""
        d = unspoken_delta(full, spoken)
        if d:
            out.append(d)
            spoken = full
    out.append(unspoken_delta(strip_markdown(final) or "", spoken))
    return "".join(out)


def test_a_final_reply_already_at_the_end_of_the_streamed_text_is_not_spoken_again():
    narration, final = "Let me check.", "It is running."
    partials = [narration, narration + "\n\n" + final]
    assert _speak_stream(partials, final).strip() == "Let me check. It is running."


def test_a_final_that_differs_from_the_streamed_text_is_still_spoken():
    draft, final = "The app is working.", "I checked: the service is active."
    out = _speak_stream([draft], final)
    assert out.startswith(draft) and out.endswith(final)

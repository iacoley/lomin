"""Unit & regression tests for streaming_brain.py.

Verifies:
1. Stream size cap on single oversized SSE lines and single 'reply' events (PR #6 fix).
2. Elimination of quadratic string concatenation around on_partial (O(N) linear scaling & incremental mode).
3. Clean interruption and timeout raising for stalled socket/stream reads.
4. Full SSE protocol parity (OpenAI deltas, sentence/delta/reply events, comments, [DONE]).
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest

from streaming_brain import (
    StreamTimeoutError,
    build_http_brain_body,
    consume_brain_sse_lines,
    http_brain_call_stream,
)


# ---------------------------------------------------------------------------
# 1. Regression Tests: Stream Size Cap on Single Oversized Lines & Reply Events
# ---------------------------------------------------------------------------

def test_single_oversized_reply_event_strictly_capped():
    """Verify that a single 'reply' event larger than max_chars is strictly capped

    and does not slip past the cap into the return value or on_partial.
    """
    giant_reply = "R" * 100_000
    lines = [
        "event: reply\n",
        f'data: {{"reply": "{giant_reply}"}}\n',
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append, max_chars=50)

    assert len(reply) == 50
    assert reply == "R" * 50
    assert len(partials) == 1
    assert len(partials[0]) == 50
    assert partials[0] == "R" * 50


def test_single_oversized_reply_event_with_prior_partials_capped():
    """Verify that a giant reply event replacing prior partials is strictly capped."""
    lines = [
        "event: sentence\n",
        'data: {"text": "Small prefix."}\n',
        "\n",
        "event: reply\n",
        f'data: {{"reply": "{"Z" * 50_000}"}}\n',
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append, max_chars=40)

    assert len(reply) == 40
    assert reply == "Z" * 40
    assert len(partials) == 2
    assert partials[0] == "Small prefix."
    assert partials[1] == "Z" * 40


def test_single_oversized_default_reply_payload_capped():
    """Verify that an untagged SSE event with a giant 'reply' field is capped."""
    lines = [
        f'data: {{"reply": "{"K" * 20_000}"}}\n',
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append, max_chars=35)

    assert len(reply) == 35
    assert reply == "K" * 35
    assert len(partials) == 1
    assert partials[0] == "K" * 35


def test_single_oversized_raw_sse_line_capped():
    """Verify that a single raw SSE line exceeding max_chars is capped."""
    giant_raw = "X" * 10_000
    lines = [
        f"data: {giant_raw}\n",
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append, max_chars=60)

    assert len(reply) == 60
    assert reply == "X" * 60
    assert len(partials) == 1
    assert partials[0] == "X" * 60


def test_single_oversized_openai_delta_capped():
    """Verify that a single oversized OpenAI delta is capped cleanly."""
    lines = [
        f'data: {{"choices": [{{"delta": {{"content": "{"D" * 5_000}"}}}}]}}\n',
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append, max_chars=25)

    assert len(reply) == 25
    assert reply == "D" * 25
    assert partials == ["D" * 25]


def test_oversized_raw_line_length_protection():
    """Verify that an excessively long raw SSE line is safely bounded before parsing."""
    # Line larger than MAX_SSE_LINE_CHARS
    huge_line = "data: " + ("A" * 80_000) + "\n"
    reply = consume_brain_sse_lines([huge_line], max_chars=100)
    assert len(reply) == 100
    assert reply == "A" * 100


# ---------------------------------------------------------------------------
# 2. Regression Tests: Linear O(N) Scaling & on_partial Improvements
# ---------------------------------------------------------------------------

def test_no_quadratic_concatenation_scaling():
    """Verify that streaming thousands of deltas completes in linear time without

    quadratic ''.join(chunks) overhead.
    """
    delta_count = 2000
    lines = [
        'data: {"choices": [{"delta": {"content": "t"}}]}\n'
        for _ in range(delta_count)
    ]
    lines.append("data: [DONE]\n")

    partials_count = 0

    def counter(text: str):
        nonlocal partials_count
        partials_count += 1

    t0 = time.monotonic()
    reply = consume_brain_sse_lines(lines, on_partial=counter, max_chars=delta_count + 100)
    elapsed = time.monotonic() - t0

    assert len(reply) == delta_count
    assert reply == "t" * delta_count
    assert partials_count == 1
    # 2000 deltas should parse in well under 0.25 seconds in linear time
    assert elapsed < 0.25, f"Took {elapsed:.3f}s; possible quadratic join overhead"


def test_on_partial_accumulated_matches_expected():
    """Verify that default on_partial receives the full accumulated prefix at each step."""
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        'data: {"choices": [{"delta": {"content": "!"}}]}\n',
    ]
    partials = []
    reply = consume_brain_sse_lines(lines, on_partial=partials.append)

    assert reply == "Hello world!"
    assert partials == ["Hello world!"]


def test_on_partial_incremental_mode():
    """Verify that incremental=True passes only the incoming delta rather than the full buffer."""
    lines = [
        'data: {"choices": [{"delta": {"content": "Part1 "}}]}\n',
        'data: {"choices": [{"delta": {"content": "Part2 "}}]}\n',
        'data: {"choices": [{"delta": {"content": "Part3"}}]}\n',
    ]
    deltas = []
    reply = consume_brain_sse_lines(lines, on_partial=deltas.append, incremental=True)

    assert reply == "Part1 Part2 Part3"
    assert deltas == ["Part1 ", "Part2 ", "Part3"]


# ---------------------------------------------------------------------------
# 3. Regression Tests: Stalled Socket / Stream Read Timeouts & Interrupts
# ---------------------------------------------------------------------------

def test_stalled_stream_read_timeout_raises_stream_timeout_error():
    """Verify that a stream stalling mid-read raises StreamTimeoutError within read_timeout bounds."""
    def stalled_stream():
        yield 'data: {"choices": [{"delta": {"content": "Chunk1"}}]}\n'
        # Stalls for 5 seconds
        time.sleep(5.0)
        yield 'data: {"choices": [{"delta": {"content": "Chunk2"}}]}\n'

    partials = []
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as exc_info:
        consume_brain_sse_lines(
            stalled_stream(),
            on_partial=partials.append,
            read_timeout=0.1,
            raise_on_timeout=True,
        )
    elapsed = time.monotonic() - t0

    # Must time out in ~0.1s, NOT wait for the 5s sleep
    assert elapsed < 0.6, f"Took {elapsed:.3f}s; stalled read was not interrupted"
    assert isinstance(exc_info.value, StreamTimeoutError)
    assert exc_info.value.partial_text == "Chunk1"
    assert partials == ["Chunk1"]


def test_stalled_stream_read_timeout_clean_interrupt():
    """Verify that a stalled stream with raise_on_timeout=False cleanly interrupts

    and returns whatever text was accumulated before the stall.
    """
    def stalled_stream():
        yield 'data: {"choices": [{"delta": {"content": "ValidStart"}}]}\n'
        time.sleep(5.0)
        yield 'data: {"choices": [{"delta": {"content": "NeverReached"}}]}\n'

    partials = []
    t0 = time.monotonic()
    reply = consume_brain_sse_lines(
        stalled_stream(),
        on_partial=partials.append,
        read_timeout=0.1,
        raise_on_timeout=False,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 0.6, f"Took {elapsed:.3f}s; stalled read was not interrupted"
    assert reply == "ValidStart"
    assert partials == ["ValidStart"]


def test_stalled_stream_closes_underlying_resource():
    """Verify that a stalled stream resource with a close() method is closed on timeout."""
    class ClosableStalledStream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield 'data: {"choices": [{"delta": {"content": "Initial"}}]}\n'
            time.sleep(3.0)
            yield 'data: {"choices": [{"delta": {"content": "After"}}]}\n'

        def close(self):
            self.closed = True

    stream = ClosableStalledStream()
    reply = consume_brain_sse_lines(stream, read_timeout=0.1, raise_on_timeout=False)

    assert reply == "Initial"
    assert stream.closed is True


def test_max_stream_seconds_total_deadline_interrupt():
    """Verify that max_stream_seconds bounds total consumption time even when chunks trickle slowly."""
    def trickling_stream():
        for i in range(20):
            yield f'data: {{"choices": [{{"delta": {{"content": "c{i} "}}}}]}}\n'
            time.sleep(0.04)

    t0 = time.monotonic()
    reply = consume_brain_sse_lines(trickling_stream(), max_stream_seconds=0.10, raise_on_timeout=False)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.35
    assert "c0 " in reply
    # Must have interrupted before the 20th chunk
    assert "c19 " not in reply


def test_max_stream_seconds_raises_when_requested():
    """Verify that max_stream_seconds raises StreamTimeoutError when raise_on_timeout=True."""
    def slow_generator():
        yield 'data: {"choices": [{"delta": {"content": "hello"}}]}\n'
        time.sleep(0.5)

    with pytest.raises(TimeoutError) as exc_info:
        consume_brain_sse_lines(slow_generator(), max_stream_seconds=0.1, raise_on_timeout=True)

    assert isinstance(exc_info.value, StreamTimeoutError)
    assert exc_info.value.partial_text == "hello"


def test_stream_exception_propagates_immediately():
    """Verify that non-timeout exceptions raised during stream iteration propagate up."""
    def broken_stream():
        yield 'data: {"choices": [{"delta": {"content": "good"}}]}\n'
        raise ConnectionResetError("Remote server closed socket")

    with pytest.raises(ConnectionResetError):
        consume_brain_sse_lines(broken_stream(), read_timeout=1.0)


# ---------------------------------------------------------------------------
# 4. Parity & Protocol Tests (Sentences, Deltas, Comments, Bytes)
# ---------------------------------------------------------------------------

def test_sse_sentence_events_with_spacing():
    """Verify sentence events are correctly spaced when punctuation/spaces allow."""
    lines = [
        "event: sentence\n",
        'data: {"text": "First sentence."}\n',
        "\n",
        "event: sentence\n",
        'data: {"text": "Second sentence."}\n',
    ]
    reply = consume_brain_sse_lines(lines)
    assert reply == "First sentence. Second sentence."


def test_sse_bytes_and_comments_ignored():
    """Verify bytes lines and SSE comments (: ...) are handled cleanly."""
    lines = [
        b": ping comment\n",
        b'data: {"choices": [{"delta": {"content": "Echo"}}]}\n',
        b"\n",
        b': another comment\n',
        b'data: {"choices": [{"delta": {"content": " message."}}]}\n',
        b"data: [DONE]\n",
    ]
    reply = consume_brain_sse_lines(lines)
    assert reply == "Echo message."


def test_sse_reply_event_overrides_earlier_deltas():
    """Verify that an explicit event: reply specifies the final reply."""
    lines = [
        "event: delta\n",
        'data: {"text": "Draft text"}\n',
        "\n",
        "event: reply\n",
        'data: {"reply": "Polished final answer."}\n',
    ]
    reply = consume_brain_sse_lines(lines)
    assert reply == "Polished final answer."


def test_truncated_json_reply_extracted_cleanly():
    """Verify that a truncated JSON line with a reply field extracts inner text safely."""
    # Simulates a line truncated right after the reply string before closing quote/brace
    truncated_line = 'data: {"reply": "Recovered text without closing brace\n'
    reply = consume_brain_sse_lines([truncated_line])
    assert reply == "Recovered text without closing brace"


# ---------------------------------------------------------------------------
# 5. Integration Tests: Mock HTTP Server & Request Builders
# ---------------------------------------------------------------------------

class MockServer:
    def __init__(self, handler_cls):
        self.server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_port

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def test_build_http_brain_body_helper():
    body_dict, body_bytes = build_http_brain_body(
        prompt="Test prompt",
        session_id="sess-123",
        node_id="node-1",
        image_b64="imgdata",
        burst_images_b64=["b1", "b2"],
        audio_b64="audiodata",
    )
    assert body_dict["prompt"] == "Test prompt"
    assert body_dict["session_id"] == "sess-123"
    assert body_dict["node_id"] == "node-1"
    assert body_dict["image_b64"] == "imgdata"
    assert body_dict["burst_images_b64"] == ["b1", "b2"]
    assert body_dict["audio_b64"] == "audiodata"
    assert json.loads(body_bytes.decode("utf-8")) == body_dict


def test_http_brain_call_stream_integration():
    class SSEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "Streamed "}}]}\n\n')
            self.wfile.flush()
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "via HTTP"}}]}\n\n')
            self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = MockServer(SSEHandler)
    server.start()
    try:
        url = f"http://127.0.0.1:{server.port}/stream"
        partials = []
        reply = http_brain_call_stream(
            prompt="Hello",
            session_id="sess-h1",
            node_id="node-1",
            url=url,
            on_partial=partials.append,
        )
        assert reply == "Streamed via HTTP"
        assert partials == ["Streamed via HTTP"]
    finally:
        server.stop()


def test_http_brain_call_stream_json_fallback_capped():
    class JSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"reply": "J" * 200}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = MockServer(JSONHandler)
    server.start()
    try:
        url = f"http://127.0.0.1:{server.port}/json"
        reply = http_brain_call_stream(
            prompt="Hello",
            session_id="sess-j1",
            node_id="node-1",
            url=url,
            max_chars=50,
        )
        assert len(reply) == 50
        assert reply == "J" * 50
    finally:
        server.stop()


def test_cumulative_on_partial_is_coalesced():
    # Punctuation-heavy stream: 20,000 deltas of "a." must not call on_partial per delta.
    lines = ['data: {"delta": "a."}\n' for _ in range(20000)]
    calls = []
    reply = consume_brain_sse_lines(lines, on_partial=calls.append, incremental=False)
    assert reply == "a." * 20000
    assert 1 <= len(calls) <= 300
    assert calls[-1] == reply  # always one final call with the full text
    # Cumulative: every call extends the previous one.
    for prev, cur in zip(calls, calls[1:]):
        assert cur.startswith(prev) and len(cur) > len(prev)


def test_cumulative_on_partial_punctuation_free_single_call():
    lines = ['data: {"delta": "ab"}\n' for _ in range(20000)]
    calls = []
    reply = consume_brain_sse_lines(lines, on_partial=calls.append, incremental=False)
    assert len(reply) == 40000
    assert len(calls) == 1
    assert calls[0] == reply


def test_cumulative_on_partial_time_gate(monkeypatch):
    import streaming_brain

    now = [0.0]
    monkeypatch.setattr(streaming_brain.time, "monotonic", lambda: now[0])
    calls = []

    def gen():
        now[0] += 1.0
        yield 'data: {"delta": "One. "}\n'
        now[0] += 0.05  # < 150 ms, < 400 chars: coalesced
        yield 'data: {"delta": "Two. "}\n'
        now[0] += 0.2  # >= 150 ms since last call: emitted
        yield 'data: {"delta": "Three. "}\n'
        now[0] += 0.01
        yield 'data: {"delta": "Four"}\n'  # no boundary: no call until stream end

    reply = consume_brain_sse_lines(gen(), on_partial=calls.append, incremental=False, read_timeout=None)
    assert reply == "One. Two. Three. Four"
    assert calls == ["One. ", "One. Two. Three. ", "One. Two. Three. Four"]

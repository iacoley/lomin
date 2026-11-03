import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GATEWAY_DIR))
import gateway  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests for consume_brain_sse_lines
# ---------------------------------------------------------------------------

def test_consume_brain_sse_openai_deltas():
    partials = []
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        'data: {"choices": [{"delta": {"content": "!"}}]}\n',
        'data: [DONE]\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Hello world!"
    assert partials == ["Hello world!"]


def test_consume_brain_sse_bytes_and_comments():
    partials = []
    lines = [
        b": keepalive comment\n",
        b'data: {"choices": [{"delta": {"content": "The"}}]}\n',
        b"\n",
        b'data: {"choices": [{"delta": {"content": " answer."}}]}\n',
        b"data: [DONE]\n",
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "The answer."
    assert partials == ["The answer."]


def test_consume_brain_sse_events_sentence():
    partials = []
    lines = [
        "event: sentence\n",
        'data: {"text": "First sentence."}\n',
        "\n",
        "event: sentence\n",
        'data: {"text": "Second sentence."}\n',
        "\n",
        "event: reply\n",
        'data: {"reply": "First sentence. Second sentence."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "First sentence. Second sentence."
    assert partials == [
        "First sentence.",
        "First sentence. Second sentence.",
    ]


def test_consume_brain_sse_events_delta():
    partials = []
    lines = [
        "event: delta\n",
        'data: {"text": "Alpha "}\n',
        "\n",
        "event: delta\n",
        'data: {"text": "Beta "}\n',
        "\n",
        "event: delta\n",
        'data: {"text": "Gamma"}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Alpha Beta Gamma"
    assert partials == ["Alpha Beta Gamma"]


def test_consume_brain_sse_sentence_with_reply_key():
    partials = []
    lines = [
        "event: sentence\n",
        'data: {"reply": "Sentence one."}\n',
        "\n",
        "event: sentence\n",
        'data: {"reply": "Sentence two."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Sentence one. Sentence two."
    assert partials == ["Sentence one.", "Sentence one. Sentence two."]


def test_consume_brain_sse_standalone_reply_event():
    partials = []
    lines = [
        "event: status\n",
        'data: {"status": "thinking"}\n',
        "\n",
        "event: reply\n",
        'data: {"reply": "One shot reply."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "One shot reply."
    assert partials == ["One shot reply."]


# ---------------------------------------------------------------------------
# Integration tests for http_brain_call_stream & BRAIN_DIRECT_AUDIO
# ---------------------------------------------------------------------------

class MockBrainServer:
    def __init__(self, handler_cls):
        self.server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_port

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def test_http_brain_call_stream_sse_ask_stream(monkeypatch):
    seen_headers = []

    class SSEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen_headers.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunks = [
                b"event: sentence\ndata: {\"text\": \"Hello there.\"}\n\n",
                b"event: sentence\ndata: {\"text\": \"How can I help?\"}\n\n",
                b"event: reply\ndata: {\"reply\": \"Hello there. How can I help?\"}\n\n",
            ]
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = MockBrainServer(SSEHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/ask/stream")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "hello", "sess-sse-1", "node-1", "alice", 0.9, 5, on_partial=partials.append
        )
        assert reply == "Hello there. How can I help?"
        assert partials == [
            "Hello there.",
            "Hello there. How can I help?",
        ]
        assert "text/event-stream" in seen_headers[0].get("Accept", "")
    finally:
        server.stop()


def test_http_brain_call_stream_sse_openai_deltas(monkeypatch):
    class OpenAIDeltaHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                b"data: {\"choices\": [{\"delta\": {\"content\": \"Streamed \"}}]}\n\n",
                b"data: {\"choices\": [{\"delta\": {\"content\": \"response.\"}}]}\n\n",
                b"data: [DONE]\n\n",
            ]
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = MockBrainServer(OpenAIDeltaHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/v1/chat")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "test prompt", "sess-openai-1", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "Streamed response."
        assert partials == ["Streamed response."]
    finally:
        server.stop()


def test_http_brain_call_stream_json_fallback(monkeypatch):
    # When HTTP_BRAIN_STREAM is false (default) and server returns application/json
    class JSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"reply": "standard json fallback response"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = MockBrainServer(JSONHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", False)
        partials = []
        reply = gateway.http_brain_call_stream(
            "json prompt", "sess-json-1", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "standard json fallback response"
        assert partials == ["standard json fallback response"]
    finally:
        server.stop()


def test_http_brain_call_stream_json_with_stream_flag(monkeypatch):
    # When HTTP_BRAIN_STREAM is true but server answers with application/json
    class JSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"reply": "json despite stream request"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = MockBrainServer(JSONHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "json prompt", "sess-json-2", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "json despite stream request"
        assert partials == ["json despite stream request"]
    finally:
        server.stop()


def test_brain_direct_audio_gating(monkeypatch):
    seen_payloads = []

    class RecordHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            seen_payloads.append(json.loads(body.decode("utf-8")))
            resp = json.dumps({"reply": "audio acknowledged"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = MockBrainServer(RecordHandler)
    server.start()
    raw_audio_b64 = base64.b64encode(b"RIFFdummywavdata").decode("ascii")
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")

        # 1. BRAIN_DIRECT_AUDIO = False -> audio_b64 should NOT be in payload
        monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", False)
        monkeypatch.setenv("MVG_BRAIN_DIRECT_AUDIO", "0")
        gateway.http_brain_call("test", "sess-1", "node-1", "", None, 5, audio_b64=raw_audio_b64)
        assert len(seen_payloads) == 1
        assert "audio_b64" not in seen_payloads[0]

        # 2. BRAIN_DIRECT_AUDIO = True via http_brain_call -> audio_b64 IS in payload
        monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", True)
        monkeypatch.setenv("MVG_BRAIN_DIRECT_AUDIO", "1")
        gateway.http_brain_call("test", "sess-2", "node-1", "", None, 5, audio_b64=raw_audio_b64)
        assert len(seen_payloads) == 2
        assert seen_payloads[1]["audio_b64"] == raw_audio_b64

        # 3. BRAIN_DIRECT_AUDIO = True via http_brain_call_stream -> audio_b64 IS in payload
        partials = []
        gateway.http_brain_call_stream(
            "test", "sess-3", "node-1", "", None, 5, on_partial=partials.append, audio_b64=raw_audio_b64
        )
        assert len(seen_payloads) == 3
        assert seen_payloads[2]["audio_b64"] == raw_audio_b64

        # 4. Context propagation via _FRAME_CTX
        gateway._FRAME_CTX.ctx = {"audio_b64": raw_audio_b64}
        gateway.http_brain_call("test", "sess-4", "node-1", "", None, 5)
        assert len(seen_payloads) == 4
        assert seen_payloads[3]["audio_b64"] == raw_audio_b64
        gateway._FRAME_CTX.ctx = None
    finally:
        server.stop()

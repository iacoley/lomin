"""tests/test_side_channel.py — Unit tests for side_channel.py and gateway parity.

Tests:
1. reply_expects_reply (whitespace, quotes, Unicode emojis, joiners, question marks).
2. forward_artifacts_to_webhook (URL / code block detection, webhook dispatch, Discord cueing, sync & async).
3. emit_turn_transcript (JSONL file append, HTTP sink dispatch, error resilience, sync & async).
4. gateway integration (SSE expects_reply payload opt-in gate and side-channel wiring).
5. security and hardening (URL validation, safe redirects, bearer token auth, 0o600 permissions, log rotation).
6. voice turn end transcript capture (STT prompt and spoken reply recording).
7. async worker queue (flush, bounded queue non-blocking overflow handling).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import stat
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest

import gateway
from side_channel import (
    SafeRedirectHandler,
    _enqueue_task,
    _transcript_queue,
    _validate_url,
    _webhook_queue,
    emit_turn_transcript,
    emit_turn_transcript_sync,
    flush_side_channel_queue,
    forward_artifacts_to_webhook,
    forward_artifacts_to_webhook_sync,
    reply_expects_reply,
    reset_side_channel_queue,
)

# RFC 5737 documentation test addresses
DOC_WEBHOOK_URL = "http://198.51.100.10:8080/webhook"
DOC_SINK_URL = "http://198.51.100.20:8080/transcript-sink"


@pytest.fixture(autouse=True)
def clean_side_channel_queue():
    flush_side_channel_queue(timeout_s=1.0)
    reset_side_channel_queue()
    yield
    flush_side_channel_queue(timeout_s=1.0)
    reset_side_channel_queue()


# ============================================================================
# 1. Tests for reply_expects_reply
# ============================================================================


class TestReplyExpectsReply:
    @pytest.mark.parametrize(
        "text",
        [
            "What time is it?",
            "How are you?",
            "Can you help me with this?",
            "Is it raining outside?",
            "Really!?",  # ends in ?
            "?",
        ],
    )
    def test_basic_questions_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What time is it? ",
            "How are you?   \t\n",
            "Are you ready?\n\n",
            "   ?   ",
        ],
    )
    def test_trailing_whitespace_returns_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            'What time is it?"',
            "What time is it?'",
            '“What time is it?”',
            "‘What time is it?’",
            "What time is it?`",
            "What time is it?»",
            "«What time is it?»",
            'What time is it?"""',
            "What time is it?'''",
        ],
    )
    def test_trailing_quotes_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What time is it? 🤔",
            "Shall we go? 🚀",
            "Do you agree? 👍",
            "Are you there? 👋",
            "Would you like that? ✨",
            "What do you think? 💭",
            "How is that? ☕️",
        ],
    )
    def test_trailing_emojis_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Any questions? 👨‍💻",  # ZWJ sequence
            "Can we meet tomorrow? 🤷‍♀️",  # ZWJ + gender + VS16
            "Is that cool? \u200d\ufe0f",
            "Ready? 🏃‍♂️",
        ],
    )
    def test_trailing_complex_emoji_joiners_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            '“What is your name?” 🤔 ',
            'How are you?" 😊  \n',
            "Are you coming?' 🚀✨",
            '“Ready?” 👋 \t',
        ],
    )
    def test_mixed_quotes_emojis_whitespace_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What do you think❓",
            "Which one❔",
            "Are you sure❓ ",
            "Is this right❓ 🤔",
        ],
    )
    def test_unicode_question_ornaments_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The weather is sunny.",
            "The weather is sunny. 😊",
            "Hello there! 👋",
            "Check this out: https://example.com/search?q=test",
            "Really?!",  # ends in !
            "Done.",
            "That's fantastic!",
            'He asked "Why is the sky blue."',
        ],
    )
    def test_non_questions_return_false(self, text: str):
        assert reply_expects_reply(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\t",
            None,
            123,
            [],
            {},
        ],
    )
    def test_empty_or_invalid_inputs_return_false(self, text: Any):
        assert reply_expects_reply(text) is False


# ============================================================================
# 2. Tests for forward_artifacts_to_webhook
# ============================================================================


class TestForwardArtifactsToWebhook:
    def test_no_artifacts_returns_unchanged(self):
        raw = "The capital of France is Paris."
        spoken = "The capital of France is Paris."
        mock_poster = MagicMock()

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert res_spoken == spoken
        assert sent is False
        mock_poster.assert_not_called()

    def test_detects_http_and_https_urls(self):
        raw = "You can read the docs at https://example.com/docs for more details."
        spoken = "You can read the docs for more details."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == "You can read the docs for more details. Sending the link to Discord."
        mock_poster.assert_called_once()
        args, _ = mock_poster.call_args
        assert args[0] == DOC_WEBHOOK_URL
        assert args[1]["content"] == raw.strip()

    def test_detects_fenced_code_blocks(self):
        raw = "Here is the implementation:\n```python\ndef hello():\n    return 'world'\n```"
        spoken = "Here is the implementation."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == "Here is the implementation. Sending the link to Discord."
        mock_poster.assert_called_once()
        assert mock_poster.call_args[0][0] == DOC_WEBHOOK_URL

    def test_uses_mvg_artifact_webhook_url_env_var(self, monkeypatch):
        monkeypatch.setenv("MVG_ARTIFACT_WEBHOOK_URL", DOC_WEBHOOK_URL)
        raw = "See https://github.com/example/repo"
        spoken = "See the GitHub repository."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url="",
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == "See the GitHub repository. Sending the link to Discord."
        assert mock_poster.call_args[0][0] == DOC_WEBHOOK_URL

    def test_missing_webhook_url_returns_false_and_does_not_post(self, monkeypatch):
        monkeypatch.delenv("MVG_ARTIFACT_WEBHOOK_URL", raising=False)
        raw = "Visit https://example.com"
        spoken = "Visit the website."
        mock_poster = MagicMock()

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url="",
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is False
        assert res_spoken == spoken
        mock_poster.assert_not_called()

    @pytest.mark.parametrize(
        "already_cued_spoken",
        [
            "I sent the link to Discord for you.",
            "Sending the link to Discord for you.",
            "I already sent that over to Discord.",
            "Check Discord for the full link.",
            "I posted the link in Discord.",
            "I dropped it in Discord.",
            "I sent it to discord chat.",
        ],
    )
    def test_does_not_duplicate_cue_if_already_cued(self, already_cued_spoken: str):
        raw = "Check out https://example.com"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=already_cued_spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == already_cued_spoken
        mock_poster.assert_called_once()

    def test_empty_spoken_reply_gets_cue(self):
        raw = "https://example.com"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply="",
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == "Sending the link to Discord."

    def test_adds_period_to_spoken_reply_if_missing(self):
        raw = "Here is the site https://example.com"
        spoken = "Here is the site"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is True
        assert res_spoken == "Here is the site. Sending the link to Discord."

    def test_poster_failure_returns_false_and_unmodified_spoken(self):
        raw = "https://example.com"
        spoken = "Here is the link."
        mock_poster = MagicMock(return_value=False)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is False
        assert res_spoken == spoken

    def test_poster_exception_returns_false_and_unmodified_spoken(self):
        raw = "https://example.com"
        spoken = "Here is the link."
        mock_poster = MagicMock(side_effect=ConnectionError("Host unreachable"))

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=True,
        )

        assert sent is False
        assert res_spoken == spoken

    def test_real_http_server_post(self):
        received_requests = []

        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received_requests.append({
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type"),
                    "payload": json.loads(body.decode("utf-8")),
                })
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{server.server_port}/api/webhooks/test"
            raw = "Look at https://example.com"
            spoken = "Look at the website"

            res_spoken, sent = forward_artifacts_to_webhook(
                raw_reply=raw,
                spoken_reply=spoken,
                webhook_url=url,
                http_poster=None,
                sync=True,
            )

            assert sent is True
            assert res_spoken == "Look at the website. Sending the link to Discord."
            assert len(received_requests) == 1
            assert received_requests[0]["payload"]["content"] == raw
            assert "application/json" in received_requests[0]["content_type"]
        finally:
            server.shutdown()
            server.server_close()


# ============================================================================
# 3. Tests for emit_turn_transcript
# ============================================================================


class TestEmitTurnTranscript:
    def test_no_sink_configured_returns_false(self, monkeypatch):
        monkeypatch.delenv("MVG_TRANSCRIPT_SINK_URL", raising=False)
        monkeypatch.delenv("MVG_TRANSCRIPT_LOG_PATH", raising=False)

        res = emit_turn_transcript({"turn_id": "turn-1", "reply": "Hello"}, sync=True)
        assert res is False

    def test_invalid_turn_record_returns_false(self):
        assert emit_turn_transcript({}, sync=True) is False
        assert emit_turn_transcript(None, sync=True) is False  # type: ignore

    def test_log_path_appends_jsonl(self, tmp_path):
        log_file = tmp_path / "transcripts.jsonl"
        record_1 = {"turn_id": "turn-1", "prompt": "Hi", "reply": "Hello!"}
        record_2 = {"turn_id": "turn-2", "prompt": "What?", "reply": "Nothing."}

        res1 = emit_turn_transcript(record_1, log_path=str(log_file), sync=True)
        res2 = emit_turn_transcript(record_2, log_path=str(log_file), sync=True)

        assert res1 is True
        assert res2 is True
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == record_1
        assert json.loads(lines[1]) == record_2

    def test_log_path_creates_parent_directory(self, tmp_path):
        nested_log = tmp_path / "deep" / "nested" / "dir" / "transcripts.jsonl"
        record = {"turn_id": "t-1", "reply": "ok"}

        res = emit_turn_transcript(record, log_path=str(nested_log), sync=True)
        assert res is True
        assert nested_log.exists()
        assert json.loads(nested_log.read_text(encoding="utf-8").strip()) == record

    def test_log_path_env_var(self, tmp_path, monkeypatch):
        log_file = tmp_path / "env_transcripts.jsonl"
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_file))
        record = {"turn_id": "env-turn", "reply": "from env"}

        res = emit_turn_transcript(record, sync=True)
        assert res is True
        assert log_file.exists()
        assert json.loads(log_file.read_text(encoding="utf-8").strip()) == record

    def test_sink_url_dispatches_http_post(self):
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "turn-http", "prompt": "Status", "reply": "All systems nominal."}

        res = emit_turn_transcript(record, sink_url=DOC_SINK_URL, http_poster=mock_poster, sync=True)

        assert res is True
        mock_poster.assert_called_once()
        args, _ = mock_poster.call_args
        assert args[0] == DOC_SINK_URL
        assert args[1] == record

    def test_sink_url_env_var(self, monkeypatch):
        monkeypatch.setenv("MVG_TRANSCRIPT_SINK_URL", DOC_SINK_URL)
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "env-sink", "reply": "sink-env"}

        res = emit_turn_transcript(record, http_poster=mock_poster, sync=True)

        assert res is True
        mock_poster.assert_called_once()
        assert mock_poster.call_args[0][0] == DOC_SINK_URL

    def test_both_log_path_and_sink_url(self, tmp_path):
        log_file = tmp_path / "both.jsonl"
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "both-1", "reply": "both"}

        res = emit_turn_transcript(
            record,
            sink_url=DOC_SINK_URL,
            log_path=str(log_file),
            http_poster=mock_poster,
            sync=True,
        )

        assert res is True
        assert log_file.exists()
        assert json.loads(log_file.read_text(encoding="utf-8").strip()) == record
        mock_poster.assert_called_once()

    def test_sink_failure_with_log_path_still_succeeds(self, tmp_path):
        log_file = tmp_path / "partial.jsonl"
        mock_poster = MagicMock(side_effect=Exception("HTTP connection failed"))
        record = {"turn_id": "partial-1", "reply": "ok"}

        res = emit_turn_transcript(
            record,
            sink_url=DOC_SINK_URL,
            log_path=str(log_file),
            http_poster=mock_poster,
            sync=True,
        )

        assert res is True
        assert log_file.exists()

    def test_sink_failure_without_log_path_returns_false(self):
        mock_poster = MagicMock(return_value=False)
        record = {"turn_id": "fail-1", "reply": "fail"}

        res = emit_turn_transcript(record, sink_url=DOC_SINK_URL, http_poster=mock_poster, sync=True)
        assert res is False

    def test_real_http_sink_server(self):
        received_records = []

        class SinkHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received_records.append(json.loads(body.decode("utf-8")))
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), SinkHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{server.server_port}/transcripts"
            record = {"turn_id": "real-server-1", "prompt": "Hello", "reply": "World"}

            res = emit_turn_transcript(record, sink_url=url, http_poster=None, sync=True)

            assert res is True
            assert len(received_records) == 1
            assert received_records[0] == record
        finally:
            server.shutdown()
            server.server_close()


# ============================================================================
# 4. Tests for gateway integration
# ============================================================================


class TestGatewayIntegration:
    def test_record_reply_preserves_ambient_state(self):
        gateway._ambient_reset_state()
        gateway.record_reply("node-42", "Are you ready?")

        with gateway._ambient_lock:
            assert "node-42" in gateway._ambient_last_reply
            ts, reply = gateway._ambient_last_reply["node-42"]
            assert reply == "Are you ready?"
            assert ts > 0

    def test_reply_expects_reply_integration(self):
        assert gateway.reply_expects_reply("Would you like me to continue? 🤔") is True
        assert gateway.reply_expects_reply("I have finished the job.") is False

    def test_side_channel_wiring_in_gateway(self, tmp_path, monkeypatch):
        log_file = tmp_path / "gateway_transcripts.jsonl"
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_file))

        record = {
            "kind": "turn",
            "turn_id": "gw-turn-1",
            "node_id": "test-node",
            "prompt": "What is the weather?",
            "reply": "It is sunny today.",
        }

        success = gateway.emit_turn_transcript(record)
        assert success is True
        assert flush_side_channel_queue(2.0) is True
        assert log_file.exists()

        data = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert data["turn_id"] == "gw-turn-1"
        assert data["reply"] == "It is sunny today."

    def test_handle_ask_sse_expects_reply_true(self, monkeypatch):
        monkeypatch.setenv("MVG_ENABLE_MIC_REOPEN", "1")
        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "Would you like coffee? 🤔")

            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Hello", "node_id": "node-1"}).encode("utf-8")
            conn.request("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200

            data = resp.read().decode("utf-8")
            conn.close()

            events = []
            ev_name = None
            for line in data.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((ev_name, json.loads(line.split(":", 1)[1])))

            reply_ev = next(d for name, d in events if name == "reply")
            done_ev = next(d for name, d in events if name == "done")

            assert reply_ev["expects_reply"] is True
            assert done_ev["expects_reply"] is True
        finally:
            srv.shutdown()
            srv.server_close()

    def test_handle_ask_sse_expects_reply_false(self, monkeypatch):
        monkeypatch.setenv("MVG_ENABLE_MIC_REOPEN", "1")
        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "The capital of France is Paris.")

            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Hello", "node_id": "node-1"}).encode("utf-8")
            conn.request("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200

            data = resp.read().decode("utf-8")
            conn.close()

            events = []
            ev_name = None
            for line in data.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((ev_name, json.loads(line.split(":", 1)[1])))

            reply_ev = next(d for name, d in events if name == "reply")
            done_ev = next(d for name, d in events if name == "done")

            assert reply_ev["expects_reply"] is False
            assert done_ev["expects_reply"] is False
        finally:
            srv.shutdown()
            srv.server_close()

    def test_handle_ask_sse_default_omits_expects_reply(self, monkeypatch):
        monkeypatch.delenv("MVG_ENABLE_MIC_REOPEN", raising=False)
        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "Would you like coffee? 🤔")

            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Hello", "node_id": "node-1"}).encode("utf-8")
            conn.request("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200

            data = resp.read().decode("utf-8")
            conn.close()

            events = []
            ev_name = None
            for line in data.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((ev_name, json.loads(line.split(":", 1)[1])))

            reply_ev = next(d for name, d in events if name == "reply")
            done_ev = next(d for name, d in events if name == "done")

            assert "expects_reply" not in reply_ev
            assert "expects_reply" not in done_ev
        finally:
            srv.shutdown()
            srv.server_close()

    def test_handle_ask_stream_sse_expects_reply_gated(self, monkeypatch):
        def fake_stream_brain(*args, **kwargs):
            on_partial = kwargs.get("on_partial")
            text = "Shall we proceed with the plan? 🤔"
            if on_partial:
                on_partial(text)
            return text

        monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", fake_stream_brain)

        # 1. Unset: expects_reply omitted
        monkeypatch.delenv("MVG_ENABLE_MIC_REOPEN", raising=False)
        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Hello", "node_id": "node-1", "voice": False}).encode("utf-8")
            conn.request("POST", "/ask/stream", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            data = resp.read().decode("utf-8")
            conn.close()

            events = []
            ev_name = None
            for line in data.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((ev_name, json.loads(line.split(":", 1)[1])))

            reply_ev = next(d for name, d in events if name == "reply")
            done_ev = next(d for name, d in events if name == "done")
            assert "expects_reply" not in reply_ev
            assert "expects_reply" not in done_ev
        finally:
            srv.shutdown()
            srv.server_close()

        # 2. Opt-in MVG_ENABLE_MIC_REOPEN=1: expects_reply included
        monkeypatch.setenv("MVG_ENABLE_MIC_REOPEN", "1")
        srv2 = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        port2 = srv2.server_address[1]

        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port2, timeout=5)
            body = json.dumps({"prompt": "Hello", "node_id": "node-1", "voice": False}).encode("utf-8")
            conn.request("POST", "/ask/stream", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            data = resp.read().decode("utf-8")
            conn.close()

            events = []
            ev_name = None
            for line in data.splitlines():
                if line.startswith("event:"):
                    ev_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((ev_name, json.loads(line.split(":", 1)[1])))

            reply_ev = next(d for name, d in events if name == "reply")
            done_ev = next(d for name, d in events if name == "done")
            assert reply_ev["expects_reply"] is True
            assert done_ev["expects_reply"] is True
        finally:
            srv2.shutdown()
            srv2.server_close()

    def test_handle_ask_with_artifact_webhook_and_transcript_log(self, tmp_path, monkeypatch):
        received_webhooks = []

        class MockWebhook(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                received_webhooks.append(json.loads(self.rfile.read(content_length).decode("utf-8")))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        wh_server = HTTPServer(("127.0.0.1", 0), MockWebhook)
        threading.Thread(target=wh_server.serve_forever, daemon=True).start()
        wh_port = wh_server.server_port

        log_path = tmp_path / "ask_transcript.jsonl"
        monkeypatch.setenv("MVG_ARTIFACT_WEBHOOK_URL", f"http://127.0.0.1:{wh_port}/hook")
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_path))
        monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "Here is https://example.com/doc for details.")

        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Documentation", "node_id": "desk"}).encode("utf-8")
            conn.request("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            _ = resp.read().decode("utf-8")
            conn.close()

            # Flush async side-channel queue
            assert flush_side_channel_queue(2.0) is True

            # Verify artifact forwarded
            assert len(received_webhooks) == 1
            assert "https://example.com/doc" in received_webhooks[0]["content"]

            # Verify transcript logged
            assert log_path.exists()
            log_entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            assert log_entry["endpoint"] == "/ask"
            assert log_entry["node_id"] == "desk"
            assert "Sending the link to Discord." in log_entry["reply"]
        finally:
            srv.shutdown()
            srv.server_close()
            wh_server.shutdown()
            wh_server.server_close()


# ============================================================================
# 5. Tests for Security & Hardening
# ============================================================================


class TestSecurityAndHardening:
    def test_url_validation_http_https_only(self):
        assert _validate_url("http://example.com/webhook") is True
        assert _validate_url("https://example.com/webhook") is True
        assert _validate_url("ftp://example.com/webhook") is False
        assert _validate_url("file:///etc/passwd") is False
        assert _validate_url("javascript:alert(1)") is False
        assert _validate_url("") is False
        assert _validate_url(None) is False  # type: ignore

    def test_forward_artifacts_rejects_non_http_schemes(self):
        raw = "Check https://example.com"
        spoken = "Check the site"
        mock_poster = MagicMock()

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url="file:///etc/shadow",
            http_poster=mock_poster,
            sync=True,
        )
        assert sent is False
        assert res_spoken == spoken
        mock_poster.assert_not_called()

    def test_transcript_sink_rejects_non_http_schemes(self):
        mock_poster = MagicMock()
        res = emit_turn_transcript(
            {"turn_id": "bad-scheme"},
            sink_url="ftp://example.com/dump",
            http_poster=mock_poster,
            sync=True,
        )
        assert res is False
        mock_poster.assert_not_called()

    def test_safe_redirect_handler_blocks_downgrade_and_disallowed_schemes(self):
        handler = SafeRedirectHandler()
        req_https = urllib.request.Request("https://secure.example.com/api")
        # HTTPS -> HTTPS allowed
        r1 = handler.redirect_request(req_https, None, 302, "Found", {}, "https://secure.example.com/api2")
        assert r1 is not None and r1.full_url == "https://secure.example.com/api2"

        # HTTPS -> HTTP downgrade blocked
        r2 = handler.redirect_request(req_https, None, 302, "Found", {}, "http://insecure.example.com/api")
        assert r2 is None

        # Redirect to non-http scheme blocked
        req_http = urllib.request.Request("http://example.com/api")
        r3 = handler.redirect_request(req_http, None, 302, "Found", {}, "ftp://example.com/file")
        assert r3 is None

    def test_transcript_sink_sends_bearer_auth_token(self, monkeypatch):
        received_headers = {}

        class AuthSinkHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                received_headers["authorization"] = self.headers.get("Authorization")
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), AuthSinkHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            sink_url = f"http://127.0.0.1:{server.server_port}/transcripts"
            monkeypatch.setenv("MVG_TRANSCRIPT_AUTH_TOKEN", "secret-bearer-99")

            res = emit_turn_transcript(
                {"turn_id": "auth-test"},
                sink_url=sink_url,
                sync=True,
            )
            assert res is True
            assert received_headers.get("authorization") == "Bearer secret-bearer-99"
        finally:
            server.shutdown()
            server.server_close()

    def test_transcript_log_permissions_0o600(self, tmp_path):
        log_file = tmp_path / "secure_transcripts.jsonl"
        res = emit_turn_transcript({"turn_id": "perm-test"}, log_path=str(log_file), sync=True)
        assert res is True
        assert log_file.exists()

        file_mode = stat.S_IMODE(os.stat(str(log_file)).st_mode)
        assert file_mode == 0o600

    def test_transcript_log_size_cap_and_rotation(self, tmp_path):
        log_file = tmp_path / "rotating_transcripts.jsonl"
        record1 = {"turn_id": "r1", "msg": "First record"}
        record2 = {"turn_id": "r2", "msg": "Second record"}

        # Write first record with small max_bytes
        emit_turn_transcript(record1, log_path=str(log_file), max_bytes=40, sync=True)
        assert log_file.exists()

        # Write second record, causing rotation to .1
        emit_turn_transcript(record2, log_path=str(log_file), max_bytes=40, sync=True)
        backup_file = tmp_path / "rotating_transcripts.jsonl.1"
        assert backup_file.exists()
        assert log_file.exists()

        rec1_data = json.loads(backup_file.read_text(encoding="utf-8").strip())
        rec2_data = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert rec1_data["turn_id"] == "r1"
        assert rec2_data["turn_id"] == "r2"


# ============================================================================
# 6. Tests for Voice Turn End Transcript Capture
# ============================================================================


class TestVoiceTurnEndTranscript:
    def test_voice_turn_end_captures_stt_prompt_and_spoken_reply(self, tmp_path, monkeypatch):
        log_path = tmp_path / "voice_turn_transcript.jsonl"
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_path))
        monkeypatch.setenv("MVG_ENABLE_MIC_REOPEN", "1")

        stt_box = {"text": "What time is the meeting?"}
        transcript = "What time is the meeting?"
        spoken_reply = "The meeting is at three. Do you need a ride?"
        node_id = "kitchen-hub"
        session_id = "sess-abc"
        turn_id = "turn-123"
        t0 = time.time()
        log_extra = {"node": node_id}

        def end(outcome: str, reply: str = "", **extra) -> None:
            prompt_text = extra.pop("prompt", None) or extra.pop("transcript", None) or transcript or stt_box.get("text", "")
            final_reply = reply or extra.pop("reply", "") or spoken_reply
            mic_reopen_on = os.environ.get("MVG_ENABLE_MIC_REOPEN", "0").strip().lower() in ("1", "true", "yes", "on")
            tr_entry = {
                "kind": "turn",
                "endpoint": "/voice/turn",
                "node_id": node_id or None,
                "session_id": session_id or None,
                "turn_id": turn_id,
                "prompt": prompt_text or "",
                "transcript": prompt_text or "",
                "reply": final_reply or "",
                "outcome": outcome,
                "total_ms": round((time.time() - t0) * 1000, 1),
                "timestamp": time.time(),
                **log_extra,
                **extra,
            }
            if mic_reopen_on and final_reply:
                tr_entry["expects_reply"] = reply_expects_reply(final_reply)
            emit_turn_transcript(tr_entry, sync=True)

        end("ok:ignored:not-addressed", score=0.45)
        assert log_path.exists()
        entry = json.loads(log_path.read_text(encoding="utf-8").strip())

        assert entry["prompt"] == "What time is the meeting?"
        assert entry["transcript"] == "What time is the meeting?"
        assert entry["reply"] == "The meeting is at three. Do you need a ride?"
        assert entry["expects_reply"] is True
        assert entry["outcome"] == "ok:ignored:not-addressed"
        assert entry["score"] == 0.45


# ============================================================================
# 7. Tests for Async Worker Queue
# ============================================================================


class TestAsyncWorkerQueue:
    def test_forward_artifacts_async_queue_enqueues_and_flushes(self):
        raw = "See https://example.com/repo"
        spoken = "See the repo"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
            sync=False,
        )

        assert sent is True
        assert res_spoken == "See the repo. Sending the link to Discord."
        assert flush_side_channel_queue(2.0) is True
        mock_poster.assert_called_once()

    def test_emit_turn_transcript_async_queue_enqueues_and_flushes(self, tmp_path):
        log_file = tmp_path / "async_queue_test.jsonl"
        record = {"turn_id": "async-1", "prompt": "Hello", "reply": "World"}

        assert emit_turn_transcript(record, log_path=str(log_file), sync=False) is True
        assert flush_side_channel_queue(2.0) is True
        assert log_file.exists()
        assert json.loads(log_file.read_text(encoding="utf-8").strip()) == record

    def test_sync_helpers(self):
        mock_poster = MagicMock(return_value=True)
        res_spoken, sent = forward_artifacts_to_webhook_sync(
            "Look at https://example.com",
            "Look at it",
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )
        assert sent is True
        assert "Sending the link to Discord." in res_spoken
        mock_poster.assert_called_once()

        mock_sink = MagicMock(return_value=True)
        assert emit_turn_transcript_sync(
            {"turn_id": "sync-helper"},
            sink_url=DOC_SINK_URL,
            http_poster=mock_sink,
        ) is True
        mock_sink.assert_called_once()

    def test_worker_loop_logs_exceptions(self, caplog):
        caplog.set_level(logging.ERROR)

        def failing_task():
            raise RuntimeError("Deliberate background worker task failure")

        assert _enqueue_task(failing_task) is True
        assert flush_side_channel_queue(2.0) is True

        assert "Deliberate background worker task failure" in caplog.text
        assert "SideChannelWebhookWorker" in caplog.text or "Error in" in caplog.text

    def test_enqueue_task_logs_dropped_tasks_when_queue_is_full(self, caplog):
        caplog.set_level(logging.ERROR)
        small_q = queue.Queue(maxsize=1)
        small_q.put_nowait(("dummy", (), {}))

        def dummy_task():
            pass

        enqueued = _enqueue_task(dummy_task, task_queue=small_q)
        assert enqueued is False
        assert "queue full" in caplog.text.lower()
        assert "dropped task" in caplog.text.lower()

    def test_slow_webhook_does_not_delay_transcript_emission(self, tmp_path):
        webhook_running = threading.Event()
        webhook_release = threading.Event()

        def slow_poster(*args, **kwargs):
            webhook_running.set()
            webhook_release.wait(timeout=3.0)
            return True

        # Enqueue slow webhook task
        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply="See https://example.com/slow",
            spoken_reply="See link",
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=slow_poster,
            sync=False,
        )
        assert sent is True
        assert webhook_running.wait(timeout=2.0) is True

        # While webhook worker is blocked, emit turn transcript to local file
        log_file = tmp_path / "independent_transcript.jsonl"
        record = {"turn_id": "fast-transcript", "text": "Instant"}
        assert emit_turn_transcript(record, log_path=str(log_file), sync=False) is True

        # Transcript worker must process the task independently without waiting for webhook
        t0 = time.time()
        written = False
        while time.time() - t0 < 1.0:
            if log_file.exists() and log_file.stat().st_size > 0:
                written = True
                break
            time.sleep(0.01)

        assert written is True, "Transcript emission was delayed by stalled webhook worker!"
        data = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert data["turn_id"] == "fast-transcript"

        # Release slow webhook and flush both queues
        webhook_release.set()
        assert flush_side_channel_queue(2.0) is True

    def test_reset_side_channel_queue_drains_both_queues(self):
        # Insert raw items directly into queues
        _webhook_queue.put_nowait(("item1", (), {}))
        _transcript_queue.put_nowait(("item2", (), {}))

        reset_side_channel_queue()

        assert _webhook_queue.empty() is True
        assert _transcript_queue.empty() is True

    def test_flush_side_channel_queue_timeout_returns_false(self):
        block_event = threading.Event()

        def blocking_task():
            block_event.wait(timeout=2.0)

        assert _enqueue_task(blocking_task) is True
        # Flushes with a tiny timeout should return False because task is still unfinished
        assert flush_side_channel_queue(timeout_s=0.05) is False

        # Unblock and flush properly
        block_event.set()
        assert flush_side_channel_queue(timeout_s=2.0) is True

#!/usr/bin/env python3
"""tests/test_defer_and_auth.py — Tests for Deferred Async Answers,
Bearer Token Auth, and [[defer]] marker stripping.
"""

from __future__ import annotations

import collections
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import patch

import pytest

import gateway
from pending_and_auth import (
    BearerTokenVerifier,
    PendingVoiceAnswerStore,
    get_device_token_verifier,
    get_pending_store,
    has_defer_marker,
    strip_defer_marker,
)

WAV = gateway.pcm_to_wav(b"\x01\x00" * 2400, 24000)


ORIG_SYNTHESIZE = gateway.synthesize


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset_env_and_state(monkeypatch, tmp_path):
    gateway._ambient_reset_state()
    get_pending_store().clear()
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "Hi there!")
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    monkeypatch.delenv("MVG_ENABLE_DEFER", raising=False)
    yield
    gateway._ambient_reset_state()
    get_pending_store().clear()


# ============================================================================
# Unit Tests: BearerTokenVerifier
# ============================================================================

def test_bearer_verifier_disabled_when_empty(monkeypatch):
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    verifier = BearerTokenVerifier()
    assert not verifier.is_enabled
    # When disabled, all requests pass
    assert verifier.verify(None, "node1")
    assert verifier.verify("", "node1")
    assert verifier.verify("Bearer any-token", "node1")
    assert verifier.verify_token("any-token", "node1")


def test_bearer_verifier_single_shared_token():
    verifier = BearerTokenVerifier(tokens_str="secret_shared_123")
    assert verifier.is_enabled

    # Allowed with correct token for any node
    assert verifier.verify("Bearer secret_shared_123", "kitchen")
    assert verifier.verify("bearer secret_shared_123", "office")
    assert verifier.verify("BEARER secret_shared_123", "")
    assert verifier.verify_token("secret_shared_123", "kitchen")

    # Denied with incorrect or missing token
    assert not verifier.verify("Bearer wrong_token", "kitchen")
    assert not verifier.verify("Bearer ", "kitchen")
    assert not verifier.verify("", "kitchen")
    assert not verifier.verify(None, "kitchen")
    assert not verifier.verify_token("wrong_token", "kitchen")
    assert not verifier.verify_token("", "kitchen")


def test_bearer_verifier_per_device_tokens():
    verifier = BearerTokenVerifier(tokens_str="node1:tok_alpha,node2:tok_beta")
    assert verifier.is_enabled

    # Correct node and matching token
    assert verifier.verify("Bearer tok_alpha", "node1")
    assert verifier.verify("Bearer tok_beta", "node2")

    # Mismatched tokens
    assert not verifier.verify("Bearer tok_beta", "node1")
    assert not verifier.verify("Bearer tok_alpha", "node2")

    # Unknown node
    assert not verifier.verify("Bearer tok_alpha", "node3")
    assert not verifier.verify("Bearer tok_alpha", "")

    # Malformed headers
    assert not verifier.verify("Basic dXNlcjpwYXNz", "node1")
    assert not verifier.verify("", "node1")
    assert not verifier.verify(None, "node1")


def test_bearer_verifier_per_device_with_shared_fallback():
    verifier = BearerTokenVerifier(tokens_str="node1:tok_alpha,fallback_shared_tok")
    assert verifier.is_enabled

    # node1 requires tok_alpha
    assert verifier.verify("Bearer tok_alpha", "node1")
    assert not verifier.verify("Bearer fallback_shared_tok", "node1")

    # other nodes accept fallback_shared_tok
    assert verifier.verify("Bearer fallback_shared_tok", "node2")
    assert verifier.verify("Bearer fallback_shared_tok", "node3")
    assert verifier.verify("Bearer fallback_shared_tok", "")


def test_bearer_verifier_dynamic_env(monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "dev_node:secret999")
    verifier = BearerTokenVerifier()  # no tokens_str passed, reads env
    assert verifier.is_enabled
    assert verifier.verify("Bearer secret999", "dev_node")
    assert not verifier.verify("Bearer wrong", "dev_node")

    # Update env
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "new_shared_token")
    assert verifier.verify("Bearer new_shared_token", "any_node")


def test_bearer_verifier_constant_time(monkeypatch):
    verifier = BearerTokenVerifier("secure_token")
    compare_calls = []

    real_compare = gateway.hmac.compare_digest

    def mock_compare(a, b):
        compare_calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("pending_and_auth.hmac.compare_digest", mock_compare)

    # Valid token
    assert verifier.verify("Bearer secure_token", "node1")
    assert len(compare_calls) == 1
    assert compare_calls[-1] == (b"secure_token", b"secure_token")

    # Invalid token (still calls compare_digest)
    compare_calls.clear()
    assert not verifier.verify("Bearer wrong_token", "node1")
    assert len(compare_calls) == 1
    assert compare_calls[-1] == (b"wrong_token", b"secure_token")

    # Missing header (still performs dummy compare_digest to prevent timing leak)
    compare_calls.clear()
    assert not verifier.verify(None, "node1")
    assert len(compare_calls) == 1


# ============================================================================
# Unit Tests: [[defer]] Marker Detection & Stripping
# ============================================================================

def test_has_defer_marker():
    assert has_defer_marker("Checking your request now. [[defer]]")
    assert has_defer_marker("[[defer]]")
    assert has_defer_marker("[[DEFER]]")
    assert has_defer_marker("[[ defer ]]")
    assert has_defer_marker("Prefix [[DeFeR]] Suffix")

    assert not has_defer_marker("Normal reply without defer.")
    assert not has_defer_marker("[defer]")
    assert not has_defer_marker("")
    assert not has_defer_marker(None)
    assert not has_defer_marker(123)  # non-string


def test_strip_defer_marker():
    assert strip_defer_marker("I will check that for you. [[defer]]") == "I will check that for you."
    assert strip_defer_marker("[[defer]] I will check that for you.") == "I will check that for you."
    assert strip_defer_marker("Checking [[defer]] your flight.") == "Checking your flight."
    assert strip_defer_marker("Multiple [[defer]] markers [[DEFER]] removed.") == "Multiple markers removed."
    assert strip_defer_marker("[[defer]]") == ""
    assert strip_defer_marker("[[ defer ]]") == ""
    assert strip_defer_marker("   [[DEFER]]   ") == ""
    assert strip_defer_marker("Normal text") == "Normal text"
    assert strip_defer_marker("") == ""
    assert strip_defer_marker(None) == ""


# ============================================================================
# Unit Tests: PendingVoiceAnswerStore
# ============================================================================

def test_pending_store_lifecycle():
    store = PendingVoiceAnswerStore(max_per_node=10)

    # Empty initially
    assert store.peek_for_node("node1") is None
    assert store.pop_for_node("node1") is None
    assert store.count_for_node("node1") == 0

    # Store first item
    item1 = store.store("node1", prompt="What's the weather?", answer="72 and sunny", topic="weather")
    assert item1["node_id"] == "node1"
    assert item1["prompt"] == "What's the weather?"
    assert item1["answer"] == "72 and sunny"
    assert item1["topic"] == "weather"
    assert "id" in item1
    assert "created_at" in item1
    assert store.count_for_node("node1") == 1

    # Store second item for same node
    item2 = store.store("node1", prompt="Any new mail?", answer="You have 2 emails", topic="email")
    assert store.count_for_node("node1") == 2

    # Peek returns oldest item without popping
    peeked1 = store.peek_for_node("node1")
    assert peeked1["id"] == item1["id"]
    assert peeked1["answer"] == "72 and sunny"
    assert store.count_for_node("node1") == 2

    # Pop returns items in FIFO order
    popped1 = store.pop_for_node("node1")
    assert popped1["id"] == item1["id"]
    assert store.count_for_node("node1") == 1

    # Next peek shows item2
    peeked2 = store.peek_for_node("node1")
    assert peeked2["id"] == item2["id"]

    popped2 = store.pop_for_node("node1")
    assert popped2["id"] == item2["id"]
    assert store.count_for_node("node1") == 0

    # Empty again
    assert store.pop_for_node("node1") is None
    assert store.peek_for_node("node1") is None


def test_pending_store_multi_node_isolation():
    store = PendingVoiceAnswerStore()
    store.store("kitchen", prompt="Timer?", answer="5 minutes left")
    store.store("living_room", prompt="Score?", answer="Dodgers up 3-1")

    assert store.count_for_node("kitchen") == 1
    assert store.count_for_node("living_room") == 1

    kitchen_item = store.pop_for_node("kitchen")
    assert kitchen_item["answer"] == "5 minutes left"
    assert store.count_for_node("kitchen") == 0

    # living_room remains intact
    assert store.count_for_node("living_room") == 1
    lr_item = store.pop_for_node("living_room")
    assert lr_item["answer"] == "Dodgers up 3-1"
    assert store.count_for_node("living_room") == 0


def test_pending_store_clear():
    store = PendingVoiceAnswerStore()
    store.store("nodeA", "p1", "a1")
    store.store("nodeB", "p2", "a2")

    store.clear("nodeA")
    assert store.count_for_node("nodeA") == 0
    assert store.count_for_node("nodeB") == 1

    store.clear()
    assert store.count_for_node("nodeB") == 0


def test_pending_store_thread_safety():
    store = PendingVoiceAnswerStore()
    errors = []

    def producer(node, count):
        try:
            for i in range(count):
                store.store(node, prompt=f"prompt_{i}", answer=f"ans_{i}")
        except Exception as e:
            errors.append(e)

    def consumer(node, count):
        try:
            popped = 0
            while popped < count:
                item = store.pop_for_node(node)
                if item is not None:
                    popped += 1
                else:
                    time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=producer, args=("node_concurrent", 100)),
        threading.Thread(target=consumer, args=("node_concurrent", 100)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors
    assert store.count_for_node("node_concurrent") == 0


# ============================================================================
# Integration Tests: Gateway Pending Endpoints (GET & POST /voice/pending)
# ============================================================================

def test_pending_endpoints_disabled_by_default(server):
    port = server
    # Without MVG_ENABLE_DEFER=1, both GET and POST should return 404
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/voice/pending?node=node1")
    resp = conn.getresponse()
    assert resp.status == 404

    conn.request(
        "POST",
        "/voice/pending",
        body=json.dumps({"node_id": "node1", "answer": "test"}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 404


def test_pending_endpoints_lifecycle(server, monkeypatch):
    monkeypatch.setenv("MVG_ENABLE_DEFER", "1")
    port = server

    # 1. Missing node parameter on GET -> 400
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", "/voice/pending")
    resp = conn.getresponse()
    assert resp.status == 400
    data = json.loads(resp.read().decode("utf-8"))
    assert "node" in data.get("error", "").lower()

    # 2. GET when no pending items -> has_pending=False
    conn.request("GET", "/voice/pending?node=node1")
    resp = conn.getresponse()
    assert resp.status == 200
    data = json.loads(resp.read().decode("utf-8"))
    assert data["has_pending"] is False
    assert data["pending"] is None

    # 3. POST /voice/pending with missing fields -> 400
    conn.request("POST", "/voice/pending", body=json.dumps({"prompt": "hi"}), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 400

    conn.request("POST", "/voice/pending", body=json.dumps({"node_id": "node1"}), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 400

    # 4. POST /voice/pending valid -> 200 stored
    payload = {
        "node_id": "node1",
        "prompt": "What time is the game?",
        "answer": "The game starts at 7:05 PM.",
        "topic": "sports",
    }
    conn.request("POST", "/voice/pending", body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    stored_data = json.loads(resp.read().decode("utf-8"))
    assert stored_data["status"] == "ok"
    assert stored_data["pending"]["answer"] == "The game starts at 7:05 PM."

    # 5. GET /voice/pending with peek=1 does not remove the item
    conn.request("GET", "/voice/pending?node=node1&peek=1")
    resp = conn.getresponse()
    assert resp.status == 200
    peek_data = json.loads(resp.read().decode("utf-8"))
    assert peek_data["has_pending"] is True
    assert peek_data["pending"]["answer"] == "The game starts at 7:05 PM."

    # 6. GET /voice/pending default (pop=1) pops the item
    conn.request("GET", "/voice/pending?node=node1")
    resp = conn.getresponse()
    assert resp.status == 200
    pop_data = json.loads(resp.read().decode("utf-8"))
    assert pop_data["has_pending"] is True
    assert pop_data["pending"]["answer"] == "The game starts at 7:05 PM."
    assert pop_data["popped"] is True

    # 7. Next GET returns has_pending=False
    conn.request("GET", "/voice/pending?node=node1")
    resp = conn.getresponse()
    assert resp.status == 200
    empty_data = json.loads(resp.read().decode("utf-8"))
    assert empty_data["has_pending"] is False
    assert empty_data["pending"] is None


# ============================================================================
# Integration Tests: Gateway Bearer Auth on /voice/turn
# ============================================================================

def _post_multipart_turn(port: int, fields: dict, files: dict, headers: dict | None = None) -> tuple[int, bytes]:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for k, v in fields.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))
    for k, (filename, data, content_type) in files.items():
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(data)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    conn = http.client.HTTPConnection("127.0.0.1", port)
    req_headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    if headers:
        req_headers.update(headers)
    conn.request("POST", "/voice/turn", body=bytes(body), headers=req_headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_voice_turn_bearer_auth_disabled_by_default(server, monkeypatch):
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "hello")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *args, **kwargs: "Hi there!")

    # Without tokens configured, turn proceeds without Authorization header
    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "living_room", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
    )
    assert status == 200
    assert b"Hi there!" in body


def test_voice_turn_bearer_auth_enforced(server, monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "living_room:tok_live_123,kitchen:tok_kitch_456")
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "hello")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *args, **kwargs: "Hi there!")

    # 1. No Authorization header -> 401
    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "living_room", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
    )
    assert status == 401
    assert b"unauthorized" in body

    # 2. Invalid token -> 401
    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "living_room", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
        headers={"Authorization": "Bearer wrong_token"},
    )
    assert status == 401
    assert b"unauthorized" in body

    # 3. Wrong node's token -> 401
    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "living_room", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
        headers={"Authorization": "Bearer tok_kitch_456"},
    )
    assert status == 401
    assert b"unauthorized" in body

    # 4. Valid token -> 200
    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "living_room", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
        headers={"Authorization": "Bearer tok_live_123"},
    )
    assert status == 200
    assert b"Hi there!" in body


# ============================================================================
# Integration Tests: [[defer]] Marker Stripping in Gateway & Record Reply
# ============================================================================

def test_record_reply_strips_defer_marker():
    gateway.record_reply("test_node", "Checking the garage door now. [[defer]]")
    with gateway._ambient_lock:
        _, recorded_text = gateway._ambient_last_reply["test_node"]
        assert "[[defer]]" not in recorded_text
        assert recorded_text == "Checking the garage door now."

        # Check rolling context deque
        dq = gateway._ambient_context["test_node"]
        assert len(dq) == 1
        _, speaker, context_text = dq[0]
        assert speaker == gateway.AGENT_LABEL
        assert "[[defer]]" not in context_text
        assert context_text == "Checking the garage door now."


def test_synthesize_strips_defer_marker(monkeypatch):
    spoken = []

    def mock_backend(text: str):
        spoken.append(text)
        return WAV

    monkeypatch.setattr(gateway, "TTS_BACKENDS", {"mock": mock_backend})
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["mock"])
    monkeypatch.setattr(gateway, "synthesize", ORIG_SYNTHESIZE)

    wav, engine = gateway.synthesize("Looking into that now. [[defer]]")
    assert engine == "mock"
    assert len(spoken) == 1
    assert "[[defer]]" not in spoken[0]
    assert spoken[0] == "Looking into that now."


def test_voice_turn_strips_defer_marker_from_reply(server, monkeypatch):
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "check my flight")
    monkeypatch.setattr(
        gateway,
        "STREAM_BRAIN_CALL",
        lambda *args, **kwargs: "I am checking your flight details now. [[defer]]",
    )

    status, body = _post_multipart_turn(
        server,
        fields={"node_id": "node_defer", "tts": "none"},
        files={"audio": ("audio.wav", WAV, "audio/wav")},
    )
    assert status == 200
    # Marker must be stripped from the SSE reply event
    assert b"[[defer]]" not in body
    assert b"I am checking your flight details now." in body

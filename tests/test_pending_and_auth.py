#!/usr/bin/env python3
"""tests/test_pending_and_auth.py — Tests for Deferred Async Answers,
Bearer Token Authentication & Lockout across all Gateway endpoints.
"""

from __future__ import annotations

import http.client
import json
import threading

import pytest

import gateway
from pending_and_auth import (
    BearerTokenVerifier,
    PendingVoiceAnswerStore,
    get_device_token_verifier,
    get_pending_store,
    has_defer_marker,
    is_defer_enabled,
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
    _v = get_device_token_verifier()
    _v.reset_failures()
    _v._tokens_str = None
    _v._max_failures = None
    _v._lockout_sec = None
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", lambda *a, **k: "Hi there!")
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    monkeypatch.delenv("MVG_ENABLE_DEFER", raising=False)
    monkeypatch.delenv("MVG_AUTH_MAX_FAILURES", raising=False)
    monkeypatch.delenv("MVG_AUTH_LOCKOUT_SEC", raising=False)
    yield
    gateway._ambient_reset_state()
    get_pending_store().clear()
    get_device_token_verifier().reset_failures()


# ============================================================================
# 1. BearerTokenVerifier Unit Tests
# ============================================================================

def test_bearer_verifier_disabled_when_empty(monkeypatch):
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    verifier = BearerTokenVerifier()
    assert not verifier.is_enabled
    assert verifier.verify(None, "node1")
    assert verifier.verify("", "node1")
    assert verifier.verify("Bearer any-token", "node1")
    assert verifier.verify_token("any-token", "node1")


def test_bearer_verifier_single_shared_token():
    verifier = BearerTokenVerifier(tokens_str="secret_shared_123")
    assert verifier.is_enabled
    assert verifier.verify("Bearer secret_shared_123", "kitchen")
    assert verifier.verify("bearer secret_shared_123", "office")
    assert verifier.verify_token("secret_shared_123", "kitchen")

    assert not verifier.verify("Bearer wrong_token", "kitchen")
    assert not verifier.verify("Bearer ", "kitchen")
    assert not verifier.verify("", "kitchen")
    assert not verifier.verify(None, "kitchen")


def test_bearer_verifier_per_device_tokens():
    verifier = BearerTokenVerifier(tokens_str="node1:tok_alpha,node2:tok_beta")
    assert verifier.is_enabled
    assert verifier.verify("Bearer tok_alpha", "node1")
    assert verifier.verify("Bearer tok_beta", "node2")

    assert not verifier.verify("Bearer tok_beta", "node1")
    assert not verifier.verify("Bearer tok_alpha", "node2")
    assert not verifier.verify("Bearer tok_alpha", "node3")


def test_bearer_verifier_shared_fallback():
    verifier = BearerTokenVerifier(tokens_str="node1:tok_alpha,shared:fallback_shared")
    assert verifier.is_enabled
    assert verifier.verify("Bearer tok_alpha", "node1")
    assert not verifier.verify("Bearer fallback_shared", "node1")
    assert verifier.verify("Bearer fallback_shared", "node2")
    assert verifier.verify("Bearer fallback_shared", "")


def test_bearer_verifier_lockout_unit():
    verifier = BearerTokenVerifier("secret_token", max_failures=3, lockout_sec=10.0)
    client = "198.51.100.1"

    assert not verifier.is_locked_out(client)

    verifier.record_failure(client)
    assert not verifier.is_locked_out(client)

    verifier.record_failure(client)
    assert not verifier.is_locked_out(client)

    verifier.record_failure(client)
    assert verifier.is_locked_out(client)

    # Success clears lockout
    verifier.record_success(client)
    assert not verifier.is_locked_out(client)

    # Reset clears all
    verifier.record_failure(client)
    verifier.record_failure(client)
    verifier.record_failure(client)
    assert verifier.is_locked_out(client)
    verifier.reset_failures()
    assert not verifier.is_locked_out(client)


# ============================================================================
# 2. Defer Marker & Helpers Unit Tests
# ============================================================================

def test_is_defer_enabled(monkeypatch):
    monkeypatch.delenv("MVG_ENABLE_DEFER", raising=False)
    assert not is_defer_enabled()
    for val in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv("MVG_ENABLE_DEFER", val)
        assert is_defer_enabled()
    for val in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MVG_ENABLE_DEFER", val)
        assert not is_defer_enabled()


def test_has_and_strip_defer_marker():
    text = "Processing your query. [[defer]] Please wait."
    assert has_defer_marker(text)
    assert strip_defer_marker(text) == "Processing your query. Please wait."

    assert has_defer_marker("[[ DEFER ]]")
    assert strip_defer_marker("[[ DEFER ]]") == ""

    assert not has_defer_marker("Normal text")
    assert strip_defer_marker("Normal text") == "Normal text"
    assert strip_defer_marker(None) == ""


# ============================================================================
# 3. PendingVoiceAnswerStore Unit Tests
# ============================================================================

def test_pending_store_lifecycle():
    store = PendingVoiceAnswerStore(max_per_node=5)
    assert store.count_for_node("nodeA") == 0
    assert store.pop_for_node("nodeA") is None

    store.store("nodeA", "prompt1", "answer1", "topic1")
    store.store("nodeA", "prompt2", "answer2", "topic2")
    assert store.count_for_node("nodeA") == 2

    peek = store.peek_for_node("nodeA")
    assert peek["answer"] == "answer1"
    assert store.count_for_node("nodeA") == 2

    popped1 = store.pop_for_node("nodeA")
    assert popped1["answer"] == "answer1"
    assert store.count_for_node("nodeA") == 1

    popped2 = store.pop_for_node("nodeA")
    assert popped2["answer"] == "answer2"
    assert store.count_for_node("nodeA") == 0


# ============================================================================
# 4. Gateway Multimodal Image Magic Byte Validation Tests
# ============================================================================

def test_validate_multimodal_raw_image_magic(monkeypatch):
    monkeypatch.setenv("MVG_VISION_ADPCM", "1")
    # Valid JPEG
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    assert gateway.validate_multimodal_payload({"image": jpeg}) == (None, None)
    assert gateway.validate_multimodal_payload({"image_bytes": jpeg}) == (None, None)

    # Valid PNG
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert gateway.validate_multimodal_payload({"image": png}) == (None, None)

    # Valid WEBP
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    assert gateway.validate_multimodal_payload({"image": webp}) == (None, None)

    # Invalid raw image bytes (no valid magic)
    bad_raw = b"NOT_AN_IMAGE_DATA_BYTES"
    code, msg = gateway.validate_multimodal_payload({"image": bad_raw})
    assert code == 400
    assert "valid image" in msg

    # Invalid burst_images item
    code, msg = gateway.validate_multimodal_payload({"burst_images": [jpeg, bad_raw]})
    assert code == 400
    assert "burst_images[1]" in msg


def test_validate_multimodal_is_noop_when_flags_off(monkeypatch):
    for k in ("MVG_ENABLE_VISION", "MVG_VISION_ENABLED", "MVG_VISION_ADPCM", "MVG_BRAIN_DIRECT_AUDIO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(gateway, "ENABLE_VISION", False)
    monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", False)
    assert gateway.validate_multimodal_payload({"image": b"junk", "image_b64": "%%%", "audio_b64": "%%%"}) == (None, None)


# ============================================================================
# 5. Endpoint Bearer Auth & Lockout Integration Tests Across ALL Endpoints
# ============================================================================

def _request(port: int, method: str, path: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port)
    req_headers = headers.copy() if headers else {}
    if body and "Content-Length" not in req_headers:
        req_headers["Content-Length"] = str(len(body))
    conn.request(method, path, body=body, headers=req_headers)
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_endpoint_bearer_auth_all_endpoints(server, monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "living_room:tok_live,kitchen:tok_kitch,shared:tok_shared")
    monkeypatch.setenv("MVG_ENABLE_DEFER", "1")
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "hello")
    monkeypatch.setattr(gateway, "proxy_embed", lambda wav: (200, b'{"embedding": [1.0]}'))
    port = server

    # 1. GET /voice/pending
    status, _ = _request(port, "GET", "/voice/pending?node=living_room")
    assert status == 401
    status, _ = _request(port, "GET", "/voice/pending?node=living_room", headers={"Authorization": "Bearer wrong"})
    assert status == 401
    status, _ = _request(port, "GET", "/voice/pending?node=living_room", headers={"Authorization": "Bearer tok_live"})
    assert status == 200

    # 2. POST /voice/pending
    post_body = json.dumps({"node_id": "living_room", "prompt": "p", "answer": "a"}).encode("utf-8")
    status, _ = _request(port, "POST", "/voice/pending", body=post_body, headers={"Content-Type": "application/json"})
    assert status == 401
    status, _ = _request(port, "POST", "/voice/pending", body=post_body, headers={"Content-Type": "application/json", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 3. POST /ask
    ask_body = json.dumps({"prompt": "hi", "node_id": "living_room"}).encode("utf-8")
    status, _ = _request(port, "POST", "/ask", body=ask_body, headers={"Content-Type": "application/json"})
    assert status == 401
    status, _ = _request(port, "POST", "/ask", body=ask_body, headers={"Content-Type": "application/json", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 4. POST /ask/stream
    ask_stream_body = json.dumps({"prompt": "hi", "node_id": "living_room"}).encode("utf-8")
    status, _ = _request(port, "POST", "/ask/stream", body=ask_stream_body, headers={"Content-Type": "application/json"})
    assert status == 401
    status, _ = _request(port, "POST", "/ask/stream", body=ask_stream_body, headers={"Content-Type": "application/json", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 5. POST /stt (now locked out after 5 failures: missing token returns 429, valid token returns 200)
    status, _ = _request(port, "POST", "/stt?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav"})
    assert status == 429
    status, _ = _request(port, "POST", "/stt?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 6. POST /embed
    status, _ = _request(port, "POST", "/embed?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav"})
    assert status == 429
    status, _ = _request(port, "POST", "/embed?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 7. POST /ambient
    status, _ = _request(port, "POST", "/ambient?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav"})
    assert status == 429
    status, _ = _request(port, "POST", "/ambient?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav", "Authorization": "Bearer tok_live"})
    assert status == 200

    # 8. POST /intent
    intent_body = json.dumps({"text": "turn on lights", "node": "living_room"}).encode("utf-8")
    status, _ = _request(port, "POST", "/intent", body=intent_body, headers={"Content-Type": "application/json"})
    assert status == 429
    status, _ = _request(port, "POST", "/intent", body=intent_body, headers={"Content-Type": "application/json", "Authorization": "Bearer tok_live"})
    # 204 or 200 (intent miss or match)
    assert status in (200, 204)

    # 9. POST /voice/turn
    status, _ = _request(port, "POST", "/voice/turn?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav"})
    assert status == 429
    status, _ = _request(port, "POST", "/voice/turn?node_id=living_room", body=WAV, headers={"Content-Type": "audio/wav", "Authorization": "Bearer wrong"})
    assert status == 429


def test_endpoint_lockout_returns_429(server, monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "living_room:tok_live")
    monkeypatch.setenv("MVG_AUTH_MAX_FAILURES", "3")
    monkeypatch.setenv("MVG_AUTH_LOCKOUT_SEC", "30.0")
    port = server

    ask_body = json.dumps({"prompt": "hi", "node_id": "living_room"}).encode("utf-8")
    headers_bad = {"Content-Type": "application/json", "Authorization": "Bearer wrong"}
    headers_good = {"Content-Type": "application/json", "Authorization": "Bearer tok_live"}

    # Fail 1
    s, _ = _request(port, "POST", "/ask", body=ask_body, headers=headers_bad)
    assert s == 401

    # Fail 2
    s, _ = _request(port, "POST", "/ask", body=ask_body, headers=headers_bad)
    assert s == 401

    # Fail 3
    s, _ = _request(port, "POST", "/ask", body=ask_body, headers=headers_bad)
    assert s == 401

    # Attempt 4: now locked out -> 429 for bad/missing tokens
    s, body = _request(port, "POST", "/ask", body=ask_body, headers=headers_bad)
    assert s == 429
    data = json.loads(body.decode("utf-8"))
    assert "locked out" in data.get("error", "").lower() or "too many" in data.get("error", "").lower()

    s, body = _request(port, "POST", "/ask", body=ask_body, headers={"Content-Type": "application/json"})
    assert s == 429
    data = json.loads(body.decode("utf-8"))
    assert "locked out" in data.get("error", "").lower() or "too many" in data.get("error", "").lower()

    # Valid token during lockout -> exempt from lockout, returns 200
    s, body = _request(port, "POST", "/ask", body=ask_body, headers=headers_good)
    assert s == 200


# ============================================================================
# 6. Deferred Answers Integration in Gateway
# ============================================================================

def test_defer_answer_stored_when_marker_present(server, monkeypatch):
    monkeypatch.setenv("MVG_ENABLE_DEFER", "1")
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "Looking into the status now. [[defer]]")
    port = server

    store = get_pending_store()
    store.clear()

    ask_body = json.dumps({"prompt": "check the pool pump", "node_id": "pool_node"}).encode("utf-8")
    status, body = _request(port, "POST", "/ask", body=ask_body, headers={"Content-Type": "application/json"})
    assert status == 200

    # Ensure answer was stored in pending store with marker stripped
    assert store.count_for_node("pool_node") == 1
    item = store.pop_for_node("pool_node")
    assert item["node_id"] == "pool_node"
    assert item["prompt"] == "check the pool pump"
    assert item["answer"] == "Looking into the status now."
    assert "[[defer]]" not in item["answer"]

    # SSE body must have marker stripped as well
    assert b"[[defer]]" not in body
    assert b"Looking into the status now." in body

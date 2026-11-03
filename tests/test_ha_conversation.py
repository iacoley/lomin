"""tests/test_ha_conversation.py — Home Assistant Conversation Fast-Path & Authorization.

Covers:
- PR #7 Blocker: `_is_ha_authorized` removes speaker shortcut so `{"speaker": "x"}`
  cannot bypass bearer token auth when `MVG_HA_INTENT_TOKEN` or `MVG_DEVICE_TOKENS` is set.
- `_intent`: respects `ha_authorized=False` by returning `{"hit": False, "reason": "ha_unauthorized"}`
  without invoking Home Assistant upstream.
- HTTP POST /intent: verifies bearer token enforcement, speaker bypass rejection,
  trusted claimed speaker auth, device token verifier integration, and lockout response (429).
"""

import http.client
import json
import threading

import pytest

import gateway
from gateway import _intent, _is_ha_authorized
from pending_and_auth import get_device_token_verifier, get_pending_store


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _reset_state():
    get_pending_store().clear()
    verifier = get_device_token_verifier()
    verifier._tokens_str = None
    with verifier._lock:
        verifier._failures.clear()
    yield
    get_pending_store().clear()
    verifier = get_device_token_verifier()
    verifier._tokens_str = None
    with verifier._lock:
        verifier._failures.clear()


def test_is_ha_authorized_tokens_unconfigured(monkeypatch):
    """When neither MVG_HA_INTENT_TOKEN nor MVG_DEVICE_TOKENS is set, all callers are authorized."""
    monkeypatch.delenv("MVG_HA_INTENT_TOKEN", raising=False)
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    # Ensure verifier singleton is disabled
    verifier = get_device_token_verifier()
    verifier._tokens_str = None
    assert not verifier.is_enabled

    assert _is_ha_authorized() is True
    assert _is_ha_authorized(node_id="kitchen", speaker="alice") is True
    assert _is_ha_authorized(auth_header=None, claimed_speaker="") is True


def test_is_ha_authorized_speaker_bypass_prohibited(monkeypatch):
    """PR #7 Blocker: passing speaker="x" must NEVER bypass auth when tokens are configured."""
    monkeypatch.setenv("MVG_HA_INTENT_TOKEN", "secret-ha-token")
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    verifier = get_device_token_verifier()
    verifier._tokens_str = None

    # 1. Plain speaker field without valid auth header -> MUST return False
    assert _is_ha_authorized(speaker="alice") is False
    assert _is_ha_authorized(speaker="admin", node_id="node1") is False

    # 2. Speaker field with wrong token -> MUST return False
    assert (
        _is_ha_authorized(
            speaker="alice",
            auth_header="Bearer wrong-token",
            node_id="node1",
        )
        is False
    )

    # 3. Speaker field with valid token -> True
    assert (
        _is_ha_authorized(
            speaker="alice",
            auth_header="Bearer secret-ha-token",
            node_id="node1",
        )
        is True
    )

    # 4. Valid token alone -> True
    assert _is_ha_authorized(auth_header="Bearer secret-ha-token") is True

    # 5. Invalid token alone -> False
    assert _is_ha_authorized(auth_header="Bearer wrong-token") is False


def test_is_ha_authorized_voice_id_and_trusted_speaker(monkeypatch):
    """Verified voice_id and trusted claimed_speaker are allowed to authorize."""
    monkeypatch.setenv("MVG_HA_INTENT_TOKEN", "secret-ha-token")

    # Trusted claimed speaker (from X-Voice-Claimed-Speaker validated against MVG_TRUSTED_KEY)
    assert _is_ha_authorized(claimed_speaker="alice") is True
    assert _is_ha_authorized(claimed_speaker="") is False

    # Voice ID status
    assert _is_ha_authorized(voice_id={"status": "verified", "speaker": "bob"}) is True
    assert _is_ha_authorized(voice_id={"status": "unverified", "speaker": "bob"}) is False
    assert _is_ha_authorized(voice_id={"status": "unknown"}) is False


def test_is_ha_authorized_device_verifier(monkeypatch):
    """DeviceTokenVerifier is checked when MVG_DEVICE_TOKENS is set."""
    monkeypatch.delenv("MVG_HA_INTENT_TOKEN", raising=False)
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "bed:tok-bed,living:tok-living,shared-tok")
    verifier = get_device_token_verifier()
    verifier._tokens_str = None
    assert verifier.is_enabled

    # Speaker without token must NOT bypass
    assert _is_ha_authorized(node_id="bed", speaker="alice") is False

    # Valid per-device token
    assert _is_ha_authorized(auth_header="Bearer tok-bed", node_id="bed") is True

    # Token for wrong device
    assert _is_ha_authorized(auth_header="Bearer tok-bed", node_id="living") is False

    # Shared fallback token
    assert _is_ha_authorized(auth_header="Bearer shared-tok", node_id="other") is True


def test_intent_ha_unauthorized_short_circuit(monkeypatch):
    """_intent returns ha_unauthorized immediately when ha_authorized is False without calling HA."""
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    called = []

    def mock_try_ha(text, has_images=False):
        called.append(text)
        return True, "Done", {"reason": "action_done"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_try_ha)

    # When ha_authorized=False, HA upstream must not be called
    res = _intent("turn on light", ha_authorized=False)
    assert res["hit"] is False
    assert res["reason"] == "ha_unauthorized"
    assert len(called) == 0

    # When ha_authorized=True, HA upstream is called
    res_auth = _intent("turn on light", ha_authorized=True)
    assert res_auth["hit"] is True
    assert res_auth["reply"] == "Done"
    assert len(called) == 1


def test_post_intent_speaker_cannot_bypass(server, monkeypatch):
    """PR #7 Blocker end-to-end: POST /intent with speaker='x' cannot bypass bearer auth."""
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setenv("MVG_HA_INTENT_TOKEN", "ha-guard-token")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    ha_calls = []

    def mock_ha(text, has_images=False, **kwargs):
        ha_calls.append(text)
        return True, "Light turned on", {"reason": "action_done", "response_type": "action_done"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_ha)

    port = server

    def _post(payload: dict, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        body = json.dumps(payload).encode("utf-8")
        h = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if headers:
            h.update(headers)
        conn.request("POST", "/intent", body=body, headers=h)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")

    # 1. Unauthenticated request without speaker -> 401
    status, body = _post({"text": "turn on lights", "node": "living"})
    assert status == 401
    assert "unauthorized" in body.lower()
    assert len(ha_calls) == 0

    # 2. Attacker passes speaker="alice" without auth header -> MUST return 401!
    status, body = _post({"text": "turn on lights", "node": "living", "speaker": "alice"})
    assert status == 401
    assert "unauthorized" in body.lower()
    assert len(ha_calls) == 0

    # 3. Attacker passes speaker="alice" with wrong token -> 401
    status, body = _post(
        {"text": "turn on lights", "node": "living", "speaker": "alice"},
        headers={"Authorization": "Bearer evil-token"},
    )
    assert status == 401
    assert len(ha_calls) == 0

    # 4. Legitimate request with valid bearer token -> 200
    status, body = _post(
        {"text": "turn on lights", "node": "living", "speaker": "alice"},
        headers={"Authorization": "Bearer ha-guard-token"},
    )
    assert status == 200
    assert "Light turned on" in body
    assert len(ha_calls) == 1


def test_post_intent_device_tokens_and_lockout(server, monkeypatch):
    """POST /intent respects MVG_DEVICE_TOKENS and enforces lockout (429) after max failures."""
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.delenv("MVG_HA_INTENT_TOKEN", raising=False)
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "bed:bed-token-123")
    monkeypatch.setenv("MVG_AUTH_MAX_FAILURES", "3")
    monkeypatch.setenv("MVG_AUTH_LOCKOUT_SEC", "30")

    verifier = get_device_token_verifier()
    monkeypatch.setattr(verifier, "_tokens_str", None)
    monkeypatch.setattr(verifier, "_max_failures", 3)
    monkeypatch.setattr(verifier, "_lockout_sec", 30.0)
    with verifier._lock:
        verifier._failures.clear()

    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)
    monkeypatch.setattr(
        gateway,
        "try_ha_conversation",
        lambda text, has_images=False: (True, "OK", {"reason": "action_done"}),
    )

    port = server

    def _post(node: str, auth: str | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        body = json.dumps({"text": "status", "node": node}).encode("utf-8")
        h = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if auth:
            h["Authorization"] = auth
        conn.request("POST", "/intent", body=body, headers=h)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")

    # 3 failures -> 401
    for _ in range(3):
        st, _ = _post("bed", "Bearer wrong")
        assert st == 401

    # 4th failure -> locked out, returns 429
    st, body = _post("bed", "Bearer wrong")
    assert st == 429
    assert "locked out" in body.lower()

    # Missing token during lockout -> 429
    st, body = _post("bed", None)
    assert st == 429
    assert "locked out" in body.lower()

    # Valid token during lockout -> exempt from lockout, returns 200
    st, body = _post("bed", "Bearer bed-token-123")
    assert st == 200

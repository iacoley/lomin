"""/voice/live over the stdlib WebSocket layer, with a mock backend and a raw-socket client."""
import base64
import json
import os
import socket
import struct
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
import live_ws  # noqa: E402
import talker_backend as tb  # noqa: E402


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class EchoBackend(tb.MockBackend):
    """Echoes each audio frame back as an audio event, then ends on cancel/close."""
    def __init__(self):
        super().__init__()
        import asyncio
        self.q = asyncio.Queue()

    async def send_audio(self, pcm):
        await super().send_audio(pcm)
        await self.q.put({"type": "audio", "pcm": pcm[::-1]})

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        await super().close()
        await self.q.put(None)


class FailBackend(tb.MockBackend):
    def __init__(self):
        super().__init__([{"type": "error", "message": "boom"}])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    monkeypatch.setenv("MVG_TALKER_BACKEND", "echo")
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "echo=test_live_ws:EchoBackend,fail=test_live_ws:FailBackend")
    sys.modules.setdefault("test_live_ws", sys.modules[__name__])


class Client:
    def __init__(self, port, path="/voice/live?node_id=n1", headers=None, upgrade=True):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        h = {"Host": "x"}
        if upgrade:
            h.update({"Upgrade": "websocket", "Connection": "Upgrade", "Sec-WebSocket-Key": key,
                      "Sec-WebSocket-Version": "13"})
        h.update(headers or {})
        req = f"GET {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
        self.s.sendall(req.encode())
        self.key = key
        self.head = self._read_head()

    def _read_head(self):
        buf = b""
        while b"\r\n\r\n" not in buf:
            c = self.s.recv(1)
            if not c:
                break
            buf += c
        return buf.decode()

    @property
    def status(self):
        return int(self.head.split()[1])

    def send(self, op, payload=b""):
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x80 | op])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            c = self.s.recv(n - len(buf))
            if not c:
                raise ConnectionError
            buf += c
        return buf

    def recv(self):
        b1, b2 = self._read(2)
        ln = b2 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", self._read(2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", self._read(8))[0]
        return b1 & 0x0F, self._read(ln)


def test_accept_key_rfc_example():
    assert live_ws.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_happy_path_audio_round_trip(server):
    c = Client(server)
    assert c.status == 101
    assert live_ws.accept_key(c.key) in c.head
    c.send(live_ws.OP_BIN, b"\x01\x02\x03\x04")
    op, payload = c.recv()
    assert (op, payload) == (live_ws.OP_BIN, b"\x04\x03\x02\x01")
    c.send(live_ws.OP_PING, b"hi")
    assert c.recv() == (live_ws.OP_PONG, b"hi")
    c.send(live_ws.OP_TEXT, json.dumps({"event": "end"}).encode())
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1000


def test_auth_failure_is_401(server, monkeypatch):
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "n1:secret")
    assert Client(server).status == 401
    ok = Client(server, headers={"Authorization": "Bearer secret"})
    assert ok.status == 101


def test_not_configured_is_501(server, monkeypatch):
    monkeypatch.delenv("MVG_TALKER_BACKEND")
    assert Client(server).status == 501


def test_unknown_backend_is_501(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKEND", "nope")
    assert Client(server).status == 501


def test_plain_get_without_upgrade_is_426(server):
    assert Client(server, upgrade=False).status == 426


def test_oversized_frame_closes_1009(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_MAX_FRAME", "1000")
    c = Client(server)
    c.send(live_ws.OP_BIN, b"\x00" * 2000)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1009


def test_idle_timeout_closes_4000(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_IDLE_SEC", "0.2")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 4000


def test_max_duration_closes_4001(server, monkeypatch):
    monkeypatch.setenv("MVG_LIVE_MAX_SEC", "0.2")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 4001


def test_backend_error_closes_1011_with_error_event(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKEND", "fail")
    c = Client(server)
    op, payload = c.recv()
    assert op == live_ws.OP_TEXT and json.loads(payload)["event"] == "error"
    op, payload = c.recv()
    fb = json.loads(payload)
    assert op == live_ws.OP_TEXT and fb["event"] == "fallback" and fb["endpoint"] == "/voice/turn"
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1011


def test_unmasked_frame_rejected():
    class S:
        def __init__(self): self.data = bytes([0x82, 0x01, 0x00])
        def recv(self, n):
            d, self.data = self.data[:n], self.data[n:]
            return d
    with pytest.raises(ConnectionError):
        live_ws.WSConn(S(), 100).recv()


class OpenFailBackend(tb.MockBackend):
    async def open(self, config):
        raise ConnectionError("no route")


def test_open_failure_sends_fallback_then_1011(server, monkeypatch):
    monkeypatch.setenv("MVG_TALKER_BACKENDS", "openfail=test_live_ws:OpenFailBackend")
    monkeypatch.setenv("MVG_TALKER_BACKEND", "openfail")
    c = Client(server)
    op, payload = c.recv()
    fb = json.loads(payload)
    assert fb["event"] == "fallback" and "open_failed" in fb["reason"]
    op, payload = c.recv()
    assert op == live_ws.OP_CLOSE and struct.unpack(">H", payload[:2])[0] == 1011

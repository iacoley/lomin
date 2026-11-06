"""GeminiLiveBackend against a fake Live server (a real local WebSocket, no network)."""
import asyncio
import base64
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gemini_live as gl  # noqa: E402
import talker_backend  # noqa: E402


async def _ws_read(reader):
    b1, b2 = await reader.readexactly(2)
    n = b2 & 0x7F
    if n == 126:
        n = struct.unpack(">H", await reader.readexactly(2))[0]
    mask = await reader.readexactly(4)
    data = await reader.readexactly(n)
    return b1 & 0xF, bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def _frame(obj):
    p = json.dumps(obj).encode()
    hdr = bytes([0x81]) + (bytes([len(p)]) if len(p) < 126 else bytes([126]) + struct.pack(">H", len(p)))
    return hdr + p


def run_with_server(script, client):
    """script(received, write) -> awaitable run after setup; client(backend) -> awaitable."""
    received = []

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n")
        op, data = await _ws_read(reader)
        received.append(json.loads(data))
        writer.write(_frame({"setupComplete": {}}))
        await writer.drain()
        await script(reader, writer, received)
        writer.close()

    async def main():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        import os
        os.environ["MVG_GEMINI_URL"] = f"ws://127.0.0.1:{port}/live"
        os.environ.pop("MVG_GEMINI_API_KEY", None)
        b = gl.GeminiLiveBackend()
        await b.open({})
        try:
            return await asyncio.wait_for(client(b, received), 5)
        finally:
            await b.close()
            server.close()
    return asyncio.run(main())


def test_registered_contract():
    talker_backend.requires(gl.GeminiLiveBackend(), "audio_in", "audio_out", "tool_calls", "silent_context")


def test_setup_and_audio_and_events():
    msgs = [
        {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000",
                                                                    "data": base64.b64encode(b"\x01\x02").decode()}}]}}},
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "call_thinker", "args": {"request": "x"}}]}},
        {"serverContent": {"interrupted": True}},
        {"serverContent": {"turnComplete": True}},
    ]

    async def script(reader, writer, received):
        op, data = await _ws_read(reader)
        received.append(json.loads(data))
        for m in msgs:
            writer.write(_frame(m))
        await writer.drain()
        op, data = await _ws_read(reader)
        received.append(json.loads(data))
        await asyncio.sleep(0.2)

    async def client(b, received):
        await b.send_audio(b"\x00\x01")
        evs = []
        async for ev in b.events():
            evs.append(ev)
            if ev["type"] == "tool_call":
                await b.send_tool_result("c1", {"status": "started"})
            if ev["type"] == "turn_end":
                break
        await asyncio.sleep(0.1)
        return evs

    # received is shared; fetch after
    holder = {}

    async def wrapped(b, received):
        holder["r"] = received
        return await client(b, received)
    evs = run_with_server(script, wrapped)
    assert [e["type"] for e in evs] == ["audio", "tool_call", "interrupted", "turn_end"]
    assert evs[0]["pcm"] == b"\x01\x02"
    r = holder["r"]
    setup = r[0]["setup"]
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert {t["name"] for t in setup["tools"][0]["functionDeclarations"]} == set(
        ["call_thinker", "cancel_thinker", "answer_prompt", "set_progress_updates"])
    assert r[1]["realtimeInput"]["audio"]["mimeType"] == "audio/pcm;rate=16000"
    assert r[2]["toolResponse"]["functionResponses"][0] == {"id": "c1", "name": "call_thinker", "response": {"status": "started"}}


def test_silent_text_does_not_complete_turn():
    async def script(reader, writer, received):
        op, data = await _ws_read(reader)
        received.append(json.loads(data))
        op, data = await _ws_read(reader)
        received.append(json.loads(data))

    async def client(b, received):
        await b.send_text("ctx", silent=True)
        await b.send_text("hi")
        await asyncio.sleep(0.2)
        return received
    r = run_with_server(script, client)
    assert r[1]["clientContent"]["turnComplete"] is False
    assert r[2]["clientContent"]["turnComplete"] is True


def test_connection_drop_is_error_event():
    async def script(reader, writer, received):
        return None

    async def client(b, received):
        return [e async for e in b.events()]
    evs = run_with_server(script, client)
    assert evs and evs[-1]["type"] == "error"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MVG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MVG_GEMINI_URL", raising=False)
    import pytest
    with pytest.raises(ValueError):
        asyncio.run(gl.GeminiLiveBackend().open({}))

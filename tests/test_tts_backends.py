"""tts_backends.py: the openai-compatible, generic HTTP and ElevenLabs TTS
backends (ERED LOMIN P1b). The OpenAI and HTTP backends talk to a local
http.server; ElevenLabs is checked by patching urllib.request.urlopen, so the
real API is never called."""

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tts_backends  # noqa: E402


def _wav(frames: bytes = b"\x01\x00" * 480) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(frames)
    return buf.getvalue()


WAV = _wav()


@pytest.fixture
def tts_server():
    """A local server that records each POST and answers with state["body"]
    at state["status"]."""
    seen = []
    state = {"status": 200, "body": WAV}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append({"path": self.path, "headers": self.headers, "body": self.rfile.read(n)})
            self.send_response(state["status"])
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(state["body"])))
            self.end_headers()
            self.wfile.write(state["body"])

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(
        url=f"http://127.0.0.1:{server.server_address[1]}/speak", seen=seen, state=state)
    server.shutdown()
    server.server_close()


# --- openai_speech_speak -----------------------------------------------------

def test_openai_posts_speech_request_and_returns_wav(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_OPENAI_TTS_URL", tts_server.url)
    for k in ("MVG_OPENAI_TTS_KEY", "MVG_OPENAI_TTS_MODEL", "MVG_OPENAI_TTS_VOICE"):
        monkeypatch.delenv(k, raising=False)
    out = tts_backends.openai_speech_speak("hello there")
    assert out == WAV
    req = tts_server.seen[-1]
    assert json.loads(req["body"]) == {
        "model": "kokoro", "voice": "af_sky",
        "input": "hello there", "response_format": "wav",
    }
    assert "Authorization" not in req["headers"]


def test_openai_model_voice_and_bearer_from_env(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_OPENAI_TTS_URL", tts_server.url)
    monkeypatch.setenv("MVG_OPENAI_TTS_MODEL", "tts-1")
    monkeypatch.setenv("MVG_OPENAI_TTS_VOICE", "bf_emma")
    monkeypatch.setenv("MVG_OPENAI_TTS_KEY", "sekrit")
    tts_backends.openai_speech_speak("x")
    req = tts_server.seen[-1]
    body = json.loads(req["body"])
    assert body["model"] == "tts-1" and body["voice"] == "bf_emma"
    assert req["headers"]["Authorization"] == "Bearer sekrit"


def test_openai_non_wav_body_raises(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_OPENAI_TTS_URL", tts_server.url)
    tts_server.state["body"] = b"<html>not audio</html>"
    with pytest.raises(RuntimeError):
        tts_backends.openai_speech_speak("x")


def test_openai_non_2xx_raises(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_OPENAI_TTS_URL", tts_server.url)
    tts_server.state["status"] = 500
    with pytest.raises(Exception):  # urllib raises HTTPError for 5xx
        tts_backends.openai_speech_speak("x")


def test_openai_missing_url_raises(monkeypatch):
    monkeypatch.delenv("MVG_OPENAI_TTS_URL", raising=False)
    with pytest.raises(RuntimeError):
        tts_backends.openai_speech_speak("x")


# --- http_tts_speak ----------------------------------------------------------

def test_http_hook_posts_text_and_returns_wav(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_HTTP_TTS_URL", tts_server.url)
    assert tts_backends.http_tts_speak("hi there") == WAV
    req = tts_server.seen[-1]
    assert json.loads(req["body"]) == {"text": "hi there"}
    assert req["headers"]["Content-Type"] == "application/json"


def test_http_hook_non_wav_body_raises(tts_server, monkeypatch):
    monkeypatch.setenv("MVG_HTTP_TTS_URL", tts_server.url)
    tts_server.state["body"] = b"\x00\x01\x02 not a riff"
    with pytest.raises(RuntimeError):
        tts_backends.http_tts_speak("x")


def test_http_hook_missing_url_raises(monkeypatch):
    monkeypatch.delenv("MVG_HTTP_TTS_URL", raising=False)
    with pytest.raises(RuntimeError):
        tts_backends.http_tts_speak("x")


# --- elevenlabs_speak (urlopen patched; the real API is never called) ---------

class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self._pos = 0
        self.status = status

    def read(self, n: int = -1):
        if n is None or n < 0:
            rem = self._body[self._pos:]
            self._pos = len(self._body)
            return rem
        chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_urlopen(monkeypatch):
    calls = []
    state = {"body": WAV, "status": 200}

    def fake(req, timeout=None):
        calls.append({"req": req, "timeout": timeout})
        return _FakeResponse(state["body"], state["status"])

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return SimpleNamespace(calls=calls, state=state)


def test_elevenlabs_request_shape(fake_urlopen, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-key-1")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "VOICE123")
    monkeypatch.delenv("MVG_ELEVENLABS_MODEL", raising=False)
    assert tts_backends.elevenlabs_speak("hi") == WAV
    call = fake_urlopen.calls[-1]
    req = call["req"]
    assert req.get_method() == "POST"
    assert req.full_url == (
        "https://api.elevenlabs.io/v1/text-to-speech/VOICE123?output_format=wav_24000")
    assert req.get_header("Xi-api-key") == "xi-key-1"
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Accept") == "audio/wav"
    assert json.loads(req.data) == {"text": "hi", "model_id": "eleven_multilingual_v2"}
    assert call["timeout"] == 10


def test_elevenlabs_model_from_env(fake_urlopen, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setenv("MVG_ELEVENLABS_MODEL", "eleven_turbo_v2")
    tts_backends.elevenlabs_speak("hi")
    assert json.loads(fake_urlopen.calls[-1]["req"].data)["model_id"] == "eleven_turbo_v2"


def test_elevenlabs_non_2xx_raises(fake_urlopen, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "v")
    fake_urlopen.state["status"] = 401
    with pytest.raises(RuntimeError):
        tts_backends.elevenlabs_speak("hi")


def test_elevenlabs_non_wav_body_raises(fake_urlopen, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "v")
    fake_urlopen.state["body"] = b'{"detail": "quota"}'
    with pytest.raises(RuntimeError):
        tts_backends.elevenlabs_speak("hi")


@pytest.mark.parametrize("missing", ["ELEVENLABS_API_KEY", "MVG_ELEVENLABS_VOICE_ID"])
def test_elevenlabs_missing_config_raises_without_calling_api(fake_urlopen, monkeypatch, missing):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "v")
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError):
        tts_backends.elevenlabs_speak("hi")
    assert fake_urlopen.calls == []

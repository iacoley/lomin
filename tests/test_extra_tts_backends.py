"""Unit tests for GeminiTTSBackend, KokoroTTSBackend, and get_backend factory resolution.

Covers:
- GeminiTTSBackend header auth (x-goog-api-key, never in URL query string)
- GeminiTTSBackend payload generation and base64 PCM / WAV container decoding
- GeminiTTSBackend synthesize and synthesize_stream chunking
- KokoroTTSBackend request payload format (model, input, voice, response_format=pcm)
- KokoroTTSBackend streaming and chunked PCM yield
- Factory resolution via get_backend() with env vars, explicit names, and registration
"""

import base64
import json
from io import BytesIO
import urllib.request
import wave

import pytest

import tts_backends
from tts_backends import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_VOICE,
    DEFAULT_KOKORO_URL,
    DEFAULT_KOKORO_VOICE,
    GeminiTTSBackend,
    KokoroTTSBackend,
    extract_pcm_from_wav_or_raw,
    gemini_speak,
    get_backend,
    kokoro_speak,
    pcm_to_wav,
    register_backend,
)


def _make_dummy_pcm(num_samples: int = 2400) -> bytes:
    """Generate deterministic 16-bit mono PCM bytes."""
    return b"\x01\x00" * num_samples


def _make_wav_from_pcm(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Create a standard WAV container around PCM data."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_data)
    return buf.getvalue()


class _MockHTTPResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict = None):
        self._body = body
        self._pos = 0
        self.status = status
        self.headers = headers or {}

    def read(self, amt: int = -1) -> bytes:
        if amt is None or amt < 0:
            rem = self._body[self._pos:]
            self._pos = len(self._body)
            return rem
        chunk = self._body[self._pos : self._pos + amt]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# GeminiTTSBackend Tests
# ---------------------------------------------------------------------------


def test_gemini_header_auth_and_no_key_in_url(monkeypatch):
    """API key MUST be passed strictly in x-goog-api-key header and never in URL query."""
    captured = {}
    test_pcm = _make_dummy_pcm(1200)
    b64_audio = base64.b64encode(test_pcm).decode("ascii")

    resp_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=24000",
                                "data": b64_audio,
                            }
                        }
                    ]
                }
            }
        ]
    }
    resp_bytes = json.dumps(resp_json).encode("utf-8")

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _MockHTTPResponse(resp_bytes, status=200)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    monkeypatch.setenv("MVG_GEMINI_API_KEY", "secret-test-key-12345")
    monkeypatch.delenv("MVG_GEMINI_VOICE", raising=False)
    monkeypatch.delenv("MVG_GEMINI_TTS_MODEL", raising=False)

    backend = GeminiTTSBackend()
    result = backend.synthesize("Hello Gemini")

    assert result == test_pcm
    req = captured["req"]
    assert req.get_method() == "POST"

    # Verify x-goog-api-key header is set
    assert req.get_header("X-goog-api-key") == "secret-test-key-12345"

    # Verify API key is NOT leaked in the URL query string
    url = req.full_url
    assert "secret-test-key-12345" not in url
    assert "?" not in url
    assert url == f"https://generativelanguage.googleapis.com/v1beta/models/{DEFAULT_GEMINI_MODEL}:generateContent"

    # Verify payload format
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["contents"] == [{"parts": [{"text": "Hello Gemini"}]}]
    assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert (
        payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
        == DEFAULT_GEMINI_VOICE
    )


def test_gemini_custom_voice_and_model(monkeypatch):
    """Custom voice, model, and custom endpoint URL are respected."""
    captured = {}
    test_pcm = _make_dummy_pcm(400)
    b64_audio = base64.b64encode(test_pcm).decode("ascii")

    resp_json = {
        "candidates": [
            {
                "content": {
                    "parts": [{"inlineData": {"data": b64_audio}}]
                }
            }
        ]
    }

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        return _MockHTTPResponse(json.dumps(resp_json).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    monkeypatch.setenv("MVG_GEMINI_API_KEY", "key-custom")
    monkeypatch.setenv("MVG_GEMINI_VOICE", "Puck")
    monkeypatch.setenv("MVG_GEMINI_TTS_MODEL", "models/custom-gemini-tts")

    backend = GeminiTTSBackend()
    res = backend.synthesize("Testing voice override", voice="Fenrir")
    assert res == test_pcm

    req = captured["req"]
    assert req.full_url == "https://generativelanguage.googleapis.com/v1beta/models/custom-gemini-tts:generateContent"
    payload = json.loads(req.data.decode("utf-8"))
    assert (
        payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
        == "Fenrir"
    )


def test_gemini_decodes_wav_container_to_raw_pcm(monkeypatch):
    """If Gemini returns base64 WAV with RIFF header, it is cleanly unwrapped to raw PCM."""
    test_pcm = _make_dummy_pcm(1600)
    wav_bytes = _make_wav_from_pcm(test_pcm, sample_rate=24000)
    b64_wav = base64.b64encode(wav_bytes).decode("ascii")

    resp_json = {
        "candidates": [
            {
                "content": {
                    "parts": [{"inlineData": {"data": b64_wav}}]
                }
            }
        ]
    }

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(json.dumps(resp_json).encode("utf-8")),
    )
    backend = GeminiTTSBackend(api_key="mock-key")
    out = backend.synthesize("WAV wrapped audio")
    assert out == test_pcm
    assert not out.startswith(b"RIFF")


def test_gemini_missing_api_key_raises(monkeypatch):
    """Missing API key raises RuntimeError."""
    monkeypatch.delenv("MVG_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    backend = GeminiTTSBackend()
    with pytest.raises(RuntimeError, match="MVG_GEMINI_API_KEY not set"):
        backend.synthesize("No key")


def test_gemini_missing_audio_parts_raises(monkeypatch):
    """Empty parts or missing audio data in Gemini response raises RuntimeError."""
    resp_json = {"candidates": [{"content": {"parts": []}}]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(json.dumps(resp_json).encode("utf-8")),
    )
    backend = GeminiTTSBackend(api_key="mock-key")
    with pytest.raises(RuntimeError, match="gemini tts returned no audio inlineData"):
        backend.synthesize("Empty parts")


def test_gemini_synthesize_stream(monkeypatch):
    """synthesize_stream yields correctly sized PCM chunks."""
    # 9600 bytes of PCM (2 full 4800 byte chunks)
    test_pcm = b"\x02\x00" * 4800
    b64_audio = base64.b64encode(test_pcm).decode("ascii")
    resp_json = {
        "candidates": [
            {
                "content": {
                    "parts": [{"inlineData": {"data": b64_audio}}]
                }
            }
        ]
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(json.dumps(resp_json).encode("utf-8")),
    )

    backend = GeminiTTSBackend(api_key="mock-key")
    chunks = list(backend.synthesize_stream("Stream test", chunk_size=4800))
    assert len(chunks) == 2
    assert chunks[0] == test_pcm[:4800]
    assert chunks[1] == test_pcm[4800:]
    assert b"".join(chunks) == test_pcm


# ---------------------------------------------------------------------------
# KokoroTTSBackend Tests
# ---------------------------------------------------------------------------


def test_kokoro_request_payload_and_defaults(monkeypatch):
    """KokoroTTSBackend posts {"model": "kokoro", "input": text, "voice": voice, "response_format": "pcm"}."""
    captured = {}
    test_pcm = _make_dummy_pcm(2400)

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _MockHTTPResponse(test_pcm, status=200)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    monkeypatch.delenv("MVG_KOKORO_URL", raising=False)
    monkeypatch.delenv("MVG_KOKORO_VOICE", raising=False)
    monkeypatch.delenv("MVG_KOKORO_MODEL", raising=False)
    monkeypatch.delenv("MVG_KOKORO_API_KEY", raising=False)

    backend = KokoroTTSBackend()
    res = backend.synthesize("Hello Kokoro")
    assert res == test_pcm

    req = captured["req"]
    assert req.full_url == DEFAULT_KOKORO_URL
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    assert "Authorization" not in req.headers

    payload = json.loads(req.data.decode("utf-8"))
    assert payload == {
        "model": "kokoro",
        "input": "Hello Kokoro",
        "voice": DEFAULT_KOKORO_VOICE,
        "response_format": "pcm",
    }


def test_kokoro_custom_voice_model_auth(monkeypatch):
    """Custom voice, model, and Bearer auth token are handled properly."""
    captured = {}
    test_pcm = _make_dummy_pcm(600)

    def mock_urlopen(req, timeout=None):
        captured["req"] = req
        return _MockHTTPResponse(test_pcm, status=200)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    monkeypatch.setenv("MVG_KOKORO_URL", "http://127.0.0.1:9999/v1/audio/speech")
    monkeypatch.setenv("MVG_KOKORO_VOICE", "af_bella")
    monkeypatch.setenv("MVG_KOKORO_MODEL", "kokoro-v1")
    monkeypatch.setenv("MVG_KOKORO_API_KEY", "bearer-token-xyz")

    backend = KokoroTTSBackend()
    res = backend.synthesize("Test with bearer", voice="af_sarah")
    assert res == test_pcm

    req = captured["req"]
    assert req.full_url == "http://127.0.0.1:9999/v1/audio/speech"
    assert req.get_header("Authorization") == "Bearer bearer-token-xyz"

    payload = json.loads(req.data.decode("utf-8"))
    assert payload["model"] == "kokoro-v1"
    assert payload["voice"] == "af_sarah"
    assert payload["input"] == "Test with bearer"
    assert payload["response_format"] == "pcm"


def test_kokoro_streaming_yields_pcm_chunks(monkeypatch):
    """synthesize_stream reads and yields PCM chunks from the HTTP response."""
    test_pcm = b"\x03\x00" * 4800  # 9600 bytes
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(test_pcm, status=200),
    )

    backend = KokoroTTSBackend(url="http://127.0.0.1:8880/v1/audio/speech")
    chunks = list(backend.synthesize_stream("Stream Kokoro", chunk_size=2400))
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 2400
    assert b"".join(chunks) == test_pcm


def test_kokoro_non_2xx_raises(monkeypatch):
    """Non-2xx HTTP status raises an exception."""
    def mock_urlopen(req, timeout=None):
        return _MockHTTPResponse(b"Internal Server Error", status=500)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)
    backend = KokoroTTSBackend()
    with pytest.raises(RuntimeError, match="kokoro tts HTTP 500"):
        backend.synthesize("Fail")


def test_kokoro_unwraps_wav_if_server_returns_wav(monkeypatch):
    """If a server returns WAV container despite response_format=pcm, unwraps to PCM."""
    test_pcm = _make_dummy_pcm(1200)
    wav_bytes = _make_wav_from_pcm(test_pcm, sample_rate=24000)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(wav_bytes, status=200),
    )

    backend = KokoroTTSBackend()
    res = backend.synthesize("WAV wrap")
    assert res == test_pcm


# ---------------------------------------------------------------------------
# get_backend Factory Resolution Tests
# ---------------------------------------------------------------------------


def test_get_backend_by_name():
    """get_backend resolves instances of GeminiTTSBackend and KokoroTTSBackend."""
    b_gemini = get_backend("gemini")
    assert isinstance(b_gemini, GeminiTTSBackend)
    assert b_gemini.sample_rate == 24000

    b_kokoro = get_backend("kokoro")
    assert isinstance(b_kokoro, KokoroTTSBackend)
    assert b_kokoro.sample_rate == 24000

    # Case-insensitive resolution
    assert isinstance(get_backend("GEMINI"), GeminiTTSBackend)
    assert isinstance(get_backend("  Kokoro  "), KokoroTTSBackend)


def test_get_backend_instantiate_false():
    """get_backend with instantiate=False returns the backend class."""
    cls_gemini = get_backend("gemini", instantiate=False)
    assert cls_gemini is GeminiTTSBackend

    cls_kokoro = get_backend("kokoro", instantiate=False)
    assert cls_kokoro is KokoroTTSBackend


def test_get_backend_kwargs_forwarding():
    """Kwargs are passed through to the backend constructor."""
    b_gemini = get_backend("gemini", voice="Puck", api_key="test-key")
    assert b_gemini.voice == "Puck"
    assert b_gemini.api_key == "test-key"

    b_kokoro = get_backend("kokoro", voice="af_nicole", url="http://127.0.0.1:9090/speak")
    assert b_kokoro.voice == "af_nicole"
    assert b_kokoro.url == "http://127.0.0.1:9090/speak"


def test_get_backend_env_resolution(monkeypatch):
    """get_backend() without args resolves from MVG_TTS env var, defaulting to kokoro."""
    monkeypatch.setenv("MVG_TTS", "gemini")
    b = get_backend()
    assert isinstance(b, GeminiTTSBackend)

    monkeypatch.setenv("MVG_TTS", "kokoro")
    b = get_backend()
    assert isinstance(b, KokoroTTSBackend)

    monkeypatch.delenv("MVG_TTS", raising=False)
    b = get_backend()
    assert b is None


def test_get_backend_unknown_raises():
    """Unknown backend name raises ValueError with available list."""
    with pytest.raises(ValueError, match="Unknown TTS backend: 'nonexistent'"):
        get_backend("nonexistent")


def test_register_backend_custom():
    """register_backend dynamically registers new backends."""
    class CustomBackend:
        def __init__(self, greeting="hello"):
            self.greeting = greeting

    register_backend("custom_dummy", CustomBackend)
    try:
        inst = get_backend("custom_dummy", greeting="hi there")
        assert isinstance(inst, CustomBackend)
        assert inst.greeting == "hi there"
    finally:
        tts_backends.BACKENDS.pop("custom_dummy", None)


# ---------------------------------------------------------------------------
# Utility & Convenience Helper Tests
# ---------------------------------------------------------------------------


def test_extract_pcm_from_wav_or_raw():
    """extract_pcm_from_wav_or_raw extracts audio frames from WAV and leaves raw PCM unchanged."""
    raw_pcm = b"\x10\x00\x20\x00\x30\x00"
    assert extract_pcm_from_wav_or_raw(raw_pcm) == raw_pcm

    wav_bytes = _make_wav_from_pcm(raw_pcm, sample_rate=24000)
    assert wav_bytes.startswith(b"RIFF")
    extracted = extract_pcm_from_wav_or_raw(wav_bytes)
    assert extracted == raw_pcm


def test_pcm_to_wav():
    """pcm_to_wav wraps raw PCM in valid RIFF header."""
    raw_pcm = _make_dummy_pcm(480)
    wav_bytes = pcm_to_wav(raw_pcm, sample_rate=24000)
    assert wav_bytes.startswith(b"RIFF")
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
        assert w.readframes(w.getnframes()) == raw_pcm


def test_gemini_and_kokoro_speak_helpers(monkeypatch):
    """gemini_speak and kokoro_speak return complete WAV bytes."""
    test_pcm = _make_dummy_pcm(480)
    b64_audio = base64.b64encode(test_pcm).decode("ascii")
    resp_json = {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"data": b64_audio}}]}}
        ]
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(
            json.dumps(resp_json).encode("utf-8") if "generateContent" in req.full_url else test_pcm
        ),
    )
    monkeypatch.setenv("MVG_GEMINI_API_KEY", "mock-key")

    wav_gemini = gemini_speak("hello gemini speak")
    assert wav_gemini.startswith(b"RIFF")
    assert extract_pcm_from_wav_or_raw(wav_gemini) == test_pcm

    wav_kokoro = kokoro_speak("hello kokoro speak")
    assert wav_kokoro.startswith(b"RIFF")
    assert extract_pcm_from_wav_or_raw(wav_kokoro) == test_pcm


def test_callable_interface(monkeypatch):
    """backend(text) behaves identically to backend.synthesize(text)."""
    test_pcm = _make_dummy_pcm(300)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(test_pcm),
    )
    backend = KokoroTTSBackend()
    assert backend("Callable test") == test_pcm

"""Swappable TTS backends for the voice gateway (ERED LOMIN P1b).

Each backend is `(text: str) -> bytes`: a complete WAV, starting with RIFF.
It raises on any failure: a missing config, a transport error, a non-2xx
status, or a body that is not a WAV. gateway.py builds its chain from these
(MVG_TTS_CHAIN); see the README "Backends" section.

Stdlib only, and no import of gateway: config is read from the environment
on every call, so a backend can be swapped or reconfigured without a reload.
"""

import base64
import json
import os
import struct
import urllib.parse
import urllib.request
import wave
from io import BytesIO
from typing import Any, Dict, Iterator, Optional

ELEVENLABS_TIMEOUT = 10
OPENAI_TTS_TIMEOUT = 15
HTTP_TTS_TIMEOUT = 15
DEFAULT_GEMINI_VOICE = "Schedar"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_URL = "http://127.0.0.1:8880/v1/audio/speech"
MAX_TTS_RESPONSE_BYTES = int(os.environ.get("MVG_MAX_TTS_RESPONSE_BYTES") or 25 * 1024 * 1024)


def _post_wav(url: str, data: bytes, headers: dict, timeout: float) -> bytes:
    """POST `data`, return the body if it is a 2xx WAV. Raises otherwise."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        body = resp.read(MAX_TTS_RESPONSE_BYTES + 1)
    if not 200 <= status < 300:
        raise RuntimeError(f"tts HTTP {status}")
    if len(body) > MAX_TTS_RESPONSE_BYTES:
        raise RuntimeError(f"tts response exceeded maximum size ({MAX_TTS_RESPONSE_BYTES} bytes)")
    if not body or not body.startswith(b"RIFF"):
        raise RuntimeError(f"tts returned a non-WAV body ({len(body or b'')} bytes)")
    return body


def elevenlabs_speak(text: str) -> bytes:
    """ElevenLabs text-to-speech, WAV output at 24 kHz.

    Env: ELEVENLABS_API_KEY (required), MVG_ELEVENLABS_VOICE_ID (required),
    MVG_ELEVENLABS_MODEL (default eleven_multilingual_v2)."""
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice = os.environ.get("MVG_ELEVENLABS_VOICE_ID", "")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    if not voice:
        raise RuntimeError("MVG_ELEVENLABS_VOICE_ID not set")
    model = os.environ.get("MVG_ELEVENLABS_MODEL") or "eleven_multilingual_v2"
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           + urllib.parse.quote(voice, safe="")
           + "?output_format=wav_24000")
    body = json.dumps({"text": text, "model_id": model}).encode("utf-8")
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    return _post_wav(url, body, headers, ELEVENLABS_TIMEOUT)


def openai_speech_speak(text: str) -> bytes:
    """Any OpenAI-compatible /v1/audio/speech endpoint. Kokoro-FastAPI serves
    one at http://localhost:8880/v1/audio/speech.

    Env: MVG_OPENAI_TTS_URL (required), MVG_OPENAI_TTS_MODEL (default kokoro),
    MVG_OPENAI_TTS_VOICE (default af_sky), MVG_OPENAI_TTS_KEY (optional; sent
    as a bearer token only when set)."""
    url = os.environ.get("MVG_OPENAI_TTS_URL", "")
    if not url:
        raise RuntimeError("MVG_OPENAI_TTS_URL not set")
    payload = {
        "model": os.environ.get("MVG_OPENAI_TTS_MODEL") or "kokoro",
        "voice": os.environ.get("MVG_OPENAI_TTS_VOICE") or "af_sky",
        "input": text,
        "response_format": "wav",
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("MVG_OPENAI_TTS_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload).encode("utf-8")
    return _post_wav(url, body, headers, OPENAI_TTS_TIMEOUT)


def http_tts_speak(text: str) -> bytes:
    """Generic hook: POST {"text": text} as JSON to MVG_HTTP_TTS_URL; the
    response body is the WAV."""
    url = os.environ.get("MVG_HTTP_TTS_URL", "")
    if not url:
        raise RuntimeError("MVG_HTTP_TTS_URL not set")
    body = json.dumps({"text": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    return _post_wav(url, body, headers, HTTP_TTS_TIMEOUT)


def extract_pcm_from_wav_or_raw(raw_bytes: bytes) -> bytes:
    """Extract raw PCM audio from a WAV container if RIFF header is present,
    otherwise return raw_bytes unchanged."""
    if len(raw_bytes) < 12 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WAVE":
        return raw_bytes
    offset = 12
    total_len = len(raw_bytes)
    while offset + 8 <= total_len:
        chunk_id = raw_bytes[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw_bytes, offset + 4)[0]
        offset += 8
        if chunk_id == b"data":
            if 0 < chunk_size < 0xFFFFFFFF and offset + chunk_size <= total_len:
                return raw_bytes[offset : offset + chunk_size]
            return raw_bytes[offset:]
        skip = chunk_size + (chunk_size % 2)
        offset += skip
    return raw_bytes[44:] if total_len > 44 else raw_bytes


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw 16-bit PCM bytes in a standard RIFF/WAVE container."""
    if pcm_bytes.startswith(b"RIFF"):
        return pcm_bytes
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


class GeminiTTSBackend:
    """Gemini API TTS backend (`MVG_TTS=gemini`, 24 kHz 16-bit mono PCM).

    Env: MVG_GEMINI_API_KEY (or GEMINI_API_KEY, required),
    MVG_GEMINI_VOICE (default Schedar),
    MVG_GEMINI_TTS_MODEL (default gemini-2.5-flash-preview-tts),
    MVG_GEMINI_TTS_URL (optional custom endpoint).
    Passes API key strictly via x-goog-api-key header (never in URL query string).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        sample_rate: int = 24000,
        url: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self.sample_rate = sample_rate
        self._url = url
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        return (
            self._api_key
            or os.environ.get("MVG_GEMINI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )

    @api_key.setter
    def api_key(self, val: Optional[str]) -> None:
        self._api_key = val

    @property
    def voice(self) -> str:
        return (
            self._voice
            or os.environ.get("MVG_GEMINI_VOICE")
            or DEFAULT_GEMINI_VOICE
        )

    @voice.setter
    def voice(self, val: Optional[str]) -> None:
        self._voice = val

    @property
    def model(self) -> str:
        return (
            self._model
            or os.environ.get("MVG_GEMINI_TTS_MODEL")
            or DEFAULT_GEMINI_MODEL
        )

    @model.setter
    def model(self, val: Optional[str]) -> None:
        self._model = val

    @property
    def url(self) -> str:
        if self._url:
            return self._url
        env_url = os.environ.get("MVG_GEMINI_TTS_URL", "").strip()
        if env_url:
            return env_url
        m = self.model
        model_path = m if m.startswith("models/") else f"models/{m}"
        return f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"

    @url.setter
    def url(self, val: Optional[str]) -> None:
        self._url = val

    def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text into raw 24 kHz 16-bit mono PCM bytes."""
        api_key = self.api_key
        if not api_key:
            raise RuntimeError("MVG_GEMINI_API_KEY not set")

        target_voice = voice or self.voice
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": target_voice,
                        }
                    }
                },
            },
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = getattr(resp, "status", 200)
            resp_bytes = resp.read(MAX_TTS_RESPONSE_BYTES + 1)

        if not 200 <= status < 300:
            raise RuntimeError(f"gemini tts HTTP {status}")
        if len(resp_bytes) > MAX_TTS_RESPONSE_BYTES:
            raise RuntimeError(
                f"gemini tts response exceeded maximum size ({MAX_TTS_RESPONSE_BYTES} bytes)"
            )

        data = json.loads(resp_bytes.decode("utf-8"))
        pcm_chunks = []
        candidates = data.get("candidates") or []
        for cand in candidates:
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                inline = part.get("inlineData") or part.get("inline_data") or {}
                b64_audio = inline.get("data")
                if b64_audio:
                    raw = base64.b64decode(b64_audio)
                    pcm_chunks.append(extract_pcm_from_wav_or_raw(raw))
        if pcm_chunks:
            return b"".join(pcm_chunks)
        raise RuntimeError("gemini tts returned no audio inlineData")

    def synthesize_stream(
        self, text: str, voice: Optional[str] = None, chunk_size: int = 4800
    ) -> Iterator[bytes]:
        """Synthesize text and yield PCM16 chunks."""
        pcm = self.synthesize(text, voice=voice)
        if not chunk_size or chunk_size <= 0:
            yield pcm
            return
        for i in range(0, len(pcm), chunk_size):
            yield pcm[i : i + chunk_size]

    def __call__(self, text: str) -> bytes:
        return self.synthesize(text)


class KokoroTTSBackend:
    """Kokoro TTS backend (`MVG_TTS=kokoro`, 24 kHz 16-bit mono PCM).

    Env: MVG_KOKORO_URL (default http://127.0.0.1:8880/v1/audio/speech),
    MVG_KOKORO_VOICE (default af_heart),
    MVG_KOKORO_MODEL (default kokoro),
    MVG_KOKORO_API_KEY or MVG_KOKORO_KEY (optional bearer token).
    Posts {"model": "kokoro", "input": text, "voice": voice, "response_format": "pcm"}.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        sample_rate: int = 24000,
        api_key: Optional[str] = None,
        timeout: float = 15.0,
    ):
        self._url = url
        self._voice = voice
        self._model = model
        self.sample_rate = sample_rate
        self._api_key = api_key
        self.timeout = timeout

    @property
    def url(self) -> str:
        return (
            self._url
            or os.environ.get("MVG_KOKORO_URL")
            or DEFAULT_KOKORO_URL
        )

    @url.setter
    def url(self, val: Optional[str]) -> None:
        self._url = val

    @property
    def voice(self) -> str:
        return (
            self._voice
            or os.environ.get("MVG_KOKORO_VOICE")
            or DEFAULT_KOKORO_VOICE
        )

    @voice.setter
    def voice(self, val: Optional[str]) -> None:
        self._voice = val

    @property
    def model(self) -> str:
        return (
            self._model
            or os.environ.get("MVG_KOKORO_MODEL")
            or "kokoro"
        )

    @model.setter
    def model(self, val: Optional[str]) -> None:
        self._model = val

    @property
    def api_key(self) -> str:
        return (
            self._api_key
            or os.environ.get("MVG_KOKORO_API_KEY")
            or os.environ.get("MVG_KOKORO_KEY")
            or ""
        )

    @api_key.setter
    def api_key(self, val: Optional[str]) -> None:
        self._api_key = val

    def synthesize_stream(
        self, text: str, voice: Optional[str] = None, chunk_size: int = 4800
    ) -> Iterator[bytes]:
        """Post request and yield PCM16 chunks."""
        url = self.url
        if not url:
            raise RuntimeError("MVG_KOKORO_URL not set")

        payload = {
            "model": self.model,
            "input": text,
            "voice": voice or self.voice,
            "response_format": "pcm",
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = getattr(resp, "status", 200)
            if not 200 <= status < 300:
                raise RuntimeError(f"kokoro tts HTTP {status}")
            total_read = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > MAX_TTS_RESPONSE_BYTES:
                    raise RuntimeError(
                        f"kokoro tts response exceeded maximum size ({MAX_TTS_RESPONSE_BYTES} bytes)"
                    )
                yield chunk

    def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Synthesize text into raw 24 kHz 16-bit mono PCM bytes."""
        url = self.url
        if not url:
            raise RuntimeError("MVG_KOKORO_URL not set")

        payload = {
            "model": self.model,
            "input": text,
            "voice": voice or self.voice,
            "response_format": "pcm",
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            status = getattr(resp, "status", 200)
            if not 200 <= status < 300:
                raise RuntimeError(f"kokoro tts HTTP {status}")
            resp_bytes = resp.read(MAX_TTS_RESPONSE_BYTES + 1)

        if len(resp_bytes) > MAX_TTS_RESPONSE_BYTES:
            raise RuntimeError(
                f"kokoro tts response exceeded maximum size ({MAX_TTS_RESPONSE_BYTES} bytes)"
            )
        return extract_pcm_from_wav_or_raw(resp_bytes)

    def __call__(self, text: str) -> bytes:
        return self.synthesize(text)


BACKENDS: Dict[str, Any] = {
    "gemini": GeminiTTSBackend,
    "kokoro": KokoroTTSBackend,
}


def register_backend(name: str, backend_cls_or_factory: Any) -> None:
    """Register a TTS backend class or factory under name."""
    BACKENDS[name.strip().lower()] = backend_cls_or_factory


def get_backend(name: Optional[str] = None, *, instantiate: bool = True, **kwargs) -> Any:
    """Factory resolution for TTS backends.

    Resolves backend by name (defaulting to MVG_TTS environment variable).
    Returns an instance of the backend class (or class if instantiate=False),
    or None if name / MVG_TTS is unset or empty.
    """
    if name is None:
        name = os.environ.get("MVG_TTS", "").strip()
    if not name or name.lower() in ("none", "off", "0", "false"):
        return None
    name = name.strip().lower()
    backend = BACKENDS.get(name)
    if backend is None:
        raise ValueError(f"Unknown TTS backend: {name!r}. Available: {list(BACKENDS.keys())}")
    if not instantiate:
        return backend
    if isinstance(backend, type):
        return backend(**kwargs)
    if callable(backend):
        return backend(**kwargs) if kwargs else backend()
    return backend


get_backend.registry = BACKENDS  # type: ignore[attr-defined]


def gemini_speak(text: str) -> bytes:
    """Convenience helper: Gemini TTS, WAV output at 24 kHz."""
    backend = GeminiTTSBackend()
    pcm = backend.synthesize(text)
    return pcm_to_wav(pcm, sample_rate=backend.sample_rate)


def kokoro_speak(text: str) -> bytes:
    """Convenience helper: Kokoro TTS, WAV output at 24 kHz."""
    backend = KokoroTTSBackend()
    pcm = backend.synthesize(text)
    return pcm_to_wav(pcm, sample_rate=backend.sample_rate)

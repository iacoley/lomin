"""Streaming SSE Brain Consumer and HTTP Stream Client.

High-performance, bounded Server-Sent Events (SSE) consumer for LLM brain streaming.
Features:
- Strict character caps on total stream, oversized SSE lines, and single 'reply' events.
- Linear-time O(N) accumulator avoiding quadratic string join overhead.
- Timeout-guarded line iterator interrupting stalled socket/stream reads cleanly.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.request import Request, urlopen

# Default character ceiling for SSE brain streams (256 KiB default)
MAX_SSE_CHARS: int = int(os.environ.get("MVG_MAX_SSE_CHARS", 256 * 1024))

# Default ceiling for a single SSE line (to prevent memory bloat on runaway non-JSON streams)
MAX_SSE_LINE_CHARS: int = int(os.environ.get("MVG_MAX_SSE_LINE_CHARS", 16 * 1024 * 1024))

# Default per-read timeout for stream reads (in seconds; None = disabled by default)
DEFAULT_STREAM_READ_TIMEOUT: Optional[float] = (
    float(os.environ["MVG_STREAM_READ_TIMEOUT"])
    if "MVG_STREAM_READ_TIMEOUT" in os.environ
    else None
)


class StreamTimeoutError(TimeoutError):
    """Raised when an SSE stream read stalls beyond timeout bounds."""

    def __init__(self, message: str = "SSE stream read timed out", partial_text: str = ""):
        super().__init__(message)
        self.partial_text = partial_text


def iter_lines_with_timeout(
    lines: Iterable[Union[str, bytes]],
    read_timeout: Optional[float] = None,
    max_stream_seconds: Optional[float] = None,
    raise_on_timeout: bool = False,
    accumulated_getter: Optional[Callable[[], str]] = None,
) -> Iterator[Union[str, bytes]]:
    """Iterate over lines enforcing a per-read timeout and total stream deadline.

    If a read stalls beyond read_timeout or max_stream_seconds, interrupts cleanly
    (terminating iteration) or raises StreamTimeoutError if raise_on_timeout is True.
    """
    if read_timeout is None and max_stream_seconds is None:
        for line in lines:
            yield line
        return

    # In-memory sequences (list, tuple) cannot block on socket I/O
    if isinstance(lines, (list, tuple)):
        start_time = time.monotonic()
        for line in lines:
            if max_stream_seconds is not None and (time.monotonic() - start_time) > max_stream_seconds:
                if raise_on_timeout:
                    partial = accumulated_getter() if accumulated_getter else ""
                    raise StreamTimeoutError(
                        f"Stream deadline exceeded ({max_stream_seconds}s)",
                        partial_text=partial,
                    )
                break
            yield line
        return

    # Dynamic iterable / stream / socket: run reader in daemon thread with bounded queue
    start_time = time.monotonic()
    q: queue.Queue = queue.Queue(maxsize=128)
    sentinel = object()
    exception_sentinel = object()
    stop_event = threading.Event()

    # Attempt to set raw socket timeout if available
    raw_sock = None
    if hasattr(lines, "fp") and hasattr(lines.fp, "_sock"):
        raw_sock = lines.fp._sock
    elif hasattr(lines, "raw") and hasattr(lines.raw, "_sock"):
        raw_sock = lines.raw._sock
    elif hasattr(lines, "_sock"):
        raw_sock = lines._sock

    if raw_sock is not None and read_timeout is not None:
        try:
            raw_sock.settimeout(read_timeout)
        except Exception:
            pass

    def reader_worker():
        try:
            for raw_item in lines:
                if stop_event.is_set():
                    break
                while not stop_event.is_set():
                    try:
                        q.put(raw_item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            if not stop_event.is_set():
                q.put((exception_sentinel, exc))
        finally:
            q.put(sentinel)

    t = threading.Thread(target=reader_worker, daemon=True)
    t.start()

    try:
        while True:
            now = time.monotonic()
            remaining_deadline = None
            if max_stream_seconds is not None:
                elapsed = now - start_time
                remaining_deadline = max_stream_seconds - elapsed
                if remaining_deadline <= 0:
                    stop_event.set()
                    if hasattr(lines, "close"):
                        try:
                            lines.close()
                        except Exception:
                            pass
                    if raise_on_timeout:
                        partial = accumulated_getter() if accumulated_getter else ""
                        raise StreamTimeoutError(
                            f"Stream deadline exceeded ({max_stream_seconds}s)",
                            partial_text=partial,
                        )
                    break

            if read_timeout is not None and remaining_deadline is not None:
                wait_time = max(0.001, min(read_timeout, remaining_deadline))
            elif read_timeout is not None:
                wait_time = max(0.001, read_timeout)
            elif remaining_deadline is not None:
                wait_time = max(0.001, remaining_deadline)
            else:
                wait_time = None

            try:
                item = q.get(timeout=wait_time)
            except queue.Empty:
                # Stalled read detected!
                stop_event.set()
                if hasattr(lines, "close"):
                    try:
                        lines.close()
                    except Exception:
                        pass
                partial = accumulated_getter() if accumulated_getter else ""
                if raise_on_timeout:
                    timeout_kind = (
                        "read_timeout"
                        if (
                            read_timeout is not None
                            and (wait_time == read_timeout or remaining_deadline is None or read_timeout <= remaining_deadline)
                        )
                        else "stream_deadline"
                    )
                    raise StreamTimeoutError(
                        f"Stream stalled beyond {timeout_kind} ({wait_time:.3f}s)",
                        partial_text=partial,
                    )
                break

            if item is sentinel:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] is exception_sentinel:
                raise item[1]

            yield item
    finally:
        stop_event.set()


# on_partial coalescing (cumulative mode): see _emit_partial in consume_brain_sse_lines.
SENTENCE_BOUNDARY_CHARS = (".", "?", "!", "\n")
EMIT_MIN_INTERVAL_SEC = 0.150
EMIT_MIN_NEW_CHARS = 400


def consume_brain_sse_lines(
    lines: Iterable[Union[str, bytes]],
    on_partial: Optional[Callable[[str], None]] = None,
    max_stream_seconds: Optional[float] = None,
    max_chars: Optional[int] = None,
    read_timeout: Optional[float] = None,
    raise_on_timeout: bool = False,
    incremental: bool = False,
) -> str:
    """Consume Server-Sent Events (SSE) lines from an HTTP brain stream.

    Supports SSE `event: sentence`, `event: delta`, and `event: reply`
    (with JSON {"text": ...} or {"reply": ...}) as well as OpenAI-style
    `data: {"choices": [{"delta": {"content": ...}}]}` lines.

    Fixes & Guards:
    - Enforces max_chars on total stream, single oversized SSE lines, and single 'reply' events.
    - Avoids quadratic string join overhead: maintains a linear running string accumulator.
    - Enforces read_timeout and max_stream_seconds: interrupts cleanly or raises StreamTimeoutError.
    - Supports incremental delivery via incremental=True (on_partial receives only the delta).

    Returns:
        The final reply string (strictly bounded by max_chars).
    """
    if max_chars is None:
        max_chars = int(os.environ.get("MVG_MAX_SSE_CHARS", 256 * 1024))
    if read_timeout is None:
        read_timeout = DEFAULT_STREAM_READ_TIMEOUT

    current_event = ""
    accumulated = ""
    current_len = 0
    final_reply = ""
    last_emitted_acc: Optional[str] = None
    start_time = time.monotonic()

    def _get_accumulated() -> str:
        return (final_reply[:max_chars] if max_chars else final_reply) if final_reply else (accumulated[:max_chars] if max_chars else accumulated)

    # Cumulative on_partial is coalesced: a call is made only when text added since
    # the last call contains a sentence boundary AND (at least EMIT_MIN_INTERVAL_SEC
    # has passed OR at least EMIT_MIN_NEW_CHARS new characters have arrived). The
    # stream end always flushes. Boundaries are tracked per delta (O(delta)), never
    # by rescanning the accumulated text, so cost stays linear.
    last_emit_time = float("-inf")
    last_emit_len = 0
    pending_boundary = False

    def _emit_partial(new_acc: str, delta: str, force_sentence: bool = False) -> None:
        nonlocal last_emitted_acc, last_emit_time, last_emit_len, pending_boundary
        if on_partial is None:
            return
        if incremental:
            if delta:
                on_partial(delta)
            return
        if len(new_acc) < last_emit_len:
            # A reply event replaced the accumulated text with something shorter.
            last_emit_len = 0
        if force_sentence or (delta and any(ch in delta for ch in SENTENCE_BOUNDARY_CHARS)):
            pending_boundary = True
        if not pending_boundary:
            return
        now = time.monotonic()
        if (now - last_emit_time) >= EMIT_MIN_INTERVAL_SEC or (len(new_acc) - last_emit_len) >= EMIT_MIN_NEW_CHARS:
            on_partial(new_acc)
            last_emitted_acc = new_acc
            last_emit_time = now
            last_emit_len = len(new_acc)
            pending_boundary = False

    timed_lines = iter_lines_with_timeout(
        lines,
        read_timeout=read_timeout,
        max_stream_seconds=max_stream_seconds,
        raise_on_timeout=raise_on_timeout,
        accumulated_getter=_get_accumulated,
    )

    # Line ceiling: prevent memory blowup on oversized raw SSE lines
    max_line_len = max(max_chars + 4096, MAX_SSE_LINE_CHARS)

    try:
        for raw_line in timed_lines:
            if max_stream_seconds is not None and (time.monotonic() - start_time) > max_stream_seconds:
                if raise_on_timeout:
                    raise StreamTimeoutError(
                        f"Stream deadline exceeded ({max_stream_seconds}s)",
                        partial_text=_get_accumulated(),
                    )
                break

            if isinstance(raw_line, (bytes, bytearray)):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = str(raw_line)

            # Enforce cap on single oversized SSE lines
            if len(line) > max_line_len:
                line = line[:max_line_len]

            line = line.rstrip("\r\n")

            if not line:
                current_event = ""
                continue

            if line.startswith(":"):
                continue

            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    payload = json.loads(data_str)
                except Exception:
                    payload = data_str
                    # If JSON parsing failed but string looks like a JSON reply field, extract inner text safely
                    if isinstance(data_str, str) and '"reply":' in data_str:
                        m = re.search(r'"reply":\s*"([^"]*)', data_str)
                        if m:
                            payload = {"reply": m.group(1)}

                # 1. OpenAI-style choices -> delta -> content
                if isinstance(payload, dict) and "choices" in payload:
                    choices = payload.get("choices")
                    if isinstance(choices, list) and choices:
                        c0 = choices[0]
                        if isinstance(c0, dict):
                            delta = c0.get("delta")
                            if isinstance(delta, dict):
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    if current_len + len(content) > max_chars:
                                        allowed = max(0, max_chars - current_len)
                                        if allowed > 0:
                                            piece = content[:allowed]
                                            accumulated += piece
                                            current_len += allowed
                                            _emit_partial(accumulated, piece)
                                        break
                                    accumulated += content
                                    current_len += len(content)
                                    _emit_partial(accumulated, content)
                                    continue

                # 2. Extract payload chunk and detect reply event
                chunk = ""
                is_reply_event = False
                if isinstance(payload, dict):
                    if current_event == "reply":
                        is_reply_event = True
                        chunk = payload.get("reply") or payload.get("text") or ""
                    elif current_event == "sentence":
                        chunk = payload.get("text") or payload.get("reply") or ""
                    elif current_event == "delta":
                        chunk = payload.get("text") or payload.get("delta") or payload.get("reply") or ""
                    else:
                        if "reply" in payload and not ("text" in payload or "delta" in payload):
                            is_reply_event = True
                            chunk = payload["reply"]
                        elif "text" in payload:
                            chunk = payload["text"]
                        elif "delta" in payload:
                            chunk = payload["delta"]
                        else:
                            chunk = payload.get("reply") or ""
                elif isinstance(payload, str):
                    chunk = payload

                if not isinstance(chunk, str) or not chunk:
                    continue

                # 3. Handle 'reply' event: strictly bound against max_chars
                if is_reply_event or current_event == "reply":
                    if len(chunk) > max_chars:
                        chunk = chunk[:max_chars]
                    final_reply = chunk[:max_chars] if max_chars else chunk
                    accumulated = chunk
                    current_len = len(chunk)
                    _emit_partial(accumulated, chunk, force_sentence=True)
                    continue

                # 4. Handle 'sentence' event
                if current_event == "sentence":
                    sep = ""
                    if accumulated and not accumulated.endswith((" ", "\n")) and not chunk.startswith(" "):
                        sep = " "
                    to_add = sep + chunk if sep else chunk
                    if current_len + len(to_add) > max_chars:
                        allowed = max(0, max_chars - current_len)
                        if allowed > 0:
                            piece = to_add[:allowed]
                            accumulated += piece
                            current_len += allowed
                            _emit_partial(accumulated, piece)
                        break
                    accumulated += to_add
                    current_len += len(to_add)
                    _emit_partial(accumulated, to_add, force_sentence=True)

                # 5. Handle 'delta' event
                elif current_event == "delta":
                    to_add = chunk
                    if current_len + len(to_add) > max_chars:
                        allowed = max(0, max_chars - current_len)
                        if allowed > 0:
                            piece = to_add[:allowed]
                            accumulated += piece
                            current_len += allowed
                            _emit_partial(accumulated, piece)
                        break
                    accumulated += to_add
                    current_len += len(to_add)
                    _emit_partial(accumulated, to_add)

                # 6. Fallback / generic text chunks
                else:
                    to_add = chunk
                    if current_len + len(to_add) > max_chars:
                        allowed = max(0, max_chars - current_len)
                        if allowed > 0:
                            piece = to_add[:allowed]
                            accumulated += piece
                            current_len += allowed
                            _emit_partial(accumulated, piece)
                        break
                    accumulated += to_add
                    current_len += len(to_add)
                    _emit_partial(accumulated, to_add)
    finally:
        if not incremental and on_partial is not None and accumulated and last_emitted_acc != accumulated:
            on_partial(accumulated)
            last_emitted_acc = accumulated
    result = final_reply if final_reply else accumulated
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


def build_http_brain_body(
    prompt: str,
    session_id: str,
    node_id: str,
    speaker: str = "",
    speaker_score: Optional[float] = None,
    image_b64: Optional[str] = None,
    burst_images_b64: Optional[List[str]] = None,
    audio_b64: Optional[str] = None,
    target: str = "agent",
) -> Tuple[Dict[str, Any], bytes]:
    """Shared request body builder for HTTP brain calls."""
    body_dict: Dict[str, Any] = {
        "prompt": prompt,
        "session_id": session_id,
        "node_id": node_id,
        "target": target,
    }
    if image_b64:
        body_dict["image_b64"] = image_b64
    if burst_images_b64:
        body_dict["burst_images_b64"] = burst_images_b64
    if audio_b64:
        body_dict["audio_b64"] = audio_b64
    return body_dict, json.dumps(body_dict).encode("utf-8")


def http_brain_call_stream(
    prompt: str,
    session_id: str,
    node_id: str,
    speaker: str = "",
    speaker_score: Optional[float] = None,
    timeout: int = 45,
    on_partial: Optional[Callable[[str], None]] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    image_b64: Optional[str] = None,
    burst_images_b64: Optional[List[str]] = None,
    audio_b64: Optional[str] = None,
    target: str = "agent",
    read_timeout: Optional[float] = None,
    raise_on_timeout: bool = False,
    incremental: bool = False,
    max_chars: Optional[int] = None,
) -> str:
    """HTTP brain streaming client: POSTs request and consumes SSE response."""
    endpoint = url or os.environ.get("MVG_BRAIN_URL", "")
    if not endpoint:
        raise RuntimeError("MVG_BRAIN_URL not set")

    auth_token = token if token is not None else os.environ.get("MVG_BRAIN_TOKEN", "")
    _, body = build_http_brain_body(
        prompt=prompt,
        session_id=session_id,
        node_id=node_id,
        speaker=speaker,
        speaker_score=speaker_score,
        image_b64=image_b64,
        burst_images_b64=burst_images_b64,
        audio_b64=audio_b64,
        target=target,
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    req = Request(endpoint, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        headers_obj = getattr(resp, "headers", None)
        content_type = headers_obj.get("Content-Type", "") if hasattr(headers_obj, "get") else ""
        if "text/event-stream" in content_type:
            return consume_brain_sse_lines(
                resp,
                on_partial=on_partial,
                max_stream_seconds=float(timeout),
                max_chars=max_chars,
                read_timeout=read_timeout,
                raise_on_timeout=raise_on_timeout,
                incremental=incremental,
            )
        data = json.loads(resp.read().decode("utf-8"))
        reply = data.get("reply")
        if not isinstance(reply, str):
            raise RuntimeError("MVG_BRAIN_URL response has no reply string")
        if max_chars is not None and len(reply) > max_chars:
            reply = reply[:max_chars]
        if on_partial:
            on_partial(reply)
        return reply

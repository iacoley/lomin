"""ha_fastpath.py - Home Assistant Conversation Intent Fast-Path for Lomin.

Evaluates user voice transcripts against Home Assistant's local
`/api/conversation/process` endpoint when MVG_HA_ENABLED is '1'.
Short-circuits before calling the LLM brain when a deterministic smart-home
intent matches ('action_done' or 'query_answer').
"""

import json
import os
import urllib.error
import urllib.request


def _default_http_poster(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_s: float = 2.5,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}


ERROR_CODES = frozenset({
    "no_intent_matched",
    "intent_not_recognized",
    "failed_to_handle",
    "unknown",
    "timer_not_supported",
})


def try_ha_conversation(
    transcript: str,
    *,
    has_images: bool = False,
    enabled: bool | None = None,
    ha_url: str | None = None,
    ha_token: str | None = None,
    timeout_s: float = 2.5,
    http_poster=None,
) -> tuple[bool, str, dict]:
    """Evaluate transcript against Home Assistant's conversation API.

    Returns:
        (True, speech, meta) on matched intent with non-empty speech.
        (False, '', meta) when disabled, vision turn, no intent matched,
        or on network/HTTP errors.
    """
    eff_enabled = (os.environ.get("MVG_HA_ENABLED", "0") == "1" or os.environ.get("MVG_HA_FASTPATH", "0") == "1") if enabled is None else bool(enabled)
    if not eff_enabled or has_images:
        return False, "", {"reason": "disabled_or_vision"}

    clean_text = (transcript or "").strip()
    if not clean_text:
        return False, "", {"reason": "empty_transcript"}

    eff_url = os.environ.get("MVG_HA_URL", "") if ha_url is None else str(ha_url or "")
    eff_url = eff_url.strip().rstrip("/")
    if not eff_url:
        return False, "", {"reason": "missing_ha_url"}

    eff_token = os.environ.get("MVG_HA_TOKEN", "") if ha_token is None else str(ha_token or "")
    eff_token = eff_token.strip()

    endpoint = eff_url if eff_url.endswith("/api/conversation/process") else f"{eff_url}/api/conversation/process"

    headers = {"Content-Type": "application/json"}
    if eff_token:
        headers["Authorization"] = f"Bearer {eff_token}"

    payload = {"text": clean_text, "language": "en"}
    poster = http_poster or _default_http_poster

    try:
        try:
            raw_res = poster(endpoint, headers, payload, timeout_s)
        except TypeError:
            raw_res = poster(endpoint, headers, payload)
    except Exception as exc:
        return False, "", {
            "reason": "network_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }

    if hasattr(raw_res, "read"):
        raw_res = raw_res.read()
    if isinstance(raw_res, (bytes, bytearray)):
        raw_res = raw_res.decode("utf-8", errors="replace")
    if isinstance(raw_res, str):
        try:
            raw_res = json.loads(raw_res) if raw_res.strip() else {}
        except Exception:
            return False, "", {"reason": "invalid_json"}

    if not isinstance(raw_res, dict):
        return False, "", {"reason": "invalid_json"}

    resp_obj = raw_res.get("response")
    if not isinstance(resp_obj, dict):
        if "response_type" in raw_res:
            resp_obj = raw_res
        else:
            return False, "", {"reason": "missing_response_object"}

    resp_type = str(resp_obj.get("response_type") or "").strip().lower()
    data_obj = resp_obj.get("data") if isinstance(resp_obj.get("data"), dict) else {}
    err_code = str(data_obj.get("code") or "").strip().lower()

    if resp_type == "error" or err_code in ERROR_CODES:
        return False, "", {
            "reason": "no_intent_matched",
            "response_type": resp_type,
            "code": err_code or "no_intent_matched",
            "data": data_obj,
        }

    if resp_type not in ("action_done", "query_answer"):
        return False, "", {
            "reason": f"unsupported_response_type:{resp_type}",
            "response_type": resp_type,
            "data": data_obj,
        }

    speech_val = resp_obj.get("speech")
    speech_text = ""
    if isinstance(speech_val, dict):
        plain = speech_val.get("plain")
        if isinstance(plain, dict):
            speech_text = str(plain.get("speech") or "").strip()
        elif isinstance(plain, str):
            speech_text = plain.strip()
        elif "speech" in speech_val and isinstance(speech_val["speech"], str):
            speech_text = speech_val["speech"].strip()
    elif isinstance(speech_val, str):
        speech_text = speech_val.strip()

    if not speech_text:
        return False, "", {
            "reason": "empty_speech",
            "response_type": resp_type,
            "data": data_obj,
        }

    meta = {
        "reason": "action_done" if resp_type == "action_done" else resp_type,
        "response_type": resp_type,
        "data": data_obj,
        "conversation_id": raw_res.get("conversation_id"),
    }
    return True, speech_text, meta

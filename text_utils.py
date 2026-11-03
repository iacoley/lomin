"""Text helpers for spoken replies (moved out of gateway.py, ERED LOMIN P1b).

Stdlib only. gateway.py re-imports every name."""

import re


# --- Markdown stripping (for spoken replies) -----------------------------

_MD_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


def stable_markdown_prefix(text: str) -> str:
    """The part of a still-streaming reply whose markdown cannot change once
    more text arrives. Cuts before an unclosed code fence, an unclosed inline
    backtick, or a link whose "](url)" has not closed yet. Without the cut,
    strip_markdown of a later partial can be shorter than an earlier one
    (a link collapses to its label), and slicing by the earlier length skips
    or drops reply text."""
    if not text:
        return text
    cut = len(text)
    fences = [m.start() for m in re.finditer(r"```", text)]
    if len(fences) % 2:
        cut = min(cut, fences[-1])
    head = text[:cut]
    # inline backticks outside closed fences
    body = _MD_CODE_BLOCK_RE.sub(lambda m: " " * len(m.group(0)), head)
    ticks = [i for i, ch in enumerate(body) if ch == "`"]
    if len(ticks) % 2:
        cut = min(cut, ticks[-1])
    head = text[:cut]
    # a "[" that may still become a link: no "]" yet, "]" at the very end,
    # or "](" whose ")" has not arrived
    for m in re.finditer(r"\[", head):
        rest = head[m.start():]
        if _MD_LINK_RE.match(rest):
            continue
        close = rest.find("]")
        after = rest[close + 1:] if close != -1 else ""
        if close == -1 or after == "" or (after.startswith("(") and ")" not in after):
            cut = min(cut, m.start())
            break
    return text[:cut]


def unspoken_delta(full_stripped: str, spoken: str) -> str:
    """Text in full_stripped not yet spoken. Normally full_stripped extends
    spoken. If the streamed text already ends with the whole of full_stripped
    (an earlier block, then this reply), nothing is left. If an already-spoken
    span was rewritten, resume at the longest common prefix: repeating a few
    words beats silently skipping new ones."""
    if full_stripped.startswith(spoken):
        return full_stripped[len(spoken):]
    tail = full_stripped.strip()
    if tail and spoken.rstrip().endswith(tail):
        # The final reply is already the last thing spoken: the streamed text
        # carried an earlier block (narration before a tool call) and then
        # this reply, so the final does not extend it but was heard in full.
        return ""
    n = 0
    for a, b in zip(full_stripped, spoken):
        if a != b:
            break
        n += 1
    return full_stripped[n:]


def strip_markdown(text: str) -> str:
    """Best-effort strip of markdown so a spoken reply doesn't say
    "asterisk asterisk" or read a code block aloud. Not a full parser —
    good enough for the short spoken sentences /ask is asked to produce."""
    if not text:
        return text
    text = _MD_CODE_BLOCK_RE.sub(" ", text)
    text = _MD_INLINE_CODE_RE.sub(r"\1", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

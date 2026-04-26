"""Mechanical secret redaction for logs and stored memories.

Replaces known secret patterns with `[REDACTED:<class>]` tokens. Designed to run
on user-supplied content before it reaches the MemReader LLM, on the structured
output the LLM returns, and as a logging filter (defense in depth).

False-positive cost is intentionally low: tagging a legitimate email address as
`[REDACTED:email]` is acceptable. False-negative cost is high: a missed API key
ends up persisted in Qdrant + Neo4j and on disk in the log file. The patterns
err on aggressive.

Performance contract: compiled regex objects, no per-call rebuilds; ≤1ms per
1KB, ≤100ms per 1MB on contemporary hardware.
"""

from __future__ import annotations

import re

from typing import Any


__all__ = ["redact", "redact_dict", "REDACTION_CLASSES"]


# Per-pattern redactors. Each entry: (class_name, sentinel_substring,
# compiled_regex, replacement). The sentinel substring is used as a cheap
# substring fast-path: if it isn't present in the text, the regex is skipped
# entirely. Empirically this is ~3× faster than a single alternation regex
# on multi-megabyte payloads, because Python's `re` engine can use
# Boyer-Moore-style literal-prefix scanning on each individual pattern but
# falls back to per-position state-machine work on alternation.
#
# Order matters: each pass operates on the output of the previous, and once
# a span has been replaced with `[REDACTED:<class>]`, later patterns will
# not re-match it. So put the most specific / highest-confidence patterns
# first.
#
# - PEM first: the body of a multiline PEM block must not be re-matched by
#   per-line patterns.
# - JWT before bearer: `Bearer eyJ...` should be tagged jwt (the JWT body
#   is the actual secret, not the literal word "Bearer").
# - SSN before phone: `\d{3}-\d{2}-\d{4}` would otherwise match phone too.
# - Email before phone: an email address contains digits adjacent to '@'
#   that we don't want grabbed by phone.
# - Card has its own pass with Luhn validation; phone runs last.
#
# The sentinel may be `None` for patterns we always run (no cheap literal).
_PATTERNS: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = (
    (
        "pem",
        ("BEGIN ",),
        re.compile(r"-----BEGIN [A-Z0-9 ]+-----[\s\S]+?-----END [A-Z0-9 ]+-----"),
    ),
    (
        "jwt",
        ("eyJ",),
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    ),
    (
        "bearer",
        ("bearer", "Bearer", "BEARER"),
        re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE),
    ),
    (
        "sk-key",
        ("sk-",),
        re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    ),
    (
        "aws-key",
        ("AKIA", "aws_secret", "AWS_SECRET"),
        re.compile(
            r"\bAKIA[0-9A-Z]{16}\b"
            r"|aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}",
            re.IGNORECASE,
        ),
    ),
    (
        "ssn",
        ("-",),  # any dash is required for an SSN; cheap test
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "email",
        ("@",),
        re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    ),
)


# Card numbers need Luhn validation. Match candidate digit run of 13–19 digits
# (with optional spaces / dashes as separators), then check Luhn.
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")

# Phone — loose but anchored to plausible international/local shape. Runs
# AFTER the per-pattern passes and the card pass, so a Luhn-valid card never
# falls through here, and an SSN or email-embedded digit run is already gone.
_PHONE = re.compile(
    r"(?<![\w.])"  # not glued to a word/number to the left
    r"\+?\d{1,3}[\s\-.]?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}"
    r"(?!\d)",
)

# Quick "any digit?" check used to skip the card / phone passes on text that
# has no digits at all (typical for prose snippets).
_HAS_DIGIT = re.compile(r"\d")


REDACTION_CLASSES: tuple[str, ...] = (
    "pem",
    "jwt",
    "bearer",
    "sk-key",
    "aws-key",
    "ssn",
    "email",
    "card",
    "phone",
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = ord(ch) - 48  # ord('0') == 48
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_card(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        digits = re.sub(r"[ \-]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "[REDACTED:card]"
        return match.group(0)

    return _CARD_CANDIDATE.sub(_repl, text)


def redact(text: Any) -> Any:
    """Replace known secret patterns in ``text`` with ``[REDACTED:<class>]``.

    Non-string inputs are returned unchanged so callers can pass arbitrary log
    args through without isinstance gymnastics. Always returns a new string;
    the input is never mutated.
    """
    if not isinstance(text, str) or not text:
        return text

    out = text
    for cls, sentinels, pattern in _PATTERNS:
        if not any(s in out for s in sentinels):
            continue
        out = pattern.sub(f"[REDACTED:{cls}]", out)

    if _HAS_DIGIT.search(out):
        out = _redact_card(out)
        out = _PHONE.sub("[REDACTED:phone]", out)
    return out


def redact_dict(obj: Any) -> Any:
    """Recursively redact every string leaf in ``obj``.

    Walks dicts and lists/tuples/sets. Non-string scalars (int, bool, None,
    datetime, etc.) pass through unchanged. The structure is rebuilt; the
    input is not mutated.
    """
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_dict(v) for v in obj)
    if isinstance(obj, set):
        return {redact_dict(v) for v in obj}
    return obj

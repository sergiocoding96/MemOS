"""Integration tests for the redaction wiring across MemOS.

These exercise the actual call sites that the redactor hooks into:

1. ``_get_llm_response`` — pre-extraction redaction of the conversation text
   that reaches the MemReader LLM, and post-extraction redaction of the
   structured output the LLM returns.
2. ``_build_fast_node`` (via the fast-mode chat path) — fast-mode bypasses
   the LLM entirely and writes user content straight into Qdrant + Neo4j.
3. ``add_handler.log_add_messages`` — secrets in the message body must not
   appear verbatim in the parse-error log line.
4. ``RedactionFilter`` — defense-in-depth filter that scrubs every log
   record before it reaches a handler.

Real Qdrant + Neo4j storage is not exercised here — instead we verify that
the value reaching the persistence layer is already redacted.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. MemReader LLM call site: pre + post redaction.
# ---------------------------------------------------------------------------


class _StubReader:
    """Just enough of SimpleStructMemReader to drive ``_get_llm_response``.

    We bind the real method via ``__get__`` so the test is hitting the actual
    implementation path, not a reimplementation.
    """

    def __init__(self, llm_raw_response: str) -> None:
        self.llm_raw_response = llm_raw_response
        self.config = MagicMock()
        self.config.remove_prompt_example = False
        self.last_llm_messages: list | None = None

    def _safe_generate(self, messages):
        self.last_llm_messages = messages
        return self.llm_raw_response

    def _safe_parse(self, text):
        # Pretend the LLM returned a structured result that quotes the
        # already-redacted prompt content. (Real MemReader parses JSON; we
        # just hand back a dict to drive the post-extraction path.)
        return {
            "memory_list": [
                {
                    "key": "k",
                    "memory_type": "UserMemory",
                    "value": text,  # echoes back; could carry secrets
                    "tags": [],
                }
            ],
            "summary": text,
        }


def _bind_get_llm_response(stub):
    from memos.mem_reader.simple_struct import SimpleStructMemReader

    return SimpleStructMemReader._get_llm_response.__get__(stub, SimpleStructMemReader)


def test_get_llm_response_pre_extraction_redacts_input() -> None:
    """The prompt that hits the LLM must not contain raw secrets."""
    stub = _StubReader(llm_raw_response='{"memory_list": []}')
    get_llm_response = _bind_get_llm_response(stub)

    user_text = "user pasted Bearer abc123def456ghi789 and sk-test-1234567890ABCDEF"
    get_llm_response(user_text, custom_tags=None)

    assert stub.last_llm_messages is not None
    prompt = stub.last_llm_messages[0]["content"]
    assert "abc123def456ghi789" not in prompt, "raw bearer token reached LLM prompt"
    assert "sk-test-1234567890ABCDEF" not in prompt, "raw sk-key reached LLM prompt"
    assert "[REDACTED:bearer]" in prompt
    assert "[REDACTED:sk-key]" in prompt


def test_get_llm_response_post_extraction_redacts_output() -> None:
    """The structured output must not contain raw secrets even if the LLM
    re-quoted them or hallucinated them back."""

    # Stub returns a "parsed" dict that contains a fresh secret which was
    # NOT in the prompt — simulating the LLM hallucinating a key.
    class _LeakyReader(_StubReader):
        def _safe_parse(self, text):
            return {
                "memory_list": [
                    {
                        "key": "k",
                        "memory_type": "UserMemory",
                        "value": "ok",
                        "tags": ["see AKIAIOSFODNN7EXAMPLE"],
                    }
                ],
                "summary": "leaked: alice@example.com",
            }

    stub = _LeakyReader(llm_raw_response="anything")
    get_llm_response = _bind_get_llm_response(stub)

    out = get_llm_response("benign text with no secrets", custom_tags=None)
    assert "AKIAIOSFODNN7EXAMPLE" not in str(out)
    assert "alice@example.com" not in str(out)
    assert "[REDACTED:aws-key]" in out["memory_list"][0]["tags"][0]
    assert "[REDACTED:email]" in out["summary"]


def test_get_llm_response_fallback_path_redacts() -> None:
    """When the LLM fails to produce a parseable response, the fallback
    memory_list and summary echo back the user content. They must echo back
    the *redacted* text, not the raw secret."""

    class _FailingReader(_StubReader):
        def _safe_parse(self, text):
            return None  # forces the fallback branch

    stub = _FailingReader(llm_raw_response="garbage")
    get_llm_response = _bind_get_llm_response(stub)

    user_text = "leaked sk-test-1234567890ABCDEF here"
    out = get_llm_response(user_text, custom_tags=None)

    # Fallback path quotes mem_str into both `value` and `summary`.
    assert "sk-test-1234567890ABCDEF" not in out["summary"]
    assert "sk-test-1234567890ABCDEF" not in out["memory_list"][0]["value"]
    assert "[REDACTED:sk-key]" in out["summary"]


# ---------------------------------------------------------------------------
# 2. add_handler — parse error path on line 56.
# ---------------------------------------------------------------------------


def test_add_handler_parse_error_redacts_content_in_log(caplog: pytest.LogCaptureFixture) -> None:
    """The `add_handler.log_add_messages` parse-error path must redact the
    content before logging it. Mirrors the F-09 finding from the audit."""
    from memos.mem_scheduler.task_schedule_modules.handlers.add_handler import (
        AddMessageHandler,
    )

    handler = AddMessageHandler.__new__(AddMessageHandler)
    handler.scheduler_context = MagicMock()
    handler.scheduler_context.get_mem_cube.return_value = MagicMock(
        text_mem=MagicMock(get=MagicMock(side_effect=ValueError("boom"))),
    )

    msg = MagicMock()
    msg.content = "not-json: Bearer abc123def456ghi789 sk-test-AAAAAAAAAAAAAAAA"
    msg.user_id = "u"
    msg.mem_cube_id = "c"
    msg.task_id = "t"
    msg.item_id = "i"
    msg.label = "L"

    with caplog.at_level(logging.ERROR, logger=""):
        handler.log_add_messages(msg)

    captured = caplog.text
    assert "abc123def456ghi789" not in captured, f"raw bearer leaked: {captured!r}"
    assert "sk-test-AAAAAAAAAAAAAAAA" not in captured, f"raw sk-key leaked: {captured!r}"
    assert "[REDACTED:" in captured


# ---------------------------------------------------------------------------
# 3. RedactionFilter — defense in depth.
# ---------------------------------------------------------------------------


def test_redaction_filter_scrubs_msg_and_args() -> None:
    from memos.log import RedactionFilter

    f = RedactionFilter()
    record = logging.LogRecord(
        name="t",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="user pasted %s plus %s",
        args=("Bearer abc123def456ghi789", "sk-test-1234567890ABCDEF"),
        exc_info=None,
    )
    assert f.filter(record) is True
    out = record.getMessage()
    assert "abc123def456ghi789" not in out
    assert "sk-test-1234567890ABCDEF" not in out
    assert "[REDACTED:bearer]" in out
    assert "[REDACTED:sk-key]" in out


def test_redaction_filter_passes_clean_records_through() -> None:
    from memos.log import RedactionFilter

    f = RedactionFilter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="processed %d items",
        args=(7,),
        exc_info=None,
    )
    assert f.filter(record) is True
    # Clean records pass through with original args intact.
    assert record.args == (7,)
    assert "processed 7 items" == record.getMessage()


def test_redaction_filter_end_to_end_through_handler() -> None:
    """A full handler path: configure a fresh logger with a memory handler
    and the RedactionFilter, log a record with a secret, and check the
    rendered output."""
    from memos.log import RedactionFilter

    buf = io.StringIO()
    logger = logging.getLogger("redaction_filter_e2e_test")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(RedactionFilter())
    logger.addHandler(handler)

    logger.error(
        "auth failed for token=%s and key=%s",
        "Bearer abc123def456ghi789",
        "sk-test-1234567890ABCDEF",
    )

    rendered = buf.getvalue()
    assert "abc123def456ghi789" not in rendered, rendered
    assert "sk-test-1234567890ABCDEF" not in rendered, rendered
    assert "[REDACTED:bearer]" in rendered
    assert "[REDACTED:sk-key]" in rendered


# ---------------------------------------------------------------------------
# 4. Fast-mode chat node build site.
# ---------------------------------------------------------------------------


def test_fast_mode_value_is_redacted_before_persistence() -> None:
    """Fast mode skips the LLM entirely. Without a redact pass at the build
    site, an `sk-...` key pasted into chat is stored verbatim in Qdrant +
    Neo4j. This test mirrors that path: build the fast node, capture the
    `value=` arg passed into ``_make_memory_item``, and assert it is
    redacted."""
    from memos.mem_reader.simple_struct import SimpleStructMemReader

    reader = SimpleStructMemReader.__new__(SimpleStructMemReader)
    reader._count_tokens = lambda s: max(1, len(s) // 4)
    reader.chat_window_max_tokens = 10000
    captured: dict[str, object] = {}

    def fake_make_memory_item(value, info, memory_type, tags, sources, **kwargs):
        captured["value"] = value
        captured["tags"] = tags
        return MagicMock()

    reader._make_memory_item = fake_make_memory_item

    scene = [
        {"role": "user", "content": "leaked Bearer abc123def456ghi789", "chat_time": ""},
        {"role": "user", "content": "leaked sk-test-1234567890ABCDEF", "chat_time": ""},
    ]

    # Force the fast-mode path; chunk_tokens large so no sub-chunking.
    with patch.dict("os.environ", {"MOS_FAST_CHUNK_TOKENS": "10000"}):
        reader._process_chat_data(
            scene,
            info={"user_id": "u", "session_id": "s"},
            mode="fast",
        )

    val = captured["value"]
    assert isinstance(val, str)
    assert "abc123def456ghi789" not in val, val
    assert "sk-test-1234567890ABCDEF" not in val, val
    assert "[REDACTED:bearer]" in val
    assert "[REDACTED:sk-key]" in val

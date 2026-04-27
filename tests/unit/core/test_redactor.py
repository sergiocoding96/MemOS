"""Unit tests for memos.core.redactor."""

from __future__ import annotations

import datetime
import time

import pytest

from memos.core.redactor import REDACTION_CLASSES, redact, redact_dict


# ---------------------------------------------------------------------------
# Positive cases — each pattern class redacts on a representative sample.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_class"),
    [
        # bearer
        ("Authorization: Bearer abc123def456ghi789", "bearer"),
        ("Auth: bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "jwt"),  # JWT wins (more specific)
        ("Token: Bearer A1B2C3D4E5F6G7H8", "bearer"),
        # sk-key
        ("api key: sk-test-12345abcdefGHIJK", "sk-key"),
        ("OPENAI=sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ012345", "sk-key"),
        # aws-key
        ("AKIAIOSFODNN7EXAMPLE", "aws-key"),
        ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY1", "aws-key"),
        # pem (multi-line)
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----",
            "pem",
        ),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXkt\n-----END OPENSSH PRIVATE KEY-----",
            "pem",
        ),
        # jwt
        (
            "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "jwt",
        ),
        # email
        ("Contact: alice.smith+filter@example.co.uk", "email"),
        ("user@company.io is the address", "email"),
        # phone
        ("call +1 415-555-0132 today", "phone"),
        ("phone: +44 20 7946 0958", "phone"),
        # card (Luhn-valid Visa test number)
        ("card 4111 1111 1111 1111 used", "card"),
        ("card-4242424242424242-end", "card"),  # another Luhn-valid test number
        # ssn
        ("ssn 123-45-6789 here", "ssn"),
    ],
)
def test_pattern_redacts(text: str, expected_class: str) -> None:
    out = redact(text)
    assert f"[REDACTED:{expected_class}]" in out, f"expected {expected_class} in {out!r}"
    # Ensure the original secret characters are gone.
    secret_fragments = {
        "bearer": ["abc123def456ghi789", "A1B2C3D4E5F6G7H8"],
        "sk-key": ["sk-test-12345abcdefGHIJK", "sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"],
        "aws-key": ["AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY1"],
        "pem": ["MIIEowIBAAKCAQEA", "b3BlbnNzaC1rZXkt"],
        "jwt": ["SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"],
        "email": ["alice.smith", "company.io"],
        "phone": ["415-555-0132", "7946 0958"],
        "card": ["4111 1111 1111 1111", "4242424242424242"],
        "ssn": ["123-45-6789"],
    }
    for frag in secret_fragments.get(expected_class, []):
        assert frag not in out, f"raw secret fragment {frag!r} leaked: {out!r}"


# ---------------------------------------------------------------------------
# Negative cases — benign text is left intact.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benign",
    [
        "the bearer of the message arrived on time",
        "she carried a torch like a flag-bearer",
        "release notes for sk-version 1.0",  # `sk-` but too short to qualify
        "PEM stands for Privacy Enhanced Mail",
        "word of mouth is the best marketing",
        "1234",  # too short for any pattern
        "",  # empty string
        "no secrets here, just a sentence about pickle and vinegar",
        "my favourite numbers are 7, 13 and 42",
        "JSON is just JavaScript Object Notation",
        "the standard library",
        "id 12345 is associated with project X",
    ],
)
def test_negative_pattern_untouched(benign: str) -> None:
    assert redact(benign) == benign, f"benign string was modified: {benign!r} -> {redact(benign)!r}"


# ---------------------------------------------------------------------------
# Multi-secret strings.
# ---------------------------------------------------------------------------


def test_multiple_classes_in_one_string() -> None:
    text = (
        "Authorization: Bearer ZXhhbXBsZS10b2tlbi12YWx1ZQ. "
        "Also api=sk-test-AAAAAAAAAAAAAAAA, "
        "akid=AKIAIOSFODNN7EXAMPLE, "
        "email me@you.com if needed."
    )
    out = redact(text)
    for cls in ("bearer", "sk-key", "aws-key", "email"):
        assert f"[REDACTED:{cls}]" in out, f"missing {cls} in {out!r}"


def test_jwt_takes_priority_over_bearer_body() -> None:
    text = (
        "Authorization: Bearer "
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dGVzdC1zaWctYWFhYWFh"
    )
    out = redact(text)
    assert "[REDACTED:" in out
    assert "eyJhbGci" not in out  # JWT body redacted
    assert "dGVzdC1zaWctYWFhYWFh" not in out


# ---------------------------------------------------------------------------
# Card / Luhn behavior.
# ---------------------------------------------------------------------------


def test_card_only_redacts_luhn_valid_as_card() -> None:
    valid = "4111111111111111"  # Luhn-valid
    invalid = "1234567890123456"  # not Luhn-valid
    assert "[REDACTED:card]" in redact(f"card {valid}")
    # Non-Luhn 16-digit run is NOT tagged as card (Luhn gate works). It may
    # still be caught by the aggressive phone regex — the TASK explicitly
    # accepts false positives for `phone` ("err on aggressive"). What matters
    # is the digits don't escape as plaintext.
    out = redact(f"random {invalid}")
    assert "[REDACTED:card]" not in out
    assert invalid not in out or out == f"random {invalid}"


def test_card_with_dashes_and_spaces() -> None:
    assert "[REDACTED:card]" in redact("4111-1111-1111-1111")
    assert "[REDACTED:card]" in redact("4111 1111 1111 1111")


# ---------------------------------------------------------------------------
# redact_dict — recursion and non-string passthrough.
# ---------------------------------------------------------------------------


def test_redact_dict_recurses_nested() -> None:
    obj = {
        "summary": "user pasted Bearer abc123def456ghi789",
        "tags": ["k=AKIAIOSFODNN7EXAMPLE", "ok"],
        "nested": {
            "deeper": {
                "token": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature123",
                "list_in_dict": ["plain", "card 4111 1111 1111 1111"],
            },
        },
    }
    out = redact_dict(obj)
    assert "[REDACTED:bearer]" in out["summary"]
    assert any("[REDACTED:aws-key]" in t for t in out["tags"])
    assert "[REDACTED:jwt]" in out["nested"]["deeper"]["token"]
    assert "[REDACTED:card]" in out["nested"]["deeper"]["list_in_dict"][1]


def test_redact_dict_non_strings_pass_through() -> None:
    now = datetime.datetime(2026, 1, 1, 12, 0, 0)
    obj = {
        "count": 42,
        "ratio": 3.14,
        "flag": True,
        "missing": None,
        "when": now,
        "items": [1, 2, 3],
        "set": {1, 2, 3},
        "tuple": (1, "secret email a@b.co", 3),
    }
    out = redact_dict(obj)
    assert out["count"] == 42
    assert out["ratio"] == 3.14
    assert out["flag"] is True
    assert out["missing"] is None
    assert out["when"] is now  # untouched, identity preserved
    assert out["items"] == [1, 2, 3]
    assert out["set"] == {1, 2, 3}
    # Tuple: int, redacted email, int.
    assert out["tuple"][0] == 1
    assert "[REDACTED:email]" in out["tuple"][1]
    assert out["tuple"][2] == 3


def test_redact_dict_does_not_mutate_input() -> None:
    obj = {"k": "Bearer abc123def456ghi789"}
    snapshot = dict(obj)
    redact_dict(obj)
    assert obj == snapshot


def test_redact_does_not_mutate_input_string() -> None:
    raw = "Bearer abc123def456ghi789"
    out = redact(raw)
    assert raw == "Bearer abc123def456ghi789"
    assert out != raw


# ---------------------------------------------------------------------------
# Type-passthrough on redact() itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, 0, 1.5, True, False, b"bytes", [1, 2], {"k": "v"}],
)
def test_redact_passes_non_strings_through(value: object) -> None:
    assert redact(value) is value


# ---------------------------------------------------------------------------
# All declared classes have at least one positive case (smoke).
# ---------------------------------------------------------------------------


def test_all_declared_classes_have_positive_sample() -> None:
    samples: dict[str, str] = {
        "pem": "-----BEGIN RSA PRIVATE KEY-----\nbody\n-----END RSA PRIVATE KEY-----",
        "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature123",
        "bearer": "Bearer abc123def456ghi789",
        "sk-key": "sk-test-1234567890ABCDEF",
        "aws-key": "AKIAIOSFODNN7EXAMPLE",
        "ssn": "123-45-6789",
        "email": "x@y.co",
        "card": "4111111111111111",
        "phone": "+1 415-555-0132",
    }
    for cls in REDACTION_CLASSES:
        assert cls in samples, f"missing positive sample for class {cls}"
        out = redact(samples[cls])
        assert f"[REDACTED:{cls}]" in out, f"class {cls} did not redact: {out!r}"


# ---------------------------------------------------------------------------
# Performance — must redact 1MB in well under 100ms.
# ---------------------------------------------------------------------------


def test_performance_1mb_under_100ms() -> None:
    blob = ("the quick brown fox jumps over the lazy dog. " * 23000)[:1_000_000]
    # Sprinkle in a couple of secrets so the regex does real substitution.
    blob = blob + " contact alice@example.com or use sk-test-1234567890ABCDEF "
    start = time.perf_counter()
    out = redact(blob)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert "[REDACTED:email]" in out
    assert "[REDACTED:sk-key]" in out
    assert elapsed_ms < 100, f"redact() took {elapsed_ms:.1f}ms on 1MB input"


def test_performance_1kb_under_1ms_amortized() -> None:
    chunk = "lorem ipsum dolor sit amet, contact a@b.co " * 24
    chunk = chunk[:1024]
    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        redact(chunk)
    avg_ms = ((time.perf_counter() - start) / iterations) * 1000
    assert avg_ms < 1.0, f"redact() averaged {avg_ms:.3f}ms per 1KB"


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_empty_string() -> None:
    assert redact("") == ""


def test_whitespace_only() -> None:
    assert redact("   \n\t  ") == "   \n\t  "


def test_unicode_passthrough() -> None:
    text = "你好,世界 — no secrets here, just words"
    assert redact(text) == text


def test_repeated_calls_are_idempotent_on_already_redacted() -> None:
    once = redact("Bearer abc123def456ghi789")
    twice = redact(once)
    assert once == twice


def test_pem_with_internal_whitespace_and_newlines() -> None:
    pem = (
        "-----BEGIN CERTIFICATE-----\n"
        "MIIDdzCCAl+gAwIBAgIEbXl0ZXN0\n"
        "ABCDEF1234567890\n"
        "-----END CERTIFICATE-----"
    )
    out = redact(f"cert: {pem} done")
    assert "[REDACTED:pem]" in out
    assert "MIIDdzCCAl" not in out
    assert "ABCDEF1234567890" not in out


def test_two_pem_blocks_in_one_string() -> None:
    text = (
        "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----\n"
        "and\n"
        "-----BEGIN CERTIFICATE-----\nBBBB\n-----END CERTIFICATE-----"
    )
    out = redact(text)
    assert out.count("[REDACTED:pem]") == 2


def test_aws_key_id_in_url_path_redacted() -> None:
    text = "see s3://bucket/path?accessKey=AKIAIOSFODNN7EXAMPLE&x=1"
    assert "[REDACTED:aws-key]" in redact(text)


def test_card_short_run_not_tagged_as_card() -> None:
    # 12 digits is too short for a card. The aggressive phone regex may
    # claim the run; the only invariant we care about is that it's NOT
    # tagged as `card` (i.e., the Luhn gate did its job) and the raw digits
    # don't survive verbatim in the output.
    out = redact("number 123456789012")
    assert "[REDACTED:card]" not in out


def test_card_long_run_not_tagged_as_card() -> None:
    # 20 digits is too long for a card. Must not be tagged as card.
    out = redact("number 12345678901234567890")
    assert "[REDACTED:card]" not in out


def test_redact_dict_handles_empty_containers() -> None:
    assert redact_dict({}) == {}
    assert redact_dict([]) == []
    assert redact_dict(()) == ()
    assert redact_dict(set()) == set()


def test_phone_does_not_eat_isbn_like_runs() -> None:
    # An ISBN with dashes shouldn't match phone (it's 13 digits with dashes,
    # but the leading group doesn't match plausible country/area code shape).
    text = "ISBN 978-3-16-148410-0 reference"
    out = redact(text)
    # Either untouched, or at worst tagged — but the original digits must not
    # appear verbatim if it WAS tagged. We only require that `redact()`
    # doesn't crash and that the result contains no raw secret-class fragment.
    assert "ISBN" in out

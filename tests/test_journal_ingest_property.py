"""
Property-based tests for MindGuard AI Journal Ingest Lambda.

# Feature: mindguard-ai, Property 3: Text Entry Validation

For any text input, the system should reject it if it is empty, composed entirely
of whitespace, or exceeds 5,000 characters; otherwise it should be accepted.

Validates: Requirements 2.1, 2.2, 2.4
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.lambdas.journal_ingest_lambda import validate_text_entry, MAX_TEXT_LENGTH

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Whitespace-only strings: build directly from a fixed whitespace alphabet
_WHITESPACE_CHARS = " \t\n\r\x0b\x0c"
whitespace_st = st.text(
    alphabet=_WHITESPACE_CHARS,
    min_size=1,
    max_size=200,
)

# Strings that exceed the 5,000-character limit: use a short repeated pattern
# to avoid data_too_large health check — we just need len > 5000
oversized_st = st.integers(min_value=MAX_TEXT_LENGTH + 1, max_value=MAX_TEXT_LENGTH + 200).map(
    lambda n: "a" * n
)

# Valid text: non-empty, not purely whitespace, within 5,000 chars
valid_text_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=MAX_TEXT_LENGTH,
).filter(lambda t: t.strip() != "")


# ---------------------------------------------------------------------------
# Property 3: Text Entry Validation
# Validates: Requirements 2.1, 2.2, 2.4
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(text=st.just(""))
def test_empty_string_is_rejected(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    An empty string must be rejected with HTTP 400.

    Validates: Requirements 2.4
    """
    is_valid, status_code, error_message = validate_text_entry(text)
    assert is_valid is False
    assert status_code == 400
    assert error_message != ""


@settings(max_examples=100)
@given(text=whitespace_st)
def test_whitespace_only_is_rejected(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    Any text composed entirely of whitespace must be rejected with HTTP 400.

    Validates: Requirements 2.4
    """
    is_valid, status_code, error_message = validate_text_entry(text)
    assert is_valid is False
    assert status_code == 400
    assert error_message != ""


@settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example])
@given(text=oversized_st)
def test_oversized_text_is_rejected(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    Any text exceeding 5,000 characters must be rejected with HTTP 413.

    Validates: Requirements 2.2
    """
    assert len(text) > MAX_TEXT_LENGTH
    is_valid, status_code, error_message = validate_text_entry(text)
    assert is_valid is False
    assert status_code == 413
    assert error_message != ""


@settings(max_examples=100)
@given(text=valid_text_st)
def test_valid_text_is_accepted(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    Any non-empty, non-whitespace text within 5,000 characters must be accepted
    (is_valid=True, status_code=200).

    Validates: Requirements 2.1, 2.2
    """
    assert len(text) <= MAX_TEXT_LENGTH
    assert text.strip() != ""
    is_valid, status_code, error_message = validate_text_entry(text)
    assert is_valid is True
    assert status_code == 200
    assert error_message == ""


@settings(max_examples=100)
@given(
    text=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=MAX_TEXT_LENGTH,
    ).filter(lambda t: t.strip() != "")
)
def test_valid_text_never_returns_error_message(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    For any accepted text entry, the error message must be empty.

    Validates: Requirements 2.1
    """
    is_valid, status_code, error_message = validate_text_entry(text)
    if is_valid:
        assert error_message == ""


@settings(max_examples=100)
@given(
    text=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=0,
        max_size=MAX_TEXT_LENGTH + 500,
    )
)
def test_rejection_always_has_error_message(text: str):
    """
    # Feature: mindguard-ai, Property 3: Text Entry Validation

    For any rejected text entry, the error message must be non-empty.

    Validates: Requirements 2.1, 2.2, 2.4
    """
    is_valid, status_code, error_message = validate_text_entry(text)
    if not is_valid:
        assert error_message != ""
        assert status_code in (400, 413)

"""
Property-based tests for MindGuard AI Auth Lambda.

# Feature: mindguard-ai, Property 20: Password Complexity Enforcement
# Feature: mindguard-ai, Property 21: Token Expiry
# Feature: mindguard-ai, Property 22: Account Lockout
"""
from __future__ import annotations

import string
from datetime import datetime, timezone, timedelta

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.lambdas.auth_lambda import (
    validate_password,
    is_token_valid,
    is_account_locked,
    compute_lockout_until,
    ACCESS_TOKEN_EXPIRY_MINUTES,
    LOCKOUT_DURATION_MINUTES,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Any printable ASCII string
printable_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Zs")),
    min_size=0,
    max_size=100,
)

# Passwords that are too short (0–11 chars)
short_password_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S")),
    min_size=0,
    max_size=11,
)

# Passwords that are long enough but missing required character classes
_UPPER = string.ascii_uppercase
_LOWER = string.ascii_lowercase
_DIGITS = string.digits
_SPECIAL = string.punctuation

# A valid password base: 12+ chars with all required classes
valid_password_st = st.builds(
    lambda base, upper, digit, special: base + upper + digit + special,
    base=st.text(alphabet=_LOWER, min_size=9, max_size=50),
    upper=st.sampled_from(list(_UPPER)),
    digit=st.sampled_from(list(_DIGITS)),
    special=st.sampled_from(list(_SPECIAL)),
)

# UTC datetimes for token tests
utc_datetime_st = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
).map(lambda dt: dt.replace(tzinfo=timezone.utc))

# Non-empty token strings
token_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=500,
)

# Offsets in minutes: within validity window (0 to <60)
within_window_offset_st = st.integers(min_value=0, max_value=ACCESS_TOKEN_EXPIRY_MINUTES - 1)

# Offsets in minutes: past expiry (60 to 1440)
past_expiry_offset_st = st.integers(
    min_value=ACCESS_TOKEN_EXPIRY_MINUTES,
    max_value=ACCESS_TOKEN_EXPIRY_MINUTES * 24,
)

# Offsets in seconds for lockout tests
within_lockout_offset_st = st.integers(min_value=0, max_value=LOCKOUT_DURATION_MINUTES * 60 - 1)
past_lockout_offset_st = st.integers(
    min_value=LOCKOUT_DURATION_MINUTES * 60,
    max_value=LOCKOUT_DURATION_MINUTES * 60 * 24,
)


# ---------------------------------------------------------------------------
# Property 20: Password Complexity Enforcement
# Validates: Requirements 11.1
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(password=short_password_st)
def test_password_complexity_rejects_short_passwords(password: str):
    """
    # Feature: mindguard-ai, Property 20: Password Complexity Enforcement

    For any password shorter than 12 characters, the system must reject it.

    Validates: Requirements 11.1
    """
    assert validate_password(password) is False, (
        f"Expected short password '{password}' (len={len(password)}) to be rejected"
    )


@settings(max_examples=100)
@given(password=valid_password_st)
def test_password_complexity_accepts_valid_passwords(password: str):
    """
    # Feature: mindguard-ai, Property 20: Password Complexity Enforcement

    For any password that meets all complexity requirements (≥12 chars, uppercase,
    digit, special character), the system must accept it.

    Validates: Requirements 11.1
    """
    assert validate_password(password) is True, (
        f"Expected valid password '{password}' to be accepted"
    )


@settings(max_examples=100)
@given(
    base=st.text(alphabet=_LOWER + _DIGITS + _SPECIAL, min_size=12, max_size=50)
)
def test_password_complexity_rejects_no_uppercase(base: str):
    """
    # Feature: mindguard-ai, Property 20: Password Complexity Enforcement

    For any password lacking an uppercase letter, the system must reject it,
    even if it meets all other requirements.

    Validates: Requirements 11.1
    """
    # Ensure no uppercase letters
    password = base.lower()
    # Make sure it has a digit and special char so only uppercase is missing
    password = password + "1!"
    assume(not any(c.isupper() for c in password))
    assert validate_password(password) is False, (
        f"Expected password without uppercase '{password}' to be rejected"
    )


@settings(max_examples=100)
@given(
    base=st.text(alphabet=_LOWER + _UPPER + _SPECIAL, min_size=12, max_size=50)
)
def test_password_complexity_rejects_no_digit(base: str):
    """
    # Feature: mindguard-ai, Property 20: Password Complexity Enforcement

    For any password lacking a digit, the system must reject it.

    Validates: Requirements 11.1
    """
    assume(not any(c.isdigit() for c in base))
    assume(any(c.isupper() for c in base))
    assume(any(c in _SPECIAL for c in base))
    assert validate_password(base) is False, (
        f"Expected password without digit '{base}' to be rejected"
    )


@settings(max_examples=100)
@given(
    base=st.text(alphabet=_LOWER + _UPPER + _DIGITS, min_size=12, max_size=50)
)
def test_password_complexity_rejects_no_special_char(base: str):
    """
    # Feature: mindguard-ai, Property 20: Password Complexity Enforcement

    For any password lacking a special character, the system must reject it.

    Validates: Requirements 11.1
    """
    assume(not any(c in _SPECIAL for c in base))
    assume(any(c.isupper() for c in base))
    assume(any(c.isdigit() for c in base))
    assert validate_password(base) is False, (
        f"Expected password without special char '{base}' to be rejected"
    )


# ---------------------------------------------------------------------------
# Property 21: Token Expiry
# Validates: Requirements 11.2
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    token=token_st,
    issued_at=utc_datetime_st,
    offset_minutes=within_window_offset_st,
)
def test_token_valid_within_60_minutes(token: str, issued_at: datetime, offset_minutes: int):
    """
    # Feature: mindguard-ai, Property 21: Token Expiry

    For any issued JWT access token, the token must be valid (accepted by the
    auth layer) within the 60-minute validity window from issuance.

    Validates: Requirements 11.2
    """
    current_time = issued_at + timedelta(minutes=offset_minutes)
    assert is_token_valid(token, issued_at, current_time) is True, (
        f"Token should be valid at offset {offset_minutes} minutes (< 60)"
    )


@settings(max_examples=100)
@given(
    token=token_st,
    issued_at=utc_datetime_st,
    offset_minutes=past_expiry_offset_st,
)
def test_token_invalid_after_60_minutes(token: str, issued_at: datetime, offset_minutes: int):
    """
    # Feature: mindguard-ai, Property 21: Token Expiry

    For any issued JWT access token, the token must be invalid (rejected by the
    auth layer) after 60 minutes from issuance.

    Validates: Requirements 11.2
    """
    current_time = issued_at + timedelta(minutes=offset_minutes)
    assert is_token_valid(token, issued_at, current_time) is False, (
        f"Token should be invalid at offset {offset_minutes} minutes (≥ 60)"
    )


@settings(max_examples=100)
@given(issued_at=utc_datetime_st, offset_minutes=within_window_offset_st)
def test_empty_token_always_invalid(issued_at: datetime, offset_minutes: int):
    """
    # Feature: mindguard-ai, Property 21: Token Expiry

    An empty token string must always be rejected regardless of timing.

    Validates: Requirements 11.2
    """
    current_time = issued_at + timedelta(minutes=offset_minutes)
    assert is_token_valid("", issued_at, current_time) is False


# ---------------------------------------------------------------------------
# Property 22: Account Lockout
# Validates: Requirements 11.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    lock_base_time=utc_datetime_st,
    offset_seconds=within_lockout_offset_st,
)
def test_account_locked_within_30_minutes(lock_base_time: datetime, offset_seconds: int):
    """
    # Feature: mindguard-ai, Property 22: Account Lockout

    For any account that has been locked, the system must reject all login
    attempts within the 30-minute lockout window, regardless of password
    correctness.

    Validates: Requirements 11.5
    """
    # Compute the lockout_until timestamp from the lock base time
    account_locked_until = compute_lockout_until(lock_base_time)

    # current_time is within the lockout window
    current_time = lock_base_time + timedelta(seconds=offset_seconds)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    assert is_account_locked(account_locked_until, current_time) is True, (
        f"Account should be locked at offset {offset_seconds}s "
        f"(lock_until={account_locked_until}, current={current_time.isoformat()})"
    )


@settings(max_examples=100)
@given(
    lock_base_time=utc_datetime_st,
    offset_seconds=past_lockout_offset_st,
)
def test_account_unlocked_after_30_minutes(lock_base_time: datetime, offset_seconds: int):
    """
    # Feature: mindguard-ai, Property 22: Account Lockout

    For any account that has been locked, the system must allow login attempts
    after the 30-minute lockout window has expired.

    Validates: Requirements 11.5
    """
    account_locked_until = compute_lockout_until(lock_base_time)

    # current_time is past the lockout window
    current_time = lock_base_time + timedelta(seconds=offset_seconds)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    assert is_account_locked(account_locked_until, current_time) is False, (
        f"Account should be unlocked at offset {offset_seconds}s "
        f"(lock_until={account_locked_until}, current={current_time.isoformat()})"
    )


@settings(max_examples=100)
@given(current_time=utc_datetime_st)
def test_no_lockout_when_none(current_time: datetime):
    """
    # Feature: mindguard-ai, Property 22: Account Lockout

    When account_locked_until is None, the account must never be considered locked.

    Validates: Requirements 11.5
    """
    assert is_account_locked(None, current_time) is False

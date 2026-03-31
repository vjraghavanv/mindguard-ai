"""
Property-based tests for MindGuard AI Account Settings Lambda.

# Feature: mindguard-ai, Property 18: Session Re-Authentication

For any user session that has been inactive for 15 or more minutes, any attempt
to access journal entries or reports must require re-authentication before succeeding.

Validates: Requirements 9.5
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.lambdas.account_settings_lambda import (
    is_session_expired,
    SESSION_TIMEOUT_MINUTES,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# UTC datetimes for session tests
utc_datetime_st = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
).map(lambda dt: dt.replace(tzinfo=timezone.utc))

# Inactivity offsets in seconds: >= 15 minutes (session expired)
expired_offset_st = st.integers(
    min_value=SESSION_TIMEOUT_MINUTES * 60,
    max_value=SESSION_TIMEOUT_MINUTES * 60 * 48,  # up to 48 hours
)

# Inactivity offsets in seconds: < 15 minutes (session still active)
active_offset_st = st.integers(
    min_value=0,
    max_value=SESSION_TIMEOUT_MINUTES * 60 - 1,
)


# ---------------------------------------------------------------------------
# Property 18: Session Re-Authentication
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(
    last_activity=utc_datetime_st,
    inactive_seconds=expired_offset_st,
)
def test_session_expired_after_15_minutes(
    last_activity: datetime, inactive_seconds: int
):
    """
    # Feature: mindguard-ai, Property 18: Session Re-Authentication

    For any user session that has been inactive for 15 or more minutes,
    is_session_expired must return True, indicating re-authentication is required
    before granting access to journal entries or reports.

    Validates: Requirements 9.5
    """
    current_time = last_activity + timedelta(seconds=inactive_seconds)
    result = is_session_expired(last_activity, current_time)
    assert result is True, (
        f"Session inactive for {inactive_seconds}s (>= {SESSION_TIMEOUT_MINUTES * 60}s) "
        f"must be expired. last_activity={last_activity.isoformat()}, "
        f"current_time={current_time.isoformat()}"
    )


@settings(max_examples=100)
@given(
    last_activity=utc_datetime_st,
    inactive_seconds=active_offset_st,
)
def test_session_active_within_15_minutes(
    last_activity: datetime, inactive_seconds: int
):
    """
    # Feature: mindguard-ai, Property 18: Session Re-Authentication

    For any user session that has been inactive for less than 15 minutes,
    is_session_expired must return False, meaning no re-authentication is needed.

    Validates: Requirements 9.5
    """
    current_time = last_activity + timedelta(seconds=inactive_seconds)
    result = is_session_expired(last_activity, current_time)
    assert result is False, (
        f"Session inactive for {inactive_seconds}s (< {SESSION_TIMEOUT_MINUTES * 60}s) "
        f"must NOT be expired. last_activity={last_activity.isoformat()}, "
        f"current_time={current_time.isoformat()}"
    )


@settings(max_examples=100)
@given(last_activity=utc_datetime_st)
def test_session_expired_at_exactly_15_minutes(last_activity: datetime):
    """
    # Feature: mindguard-ai, Property 18: Session Re-Authentication

    At exactly 15 minutes of inactivity, the session must be considered expired
    (boundary condition: >= 15 minutes triggers re-authentication).

    Validates: Requirements 9.5
    """
    current_time = last_activity + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    result = is_session_expired(last_activity, current_time)
    assert result is True, (
        f"Session inactive for exactly {SESSION_TIMEOUT_MINUTES} minutes "
        f"must be expired (boundary). last_activity={last_activity.isoformat()}"
    )


@settings(max_examples=100)
@given(
    last_activity=utc_datetime_st,
    inactive_seconds=expired_offset_st,
    custom_timeout=st.integers(min_value=1, max_value=60),
)
def test_session_expired_respects_custom_timeout(
    last_activity: datetime, inactive_seconds: int, custom_timeout: int
):
    """
    # Feature: mindguard-ai, Property 18: Session Re-Authentication

    For any custom timeout_minutes value, is_session_expired must correctly
    identify sessions inactive for >= timeout_minutes as expired.

    Validates: Requirements 9.5
    """
    # Use inactive_seconds relative to the custom timeout
    actual_inactive = custom_timeout * 60 + (inactive_seconds % (custom_timeout * 60 + 1))
    current_time = last_activity + timedelta(seconds=actual_inactive)
    result = is_session_expired(last_activity, current_time, timeout_minutes=custom_timeout)
    assert result is True, (
        f"Session inactive for {actual_inactive}s with timeout={custom_timeout}min "
        f"must be expired."
    )


@settings(max_examples=100)
@given(
    last_activity=utc_datetime_st,
    inactive_seconds=active_offset_st,
    custom_timeout=st.integers(min_value=1, max_value=60),
)
def test_session_active_respects_custom_timeout(
    last_activity: datetime, inactive_seconds: int, custom_timeout: int
):
    """
    # Feature: mindguard-ai, Property 18: Session Re-Authentication

    For any custom timeout_minutes value, is_session_expired must correctly
    identify sessions inactive for < timeout_minutes as still active.

    Validates: Requirements 9.5
    """
    # Clamp inactive_seconds to be strictly less than custom_timeout * 60
    actual_inactive = inactive_seconds % (custom_timeout * 60)
    assume(actual_inactive < custom_timeout * 60)
    current_time = last_activity + timedelta(seconds=actual_inactive)
    result = is_session_expired(last_activity, current_time, timeout_minutes=custom_timeout)
    assert result is False, (
        f"Session inactive for {actual_inactive}s with timeout={custom_timeout}min "
        f"must NOT be expired."
    )

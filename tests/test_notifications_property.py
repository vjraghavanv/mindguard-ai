"""
Property-based tests for MindGuard AI Notification Service.

# Feature: mindguard-ai, Property 7: Burnout Alert Threshold
# Feature: mindguard-ai, Property 10: Notification Channel Routing
# Feature: mindguard-ai, Property 11: Notification Gating
# Feature: mindguard-ai, Property 13: Escalation and Event Recording
# Feature: mindguard-ai, Property 14: Crisis Helpline Presentation
# Feature: mindguard-ai, Property 15: Escalation Cancellation Window

Validates: Requirements 4.4, 5.4, 6.4, 6.5, 7.1, 7.2, 7.4, 7.6
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.utils.notifications import (
    should_send_burnout_alert,
    should_send_escalation,
    should_show_crisis_helpline,
    is_notification_gated,
    can_cancel_escalation,
    route_notification_channel,
    BURNOUT_ALERT_THRESHOLD,
    CRISIS_HELPLINE_THRESHOLD,
    ESCALATION_CANCEL_WINDOW_SECONDS,
    VALID_CHANNELS,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_burnout_score_st = st.integers(min_value=0, max_value=100)
_escalation_threshold_st = st.integers(min_value=1, max_value=100)
_channel_st = st.sampled_from(["in_app", "push", "sms"])
_notification_type_st = st.sampled_from(["nudge", "alert", "escalation", "report"])
_user_id_st = st.uuids().map(str)


def _notification_prefs_st(
    enabled: bool | None = None,
    channel: str | None = None,
    snooze_until: str | None = None,
):
    """Strategy for a notification_prefs dict."""
    enabled_st = st.just(enabled) if enabled is not None else st.booleans()
    channel_st = st.just(channel) if channel is not None else _channel_st
    snooze_st = st.just(snooze_until)
    return st.fixed_dictionaries({
        "enabled": enabled_st,
        "channel": channel_st,
        "snooze_until": snooze_st,
    })


# ---------------------------------------------------------------------------
# Property 7: Burnout Alert Threshold
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(score=st.integers(min_value=71, max_value=100))
def test_burnout_alert_sent_when_score_above_70(score: int):
    """
    # Feature: mindguard-ai, Property 7: Burnout Alert Threshold

    For any user whose computed Burnout_Score exceeds 70, the Notification_Service
    must send a burnout risk alert.

    Validates: Requirements 4.4
    """
    assert should_send_burnout_alert(score) is True, (
        f"Expected alert for score={score} (> 70)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(score=st.integers(min_value=0, max_value=70))
def test_burnout_alert_not_sent_when_score_at_or_below_70(score: int):
    """
    # Feature: mindguard-ai, Property 7: Burnout Alert Threshold

    For any score ≤ 70, no burnout alert should be sent.

    Validates: Requirements 4.4
    """
    assert should_send_burnout_alert(score) is False, (
        f"Expected no alert for score={score} (≤ 70)"
    )


# ---------------------------------------------------------------------------
# Property 10: Notification Channel Routing
# Validates: Requirements 5.4, 6.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(channel=_channel_st)
def test_notification_routed_to_configured_channel(channel: str):
    """
    # Feature: mindguard-ai, Property 10: Notification Channel Routing

    For any user with a configured notification channel preference, all outbound
    notifications must be delivered exclusively through that channel.

    Validates: Requirements 5.4, 6.3
    """
    prefs = {"channel": channel, "enabled": True, "snooze_until": None}
    result = route_notification_channel(prefs)
    assert result == channel, (
        f"Expected channel={channel}, got {result}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(channel=_channel_st)
def test_notification_channel_is_always_valid(channel: str):
    """
    # Feature: mindguard-ai, Property 10: Notification Channel Routing

    The routed channel must always be one of the valid channels.

    Validates: Requirements 5.4
    """
    prefs = {"channel": channel, "enabled": True, "snooze_until": None}
    result = route_notification_channel(prefs)
    assert result in VALID_CHANNELS, (
        f"Routed channel '{result}' is not a valid channel"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(invalid_channel=st.text(min_size=1).filter(lambda c: c not in VALID_CHANNELS))
def test_invalid_channel_falls_back_to_in_app(invalid_channel: str):
    """
    # Feature: mindguard-ai, Property 10: Notification Channel Routing

    An invalid channel preference must fall back to 'in_app'.

    Validates: Requirements 5.4
    """
    prefs = {"channel": invalid_channel, "enabled": True, "snooze_until": None}
    result = route_notification_channel(prefs)
    assert result == "in_app", (
        f"Expected fallback to 'in_app' for invalid channel '{invalid_channel}', got '{result}'"
    )


# ---------------------------------------------------------------------------
# Property 11: Notification Gating
# Validates: Requirements 6.4, 6.5
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    notification_type=_notification_type_st,
    channel=_channel_st,
)
def test_disabled_notifications_gate_all_types(
    notification_type: str,
    channel: str,
):
    """
    # Feature: mindguard-ai, Property 11: Notification Gating

    For any user who has disabled all notifications, the Notification_Service
    must not send any nudges or alerts.

    Validates: Requirements 6.5
    """
    prefs = {"enabled": False, "channel": channel, "snooze_until": None}
    current_time = datetime.now(timezone.utc)
    result = is_notification_gated(prefs, notification_type, current_time)
    assert result is True, (
        f"Expected gating for disabled notifications, type={notification_type}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    seconds_remaining=st.integers(min_value=1, max_value=24 * 3600),
    channel=_channel_st,
)
def test_active_snooze_gates_nudges(seconds_remaining: int, channel: str):
    """
    # Feature: mindguard-ai, Property 11: Notification Gating

    For any user with an active snooze, no nudge must be sent until the snooze
    period expires.

    Validates: Requirements 6.4
    """
    current_time = datetime.now(timezone.utc)
    snooze_until = (current_time + timedelta(seconds=seconds_remaining)).isoformat()
    prefs = {"enabled": True, "channel": channel, "snooze_until": snooze_until}
    result = is_notification_gated(prefs, "nudge", current_time)
    assert result is True, (
        f"Expected nudge gated during active snooze ({seconds_remaining}s remaining)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    seconds_elapsed=st.integers(min_value=1, max_value=24 * 3600),
    channel=_channel_st,
)
def test_expired_snooze_does_not_gate_nudges(seconds_elapsed: int, channel: str):
    """
    # Feature: mindguard-ai, Property 11: Notification Gating

    After the snooze period expires, nudges must be allowed through.

    Validates: Requirements 6.4
    """
    current_time = datetime.now(timezone.utc)
    snooze_until = (current_time - timedelta(seconds=seconds_elapsed)).isoformat()
    prefs = {"enabled": True, "channel": channel, "snooze_until": snooze_until}
    result = is_notification_gated(prefs, "nudge", current_time)
    assert result is False, (
        f"Expected nudge allowed after snooze expired ({seconds_elapsed}s ago)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    seconds_remaining=st.integers(min_value=1, max_value=24 * 3600),
    notification_type=st.sampled_from(["alert", "escalation", "report"]),
    channel=_channel_st,
)
def test_snooze_does_not_gate_non_nudge_notifications(
    seconds_remaining: int,
    notification_type: str,
    channel: str,
):
    """
    # Feature: mindguard-ai, Property 11: Notification Gating

    Snooze only suppresses nudge-type notifications; alerts and escalations
    must still be delivered during a snooze period.

    Validates: Requirements 6.4
    """
    current_time = datetime.now(timezone.utc)
    snooze_until = (current_time + timedelta(seconds=seconds_remaining)).isoformat()
    prefs = {"enabled": True, "channel": channel, "snooze_until": snooze_until}
    result = is_notification_gated(prefs, notification_type, current_time)
    assert result is False, (
        f"Expected {notification_type} NOT gated during snooze"
    )


# ---------------------------------------------------------------------------
# Property 13: Escalation and Event Recording
# Validates: Requirements 7.1, 7.4
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    burnout_score=st.integers(min_value=0, max_value=100),
    escalation_threshold=_escalation_threshold_st,
)
def test_escalation_sent_iff_score_exceeds_threshold(
    burnout_score: int,
    escalation_threshold: int,
):
    """
    # Feature: mindguard-ai, Property 13: Escalation and Event Recording

    For any user whose Burnout_Score exceeds their configured Escalation_Threshold,
    the Notification_Service must send an escalation alert.
    For scores at or below the threshold, no escalation should be triggered.

    Validates: Requirements 7.1
    """
    result = should_send_escalation(burnout_score, escalation_threshold)
    expected = burnout_score > escalation_threshold
    assert result == expected, (
        f"should_send_escalation({burnout_score}, {escalation_threshold}) "
        f"returned {result}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Property 14: Crisis Helpline Presentation
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(score=st.integers(min_value=86, max_value=100))
def test_crisis_helpline_shown_when_score_above_85(score: int):
    """
    # Feature: mindguard-ai, Property 14: Crisis Helpline Presentation

    For any user with a Burnout_Score exceeding 85, the system must present
    a crisis helpline option in the UI response.

    Validates: Requirements 7.2
    """
    assert should_show_crisis_helpline(score) is True, (
        f"Expected crisis helpline for score={score} (> 85)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(score=st.integers(min_value=0, max_value=85))
def test_crisis_helpline_not_shown_when_score_at_or_below_85(score: int):
    """
    # Feature: mindguard-ai, Property 14: Crisis Helpline Presentation

    For any score ≤ 85, the crisis helpline must NOT be shown.

    Validates: Requirements 7.2
    """
    assert should_show_crisis_helpline(score) is False, (
        f"Expected no crisis helpline for score={score} (≤ 85)"
    )


# ---------------------------------------------------------------------------
# Property 15: Escalation Cancellation Window
# Validates: Requirements 7.6
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(elapsed_seconds=st.integers(min_value=0, max_value=59))
def test_escalation_cancellable_within_60_seconds(elapsed_seconds: int):
    """
    # Feature: mindguard-ai, Property 15: Escalation Cancellation Window

    For any triggered escalation alert, the user must be able to cancel it
    within 60 seconds.

    Validates: Requirements 7.6
    """
    triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    current_time = triggered_at + timedelta(seconds=elapsed_seconds)
    result = can_cancel_escalation(triggered_at, current_time)
    assert result is True, (
        f"Expected cancellation allowed at {elapsed_seconds}s (< 60s)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(elapsed_seconds=st.integers(min_value=60, max_value=3600))
def test_escalation_not_cancellable_after_60_seconds(elapsed_seconds: int):
    """
    # Feature: mindguard-ai, Property 15: Escalation Cancellation Window

    After 60 seconds, cancellation must not be possible.

    Validates: Requirements 7.6
    """
    triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    current_time = triggered_at + timedelta(seconds=elapsed_seconds)
    result = can_cancel_escalation(triggered_at, current_time)
    assert result is False, (
        f"Expected cancellation denied at {elapsed_seconds}s (≥ 60s)"
    )

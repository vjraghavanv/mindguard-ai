"""
Property-based tests for MindGuard AI Nudge Lambda.

# Feature: mindguard-ai, Property 12: Burnout Trend Alert

For any user whose Burnout_Score increases by 15 or more points within a 7-day
period, the Notification_Service must send a trend alert.

Validates: Requirements 6.2
"""
from __future__ import annotations

import os

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.lambdas.nudge_lambda import should_send_trend_alert, TREND_ALERT_THRESHOLD

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_score_st = st.integers(min_value=0, max_value=100)
_scores_list_st = st.lists(_score_st, min_size=2, max_size=7)


# ---------------------------------------------------------------------------
# Property 12: Burnout Trend Alert
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    min_score=st.integers(min_value=0, max_value=85),
    delta=st.integers(min_value=15, max_value=100),
    extra_scores=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=5),
)
def test_trend_alert_sent_when_increase_gte_15(
    min_score: int,
    delta: int,
    extra_scores: list[int],
):
    """
    # Feature: mindguard-ai, Property 12: Burnout Trend Alert

    For any user whose Burnout_Score increases by 15 or more points within a
    7-day period, the Notification_Service must send a trend alert.

    Validates: Requirements 6.2
    """
    max_score = min(min_score + delta, 100)
    scores = [min_score, max_score] + extra_scores
    result = should_send_trend_alert(scores)
    assert result is True, (
        f"Expected trend alert for scores={scores} "
        f"(max={max(scores)}, min={min(scores)}, diff={max(scores) - min(scores)} >= 15)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    base_score=st.integers(min_value=0, max_value=100),
    delta=st.integers(min_value=0, max_value=14),
    extra_scores=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=5),
)
def test_trend_alert_not_sent_when_increase_lt_15(
    base_score: int,
    delta: int,
    extra_scores: list[int],
):
    """
    # Feature: mindguard-ai, Property 12: Burnout Trend Alert

    When the Burnout_Score increase is less than 15 points within the 7-day
    window, no trend alert should be sent.

    Validates: Requirements 6.2
    """
    # Construct a list where all scores are within [base_score, base_score + delta]
    # so max - min < 15
    scores = [base_score, min(base_score + delta, 100)]
    # Clamp extra scores to the same narrow band
    clamped_extras = [
        max(base_score, min(base_score + delta, s)) for s in extra_scores
    ]
    all_scores = scores + clamped_extras

    actual_diff = max(all_scores) - min(all_scores)
    if actual_diff >= TREND_ALERT_THRESHOLD:
        # Skip this example — clamping may not be perfect for edge cases
        return

    result = should_send_trend_alert(all_scores)
    assert result is False, (
        f"Expected no trend alert for scores={all_scores} "
        f"(diff={actual_diff} < 15)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scores=st.lists(_score_st, min_size=2, max_size=7))
def test_trend_alert_consistent_with_max_minus_min(scores: list[int]):
    """
    # Feature: mindguard-ai, Property 12: Burnout Trend Alert

    The trend alert decision must be consistent with max(scores) - min(scores) >= 15.
    This is a metamorphic invariant: the function must agree with the threshold formula.

    Validates: Requirements 6.2
    """
    expected = (max(scores) - min(scores)) >= TREND_ALERT_THRESHOLD
    result = should_send_trend_alert(scores)
    assert result == expected, (
        f"should_send_trend_alert({scores}) returned {result}, "
        f"expected {expected} (diff={max(scores) - min(scores)})"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(score=_score_st)
def test_trend_alert_false_for_single_score(score: int):
    """
    # Feature: mindguard-ai, Property 12: Burnout Trend Alert

    A single score cannot represent a trend — no alert should be sent.

    Validates: Requirements 6.2
    """
    assert should_send_trend_alert([score]) is False, (
        f"Expected no trend alert for single score [{score}]"
    )


def test_trend_alert_false_for_empty_list():
    """
    # Feature: mindguard-ai, Property 12: Burnout Trend Alert

    An empty score list cannot represent a trend — no alert should be sent.

    Validates: Requirements 6.2
    """
    assert should_send_trend_alert([]) is False


def test_trend_alert_threshold_constant_is_15():
    """The TREND_ALERT_THRESHOLD constant must be 15."""
    assert TREND_ALERT_THRESHOLD == 15

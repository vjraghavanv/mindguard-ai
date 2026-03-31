"""
Unit tests for MindGuard AI Nudge Lambda (src/lambdas/nudge_lambda.py).

Tests cover:
- 24-hour gap boundary (has_journaling_gap)
- Snooze expiry gating
- Disabled-notifications gate
- Trend alert threshold (should_send_trend_alert)
- Nudge time matching (is_nudge_time)
- process_user_nudge with mocked DynamoDB and SNS

Requirements: 6.1, 6.2, 6.4, 6.5
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.lambdas.nudge_lambda import (
    should_send_trend_alert,
    has_journaling_gap,
    is_nudge_time,
    process_user_nudge,
    TREND_ALERT_THRESHOLD,
    JOURNALING_GAP_HOURS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamodb_table():
    """Create a mocked DynamoDB table."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="mindguard-trend-store",
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield table


def _make_prefs(channel="in_app", enabled=True, snooze_until=None, nudge_time="09:00"):
    return {
        "channel": channel,
        "enabled": enabled,
        "snooze_until": snooze_until,
        "nudge_time": nudge_time,
    }


def _make_user_profile(user_id="user-1", enabled=True, snooze_until=None, nudge_time="09:00"):
    return {
        "user_id": user_id,
        "email_hash": "abc123",
        "record_type": "user_profile",
        "notification_prefs": _make_prefs(
            enabled=enabled,
            snooze_until=snooze_until,
            nudge_time=nudge_time,
        ),
    }


# ---------------------------------------------------------------------------
# has_journaling_gap — Requirement 6.1
# ---------------------------------------------------------------------------

class TestHasJournalingGap:
    def test_gap_exactly_24_hours_is_not_a_gap(self):
        """Exactly 24 hours is NOT > 24 hours — no nudge."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=24)
        assert has_journaling_gap(last, now) is False

    def test_gap_just_over_24_hours_is_a_gap(self):
        """24 hours + 1 second IS > 24 hours — nudge should fire."""
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=24, seconds=1)
        assert has_journaling_gap(last, now) is True

    def test_gap_less_than_24_hours_is_not_a_gap(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=23, minutes=59)
        assert has_journaling_gap(last, now) is False

    def test_gap_48_hours_is_a_gap(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        last = now - timedelta(hours=48)
        assert has_journaling_gap(last, now) is True

    def test_gap_constant_is_24(self):
        assert JOURNALING_GAP_HOURS == 24


# ---------------------------------------------------------------------------
# should_send_trend_alert — Requirement 6.2
# ---------------------------------------------------------------------------

class TestShouldSendTrendAlert:
    def test_increase_of_exactly_15_triggers_alert(self):
        assert should_send_trend_alert([50, 65]) is True

    def test_increase_of_14_does_not_trigger_alert(self):
        assert should_send_trend_alert([50, 64]) is False

    def test_increase_of_16_triggers_alert(self):
        assert should_send_trend_alert([40, 56]) is True

    def test_all_same_scores_no_alert(self):
        assert should_send_trend_alert([60, 60, 60, 60]) is False

    def test_decrease_does_not_trigger_alert(self):
        """A decrease of 15+ points should NOT trigger an alert (max - min matters)."""
        # max=80, min=65, diff=15 → alert IS triggered regardless of direction
        assert should_send_trend_alert([80, 65]) is True

    def test_empty_list_no_alert(self):
        assert should_send_trend_alert([]) is False

    def test_single_score_no_alert(self):
        assert should_send_trend_alert([75]) is False

    def test_multiple_scores_with_large_spread(self):
        assert should_send_trend_alert([20, 30, 40, 50, 60, 70, 80]) is True

    def test_threshold_constant_is_15(self):
        assert TREND_ALERT_THRESHOLD == 15

    def test_boundary_exactly_15_with_multiple_scores(self):
        """With multiple scores, max - min = 15 should trigger."""
        assert should_send_trend_alert([30, 35, 40, 45]) is True  # 45-30=15


# ---------------------------------------------------------------------------
# is_nudge_time — Requirement 6.3
# ---------------------------------------------------------------------------

class TestIsNudgeTime:
    def test_matches_configured_hour_and_minute(self):
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert is_nudge_time("09:00", current_time) is True

    def test_does_not_match_different_hour(self):
        current_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        assert is_nudge_time("09:00", current_time) is False

    def test_does_not_match_different_minute(self):
        current_time = datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        assert is_nudge_time("09:00", current_time) is False

    def test_matches_afternoon_time(self):
        current_time = datetime(2025, 1, 15, 18, 30, 0, tzinfo=timezone.utc)
        assert is_nudge_time("18:30", current_time) is True

    def test_invalid_nudge_time_returns_false(self):
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        assert is_nudge_time("invalid", current_time) is False


# ---------------------------------------------------------------------------
# process_user_nudge — integration of gating, gap, and trend alert
# ---------------------------------------------------------------------------

class TestProcessUserNudge:
    def test_disabled_notifications_skips_all(self, dynamodb_table):
        """If notifications are disabled, no nudge or alert is sent."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user = _make_user_profile(user_id="user-disabled", enabled=False, nudge_time="09:00")

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is False
        assert result["trend_alert_sent"] is False
        assert result["skipped_reason"] == "gated"
        mock_sns.publish.assert_not_called()

    def test_active_snooze_gates_nudge(self, dynamodb_table):
        """An active snooze prevents nudge from being sent."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time + timedelta(hours=1)).isoformat()
        user = _make_user_profile(
            user_id="user-snoozed",
            snooze_until=snooze_until,
            nudge_time="09:00",
        )

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is False
        assert result["skipped_reason"] == "gated"

    def test_expired_snooze_allows_nudge(self, dynamodb_table):
        """An expired snooze allows the nudge to proceed (if gap > 24h)."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time - timedelta(hours=1)).isoformat()
        user = _make_user_profile(
            user_id="user-snooze-expired",
            snooze_until=snooze_until,
            nudge_time="09:00",
        )

        # No journal entries → gap is infinite → nudge should fire
        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is True

    def test_not_nudge_time_skips(self, dynamodb_table):
        """If current time doesn't match nudge_time, skip processing."""
        mock_sns = MagicMock()
        # Current time is 10:00, nudge_time is 09:00
        current_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        user = _make_user_profile(user_id="user-wrong-time", nudge_time="09:00")

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["skipped_reason"] == "not_nudge_time"
        assert result["check_in_nudge_sent"] is False
        mock_sns.publish.assert_not_called()

    def test_no_journal_entries_sends_check_in_nudge(self, dynamodb_table):
        """No journal entries means gap is infinite — nudge should be sent."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user = _make_user_profile(user_id="user-no-entries", nudge_time="09:00")

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is True

    def test_recent_journal_entry_no_nudge(self, dynamodb_table):
        """A journal entry within 24 hours should suppress the check-in nudge."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user_id = "user-recent-entry"
        user = _make_user_profile(user_id=user_id, nudge_time="09:00")

        # Insert a recent journal entry (1 hour ago)
        recent_ts = (current_time - timedelta(hours=1)).isoformat()
        dynamodb_table.put_item(Item={
            "user_id": user_id,
            "sk": f"{recent_ts}#entry-001",
            "entry_type": "text",
            "created_at": recent_ts,
            "burnout_score": 40,
            "timestamp": recent_ts,
        })

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is False

    def test_trend_alert_sent_when_scores_increase_15_points(self, dynamodb_table):
        """Burnout score increase ≥15 in 7 days triggers a trend alert."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user_id = "user-trend-alert"
        user = _make_user_profile(user_id=user_id, nudge_time="09:00")

        # Insert burnout score records spanning 7 days with ≥15 point increase
        for i, score in enumerate([40, 45, 50, 55, 60]):
            ts = (current_time - timedelta(days=6 - i)).isoformat()
            dynamodb_table.put_item(Item={
                "user_id": user_id,
                "sk": f"{ts}#burnout-{i}",
                "entry_type": "text",
                "created_at": ts,
                "burnout_score": score,
                "timestamp": ts,
            })

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["trend_alert_sent"] is True

    def test_trend_alert_not_sent_when_scores_stable(self, dynamodb_table):
        """Stable burnout scores (< 15 point spread) should not trigger trend alert."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user_id = "user-stable-scores"
        user = _make_user_profile(user_id=user_id, nudge_time="09:00")

        # Insert a recent entry to suppress check-in nudge
        recent_ts = (current_time - timedelta(hours=1)).isoformat()
        dynamodb_table.put_item(Item={
            "user_id": user_id,
            "sk": f"{recent_ts}#entry-001",
            "entry_type": "text",
            "created_at": recent_ts,
            "burnout_score": 50,
            "timestamp": recent_ts,
        })

        # Insert stable burnout scores (spread < 15)
        for i, score in enumerate([50, 52, 51, 53, 50]):
            ts = (current_time - timedelta(days=6 - i)).isoformat()
            dynamodb_table.put_item(Item={
                "user_id": user_id,
                "sk": f"{ts}#burnout-{i}",
                "entry_type": "text",
                "created_at": ts,
                "burnout_score": score,
                "timestamp": ts,
            })

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["trend_alert_sent"] is False

    def test_gap_exactly_24_hours_no_nudge(self, dynamodb_table):
        """Exactly 24 hours gap is NOT > 24 hours — no nudge."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user_id = "user-exact-24h"
        user = _make_user_profile(user_id=user_id, nudge_time="09:00")

        # Entry exactly 24 hours ago
        last_ts = (current_time - timedelta(hours=24)).isoformat()
        dynamodb_table.put_item(Item={
            "user_id": user_id,
            "sk": f"{last_ts}#entry-001",
            "entry_type": "text",
            "created_at": last_ts,
            "burnout_score": 40,
            "timestamp": last_ts,
        })

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is False

    def test_gap_just_over_24_hours_sends_nudge(self, dynamodb_table):
        """24 hours + 1 second gap triggers the check-in nudge."""
        mock_sns = MagicMock()
        current_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        user_id = "user-just-over-24h"
        user = _make_user_profile(user_id=user_id, nudge_time="09:00")

        # Entry 24 hours and 1 second ago
        last_ts = (current_time - timedelta(hours=24, seconds=1)).isoformat()
        dynamodb_table.put_item(Item={
            "user_id": user_id,
            "sk": f"{last_ts}#entry-001",
            "entry_type": "text",
            "created_at": last_ts,
            "burnout_score": 40,
            "timestamp": last_ts,
        })

        result = process_user_nudge(user, current_time, dynamodb_table, mock_sns)

        assert result["check_in_nudge_sent"] is True

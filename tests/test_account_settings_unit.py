"""
Unit tests for MindGuard AI Account Settings Lambda.

Tests cover:
- Snooze durations (1h, 4h, 24h) and invalid durations
- Trusted contact update
- Escalation threshold update
- Data deletion request (marks for deletion with 30-day deadline)
- Session re-authentication boundary (exactly 15 minutes)

Requirements: 6.4, 7.3, 9.4, 9.5, 11.4
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta

import boto3
import pytest
from moto import mock_aws

# Fake AWS credentials for moto
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.lambdas.account_settings_lambda import (
    is_session_expired,
    compute_snooze_until,
    update_notification_prefs,
    update_trusted_contact,
    update_escalation_threshold,
    request_data_deletion,
    SESSION_TIMEOUT_MINUTES,
    VALID_SNOOZE_DURATIONS_HOURS,
    DATA_DELETION_DEADLINE_DAYS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamodb_table():
    """Create a mocked DynamoDB table for tests."""
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


def _seed_user(table, user_id: str) -> dict:
    """Insert a minimal UserProfile into the mocked table."""
    profile = {
        "user_id": user_id,
        "sk": "profile",
        "email_hash": "abc123",
        "notification_prefs": {
            "channel": "in_app",
            "nudge_time": "09:00",
            "enabled": True,
            "snooze_until": None,
        },
        "trusted_contact": {"name": "", "contact": ""},
        "escalation_threshold": 80,
        "account_locked_until": None,
    }
    table.put_item(Item=profile)
    return profile


# ---------------------------------------------------------------------------
# Session re-authentication tests (Requirement 9.5)
# ---------------------------------------------------------------------------

class TestSessionExpiry:
    def test_session_active_at_0_seconds(self):
        now = datetime.now(timezone.utc)
        assert is_session_expired(now, now) is False

    def test_session_active_at_14_minutes_59_seconds(self):
        now = datetime.now(timezone.utc)
        current = now + timedelta(seconds=SESSION_TIMEOUT_MINUTES * 60 - 1)
        assert is_session_expired(now, current) is False

    def test_session_expired_at_exactly_15_minutes(self):
        """Boundary: exactly 15 minutes of inactivity must trigger re-auth."""
        now = datetime.now(timezone.utc)
        current = now + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        assert is_session_expired(now, current) is True

    def test_session_expired_at_16_minutes(self):
        now = datetime.now(timezone.utc)
        current = now + timedelta(minutes=16)
        assert is_session_expired(now, current) is True

    def test_session_expired_at_1_hour(self):
        now = datetime.now(timezone.utc)
        current = now + timedelta(hours=1)
        assert is_session_expired(now, current) is True

    def test_session_timeout_constant_is_15(self):
        assert SESSION_TIMEOUT_MINUTES == 15

    def test_naive_datetimes_treated_as_utc(self):
        """Naive datetimes should be handled without raising errors."""
        last_activity = datetime(2025, 1, 15, 12, 0, 0)  # naive
        current = datetime(2025, 1, 15, 12, 20, 0)       # naive, 20 min later
        assert is_session_expired(last_activity, current) is True

    def test_custom_timeout_respected(self):
        now = datetime.now(timezone.utc)
        current = now + timedelta(minutes=5)
        # With 5-minute timeout, 5 minutes is expired
        assert is_session_expired(now, current, timeout_minutes=5) is True
        # With 10-minute timeout, 5 minutes is still active
        assert is_session_expired(now, current, timeout_minutes=10) is False


# ---------------------------------------------------------------------------
# Snooze duration tests (Requirement 6.4)
# ---------------------------------------------------------------------------

class TestComputeSnoozeUntil:
    def test_snooze_1_hour(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_snooze_until(1, now)
        expected = "2025-01-15T13:00:00Z"
        assert result == expected

    def test_snooze_4_hours(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_snooze_until(4, now)
        expected = "2025-01-15T16:00:00Z"
        assert result == expected

    def test_snooze_24_hours(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_snooze_until(24, now)
        expected = "2025-01-16T12:00:00Z"
        assert result == expected

    def test_invalid_snooze_duration_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError):
            compute_snooze_until(2, now)

    def test_invalid_snooze_duration_8_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError):
            compute_snooze_until(8, now)

    def test_invalid_snooze_duration_0_raises(self):
        now = datetime.now(timezone.utc)
        with pytest.raises(ValueError):
            compute_snooze_until(0, now)

    def test_valid_snooze_durations_constant(self):
        assert VALID_SNOOZE_DURATIONS_HOURS == {1, 4, 24}

    def test_snooze_result_is_iso8601_utc(self):
        now = datetime.now(timezone.utc)
        result = compute_snooze_until(1, now)
        # Must be parseable as ISO-8601
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_snooze_naive_datetime_handled(self):
        """Naive datetime should be treated as UTC without raising."""
        now = datetime(2025, 1, 15, 12, 0, 0)  # naive
        result = compute_snooze_until(4, now)
        assert result == "2025-01-15T16:00:00Z"


# ---------------------------------------------------------------------------
# Notification prefs update tests (Requirement 11.4)
# ---------------------------------------------------------------------------

class TestUpdateNotificationPrefs:
    def test_update_channel(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_notification_prefs(user_id, {"channel": "sms"}, dynamodb_table)

        assert result["updated"] is True
        assert result["notification_prefs"]["channel"] == "sms"

    def test_update_enabled_false(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_notification_prefs(user_id, {"enabled": False}, dynamodb_table)

        assert result["updated"] is True
        assert result["notification_prefs"]["enabled"] is False

    def test_update_snooze_until(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)
        snooze_until = "2025-01-15T13:00:00Z"

        result = update_notification_prefs(
            user_id, {"snooze_until": snooze_until}, dynamodb_table
        )

        assert result["updated"] is True
        assert result["notification_prefs"]["snooze_until"] == snooze_until

    def test_update_nudge_time(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_notification_prefs(user_id, {"nudge_time": "18:00"}, dynamodb_table)

        assert result["updated"] is True
        assert result["notification_prefs"]["nudge_time"] == "18:00"

    def test_update_persists_to_dynamodb(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        update_notification_prefs(user_id, {"channel": "push"}, dynamodb_table)

        # Re-read from DynamoDB
        item = dynamodb_table.get_item(Key={"user_id": user_id, "sk": "profile"}).get("Item")
        assert item["notification_prefs"]["channel"] == "push"

    def test_update_unknown_user_returns_error(self, dynamodb_table):
        result = update_notification_prefs("nonexistent-user", {"channel": "sms"}, dynamodb_table)
        assert "error" in result

    def test_partial_update_preserves_other_prefs(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        # Only update channel; nudge_time should remain "09:00"
        result = update_notification_prefs(user_id, {"channel": "push"}, dynamodb_table)

        assert result["notification_prefs"]["nudge_time"] == "09:00"
        assert result["notification_prefs"]["enabled"] is True


# ---------------------------------------------------------------------------
# Trusted contact update tests (Requirement 7.3)
# ---------------------------------------------------------------------------

class TestUpdateTrustedContact:
    def test_update_trusted_contact(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_trusted_contact(
            user_id,
            {"name": "Alice Smith", "contact": "alice@example.com"},
            dynamodb_table,
        )

        assert result["updated"] is True
        assert result["trusted_contact"]["name"] == "Alice Smith"
        assert result["trusted_contact"]["contact"] == "alice@example.com"

    def test_update_trusted_contact_phone(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_trusted_contact(
            user_id,
            {"name": "Bob", "contact": "+15551234567"},
            dynamodb_table,
        )

        assert result["trusted_contact"]["contact"] == "+15551234567"

    def test_update_trusted_contact_persists(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        update_trusted_contact(
            user_id,
            {"name": "Carol", "contact": "carol@example.com"},
            dynamodb_table,
        )

        item = dynamodb_table.get_item(Key={"user_id": user_id, "sk": "profile"}).get("Item")
        assert item["trusted_contact"]["name"] == "Carol"

    def test_update_trusted_contact_unknown_user(self, dynamodb_table):
        result = update_trusted_contact(
            "nonexistent",
            {"name": "X", "contact": "x@example.com"},
            dynamodb_table,
        )
        assert "error" in result

    def test_update_trusted_contact_empty_clears_contact(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_trusted_contact(user_id, {"name": "", "contact": ""}, dynamodb_table)

        assert result["trusted_contact"]["name"] == ""
        assert result["trusted_contact"]["contact"] == ""


# ---------------------------------------------------------------------------
# Escalation threshold update tests (Requirement 9.4)
# ---------------------------------------------------------------------------

class TestUpdateEscalationThreshold:
    def test_update_threshold_to_90(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_escalation_threshold(user_id, 90, dynamodb_table)

        assert result["updated"] is True
        assert result["escalation_threshold"] == 90

    def test_update_threshold_to_0(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_escalation_threshold(user_id, 0, dynamodb_table)

        assert result["updated"] is True
        assert result["escalation_threshold"] == 0

    def test_update_threshold_to_100(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_escalation_threshold(user_id, 100, dynamodb_table)

        assert result["updated"] is True
        assert result["escalation_threshold"] == 100

    def test_invalid_threshold_above_100(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_escalation_threshold(user_id, 101, dynamodb_table)

        assert "error" in result

    def test_invalid_threshold_negative(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = update_escalation_threshold(user_id, -1, dynamodb_table)

        assert "error" in result

    def test_threshold_persists_to_dynamodb(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        update_escalation_threshold(user_id, 75, dynamodb_table)

        item = dynamodb_table.get_item(Key={"user_id": user_id, "sk": "profile"}).get("Item")
        assert int(item["escalation_threshold"]) == 75

    def test_update_threshold_unknown_user(self, dynamodb_table):
        result = update_escalation_threshold("nonexistent", 80, dynamodb_table)
        assert "error" in result


# ---------------------------------------------------------------------------
# Data deletion request tests (Requirement 9.4)
# ---------------------------------------------------------------------------

class TestRequestDataDeletion:
    def test_deletion_request_marks_profile(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = request_data_deletion(user_id, dynamodb_table)

        assert result["deletion_requested"] is True
        assert result["user_id"] == user_id

    def test_deletion_deadline_is_30_days(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = request_data_deletion(user_id, dynamodb_table)

        requested_at = datetime.fromisoformat(result["requested_at"].replace("Z", "+00:00"))
        deadline = datetime.fromisoformat(result["deletion_deadline"].replace("Z", "+00:00"))
        delta = deadline - requested_at
        assert delta.days == DATA_DELETION_DEADLINE_DAYS

    def test_deletion_deadline_constant_is_30(self):
        assert DATA_DELETION_DEADLINE_DAYS == 30

    def test_deletion_status_set_to_pending(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        request_data_deletion(user_id, dynamodb_table)

        item = dynamodb_table.get_item(Key={"user_id": user_id, "sk": "profile"}).get("Item")
        assert item["deletion_status"] == "pending"

    def test_deletion_timestamps_are_iso8601(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = request_data_deletion(user_id, dynamodb_table)

        # Both timestamps must be parseable ISO-8601
        datetime.fromisoformat(result["requested_at"].replace("Z", "+00:00"))
        datetime.fromisoformat(result["deletion_deadline"].replace("Z", "+00:00"))

    def test_deletion_message_mentions_30_days(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        result = request_data_deletion(user_id, dynamodb_table)

        assert "30" in result["message"]

    def test_deletion_request_unknown_user(self, dynamodb_table):
        result = request_data_deletion("nonexistent", dynamodb_table)
        assert "error" in result

    def test_deletion_persists_to_dynamodb(self, dynamodb_table):
        user_id = str(uuid.uuid4())
        _seed_user(dynamodb_table, user_id)

        request_data_deletion(user_id, dynamodb_table)

        item = dynamodb_table.get_item(Key={"user_id": user_id, "sk": "profile"}).get("Item")
        assert "deletion_requested_at" in item
        assert "deletion_deadline" in item
        assert item["deletion_status"] == "pending"

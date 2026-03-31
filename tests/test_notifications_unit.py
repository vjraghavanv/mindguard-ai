"""
Unit tests for MindGuard AI Notification Service (src/utils/notifications.py).

Tests cover:
- Channel routing for each valid channel
- Snooze boundary (exactly at snooze_until)
- Disabled-notifications gate
- Escalation cancellation at exactly 60 seconds
- Missing Trusted_Contact path (prompt user + show helpline)
- send_notification with mocked SNS
- record_escalation_event with mocked DynamoDB

Requirements: 5.4, 6.4, 6.5, 7.1, 7.2, 7.5, 7.6
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.utils.notifications import (
    send_notification,
    send_burnout_alert,
    send_escalation_alert,
    record_escalation_event,
    cancel_escalation_event,
    build_response_payload,
    route_notification_channel,
    is_notification_gated,
    can_cancel_escalation,
    should_send_burnout_alert,
    should_send_escalation,
    should_show_crisis_helpline,
    CRISIS_HELPLINE_INFO,
    ESCALATION_CANCEL_WINDOW_SECONDS,
    BURNOUT_ALERT_THRESHOLD,
    CRISIS_HELPLINE_THRESHOLD,
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


def _make_prefs(channel="in_app", enabled=True, snooze_until=None):
    return {"channel": channel, "enabled": enabled, "snooze_until": snooze_until}


def _make_trusted_contact(name="Alice", contact="alice@example.com"):
    return {"name": name, "contact": contact}


# ---------------------------------------------------------------------------
# Channel routing tests (Requirement 5.4)
# ---------------------------------------------------------------------------

class TestChannelRouting:
    def test_routes_to_in_app(self):
        prefs = _make_prefs(channel="in_app")
        assert route_notification_channel(prefs) == "in_app"

    def test_routes_to_push(self):
        prefs = _make_prefs(channel="push")
        assert route_notification_channel(prefs) == "push"

    def test_routes_to_sms(self):
        prefs = _make_prefs(channel="sms")
        assert route_notification_channel(prefs) == "sms"

    def test_invalid_channel_falls_back_to_in_app(self):
        prefs = _make_prefs(channel="telegram")
        assert route_notification_channel(prefs) == "in_app"

    def test_missing_channel_defaults_to_in_app(self):
        prefs = {"enabled": True}
        assert route_notification_channel(prefs) == "in_app"

    def test_send_notification_uses_configured_channel(self):
        """send_notification must use the user's preferred channel."""
        mock_sns = MagicMock()
        prefs = _make_prefs(channel="sms")
        result = send_notification("user-1", "Hello", "alert", prefs, sns_client=mock_sns)
        assert result["sent"] is True
        assert result["channel"] == "sms"

    def test_send_notification_channel_in_sns_payload(self):
        """The SNS publish call must include the channel in MessageAttributes."""
        mock_sns = MagicMock()
        prefs = _make_prefs(channel="push")
        send_notification("user-1", "Test", "alert", prefs, sns_client=mock_sns)
        call_kwargs = mock_sns.publish.call_args[1]
        assert call_kwargs["MessageAttributes"]["channel"]["StringValue"] == "push"


# ---------------------------------------------------------------------------
# Notification gating — disabled (Requirement 6.5)
# ---------------------------------------------------------------------------

class TestDisabledNotificationsGate:
    def test_disabled_gates_nudge(self):
        prefs = _make_prefs(enabled=False)
        current_time = datetime.now(timezone.utc)
        assert is_notification_gated(prefs, "nudge", current_time) is True

    def test_disabled_gates_alert(self):
        prefs = _make_prefs(enabled=False)
        current_time = datetime.now(timezone.utc)
        assert is_notification_gated(prefs, "alert", current_time) is True

    def test_disabled_gates_escalation(self):
        prefs = _make_prefs(enabled=False)
        current_time = datetime.now(timezone.utc)
        assert is_notification_gated(prefs, "escalation", current_time) is True

    def test_disabled_gates_report(self):
        prefs = _make_prefs(enabled=False)
        current_time = datetime.now(timezone.utc)
        assert is_notification_gated(prefs, "report", current_time) is True

    def test_send_notification_returns_not_sent_when_disabled(self):
        mock_sns = MagicMock()
        prefs = _make_prefs(enabled=False)
        result = send_notification("user-1", "Hello", "nudge", prefs, sns_client=mock_sns)
        assert result["sent"] is False
        assert result["reason"] == "notifications_disabled"
        mock_sns.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Notification gating — snooze boundary (Requirement 6.4)
# ---------------------------------------------------------------------------

class TestSnoozeBoundary:
    def test_snooze_gates_nudge_when_active(self):
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time + timedelta(hours=1)).isoformat()
        prefs = _make_prefs(snooze_until=snooze_until)
        assert is_notification_gated(prefs, "nudge", current_time) is True

    def test_snooze_does_not_gate_nudge_when_expired(self):
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time - timedelta(seconds=1)).isoformat()
        prefs = _make_prefs(snooze_until=snooze_until)
        assert is_notification_gated(prefs, "nudge", current_time) is False

    def test_snooze_boundary_exactly_at_snooze_until(self):
        """At exactly snooze_until, the snooze has expired — nudge should be allowed."""
        snooze_until_dt = datetime(2025, 1, 15, 13, 0, 0, tzinfo=timezone.utc)
        current_time = snooze_until_dt  # exactly at boundary
        prefs = _make_prefs(snooze_until=snooze_until_dt.isoformat())
        # current_time == snooze_until → NOT in future → not gated
        assert is_notification_gated(prefs, "nudge", current_time) is False

    def test_snooze_does_not_gate_alert(self):
        """Alerts must not be suppressed by snooze."""
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time + timedelta(hours=4)).isoformat()
        prefs = _make_prefs(snooze_until=snooze_until)
        assert is_notification_gated(prefs, "alert", current_time) is False

    def test_snooze_does_not_gate_escalation(self):
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        snooze_until = (current_time + timedelta(hours=24)).isoformat()
        prefs = _make_prefs(snooze_until=snooze_until)
        assert is_notification_gated(prefs, "escalation", current_time) is False

    def test_send_notification_returns_snoozed_reason(self):
        mock_sns = MagicMock()
        current_time = datetime.now(timezone.utc)
        snooze_until = (current_time + timedelta(hours=1)).isoformat()
        prefs = _make_prefs(snooze_until=snooze_until)
        result = send_notification("user-1", "Check in!", "nudge", prefs, sns_client=mock_sns)
        assert result["sent"] is False
        assert result["reason"] == "snoozed"
        mock_sns.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Burnout alert threshold tests (Requirement 4.4)
# ---------------------------------------------------------------------------

class TestBurnoutAlertThreshold:
    def test_alert_sent_for_score_71(self):
        assert should_send_burnout_alert(71) is True

    def test_alert_sent_for_score_100(self):
        assert should_send_burnout_alert(100) is True

    def test_no_alert_for_score_70(self):
        assert should_send_burnout_alert(70) is False

    def test_no_alert_for_score_0(self):
        assert should_send_burnout_alert(0) is False

    def test_threshold_constant_is_70(self):
        assert BURNOUT_ALERT_THRESHOLD == 70

    def test_send_burnout_alert_publishes_to_sns(self):
        mock_sns = MagicMock()
        prefs = _make_prefs(channel="push")
        result = send_burnout_alert("user-1", 75, prefs, sns_client=mock_sns)
        assert result["alert_sent"] is True
        mock_sns.publish.assert_called_once()

    def test_send_burnout_alert_skipped_for_low_score(self):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        result = send_burnout_alert("user-1", 65, prefs, sns_client=mock_sns)
        assert result["alert_sent"] is False
        assert result["reason"] == "score_below_threshold"
        mock_sns.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Escalation cancellation at exactly 60 seconds (Requirement 7.6)
# ---------------------------------------------------------------------------

class TestEscalationCancellationWindow:
    def test_cancellable_at_0_seconds(self):
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert can_cancel_escalation(triggered_at, triggered_at) is True

    def test_cancellable_at_59_seconds(self):
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = triggered_at + timedelta(seconds=59)
        assert can_cancel_escalation(triggered_at, current_time) is True

    def test_not_cancellable_at_exactly_60_seconds(self):
        """At exactly 60 seconds, cancellation must not be possible."""
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = triggered_at + timedelta(seconds=ESCALATION_CANCEL_WINDOW_SECONDS)
        assert can_cancel_escalation(triggered_at, current_time) is False

    def test_not_cancellable_at_61_seconds(self):
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = triggered_at + timedelta(seconds=61)
        assert can_cancel_escalation(triggered_at, current_time) is False

    def test_cancel_window_constant_is_60(self):
        assert ESCALATION_CANCEL_WINDOW_SECONDS == 60

    def test_cancel_escalation_event_within_window(self, dynamodb_table):
        """cancel_escalation_event succeeds within 60 seconds."""
        user_id = "user-cancel-test"
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = triggered_at + timedelta(seconds=30)

        # First record an event
        event = record_escalation_event(
            user_id=user_id,
            burnout_score=85,
            escalation_threshold=80,
            contact_notified=True,
            dynamodb_table=dynamodb_table,
        )
        sort_key = event["sk"]

        result = cancel_escalation_event(
            user_id=user_id,
            sort_key=sort_key,
            triggered_at=triggered_at,
            current_time=current_time,
            dynamodb_table=dynamodb_table,
        )
        assert result["cancelled"] is True
        assert "cancelled_at" in result

    def test_cancel_escalation_event_after_window(self, dynamodb_table):
        """cancel_escalation_event fails after 60 seconds."""
        user_id = "user-cancel-expired"
        triggered_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        current_time = triggered_at + timedelta(seconds=61)

        event = record_escalation_event(
            user_id=user_id,
            burnout_score=85,
            escalation_threshold=80,
            contact_notified=True,
            dynamodb_table=dynamodb_table,
        )
        sort_key = event["sk"]

        result = cancel_escalation_event(
            user_id=user_id,
            sort_key=sort_key,
            triggered_at=triggered_at,
            current_time=current_time,
            dynamodb_table=dynamodb_table,
        )
        assert result["cancelled"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Missing Trusted_Contact path (Requirement 7.5)
# ---------------------------------------------------------------------------

class TestMissingTrustedContact:
    def test_no_trusted_contact_prompts_user(self, dynamodb_table):
        """When no Trusted_Contact is configured, user is prompted to add one."""
        mock_sns = MagicMock()
        prefs = _make_prefs()
        empty_contact = {"name": "", "contact": ""}

        result = send_escalation_alert(
            user_id="user-no-contact",
            burnout_score=90,
            escalation_threshold=80,
            trusted_contact=empty_contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert result.get("prompt_add_trusted_contact") is True
        assert "crisis_helpline" in result
        assert result["crisis_helpline"]["number"] == "988"
        # SNS escalation should NOT be published (no contact to notify)
        mock_sns.publish.assert_not_called()

    def test_no_trusted_contact_shows_helpline_info(self, dynamodb_table):
        """Crisis helpline info must be present when no Trusted_Contact is set."""
        mock_sns = MagicMock()
        prefs = _make_prefs()
        empty_contact = {"name": "", "contact": ""}

        result = send_escalation_alert(
            user_id="user-no-contact-2",
            burnout_score=88,
            escalation_threshold=80,
            trusted_contact=empty_contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        helpline = result.get("crisis_helpline", {})
        assert helpline.get("number") == "988"
        assert "url" in helpline

    def test_no_trusted_contact_records_escalation_event(self, dynamodb_table):
        """Even without a contact, an EscalationEvent must be recorded."""
        mock_sns = MagicMock()
        prefs = _make_prefs()
        empty_contact = {"name": "", "contact": ""}

        send_escalation_alert(
            user_id="user-no-contact-3",
            burnout_score=90,
            escalation_threshold=80,
            trusted_contact=empty_contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        # Verify the event was stored in DynamoDB
        response = dynamodb_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq("user-no-contact-3")
        )
        items = response.get("Items", [])
        assert len(items) >= 1
        assert items[0]["contact_notified"] is False

    def test_escalation_not_triggered_below_threshold(self, dynamodb_table):
        """No escalation when score is at or below threshold."""
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-low-score",
            burnout_score=80,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert result["escalation_sent"] is False
        assert result["reason"] == "score_below_escalation_threshold"
        mock_sns.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Escalation with Trusted_Contact (Requirements 7.1, 7.4)
# ---------------------------------------------------------------------------

class TestEscalationWithTrustedContact:
    def test_escalation_sent_to_trusted_contact(self, dynamodb_table):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-escalate",
            burnout_score=90,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert result["escalation_sent"] is True
        assert result["contact_notified"] is True
        mock_sns.publish.assert_called_once()

    def test_escalation_event_recorded_in_dynamodb(self, dynamodb_table):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-event-record",
            burnout_score=85,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert "escalation_event" in result
        event = result["escalation_event"]
        assert event["user_id"] == "user-event-record"
        assert event["burnout_score"] == 85
        assert event["escalation_threshold"] == 80
        assert event["contact_notified"] is True
        assert "timestamp" in event

    def test_escalation_event_has_utc_timestamp(self, dynamodb_table):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-ts-check",
            burnout_score=90,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        timestamp_str = result["escalation_event"]["timestamp"]
        # Must be parseable as ISO-8601 UTC
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# Crisis helpline in response payload (Requirement 7.2)
# ---------------------------------------------------------------------------

class TestCrisisHelplineInPayload:
    def test_helpline_in_payload_for_score_above_85(self):
        payload = build_response_payload(
            burnout_score=86,
            sentiment={"sentiment": "NEGATIVE", "sentiment_score": 0.9},
            coping_suggestion="Take a break.",
        )
        assert "crisis_helpline" in payload
        assert payload["crisis_helpline"]["number"] == "988"

    def test_helpline_not_in_payload_for_score_at_85(self):
        payload = build_response_payload(
            burnout_score=85,
            sentiment={"sentiment": "NEGATIVE", "sentiment_score": 0.85},
            coping_suggestion="Breathe deeply.",
        )
        assert "crisis_helpline" not in payload

    def test_helpline_not_in_payload_for_low_score(self):
        payload = build_response_payload(
            burnout_score=50,
            sentiment={"sentiment": "NEUTRAL", "sentiment_score": 0.6},
            coping_suggestion="Keep journaling.",
        )
        assert "crisis_helpline" not in payload

    def test_payload_always_contains_core_fields(self):
        payload = build_response_payload(
            burnout_score=90,
            sentiment={"sentiment": "NEGATIVE", "sentiment_score": 0.9},
            coping_suggestion="Rest.",
        )
        assert "sentiment" in payload
        assert "burnout_score" in payload
        assert "coping_suggestion" in payload

    def test_crisis_helpline_threshold_constant_is_85(self):
        assert CRISIS_HELPLINE_THRESHOLD == 85

    def test_escalation_shows_helpline_when_score_above_85(self, dynamodb_table):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-helpline",
            burnout_score=90,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert result["show_crisis_helpline"] is True
        assert "crisis_helpline" in result

    def test_escalation_no_helpline_when_score_at_85(self, dynamodb_table):
        mock_sns = MagicMock()
        prefs = _make_prefs()
        contact = _make_trusted_contact()

        result = send_escalation_alert(
            user_id="user-no-helpline",
            burnout_score=85,
            escalation_threshold=80,
            trusted_contact=contact,
            notification_prefs=prefs,
            sns_client=mock_sns,
            dynamodb_table=dynamodb_table,
        )

        assert result["show_crisis_helpline"] is False


# ---------------------------------------------------------------------------
# record_escalation_event unit tests (Requirement 7.4)
# ---------------------------------------------------------------------------

class TestRecordEscalationEvent:
    def test_event_stored_with_correct_fields(self, dynamodb_table):
        event = record_escalation_event(
            user_id="user-record-1",
            burnout_score=88,
            escalation_threshold=80,
            contact_notified=True,
            dynamodb_table=dynamodb_table,
        )

        assert event["user_id"] == "user-record-1"
        assert event["burnout_score"] == 88
        assert event["escalation_threshold"] == 80
        assert event["contact_notified"] is True
        assert event["cancelled"] is False
        assert "timestamp" in event
        assert "sk" in event

    def test_event_retrievable_from_dynamodb(self, dynamodb_table):
        event = record_escalation_event(
            user_id="user-record-2",
            burnout_score=75,
            escalation_threshold=70,
            contact_notified=False,
            dynamodb_table=dynamodb_table,
        )

        response = dynamodb_table.get_item(
            Key={"user_id": "user-record-2", "sk": event["sk"]}
        )
        item = response.get("Item")
        assert item is not None
        assert int(item["burnout_score"]) == 75
        assert item["contact_notified"] is False

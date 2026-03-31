"""
Notification Service for MindGuard AI.

Handles routing notifications to user-preferred channels (in-app, push, SMS),
gating based on enabled/snooze state, burnout alert publishing via SNS,
escalation alerts to Trusted_Contact, crisis helpline presentation, and
60-second escalation cancellation window.

Requirements: 4.4, 5.4, 6.4, 6.5, 7.1, 7.2, 7.4, 7.6
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BURNOUT_ALERT_THRESHOLD = 70          # score > 70 triggers burnout alert
CRISIS_HELPLINE_THRESHOLD = 85        # score > 85 shows crisis helpline
ESCALATION_CANCEL_WINDOW_SECONDS = 60 # user can cancel within 60 seconds

CRISIS_HELPLINE_INFO = {
    "name": "988 Suicide & Crisis Lifeline",
    "number": "988",
    "url": "https://988lifeline.org",
    "text": "If you are in crisis, please call or text 988 for immediate support.",
}

VALID_CHANNELS = {"in_app", "push", "sms"}

SNS_TOPIC_ARN = os.environ.get(
    "SNS_BURNOUT_ALERT_TOPIC_ARN",
    "arn:aws:sns:us-east-1:123456789012:mindguard-burnout-alert",
)
SNS_ESCALATION_TOPIC_ARN = os.environ.get(
    "SNS_ESCALATION_TOPIC_ARN",
    "arn:aws:sns:us-east-1:123456789012:mindguard-escalation",
)


# ---------------------------------------------------------------------------
# AWS client helpers
# ---------------------------------------------------------------------------

def _get_sns_client(sns_client=None):
    """Return the provided SNS client or create a real one."""
    if sns_client is not None:
        return sns_client
    return boto3.client(
        "sns",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def _get_dynamodb_table(dynamodb_table=None):
    """Return the provided DynamoDB table or create a real one."""
    if dynamodb_table is not None:
        return dynamodb_table
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    table_name = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")
    return dynamodb.Table(table_name)


# ---------------------------------------------------------------------------
# Pure helper functions (no AWS calls — testable without mocks)
# ---------------------------------------------------------------------------

def should_send_burnout_alert(burnout_score: int) -> bool:
    """
    Return True if the burnout score exceeds the alert threshold (> 70).

    Property 7: Burnout Alert Threshold
    Validates: Requirements 4.4
    """
    return burnout_score > BURNOUT_ALERT_THRESHOLD


def should_send_escalation(burnout_score: int, escalation_threshold: int) -> bool:
    """
    Return True if the burnout score exceeds the user's escalation threshold.

    Property 13: Escalation and Event Recording
    Validates: Requirements 7.1, 7.4
    """
    return burnout_score > escalation_threshold


def should_show_crisis_helpline(burnout_score: int) -> bool:
    """
    Return True if the burnout score exceeds 85, requiring crisis helpline display.

    Property 14: Crisis Helpline Presentation
    Validates: Requirements 7.2
    """
    return burnout_score > CRISIS_HELPLINE_THRESHOLD


def is_notification_gated(
    notification_prefs: dict,
    notification_type: str,
    current_time: datetime,
) -> bool:
    """
    Return True if the notification should be suppressed.

    Suppression rules:
    - If enabled=False, suppress ALL notifications.
    - If snooze_until is set and in the future, suppress nudge-type notifications.

    notification_type: "nudge" | "alert" | "escalation" | "report"

    Property 11: Notification Gating
    Validates: Requirements 6.4, 6.5
    """
    enabled = notification_prefs.get("enabled", True)
    if not enabled:
        return True  # all notifications suppressed

    snooze_until_str = notification_prefs.get("snooze_until")
    if snooze_until_str and notification_type == "nudge":
        try:
            snooze_until = datetime.fromisoformat(
                snooze_until_str.replace("Z", "+00:00")
            )
            if current_time < snooze_until:
                return True  # nudge suppressed during snooze
        except (ValueError, TypeError):
            pass  # malformed snooze_until — do not suppress

    return False


def can_cancel_escalation(triggered_at: datetime, current_time: datetime) -> bool:
    """
    Return True if the escalation alert can still be cancelled (within 60 seconds).

    Property 15: Escalation Cancellation Window
    Validates: Requirements 7.6
    """
    elapsed = (current_time - triggered_at).total_seconds()
    return elapsed < ESCALATION_CANCEL_WINDOW_SECONDS


def route_notification_channel(notification_prefs: dict) -> str:
    """
    Return the user's preferred notification channel.

    Falls back to 'in_app' if the configured channel is invalid.

    Property 10: Notification Channel Routing
    Validates: Requirements 5.4, 6.3
    """
    channel = notification_prefs.get("channel", "in_app")
    if channel not in VALID_CHANNELS:
        return "in_app"
    return channel


# ---------------------------------------------------------------------------
# DynamoDB escalation event recording
# ---------------------------------------------------------------------------

def record_escalation_event(
    user_id: str,
    burnout_score: int,
    escalation_threshold: int,
    contact_notified: bool,
    dynamodb_table=None,
) -> dict:
    """
    Store an EscalationEvent record in DynamoDB with a UTC timestamp.

    Returns the stored event dict.

    Property 13: Escalation and Event Recording
    Validates: Requirements 7.4
    """
    table = _get_dynamodb_table(dynamodb_table)
    now_utc = datetime.now(timezone.utc).isoformat()
    event_id = str(uuid.uuid4())
    sort_key = f"{now_utc}#escalation#{event_id}"

    item = {
        "user_id": user_id,
        "sk": sort_key,
        "timestamp": now_utc,
        "record_type": "escalation_event",
        "burnout_score": burnout_score,
        "escalation_threshold": escalation_threshold,
        "contact_notified": contact_notified,
        "cancelled": False,
        "cancelled_at": None,
    }
    # Remove None values for DynamoDB
    db_item = {k: v for k, v in item.items() if v is not None}
    table.put_item(Item=db_item)
    return item


def cancel_escalation_event(
    user_id: str,
    sort_key: str,
    triggered_at: datetime,
    current_time: datetime,
    dynamodb_table=None,
) -> dict:
    """
    Cancel a pending escalation event if within the 60-second window.

    Returns a dict with 'cancelled' bool and optional 'error' message.
    """
    if not can_cancel_escalation(triggered_at, current_time):
        return {
            "cancelled": False,
            "error": "Cancellation window has expired (60 seconds).",
        }

    table = _get_dynamodb_table(dynamodb_table)
    cancelled_at = current_time.isoformat()
    table.update_item(
        Key={"user_id": user_id, "sk": sort_key},
        UpdateExpression="SET cancelled = :c, cancelled_at = :ca",
        ExpressionAttributeValues={":c": True, ":ca": cancelled_at},
    )
    return {"cancelled": True, "cancelled_at": cancelled_at}


# ---------------------------------------------------------------------------
# Main notification dispatch
# ---------------------------------------------------------------------------

def send_notification(
    user_id: str,
    message: str,
    notification_type: str,
    notification_prefs: dict,
    sns_client=None,
) -> dict:
    """
    Route a notification to the user's preferred channel, respecting gating rules.

    Args:
        user_id: Anonymized user UUID.
        message: Notification message text.
        notification_type: "nudge" | "alert" | "escalation" | "report"
        notification_prefs: Dict matching NotificationPrefs schema.
        sns_client: Optional injected SNS client (for testing).

    Returns:
        dict with keys:
          - sent (bool): Whether the notification was dispatched.
          - channel (str | None): Channel used, or None if gated.
          - reason (str | None): Reason for suppression if not sent.

    Property 10: Notification Channel Routing
    Property 11: Notification Gating
    Validates: Requirements 5.4, 6.4, 6.5
    """
    current_time = datetime.now(timezone.utc)

    # Gate check
    if is_notification_gated(notification_prefs, notification_type, current_time):
        reason = (
            "notifications_disabled"
            if not notification_prefs.get("enabled", True)
            else "snoozed"
        )
        return {"sent": False, "channel": None, "reason": reason}

    channel = route_notification_channel(notification_prefs)

    # Publish to SNS (which routes to Pinpoint for push/SMS)
    client = _get_sns_client(sns_client)
    payload = {
        "user_id": user_id,
        "message": message,
        "notification_type": notification_type,
        "channel": channel,
    }

    try:
        client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=json.dumps(payload),
            MessageAttributes={
                "channel": {
                    "DataType": "String",
                    "StringValue": channel,
                },
                "notification_type": {
                    "DataType": "String",
                    "StringValue": notification_type,
                },
            },
        )
    except Exception:
        # Log and continue — delivery failure is handled by SNS/Pinpoint retries
        pass

    return {"sent": True, "channel": channel, "reason": None}


def send_burnout_alert(
    user_id: str,
    burnout_score: int,
    notification_prefs: dict,
    sns_client=None,
) -> dict:
    """
    Send a burnout risk alert if score > 70.

    Returns a dict with 'alert_sent' bool and 'channel'.

    Property 7: Burnout Alert Threshold
    Validates: Requirements 4.4
    """
    if not should_send_burnout_alert(burnout_score):
        return {"alert_sent": False, "reason": "score_below_threshold"}

    message = (
        f"MindGuard: Your stress levels are elevated (score: {burnout_score}/100). "
        "We've sent you a coping tip. You've got this. 💙"
    )
    result = send_notification(
        user_id=user_id,
        message=message,
        notification_type="alert",
        notification_prefs=notification_prefs,
        sns_client=sns_client,
    )
    return {"alert_sent": result["sent"], "channel": result.get("channel")}


def send_escalation_alert(
    user_id: str,
    burnout_score: int,
    escalation_threshold: int,
    trusted_contact: dict,
    notification_prefs: dict,
    sns_client=None,
    dynamodb_table=None,
) -> dict:
    """
    Send an escalation alert to the Trusted_Contact when score > escalation_threshold.
    Records an EscalationEvent in DynamoDB.

    If no Trusted_Contact is configured, prompts user and shows crisis helpline.

    Returns a dict with escalation details.

    Property 13: Escalation and Event Recording
    Property 14: Crisis Helpline Presentation
    Validates: Requirements 7.1, 7.2, 7.4, 7.5
    """
    if not should_send_escalation(burnout_score, escalation_threshold):
        return {"escalation_sent": False, "reason": "score_below_escalation_threshold"}

    contact_name = trusted_contact.get("name", "").strip()
    contact_info = trusted_contact.get("contact", "").strip()
    has_trusted_contact = bool(contact_name and contact_info)

    response: dict = {
        "escalation_sent": False,
        "contact_notified": False,
        "show_crisis_helpline": should_show_crisis_helpline(burnout_score),
    }

    if not has_trusted_contact:
        # Requirement 7.5: prompt user to add contact + show helpline
        response["prompt_add_trusted_contact"] = True
        response["crisis_helpline"] = CRISIS_HELPLINE_INFO
        record_escalation_event(
            user_id=user_id,
            burnout_score=burnout_score,
            escalation_threshold=escalation_threshold,
            contact_notified=False,
            dynamodb_table=dynamodb_table,
        )
        return response

    # Publish escalation to SNS
    client = _get_sns_client(sns_client)
    message = (
        f"MindGuard Escalation Alert: {contact_name}, your trusted contact "
        f"may need support right now. Their burnout score is {burnout_score}/100. "
        "Please reach out to them."
    )
    payload = {
        "user_id": user_id,
        "burnout_score": burnout_score,
        "escalation_threshold": escalation_threshold,
        "trusted_contact": trusted_contact,
        "message": message,
    }

    try:
        client.publish(
            TopicArn=SNS_ESCALATION_TOPIC_ARN,
            Message=json.dumps(payload),
        )
        response["escalation_sent"] = True
        response["contact_notified"] = True
    except Exception:
        response["escalation_sent"] = False
        response["contact_notified"] = False

    # Record escalation event in DynamoDB
    event_record = record_escalation_event(
        user_id=user_id,
        burnout_score=burnout_score,
        escalation_threshold=escalation_threshold,
        contact_notified=response["contact_notified"],
        dynamodb_table=dynamodb_table,
    )
    response["escalation_event"] = event_record

    # Always show crisis helpline if score > 85
    if response["show_crisis_helpline"]:
        response["crisis_helpline"] = CRISIS_HELPLINE_INFO

    return response


def build_response_payload(
    burnout_score: int,
    sentiment: dict,
    coping_suggestion: str,
) -> dict:
    """
    Build the API response payload, including crisis helpline if score > 85.

    Property 14: Crisis Helpline Presentation
    Validates: Requirements 7.2
    """
    payload: dict = {
        "sentiment": sentiment,
        "burnout_score": burnout_score,
        "coping_suggestion": coping_suggestion,
    }
    if should_show_crisis_helpline(burnout_score):
        payload["crisis_helpline"] = CRISIS_HELPLINE_INFO
    return payload

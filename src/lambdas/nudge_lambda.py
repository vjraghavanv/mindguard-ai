"""
Nudge Lambda for MindGuard AI.

Triggered hourly by EventBridge Scheduler. For each active user:
- Sends a check-in nudge if the last journal entry gap > 24 hours
  (respecting snooze and disabled-notifications flags).
- Sends a trend alert if burnout_score increased ≥15 points within the last 7 days.
- Respects user-configured nudge schedule (nudge_time in UserProfile).

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key

from src.utils.notifications import (
    send_notification,
    is_notification_gated,
)

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")
SNS_TOPIC_ARN = os.environ.get(
    "SNS_NUDGE_TOPIC_ARN",
    "arn:aws:sns:us-east-1:123456789012:mindguard-nudge",
)

JOURNALING_GAP_HOURS = 24          # Requirement 6.1
TREND_ALERT_THRESHOLD = 15         # Requirement 6.2: ≥15-point increase triggers alert


# ---------------------------------------------------------------------------
# Pure helper functions (no AWS calls — testable without mocks)
# ---------------------------------------------------------------------------


def should_send_trend_alert(scores_7_days: list[int]) -> bool:
    """
    Return True if the burnout score increased by ≥15 points within the 7-day window.

    The increase is measured as max(scores) - min(scores) within the window.
    Returns False for empty or single-element lists.

    Property 12: Burnout Trend Alert
    Validates: Requirements 6.2
    """
    if len(scores_7_days) < 2:
        return False
    return (max(scores_7_days) - min(scores_7_days)) >= TREND_ALERT_THRESHOLD


def has_journaling_gap(last_entry_time: datetime, current_time: datetime) -> bool:
    """
    Return True if the gap between last_entry_time and current_time exceeds 24 hours.

    Validates: Requirements 6.1
    """
    gap = current_time - last_entry_time
    return gap > timedelta(hours=JOURNALING_GAP_HOURS)


def is_nudge_time(nudge_time_str: str, current_time: datetime) -> bool:
    """
    Return True if the current UTC hour matches the user's configured nudge_time (HH:MM).

    Validates: Requirements 6.3
    """
    try:
        nudge_hour, nudge_minute = map(int, nudge_time_str.split(":"))
        return (
            current_time.hour == nudge_hour
            and current_time.minute == nudge_minute
        )
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------


def _get_table():
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return dynamodb.Table(TABLE_NAME)


def _get_sns_client():
    return boto3.client(
        "sns",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def get_active_users(table) -> list[dict]:
    """
    Scan DynamoDB for all UserProfile records (record_type = 'user_profile').

    Returns a list of user profile dicts.
    """
    response = table.scan(
        FilterExpression="record_type = :rt",
        ExpressionAttributeValues={":rt": "user_profile"},
    )
    return response.get("Items", [])


def get_last_journal_entry_time(user_id: str, table) -> Optional[datetime]:
    """
    Query DynamoDB for the most recent journal entry for a user.

    Returns the entry's created_at as a datetime, or None if no entries exist.
    """
    response = table.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
        ScanIndexForward=False,  # descending order — most recent first
        Limit=10,
    )
    items = response.get("Items", [])
    for item in items:
        if item.get("record_type") == "journal_entry" or item.get("entry_type") in ("text", "voice"):
            created_at = item.get("created_at") or item.get("timestamp")
            if created_at:
                try:
                    return datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    continue
    return None


def get_burnout_scores_7_days(user_id: str, table, current_time: datetime) -> list[int]:
    """
    Query DynamoDB for burnout score records within the last 7 days for a user.

    Returns a list of burnout_score integers, ordered by time.
    """
    cutoff = (current_time - timedelta(days=7)).isoformat()
    response = table.query(
        KeyConditionExpression=(
            Key("user_id").eq(user_id) & Key("sk").begins_with(cutoff[:10])
        ),
    )
    # Fallback: query all and filter in memory
    response = table.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
    )
    scores = []
    for item in response.get("Items", []):
        if "burnout_score" in item:
            ts_str = item.get("timestamp") or item.get("created_at", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts >= current_time - timedelta(days=7):
                    scores.append(int(item["burnout_score"]))
            except (ValueError, TypeError):
                continue
    return scores


# ---------------------------------------------------------------------------
# Per-user nudge processing
# ---------------------------------------------------------------------------


def process_user_nudge(
    user_profile: dict,
    current_time: datetime,
    table,
    sns_client,
) -> dict:
    """
    Evaluate and send nudges/alerts for a single user.

    Returns a summary dict describing what was sent.
    """
    user_id = user_profile.get("user_id", "")
    notification_prefs = user_profile.get("notification_prefs", {})
    nudge_time_str = notification_prefs.get("nudge_time", "09:00")

    result = {
        "user_id": user_id,
        "check_in_nudge_sent": False,
        "trend_alert_sent": False,
        "skipped_reason": None,
    }

    # Gate: notifications disabled
    if is_notification_gated(notification_prefs, "nudge", current_time):
        result["skipped_reason"] = "gated"
        return result

    # Requirement 6.3: only send at user's configured nudge time (hour match)
    if not is_nudge_time(nudge_time_str, current_time):
        result["skipped_reason"] = "not_nudge_time"
        return result

    # Requirement 6.1: check-in nudge if gap > 24 hours
    last_entry_time = get_last_journal_entry_time(user_id, table)
    if last_entry_time is None or has_journaling_gap(last_entry_time, current_time):
        nudge_result = send_notification(
            user_id=user_id,
            message="How are you feeling today? Take a moment to check in with yourself. 💙",
            notification_type="nudge",
            notification_prefs=notification_prefs,
            sns_client=sns_client,
        )
        result["check_in_nudge_sent"] = nudge_result.get("sent", False)

    # Requirement 6.2: trend alert if burnout score increased ≥15 points in 7 days
    scores = get_burnout_scores_7_days(user_id, table, current_time)
    if should_send_trend_alert(scores):
        alert_result = send_notification(
            user_id=user_id,
            message=(
                "MindGuard: Your burnout score has increased significantly over the past week. "
                "Please take care of yourself. 💙"
            ),
            notification_type="alert",
            notification_prefs=notification_prefs,
            sns_client=sns_client,
        )
        result["trend_alert_sent"] = alert_result.get("sent", False)

    return result


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def handler(event: dict, context) -> dict:
    """
    EventBridge Scheduler handler — runs hourly.

    Queries DynamoDB for active users and processes nudges/alerts for each.
    """
    current_time = datetime.now(timezone.utc)
    table = _get_table()
    sns_client = _get_sns_client()

    users = get_active_users(table)
    results = []

    for user_profile in users:
        try:
            result = process_user_nudge(user_profile, current_time, table, sns_client)
            results.append(result)
        except Exception as exc:
            logger.error(
                "Error processing nudge for user %s: %s",
                user_profile.get("user_id", "unknown"),
                str(exc),
            )

    logger.info("Nudge Lambda processed %d users at %s", len(users), current_time.isoformat())
    return {
        "statusCode": 200,
        "processed_users": len(users),
        "results": results,
    }

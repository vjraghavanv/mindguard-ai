"""
Account Settings Lambda for MindGuard AI.

Handles:
- Updating notification preferences
- Updating trusted contact
- Updating escalation threshold
- Snooze management (1h, 4h, 24h)
- Data deletion requests (marks for deletion; 30-day deadline)
- Session re-authentication enforcement (15-minute inactivity timeout)

Requirements: 6.4, 7.3, 9.4, 9.5, 11.4
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TIMEOUT_MINUTES = 15
VALID_SNOOZE_DURATIONS_HOURS = {1, 4, 24}
DATA_DELETION_DEADLINE_DAYS = 30

DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def is_session_expired(
    last_activity: datetime,
    current_time: datetime,
    timeout_minutes: int = SESSION_TIMEOUT_MINUTES,
) -> bool:
    """
    Return True if the session has been inactive for >= timeout_minutes.

    Args:
        last_activity: UTC datetime of the last user activity.
        current_time: UTC datetime representing "now".
        timeout_minutes: Inactivity threshold in minutes (default 15).

    Returns:
        True if inactive for >= timeout_minutes, False otherwise.
    """
    # Ensure both datetimes are timezone-aware
    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    elapsed = current_time - last_activity
    return elapsed.total_seconds() >= timeout_minutes * 60


# ---------------------------------------------------------------------------
# Snooze helpers
# ---------------------------------------------------------------------------

def compute_snooze_until(duration_hours: int, current_time: datetime) -> str:
    """
    Compute the snooze_until timestamp for a given duration.

    Args:
        duration_hours: Snooze duration in hours. Must be one of 1, 4, or 24.
        current_time: UTC datetime representing "now".

    Returns:
        ISO-8601 UTC string for when the snooze expires.

    Raises:
        ValueError: If duration_hours is not a valid snooze duration.
    """
    if duration_hours not in VALID_SNOOZE_DURATIONS_HOURS:
        raise ValueError(
            f"Invalid snooze duration: {duration_hours}. "
            f"Must be one of {sorted(VALID_SNOOZE_DURATIONS_HOURS)}."
        )
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    snooze_until = current_time + timedelta(hours=duration_hours)
    return snooze_until.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def _get_table(table=None):
    if table is not None:
        return table
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return dynamodb.Table(DYNAMODB_TABLE)


def _get_user_profile(user_id: str, table=None) -> Optional[dict]:
    tbl = _get_table(table)
    response = tbl.get_item(Key={"user_id": user_id, "sk": "profile"})
    return response.get("Item")


def _put_user_profile(profile: dict, table=None) -> None:
    tbl = _get_table(table)
    item = {k: v for k, v in profile.items() if v is not None}
    item.setdefault("sk", "profile")
    tbl.put_item(Item=item)


# ---------------------------------------------------------------------------
# Account settings operations
# ---------------------------------------------------------------------------

def update_notification_prefs(user_id: str, prefs: dict, table=None) -> dict:
    """
    Update the notification_prefs for a user.

    Args:
        user_id: The anonymized user UUID.
        prefs: Dict with any subset of: channel, nudge_time, enabled, snooze_until.
        table: Optional DynamoDB Table resource (for testing).

    Returns:
        Updated UserProfile dict.
    """
    profile = _get_user_profile(user_id, table)
    if profile is None:
        return {"error": "User not found", "user_id": user_id}

    existing_prefs = profile.get("notification_prefs", {})
    if not isinstance(existing_prefs, dict):
        existing_prefs = {}

    # Merge provided prefs into existing
    for key in ("channel", "nudge_time", "enabled", "snooze_until"):
        if key in prefs:
            existing_prefs[key] = prefs[key]

    profile["notification_prefs"] = existing_prefs
    _put_user_profile(profile, table)
    return {"updated": True, "user_id": user_id, "notification_prefs": existing_prefs}


def update_trusted_contact(user_id: str, contact: dict, table=None) -> dict:
    """
    Update the trusted_contact for a user.

    Args:
        user_id: The anonymized user UUID.
        contact: Dict with 'name' and 'contact' (phone or email).
        table: Optional DynamoDB Table resource (for testing).

    Returns:
        Updated UserProfile dict.
    """
    profile = _get_user_profile(user_id, table)
    if profile is None:
        return {"error": "User not found", "user_id": user_id}

    profile["trusted_contact"] = {
        "name": contact.get("name", ""),
        "contact": contact.get("contact", ""),
    }
    _put_user_profile(profile, table)
    return {"updated": True, "user_id": user_id, "trusted_contact": profile["trusted_contact"]}


def update_escalation_threshold(user_id: str, threshold: int, table=None) -> dict:
    """
    Update the escalation_threshold for a user.

    Args:
        user_id: The anonymized user UUID.
        threshold: New escalation threshold (0–100).
        table: Optional DynamoDB Table resource (for testing).

    Returns:
        Updated UserProfile dict.
    """
    if not isinstance(threshold, int) or not (0 <= threshold <= 100):
        return {"error": "escalation_threshold must be an integer between 0 and 100"}

    profile = _get_user_profile(user_id, table)
    if profile is None:
        return {"error": "User not found", "user_id": user_id}

    profile["escalation_threshold"] = threshold
    _put_user_profile(profile, table)
    return {"updated": True, "user_id": user_id, "escalation_threshold": threshold}


def request_data_deletion(user_id: str, table=None) -> dict:
    """
    Mark a user's data for deletion with a 30-day deadline.

    Sets 'deletion_requested_at' and 'deletion_deadline' on the UserProfile.
    An async cleanup job is responsible for completing the deletion.

    Args:
        user_id: The anonymized user UUID.
        table: Optional DynamoDB Table resource (for testing).

    Returns:
        Dict with deletion request details.
    """
    profile = _get_user_profile(user_id, table)
    if profile is None:
        return {"error": "User not found", "user_id": user_id}

    now = datetime.now(timezone.utc)
    requested_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    deadline = (now + timedelta(days=DATA_DELETION_DEADLINE_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    profile["deletion_requested_at"] = requested_at
    profile["deletion_deadline"] = deadline
    profile["deletion_status"] = "pending"
    _put_user_profile(profile, table)

    return {
        "deletion_requested": True,
        "user_id": user_id,
        "requested_at": requested_at,
        "deletion_deadline": deadline,
        "message": f"Your data will be permanently deleted within {DATA_DELETION_DEADLINE_DAYS} days.",
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context) -> dict:
    """
    Main Lambda handler for account settings operations.

    Routes:
      PUT /account/notification-prefs   — update notification preferences
      PUT /account/trusted-contact      — update trusted contact
      PUT /account/escalation-threshold — update escalation threshold
      POST /account/snooze              — set snooze duration
      POST /account/delete              — request data deletion
    """
    path = event.get("path", "")
    body = {}
    if event.get("body"):
        try:
            body = json.loads(event["body"])
        except (json.JSONDecodeError, TypeError):
            return _response(400, {"error": "Invalid JSON body"})

    # Extract user_id from Cognito authorizer claims
    user_id = (
        event.get("requestContext", {})
        .get("authorizer", {})
        .get("claims", {})
        .get("sub", body.get("user_id", ""))
    )
    if not user_id:
        return _response(401, {"error": "Unauthorized: missing user_id"})

    if path.endswith("/notification-prefs"):
        prefs = body.get("notification_prefs", body)
        return _response(200, update_notification_prefs(user_id, prefs))

    elif path.endswith("/trusted-contact"):
        contact = body.get("trusted_contact", body)
        return _response(200, update_trusted_contact(user_id, contact))

    elif path.endswith("/escalation-threshold"):
        threshold = body.get("escalation_threshold")
        if threshold is None:
            return _response(400, {"error": "escalation_threshold is required"})
        return _response(200, update_escalation_threshold(user_id, int(threshold)))

    elif path.endswith("/snooze"):
        duration_hours = body.get("duration_hours")
        if duration_hours is None:
            return _response(400, {"error": "duration_hours is required"})
        try:
            snooze_until = compute_snooze_until(
                int(duration_hours), datetime.now(timezone.utc)
            )
        except ValueError as e:
            return _response(400, {"error": str(e)})
        result = update_notification_prefs(
            user_id, {"snooze_until": snooze_until}
        )
        result["snooze_until"] = snooze_until
        return _response(200, result)

    elif path.endswith("/delete"):
        return _response(200, request_data_deletion(user_id))

    else:
        return _response(404, {"error": "Route not found"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

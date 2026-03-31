"""
Scheduled Burnout Score Recompute Lambda for MindGuard AI.

Triggered daily by EventBridge Scheduler. For each active user, recomputes
the Burnout_Score using 30-day trend data and stores a new BurnoutScoreRecord
with trigger: "scheduled_recompute".

Requirements: 4.5
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta

import boto3
from boto3.dynamodb.conditions import Key

from src.utils.bedrock import compute_rule_based_score

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------


def _get_table():
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return dynamodb.Table(TABLE_NAME)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


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


def get_30_day_trends(user_id: str, table, current_time: datetime) -> list[dict]:
    """
    Query DynamoDB for all journal/score entries within the last 30 days for a user.

    Returns a list of item dicts ordered by timestamp.
    """
    cutoff = current_time - timedelta(days=30)
    response = table.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
    )
    trends = []
    for item in response.get("Items", []):
        ts_str = item.get("timestamp") or item.get("created_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                trends.append(dict(item))
        except (ValueError, TypeError):
            continue
    return trends


def recompute_user_score(user_id: str, table) -> dict:
    """
    Recompute the Burnout_Score for a user using 30-day trend data.

    Queries DynamoDB for 30-day trend data, calls compute_rule_based_score,
    and stores a new BurnoutScoreRecord with trigger: "scheduled_recompute"
    and a UTC timestamp.

    Args:
        user_id: The anonymized UUID of the user.
        table: The DynamoDB Table resource.

    Returns:
        Dict with user_id, burnout_score, timestamp, and trigger.
    """
    current_time = datetime.now(timezone.utc)
    trends = get_30_day_trends(user_id, table, current_time)
    burnout_score = compute_rule_based_score(trends)
    timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    record = {
        "user_id": user_id,
        "sk": f"{timestamp}#burnout",
        "timestamp": timestamp,
        "burnout_score": burnout_score,
        "trigger": "scheduled_recompute",
        "record_type": "burnout_score",
    }

    table.put_item(Item=record)
    logger.info(
        "Recomputed burnout score for user %s: %d (trigger: scheduled_recompute)",
        user_id,
        burnout_score,
    )

    return {
        "user_id": user_id,
        "burnout_score": burnout_score,
        "timestamp": timestamp,
        "trigger": "scheduled_recompute",
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def handler(event: dict, context) -> dict:
    """
    EventBridge Scheduler handler — runs daily.

    Queries DynamoDB for active users and recomputes the Burnout_Score for each.
    """
    table = _get_table()
    users = get_active_users(table)
    results = []

    for user_profile in users:
        user_id = user_profile.get("user_id", "")
        if not user_id:
            continue
        try:
            result = recompute_user_score(user_id, table)
            results.append(result)
        except Exception as exc:
            logger.error(
                "Error recomputing score for user %s: %s",
                user_id,
                str(exc),
            )

    logger.info(
        "Score recompute Lambda processed %d users",
        len(results),
    )
    return {
        "statusCode": 200,
        "processed_users": len(results),
        "results": results,
    }

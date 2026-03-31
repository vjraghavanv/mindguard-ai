"""
Report Lambda for MindGuard AI.

Triggered weekly by EventBridge Scheduler. For each active user:
- Reads 7-day trend data from DynamoDB.
- Computes sentiment distribution, average burnout score, top emotions,
  and prior-week comparison.
- Calls Bedrock to generate at least two AI insights.
- Stores EmotionalHealthReport in DynamoDB with 12-month TTL.
- Publishes SNS notification that the report is ready.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from calendar import monthrange
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from boto3.dynamodb.conditions import Key

from src.utils.notifications import send_notification

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")
SNS_REPORT_TOPIC_ARN = os.environ.get(
    "SNS_REPORT_TOPIC_ARN",
    "arn:aws:sns:us-east-1:123456789012:mindguard-report-ready",
)
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

VALID_SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}
EMOTION_KEYS = ["joy", "sadness", "anger", "fear", "disgust"]

# Minimum 12 months retention (in seconds): 365 days * 24h * 3600s
RETENTION_SECONDS = 365 * 24 * 3600


# ---------------------------------------------------------------------------
# Pure helper functions (no AWS calls — testable without mocks)
# ---------------------------------------------------------------------------


def compute_sentiment_distribution(entries: list) -> dict:
    """
    Compute the fraction of entries for each sentiment label.

    Returns {"POSITIVE": float, "NEGATIVE": float, "NEUTRAL": float, "MIXED": float}
    where values are fractions summing to 1.0 (or all 0.0 for empty input).
    """
    result = {s: 0.0 for s in VALID_SENTIMENTS}
    if not entries:
        return result

    counts: Counter = Counter()
    for entry in entries:
        label = str(entry.get("sentiment_label", "")).upper()
        if label in VALID_SENTIMENTS:
            counts[label] += 1

    total = sum(counts.values())
    if total == 0:
        return result

    for label in VALID_SENTIMENTS:
        result[label] = counts[label] / total

    return result


def compute_avg_burnout_score(entries: list) -> float:
    """
    Compute the average burnout_score across all entries.

    Returns 0.0 for empty input or when no entries have a burnout_score.
    """
    scores = [
        float(e["burnout_score"])
        for e in entries
        if "burnout_score" in e and e["burnout_score"] is not None
    ]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute_top_emotions(entries: list, n: int = 3) -> list:
    """
    Return the top N emotion names by average score across all entries.

    Each entry may have an 'emotions' dict with keys from EMOTION_KEYS.
    Returns a list of emotion name strings (up to n), ordered by descending average.
    """
    totals: dict = {k: 0.0 for k in EMOTION_KEYS}
    counts: dict = {k: 0 for k in EMOTION_KEYS}

    for entry in entries:
        emotions = entry.get("emotions") or {}
        for key in EMOTION_KEYS:
            val = emotions.get(key)
            if val is not None:
                totals[key] += float(val)
                counts[key] += 1

    averages = {
        k: (totals[k] / counts[k]) if counts[k] > 0 else 0.0
        for k in EMOTION_KEYS
    }

    sorted_emotions = sorted(averages.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_emotions[:n]]


def generate_ai_insights(trends: list, bedrock_client=None) -> list:
    """
    Call Bedrock (Claude) to generate at least 2 AI insights from 7-day trend data.

    Falls back to two static insights on any error.

    Args:
        trends: List of journal entry dicts from the past 7 days.
        bedrock_client: Optional injected Bedrock client (for testing).

    Returns:
        List of at least 2 insight strings.
    """
    static_fallback = [
        "Your emotional patterns this week suggest you may benefit from scheduling "
        "dedicated recovery time. Even 15 minutes of quiet reflection can help.",
        "Consider tracking what activities or times of day correlate with your "
        "highest stress levels — awareness is the first step to change.",
    ]

    if bedrock_client is None:
        try:
            bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        except Exception as exc:
            logger.error("Failed to create Bedrock client: %s", exc)
            return static_fallback

    prompt = (
        "You are a compassionate mental health AI assistant. "
        "Analyze the following 7-day emotional trend data and generate exactly 2 "
        "personalized, actionable insights to help the user understand their patterns "
        "and improve their wellbeing.\n\n"
        f"7-Day Trend Data: {json.dumps(trends[:50], default=str)}\n\n"
        "Return a JSON object with:\n"
        '- insights: list of exactly 2 insight strings (each max 2 sentences)\n\n'
        "Respond with valid JSON only."
    )

    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())
        result = json.loads(raw["content"][0]["text"])
        insights = result.get("insights", [])
        if isinstance(insights, list) and len(insights) >= 2:
            return [str(i) for i in insights]
        # Partial result — pad with static fallback
        while len(insights) < 2:
            insights.append(static_fallback[len(insights) % len(static_fallback)])
        return insights
    except Exception as exc:
        logger.error("Bedrock insight generation failed, using fallback: %s", exc)
        return static_fallback


def compute_ttl_timestamp(generated_at: datetime) -> int:
    """
    Return a Unix timestamp 12 months from generated_at (for DynamoDB TTL).

    Handles month-end edge cases (e.g., Jan 31 + 12 months = Jan 31 next year).

    Validates: Requirements 8.5
    """
    year = generated_at.year + 1
    month = generated_at.month
    day = min(generated_at.day, monthrange(year, month)[1])
    expiry = generated_at.replace(year=year, month=month, day=day)
    return int(expiry.timestamp())


def build_report(
    user_id: str,
    week_start: str,
    week_end: str,
    entries: list,
    prior_week_entries: list,
    ai_insights: list,
) -> dict:
    """
    Assemble an EmotionalHealthReport dict from computed components.

    Args:
        user_id: Anonymized user UUID.
        week_start: ISO-8601 UTC string for the start of the report week.
        week_end: ISO-8601 UTC string for the end of the report week.
        entries: Journal entries from the current 7-day window.
        prior_week_entries: Journal entries from the prior 7-day window.
        ai_insights: List of at least 2 AI-generated insight strings.

    Returns:
        Dict matching the EmotionalHealthReport schema.

    Validates: Requirements 8.2, 8.3
    """
    generated_at = datetime.now(timezone.utc).isoformat()
    report_id = str(uuid.uuid4())

    sentiment_distribution = compute_sentiment_distribution(entries)
    avg_burnout_score = compute_avg_burnout_score(entries)
    top_emotions = compute_top_emotions(entries)
    prior_week_avg_burnout = compute_avg_burnout_score(prior_week_entries)

    return {
        "user_id": user_id,
        "report_id": report_id,
        "week_start": week_start,
        "week_end": week_end,
        "sentiment_distribution": sentiment_distribution,
        "avg_burnout_score": avg_burnout_score,
        "top_emotions": top_emotions,
        "prior_week_avg_burnout": prior_week_avg_burnout,
        "ai_insights": ai_insights,
        "generated_at": generated_at,
    }


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


def get_active_users(table) -> list:
    """Scan DynamoDB for all UserProfile records."""
    response = table.scan(
        FilterExpression="record_type = :rt",
        ExpressionAttributeValues={":rt": "user_profile"},
    )
    return response.get("Items", [])


def get_entries_for_window(
    user_id: str,
    window_start: datetime,
    window_end: datetime,
    table,
) -> list:
    """
    Query DynamoDB for journal entries within [window_start, window_end].

    Returns a list of entry dicts.
    """
    response = table.query(
        KeyConditionExpression=Key("user_id").eq(user_id),
    )
    entries = []
    for item in response.get("Items", []):
        # Only include journal entries (not burnout records, escalation events, etc.)
        if item.get("record_type") not in (None, "journal_entry") and item.get(
            "entry_type"
        ) not in ("text", "voice"):
            continue
        ts_str = item.get("created_at") or item.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if window_start <= ts <= window_end:
                entries.append(item)
        except (ValueError, TypeError):
            continue
    return entries


def store_report(report: dict, table, ttl: int) -> None:
    """
    Persist an EmotionalHealthReport to DynamoDB with a TTL for 12-month retention.

    Validates: Requirements 8.5
    """
    from decimal import Decimal

    def _to_decimal(value):
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, dict):
            return {k: _to_decimal(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_to_decimal(v) for v in value]
        return value

    item = dict(report)
    item["sk"] = f"{report['generated_at']}#report#{report['report_id']}"
    item["record_type"] = "emotional_health_report"
    item["ttl"] = ttl  # DynamoDB TTL attribute (Unix timestamp)
    item = _to_decimal(item)
    table.put_item(Item=item)


def publish_report_ready(user_id: str, report_id: str, notification_prefs: dict, sns_client) -> None:
    """
    Publish an SNS notification that the weekly report is ready.

    Routes to the user's preferred channel via send_notification.

    Validates: Requirements 8.4
    """
    message = (
        "Your weekly MindGuard emotional health report is ready. "
        "Open the app to view your insights and trends. 💙"
    )
    send_notification(
        user_id=user_id,
        message=message,
        notification_type="report",
        notification_prefs=notification_prefs,
        sns_client=sns_client,
    )


# ---------------------------------------------------------------------------
# Per-user report generation
# ---------------------------------------------------------------------------


def process_user_report(
    user_profile: dict,
    current_time: datetime,
    table,
    sns_client,
    bedrock_client=None,
) -> dict:
    """
    Generate and store the weekly Emotional_Health_Report for a single user.

    Returns a summary dict describing the result.
    """
    user_id = user_profile.get("user_id", "")
    notification_prefs = user_profile.get("notification_prefs", {})

    week_end = current_time
    week_start = current_time - timedelta(days=7)
    prior_week_start = week_start - timedelta(days=7)

    # Fetch current and prior week entries
    entries = get_entries_for_window(user_id, week_start, week_end, table)
    prior_week_entries = get_entries_for_window(user_id, prior_week_start, week_start, table)

    # Generate AI insights
    ai_insights = generate_ai_insights(entries, bedrock_client=bedrock_client)

    # Build report
    report = build_report(
        user_id=user_id,
        week_start=week_start.isoformat(),
        week_end=week_end.isoformat(),
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=ai_insights,
    )

    # Compute TTL (12 months from now)
    generated_at_dt = datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00"))
    ttl = compute_ttl_timestamp(generated_at_dt)

    # Store report in DynamoDB
    store_report(report, table, ttl)

    # Notify user that report is ready
    publish_report_ready(user_id, report["report_id"], notification_prefs, sns_client)

    return {
        "user_id": user_id,
        "report_id": report["report_id"],
        "entries_analyzed": len(entries),
        "prior_week_entries": len(prior_week_entries),
        "ai_insights_count": len(ai_insights),
        "ttl": ttl,
    }


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------


def handler(event: dict, context) -> dict:
    """
    EventBridge Scheduler handler — runs weekly.

    Generates Emotional_Health_Reports for all active users.
    """
    current_time = datetime.now(timezone.utc)
    table = _get_table()
    sns_client = _get_sns_client()

    users = get_active_users(table)
    results = []

    for user_profile in users:
        try:
            result = process_user_report(user_profile, current_time, table, sns_client)
            results.append(result)
        except Exception as exc:
            logger.error(
                "Error generating report for user %s: %s",
                user_profile.get("user_id", "unknown"),
                str(exc),
            )

    logger.info(
        "Report Lambda processed %d users at %s",
        len(users),
        current_time.isoformat(),
    )
    return {
        "statusCode": 200,
        "processed_users": len(users),
        "results": results,
    }

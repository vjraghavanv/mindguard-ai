"""
Amazon Bedrock integration for MindGuard AI burnout scoring and coping suggestions.

Invokes Claude via Bedrock to compute a Burnout_Score (0-100) and generate a
personalized Coping_Suggestion based on 30-day emotional trend data.

Fallback: rule-based score (average of last 7 days) + static coping library on error.
Deduplication: same suggestion not repeated within 48 hours per user.

Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.5
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static coping suggestion library (4 categories, ≥8 suggestions)
# ---------------------------------------------------------------------------

STATIC_COPING_LIBRARY: list[dict] = [
    # Breathing exercises
    {"category": "breathing_exercise", "text": "Try box breathing: inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 4 times."},
    {"category": "breathing_exercise", "text": "Take 5 slow deep breaths, inhaling through your nose for 4 counts and exhaling through your mouth for 6 counts."},
    # Break reminders
    {"category": "break_reminder", "text": "Step away from your screen for 10 minutes. Stretch, get some water, and let your mind rest."},
    {"category": "break_reminder", "text": "Schedule a 15-minute break in the next hour. Even a short walk outside can reset your energy."},
    # Mindfulness prompts
    {"category": "mindfulness_prompt", "text": "Take a moment to notice 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste."},
    {"category": "mindfulness_prompt", "text": "Write down three things you are grateful for today, no matter how small. Gratitude shifts perspective."},
    # Physical activity suggestions
    {"category": "physical_activity", "text": "Do a 5-minute gentle stretch: roll your shoulders, stretch your neck side to side, and reach your arms overhead."},
    {"category": "physical_activity", "text": "Take a 10-minute walk, even just around the block. Movement helps clear mental fog and reduce stress hormones."},
    # Extra suggestions for variety
    {"category": "mindfulness_prompt", "text": "Close your eyes for 2 minutes and focus only on your breathing. Let thoughts pass without judgment."},
    {"category": "physical_activity", "text": "Try 10 jumping jacks or a quick dance to your favorite song to boost your mood with movement."},
]

VALID_COPING_CATEGORIES = {"breathing_exercise", "break_reminder", "mindfulness_prompt", "physical_activity"}

# ---------------------------------------------------------------------------
# Bedrock client
# ---------------------------------------------------------------------------

BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


def _get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


# ---------------------------------------------------------------------------
# Rule-based fallback scoring
# ---------------------------------------------------------------------------


def compute_rule_based_score(trends: list) -> int:
    """
    Compute a rule-based burnout score as the average of the last 7 days'
    burnout scores from the trend data.

    Args:
        trends: List of trend dicts, each optionally containing 'burnout_score'.

    Returns:
        Integer burnout score in [0, 100]. Returns 50 if no scores available.
    """
    scores = [
        int(entry["burnout_score"])
        for entry in trends
        if "burnout_score" in entry and entry["burnout_score"] is not None
    ]
    last_7 = scores[-7:] if len(scores) >= 7 else scores
    if not last_7:
        return 50
    avg = sum(last_7) / len(last_7)
    return max(0, min(100, round(avg)))


# ---------------------------------------------------------------------------
# Coping suggestion deduplication
# ---------------------------------------------------------------------------


def is_suggestion_duplicate(
    user_id: str,
    suggestion: str,
    current_time: datetime,
    recent_suggestions: list,
) -> bool:
    """
    Check whether a suggestion was already delivered to the user within 48 hours.

    Args:
        user_id: The user's anonymized UUID (unused here, for interface clarity).
        suggestion: The suggestion text to check.
        current_time: The current UTC datetime.
        recent_suggestions: List of dicts with 'text' and 'delivered_at' (ISO-8601 UTC str).

    Returns:
        True if the suggestion was delivered within the last 48 hours, False otherwise.
    """
    window_start = current_time - timedelta(hours=48)
    for record in recent_suggestions:
        if record.get("text") != suggestion:
            continue
        delivered_at_str = record.get("delivered_at", "")
        try:
            delivered_at = datetime.fromisoformat(delivered_at_str.replace("Z", "+00:00"))
            if delivered_at.tzinfo is None:
                delivered_at = delivered_at.replace(tzinfo=timezone.utc)
            if delivered_at > window_start:
                return True
        except (ValueError, AttributeError):
            continue
    return False


def select_coping_suggestion(
    user_id: str,
    recent_suggestions: list,
    current_time: datetime,
) -> str:
    """
    Pick a non-duplicate coping suggestion from the static library.

    Iterates through STATIC_COPING_LIBRARY and returns the first suggestion
    that has not been delivered to the user within the last 48 hours.
    Falls back to the first suggestion if all are duplicates.

    Args:
        user_id: The user's anonymized UUID.
        recent_suggestions: List of recent suggestion records (see is_suggestion_duplicate).
        current_time: The current UTC datetime.

    Returns:
        A coping suggestion text string.
    """
    for item in STATIC_COPING_LIBRARY:
        suggestion = item["text"]
        if not is_suggestion_duplicate(user_id, suggestion, current_time, recent_suggestions):
            return suggestion
    # All suggestions are duplicates — return the first one as last resort
    return STATIC_COPING_LIBRARY[0]["text"]


# ---------------------------------------------------------------------------
# Bedrock invocation
# ---------------------------------------------------------------------------


def invoke_bedrock(text: str, sentiment: dict, trends: list) -> dict:
    """
    Invoke Amazon Bedrock (Claude) to compute a Burnout_Score and Coping_Suggestion.

    Builds a structured prompt with the journal text, current sentiment, and 30-day
    trend data. Parses the JSON response for burnout_score (0-100) and
    coping_suggestion.

    On any Bedrock error or malformed response, falls back to:
      - rule-based score (average of last 7 days from trends)
      - static coping library suggestion

    Args:
        text: The journal entry text.
        sentiment: Dict with 'sentiment' label and 'sentiment_score'.
        trends: List of 30-day trend dicts (each may contain 'burnout_score',
                'sentiment_label', 'emotions', 'timestamp').

    Returns:
        Dict with 'burnout_score' (int, 0-100) and 'coping_suggestion' (str).
    """
    prompt = _build_prompt(text, sentiment, trends)

    try:
        client = _get_bedrock_client()
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
            accept="application/json",
        )
        raw = json.loads(response["body"].read())
        result = json.loads(raw["content"][0]["text"])

        burnout_score = int(result["burnout_score"])
        burnout_score = max(0, min(100, burnout_score))
        coping_suggestion = str(result["coping_suggestion"])

        return {
            "burnout_score": burnout_score,
            "coping_suggestion": coping_suggestion,
            "source": "bedrock",
        }

    except (BotoCoreError, ClientError, KeyError, ValueError, json.JSONDecodeError, Exception) as exc:
        logger.error("Bedrock invocation failed, using rule-based fallback: %s", exc)
        fallback_score = compute_rule_based_score(trends)
        fallback_suggestion = STATIC_COPING_LIBRARY[0]["text"]
        return {
            "burnout_score": fallback_score,
            "coping_suggestion": fallback_suggestion,
            "source": "fallback",
        }


def _build_prompt(text: str, sentiment: dict, trends: list) -> str:
    """Build the structured Claude prompt for burnout scoring."""
    # Summarize journaling gaps from trends
    gap_info = _compute_journaling_gaps(trends)
    negative_freq = _compute_negative_frequency(trends)

    prompt = (
        "You are a mental health AI assistant. Analyze the following journal entry "
        "and emotional trend data to assess burnout risk.\n\n"
        f"Journal Entry: {text}\n"
        f"Current Sentiment: {sentiment.get('sentiment', 'UNKNOWN')} "
        f"(score: {sentiment.get('sentiment_score', 0.0):.2f})\n"
        f"Negative Sentiment Frequency (last 30 days): {negative_freq:.1%}\n"
        f"Journaling Gaps >5 Consecutive Days: {gap_info['gaps_over_5_days']}\n"
        f"30-Day Trend Summary: {json.dumps(trends[:30])}\n\n"
        "Return a JSON object with:\n"
        "- burnout_score: integer 0-100 (higher = more burnout risk)\n"
        "- coping_suggestion: one specific, actionable suggestion (max 2 sentences)\n\n"
        "Respond with valid JSON only."
    )
    return prompt


def _compute_negative_frequency(trends: list) -> float:
    """Compute the fraction of trend entries with NEGATIVE sentiment."""
    if not trends:
        return 0.0
    negative_count = sum(
        1 for e in trends if e.get("sentiment_label", "").upper() == "NEGATIVE"
    )
    return negative_count / len(trends)


def _compute_journaling_gaps(trends: list) -> dict:
    """
    Compute journaling gap statistics from trend timestamps.

    Returns a dict with 'gaps_over_5_days' count.
    """
    timestamps = []
    for entry in trends:
        ts_str = entry.get("timestamp") or entry.get("created_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            timestamps.append(ts)
        except (ValueError, AttributeError):
            continue

    if len(timestamps) < 2:
        return {"gaps_over_5_days": 0}

    timestamps.sort()
    gaps_over_5 = 0
    for i in range(1, len(timestamps)):
        gap_days = (timestamps[i] - timestamps[i - 1]).days
        if gap_days > 5:
            gaps_over_5 += 1

    return {"gaps_over_5_days": gaps_over_5}

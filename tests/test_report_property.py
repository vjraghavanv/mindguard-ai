"""
Property-based tests for MindGuard AI Report Lambda.

# Feature: mindguard-ai, Property 16: Weekly Report Completeness

Validates: Requirements 8.2, 8.3
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone, timedelta

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.lambdas.report_lambda import (
    build_report,
    compute_sentiment_distribution,
    compute_avg_burnout_score,
    compute_top_emotions,
    generate_ai_insights,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_user_id_st = st.uuids().map(str)

_sentiment_label_st = st.sampled_from(["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"])

_emotion_dict_st = st.fixed_dictionaries({
    "joy": st.floats(min_value=0.0, max_value=1.0),
    "sadness": st.floats(min_value=0.0, max_value=1.0),
    "anger": st.floats(min_value=0.0, max_value=1.0),
    "fear": st.floats(min_value=0.0, max_value=1.0),
    "disgust": st.floats(min_value=0.0, max_value=1.0),
})

_entry_st = st.fixed_dictionaries({
    "user_id": _user_id_st,
    "sentiment_label": _sentiment_label_st,
    "burnout_score": st.integers(min_value=0, max_value=100),
    "emotions": _emotion_dict_st,
})

_entries_st = st.lists(_entry_st, min_size=0, max_size=20)

_week_start = "2025-07-07T00:00:00+00:00"
_week_end = "2025-07-14T00:00:00+00:00"

# Two static insights (no Bedrock call needed)
_static_insights = [
    "Your emotional patterns this week suggest you may benefit from scheduling recovery time.",
    "Consider tracking what activities correlate with your highest stress levels.",
]


# ---------------------------------------------------------------------------
# Property 16: Weekly Report Completeness
# Validates: Requirements 8.2, 8.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
)
def test_report_contains_sentiment_distribution(
    user_id: str,
    entries: list,
    prior_week_entries: list,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include
    sentiment distribution.

    Validates: Requirements 8.2
    """
    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=_static_insights,
    )

    assert "sentiment_distribution" in report, "Report must include sentiment_distribution"
    dist = report["sentiment_distribution"]
    assert isinstance(dist, dict), "sentiment_distribution must be a dict"
    for key in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"):
        assert key in dist, f"sentiment_distribution must contain key '{key}'"
        assert isinstance(dist[key], float), f"sentiment_distribution['{key}'] must be a float"
        assert 0.0 <= dist[key] <= 1.0, f"sentiment_distribution['{key}'] must be in [0, 1]"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
)
def test_report_contains_avg_burnout_score(
    user_id: str,
    entries: list,
    prior_week_entries: list,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include
    average Burnout_Score.

    Validates: Requirements 8.2
    """
    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=_static_insights,
    )

    assert "avg_burnout_score" in report, "Report must include avg_burnout_score"
    score = report["avg_burnout_score"]
    assert isinstance(score, float), "avg_burnout_score must be a float"
    assert 0.0 <= score <= 100.0, f"avg_burnout_score {score} must be in [0, 100]"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
)
def test_report_contains_top_emotions(
    user_id: str,
    entries: list,
    prior_week_entries: list,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include
    top detected emotions.

    Validates: Requirements 8.2
    """
    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=_static_insights,
    )

    assert "top_emotions" in report, "Report must include top_emotions"
    top = report["top_emotions"]
    assert isinstance(top, list), "top_emotions must be a list"
    # Each emotion name must be a string from the known set
    valid_emotions = {"joy", "sadness", "anger", "fear", "disgust"}
    for emotion in top:
        assert isinstance(emotion, str), "Each top emotion must be a string"
        assert emotion in valid_emotions, f"Unknown emotion: {emotion}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
)
def test_report_contains_prior_week_comparison(
    user_id: str,
    entries: list,
    prior_week_entries: list,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include
    a prior-week comparison (prior_week_avg_burnout).

    Validates: Requirements 8.2
    """
    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=_static_insights,
    )

    assert "prior_week_avg_burnout" in report, "Report must include prior_week_avg_burnout"
    prior = report["prior_week_avg_burnout"]
    assert isinstance(prior, float), "prior_week_avg_burnout must be a float"
    assert 0.0 <= prior <= 100.0, f"prior_week_avg_burnout {prior} must be in [0, 100]"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
    n_insights=st.integers(min_value=2, max_value=5),
)
def test_report_contains_at_least_two_ai_insights(
    user_id: str,
    entries: list,
    prior_week_entries: list,
    n_insights: int,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include
    at least two AI-generated insights.

    Validates: Requirements 8.3
    """
    insights = [f"Insight {i}" for i in range(n_insights)]

    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=insights,
    )

    assert "ai_insights" in report, "Report must include ai_insights"
    ai = report["ai_insights"]
    assert isinstance(ai, list), "ai_insights must be a list"
    assert len(ai) >= 2, f"Report must contain at least 2 AI insights, got {len(ai)}"
    for insight in ai:
        assert isinstance(insight, str), "Each AI insight must be a string"
        assert len(insight.strip()) > 0, "AI insight must not be empty"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    entries=_entries_st,
    prior_week_entries=_entries_st,
)
def test_report_has_all_required_fields(
    user_id: str,
    entries: list,
    prior_week_entries: list,
):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any active user, the weekly Emotional_Health_Report must include ALL
    required fields: sentiment_distribution, avg_burnout_score, top_emotions,
    prior_week_avg_burnout, and at least two AI-generated insights.

    Validates: Requirements 8.2, 8.3
    """
    report = build_report(
        user_id=user_id,
        week_start=_week_start,
        week_end=_week_end,
        entries=entries,
        prior_week_entries=prior_week_entries,
        ai_insights=_static_insights,
    )

    required_fields = [
        "user_id",
        "report_id",
        "week_start",
        "week_end",
        "sentiment_distribution",
        "avg_burnout_score",
        "top_emotions",
        "prior_week_avg_burnout",
        "ai_insights",
        "generated_at",
    ]
    for field in required_fields:
        assert field in report, f"Report is missing required field: '{field}'"

    # Structural invariants
    assert report["user_id"] == user_id
    assert len(report["ai_insights"]) >= 2
    assert isinstance(report["sentiment_distribution"], dict)
    assert isinstance(report["avg_burnout_score"], float)
    assert isinstance(report["top_emotions"], list)
    assert isinstance(report["prior_week_avg_burnout"], float)


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.fixed_dictionaries({
            "sentiment_label": _sentiment_label_st,
            "burnout_score": st.integers(0, 100),
            "emotions": _emotion_dict_st,
        }),
        min_size=1,
        max_size=20,
    )
)
def test_sentiment_distribution_sums_to_one(entries: list):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any non-empty list of entries with valid sentiment labels,
    the sentiment distribution fractions must sum to 1.0.

    Validates: Requirements 8.2
    """
    dist = compute_sentiment_distribution(entries)
    total = sum(dist.values())
    # Allow small floating-point tolerance
    assert abs(total - 1.0) < 1e-9 or total == 0.0, (
        f"Sentiment distribution must sum to 1.0, got {total}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=st.lists(
        st.fixed_dictionaries({
            "burnout_score": st.integers(0, 100),
        }),
        min_size=1,
        max_size=20,
    )
)
def test_avg_burnout_score_in_valid_range(entries: list):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    For any list of entries, compute_avg_burnout_score must return a value in [0, 100].

    Validates: Requirements 8.2
    """
    score = compute_avg_burnout_score(entries)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0, f"avg_burnout_score {score} must be in [0, 100]"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    entries=_entries_st,
    n=st.integers(min_value=1, max_value=5),
)
def test_top_emotions_length_at_most_n(entries: list, n: int):
    """
    # Feature: mindguard-ai, Property 16: Weekly Report Completeness

    compute_top_emotions must return at most n emotions.

    Validates: Requirements 8.2
    """
    top = compute_top_emotions(entries, n=n)
    assert isinstance(top, list)
    assert len(top) <= n, f"Expected at most {n} emotions, got {len(top)}"

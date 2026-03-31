"""
Property-based tests for MindGuard AI Bedrock integration.

# Feature: mindguard-ai, Property 6: Burnout Score Monotonicity
# Feature: mindguard-ai, Property 8: Coping Suggestion Generation
# Feature: mindguard-ai, Property 9: Coping Suggestion Deduplication

Validates: Requirements 4.1, 4.2, 5.1, 5.2, 5.3, 5.5
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.utils.bedrock import (
    compute_rule_based_score,
    is_suggestion_duplicate,
    select_coping_suggestion,
    STATIC_COPING_LIBRARY,
    VALID_COPING_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single trend entry with a burnout score
_burnout_score_st = st.integers(min_value=0, max_value=100)

_sentiment_label_st = st.sampled_from(["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"])


def _trend_entry_st(sentiment_label=None):
    """Strategy for a single trend entry dict."""
    label_st = st.just(sentiment_label) if sentiment_label else _sentiment_label_st
    return st.fixed_dictionaries({
        "burnout_score": _burnout_score_st,
        "sentiment_label": label_st,
    })


# Strategy for a list of trend entries with a given number of negative entries
def _trends_with_n_negative(n_negative: int, n_other: int):
    """Build a trends list with exactly n_negative NEGATIVE entries and n_other non-negative."""
    negative_entries = st.lists(
        _trend_entry_st("NEGATIVE"), min_size=n_negative, max_size=n_negative
    )
    other_entries = st.lists(
        _trend_entry_st("POSITIVE"), min_size=n_other, max_size=n_other
    )
    return st.tuples(negative_entries, other_entries).map(lambda t: t[0] + t[1])


# Strategy for a user_id
_user_id_st = st.uuids().map(str)

# Strategy for a UTC datetime within the last 30 days
_now_st = st.just(datetime.now(timezone.utc))

# Strategy for a recent suggestion record (delivered within 48h)
def _recent_suggestion_record_st(suggestion_text: str, within_48h: bool):
    if within_48h:
        # delivered between 1 second and 47h59m ago
        delta_st = st.integers(min_value=1, max_value=47 * 3600 + 59 * 60 + 59)
    else:
        # delivered more than 48h ago
        delta_st = st.integers(min_value=48 * 3600 + 1, max_value=72 * 3600)

    return delta_st.map(lambda secs: {
        "text": suggestion_text,
        "delivered_at": (
            datetime.now(timezone.utc) - timedelta(seconds=secs)
        ).isoformat(),
    })


# ---------------------------------------------------------------------------
# Property 6: Burnout Score Monotonicity
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_negative_base=st.integers(min_value=0, max_value=5),
    n_extra_negative=st.integers(min_value=1, max_value=5),
    n_other=st.integers(min_value=0, max_value=7),
    base_scores=st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=7),
)
def test_burnout_score_monotonicity(
    n_negative_base: int,
    n_extra_negative: int,
    n_other: int,
    base_scores: list,
):
    """
    # Feature: mindguard-ai, Property 6: Burnout Score Monotonicity

    For any user, increasing the frequency of negative sentiment entries in the
    30-day trend window should not decrease the computed Burnout_Score.
    (Metamorphic property: more negative input → score does not go down.)

    Validates: Requirements 4.1, 4.2
    """
    # Build base trends: n_negative_base NEGATIVE + n_other POSITIVE entries
    # Use fixed burnout scores so the rule-based average is deterministic
    base_trends = (
        [{"burnout_score": s, "sentiment_label": "NEGATIVE"} for s in base_scores[:n_negative_base]]
        + [{"burnout_score": 30, "sentiment_label": "POSITIVE"} for _ in range(n_other)]
    )

    # Determine the max score in the last 7 entries of base_trends so that
    # extra entries are guaranteed to be >= that max, ensuring the average
    # of the last 7 can only stay the same or increase.
    base_last_7_scores = [
        int(e["burnout_score"]) for e in base_trends if "burnout_score" in e
    ][-7:]
    extra_negative_score = max(base_last_7_scores) if base_last_7_scores else 80

    # Build augmented trends: add n_extra_negative NEGATIVE entries with scores
    # >= the max of the base window, so the average cannot decrease.
    augmented_trends = base_trends + [
        {"burnout_score": extra_negative_score, "sentiment_label": "NEGATIVE"}
        for _ in range(n_extra_negative)
    ]

    score_base = compute_rule_based_score(base_trends)
    score_augmented = compute_rule_based_score(augmented_trends)

    # The augmented score must be >= base score because we added entries with
    # scores >= the max of the current window, so the average cannot decrease.
    # This validates the monotonicity property: more negative (high-score) input → score does not go down
    assert score_augmented >= score_base, (
        f"Monotonicity violated: base_score={score_base}, augmented_score={score_augmented} "
        f"after adding {n_extra_negative} negative entries with score={extra_negative_score}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    trends=st.lists(
        st.fixed_dictionaries({"burnout_score": st.integers(0, 100), "sentiment_label": st.just("NEGATIVE")}),
        min_size=1,
        max_size=30,
    )
)
def test_burnout_score_in_valid_range(trends: list):
    """
    # Feature: mindguard-ai, Property 6: Burnout Score Monotonicity

    For any trend data, compute_rule_based_score must return a value in [0, 100].

    Validates: Requirements 4.1
    """
    score = compute_rule_based_score(trends)
    assert isinstance(score, int)
    assert 0 <= score <= 100, f"Score {score} is outside [0, 100]"


# ---------------------------------------------------------------------------
# Property 8: Coping Suggestion Generation
# Validates: Requirements 5.1, 5.2, 5.3
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    recent_suggestions=st.just([]),  # no prior suggestions → always get a fresh one
)
def test_coping_suggestion_always_returned(user_id: str, recent_suggestions: list):
    """
    # Feature: mindguard-ai, Property 8: Coping Suggestion Generation

    For any completed journal entry analysis, the AI_Engine must return at least
    one Coping_Suggestion.

    Validates: Requirements 5.1
    """
    current_time = datetime.now(timezone.utc)
    suggestion = select_coping_suggestion(user_id, recent_suggestions, current_time)

    assert suggestion is not None
    assert isinstance(suggestion, str)
    assert len(suggestion.strip()) > 0, "Coping suggestion must not be empty"


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    recent_suggestions=st.just([]),
)
def test_coping_suggestion_from_valid_category(user_id: str, recent_suggestions: list):
    """
    # Feature: mindguard-ai, Property 8: Coping Suggestion Generation

    The AI_Engine must select Coping_Suggestions from defined categories:
    breathing exercise, break reminder, mindfulness prompt, or physical activity.

    Validates: Requirements 5.2
    """
    current_time = datetime.now(timezone.utc)
    suggestion = select_coping_suggestion(user_id, recent_suggestions, current_time)

    # Verify the suggestion exists in the static library and belongs to a valid category
    matching = [item for item in STATIC_COPING_LIBRARY if item["text"] == suggestion]
    assert len(matching) >= 1, f"Suggestion not found in static library: {suggestion}"
    assert matching[0]["category"] in VALID_COPING_CATEGORIES, (
        f"Category '{matching[0]['category']}' is not a valid coping category"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    n_prior=st.integers(min_value=0, max_value=5),
)
def test_static_library_covers_all_required_categories(user_id: str, n_prior: int):
    """
    # Feature: mindguard-ai, Property 8: Coping Suggestion Generation

    The static coping library must contain at least one suggestion from each
    required category: breathing_exercise, break_reminder, mindfulness_prompt,
    physical_activity.

    Validates: Requirements 5.2
    """
    categories_in_library = {item["category"] for item in STATIC_COPING_LIBRARY}
    assert VALID_COPING_CATEGORIES.issubset(categories_in_library), (
        f"Missing categories: {VALID_COPING_CATEGORIES - categories_in_library}"
    )


# ---------------------------------------------------------------------------
# Property 9: Coping Suggestion Deduplication
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    suggestion_idx=st.integers(min_value=0, max_value=len(STATIC_COPING_LIBRARY) - 1),
    seconds_ago=st.integers(min_value=1, max_value=47 * 3600 + 59 * 60),
)
def test_suggestion_within_48h_is_duplicate(
    user_id: str,
    suggestion_idx: int,
    seconds_ago: int,
):
    """
    # Feature: mindguard-ai, Property 9: Coping Suggestion Deduplication

    For any user, the same Coping_Suggestion text must not appear more than once
    in any 48-hour window. A suggestion delivered within the last 48 hours must
    be detected as a duplicate.

    Validates: Requirements 5.5
    """
    suggestion = STATIC_COPING_LIBRARY[suggestion_idx]["text"]
    current_time = datetime.now(timezone.utc)
    delivered_at = current_time - timedelta(seconds=seconds_ago)

    recent_suggestions = [
        {"text": suggestion, "delivered_at": delivered_at.isoformat()}
    ]

    result = is_suggestion_duplicate(user_id, suggestion, current_time, recent_suggestions)
    assert result is True, (
        f"Expected duplicate detection for suggestion delivered {seconds_ago}s ago "
        f"(within 48h window)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    suggestion_idx=st.integers(min_value=0, max_value=len(STATIC_COPING_LIBRARY) - 1),
    seconds_ago=st.integers(min_value=48 * 3600 + 1, max_value=96 * 3600),
)
def test_suggestion_outside_48h_is_not_duplicate(
    user_id: str,
    suggestion_idx: int,
    seconds_ago: int,
):
    """
    # Feature: mindguard-ai, Property 9: Coping Suggestion Deduplication

    A suggestion delivered more than 48 hours ago must NOT be detected as a
    duplicate — it can be shown again.

    Validates: Requirements 5.5
    """
    suggestion = STATIC_COPING_LIBRARY[suggestion_idx]["text"]
    current_time = datetime.now(timezone.utc)
    delivered_at = current_time - timedelta(seconds=seconds_ago)

    recent_suggestions = [
        {"text": suggestion, "delivered_at": delivered_at.isoformat()}
    ]

    result = is_suggestion_duplicate(user_id, suggestion, current_time, recent_suggestions)
    assert result is False, (
        f"Expected no duplicate for suggestion delivered {seconds_ago}s ago "
        f"(outside 48h window)"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    user_id=_user_id_st,
    n_suggestions_used=st.integers(min_value=1, max_value=len(STATIC_COPING_LIBRARY) - 1),
)
def test_select_suggestion_avoids_recent_duplicates(
    user_id: str,
    n_suggestions_used: int,
):
    """
    # Feature: mindguard-ai, Property 9: Coping Suggestion Deduplication

    select_coping_suggestion must return a suggestion that was not delivered
    within the last 48 hours, as long as at least one unused suggestion exists.

    Validates: Requirements 5.5
    """
    current_time = datetime.now(timezone.utc)
    # Mark the first n_suggestions_used suggestions as recently delivered
    recent_suggestions = [
        {
            "text": STATIC_COPING_LIBRARY[i]["text"],
            "delivered_at": (current_time - timedelta(hours=1)).isoformat(),
        }
        for i in range(n_suggestions_used)
    ]
    used_texts = {r["text"] for r in recent_suggestions}

    # There must be at least one unused suggestion available
    unused_available = any(
        item["text"] not in used_texts for item in STATIC_COPING_LIBRARY
    )

    if unused_available:
        selected = select_coping_suggestion(user_id, recent_suggestions, current_time)
        assert selected not in used_texts, (
            f"select_coping_suggestion returned a recently-used suggestion: {selected}"
        )

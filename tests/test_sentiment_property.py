"""
Property-based tests for MindGuard AI Sentiment Analysis.

# Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

For any journal entry text, the Sentiment_Analyzer must return a sentiment label
(one of POSITIVE, NEGATIVE, NEUTRAL, MIXED), a confidence score in [0.0, 1.0],
and individual confidence scores for all five emotion categories
(joy, sadness, anger, fear, disgust).

Validates: Requirements 3.1, 3.2
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Fake AWS credentials for moto / mocked clients
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.utils.sentiment import analyze_sentiment, VALID_SENTIMENT_LABELS

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid journal entry text: non-empty, within Comprehend's limits
valid_text_st = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=500,
).filter(lambda t: t.strip() != "")

# Generate a realistic Comprehend SentimentScore dict where all values sum to ~1
_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _make_comprehend_response(sentiment_label: str, pos: float, neg: float, neu: float, mix: float) -> dict:
    """Build a mock Comprehend DetectSentiment response."""
    return {
        "Sentiment": sentiment_label,
        "SentimentScore": {
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu,
            "Mixed": mix,
        },
    }


# Strategy that generates a full mock Comprehend response for any sentiment label
comprehend_response_st = st.one_of(
    *[
        st.builds(
            _make_comprehend_response,
            sentiment_label=st.just(label),
            pos=_score_st,
            neg=_score_st,
            neu=_score_st,
            mix=_score_st,
        )
        for label in VALID_SENTIMENT_LABELS
    ]
)


# ---------------------------------------------------------------------------
# Property 5: Sentiment Analysis Completeness
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=valid_text_st, mock_response=comprehend_response_st)
def test_sentiment_returns_valid_label(text: str, mock_response: dict):
    """
    # Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

    For any journal entry text, analyze_sentiment must return a sentiment label
    that is one of POSITIVE, NEGATIVE, NEUTRAL, MIXED.

    Validates: Requirements 3.1
    """
    mock_client = MagicMock()
    mock_client.detect_sentiment.return_value = mock_response

    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
        result = analyze_sentiment(text)

    assert result.get("sentiment") in VALID_SENTIMENT_LABELS


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=valid_text_st, mock_response=comprehend_response_st)
def test_sentiment_score_in_unit_interval(text: str, mock_response: dict):
    """
    # Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

    For any journal entry text, analyze_sentiment must return a confidence score
    in [0.0, 1.0].

    Validates: Requirements 3.1
    """
    mock_client = MagicMock()
    mock_client.detect_sentiment.return_value = mock_response

    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
        result = analyze_sentiment(text)

    score = result.get("sentiment_score")
    assert score is not None
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=valid_text_st, mock_response=comprehend_response_st)
def test_all_five_emotion_categories_present(text: str, mock_response: dict):
    """
    # Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

    For any journal entry text, analyze_sentiment must return individual confidence
    scores for all five emotion categories: joy, sadness, anger, fear, disgust.

    Validates: Requirements 3.2
    """
    mock_client = MagicMock()
    mock_client.detect_sentiment.return_value = mock_response

    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
        result = analyze_sentiment(text)

    emotions = result.get("emotions")
    assert emotions is not None
    assert isinstance(emotions, dict)

    required_emotions = {"joy", "sadness", "anger", "fear", "disgust"}
    assert required_emotions.issubset(emotions.keys()), (
        f"Missing emotion keys: {required_emotions - emotions.keys()}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=valid_text_st, mock_response=comprehend_response_st)
def test_all_emotion_scores_in_unit_interval(text: str, mock_response: dict):
    """
    # Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

    For any journal entry text, each individual emotion confidence score must be
    in [0.0, 1.0].

    Validates: Requirements 3.2
    """
    mock_client = MagicMock()
    mock_client.detect_sentiment.return_value = mock_response

    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
        result = analyze_sentiment(text)

    emotions = result.get("emotions", {})
    for emotion, score in emotions.items():
        assert isinstance(score, float), f"Emotion '{emotion}' score is not a float: {score}"
        assert 0.0 <= score <= 1.0, (
            f"Emotion '{emotion}' score {score} is outside [0.0, 1.0]"
        )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(text=valid_text_st, mock_response=comprehend_response_st)
def test_sentiment_score_equals_max_of_comprehend_scores(text: str, mock_response: dict):
    """
    # Feature: mindguard-ai, Property 5: Sentiment Analysis Completeness

    The returned confidence score must equal the maximum of the four Comprehend
    SentimentScore values (Positive, Negative, Neutral, Mixed).

    Validates: Requirements 3.1
    """
    mock_client = MagicMock()
    mock_client.detect_sentiment.return_value = mock_response

    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
        result = analyze_sentiment(text)

    expected_max = max(mock_response["SentimentScore"].values())
    assert abs(result["sentiment_score"] - expected_max) < 1e-9

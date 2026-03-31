"""
Unit tests for MindGuard AI Bedrock integration (src/utils/bedrock.py).

Tests cover:
- Prompt construction with journal text, sentiment, and trend data
- JSON response parsing for burnout_score and coping_suggestion
- Fallback path on Bedrock error (rule-based score + static library)
- Deduplication boundary: exactly 48 hours

Requirements: 4.1, 4.2, 5.1, 5.5
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.utils.bedrock import (
    invoke_bedrock,
    compute_rule_based_score,
    is_suggestion_duplicate,
    select_coping_suggestion,
    STATIC_COPING_LIBRARY,
    VALID_COPING_CATEGORIES,
    _build_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bedrock_response(burnout_score: int, coping_suggestion: str) -> dict:
    """Build a mock Bedrock invoke_model response."""
    payload = json.dumps({"burnout_score": burnout_score, "coping_suggestion": coping_suggestion})
    return {
        "body": io.BytesIO(
            json.dumps({"content": [{"text": payload}]}).encode()
        )
    }


def _make_trends(scores: list[int], labels: list[str] | None = None) -> list[dict]:
    """Build a list of trend dicts with given burnout scores."""
    if labels is None:
        labels = ["NEUTRAL"] * len(scores)
    return [
        {"burnout_score": s, "sentiment_label": l}
        for s, l in zip(scores, labels)
    ]


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    def test_prompt_contains_journal_text(self):
        text = "I feel overwhelmed by deadlines."
        sentiment = {"sentiment": "NEGATIVE", "sentiment_score": 0.88}
        trends = _make_trends([60, 65, 70])
        prompt = _build_prompt(text, sentiment, trends)
        assert text in prompt

    def test_prompt_contains_sentiment_label(self):
        text = "Feeling okay today."
        sentiment = {"sentiment": "NEUTRAL", "sentiment_score": 0.75}
        trends = _make_trends([40, 45])
        prompt = _build_prompt(text, sentiment, trends)
        assert "NEUTRAL" in prompt

    def test_prompt_contains_sentiment_score(self):
        text = "Great day!"
        sentiment = {"sentiment": "POSITIVE", "sentiment_score": 0.92}
        trends = _make_trends([30])
        prompt = _build_prompt(text, sentiment, trends)
        assert "0.92" in prompt

    def test_prompt_contains_trend_data(self):
        text = "Tired again."
        sentiment = {"sentiment": "NEGATIVE", "sentiment_score": 0.80}
        trends = _make_trends([55, 60, 65, 70, 75])
        prompt = _build_prompt(text, sentiment, trends)
        # Trend data should be serialized into the prompt
        assert "burnout_score" in prompt

    def test_prompt_requests_json_response(self):
        text = "Just a test."
        sentiment = {"sentiment": "NEUTRAL", "sentiment_score": 0.5}
        trends = []
        prompt = _build_prompt(text, sentiment, trends)
        assert "burnout_score" in prompt
        assert "coping_suggestion" in prompt
        assert "JSON" in prompt

    def test_prompt_includes_negative_frequency(self):
        text = "Hard week."
        sentiment = {"sentiment": "NEGATIVE", "sentiment_score": 0.85}
        trends = _make_trends([70, 75, 80], labels=["NEGATIVE", "NEGATIVE", "POSITIVE"])
        prompt = _build_prompt(text, sentiment, trends)
        # 2/3 negative = 66.7%
        assert "66.7%" in prompt or "Negative Sentiment Frequency" in prompt


# ---------------------------------------------------------------------------
# JSON response parsing tests
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_parses_burnout_score_from_bedrock(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(72, "Take a short walk.")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Feeling stressed.", {"sentiment": "NEGATIVE", "sentiment_score": 0.8}, [])
        assert result["burnout_score"] == 72

    def test_parses_coping_suggestion_from_bedrock(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(55, "Try box breathing for 5 minutes.")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Okay day.", {"sentiment": "NEUTRAL", "sentiment_score": 0.6}, [])
        assert result["coping_suggestion"] == "Try box breathing for 5 minutes."

    def test_burnout_score_clamped_to_0_100(self):
        """Bedrock returning out-of-range score should be clamped."""
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(150, "Rest.")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Test.", {"sentiment": "NEGATIVE", "sentiment_score": 0.9}, [])
        assert result["burnout_score"] == 100

    def test_burnout_score_negative_clamped_to_zero(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(-10, "Rest.")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Test.", {"sentiment": "POSITIVE", "sentiment_score": 0.9}, [])
        assert result["burnout_score"] == 0

    def test_source_is_bedrock_on_success(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _make_bedrock_response(50, "Breathe deeply.")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Normal day.", {"sentiment": "NEUTRAL", "sentiment_score": 0.5}, [])
        assert result["source"] == "bedrock"


# ---------------------------------------------------------------------------
# Fallback path tests
# ---------------------------------------------------------------------------


class TestFallbackPath:
    def test_fallback_on_bedrock_client_error(self):
        from botocore.exceptions import ClientError
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "Bedrock unavailable"}},
            "InvokeModel",
        )
        trends = _make_trends([60, 65, 70, 75, 80, 85, 90])
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Stressed.", {"sentiment": "NEGATIVE", "sentiment_score": 0.9}, trends)
        assert result["source"] == "fallback"
        assert 0 <= result["burnout_score"] <= 100
        assert isinstance(result["coping_suggestion"], str)
        assert len(result["coping_suggestion"]) > 0

    def test_fallback_on_malformed_json(self):
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = {
            "body": io.BytesIO(b'{"content": [{"text": "not valid json {{{"}]}')
        }
        trends = _make_trends([50, 55, 60])
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Test.", {"sentiment": "NEUTRAL", "sentiment_score": 0.5}, trends)
        assert result["source"] == "fallback"

    def test_fallback_score_is_average_of_last_7_days(self):
        """Fallback score must equal the average of the last 7 trend scores."""
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("Bedrock down")
        # 10 entries; last 7 scores: 60,65,70,75,80,85,90 → avg = 75
        trends = _make_trends([10, 20, 30, 60, 65, 70, 75, 80, 85, 90])
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Test.", {"sentiment": "NEGATIVE", "sentiment_score": 0.8}, trends)
        assert result["source"] == "fallback"
        assert result["burnout_score"] == 75

    def test_fallback_with_no_trends_returns_50(self):
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("Error")
        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_client):
            result = invoke_bedrock("Test.", {"sentiment": "NEUTRAL", "sentiment_score": 0.5}, [])
        assert result["source"] == "fallback"
        assert result["burnout_score"] == 50


# ---------------------------------------------------------------------------
# compute_rule_based_score tests
# ---------------------------------------------------------------------------


class TestComputeRuleBasedScore:
    def test_average_of_exactly_7_scores(self):
        trends = _make_trends([10, 20, 30, 40, 50, 60, 70])
        # avg = 40
        assert compute_rule_based_score(trends) == 40

    def test_uses_last_7_when_more_available(self):
        # First 3 entries should be ignored; last 7: 60,65,70,75,80,85,90 → avg=75
        trends = _make_trends([10, 20, 30, 60, 65, 70, 75, 80, 85, 90])
        assert compute_rule_based_score(trends) == 75

    def test_fewer_than_7_uses_all(self):
        trends = _make_trends([40, 60])
        # avg = 50
        assert compute_rule_based_score(trends) == 50

    def test_empty_trends_returns_50(self):
        assert compute_rule_based_score([]) == 50

    def test_score_clamped_to_100(self):
        trends = _make_trends([100, 100, 100, 100, 100, 100, 100])
        assert compute_rule_based_score(trends) == 100

    def test_score_clamped_to_0(self):
        trends = _make_trends([0, 0, 0])
        assert compute_rule_based_score(trends) == 0

    def test_entries_without_burnout_score_ignored(self):
        trends = [
            {"sentiment_label": "NEGATIVE"},  # no burnout_score
            {"burnout_score": 80, "sentiment_label": "NEGATIVE"},
        ]
        assert compute_rule_based_score(trends) == 80


# ---------------------------------------------------------------------------
# Deduplication boundary tests (exactly 48 hours)
# ---------------------------------------------------------------------------


class TestDeduplicationBoundary:
    def test_suggestion_at_exactly_48h_is_not_duplicate(self):
        """A suggestion delivered exactly 48 hours ago is outside the window."""
        suggestion = STATIC_COPING_LIBRARY[0]["text"]
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        delivered_at = current_time - timedelta(hours=48)

        recent = [{"text": suggestion, "delivered_at": delivered_at.isoformat()}]
        result = is_suggestion_duplicate("user-1", suggestion, current_time, recent)
        # Exactly 48h ago means window_start == delivered_at, so NOT a duplicate
        assert result is False

    def test_suggestion_just_inside_48h_is_duplicate(self):
        """A suggestion delivered 47h59m59s ago is inside the window."""
        suggestion = STATIC_COPING_LIBRARY[0]["text"]
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        delivered_at = current_time - timedelta(hours=47, minutes=59, seconds=59)

        recent = [{"text": suggestion, "delivered_at": delivered_at.isoformat()}]
        result = is_suggestion_duplicate("user-1", suggestion, current_time, recent)
        assert result is True

    def test_suggestion_just_outside_48h_is_not_duplicate(self):
        """A suggestion delivered 48h1s ago is outside the window."""
        suggestion = STATIC_COPING_LIBRARY[0]["text"]
        current_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        delivered_at = current_time - timedelta(hours=48, seconds=1)

        recent = [{"text": suggestion, "delivered_at": delivered_at.isoformat()}]
        result = is_suggestion_duplicate("user-1", suggestion, current_time, recent)
        assert result is False

    def test_different_suggestion_not_flagged_as_duplicate(self):
        suggestion_a = STATIC_COPING_LIBRARY[0]["text"]
        suggestion_b = STATIC_COPING_LIBRARY[1]["text"]
        current_time = datetime.now(timezone.utc)
        delivered_at = current_time - timedelta(hours=1)

        recent = [{"text": suggestion_a, "delivered_at": delivered_at.isoformat()}]
        result = is_suggestion_duplicate("user-1", suggestion_b, current_time, recent)
        assert result is False

    def test_empty_recent_suggestions_never_duplicate(self):
        suggestion = STATIC_COPING_LIBRARY[0]["text"]
        current_time = datetime.now(timezone.utc)
        result = is_suggestion_duplicate("user-1", suggestion, current_time, [])
        assert result is False


# ---------------------------------------------------------------------------
# select_coping_suggestion tests
# ---------------------------------------------------------------------------


class TestSelectCopingSuggestion:
    def test_returns_non_duplicate_when_available(self):
        current_time = datetime.now(timezone.utc)
        # Mark first suggestion as recently used
        recent = [
            {
                "text": STATIC_COPING_LIBRARY[0]["text"],
                "delivered_at": (current_time - timedelta(hours=1)).isoformat(),
            }
        ]
        selected = select_coping_suggestion("user-1", recent, current_time)
        assert selected != STATIC_COPING_LIBRARY[0]["text"]

    def test_returns_suggestion_when_no_recent_history(self):
        current_time = datetime.now(timezone.utc)
        selected = select_coping_suggestion("user-1", [], current_time)
        assert isinstance(selected, str)
        assert len(selected) > 0

    def test_returns_first_suggestion_when_all_duplicates(self):
        """When all suggestions are duplicates, fall back to the first one."""
        current_time = datetime.now(timezone.utc)
        recent = [
            {
                "text": item["text"],
                "delivered_at": (current_time - timedelta(hours=1)).isoformat(),
            }
            for item in STATIC_COPING_LIBRARY
        ]
        selected = select_coping_suggestion("user-1", recent, current_time)
        assert selected == STATIC_COPING_LIBRARY[0]["text"]

    def test_selected_suggestion_is_in_static_library(self):
        current_time = datetime.now(timezone.utc)
        selected = select_coping_suggestion("user-2", [], current_time)
        library_texts = {item["text"] for item in STATIC_COPING_LIBRARY}
        assert selected in library_texts


# ---------------------------------------------------------------------------
# Static library validation tests
# ---------------------------------------------------------------------------


class TestStaticCopingLibrary:
    def test_library_has_at_least_8_suggestions(self):
        assert len(STATIC_COPING_LIBRARY) >= 8

    def test_all_required_categories_present(self):
        categories = {item["category"] for item in STATIC_COPING_LIBRARY}
        assert VALID_COPING_CATEGORIES.issubset(categories)

    def test_all_suggestions_are_non_empty_strings(self):
        for item in STATIC_COPING_LIBRARY:
            assert isinstance(item["text"], str)
            assert len(item["text"].strip()) > 0

    def test_all_categories_are_valid(self):
        for item in STATIC_COPING_LIBRARY:
            assert item["category"] in VALID_COPING_CATEGORIES, (
                f"Invalid category: {item['category']}"
            )

"""
Unit tests for MindGuard AI Sentiment Analysis (src/utils/sentiment.py).

Tests cover:
- All four sentiment labels (POSITIVE, NEGATIVE, NEUTRAL, MIXED)
- Emotion score mapping from Comprehend SentimentScore
- Retry logic: up to 3 attempts with exponential backoff
- Storage of raw entry without analysis fields on failure (analysis_status: "failed")

Requirements: 3.1, 3.2, 3.4
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")

from src.utils.sentiment import analyze_sentiment, check_transcript_fidelity, VALID_SENTIMENT_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comprehend_response(sentiment: str, pos: float, neg: float, neu: float, mix: float) -> dict:
    return {
        "Sentiment": sentiment,
        "SentimentScore": {
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu,
            "Mixed": mix,
        },
    }


def _make_client_error(code: str = "InternalServerError") -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "Simulated Comprehend error"}},
        "DetectSentiment",
    )


# ---------------------------------------------------------------------------
# Sentiment label tests
# ---------------------------------------------------------------------------


class TestSentimentLabels:
    def test_positive_sentiment(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "POSITIVE", pos=0.95, neg=0.02, neu=0.02, mix=0.01
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("I had an amazing day!")

        assert result["sentiment"] == "POSITIVE"
        assert result["sentiment_score"] == pytest.approx(0.95)
        assert result["emotions"]["joy"] == pytest.approx(0.95)

    def test_negative_sentiment(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "NEGATIVE", pos=0.02, neg=0.91, neu=0.05, mix=0.02
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("I feel terrible and exhausted.")

        assert result["sentiment"] == "NEGATIVE"
        assert result["sentiment_score"] == pytest.approx(0.91)
        # Negative score maps to sadness, anger, fear, disgust
        assert result["emotions"]["sadness"] == pytest.approx(0.91)
        assert result["emotions"]["anger"] == pytest.approx(0.91)
        assert result["emotions"]["fear"] == pytest.approx(0.91)
        assert result["emotions"]["disgust"] == pytest.approx(0.91)

    def test_neutral_sentiment(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "NEUTRAL", pos=0.1, neg=0.1, neu=0.75, mix=0.05
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Today was an ordinary day.")

        assert result["sentiment"] == "NEUTRAL"
        assert result["sentiment_score"] == pytest.approx(0.75)

    def test_mixed_sentiment(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "MIXED", pos=0.45, neg=0.40, neu=0.10, mix=0.05
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Work was stressful but I enjoyed dinner with family.")

        assert result["sentiment"] == "MIXED"
        assert result["sentiment_score"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Emotion mapping tests
# ---------------------------------------------------------------------------


class TestEmotionMapping:
    def test_joy_maps_to_positive_score(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "POSITIVE", pos=0.88, neg=0.05, neu=0.05, mix=0.02
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Wonderful news today!")

        assert result["emotions"]["joy"] == pytest.approx(0.88)

    def test_negative_emotions_map_to_negative_score(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "NEGATIVE", pos=0.03, neg=0.85, neu=0.10, mix=0.02
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Everything is going wrong.")

        emotions = result["emotions"]
        assert emotions["sadness"] == pytest.approx(0.85)
        assert emotions["anger"] == pytest.approx(0.85)
        assert emotions["fear"] == pytest.approx(0.85)
        assert emotions["disgust"] == pytest.approx(0.85)

    def test_all_five_emotion_keys_always_present(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "NEUTRAL", pos=0.2, neg=0.2, neu=0.5, mix=0.1
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Just a regular entry.")

        assert set(result["emotions"].keys()) == {"joy", "sadness", "anger", "fear", "disgust"}

    def test_sentiment_score_is_max_of_all_scores(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "MIXED", pos=0.30, neg=0.40, neu=0.20, mix=0.10
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Mixed feelings today.")

        assert result["sentiment_score"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_succeeds_on_second_attempt(self):
        """Should succeed if the first call fails but the second succeeds."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = [
            _make_client_error(),
            _make_comprehend_response("POSITIVE", pos=0.9, neg=0.05, neu=0.03, mix=0.02),
        ]
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):  # skip actual sleep
                result = analyze_sentiment("Feeling better now.")

        assert result["sentiment"] == "POSITIVE"
        assert mock_client.detect_sentiment.call_count == 2

    def test_succeeds_on_third_attempt(self):
        """Should succeed if the first two calls fail but the third succeeds."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = [
            _make_client_error(),
            _make_client_error(),
            _make_comprehend_response("NEUTRAL", pos=0.1, neg=0.1, neu=0.75, mix=0.05),
        ]
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):
                result = analyze_sentiment("Okay day overall.")

        assert result["sentiment"] == "NEUTRAL"
        assert mock_client.detect_sentiment.call_count == 3

    def test_returns_failed_status_after_three_failures(self):
        """After 3 consecutive failures, must return analysis_status: 'failed'."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = [
            _make_client_error(),
            _make_client_error(),
            _make_client_error(),
        ]
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):
                result = analyze_sentiment("This entry will fail analysis.")

        assert result["analysis_status"] == "failed"
        assert result["sentiment"] is None
        assert result["sentiment_score"] is None
        assert result["emotions"] is None
        assert mock_client.detect_sentiment.call_count == 3

    def test_exactly_three_attempts_made(self):
        """Must attempt exactly 3 times before giving up."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = Exception("Network error")
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):
                analyze_sentiment("Test entry.")

        assert mock_client.detect_sentiment.call_count == 3

    def test_backoff_sleep_called_between_retries(self):
        """Sleep must be called with 1s and 2s between the three attempts."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = [
            _make_client_error(),
            _make_client_error(),
            _make_client_error(),
        ]
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep") as mock_sleep:
                analyze_sentiment("Test backoff.")

        # Sleep called twice: after attempt 1 (1s) and after attempt 2 (2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    def test_no_sleep_after_final_failure(self):
        """Sleep must NOT be called after the third (final) failure."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = [
            _make_client_error(),
            _make_client_error(),
            _make_client_error(),
        ]
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep") as mock_sleep:
                analyze_sentiment("Test no extra sleep.")

        # Only 2 sleeps: after attempt 1 and after attempt 2
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Raw entry storage on failure (Req 3.4)
# ---------------------------------------------------------------------------


class TestFailureResult:
    def test_failed_result_has_correct_keys(self):
        """Failed result must contain analysis_status, sentiment, sentiment_score, emotions."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = Exception("Comprehend unavailable")
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):
                result = analyze_sentiment("Entry that fails.")

        assert "analysis_status" in result
        assert "sentiment" in result
        assert "sentiment_score" in result
        assert "emotions" in result

    def test_failed_result_analysis_status_is_failed(self):
        mock_client = MagicMock()
        mock_client.detect_sentiment.side_effect = Exception("Error")
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            with patch("src.utils.sentiment.time.sleep"):
                result = analyze_sentiment("Entry.")

        assert result["analysis_status"] == "failed"

    def test_successful_result_has_no_analysis_status_failed(self):
        """Successful results must not have analysis_status set to 'failed'."""
        mock_client = MagicMock()
        mock_client.detect_sentiment.return_value = _make_comprehend_response(
            "POSITIVE", pos=0.9, neg=0.05, neu=0.03, mix=0.02
        )
        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_client):
            result = analyze_sentiment("Great day!")

        assert result.get("analysis_status") != "failed"
        assert result["sentiment"] == "POSITIVE"


# ---------------------------------------------------------------------------
# check_transcript_fidelity tests
# ---------------------------------------------------------------------------


class TestTranscriptFidelity:
    def test_scores_within_tolerance_returns_true(self):
        assert check_transcript_fidelity(0.85, 0.90) is True

    def test_scores_exactly_at_tolerance_returns_true(self):
        assert check_transcript_fidelity(0.80, 0.90) is True

    def test_scores_exceeding_tolerance_returns_false(self):
        assert check_transcript_fidelity(0.80, 0.91) is False

    def test_identical_scores_returns_true(self):
        assert check_transcript_fidelity(0.75, 0.75) is True

    def test_order_independent(self):
        """Fidelity check should be symmetric."""
        assert check_transcript_fidelity(0.90, 0.85) == check_transcript_fidelity(0.85, 0.90)

"""
Sentiment and emotion analysis using Amazon Comprehend for MindGuard AI.

Calls DetectSentiment to classify journal entries and map emotion scores.
Implements retry logic with exponential backoff on Comprehend errors.

Requirements: 3.1, 3.2, 3.3, 3.4
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# Retry configuration (backoff: 1s, 2s, 4s)
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]

VALID_SENTIMENT_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"}


def _get_comprehend_client():
    return boto3.client(
        "comprehend",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment and emotion for a journal entry text using Amazon Comprehend.

    Calls DetectSentiment and maps the response to a structured dict containing:
      - sentiment: one of POSITIVE, NEGATIVE, NEUTRAL, MIXED
      - sentiment_score: float in [0.0, 1.0] — max of all sentiment scores
      - emotions: dict with keys joy, sadness, anger, fear, disgust

    On Comprehend error, retries up to 3 times with exponential backoff (1s/2s/4s).
    After 3 failures, returns {"analysis_status": "failed", "sentiment": None,
    "sentiment_score": None, "emotions": None}.

    Args:
        text: The journal entry text to analyze.

    Returns:
        A dict with sentiment analysis results or failure status.
    """
    comprehend = _get_comprehend_client()
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = comprehend.detect_sentiment(Text=text, LanguageCode="en")
            scores = response["SentimentScore"]

            # Map Comprehend scores to emotion categories.
            # joy → Positive; sadness/anger/fear/disgust → Negative
            # (anger/fear/disgust will be refined via key phrase detection in future)
            negative_score = scores.get("Negative", 0.0)
            emotions = {
                "joy": scores.get("Positive", 0.0),
                "sadness": negative_score,
                "anger": negative_score,
                "fear": negative_score,
                "disgust": negative_score,
            }

            return {
                "sentiment": response["Sentiment"],
                "sentiment_score": max(scores.values()),
                "emotions": emotions,
            }

        except (BotoCoreError, ClientError, Exception) as exc:
            last_error = exc
            logger.error(
                "Comprehend DetectSentiment failed (attempt %d/%d): %s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[attempt])

    # All retries exhausted
    logger.error(
        "Comprehend DetectSentiment failed after %d attempts. Last error: %s",
        MAX_RETRIES,
        last_error,
    )
    return {
        "analysis_status": "failed",
        "sentiment": None,
        "sentiment_score": None,
        "emotions": None,
    }


def check_transcript_fidelity(voice_score: float, transcript_score: float) -> bool:
    """
    Check that a transcript's sentiment confidence score is within 0.1 of the
    original voice entry's score.

    Args:
        voice_score: Confidence score from the original voice entry analysis.
        transcript_score: Confidence score from the re-submitted transcript.

    Returns:
        True if the scores are within 0.1 of each other, False otherwise.
    """
    within_tolerance = abs(voice_score - transcript_score) <= 0.1
    if not within_tolerance:
        logger.warning(
            "Transcript fidelity violation: voice_score=%.4f, transcript_score=%.4f, diff=%.4f",
            voice_score,
            transcript_score,
            abs(voice_score - transcript_score),
        )
    return within_tolerance

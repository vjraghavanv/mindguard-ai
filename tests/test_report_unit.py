"""
Unit tests for MindGuard AI Report Lambda (src/lambdas/report_lambda.py).

Tests cover:
- Report field completeness
- Prior-week comparison calculation
- 12-month retention policy (TTL)
- compute_sentiment_distribution
- compute_avg_burnout_score
- compute_top_emotions
- generate_ai_insights (Bedrock mock + fallback)
- build_report
- store_report (DynamoDB mock)
- process_user_report (end-to-end with mocks)

Requirements: 8.1, 8.2, 8.3, 8.5
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from io import BytesIO
from unittest.mock import MagicMock, patch

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.lambdas.report_lambda import (
    build_report,
    compute_avg_burnout_score,
    compute_sentiment_distribution,
    compute_top_emotions,
    compute_ttl_timestamp,
    generate_ai_insights,
    get_entries_for_window,
    process_user_report,
    store_report,
    RETENTION_SECONDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamodb_table():
    """Create a mocked DynamoDB table."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="mindguard-trend-store",
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield table


def _make_entry(user_id: str, ts: str, burnout_score: int = 50, sentiment: str = "NEUTRAL") -> dict:
    return {
        "user_id": user_id,
        "sk": f"{ts}#entry-{uuid.uuid4()}",
        "entry_type": "text",
        "created_at": ts,
        "timestamp": ts,
        "burnout_score": burnout_score,
        "sentiment_label": sentiment,
        "emotions": {
            "joy": Decimal("0.1"),
            "sadness": Decimal("0.3"),
            "anger": Decimal("0.2"),
            "fear": Decimal("0.2"),
            "disgust": Decimal("0.1"),
        },
    }


def _make_user_profile(user_id: str = "user-1") -> dict:
    return {
        "user_id": user_id,
        "email_hash": "abc123",
        "record_type": "user_profile",
        "notification_prefs": {
            "channel": "in_app",
            "enabled": True,
            "snooze_until": None,
            "nudge_time": "09:00",
        },
    }


# ---------------------------------------------------------------------------
# compute_sentiment_distribution
# ---------------------------------------------------------------------------

class TestComputeSentimentDistribution:
    def test_empty_entries_returns_all_zeros(self):
        dist = compute_sentiment_distribution([])
        assert dist == {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0, "MIXED": 0.0}

    def test_all_positive(self):
        entries = [{"sentiment_label": "POSITIVE"}] * 4
        dist = compute_sentiment_distribution(entries)
        assert dist["POSITIVE"] == 1.0
        assert dist["NEGATIVE"] == 0.0
        assert dist["NEUTRAL"] == 0.0
        assert dist["MIXED"] == 0.0

    def test_mixed_distribution(self):
        entries = [
            {"sentiment_label": "POSITIVE"},
            {"sentiment_label": "POSITIVE"},
            {"sentiment_label": "NEGATIVE"},
            {"sentiment_label": "NEUTRAL"},
        ]
        dist = compute_sentiment_distribution(entries)
        assert abs(dist["POSITIVE"] - 0.5) < 1e-9
        assert abs(dist["NEGATIVE"] - 0.25) < 1e-9
        assert abs(dist["NEUTRAL"] - 0.25) < 1e-9
        assert dist["MIXED"] == 0.0

    def test_fractions_sum_to_one(self):
        entries = [
            {"sentiment_label": "POSITIVE"},
            {"sentiment_label": "NEGATIVE"},
            {"sentiment_label": "NEUTRAL"},
            {"sentiment_label": "MIXED"},
        ]
        dist = compute_sentiment_distribution(entries)
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_unknown_labels_ignored(self):
        entries = [
            {"sentiment_label": "POSITIVE"},
            {"sentiment_label": "UNKNOWN"},
            {"sentiment_label": ""},
        ]
        dist = compute_sentiment_distribution(entries)
        assert dist["POSITIVE"] == 1.0
        assert abs(sum(dist.values()) - 1.0) < 1e-9

    def test_case_insensitive_labels(self):
        entries = [{"sentiment_label": "positive"}, {"sentiment_label": "NEGATIVE"}]
        dist = compute_sentiment_distribution(entries)
        assert dist["POSITIVE"] == 0.5
        assert dist["NEGATIVE"] == 0.5


# ---------------------------------------------------------------------------
# compute_avg_burnout_score
# ---------------------------------------------------------------------------

class TestComputeAvgBurnoutScore:
    def test_empty_entries_returns_zero(self):
        assert compute_avg_burnout_score([]) == 0.0

    def test_single_entry(self):
        assert compute_avg_burnout_score([{"burnout_score": 60}]) == 60.0

    def test_average_of_multiple(self):
        entries = [{"burnout_score": 40}, {"burnout_score": 60}, {"burnout_score": 80}]
        assert abs(compute_avg_burnout_score(entries) - 60.0) < 1e-9

    def test_entries_without_burnout_score_ignored(self):
        entries = [{"burnout_score": 50}, {"text_content": "no score"}]
        assert compute_avg_burnout_score(entries) == 50.0

    def test_all_entries_missing_score_returns_zero(self):
        entries = [{"text_content": "no score"}, {"text_content": "also no score"}]
        assert compute_avg_burnout_score(entries) == 0.0

    def test_result_is_float(self):
        result = compute_avg_burnout_score([{"burnout_score": 70}])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_top_emotions
# ---------------------------------------------------------------------------

class TestComputeTopEmotions:
    def test_empty_entries_returns_all_emotions_sorted(self):
        # With no data, all averages are 0.0 — returns n emotions in any order
        top = compute_top_emotions([], n=3)
        assert isinstance(top, list)
        assert len(top) <= 3

    def test_returns_at_most_n_emotions(self):
        entries = [{"emotions": {"joy": 0.9, "sadness": 0.1, "anger": 0.2, "fear": 0.3, "disgust": 0.1}}]
        top = compute_top_emotions(entries, n=2)
        assert len(top) == 2

    def test_top_emotion_is_highest_average(self):
        entries = [
            {"emotions": {"joy": 0.9, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "disgust": 0.1}},
            {"emotions": {"joy": 0.8, "sadness": 0.2, "anger": 0.1, "fear": 0.1, "disgust": 0.1}},
        ]
        top = compute_top_emotions(entries, n=1)
        assert top[0] == "joy"

    def test_default_n_is_3(self):
        entries = [{"emotions": {"joy": 0.9, "sadness": 0.8, "anger": 0.7, "fear": 0.6, "disgust": 0.5}}]
        top = compute_top_emotions(entries)
        assert len(top) == 3

    def test_entries_without_emotions_handled(self):
        entries = [{"burnout_score": 50}, {"emotions": {"joy": 0.9, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "disgust": 0.1}}]
        top = compute_top_emotions(entries, n=1)
        assert top[0] == "joy"


# ---------------------------------------------------------------------------
# compute_ttl_timestamp — 12-month retention
# ---------------------------------------------------------------------------

class TestComputeTtlTimestamp:
    def test_ttl_is_approximately_12_months_later(self):
        now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        ttl = compute_ttl_timestamp(now)
        expected = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert ttl == int(expected.timestamp())

    def test_ttl_is_integer(self):
        now = datetime.now(timezone.utc)
        ttl = compute_ttl_timestamp(now)
        assert isinstance(ttl, int)

    def test_ttl_is_in_the_future(self):
        now = datetime.now(timezone.utc)
        ttl = compute_ttl_timestamp(now)
        assert ttl > int(now.timestamp())

    def test_ttl_at_least_365_days_ahead(self):
        now = datetime.now(timezone.utc)
        ttl = compute_ttl_timestamp(now)
        min_ttl = int((now + timedelta(days=365)).timestamp())
        assert ttl >= min_ttl

    def test_ttl_handles_leap_year_edge_case(self):
        """Feb 29 in a leap year → Feb 28 in a non-leap year."""
        leap_day = datetime(2024, 2, 29, 0, 0, 0, tzinfo=timezone.utc)
        ttl = compute_ttl_timestamp(leap_day)
        # 2025 is not a leap year, so Feb 28 is the last day
        expected = datetime(2025, 2, 28, 0, 0, 0, tzinfo=timezone.utc)
        assert ttl == int(expected.timestamp())

    def test_retention_constant_is_365_days(self):
        assert RETENTION_SECONDS == 365 * 24 * 3600


# ---------------------------------------------------------------------------
# generate_ai_insights
# ---------------------------------------------------------------------------

class TestGenerateAiInsights:
    def test_fallback_returns_two_insights_on_error(self):
        """When Bedrock raises an exception, fallback returns 2 static insights."""
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = Exception("Bedrock unavailable")

        insights = generate_ai_insights([], bedrock_client=mock_client)

        assert isinstance(insights, list)
        assert len(insights) >= 2
        for insight in insights:
            assert isinstance(insight, str)
            assert len(insight.strip()) > 0

    def test_bedrock_returns_two_insights(self):
        """When Bedrock returns valid JSON with 2 insights, they are used."""
        mock_client = MagicMock()
        bedrock_response = {
            "insights": [
                "You journal less on weekends — consider a Sunday check-in.",
                "Your stress peaks align with Monday entries.",
            ]
        }
        mock_body = BytesIO(json.dumps({
            "content": [{"text": json.dumps(bedrock_response)}]
        }).encode())
        mock_client.invoke_model.return_value = {"body": mock_body}

        insights = generate_ai_insights([{"burnout_score": 60}], bedrock_client=mock_client)

        assert len(insights) == 2
        assert insights[0] == "You journal less on weekends — consider a Sunday check-in."
        assert insights[1] == "Your stress peaks align with Monday entries."

    def test_bedrock_partial_response_padded_to_two(self):
        """If Bedrock returns only 1 insight, it is padded to 2."""
        mock_client = MagicMock()
        bedrock_response = {"insights": ["Only one insight here."]}
        mock_body = BytesIO(json.dumps({
            "content": [{"text": json.dumps(bedrock_response)}]
        }).encode())
        mock_client.invoke_model.return_value = {"body": mock_body}

        insights = generate_ai_insights([], bedrock_client=mock_client)

        assert len(insights) >= 2

    def test_bedrock_malformed_json_uses_fallback(self):
        """Malformed Bedrock response falls back to static insights."""
        mock_client = MagicMock()
        mock_body = BytesIO(b"not valid json")
        mock_client.invoke_model.return_value = {"body": mock_body}

        insights = generate_ai_insights([], bedrock_client=mock_client)

        assert len(insights) >= 2


# ---------------------------------------------------------------------------
# build_report — field completeness and prior-week comparison
# ---------------------------------------------------------------------------

class TestBuildReport:
    def _make_entries(self, n: int, burnout_score: int = 60, sentiment: str = "NEGATIVE") -> list:
        return [
            {
                "burnout_score": burnout_score,
                "sentiment_label": sentiment,
                "emotions": {"joy": 0.1, "sadness": 0.7, "anger": 0.1, "fear": 0.05, "disgust": 0.05},
            }
            for _ in range(n)
        ]

    def test_report_has_all_required_fields(self):
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=self._make_entries(5),
            prior_week_entries=self._make_entries(3, burnout_score=50),
            ai_insights=["Insight 1", "Insight 2"],
        )

        required = [
            "user_id", "report_id", "week_start", "week_end",
            "sentiment_distribution", "avg_burnout_score", "top_emotions",
            "prior_week_avg_burnout", "ai_insights", "generated_at",
        ]
        for field in required:
            assert field in report, f"Missing field: {field}"

    def test_report_user_id_matches(self):
        report = build_report(
            user_id="user-abc",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=[],
            prior_week_entries=[],
            ai_insights=["A", "B"],
        )
        assert report["user_id"] == "user-abc"

    def test_report_id_is_uuid(self):
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=[],
            prior_week_entries=[],
            ai_insights=["A", "B"],
        )
        # Should not raise
        uuid.UUID(report["report_id"])

    def test_prior_week_comparison_uses_prior_entries(self):
        """prior_week_avg_burnout must reflect prior_week_entries, not current entries."""
        current_entries = self._make_entries(5, burnout_score=80)
        prior_entries = self._make_entries(5, burnout_score=40)

        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=current_entries,
            prior_week_entries=prior_entries,
            ai_insights=["A", "B"],
        )

        assert abs(report["avg_burnout_score"] - 80.0) < 1e-9
        assert abs(report["prior_week_avg_burnout"] - 40.0) < 1e-9

    def test_prior_week_zero_when_no_prior_entries(self):
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=self._make_entries(3, burnout_score=70),
            prior_week_entries=[],
            ai_insights=["A", "B"],
        )
        assert report["prior_week_avg_burnout"] == 0.0

    def test_ai_insights_preserved(self):
        insights = ["Insight one.", "Insight two.", "Insight three."]
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=[],
            prior_week_entries=[],
            ai_insights=insights,
        )
        assert report["ai_insights"] == insights

    def test_sentiment_distribution_all_negative(self):
        entries = [{"sentiment_label": "NEGATIVE", "burnout_score": 70, "emotions": {}}] * 5
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=entries,
            prior_week_entries=[],
            ai_insights=["A", "B"],
        )
        assert report["sentiment_distribution"]["NEGATIVE"] == 1.0
        assert report["sentiment_distribution"]["POSITIVE"] == 0.0

    def test_top_emotions_in_report(self):
        entries = [
            {
                "sentiment_label": "NEGATIVE",
                "burnout_score": 60,
                "emotions": {"joy": 0.05, "sadness": 0.8, "anger": 0.1, "fear": 0.03, "disgust": 0.02},
            }
        ]
        report = build_report(
            user_id="user-1",
            week_start="2025-07-07T00:00:00+00:00",
            week_end="2025-07-14T00:00:00+00:00",
            entries=entries,
            prior_week_entries=[],
            ai_insights=["A", "B"],
        )
        assert report["top_emotions"][0] == "sadness"


# ---------------------------------------------------------------------------
# store_report — 12-month retention via TTL
# ---------------------------------------------------------------------------

class TestStoreReport:
    def test_report_stored_with_ttl(self, dynamodb_table):
        """Report is stored in DynamoDB with a TTL attribute."""
        now = datetime.now(timezone.utc)
        ttl = int((now + timedelta(days=365)).timestamp())
        report = {
            "user_id": "user-1",
            "report_id": str(uuid.uuid4()),
            "week_start": "2025-07-07T00:00:00+00:00",
            "week_end": "2025-07-14T00:00:00+00:00",
            "sentiment_distribution": {"POSITIVE": 0.5, "NEGATIVE": 0.5, "NEUTRAL": 0.0, "MIXED": 0.0},
            "avg_burnout_score": 65.0,
            "top_emotions": ["sadness", "anger", "fear"],
            "prior_week_avg_burnout": 55.0,
            "ai_insights": ["Insight 1", "Insight 2"],
            "generated_at": now.isoformat(),
        }

        store_report(report, dynamodb_table, ttl)

        # Verify the item was stored
        response = dynamodb_table.scan()
        items = response.get("Items", [])
        stored = [i for i in items if i.get("report_id") == report["report_id"]]
        assert len(stored) == 1
        assert stored[0]["ttl"] == ttl
        assert stored[0]["record_type"] == "emotional_health_report"

    def test_ttl_is_at_least_12_months_from_now(self, dynamodb_table):
        """The TTL stored in DynamoDB must be at least 12 months from now."""
        now = datetime.now(timezone.utc)
        ttl = compute_ttl_timestamp(now)
        min_ttl = int((now + timedelta(days=365)).timestamp())
        assert ttl >= min_ttl


# ---------------------------------------------------------------------------
# process_user_report — end-to-end with mocked AWS
# ---------------------------------------------------------------------------

class TestProcessUserReport:
    def test_report_generated_and_stored(self, dynamodb_table):
        """process_user_report stores a report and returns a summary."""
        mock_sns = MagicMock()
        mock_bedrock = MagicMock()
        bedrock_response = {
            "insights": [
                "You journal less on weekends.",
                "Your stress peaks on Mondays.",
            ]
        }
        mock_body = BytesIO(json.dumps({
            "content": [{"text": json.dumps(bedrock_response)}]
        }).encode())
        mock_bedrock.invoke_model.return_value = {"body": mock_body}

        user_id = "user-report-test"
        current_time = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)

        # Insert some entries in the current week
        for i in range(3):
            ts = (current_time - timedelta(days=i + 1)).isoformat()
            dynamodb_table.put_item(Item=_make_entry(user_id, ts, burnout_score=60 + i * 5))

        user_profile = _make_user_profile(user_id)
        result = process_user_report(
            user_profile, current_time, dynamodb_table, mock_sns, bedrock_client=mock_bedrock
        )

        assert result["user_id"] == user_id
        assert "report_id" in result
        assert result["entries_analyzed"] == 3
        assert result["ai_insights_count"] >= 2
        assert result["ttl"] > int(current_time.timestamp())

        # Verify report is in DynamoDB
        response = dynamodb_table.scan()
        reports = [
            i for i in response.get("Items", [])
            if i.get("record_type") == "emotional_health_report"
        ]
        assert len(reports) == 1
        assert reports[0]["user_id"] == user_id

    def test_report_notification_sent(self, dynamodb_table):
        """process_user_report publishes an SNS notification."""
        mock_sns = MagicMock()
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = Exception("Bedrock unavailable")

        user_id = "user-notify-test"
        current_time = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
        user_profile = _make_user_profile(user_id)

        process_user_report(
            user_profile, current_time, dynamodb_table, mock_sns, bedrock_client=mock_bedrock
        )

        # SNS publish should have been called for the report-ready notification
        mock_sns.publish.assert_called_once()

    def test_report_with_no_entries_still_generated(self, dynamodb_table):
        """A report is generated even when there are no journal entries."""
        mock_sns = MagicMock()
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = Exception("Bedrock unavailable")

        user_id = "user-no-entries"
        current_time = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
        user_profile = _make_user_profile(user_id)

        result = process_user_report(
            user_profile, current_time, dynamodb_table, mock_sns, bedrock_client=mock_bedrock
        )

        assert result["entries_analyzed"] == 0
        assert result["ai_insights_count"] >= 2

    def test_prior_week_entries_separated_from_current(self, dynamodb_table):
        """Entries from the prior week are counted separately."""
        mock_sns = MagicMock()
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = Exception("Bedrock unavailable")

        user_id = "user-prior-week"
        current_time = datetime(2025, 7, 14, 12, 0, 0, tzinfo=timezone.utc)

        # 2 entries in current week (last 7 days)
        for i in range(2):
            ts = (current_time - timedelta(days=i + 1)).isoformat()
            dynamodb_table.put_item(Item=_make_entry(user_id, ts, burnout_score=70))

        # 3 entries in prior week (8–14 days ago)
        for i in range(3):
            ts = (current_time - timedelta(days=8 + i)).isoformat()
            dynamodb_table.put_item(Item=_make_entry(user_id, ts, burnout_score=50))

        user_profile = _make_user_profile(user_id)
        result = process_user_report(
            user_profile, current_time, dynamodb_table, mock_sns, bedrock_client=mock_bedrock
        )

        assert result["entries_analyzed"] == 2
        assert result["prior_week_entries"] == 3

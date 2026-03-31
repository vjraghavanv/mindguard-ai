"""
Unit tests for MindGuard AI Score Recompute Lambda (src/lambdas/score_recompute_lambda.py).

Tests cover:
- New BurnoutScoreRecord written with trigger: "scheduled_recompute"
- UTC timestamp is present and valid ISO-8601
- Score is computed from 30-day trend data
- Users with no trend data get a default score (50)
- handler processes all active users

Requirements: 4.5
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.lambdas.score_recompute_lambda import (
    recompute_user_score,
    get_active_users,
    get_30_day_trends,
    handler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamodb_table():
    """Create a mocked DynamoDB table for each test."""
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


def _insert_user_profile(table, user_id: str):
    table.put_item(Item={
        "user_id": user_id,
        "sk": "profile",
        "record_type": "user_profile",
        "email_hash": "abc123",
    })


def _insert_journal_entry(table, user_id: str, timestamp: str, burnout_score: int):
    table.put_item(Item={
        "user_id": user_id,
        "sk": f"{timestamp}#entry-001",
        "record_type": "journal_entry",
        "entry_type": "text",
        "timestamp": timestamp,
        "created_at": timestamp,
        "burnout_score": burnout_score,
    })


# ---------------------------------------------------------------------------
# recompute_user_score — trigger type and timestamp
# ---------------------------------------------------------------------------

class TestRecomputeUserScore:
    def test_trigger_is_scheduled_recompute(self, dynamodb_table):
        """The stored record must have trigger: 'scheduled_recompute'."""
        result = recompute_user_score("user-1", dynamodb_table)
        assert result["trigger"] == "scheduled_recompute"

    def test_result_contains_utc_timestamp(self, dynamodb_table):
        """The result must include a valid UTC ISO-8601 timestamp."""
        result = recompute_user_score("user-1", dynamodb_table)
        ts = result["timestamp"]
        # Must be parseable as ISO-8601
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None or ts.endswith("Z")

    def test_timestamp_is_recent(self, dynamodb_table):
        """The timestamp should be within a few seconds of now."""
        before = datetime.now(timezone.utc).replace(microsecond=0)
        result = recompute_user_score("user-1", dynamodb_table)
        after = datetime.now(timezone.utc)
        ts = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        assert before <= ts <= after

    def test_record_written_to_dynamodb(self, dynamodb_table):
        """A BurnoutScoreRecord must be persisted in DynamoDB."""
        result = recompute_user_score("user-db-write", dynamodb_table)
        sk = f"{result['timestamp']}#burnout"
        item = dynamodb_table.get_item(
            Key={"user_id": "user-db-write", "sk": sk}
        ).get("Item")
        assert item is not None
        assert item["trigger"] == "scheduled_recompute"
        assert int(item["burnout_score"]) == result["burnout_score"]

    def test_sort_key_format_is_timestamp_hash_burnout(self, dynamodb_table):
        """The DynamoDB sort key must follow the pattern '{timestamp}#burnout'."""
        result = recompute_user_score("user-sk", dynamodb_table)
        expected_sk = f"{result['timestamp']}#burnout"
        item = dynamodb_table.get_item(
            Key={"user_id": "user-sk", "sk": expected_sk}
        ).get("Item")
        assert item is not None

    def test_no_trend_data_returns_default_score_50(self, dynamodb_table):
        """With no trend data, compute_rule_based_score returns 50 (default)."""
        result = recompute_user_score("user-no-data", dynamodb_table)
        assert result["burnout_score"] == 50

    def test_score_computed_from_trend_data(self, dynamodb_table):
        """Score should reflect the average of available burnout scores in trends."""
        user_id = "user-with-trends"
        now = datetime.now(timezone.utc)
        # Insert 3 entries within 30 days with known scores
        for i, score in enumerate([60, 70, 80]):
            ts = (now - timedelta(days=i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            _insert_journal_entry(dynamodb_table, user_id, ts, score)

        result = recompute_user_score(user_id, dynamodb_table)
        # compute_rule_based_score averages last 7 (or all if fewer): (60+70+80)/3 = 70
        assert result["burnout_score"] == 70

    def test_only_last_7_scores_used_for_average(self, dynamodb_table):
        """compute_rule_based_score uses only the last 7 scores."""
        user_id = "user-7-scores"
        now = datetime.now(timezone.utc)
        # Insert 10 entries: first 3 have score 10, last 7 have score 90
        for i in range(3):
            ts = (now - timedelta(days=29 - i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            _insert_journal_entry(dynamodb_table, user_id, ts, 10)
        for i in range(7):
            ts = (now - timedelta(days=6 - i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            _insert_journal_entry(dynamodb_table, user_id, ts, 90)

        result = recompute_user_score(user_id, dynamodb_table)
        # Last 7 scores are all 90 → average = 90
        assert result["burnout_score"] == 90

    def test_entries_older_than_30_days_excluded(self, dynamodb_table):
        """Entries older than 30 days should not be included in trend data."""
        user_id = "user-old-entries"
        now = datetime.now(timezone.utc)
        # Insert one entry 31 days ago (outside window)
        old_ts = (now - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _insert_journal_entry(dynamodb_table, user_id, old_ts, 100)

        result = recompute_user_score(user_id, dynamodb_table)
        # Old entry excluded → no trend data → default score 50
        assert result["burnout_score"] == 50

    def test_result_contains_user_id(self, dynamodb_table):
        """The result dict must include the user_id."""
        result = recompute_user_score("user-check-id", dynamodb_table)
        assert result["user_id"] == "user-check-id"

    def test_burnout_score_in_valid_range(self, dynamodb_table):
        """Burnout score must be in [0, 100]."""
        user_id = "user-range-check"
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _insert_journal_entry(dynamodb_table, user_id, ts, 75)

        result = recompute_user_score(user_id, dynamodb_table)
        assert 0 <= result["burnout_score"] <= 100


# ---------------------------------------------------------------------------
# get_active_users
# ---------------------------------------------------------------------------

class TestGetActiveUsers:
    def test_returns_users_with_user_profile_record_type(self, dynamodb_table):
        """Only items with record_type='user_profile' should be returned."""
        _insert_user_profile(dynamodb_table, "user-a")
        _insert_user_profile(dynamodb_table, "user-b")
        # Insert a non-profile item
        dynamodb_table.put_item(Item={
            "user_id": "user-c",
            "sk": "2025-01-01T00:00:00Z#entry",
            "record_type": "journal_entry",
        })

        users = get_active_users(dynamodb_table)
        user_ids = {u["user_id"] for u in users}
        assert "user-a" in user_ids
        assert "user-b" in user_ids
        assert "user-c" not in user_ids

    def test_returns_empty_list_when_no_users(self, dynamodb_table):
        users = get_active_users(dynamodb_table)
        assert users == []


# ---------------------------------------------------------------------------
# get_30_day_trends
# ---------------------------------------------------------------------------

class TestGet30DayTrends:
    def test_includes_entries_within_30_days(self, dynamodb_table):
        user_id = "user-trends"
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _insert_journal_entry(dynamodb_table, user_id, recent_ts, 60)

        trends = get_30_day_trends(user_id, dynamodb_table, now)
        assert len(trends) == 1

    def test_excludes_entries_older_than_30_days(self, dynamodb_table):
        user_id = "user-old"
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=31)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _insert_journal_entry(dynamodb_table, user_id, old_ts, 80)

        trends = get_30_day_trends(user_id, dynamodb_table, now)
        assert len(trends) == 0

    def test_returns_empty_for_unknown_user(self, dynamodb_table):
        now = datetime.now(timezone.utc)
        trends = get_30_day_trends("unknown-user", dynamodb_table, now)
        assert trends == []


# ---------------------------------------------------------------------------
# handler — EventBridge daily trigger
# ---------------------------------------------------------------------------

@pytest.fixture
def handler_table():
    """Separate fixture for handler tests — handler uses _get_table() internally."""
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


class TestHandler:
    def test_handler_processes_all_active_users(self, handler_table):
        """handler should recompute scores for all active users."""
        _insert_user_profile(handler_table, "handler-user-1")
        _insert_user_profile(handler_table, "handler-user-2")

        result = handler({}, None)

        assert result["statusCode"] == 200
        assert result["processed_users"] == 2
        user_ids = {r["user_id"] for r in result["results"]}
        assert "handler-user-1" in user_ids
        assert "handler-user-2" in user_ids

    def test_handler_returns_200_with_no_users(self, handler_table):
        """handler should return 200 even when there are no active users."""
        result = handler({}, None)
        assert result["statusCode"] == 200
        assert result["processed_users"] == 0

    def test_handler_each_result_has_scheduled_recompute_trigger(self, handler_table):
        """Every result from handler must have trigger: 'scheduled_recompute'."""
        _insert_user_profile(handler_table, "trigger-check-user")

        result = handler({}, None)

        for r in result["results"]:
            assert r["trigger"] == "scheduled_recompute"

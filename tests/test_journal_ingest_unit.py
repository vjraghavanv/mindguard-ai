"""
Unit tests for MindGuard AI Journal Ingest Lambda.

Tests cover:
- validate_text_entry: happy path, empty, whitespace-only, boundary lengths
- handler: happy path, validation errors, DynamoDB storage

Requirements: 2.1, 2.2, 2.3, 2.4
"""
from __future__ import annotations

import json
import os
import uuid
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

# Fake AWS credentials for moto
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.lambdas.journal_ingest_lambda import validate_text_entry, MAX_TEXT_LENGTH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dynamodb_table():
    """Create a mocked DynamoDB table for tests."""
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


def _make_event(text_content: str, user_id: str = "test-user-uuid") -> dict:
    """Build a minimal API Gateway event for the journal ingest handler."""
    return {
        "body": json.dumps({"text_content": text_content}),
        "requestContext": {
            "authorizer": {
                "claims": {"sub": user_id}
            }
        },
    }


# ---------------------------------------------------------------------------
# validate_text_entry unit tests
# ---------------------------------------------------------------------------


class TestValidateTextEntry:
    def test_valid_text_accepted(self):
        is_valid, status_code, error = validate_text_entry("I had a great day today.")
        assert is_valid is True
        assert status_code == 200
        assert error == ""

    def test_empty_string_rejected_400(self):
        is_valid, status_code, error = validate_text_entry("")
        assert is_valid is False
        assert status_code == 400
        assert error != ""

    def test_whitespace_only_rejected_400(self):
        is_valid, status_code, error = validate_text_entry("   \t\n  ")
        assert is_valid is False
        assert status_code == 400
        assert error != ""

    def test_single_space_rejected_400(self):
        is_valid, status_code, error = validate_text_entry(" ")
        assert is_valid is False
        assert status_code == 400

    def test_exactly_5000_chars_accepted(self):
        text = "a" * MAX_TEXT_LENGTH
        assert len(text) == 5000
        is_valid, status_code, error = validate_text_entry(text)
        assert is_valid is True
        assert status_code == 200
        assert error == ""

    def test_5001_chars_rejected_413(self):
        text = "a" * (MAX_TEXT_LENGTH + 1)
        assert len(text) == 5001
        is_valid, status_code, error = validate_text_entry(text)
        assert is_valid is False
        assert status_code == 413
        assert error != ""

    def test_single_char_accepted(self):
        is_valid, status_code, error = validate_text_entry("x")
        assert is_valid is True
        assert status_code == 200

    def test_newline_only_rejected_400(self):
        is_valid, status_code, error = validate_text_entry("\n\n\n")
        assert is_valid is False
        assert status_code == 400

    def test_text_with_leading_trailing_whitespace_accepted(self):
        """Text that has non-whitespace content is valid even with surrounding spaces."""
        is_valid, status_code, error = validate_text_entry("  hello world  ")
        assert is_valid is True
        assert status_code == 200


# ---------------------------------------------------------------------------
# Handler unit tests
# ---------------------------------------------------------------------------


class TestHandlerHappyPath:
    @mock_aws
    def test_happy_path_returns_200(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("I feel good today.")
        response = handler(event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "entry_id" in body
        assert "sentiment" in body
        assert "burnout_score" in body
        assert "created_at" in body

    @mock_aws
    def test_happy_path_stores_entry_in_dynamodb(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        user_id = str(uuid.uuid4())
        event = _make_event("Feeling stressed about work.", user_id=user_id)
        response = handler(event, None)
        assert response["statusCode"] == 200

        # Verify the entry was stored in DynamoDB
        items = dynamodb_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 1
        assert items[0]["user_id"] == user_id
        assert items[0]["text_content"] == "Feeling stressed about work."
        assert items[0]["entry_type"] == "text"

    @mock_aws
    def test_happy_path_entry_has_utc_timestamp(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("Today was okay.")
        response = handler(event, None)
        body = json.loads(response["body"])
        # Timestamp should be ISO-8601 UTC format
        assert "T" in body["created_at"]
        assert body["created_at"].endswith("Z")


class TestHandlerValidationErrors:
    @mock_aws
    def test_empty_text_returns_400(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("")
        response = handler(event, None)
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "error" in body

    @mock_aws
    def test_whitespace_only_returns_400(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("     ")
        response = handler(event, None)
        assert response["statusCode"] == 400

    @mock_aws
    def test_5001_chars_returns_413(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("a" * 5001)
        response = handler(event, None)
        assert response["statusCode"] == 413
        body = json.loads(response["body"])
        assert "error" in body

    @mock_aws
    def test_exactly_5000_chars_returns_200(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = _make_event("a" * 5000)
        response = handler(event, None)
        assert response["statusCode"] == 200

    @mock_aws
    def test_missing_text_content_returns_400(self, dynamodb_table):
        """Missing text_content key defaults to empty string → 400."""
        from src.lambdas.journal_ingest_lambda import handler
        event = {
            "body": json.dumps({}),
            "requestContext": {"authorizer": {"claims": {"sub": "user-uuid"}}},
        }
        response = handler(event, None)
        assert response["statusCode"] == 400

    @mock_aws
    def test_invalid_json_body_returns_400(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = {
            "body": "not-valid-json",
            "requestContext": {"authorizer": {"claims": {"sub": "user-uuid"}}},
        }
        response = handler(event, None)
        assert response["statusCode"] == 400

    @mock_aws
    def test_missing_authorizer_returns_401(self, dynamodb_table):
        from src.lambdas.journal_ingest_lambda import handler
        event = {
            "body": json.dumps({"text_content": "Hello"}),
            "requestContext": {},
        }
        response = handler(event, None)
        assert response["statusCode"] == 401

    @mock_aws
    def test_empty_entry_not_stored_in_dynamodb(self, dynamodb_table):
        """Rejected entries must NOT be persisted to DynamoDB (Req 2.4)."""
        from src.lambdas.journal_ingest_lambda import handler
        user_id = str(uuid.uuid4())
        event = _make_event("", user_id=user_id)
        response = handler(event, None)
        assert response["statusCode"] == 400

        items = dynamodb_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 0

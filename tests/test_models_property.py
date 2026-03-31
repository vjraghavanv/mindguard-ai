"""
Property-based tests for MindGuard AI data models and DynamoDB storage.

# Feature: mindguard-ai, Property 4: Journal Entry Storage Round-Trip
# Feature: mindguard-ai, Property 17: No PII in Trend_Store
"""
from __future__ import annotations

import os
import re
import uuid
import datetime

import boto3
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from moto import mock_aws

# Fake AWS credentials so moto is happy
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ["DYNAMODB_TABLE"] = "mindguard-trend-store"

from src.models.models import Emotions, JournalEntry
from src.utils.dynamodb import put_item, get_item


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

anon_uuid_st = st.uuids().map(str)

timestamp_st = st.datetimes(
    min_value=datetime.datetime(2020, 1, 1),
    max_value=datetime.datetime(2030, 12, 31),
).map(lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ"))

sentiment_label_st = st.sampled_from(["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"])

# DynamoDB Decimal context supports ~38 significant digits; restrict to normal
# floats in [0.0, 1.0] with limited precision to avoid subnormal underflow.
score_st = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
).map(lambda f: round(f, 10))

burnout_st = st.integers(min_value=0, max_value=100)

# Text with no PII-like patterns (lowercase + punctuation only)
safe_text_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Zs"),
        whitelist_characters=".,!?'-",
    ),
    min_size=1,
    max_size=200,
)

emotions_st = st.builds(
    Emotions,
    joy=score_st,
    sadness=score_st,
    anger=score_st,
    fear=score_st,
    disgust=score_st,
)

journal_entry_st = st.builds(
    JournalEntry,
    user_id=anon_uuid_st,
    timestamp=timestamp_st,
    entry_id=anon_uuid_st,
    entry_type=st.sampled_from(["voice", "text"]),
    text_content=safe_text_st,
    sentiment_label=sentiment_label_st,
    sentiment_score=score_st,
    emotions=emotions_st,
    burnout_score=burnout_st,
    coping_suggestion=safe_text_st,
    created_at=timestamp_st,
    audio_s3_key=st.none(),
)


# ---------------------------------------------------------------------------
# Shared table setup — called once per test function inside mock_aws context
# ---------------------------------------------------------------------------

def _create_table():
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    try:
        dynamodb.create_table(
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
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        pass  # already exists within this mock context


# ---------------------------------------------------------------------------
# Property 4: Journal Entry Storage Round-Trip
# Validates: Requirements 2.3, 3.3, 4.3, 10.2, 10.3
# ---------------------------------------------------------------------------

@mock_aws
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(entry=journal_entry_st)
def test_journal_entry_storage_round_trip(entry: JournalEntry):
    """
    # Feature: mindguard-ai, Property 4: Journal Entry Storage Round-Trip

    For any accepted journal entry, storing it in the Trend_Store and then
    retrieving it should produce an object with all original fields intact
    (text content, UTC timestamp, user_id, sentiment label, emotion scores,
    burnout score, coping suggestion) and no data loss on deserialization.

    Validates: Requirements 2.3, 3.3, 4.3, 10.2, 10.3
    """
    _create_table()

    item = entry.to_dict()
    item["sk"] = entry.sort_key
    put_item(item)

    retrieved_raw = get_item(entry.user_id, entry.sort_key)
    assert retrieved_raw is not None, "Item not found after put_item"

    retrieved_raw["sort_key"] = retrieved_raw.pop("sk", entry.sort_key)
    retrieved = JournalEntry.from_dict(retrieved_raw)

    assert retrieved.user_id == entry.user_id
    assert retrieved.timestamp == entry.timestamp
    assert retrieved.entry_id == entry.entry_id
    assert retrieved.entry_type == entry.entry_type
    assert retrieved.text_content == entry.text_content
    assert retrieved.sentiment_label == entry.sentiment_label
    assert abs(retrieved.sentiment_score - entry.sentiment_score) < 1e-6
    assert retrieved.burnout_score == entry.burnout_score
    assert retrieved.coping_suggestion == entry.coping_suggestion
    assert retrieved.created_at == entry.created_at
    assert abs(retrieved.emotions.joy - entry.emotions.joy) < 1e-6
    assert abs(retrieved.emotions.sadness - entry.emotions.sadness) < 1e-6
    assert abs(retrieved.emotions.anger - entry.emotions.anger) < 1e-6
    assert abs(retrieved.emotions.fear - entry.emotions.fear) < 1e-6
    assert abs(retrieved.emotions.disgust - entry.emotions.disgust) < 1e-6


# ---------------------------------------------------------------------------
# Property 17: No PII in Trend_Store
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,20}\s[A-Z][a-z]{1,20}\b")


def _contains_pii(value: str) -> bool:
    return bool(
        _EMAIL_RE.search(value)
        or _PHONE_RE.search(value)
        or _NAME_RE.search(value)
    )


def _record_has_pii(record: dict) -> bool:
    for key, val in record.items():
        if isinstance(val, str) and _contains_pii(val):
            return True
        if isinstance(val, dict) and _record_has_pii(val):
            return True
    return False


def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


@mock_aws
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(entry=journal_entry_st)
def test_no_pii_in_trend_store_journal_entry(entry: JournalEntry):
    """
    # Feature: mindguard-ai, Property 17: No PII in Trend_Store

    For any record stored in the Trend_Store, the user identifier must be an
    anonymized UUID and no personally identifiable information (name, email,
    phone) must appear in the record.

    Validates: Requirements 9.3
    """
    _create_table()

    item = entry.to_dict()
    item["sk"] = entry.sort_key
    put_item(item)

    retrieved = get_item(entry.user_id, entry.sort_key)
    assert retrieved is not None

    # user_id must be a valid UUID (anonymized identifier, not PII)
    assert _is_valid_uuid(retrieved["user_id"]), (
        f"user_id '{retrieved['user_id']}' is not a valid anonymized UUID"
    )

    # No PII in any other stored field
    record_without_uid = {k: v for k, v in retrieved.items() if k != "user_id"}
    assert not _record_has_pii(record_without_uid), (
        f"PII detected in Trend_Store record: {record_without_uid}"
    )

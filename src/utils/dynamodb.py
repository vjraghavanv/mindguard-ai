"""
DynamoDB single-table helpers for MindGuard AI Trend_Store.

Table design:
  PK: user_id  (anonymized UUID)
  SK: timestamp#entry_id  (composite sort key)

Encryption at rest: AWS-managed KMS key (AES-256) — configured at table creation.
"""
from __future__ import annotations

import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")


def _get_table():
    """Return a DynamoDB Table resource (uses env-configured endpoint for moto)."""
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return dynamodb.Table(TABLE_NAME)


def _to_decimal(value: Any) -> Any:
    """Recursively convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_decimal(v) for v in value]
    return value


def _from_decimal(value: Any) -> Any:
    """Recursively convert Decimal values back to float after DynamoDB retrieval."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _from_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_decimal(v) for v in value]
    return value


def put_item(item: dict) -> dict:
    """
    Write a single item to the Trend_Store.

    The item dict must contain 'user_id' (PK) and 'sort_key' (SK).
    Floats are automatically converted to Decimal for DynamoDB.
    Returns the DynamoDB response.
    """
    table = _get_table()
    # Rename sort_key → sk for storage; drop None values
    db_item = {k: v for k, v in item.items() if v is not None}
    if "sort_key" in db_item:
        db_item["sk"] = db_item.pop("sort_key")
    db_item = _to_decimal(db_item)
    response = table.put_item(Item=db_item)
    return response


def get_item(user_id: str, sk: str) -> Optional[dict]:
    """
    Retrieve a single item by PK (user_id) and SK.

    Decimal values are converted back to float on return.
    Returns the item dict or None if not found.
    """
    table = _get_table()
    response = table.get_item(Key={"user_id": user_id, "sk": sk})
    item = response.get("Item")
    if item is not None:
        item = _from_decimal(item)
    return item


def query_by_user(user_id: str, sk_prefix: Optional[str] = None) -> list[dict]:
    """
    Query all items for a given user_id, optionally filtered by SK prefix.

    Args:
        user_id: The anonymized UUID of the user.
        sk_prefix: If provided, only items whose SK begins with this prefix are returned.

    Returns:
        List of item dicts with Decimal values converted to float.
    """
    table = _get_table()
    if sk_prefix:
        response = table.query(
            KeyConditionExpression=(
                Key("user_id").eq(user_id) & Key("sk").begins_with(sk_prefix)
            )
        )
    else:
        response = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id)
        )
    return [_from_decimal(item) for item in response.get("Items", [])]

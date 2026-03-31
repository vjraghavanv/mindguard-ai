# Project Structure

```
mindguard-ai/
├── src/
│   ├── lambdas/              # Lambda handler functions (one per API/scheduled operation)
│   │   ├── auth_lambda.py          # Register, login, token refresh, password reset
│   │   ├── journal_ingest_lambda.py # Voice/text entry pipeline orchestrator
│   │   ├── nudge_lambda.py         # EventBridge-triggered nudge/trend alert sender
│   │   ├── report_lambda.py        # Weekly emotional health report generator
│   │   ├── score_recompute_lambda.py # Scheduled daily burnout score recompute
│   │   └── account_settings_lambda.py # Notification prefs, trusted contact, escalation threshold
│   ├── models/
│   │   └── models.py         # Dataclasses: JournalEntry, UserProfile, BurnoutScoreRecord,
│   │                         #   EmotionalHealthReport, EscalationEvent, Emotions, etc.
│   └── utils/
│       ├── bedrock.py        # Amazon Bedrock (Claude) invocation helpers
│       ├── dynamodb.py       # DynamoDB single-table helpers (put_item, get_item, query_by_user)
│       ├── notifications.py  # SNS/Pinpoint notification dispatch helpers
│       └── sentiment.py      # Amazon Comprehend sentiment/emotion analysis helpers
├── tests/
│   ├── test_<module>_unit.py      # Example-based unit tests per module
│   ├── test_<module>_property.py  # Hypothesis property-based tests per module
│   └── test_pipeline_integration.py # End-to-end pipeline integration tests
├── .kiro/
│   ├── specs/mindguard-ai/   # Feature spec (requirements.md, design.md, tasks.md)
│   └── steering/             # AI steering rules (this file and siblings)
├── .hypothesis/              # Hypothesis test database (do not edit manually)
└── requirements.txt
```

## Conventions

- Lambda handlers follow the pattern: `def handler(event: dict, context) -> dict`
- All handlers return `{"statusCode": int, "headers": {...}, "body": json.dumps(...)}`
- Models use Python `@dataclass` with `to_dict()` / `from_dict()` for DynamoDB serialization
- DynamoDB single-table: PK = `user_id` (anonymized UUID), SK = composite string (e.g. `timestamp#entry_id`, `profile`, `timestamp#burnout`)
- Floats are converted to `Decimal` before writing to DynamoDB and back to `float` on read (see `dynamodb.py`)
- No PII is ever stored in the Trend_Store — emails are SHA-256 hashed, user identifiers are UUIDs
- Utils modules are stateless helpers; all AWS clients are instantiated inside functions (not at module level) to support moto mocking

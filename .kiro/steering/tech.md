# Tech Stack & Build System

## Language & Runtime
- Python 3.x
- AWS Lambda (serverless, event-driven)

## AWS Services
- API Gateway (REST + WebSocket)
- Amazon Cognito (auth, JWT tokens)
- Amazon Transcribe (voice → text)
- Amazon Comprehend (sentiment + emotion analysis)
- Amazon Bedrock / Claude (burnout scoring + coping suggestions)
- Amazon DynamoDB (Trend_Store — single-table design)
- Amazon S3 (encrypted audio storage)
- Amazon SNS + Pinpoint (notifications)
- Amazon EventBridge Scheduler (nudges, reports, score recomputes)

## Key Libraries
| Package | Purpose |
|---|---|
| `boto3` | AWS SDK |
| `pytest` | Test runner |
| `hypothesis` | Property-based testing |
| `moto[dynamodb]` | AWS mocking for tests |
| `pytest-cov` | Coverage reporting |

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (single pass)
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_auth_unit.py

# Run property-based tests only
pytest tests/ -k "property"

# Run unit tests only
pytest tests/ -k "unit"
```

## Testing Conventions
- Every module has two test files: `test_<module>_unit.py` and `test_<module>_property.py`
- Property tests use `hypothesis` with `@given` + `@settings(max_examples=100)` minimum
- AWS calls are mocked with `moto` using the `@mock_aws` decorator
- Set fake AWS credentials via `os.environ` before importing source modules in tests
- Required env vars for tests: `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DYNAMODB_TABLE`
- Property test tag format: `# Feature: mindguard-ai, Property {N}: {property_text}`

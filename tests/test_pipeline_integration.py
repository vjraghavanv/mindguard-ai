"""
Integration tests for MindGuard AI Journal Ingest Pipeline.

Tests the full end-to-end pipeline:
- Text entry → Comprehend → Bedrock → DynamoDB → SNS alert
- Voice entry → Transcribe → Comprehend → Bedrock → DynamoDB
- SNS alert published when burnout_score > 70
- SNS alert NOT published when burnout_score <= 70
- Crisis helpline in response when burnout_score > 85
- Voice pipeline: mock Transcribe returns transcript, entry stored

Requirements: 1.1, 1.2, 2.1, 3.3, 4.3, 5.3
"""
from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock, patch

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
os.environ.setdefault("AUDIO_S3_BUCKET", "mindguard-audio")
os.environ.setdefault(
    "SNS_BURNOUT_ALERT_TOPIC_ARN",
    "arn:aws:sns:us-east-1:123456789012:mindguard-burnout-alert",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aws_resources():
    """Set up mocked DynamoDB table and SNS topic for integration tests."""
    with mock_aws():
        # DynamoDB
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

        # SNS topic
        sns = boto3.client("sns", region_name="us-east-1")
        topic = sns.create_topic(Name="mindguard-burnout-alert")
        topic_arn = topic["TopicArn"]
        os.environ["SNS_BURNOUT_ALERT_TOPIC_ARN"] = topic_arn

        yield {"table": table, "sns": sns, "topic_arn": topic_arn}


def _make_text_event(text_content: str, user_id: str = None) -> dict:
    """Build a minimal API Gateway event for a text journal entry."""
    if user_id is None:
        user_id = str(uuid.uuid4())
    return {
        "body": json.dumps({"entry_type": "text", "text_content": text_content}),
        "requestContext": {"authorizer": {"claims": {"sub": user_id}}},
    }


def _make_voice_event(
    audio_format: str = "mp3",
    audio_duration: float = 60.0,
    user_id: str = None,
) -> dict:
    """Build a minimal API Gateway event for a voice journal entry."""
    if user_id is None:
        user_id = str(uuid.uuid4())
    return {
        "body": json.dumps({
            "entry_type": "voice",
            "audio_format": audio_format,
            "audio_duration_seconds": audio_duration,
            "audio_data": "fake-audio-bytes",
        }),
        "requestContext": {"authorizer": {"claims": {"sub": user_id}}},
    }


def _mock_comprehend_response(sentiment: str = "NEGATIVE", score: float = 0.85):
    """Return a mock Comprehend DetectSentiment response."""
    return {
        "Sentiment": sentiment,
        "SentimentScore": {
            "Positive": 0.02,
            "Negative": score if sentiment == "NEGATIVE" else 0.05,
            "Neutral": 0.05,
            "Mixed": 0.02,
        },
    }


def _mock_bedrock_response(burnout_score: int = 75, suggestion: str = "Take a break."):
    """Return a mock Bedrock invoke_model response."""
    result_json = json.dumps({
        "burnout_score": burnout_score,
        "coping_suggestion": suggestion,
    })
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({
        "content": [{"text": result_json}]
    })
    return {"body": mock_body}


# ---------------------------------------------------------------------------
# Test: Full text pipeline → DynamoDB storage
# ---------------------------------------------------------------------------


class TestTextPipelineIntegration:
    """End-to-end: text entry → sentiment → burnout score → DynamoDB."""

    @mock_aws
    def test_text_entry_stored_in_dynamodb(self, aws_resources):
        """Full pipeline stores entry with correct fields in DynamoDB."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("I feel overwhelmed with work today.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.85)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(75, "Try box breathing.")

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "entry_id" in body
        assert body["burnout_score"] == 75
        assert body["coping_suggestion"] == "Try box breathing."

        # Verify DynamoDB storage
        items = aws_resources["table"].query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 1
        item = items[0]
        assert item["user_id"] == user_id
        assert item["text_content"] == "I feel overwhelmed with work today."
        assert item["entry_type"] == "text"
        assert item["sentiment_label"] == "NEGATIVE"
        assert "burnout_score" in item
        assert "coping_suggestion" in item
        assert "created_at" in item

    @mock_aws
    def test_text_entry_has_anonymized_user_id(self, aws_resources):
        """DynamoDB entry must use anonymized user_id — no PII (Req 9.3)."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("Feeling stressed.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response()

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(50)

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                handler(event, None)

        items = aws_resources["table"].query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 1
        item = items[0]
        # user_id must be a UUID (anonymized), not an email or name
        assert "@" not in item["user_id"]
        assert " " not in item["user_id"]

    @mock_aws
    def test_text_entry_sentiment_stored(self, aws_resources):
        """Sentiment label and score are stored alongside the entry (Req 3.3)."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("Today was a great day!", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("POSITIVE", 0.92)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(20, "Keep it up!")

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                handler(event, None)

        items = aws_resources["table"].query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 1
        item = items[0]
        assert item["sentiment_label"] == "POSITIVE"
        assert "sentiment_score" in item
        assert "emotions" in item


# ---------------------------------------------------------------------------
# Test: SNS alert published when burnout_score > 70
# ---------------------------------------------------------------------------


class TestSNSAlertIntegration:
    """SNS burnout alert is published when score > 70, not published when <= 70."""

    @mock_aws
    def test_sns_alert_published_when_score_above_70(self, aws_resources):
        """SNS alert must be published when burnout_score > 70 (Req 4.4)."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("I am completely burned out.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.95)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(80, "Rest now.")

        mock_sns_client = MagicMock()

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                    response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["burnout_score"] == 80
        # SNS publish must have been called
        mock_sns_client.publish.assert_called_once()

    @mock_aws
    def test_sns_alert_not_published_when_score_at_70(self, aws_resources):
        """SNS alert must NOT be published when burnout_score == 70 (Req 4.4)."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("Feeling okay today.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEUTRAL", 0.6)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(70, "Take a walk.")

        mock_sns_client = MagicMock()

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                    response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["burnout_score"] == 70
        # SNS publish must NOT have been called (score is exactly 70, not > 70)
        mock_sns_client.publish.assert_not_called()

    @mock_aws
    def test_sns_alert_not_published_when_score_below_70(self, aws_resources):
        """SNS alert must NOT be published when burnout_score < 70."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("Had a good day.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("POSITIVE", 0.8)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(40, "Keep journaling!")

        mock_sns_client = MagicMock()

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                    response = handler(event, None)

        assert response["statusCode"] == 200
        mock_sns_client.publish.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Crisis helpline in response when burnout_score > 85
# ---------------------------------------------------------------------------


class TestCrisisHelplineIntegration:
    """Crisis helpline is included in response when burnout_score > 85 (Req 7.2)."""

    @mock_aws
    def test_crisis_helpline_in_response_when_score_above_85(self, aws_resources):
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("I can't cope anymore.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.98)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(90, "Please seek help.")

        mock_sns_client = MagicMock()

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                    response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["burnout_score"] == 90
        assert "crisis_helpline" in body
        assert body["crisis_helpline"]["number"] == "988"

    @mock_aws
    def test_crisis_helpline_not_in_response_when_score_at_85(self, aws_resources):
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_text_event("Feeling very stressed.", user_id=user_id)

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.88)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(85, "Breathe deeply.")

        mock_sns_client = MagicMock()

        with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
            with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                    response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["burnout_score"] == 85
        assert "crisis_helpline" not in body


# ---------------------------------------------------------------------------
# Test: Voice pipeline → Transcribe → DynamoDB
# ---------------------------------------------------------------------------


class TestVoicePipelineIntegration:
    """End-to-end: voice entry → Transcribe → sentiment → burnout score → DynamoDB."""

    @mock_aws
    def test_voice_entry_transcribed_and_stored(self, aws_resources):
        """Voice pipeline: mock Transcribe returns transcript, entry stored in DynamoDB."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_voice_event(audio_format="mp3", audio_duration=60.0, user_id=user_id)

        transcript_text = "I have been feeling very anxious about my workload."

        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": transcript_text},
            }
        }

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.88)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(72, "Take a short walk.")

        mock_sns_client = MagicMock()

        with patch("src.lambdas.journal_ingest_lambda._get_s3_client", return_value=mock_s3):
            with patch("src.lambdas.journal_ingest_lambda._get_transcribe_client", return_value=mock_transcribe):
                with patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None):
                    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
                        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                            with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                                response = handler(event, None)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "entry_id" in body
        assert body["burnout_score"] == 72

        # Verify entry stored in DynamoDB with entry_type = "voice"
        items = aws_resources["table"].query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("user_id").eq(user_id)
        )["Items"]
        assert len(items) == 1
        item = items[0]
        assert item["entry_type"] == "voice"
        assert item["user_id"] == user_id
        assert "sentiment_label" in item
        assert "burnout_score" in item

    @mock_aws
    def test_voice_entry_transcribe_failure_returns_202(self, aws_resources):
        """On Transcribe failure, return 202 with retry_available=True (Req 1.3)."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_voice_event(audio_format="wav", audio_duration=120.0, user_id=user_id)

        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "Invalid audio",
            }
        }

        with patch("src.lambdas.journal_ingest_lambda._get_s3_client", return_value=mock_s3):
            with patch("src.lambdas.journal_ingest_lambda._get_transcribe_client", return_value=mock_transcribe):
                with patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None):
                    response = handler(event, None)

        assert response["statusCode"] == 202
        body = json.loads(response["body"])
        assert body["retry_available"] is True

    @mock_aws
    def test_voice_entry_invalid_format_returns_413(self, aws_resources):
        """Invalid audio format returns 413 without calling Transcribe."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = {
            "body": json.dumps({
                "entry_type": "voice",
                "audio_format": "ogg",
                "audio_duration_seconds": 60.0,
                "audio_data": "fake",
            }),
            "requestContext": {"authorizer": {"claims": {"sub": user_id}}},
        }

        mock_transcribe = MagicMock()

        with patch("src.lambdas.journal_ingest_lambda._get_transcribe_client", return_value=mock_transcribe):
            response = handler(event, None)

        assert response["statusCode"] == 413
        mock_transcribe.start_transcription_job.assert_not_called()

    @mock_aws
    def test_voice_entry_sns_alert_published_when_score_above_70(self, aws_resources):
        """SNS alert published for voice entry when burnout_score > 70."""
        from src.lambdas.journal_ingest_lambda import handler

        user_id = str(uuid.uuid4())
        event = _make_voice_event(audio_format="m4a", audio_duration=90.0, user_id=user_id)

        mock_s3 = MagicMock()
        mock_transcribe = MagicMock()
        mock_transcribe.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "Transcript text here"},
            }
        }

        mock_comprehend = MagicMock()
        mock_comprehend.detect_sentiment.return_value = _mock_comprehend_response("NEGATIVE", 0.9)

        mock_bedrock_client = MagicMock()
        mock_bedrock_client.invoke_model.return_value = _mock_bedrock_response(85, "Seek support.")

        mock_sns_client = MagicMock()

        with patch("src.lambdas.journal_ingest_lambda._get_s3_client", return_value=mock_s3):
            with patch("src.lambdas.journal_ingest_lambda._get_transcribe_client", return_value=mock_transcribe):
                with patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None):
                    with patch("src.utils.sentiment._get_comprehend_client", return_value=mock_comprehend):
                        with patch("src.utils.bedrock._get_bedrock_client", return_value=mock_bedrock_client):
                            with patch("src.utils.notifications._get_sns_client", return_value=mock_sns_client):
                                response = handler(event, None)

        assert response["statusCode"] == 200
        mock_sns_client.publish.assert_called_once()

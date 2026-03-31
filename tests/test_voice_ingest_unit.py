"""
Unit tests for MindGuard AI Voice Journal Ingest.

Tests cover:
- validate_audio_entry: accepted formats, rejected formats, duration boundary
- transcribe_audio: successful completion, timeout path, failure path

Requirements: 1.1, 1.3, 1.4, 1.5
"""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AUDIO_S3_BUCKET", "mindguard-audio")

from src.lambdas.journal_ingest_lambda import (
    validate_audio_entry,
    transcribe_audio,
    TranscribeTimeoutError,
    MAX_AUDIO_DURATION_SECONDS,
    ACCEPTED_AUDIO_FORMATS,
)


# ---------------------------------------------------------------------------
# validate_audio_entry — accepted formats
# ---------------------------------------------------------------------------


class TestValidateAudioEntryAcceptedFormats:
    """Each accepted format (MP3, WAV, M4A) with a valid duration must be accepted."""

    def test_mp3_lowercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("mp3", 60.0)
        assert is_valid is True
        assert status_code == 200
        assert error == ""

    def test_mp3_uppercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("MP3", 60.0)
        assert is_valid is True
        assert status_code == 200

    def test_wav_lowercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("wav", 120.0)
        assert is_valid is True
        assert status_code == 200

    def test_wav_uppercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("WAV", 120.0)
        assert is_valid is True
        assert status_code == 200

    def test_m4a_lowercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("m4a", 300.0)
        assert is_valid is True
        assert status_code == 200

    def test_m4a_uppercase_accepted(self):
        is_valid, status_code, error = validate_audio_entry("M4A", 300.0)
        assert is_valid is True
        assert status_code == 200

    def test_mixed_case_accepted(self):
        is_valid, status_code, error = validate_audio_entry("Mp3", 60.0)
        assert is_valid is True
        assert status_code == 200


# ---------------------------------------------------------------------------
# validate_audio_entry — rejected formats
# ---------------------------------------------------------------------------


class TestValidateAudioEntryRejectedFormats:
    """Unsupported formats must be rejected with HTTP 413."""

    def test_ogg_rejected(self):
        is_valid, status_code, error = validate_audio_entry("ogg", 60.0)
        assert is_valid is False
        assert status_code == 413
        assert error != ""

    def test_flac_rejected(self):
        is_valid, status_code, error = validate_audio_entry("flac", 60.0)
        assert is_valid is False
        assert status_code == 413

    def test_aac_rejected(self):
        is_valid, status_code, error = validate_audio_entry("aac", 60.0)
        assert is_valid is False
        assert status_code == 413

    def test_empty_format_rejected(self):
        is_valid, status_code, error = validate_audio_entry("", 60.0)
        assert is_valid is False
        assert status_code == 413

    def test_mp4_rejected(self):
        is_valid, status_code, error = validate_audio_entry("mp4", 60.0)
        assert is_valid is False
        assert status_code == 413


# ---------------------------------------------------------------------------
# validate_audio_entry — duration boundary
# ---------------------------------------------------------------------------


class TestValidateAudioEntryDurationBoundary:
    """Duration boundary: exactly 600s is valid; 600.001s is not."""

    def test_zero_duration_accepted(self):
        is_valid, status_code, error = validate_audio_entry("mp3", 0.0)
        assert is_valid is True
        assert status_code == 200

    def test_exactly_600_seconds_accepted(self):
        is_valid, status_code, error = validate_audio_entry("mp3", MAX_AUDIO_DURATION_SECONDS)
        assert is_valid is True
        assert status_code == 200

    def test_600_001_seconds_rejected(self):
        is_valid, status_code, error = validate_audio_entry("mp3", MAX_AUDIO_DURATION_SECONDS + 0.001)
        assert is_valid is False
        assert status_code == 413
        assert error != ""

    def test_601_seconds_rejected(self):
        is_valid, status_code, error = validate_audio_entry("wav", 601.0)
        assert is_valid is False
        assert status_code == 413

    def test_1200_seconds_rejected(self):
        is_valid, status_code, error = validate_audio_entry("m4a", 1200.0)
        assert is_valid is False
        assert status_code == 413

    def test_valid_format_invalid_duration_error_message_non_empty(self):
        is_valid, status_code, error = validate_audio_entry("mp3", 700.0)
        assert not is_valid
        assert error != ""

    def test_invalid_format_valid_duration_error_message_non_empty(self):
        is_valid, status_code, error = validate_audio_entry("ogg", 60.0)
        assert not is_valid
        assert error != ""


# ---------------------------------------------------------------------------
# transcribe_audio — successful completion
# ---------------------------------------------------------------------------


class TestTranscribeAudioSuccess:
    """transcribe_audio returns transcript URI when job completes."""

    @patch("src.lambdas.journal_ingest_lambda._get_transcribe_client")
    @patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None)
    def test_returns_transcript_uri_on_completion(self, mock_sleep, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "https://s3.amazonaws.com/bucket/transcript.json"},
            }
        }

        result = transcribe_audio("audio/test.mp3", "job-123")
        assert result == "https://s3.amazonaws.com/bucket/transcript.json"
        mock_client.start_transcription_job.assert_called_once()

    @patch("src.lambdas.journal_ingest_lambda._get_transcribe_client")
    @patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None)
    def test_start_job_called_with_correct_params(self, mock_sleep, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {"TranscriptFileUri": "https://example.com/t.json"},
            }
        }

        transcribe_audio("audio/recording.wav", "job-wav-001")

        call_kwargs = mock_client.start_transcription_job.call_args[1]
        assert call_kwargs["TranscriptionJobName"] == "job-wav-001"
        assert "mindguard-audio" in call_kwargs["Media"]["MediaFileUri"]
        assert call_kwargs["MediaFormat"] == "wav"
        assert call_kwargs["LanguageCode"] == "en-US"


# ---------------------------------------------------------------------------
# transcribe_audio — Transcribe failure path
# ---------------------------------------------------------------------------


class TestTranscribeAudioFailure:
    """transcribe_audio raises TranscribeTimeoutError on job failure."""

    @patch("src.lambdas.journal_ingest_lambda._get_transcribe_client")
    @patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None)
    def test_raises_on_failed_job(self, mock_sleep, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "Invalid audio format",
            }
        }

        with pytest.raises(TranscribeTimeoutError, match="Transcribe job failed"):
            transcribe_audio("audio/bad.mp3", "job-fail-001")


# ---------------------------------------------------------------------------
# transcribe_audio — timeout path
# ---------------------------------------------------------------------------


class TestTranscribeAudioTimeout:
    """transcribe_audio raises TranscribeTimeoutError when deadline is exceeded."""

    @patch("src.lambdas.journal_ingest_lambda._get_transcribe_client")
    @patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None)
    @patch("src.lambdas.journal_ingest_lambda.time.monotonic")
    def test_raises_timeout_when_deadline_exceeded(self, mock_monotonic, mock_sleep, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Simulate: first call returns start time, subsequent calls exceed deadline
        # monotonic() is called: once for deadline calc, then once per loop iteration
        mock_monotonic.side_effect = [
            0.0,   # deadline = 0.0 + 30 = 30.0
            31.0,  # first loop check: 31.0 >= 30.0 → exit loop
        ]

        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}
        }

        with pytest.raises(TranscribeTimeoutError, match="did not complete within"):
            transcribe_audio("audio/slow.mp3", "job-timeout-001")

    @patch("src.lambdas.journal_ingest_lambda._get_transcribe_client")
    @patch("src.lambdas.journal_ingest_lambda.time.sleep", return_value=None)
    @patch("src.lambdas.journal_ingest_lambda.time.monotonic")
    def test_polls_multiple_times_before_timeout(self, mock_monotonic, mock_sleep, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Allow 3 polls before timeout
        mock_monotonic.side_effect = [
            0.0,   # deadline = 30.0
            5.0,   # poll 1: within deadline
            15.0,  # poll 2: within deadline
            35.0,  # poll 3: exceeds deadline → exit
        ]

        mock_client.get_transcription_job.return_value = {
            "TranscriptionJob": {"TranscriptionJobStatus": "IN_PROGRESS"}
        }

        with pytest.raises(TranscribeTimeoutError):
            transcribe_audio("audio/slow.mp3", "job-multi-poll")

        # Should have polled twice (within deadline) before timing out
        assert mock_client.get_transcription_job.call_count == 2

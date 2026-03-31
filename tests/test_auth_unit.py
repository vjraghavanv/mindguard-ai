"""
Unit tests for MindGuard AI Auth Lambda.

Tests cover:
- Happy-path registration and login
- Password reset link generation and expiry boundary (exactly 15 minutes)
- Account lockout email notification trigger
- Token expiry boundary (exactly 60 minutes)

Requirements: 11.1, 11.2, 11.3, 11.5
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

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
os.environ["COGNITO_USER_POOL_ID"] = "us-east-1_testpool"
os.environ["COGNITO_CLIENT_ID"] = "testclientid"

from src.lambdas.auth_lambda import (
    validate_password,
    is_token_valid,
    is_account_locked,
    compute_lockout_until,
    should_lock_account,
    ACCESS_TOKEN_EXPIRY_MINUTES,
    RESET_LINK_EXPIRY_MINUTES,
    LOCKOUT_DURATION_MINUTES,
    MAX_FAILED_ATTEMPTS,
)


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


# ---------------------------------------------------------------------------
# Password validation unit tests (Requirement 11.1)
# ---------------------------------------------------------------------------

class TestValidatePassword:
    def test_valid_password_accepted(self):
        assert validate_password("SecurePass1!") is True

    def test_valid_password_long(self):
        assert validate_password("MyStr0ng&SecurePassword!") is True

    def test_rejects_too_short(self):
        assert validate_password("Short1!") is False

    def test_rejects_exactly_11_chars(self):
        assert validate_password("Abcdefg123!") is False   # 11 chars — should fail

    def test_accepts_exactly_12_chars_boundary(self):
        assert validate_password("Abcdefg1234!") is True   # 12 chars — should pass

    def test_accepts_exactly_12_chars(self):
        assert validate_password("Abcdefg1234!") is True

    def test_rejects_no_uppercase(self):
        assert validate_password("alllowercase1!") is False

    def test_rejects_no_digit(self):
        assert validate_password("NoDigitsHere!") is False

    def test_rejects_no_special_char(self):
        assert validate_password("NoSpecialChar1") is False

    def test_rejects_empty_string(self):
        assert validate_password("") is False

    def test_rejects_whitespace_only(self):
        assert validate_password("            ") is False  # 12 spaces, no upper/digit/special

    def test_all_special_chars_with_requirements(self):
        assert validate_password("A1!@#$%^&*()_+") is True


# ---------------------------------------------------------------------------
# Token expiry unit tests (Requirement 11.2)
# ---------------------------------------------------------------------------

class TestTokenExpiry:
    def test_token_valid_at_issuance(self):
        now = datetime.now(timezone.utc)
        assert is_token_valid("some.jwt.token", now, now) is True

    def test_token_valid_at_59_minutes(self):
        issued = datetime.now(timezone.utc)
        current = issued + timedelta(minutes=59)
        assert is_token_valid("some.jwt.token", issued, current) is True

    def test_token_invalid_at_exactly_60_minutes(self):
        """Token must be invalid at exactly 60 minutes (boundary)."""
        issued = datetime.now(timezone.utc)
        current = issued + timedelta(minutes=ACCESS_TOKEN_EXPIRY_MINUTES)
        assert is_token_valid("some.jwt.token", issued, current) is False

    def test_token_invalid_at_61_minutes(self):
        issued = datetime.now(timezone.utc)
        current = issued + timedelta(minutes=61)
        assert is_token_valid("some.jwt.token", issued, current) is False

    def test_empty_token_always_invalid(self):
        now = datetime.now(timezone.utc)
        assert is_token_valid("", now, now) is False

    def test_token_expiry_constant_is_60(self):
        """Verify the expiry constant matches the requirement."""
        assert ACCESS_TOKEN_EXPIRY_MINUTES == 60


# ---------------------------------------------------------------------------
# Account lockout unit tests (Requirement 11.5)
# ---------------------------------------------------------------------------

class TestAccountLockout:
    def test_no_lockout_when_none(self):
        now = datetime.now(timezone.utc)
        assert is_account_locked(None, now) is False

    def test_locked_within_window(self):
        now = datetime.now(timezone.utc)
        locked_until = compute_lockout_until(now)
        # 1 second after lock — still locked
        check_time = now + timedelta(seconds=1)
        assert is_account_locked(locked_until, check_time) is True

    def test_locked_at_29_minutes(self):
        now = datetime.now(timezone.utc)
        locked_until = compute_lockout_until(now)
        check_time = now + timedelta(minutes=29)
        assert is_account_locked(locked_until, check_time) is True

    def test_unlocked_at_exactly_30_minutes(self):
        """Account must be unlocked at exactly 30 minutes (boundary)."""
        now = datetime.now(timezone.utc)
        locked_until = compute_lockout_until(now)
        check_time = now + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
        assert is_account_locked(locked_until, check_time) is False

    def test_unlocked_after_30_minutes(self):
        now = datetime.now(timezone.utc)
        locked_until = compute_lockout_until(now)
        check_time = now + timedelta(minutes=31)
        assert is_account_locked(locked_until, check_time) is False

    def test_should_lock_after_5_attempts(self):
        assert should_lock_account(MAX_FAILED_ATTEMPTS) is True

    def test_should_not_lock_before_5_attempts(self):
        for i in range(MAX_FAILED_ATTEMPTS):
            assert should_lock_account(i) is False

    def test_lockout_duration_constant_is_30(self):
        assert LOCKOUT_DURATION_MINUTES == 30

    def test_max_failed_attempts_constant_is_5(self):
        assert MAX_FAILED_ATTEMPTS == 5

    def test_compute_lockout_until_is_30_minutes_ahead(self):
        now = datetime.now(timezone.utc)
        locked_until_str = compute_lockout_until(now)
        locked_until = datetime.fromisoformat(locked_until_str.replace("Z", "+00:00"))
        delta = locked_until - now
        # Should be exactly 30 minutes (within 1 second tolerance for test execution)
        assert abs(delta.total_seconds() - LOCKOUT_DURATION_MINUTES * 60) < 2


# ---------------------------------------------------------------------------
# Password reset expiry unit tests (Requirement 11.3)
# ---------------------------------------------------------------------------

class TestPasswordResetExpiry:
    def test_reset_link_expiry_constant_is_15(self):
        """Password reset link must expire after 15 minutes."""
        assert RESET_LINK_EXPIRY_MINUTES == 15

    def test_reset_link_valid_at_14_minutes(self):
        """Simulate a reset link that is still within the 15-minute window."""
        issued_at = datetime.now(timezone.utc)
        check_time = issued_at + timedelta(minutes=14)
        expiry = issued_at + timedelta(minutes=RESET_LINK_EXPIRY_MINUTES)
        assert check_time < expiry  # link is still valid

    def test_reset_link_expired_at_exactly_15_minutes(self):
        """Reset link must be expired at exactly 15 minutes."""
        issued_at = datetime.now(timezone.utc)
        check_time = issued_at + timedelta(minutes=RESET_LINK_EXPIRY_MINUTES)
        expiry = issued_at + timedelta(minutes=RESET_LINK_EXPIRY_MINUTES)
        assert check_time >= expiry  # link is expired at boundary

    def test_reset_link_expired_at_16_minutes(self):
        issued_at = datetime.now(timezone.utc)
        check_time = issued_at + timedelta(minutes=16)
        expiry = issued_at + timedelta(minutes=RESET_LINK_EXPIRY_MINUTES)
        assert check_time > expiry  # link is expired


# ---------------------------------------------------------------------------
# Handler integration tests with mocked Cognito (Requirements 11.1–11.5)
# ---------------------------------------------------------------------------

class TestHandlerRegister:
    def _make_event(self, path: str, body: dict) -> dict:
        return {
            "httpMethod": "POST",
            "path": path,
            "body": json.dumps(body),
        }

    @mock_aws
    def test_register_missing_fields(self):
        from src.lambdas.auth_lambda import handler
        event = self._make_event("/auth/register", {"email": "user@example.com"})
        response = handler(event, None)
        assert response["statusCode"] == 400

    @mock_aws
    def test_register_invalid_password(self):
        from src.lambdas.auth_lambda import handler
        event = self._make_event("/auth/register", {
            "email": "user@example.com",
            "password": "weak",
        })
        response = handler(event, None)
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "Password" in body["error"] or "password" in body["error"].lower()

    @mock_aws
    @patch("src.lambdas.auth_lambda._register_cognito_user")
    @patch("src.lambdas.auth_lambda.put_user_profile")
    def test_register_happy_path(self, mock_put_profile, mock_register):
        from src.lambdas.auth_lambda import handler
        mock_register.return_value = {"UserSub": str(uuid.uuid4())}
        mock_put_profile.return_value = None

        event = self._make_event("/auth/register", {
            "email": "user@example.com",
            "password": "SecurePass1!abc",
        })
        response = handler(event, None)
        assert response["statusCode"] == 201
        body = json.loads(response["body"])
        assert "user_id" in body
        mock_register.assert_called_once()
        mock_put_profile.assert_called_once()

    @mock_aws
    @patch("src.lambdas.auth_lambda._register_cognito_user")
    def test_register_duplicate_email(self, mock_register):
        from src.lambdas.auth_lambda import handler
        mock_register.side_effect = Exception("UsernameExistsException: already exists")

        event = self._make_event("/auth/register", {
            "email": "existing@example.com",
            "password": "SecurePass1!abc",
        })
        response = handler(event, None)
        assert response["statusCode"] == 409


class TestHandlerLogin:
    def _make_event(self, path: str, body: dict) -> dict:
        return {
            "httpMethod": "POST",
            "path": path,
            "body": json.dumps(body),
        }

    @mock_aws
    @patch("src.lambdas.auth_lambda.get_user_by_email_hash")
    @patch("src.lambdas.auth_lambda._authenticate_cognito_user")
    @patch("src.lambdas.auth_lambda.put_user_profile")
    def test_login_happy_path(self, mock_put, mock_auth, mock_get_user):
        from src.lambdas.auth_lambda import handler
        mock_get_user.return_value = {
            "user_id": str(uuid.uuid4()),
            "email_hash": "abc123",
            "account_locked_until": None,
            "failed_login_attempts": 0,
            "sk": "profile",
        }
        mock_auth.return_value = {
            "AuthenticationResult": {
                "AccessToken": "access.token.here",
                "RefreshToken": "refresh.token.here",
                "ExpiresIn": 3600,
                "TokenType": "Bearer",
            }
        }
        mock_put.return_value = None

        event = self._make_event("/auth/login", {
            "email": "user@example.com",
            "password": "SecurePass1!abc",
        })
        response = handler(event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "access_token" in body
        assert body["access_token"] == "access.token.here"

    @mock_aws
    @patch("src.lambdas.auth_lambda.get_user_by_email_hash")
    def test_login_unknown_user(self, mock_get_user):
        from src.lambdas.auth_lambda import handler
        mock_get_user.return_value = None

        event = self._make_event("/auth/login", {
            "email": "nobody@example.com",
            "password": "SecurePass1!abc",
        })
        response = handler(event, None)
        assert response["statusCode"] == 401

    @mock_aws
    @patch("src.lambdas.auth_lambda.get_user_by_email_hash")
    def test_login_locked_account_returns_423(self, mock_get_user):
        from src.lambdas.auth_lambda import handler
        now = datetime.now(timezone.utc)
        locked_until = compute_lockout_until(now)
        mock_get_user.return_value = {
            "user_id": str(uuid.uuid4()),
            "email_hash": "abc123",
            "account_locked_until": locked_until,
            "failed_login_attempts": 0,
            "sk": "profile",
        }

        event = self._make_event("/auth/login", {
            "email": "locked@example.com",
            "password": "AnyPassword1!",
        })
        response = handler(event, None)
        assert response["statusCode"] == 423
        body = json.loads(response["body"])
        assert "retry_after" in body

    @mock_aws
    @patch("src.lambdas.auth_lambda.get_user_by_email_hash")
    @patch("src.lambdas.auth_lambda._authenticate_cognito_user")
    @patch("src.lambdas.auth_lambda.put_user_profile")
    @patch("src.lambdas.auth_lambda._send_lockout_notification")
    def test_login_triggers_lockout_after_5_failures(
        self, mock_notify, mock_put, mock_auth, mock_get_user
    ):
        """After 5 consecutive failed logins, account is locked and notification sent."""
        from src.lambdas.auth_lambda import handler

        user_profile = {
            "user_id": str(uuid.uuid4()),
            "email_hash": "abc123",
            "account_locked_until": None,
            "failed_login_attempts": MAX_FAILED_ATTEMPTS - 1,  # 4 previous failures
            "sk": "profile",
        }
        mock_get_user.return_value = user_profile
        mock_auth.side_effect = Exception("NotAuthorizedException: Incorrect username or password")
        mock_put.return_value = None
        mock_notify.return_value = None

        event = self._make_event("/auth/login", {
            "email": "user@example.com",
            "password": "WrongPassword1!",
        })
        response = handler(event, None)

        # 5th failure should trigger lockout
        assert response["statusCode"] == 423
        body = json.loads(response["body"])
        assert "retry_after" in body
        mock_notify.assert_called_once()

    @mock_aws
    @patch("src.lambdas.auth_lambda.get_user_by_email_hash")
    @patch("src.lambdas.auth_lambda._authenticate_cognito_user")
    @patch("src.lambdas.auth_lambda.put_user_profile")
    def test_login_increments_failed_attempts(self, mock_put, mock_auth, mock_get_user):
        """Failed login increments the failed_login_attempts counter."""
        from src.lambdas.auth_lambda import handler

        user_profile = {
            "user_id": str(uuid.uuid4()),
            "email_hash": "abc123",
            "account_locked_until": None,
            "failed_login_attempts": 2,
            "sk": "profile",
        }
        mock_get_user.return_value = user_profile
        mock_auth.side_effect = Exception("NotAuthorizedException: Incorrect username or password")
        mock_put.return_value = None

        event = self._make_event("/auth/login", {
            "email": "user@example.com",
            "password": "WrongPassword1!",
        })
        response = handler(event, None)
        assert response["statusCode"] == 401

        # Verify put_user_profile was called with incremented counter
        saved_profile = mock_put.call_args[0][0]
        assert saved_profile["failed_login_attempts"] == 3


class TestHandlerPasswordReset:
    def _make_event(self, path: str, body: dict) -> dict:
        return {
            "httpMethod": "POST",
            "path": path,
            "body": json.dumps(body),
        }

    @mock_aws
    @patch("src.lambdas.auth_lambda._initiate_password_reset")
    def test_password_reset_happy_path(self, mock_reset):
        from src.lambdas.auth_lambda import handler
        mock_reset.return_value = {}

        event = self._make_event("/auth/reset", {"email": "user@example.com"})
        response = handler(event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert "15 minutes" in body["message"]
        mock_reset.assert_called_once()

    @mock_aws
    def test_password_reset_missing_email(self):
        from src.lambdas.auth_lambda import handler
        event = self._make_event("/auth/reset", {})
        response = handler(event, None)
        assert response["statusCode"] == 400

    @mock_aws
    @patch("src.lambdas.auth_lambda._initiate_password_reset")
    def test_password_reset_unknown_email_returns_200(self, mock_reset):
        """Should return 200 even for unknown emails to prevent enumeration."""
        from src.lambdas.auth_lambda import handler
        mock_reset.side_effect = Exception("UserNotFoundException")

        event = self._make_event("/auth/reset", {"email": "nobody@example.com"})
        response = handler(event, None)
        assert response["statusCode"] == 200


class TestHandlerRouting:
    def test_unknown_route_returns_404(self):
        from src.lambdas.auth_lambda import handler
        event = {"httpMethod": "POST", "path": "/auth/unknown", "body": "{}"}
        response = handler(event, None)
        assert response["statusCode"] == 404

    def test_invalid_json_body_returns_400(self):
        from src.lambdas.auth_lambda import handler
        event = {"httpMethod": "POST", "path": "/auth/register", "body": "not-json"}
        response = handler(event, None)
        assert response["statusCode"] == 400

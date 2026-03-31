"""
Auth Lambda for MindGuard AI — handles register, login, token refresh, and password reset.

Integrates with Amazon Cognito for user management and JWT token issuance.
Account lockout state is stored in DynamoDB (UserProfile.account_locked_until).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import string
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCKOUT_DURATION_MINUTES = 30
MAX_FAILED_ATTEMPTS = 5
ACCESS_TOKEN_EXPIRY_MINUTES = 60
RESET_LINK_EXPIRY_MINUTES = 15

SPECIAL_CHARACTERS = set(string.punctuation)

DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "mindguard-trend-store")
COGNITO_USER_POOL_ID = os.environ.get("COGNITO_USER_POOL_ID", "")
COGNITO_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------

def validate_password(password: str) -> bool:
    """
    Validate password complexity requirements.

    Rules:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one digit
    - At least one special character (punctuation)

    Returns True if valid, False otherwise.
    """
    if len(password) < 12:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    if not any(c in SPECIAL_CHARACTERS for c in password):
        return False
    return True


# ---------------------------------------------------------------------------
# JWT token validation
# ---------------------------------------------------------------------------

def is_token_valid(token: str, issued_at: datetime, current_time: datetime) -> bool:
    """
    Check whether a JWT access token is still valid based on its issuance time.

    A token is valid if current_time is strictly before issued_at + 60 minutes.
    The token string itself is treated as opaque here; expiry is enforced by time.

    Args:
        token: The JWT token string (must be non-empty).
        issued_at: UTC datetime when the token was issued.
        current_time: UTC datetime representing "now".

    Returns:
        True if the token is within its 60-minute validity window, False otherwise.
    """
    if not token:
        return False
    expiry = issued_at + timedelta(minutes=ACCESS_TOKEN_EXPIRY_MINUTES)
    return current_time < expiry


# ---------------------------------------------------------------------------
# Account lockout logic
# ---------------------------------------------------------------------------

def is_account_locked(account_locked_until: Optional[str], current_time: datetime) -> bool:
    """
    Determine whether an account is currently locked.

    Args:
        account_locked_until: ISO-8601 UTC string from UserProfile, or None.
        current_time: UTC datetime representing "now".

    Returns:
        True if the account is locked, False otherwise.
    """
    if account_locked_until is None:
        return False
    try:
        lock_expiry = datetime.fromisoformat(account_locked_until.replace("Z", "+00:00"))
        # Ensure current_time is timezone-aware for comparison
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        return current_time < lock_expiry
    except (ValueError, AttributeError):
        return False


def compute_lockout_until(current_time: datetime) -> str:
    """Return an ISO-8601 UTC string for 30 minutes from current_time."""
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    lockout_until = current_time + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
    return lockout_until.strftime("%Y-%m-%dT%H:%M:%SZ")


def should_lock_account(failed_attempts: int) -> bool:
    """Return True if the number of consecutive failed attempts triggers a lockout."""
    return failed_attempts >= MAX_FAILED_ATTEMPTS


# ---------------------------------------------------------------------------
# DynamoDB helpers for UserProfile
# ---------------------------------------------------------------------------

def _get_dynamodb_table():
    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return dynamodb.Table(DYNAMODB_TABLE)


def _hash_email(email: str) -> str:
    """Return a SHA-256 hash of the email address (no PII stored)."""
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()


def get_user_profile(user_id: str) -> Optional[dict]:
    """Retrieve a UserProfile item from DynamoDB."""
    table = _get_dynamodb_table()
    response = table.get_item(Key={"user_id": user_id, "sk": "profile"})
    return response.get("Item")


def put_user_profile(profile: dict) -> None:
    """Write a UserProfile item to DynamoDB."""
    table = _get_dynamodb_table()
    item = {k: v for k, v in profile.items() if v is not None}
    item["sk"] = "profile"
    table.put_item(Item=item)


def get_user_by_email_hash(email_hash: str) -> Optional[dict]:
    """Scan for a user profile by email_hash (used during login)."""
    table = _get_dynamodb_table()
    response = table.scan(
        FilterExpression=boto3.dynamodb.conditions.Attr("email_hash").eq(email_hash)
        & boto3.dynamodb.conditions.Attr("sk").eq("profile")
    )
    items = response.get("Items", [])
    return items[0] if items else None


# ---------------------------------------------------------------------------
# Cognito helpers
# ---------------------------------------------------------------------------

def _get_cognito_client():
    return boto3.client("cognito-idp", region_name=AWS_REGION)


def _register_cognito_user(email: str, password: str) -> dict:
    """Register a new user in Cognito User Pool."""
    client = _get_cognito_client()
    return client.sign_up(
        ClientId=COGNITO_CLIENT_ID,
        Username=email,
        Password=password,
        UserAttributes=[{"Name": "email", "Value": email}],
    )


def _authenticate_cognito_user(email: str, password: str) -> dict:
    """Authenticate a user via Cognito and return auth tokens."""
    client = _get_cognito_client()
    return client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": email, "PASSWORD": password},
        ClientId=COGNITO_CLIENT_ID,
    )


def _initiate_password_reset(email: str) -> dict:
    """Trigger Cognito forgot-password flow (sends reset link via email)."""
    client = _get_cognito_client()
    return client.forgot_password(
        ClientId=COGNITO_CLIENT_ID,
        Username=email,
    )


def _send_lockout_notification(email: str, retry_after: str) -> None:
    """Send an email notification to the user about account lockout via SNS."""
    sns_topic_arn = os.environ.get("LOCKOUT_SNS_TOPIC_ARN", "")
    if not sns_topic_arn:
        return
    sns = boto3.client("sns", region_name=AWS_REGION)
    sns.publish(
        TopicArn=sns_topic_arn,
        Message=json.dumps({
            "email": email,
            "message": "Your MindGuard AI account has been locked due to 5 consecutive failed login attempts.",
            "retry_after": retry_after,
        }),
        Subject="MindGuard AI: Account Locked",
    )


# ---------------------------------------------------------------------------
# Lambda handler
# ---------------------------------------------------------------------------

def handler(event: dict, context) -> dict:
    """
    Main Lambda handler for auth operations.

    Routes:
      POST /auth/register   — register a new user
      POST /auth/login      — authenticate and return tokens
      POST /auth/refresh    — refresh access token
      POST /auth/reset      — request password reset link
    """
    http_method = event.get("httpMethod", "POST")
    path = event.get("path", "")
    body = {}
    if event.get("body"):
        try:
            body = json.loads(event["body"])
        except (json.JSONDecodeError, TypeError):
            return _response(400, {"error": "Invalid JSON body"})

    if path.endswith("/register"):
        return _handle_register(body)
    elif path.endswith("/login"):
        return _handle_login(body)
    elif path.endswith("/refresh"):
        return _handle_refresh(body)
    elif path.endswith("/reset"):
        return _handle_password_reset(body)
    else:
        return _response(404, {"error": "Route not found"})


def _handle_register(body: dict) -> dict:
    """Register a new user account."""
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        return _response(400, {"error": "email and password are required"})

    if not validate_password(password):
        return _response(400, {
            "error": (
                "Password must be at least 12 characters and contain "
                "at least one uppercase letter, one digit, and one special character."
            )
        })

    try:
        _register_cognito_user(email, password)
    except Exception as e:
        error_msg = str(e)
        if "UsernameExistsException" in error_msg:
            return _response(409, {"error": "An account with this email already exists."})
        return _response(500, {"error": f"Registration failed: {error_msg}"})

    # Create UserProfile in DynamoDB (no PII — only email_hash)
    user_id = str(uuid.uuid4())
    profile = {
        "user_id": user_id,
        "email_hash": _hash_email(email),
        "notification_prefs": {
            "channel": "in_app",
            "nudge_time": "09:00",
            "enabled": True,
            "snooze_until": None,
        },
        "trusted_contact": {"name": "", "contact": ""},
        "escalation_threshold": 80,
        "account_locked_until": None,
        "failed_login_attempts": 0,
    }
    put_user_profile(profile)

    return _response(201, {"message": "Registration successful.", "user_id": user_id})


def _handle_login(body: dict) -> dict:
    """Authenticate a user and return JWT tokens."""
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        return _response(400, {"error": "email and password are required"})

    email_hash = _hash_email(email)
    profile = get_user_by_email_hash(email_hash)

    if profile is None:
        return _response(401, {"error": "Invalid credentials."})

    # Check account lockout
    now = datetime.now(timezone.utc)
    if is_account_locked(profile.get("account_locked_until"), now):
        retry_after = profile["account_locked_until"]
        return _response(423, {
            "error": "Account is temporarily locked due to too many failed login attempts.",
            "retry_after": retry_after,
        })

    # Attempt Cognito authentication
    try:
        auth_result = _authenticate_cognito_user(email, password)
        tokens = auth_result["AuthenticationResult"]

        # Reset failed attempts on success
        profile["failed_login_attempts"] = 0
        profile["account_locked_until"] = None
        put_user_profile(profile)

        return _response(200, {
            "access_token": tokens["AccessToken"],
            "refresh_token": tokens.get("RefreshToken"),
            "expires_in": tokens.get("ExpiresIn", ACCESS_TOKEN_EXPIRY_MINUTES * 60),
            "token_type": tokens.get("TokenType", "Bearer"),
        })

    except Exception as e:
        error_msg = str(e)
        if "NotAuthorizedException" in error_msg or "UserNotFoundException" in error_msg:
            # Increment failed attempts
            failed_attempts = int(profile.get("failed_login_attempts", 0)) + 1
            profile["failed_login_attempts"] = failed_attempts

            if should_lock_account(failed_attempts):
                retry_after = compute_lockout_until(now)
                profile["account_locked_until"] = retry_after
                profile["failed_login_attempts"] = 0
                put_user_profile(profile)
                _send_lockout_notification(email, retry_after)
                return _response(423, {
                    "error": "Account locked due to too many failed login attempts.",
                    "retry_after": retry_after,
                })

            put_user_profile(profile)
            return _response(401, {"error": "Invalid credentials."})

        return _response(500, {"error": f"Authentication failed: {error_msg}"})


def _handle_refresh(body: dict) -> dict:
    """Refresh an access token using a refresh token."""
    refresh_token = body.get("refresh_token", "")
    if not refresh_token:
        return _response(400, {"error": "refresh_token is required"})

    try:
        client = _get_cognito_client()
        result = client.initiate_auth(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": refresh_token},
            ClientId=COGNITO_CLIENT_ID,
        )
        tokens = result["AuthenticationResult"]
        return _response(200, {
            "access_token": tokens["AccessToken"],
            "expires_in": tokens.get("ExpiresIn", ACCESS_TOKEN_EXPIRY_MINUTES * 60),
            "token_type": tokens.get("TokenType", "Bearer"),
        })
    except Exception as e:
        return _response(401, {"error": f"Token refresh failed: {str(e)}"})


def _handle_password_reset(body: dict) -> dict:
    """Initiate a password reset — sends a reset link expiring in 15 minutes."""
    email = body.get("email", "").strip().lower()
    if not email:
        return _response(400, {"error": "email is required"})

    try:
        _initiate_password_reset(email)
    except Exception as e:
        # Return success even on error to avoid email enumeration
        pass

    return _response(200, {
        "message": (
            "If an account with that email exists, a password reset link "
            f"(valid for {RESET_LINK_EXPIRY_MINUTES} minutes) has been sent."
        )
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }

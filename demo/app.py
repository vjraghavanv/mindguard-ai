"""
MindGuard AI — Local Demo Server
Runs the full journal pipeline locally with mocked AWS services (moto).
Start with: python3 demo/app.py
Then open:  http://localhost:5001
"""
import json
import os
import sys
import uuid
from datetime import datetime, timezone

# ── Fake AWS creds so boto3/moto don't complain ──────────────────────────────
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "demo")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "demo")
os.environ.setdefault("AWS_SECURITY_TOKEN", "demo")
os.environ.setdefault("AWS_SESSION_TOKEN", "demo")
os.environ["DYNAMODB_TABLE"] = "mindguard-demo"
os.environ["AUDIO_S3_BUCKET"] = "mindguard-demo-audio"

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moto import mock_aws
import boto3

# ── Start moto mock BEFORE importing src modules ─────────────────────────────
_mock = mock_aws()
_mock.start()

# ── Create mocked DynamoDB table ─────────────────────────────────────────────
_ddb = boto3.resource("dynamodb", region_name="us-east-1")
_ddb.create_table(
    TableName="mindguard-demo",
    KeySchema=[
        {"AttributeName": "user_id", "KeyType": "HASH"},
        {"AttributeName": "sk",      "KeyType": "RANGE"},
    ],
    AttributeDefinitions=[
        {"AttributeName": "user_id", "AttributeType": "S"},
        {"AttributeName": "sk",      "AttributeType": "S"},
    ],
    BillingMode="PAY_PER_REQUEST",
)

# ── Create mocked S3 bucket ───────────────────────────────────────────────────
_s3 = boto3.client("s3", region_name="us-east-1")
_s3.create_bucket(Bucket="mindguard-demo-audio")

# ── Create mocked SNS topic ───────────────────────────────────────────────────
_sns = boto3.client("sns", region_name="us-east-1")
_topic = _sns.create_topic(Name="mindguard-burnout-alert")
os.environ["SNS_BURNOUT_ALERT_TOPIC_ARN"] = _topic["TopicArn"]

# ── Now import src modules (they'll use the mocked clients) ──────────────────
from src.lambdas.journal_ingest_lambda import handler as journal_handler
from src.lambdas.auth_lambda import (
    validate_password, _hash_email, put_user_profile, get_user_by_email_hash
)
from src.utils.dynamodb import query_by_user

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("\n❌  Flask not found. Run:  pip3 install flask\n")
    sys.exit(1)

app = Flask(__name__, static_folder="static")

# ── In-memory user store for demo (no real Cognito) ──────────────────────────
_users: dict = {}   # email → {user_id, password_hash}

# ── Serve the frontend ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")

# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def register():
    body = request.get_json() or {}
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    if not validate_password(password):
        return jsonify({"error": "Password needs 12+ chars, uppercase, digit, special char"}), 400
    if email in _users:
        return jsonify({"error": "Email already registered"}), 409
    user_id = str(uuid.uuid4())
    _users[email] = {"user_id": user_id, "password": password}
    profile = {
        "user_id": user_id,
        "email_hash": _hash_email(email),
        "notification_prefs": {"channel": "in_app", "enabled": True},
        "trusted_contact": {"name": "", "contact": ""},
        "escalation_threshold": 80,
        "account_locked_until": None,
        "failed_login_attempts": 0,
    }
    put_user_profile(profile)
    return jsonify({"message": "Registered!", "user_id": user_id}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    body = request.get_json() or {}
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")
    user = _users.get(email)
    if not user or user["password"] != password:
        return jsonify({"error": "Invalid credentials"}), 401
    return jsonify({
        "access_token": f"demo-token-{user['user_id']}",
        "user_id": user["user_id"],
        "message": "Login successful"
    }), 200

# ── Journal endpoint ──────────────────────────────────────────────────────────
@app.route("/journal", methods=["POST"])
def journal():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("demo-token-"):
        return jsonify({"error": "Unauthorized"}), 401
    user_id = auth.replace("demo-token-", "")

    body = request.get_json() or {}
    event = {
        "body": json.dumps(body),
        "requestContext": {"authorizer": {"claims": {"sub": user_id}}},
    }
    result = journal_handler(event, None)
    return jsonify(json.loads(result["body"])), result["statusCode"]

# ── History endpoint ──────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("demo-token-"):
        return jsonify({"error": "Unauthorized"}), 401
    user_id = auth.replace("demo-token-", "")
    items = query_by_user(user_id)
    entries = [i for i in items if i.get("entry_type") in ("text", "voice")]
    entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jsonify({"entries": entries[:20]}), 200

if __name__ == "__main__":
    print("\n🧠  MindGuard AI Demo  →  http://localhost:5001\n")
    app.run(port=5001, debug=False)

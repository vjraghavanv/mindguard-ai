"""
Data models for MindGuard AI.
All models use anonymized user_id (UUID) — no PII stored in Trend_Store.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------

@dataclass
class Emotions:
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Emotions":
        return cls(
            joy=float(d.get("joy", 0.0)),
            sadness=float(d.get("sadness", 0.0)),
            anger=float(d.get("anger", 0.0)),
            fear=float(d.get("fear", 0.0)),
            disgust=float(d.get("disgust", 0.0)),
        )


@dataclass
class NotificationPrefs:
    channel: str = "in_app"   # "in_app" | "push" | "sms"
    nudge_time: str = "09:00"  # HH:MM UTC
    enabled: bool = True
    snooze_until: Optional[str] = None  # ISO-8601 UTC or None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationPrefs":
        return cls(
            channel=d.get("channel", "in_app"),
            nudge_time=d.get("nudge_time", "09:00"),
            enabled=bool(d.get("enabled", True)),
            snooze_until=d.get("snooze_until"),
        )


@dataclass
class TrustedContact:
    name: str = ""
    contact: str = ""  # phone or email

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrustedContact":
        return cls(name=d.get("name", ""), contact=d.get("contact", ""))


@dataclass
class SentimentDistribution:
    POSITIVE: float = 0.0
    NEGATIVE: float = 0.0
    NEUTRAL: float = 0.0
    MIXED: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SentimentDistribution":
        return cls(
            POSITIVE=float(d.get("POSITIVE", 0.0)),
            NEGATIVE=float(d.get("NEGATIVE", 0.0)),
            NEUTRAL=float(d.get("NEUTRAL", 0.0)),
            MIXED=float(d.get("MIXED", 0.0)),
        )


# ---------------------------------------------------------------------------
# Primary models
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """Represents a single journal entry stored in the Trend_Store."""
    user_id: str                    # anonymized UUID (PK)
    timestamp: str                  # ISO-8601 UTC (SK prefix)
    entry_id: str                   # UUID
    entry_type: str                 # "voice" | "text"
    text_content: str               # transcribed or raw text
    sentiment_label: str            # POSITIVE | NEGATIVE | NEUTRAL | MIXED
    sentiment_score: float          # 0.0 – 1.0
    emotions: Emotions              # per-emotion confidence scores
    burnout_score: int              # 0 – 100
    coping_suggestion: str
    created_at: str                 # UTC timestamp
    audio_s3_key: Optional[str] = None  # voice entries only

    # DynamoDB sort key: "timestamp#entry_id"
    @property
    def sort_key(self) -> str:
        return f"{self.timestamp}#{self.entry_id}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["emotions"] = self.emotions.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        return cls(
            user_id=d["user_id"],
            timestamp=d["timestamp"],
            entry_id=d["entry_id"],
            entry_type=d["entry_type"],
            text_content=d["text_content"],
            sentiment_label=d["sentiment_label"],
            sentiment_score=float(d["sentiment_score"]),
            emotions=Emotions.from_dict(d.get("emotions", {})),
            burnout_score=int(d["burnout_score"]),
            coping_suggestion=d["coping_suggestion"],
            created_at=d["created_at"],
            audio_s3_key=d.get("audio_s3_key"),
        )


@dataclass
class BurnoutScoreRecord:
    """Tracks each burnout score computation event."""
    user_id: str
    timestamp: str
    burnout_score: int
    trigger: str  # "journal_entry" | "scheduled_recompute"

    @property
    def sort_key(self) -> str:
        return f"{self.timestamp}#burnout"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BurnoutScoreRecord":
        return cls(
            user_id=d["user_id"],
            timestamp=d["timestamp"],
            burnout_score=int(d["burnout_score"]),
            trigger=d["trigger"],
        )


@dataclass
class UserProfile:
    """User account and preferences — stored separately from emotional data."""
    user_id: str                          # anonymized UUID
    email_hash: str                       # hashed, NOT plaintext
    notification_prefs: NotificationPrefs = field(default_factory=NotificationPrefs)
    trusted_contact: TrustedContact = field(default_factory=TrustedContact)
    escalation_threshold: int = 80
    account_locked_until: Optional[str] = None  # ISO-8601 UTC or None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notification_prefs"] = self.notification_prefs.to_dict()
        d["trusted_contact"] = self.trusted_contact.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(
            user_id=d["user_id"],
            email_hash=d["email_hash"],
            notification_prefs=NotificationPrefs.from_dict(d.get("notification_prefs", {})),
            trusted_contact=TrustedContact.from_dict(d.get("trusted_contact", {})),
            escalation_threshold=int(d.get("escalation_threshold", 80)),
            account_locked_until=d.get("account_locked_until"),
        )


@dataclass
class EmotionalHealthReport:
    """Weekly emotional health summary for a user."""
    user_id: str
    report_id: str
    week_start: str                          # ISO-8601 UTC
    week_end: str
    sentiment_distribution: SentimentDistribution = field(default_factory=SentimentDistribution)
    avg_burnout_score: float = 0.0
    top_emotions: list = field(default_factory=list)
    prior_week_avg_burnout: float = 0.0
    ai_insights: list = field(default_factory=list)  # min 2 items
    generated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sentiment_distribution"] = self.sentiment_distribution.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalHealthReport":
        return cls(
            user_id=d["user_id"],
            report_id=d["report_id"],
            week_start=d["week_start"],
            week_end=d["week_end"],
            sentiment_distribution=SentimentDistribution.from_dict(d.get("sentiment_distribution", {})),
            avg_burnout_score=float(d.get("avg_burnout_score", 0.0)),
            top_emotions=list(d.get("top_emotions", [])),
            prior_week_avg_burnout=float(d.get("prior_week_avg_burnout", 0.0)),
            ai_insights=list(d.get("ai_insights", [])),
            generated_at=d.get("generated_at", ""),
        )


@dataclass
class EscalationEvent:
    """Records each escalation alert triggered for a user."""
    user_id: str
    timestamp: str
    burnout_score: int
    escalation_threshold: int
    contact_notified: bool
    cancelled: bool
    cancelled_at: Optional[str] = None  # ISO-8601 UTC or None

    @property
    def sort_key(self) -> str:
        return f"{self.timestamp}#escalation"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EscalationEvent":
        return cls(
            user_id=d["user_id"],
            timestamp=d["timestamp"],
            burnout_score=int(d["burnout_score"]),
            escalation_threshold=int(d["escalation_threshold"]),
            contact_notified=bool(d["contact_notified"]),
            cancelled=bool(d["cancelled"]),
            cancelled_at=d.get("cancelled_at"),
        )

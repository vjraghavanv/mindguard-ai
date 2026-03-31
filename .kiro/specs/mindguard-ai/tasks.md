# Implementation Plan: MindGuard AI

## Overview

Implement MindGuard AI as a serverless Python application on AWS. Tasks follow the processing pipeline order: auth → journal ingest → sentiment analysis → burnout scoring → notifications → reporting → nudges. Each task builds on the previous, ending with full integration.

## Tasks

- [x] 1. Set up project structure, data models, and DynamoDB schema
  - Create directory layout: `src/models/`, `src/lambdas/`, `src/utils/`, `tests/`
  - Define Python dataclasses for `JournalEntry`, `BurnoutScoreRecord`, `UserProfile`, `EmotionalHealthReport`, `EscalationEvent`
  - Implement DynamoDB single-table helpers: `put_item`, `get_item`, `query_by_user` with composite key `user_id` / `timestamp#entry_id`
  - Add serialization/deserialization methods for each model
  - _Requirements: 2.3, 3.3, 4.3, 9.1, 9.3, 10.2, 10.3_

  - [x] 1.1 Write property test for journal entry storage round-trip
    - **Property 4: Journal Entry Storage Round-Trip**
    - **Validates: Requirements 2.3, 3.3, 4.3, 10.2, 10.3**

  - [x] 1.2 Write property test for no PII in Trend_Store records
    - **Property 17: No PII in Trend_Store**
    - **Validates: Requirements 9.3**

- [x] 2. Implement Auth Lambda with Amazon Cognito
  - Create `src/lambdas/auth_lambda.py` handling register, login, token refresh, and password reset
  - Enforce password complexity: min 12 chars, 1 uppercase, 1 digit, 1 special character
  - Implement account lockout: lock for 30 minutes after 5 consecutive failed logins; return HTTP 423 with `retry_after`
  - Issue JWT access tokens with 60-minute expiry via Cognito; send password reset link expiring after 15 minutes
  - _Requirements: 11.1, 11.2, 11.3, 11.5_

  - [x] 2.1 Write property test for password complexity enforcement
    - **Property 20: Password Complexity Enforcement**
    - **Validates: Requirements 11.1**

  - [x] 2.2 Write property test for JWT token expiry
    - **Property 21: Token Expiry**
    - **Validates: Requirements 11.2**

  - [x] 2.3 Write property test for account lockout
    - **Property 22: Account Lockout**
    - **Validates: Requirements 11.5**

  - [x] 2.4 Write unit tests for Auth Lambda
    - Test happy-path registration and login
    - Test password reset link generation and expiry boundary (exactly 15 minutes)
    - Test lockout email notification trigger
    - _Requirements: 11.1, 11.2, 11.3, 11.5_

- [x] 3. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement input validation and text journal ingest
  - Create `src/lambdas/journal_ingest_lambda.py`
  - Validate text entries: reject empty/whitespace (HTTP 400), reject >5,000 chars (HTTP 413)
  - Accept and store text `JournalEntry` to DynamoDB with UTC timestamp and anonymized `user_id`
  - Wire API Gateway JSON route to the Lambda handler
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 4.1 Write property test for text entry validation
    - **Property 3: Text Entry Validation**
    - **Validates: Requirements 2.1, 2.2, 2.4**

  - [x] 4.2 Write unit tests for text journal ingest
    - Test happy path, empty entry, whitespace-only entry, exactly 5,000 chars, 5,001 chars
    - _Requirements: 2.1, 2.2, 2.4_

- [x] 5. Implement voice journal ingest with Amazon Transcribe
  - Extend `journal_ingest_lambda.py` to handle multipart audio uploads
  - Validate audio format (MP3, WAV, M4A) and duration (≤10 minutes); return HTTP 413 for violations
  - Upload audio to S3 (encrypted); start Transcribe job; poll with exponential backoff; timeout at 30 seconds
  - On Transcribe failure: log error, retain audio in S3, return 202 with `retry_available: true`, notify user
  - Pass transcript text to the sentiment analysis step
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 5.1 Write property test for audio format and duration validation
    - **Property 2: Audio Format Acceptance**
    - **Validates: Requirements 1.4, 1.5**

  - [x] 5.2 Write property test for voice pipeline round-trip
    - **Property 1: Voice Entry Pipeline Round-Trip**
    - **Validates: Requirements 1.1, 1.2**

  - [x] 5.3 Write unit tests for voice ingest
    - Test each accepted format, rejected format, duration boundary, Transcribe timeout path
    - _Requirements: 1.1, 1.3, 1.4, 1.5_

- [x] 6. Implement sentiment and emotion analysis with Amazon Comprehend
  - Create `src/utils/sentiment.py` with `analyze_sentiment(text) -> dict`
  - Call `DetectSentiment` and return label (POSITIVE/NEGATIVE/NEUTRAL/MIXED), confidence score in [0.0, 1.0], and per-emotion scores for joy, sadness, anger, fear, disgust
  - On Comprehend error: log, store raw entry without analysis fields, retry up to 3 times (backoff 1s/2s/4s); after 3 failures set `analysis_status: "failed"`
  - Store sentiment label, emotion scores, and confidence in DynamoDB alongside the entry
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 6.1 Write property test for sentiment analysis completeness
    - **Property 5: Sentiment Analysis Completeness**
    - **Validates: Requirements 3.1, 3.2**

  - [x] 6.2 Write unit tests for sentiment analysis
    - Test all four sentiment labels, error retry logic, storage of raw entry on failure
    - _Requirements: 3.1, 3.2, 3.4_

- [x] 7. Implement burnout scoring and coping suggestions with Amazon Bedrock
  - Create `src/utils/bedrock.py` with `invoke_bedrock(text, sentiment, trends) -> dict`
  - Build structured Claude prompt with 30-day trend data, current sentiment, and journaling gap history
  - Parse response JSON for `burnout_score` (0–100) and `coping_suggestion`
  - Factor in: frequency of negative sentiment, stress/fatigue emotion intensity, journaling gaps >5 consecutive days
  - Implement fallback: rule-based score (average of last 7 days) + static coping library on Bedrock error
  - Enforce coping suggestion deduplication: do not repeat the same suggestion within 48 hours per user
  - Store `BurnoutScoreRecord` in DynamoDB with UTC timestamp and trigger type
  - _Requirements: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 5.5_

  - [x] 7.1 Write property test for burnout score monotonicity
    - **Property 6: Burnout Score Monotonicity**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 7.2 Write property test for coping suggestion generation
    - **Property 8: Coping Suggestion Generation**
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [x] 7.3 Write property test for coping suggestion deduplication
    - **Property 9: Coping Suggestion Deduplication**
    - **Validates: Requirements 5.5**

  - [x] 7.4 Write unit tests for Bedrock integration
    - Test prompt construction, JSON parsing, fallback path, deduplication boundary (exactly 48 hours)
    - _Requirements: 4.1, 4.2, 5.1, 5.5_

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Notification Service (SNS + Pinpoint)
  - Create `src/utils/notifications.py` with `send_notification(user_id, message, notification_type)`
  - Route notifications to user's preferred channel (in-app, push, SMS) from `UserProfile.notification_prefs`
  - Gate all outbound notifications: skip if `enabled=False`; skip nudges if `snooze_until` is in the future
  - Publish burnout alert to SNS when `burnout_score > 70`
  - Publish escalation alert to Trusted_Contact when score exceeds `escalation_threshold`; record `EscalationEvent` in DynamoDB
  - Present crisis helpline option in response payload when `burnout_score > 85`
  - Implement 60-second cancellation window for escalation alerts using a delayed SNS publish + cancellation flag in DynamoDB
  - _Requirements: 4.4, 5.4, 6.4, 6.5, 7.1, 7.2, 7.4, 7.6_

  - [x] 9.1 Write property test for burnout alert threshold
    - **Property 7: Burnout Alert Threshold**
    - **Validates: Requirements 4.4**

  - [x] 9.2 Write property test for notification channel routing
    - **Property 10: Notification Channel Routing**
    - **Validates: Requirements 5.4, 6.3**

  - [x] 9.3 Write property test for notification gating
    - **Property 11: Notification Gating**
    - **Validates: Requirements 6.4, 6.5**

  - [x] 9.4 Write property test for escalation and event recording
    - **Property 13: Escalation and Event Recording**
    - **Validates: Requirements 7.1, 7.4**

  - [x] 9.5 Write property test for crisis helpline presentation
    - **Property 14: Crisis Helpline Presentation**
    - **Validates: Requirements 7.2**

  - [x] 9.6 Write property test for escalation cancellation window
    - **Property 15: Escalation Cancellation Window**
    - **Validates: Requirements 7.6**

  - [x] 9.7 Write unit tests for Notification Service
    - Test channel routing, snooze boundary, disabled-notifications gate, escalation cancellation at exactly 60 seconds
    - Test missing Trusted_Contact path (prompt user + show helpline)
    - _Requirements: 5.4, 6.4, 6.5, 7.1, 7.2, 7.5, 7.6_

- [x] 10. Implement Nudge Lambda with EventBridge Scheduler
  - Create `src/lambdas/nudge_lambda.py` triggered hourly by EventBridge
  - Query DynamoDB for active users; send check-in nudge if last entry gap > 24 hours (respecting snooze and disabled flags)
  - Send trend alert if `burnout_score` increased ≥15 points within the last 7 days
  - Respect user-configured nudge schedule (`nudge_time` in `UserProfile`)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 10.1 Write property test for burnout trend alert
    - **Property 12: Burnout Trend Alert**
    - **Validates: Requirements 6.2**

  - [x] 10.2 Write unit tests for Nudge Lambda
    - Test 24-hour gap boundary, snooze expiry, disabled-notifications gate, trend alert threshold
    - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [x] 11. Implement scheduled Burnout_Score recompute job
  - Extend `nudge_lambda.py` or create `src/lambdas/score_recompute_lambda.py` triggered daily by EventBridge
  - For each active user, recompute `Burnout_Score` using 30-day trend data and store a new `BurnoutScoreRecord` with `trigger: "scheduled_recompute"`
  - _Requirements: 4.5_

  - [x] 11.1 Write unit tests for scheduled recompute
    - Test that a new score record is written with correct trigger type and UTC timestamp
    - _Requirements: 4.5_

- [x] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Report Lambda and weekly Emotional_Health_Report generation
  - Create `src/lambdas/report_lambda.py` triggered weekly by EventBridge Scheduler
  - Read 7-day trend data from DynamoDB; compute sentiment distribution, average burnout score, top emotions, prior-week comparison
  - Call Bedrock to generate at least two AI insights
  - Store `EmotionalHealthReport` in DynamoDB; retain for minimum 12 months (set TTL or lifecycle policy)
  - Publish SNS notification that report is ready; route to user's preferred channel
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 13.1 Write property test for weekly report completeness
    - **Property 16: Weekly Report Completeness**
    - **Validates: Requirements 8.2, 8.3**

  - [x] 13.2 Write unit tests for Report Lambda
    - Test report field completeness, prior-week comparison calculation, 12-month retention policy
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [x] 14. Implement account settings and user profile management
  - Add API routes for updating `notification_prefs`, `trusted_contact`, and `escalation_threshold` in `UserProfile`
  - Implement snooze: accept duration (1h, 4h, 24h) and set `snooze_until` timestamp in DynamoDB
  - Implement data deletion: mark user data for deletion; complete within 30 days (schedule async cleanup job)
  - Enforce session re-authentication after 15 minutes of inactivity
  - _Requirements: 6.4, 7.3, 9.4, 9.5, 11.4_

  - [x] 14.1 Write property test for session re-authentication
    - **Property 18: Session Re-Authentication**
    - **Validates: Requirements 9.5**

  - [x] 14.2 Write unit tests for account settings
    - Test snooze durations, trusted contact update, escalation threshold update, data deletion request
    - _Requirements: 6.4, 7.3, 9.4, 11.4_

- [x] 15. Implement transcript sentiment fidelity check
  - In `src/utils/sentiment.py`, add `check_transcript_fidelity(voice_score, transcript_score) -> bool`
  - Validate that re-submitting a transcript as text yields a sentiment confidence score within 0.1 of the original voice entry's score
  - Log fidelity violations for monitoring
  - _Requirements: 10.1_

  - [x] 15.1 Write property test for transcript sentiment fidelity
    - **Property 19: Transcript Sentiment Fidelity**
    - **Validates: Requirements 10.1**

- [x] 16. Wire all components together in the Journal Ingest Lambda
  - Connect the full pipeline in `journal_ingest_lambda.py`: validate input → (Transcribe if voice) → Comprehend → Bedrock → DynamoDB → SNS alerts → return response within 60 seconds
  - Ensure coping suggestion is returned to the user within 60 seconds of submission
  - Apply security: all DynamoDB writes use anonymized `user_id`; no PII stored in Trend_Store; TLS enforced on all API Gateway routes
  - _Requirements: 1.1, 1.2, 2.1, 3.3, 4.3, 5.3, 9.1, 9.2, 9.3, 9.6_

  - [x] 16.1 Write integration tests for the full journal entry pipeline
    - Test end-to-end: text entry → sentiment → burnout score → DynamoDB → SNS alert (mocked AWS clients)
    - Test end-to-end: voice entry → Transcribe → sentiment → burnout score → DynamoDB
    - _Requirements: 1.1, 1.2, 2.1, 3.3, 4.3, 5.3_

- [x] 17. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All property tests use `hypothesis` (Python) with a minimum of 100 iterations, tagged with `# Feature: mindguard-ai, Property {N}: {property_text}`
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical pipeline boundaries

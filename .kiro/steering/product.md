# MindGuard AI — Product Overview

MindGuard AI is a serverless, proactive mental health and burnout prevention companion designed for women navigating career and family pressures. It continuously monitors emotional patterns through voice and text journaling, applies AI-driven sentiment analysis, and delivers personalized coping strategies and proactive burnout alerts before a crisis occurs.

## Core Capabilities

- Voice and text journal ingestion with real-time sentiment and emotion analysis (Amazon Comprehend)
- Burnout score computation (0–100) via Amazon Bedrock (Claude)
- Personalized coping suggestions delivered within 60 seconds of entry submission
- Proactive smart nudges and trend alerts via Amazon SNS + Pinpoint
- Emergency escalation to a user-configured trusted contact
- Weekly emotional health reports with AI-generated insights
- Privacy-first: all data stored with anonymized UUIDs, no PII in the Trend_Store

## Key Business Rules

- Burnout score > 70 triggers a push/SMS alert
- Burnout score > 85 triggers crisis helpline presentation
- Burnout score exceeding the user's `escalation_threshold` (default 80) notifies their trusted contact
- Escalation can be cancelled within 60 seconds of triggering
- Same coping suggestion must not repeat within a 48-hour window
- Users with notifications disabled must never receive nudges or alerts

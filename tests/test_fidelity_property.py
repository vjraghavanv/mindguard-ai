"""
Property-based tests for MindGuard AI Transcript Sentiment Fidelity.

# Feature: mindguard-ai, Property 19: Transcript Sentiment Fidelity

For any voice journal entry, the sentiment confidence score produced from the
transcript (re-submitted as text) must be within 0.1 of the confidence score
produced from the original voice entry's analysis.

Validates: Requirements 10.1
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.sentiment import check_transcript_fidelity

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid sentiment confidence scores in [0.0, 1.0]
score_st = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Pairs of scores that are within 0.1 of each other
@st.composite
def within_tolerance_scores(draw):
    voice = draw(score_st)
    # Use a tolerance slightly below 0.1 to avoid floating-point boundary issues
    _tol = 0.09999
    max_offset = min(_tol, 1.0 - voice)
    min_offset = max(-_tol, 0.0 - voice)
    offset = draw(
        st.floats(
            min_value=min_offset,
            max_value=max_offset,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    transcript = voice + offset
    return voice, transcript


# Pairs of scores that are strictly more than 0.1 apart
@st.composite
def outside_tolerance_scores(draw):
    voice = draw(score_st)
    # Pick a transcript score that is > 0.1 away from voice
    # Either voice + gap (if room) or voice - gap (if room)
    gap = draw(
        st.floats(
            min_value=0.1 + 1e-9,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    # Prefer adding gap; fall back to subtracting if out of range
    if voice + gap <= 1.0:
        transcript = voice + gap
    elif voice - gap >= 0.0:
        transcript = voice - gap
    else:
        # Can't place a score > 0.1 away — shrink voice toward 0.5 to make room
        voice = 0.5
        transcript = voice + gap if voice + gap <= 1.0 else voice - gap
    return voice, transcript


# ---------------------------------------------------------------------------
# Property 19: Transcript Sentiment Fidelity
# Validates: Requirements 10.1
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scores=within_tolerance_scores())
def test_fidelity_true_when_within_tolerance(scores):
    """
    # Feature: mindguard-ai, Property 19: Transcript Sentiment Fidelity

    Scores within 0.1 of each other must return True.

    Validates: Requirements 10.1
    """
    voice_score, transcript_score = scores
    assert check_transcript_fidelity(voice_score, transcript_score) is True


@settings(max_examples=100)
@given(scores=outside_tolerance_scores())
def test_fidelity_false_when_outside_tolerance(scores):
    """
    # Feature: mindguard-ai, Property 19: Transcript Sentiment Fidelity

    Scores more than 0.1 apart must return False.

    Validates: Requirements 10.1
    """
    voice_score, transcript_score = scores
    assert check_transcript_fidelity(voice_score, transcript_score) is False


@settings(max_examples=100)
@given(scores=within_tolerance_scores())
def test_fidelity_is_symmetric(scores):
    """
    # Feature: mindguard-ai, Property 19: Transcript Sentiment Fidelity

    The function must be symmetric: order of arguments does not affect the result.

    Validates: Requirements 10.1
    """
    voice_score, transcript_score = scores
    assert check_transcript_fidelity(voice_score, transcript_score) == \
           check_transcript_fidelity(transcript_score, voice_score)


@settings(max_examples=100)
@given(score=score_st)
def test_fidelity_identical_scores_always_true(score):
    """
    # Feature: mindguard-ai, Property 19: Transcript Sentiment Fidelity

    Identical scores must always return True (difference is 0.0 ≤ 0.1).

    Validates: Requirements 10.1
    """
    assert check_transcript_fidelity(score, score) is True

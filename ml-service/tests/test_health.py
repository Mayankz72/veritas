from app.services.health import LOW_SCORE_THRESHOLD, SectionStats


def test_no_claims_is_flagged_as_thin_evidence():
    stats = SectionStats(section="1 Introduction")
    flagged, reason = stats.flag()
    assert flagged is True
    assert "no claims" in reason


def test_any_unsupported_claim_flags_the_section():
    stats = SectionStats(section="3.2 Attention", claim_count=3, supported_count=2)
    stats.unsupported_count = 1
    stats.scores = [1.0, 1.0, 0.2]
    flagged, reason = stats.flag()
    assert flagged is True
    assert "unsupported" in reason


def test_low_average_score_flags_even_without_unsupported_claims():
    stats = SectionStats(section="4 Why Self-Attention", claim_count=2, partial_count=2)
    stats.scores = [LOW_SCORE_THRESHOLD - 0.1, LOW_SCORE_THRESHOLD - 0.2]
    flagged, reason = stats.flag()
    assert flagged is True
    assert "confidence" in reason


def test_healthy_section_is_not_flagged():
    stats = SectionStats(section="5.3 Optimizer", claim_count=2, supported_count=2)
    stats.scores = [1.0, 0.9]
    flagged, reason = stats.flag()
    assert flagged is False
    assert reason is None


def test_average_grounding_score_is_none_when_no_scores_recorded():
    stats = SectionStats(section="Abstract", claim_count=1, not_checked_count=1)
    assert stats.average_grounding_score is None

"""Health panel: surfaces sections with thin or weak evidence so an author
knows where to click "Strengthen" (re-run generation with a wider net)
rather than trusting a paper page is fully covered just because it renders.
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Claim

LOW_SCORE_THRESHOLD = 0.6


@dataclass
class SectionStats:
    section: str | None
    claim_count: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    not_checked_count: int = 0
    scores: list[float] = field(default_factory=list)

    @property
    def average_grounding_score(self) -> float | None:
        return sum(self.scores) / len(self.scores) if self.scores else None

    def flag(self) -> tuple[bool, str | None]:
        if self.claim_count == 0:
            return True, "no claims generated for this section yet"
        if self.unsupported_count > 0:
            return True, f"{self.unsupported_count} unsupported claim(s)"
        avg = self.average_grounding_score
        if avg is not None and avg < LOW_SCORE_THRESHOLD:
            return True, f"average grounding confidence {avg:.2f} below {LOW_SCORE_THRESHOLD}"
        return False, None


def compute_document_health(db: Session, document_id: str) -> list[SectionStats]:
    sections_in_order: list[str | None] = []
    seen: set[str | None] = set()
    for (section,) in db.execute(
        select(Chunk.section)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.order_index)
    ):
        if section not in seen:
            seen.add(section)
            sections_in_order.append(section)

    stats_by_section: dict[str | None, SectionStats] = {
        section: SectionStats(section=section) for section in sections_in_order
    }

    current_claims = db.execute(
        select(Claim).where(Claim.document_id == document_id, Claim.is_current.is_(True))
    ).scalars()

    for claim in current_claims:
        stats = stats_by_section.setdefault(claim.section, SectionStats(section=claim.section))
        stats.claim_count += 1
        if claim.grounding_label == "supported":
            stats.supported_count += 1
        elif claim.grounding_label == "partial":
            stats.partial_count += 1
        elif claim.grounding_label == "unsupported":
            stats.unsupported_count += 1
        else:
            stats.not_checked_count += 1
        if claim.grounding_score is not None:
            stats.scores.append(claim.grounding_score)

    return list(stats_by_section.values())

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document as DocumentModel
from app.models import QuizQuestion as QuizQuestionModel
from app.schemas.document import QuizQuestion as QuizQuestionSchema
from app.services.grounding_verifier import get_verifier
from app.services.quiz_generation import get_quiz_generator
from app.services.retrieval import retrieve_chunks

router = APIRouter(prefix="/documents/{document_id}/quiz", tags=["quiz"])


class GenerateQuizRequest(BaseModel):
    query: str
    section: str | None = None
    k: int = 5
    max_questions: int = 5


def _to_schema(q: QuizQuestionModel) -> QuizQuestionSchema:
    return QuizQuestionSchema(
        id=q.id,
        document_id=q.document_id,
        section=q.section,
        question=q.question,
        answer=q.answer,
        source_chunk_id=q.source_chunk_id,
        page=q.page,
        grounding_label=q.grounding_label,
        grounding_score=q.grounding_score,
    )


@router.post("/generate", response_model=list[QuizQuestionSchema])
def generate_quiz(
    document_id: str, payload: GenerateQuizRequest, db: Session = Depends(get_db)
) -> list[QuizQuestionSchema]:
    document = db.get(DocumentModel, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    retrieved = retrieve_chunks(db, document_id, payload.query, k=payload.k)
    if not retrieved:
        raise HTTPException(
            status_code=422,
            detail="No embedded chunks found for this document - was it fully ingested?",
        )

    passages = [rc.chunk.text for rc in retrieved]
    generator = get_quiz_generator()
    generated = generator.generate(passages, max_questions=payload.max_questions)

    verifier = get_verifier()
    question_models: list[QuizQuestionModel] = []
    for gen_q in generated:
        source = retrieved[gen_q.source_chunk_index]

        grounding_label, grounding_score = None, None
        if verifier is not None:
            grounding_label, grounding_score = verifier.predict(
                gen_q.supporting_sentence, source.chunk.text
            )

        question_models.append(
            QuizQuestionModel(
                id=uuid.uuid4().hex,
                document_id=document_id,
                section=payload.section,
                question=gen_q.question,
                answer=gen_q.answer,
                source_chunk_id=source.chunk.id,
                page=source.chunk.page,
                grounding_label=grounding_label,
                grounding_score=grounding_score,
            )
        )
    db.add_all(question_models)
    db.commit()

    return [_to_schema(q) for q in question_models]


@router.get("", response_model=list[QuizQuestionSchema])
def list_quiz(document_id: str, db: Session = Depends(get_db)) -> list[QuizQuestionSchema]:
    questions = (
        db.execute(select(QuizQuestionModel).where(QuizQuestionModel.document_id == document_id))
        .scalars()
        .all()
    )
    return [_to_schema(q) for q in questions]

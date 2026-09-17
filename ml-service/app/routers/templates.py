from fastapi import APIRouter

from app.services.templates import NarrativeTemplate, list_templates

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("", response_model=list[NarrativeTemplate])
def get_templates() -> list[NarrativeTemplate]:
    return list_templates()

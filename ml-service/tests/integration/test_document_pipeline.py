"""End-to-end integration tests against a real Postgres+pgvector instance:
upload -> extract -> embed -> retrieve -> generate claims/quiz -> grounding
verification -> health -> publish -> public fetch. Exercises the exact
request/response contracts the frontend relies on.
"""

import io

import fitz
import pytest

pytestmark = pytest.mark.integration


def _build_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "Synthetic Test Paper", fontsize=24, fontname="hebo")
    page.insert_text((72, 140), "Abstract", fontsize=16, fontname="hebo")
    rc = page.insert_textbox(
        fitz.Rect(72, 180, 500, 260),
        "This paper introduces the Widget Transformer, which achieves 42.0 "
        "accuracy on the Foo benchmark using 12 attention heads.",
        fontsize=11,
        fontname="helv",
    )
    assert rc >= 0, "abstract paragraph did not fit its textbox"
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture()
def uploaded_document(client):
    pdf_bytes = _build_pdf_bytes()
    response = client.post(
        "/documents/upload",
        files={"file": ("synthetic.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_extracts_title_and_chunks(uploaded_document):
    # /documents/upload intentionally uses the filename as a title hint,
    # which takes precedence over guessing a title from the PDF content
    # (see extract_document's title_hint param) - "synthetic.pdf" in, so
    # "synthetic" out, regardless of what the page itself says.
    assert uploaded_document["title"] == "synthetic"
    assert uploaded_document["pageCount"] == 1
    assert len(uploaded_document["chunks"]) >= 1
    assert any(c["section"] == "Abstract" for c in uploaded_document["chunks"])


def test_get_document_matches_upload_response(client, uploaded_document):
    response = client.get(f"/documents/{uploaded_document['id']}")
    assert response.status_code == 200
    assert response.json()["title"] == uploaded_document["title"]


def test_get_missing_document_is_404(client):
    response = client.get("/documents/does-not-exist")
    assert response.status_code == 404


def test_claims_generate_returns_grounded_supported_claim(client, uploaded_document):
    response = client.post(
        f"/documents/{uploaded_document['id']}/claims/generate",
        json={"query": "What does the Widget Transformer achieve?", "k": 1, "max_claims": 1},
    )
    assert response.status_code == 200, response.text
    claims = response.json()
    assert len(claims) == 1
    claim = claims[0]
    assert claim["page"] == 1
    assert 0.0 <= claim["retrievalScore"] <= 1.0
    # ExtractiveProvider always produces a literal substring of the source
    # chunk, so the grounding verifier's exact-substring short-circuit
    # should fire (see app/services/grounding_verifier.py).
    assert claim["groundingLabel"] == "supported"
    assert claim["groundingScore"] == 1.0


def test_claims_are_persisted_and_listable(client, uploaded_document):
    client.post(
        f"/documents/{uploaded_document['id']}/claims/generate",
        json={"query": "Widget Transformer accuracy", "k": 1, "max_claims": 1},
    )
    response = client.get(f"/documents/{uploaded_document['id']}/claims")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_regenerate_supersedes_prior_version(client, uploaded_document):
    doc_id = uploaded_document["id"]
    first = client.post(
        f"/documents/{doc_id}/claims/generate",
        json={"query": "Widget Transformer", "section": "Abstract", "k": 1, "max_claims": 1},
    ).json()
    assert first[0]["version"] == 1

    second = client.post(
        f"/documents/{doc_id}/claims/regenerate",
        json={"query": "Foo benchmark", "section": "Abstract", "k": 1, "max_claims": 1},
    ).json()
    assert second[0]["version"] == 2

    current = client.get(f"/documents/{doc_id}/claims").json()
    assert all(c["isCurrent"] for c in current)
    assert len(current) == 1

    everything = client.get(f"/documents/{doc_id}/claims?current_only=false").json()
    assert len(everything) == 2


def test_health_flags_sections_with_no_claims_and_clears_after_generation(
    client, uploaded_document
):
    doc_id = uploaded_document["id"]

    before = client.get(f"/documents/{doc_id}/health").json()
    abstract_health = next(s for s in before if s["section"] == "Abstract")
    assert abstract_health["flagged"] is True
    assert "no claims" in abstract_health["flagReason"]

    client.post(
        f"/documents/{doc_id}/claims/generate",
        json={"query": "Widget Transformer", "section": "Abstract", "k": 1, "max_claims": 1},
    )

    after = client.get(f"/documents/{doc_id}/health").json()
    abstract_health_after = next(s for s in after if s["section"] == "Abstract")
    assert abstract_health_after["flagged"] is False


def test_quiz_generate_produces_a_cloze_question_with_the_masked_number(
    client, uploaded_document
):
    response = client.post(
        f"/documents/{uploaded_document['id']}/quiz/generate",
        json={"query": "Widget Transformer accuracy", "k": 1, "max_questions": 1},
    )
    assert response.status_code == 200, response.text
    questions = response.json()
    assert len(questions) == 1
    assert "____" in questions[0]["question"]
    assert questions[0]["answer"] in ("42.0", "12")


def test_publish_and_fetch_public_bundle(client, uploaded_document):
    doc_id = uploaded_document["id"]
    client.post(
        f"/documents/{doc_id}/claims/generate",
        json={"query": "Widget Transformer", "k": 1, "max_claims": 1},
    )

    publish_response = client.post(f"/documents/{doc_id}/publish", json={})
    assert publish_response.status_code == 200
    publication_id = publish_response.json()["id"]

    bundle_response = client.get(f"/publications/{publication_id}")
    assert bundle_response.status_code == 200
    bundle = bundle_response.json()
    assert bundle["document"]["id"] == doc_id
    assert len(bundle["claims"]) == 1

    export_response = client.get(f"/publications/{publication_id}/export")
    assert export_response.status_code == 200
    assert export_response.json()["format"] == "veritas-export-v1"


def test_expired_publication_returns_410(client, uploaded_document):
    publish_response = client.post(
        f"/documents/{uploaded_document['id']}/publish",
        json={"expires_in_hours": -1},
    )
    publication_id = publish_response.json()["id"]

    response = client.get(f"/publications/{publication_id}")
    assert response.status_code == 410

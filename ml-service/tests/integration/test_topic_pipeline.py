"""The topic pipeline end to end against real Postgres: POST /topics runs the
search->ingest->brief->synthesize job (arXiv downloads are mocked, so this
runs offline) and GET /topics/{id} reports progress and results."""

import fitz
import pytest

pytestmark = pytest.mark.integration


def _pdf(model_name: str, score: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), f"The {model_name} Paper", fontsize=24, fontname="hebo")
    page.insert_text((72, 130), "Abstract", fontsize=16, fontname="hebo")
    rc = page.insert_textbox(
        fitz.Rect(72, 160, 520, 420),
        f"Existing sequence models are slow because they process tokens one at a time. "
        f"We propose the {model_name}, a new architecture that relies entirely on attention. "
        f"Our model achieves {score} accuracy on the Foo benchmark, outperforming all baselines. "
        f"We believe this approach will enable much longer contexts in future work.",
        fontsize=11,
        fontname="helv",
    )
    assert rc >= 0, "abstract did not fit its textbox"
    data = doc.tobytes()
    doc.close()
    return data


def _meta(arxiv_id: str, title: str, published: str) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["A. Author", "B. Writer"],
        "published": published,
        "abstract": f"{title} abstract about attention models.",
    }


@pytest.fixture()
def fake_arxiv(monkeypatch):
    """Serves synthetic PDFs by arXiv id and records every download."""
    pdfs = {
        "1111.00001": _pdf("Widget Transformer", "42.0"),
        "1111.00002": _pdf("Gadget Network", "57.5"),
    }
    calls: list[str] = []

    def download(arxiv_id: str) -> bytes:
        calls.append(arxiv_id)
        if arxiv_id not in pdfs:
            raise RuntimeError("404 from arXiv")
        return pdfs[arxiv_id]

    monkeypatch.setattr("app.services.topic_pipeline.download_arxiv_pdf", download)
    monkeypatch.setattr("app.services.topic_pipeline.ARXIV_POLITE_DELAY_SECONDS", 0)
    return calls


def _create(client, papers):
    response = client.post("/topics", json={"query": "attention models", "papers": papers})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_topic_pipeline_builds_briefs_and_synthesis(client, fake_arxiv):
    topic_id = _create(
        client,
        [
            _meta("1111.00001", "Widget Transformer", "2018-01-01"),
            _meta("1111.00002", "Gadget Network", "2019-01-01"),
        ],
    )

    topic = client.get(f"/topics/{topic_id}").json()
    assert topic["status"] == "done", topic
    assert [p["status"] for p in topic["papers"]] == ["done", "done"]

    widget = topic["papers"][0]
    assert widget["briefMode"] == "extractive"  # tests pin the keyless provider
    results = widget["brief"]["results"]
    assert results, "every brief needs key-result evidence"
    assert "42.0" in results[0]["text"]
    assert results[0]["page"] == 1
    assert results[0]["groundingLabel"] == "supported"  # verbatim quote from the paper

    synthesis = topic["synthesis"]
    assert synthesis["mode"] == "template"
    assert [row["title"] for row in synthesis["comparison"]] == [
        "Widget Transformer",
        "Gadget Network",
    ]
    assert synthesis["readingOrder"] == [p["id"] for p in topic["papers"]]

    # Briefs are ordinary claims, so they also show up in the paper's own workspace.
    claims = client.get(f"/documents/{widget['documentId']}/claims").json()
    assert {c["section"] for c in claims} >= {"Problem", "Method", "Key results"}

    listing = client.get("/topics").json()
    assert listing[0]["id"] == topic_id and listing[0]["paperCount"] == 2


def test_one_failing_paper_does_not_sink_the_topic(client, fake_arxiv):
    topic_id = _create(
        client,
        [
            _meta("1111.00001", "Widget Transformer", "2018-01-01"),
            _meta("9999.99999", "Missing Paper", "2019-01-01"),  # download raises
        ],
    )
    topic = client.get(f"/topics/{topic_id}").json()
    assert topic["status"] == "done"
    assert [p["status"] for p in topic["papers"]] == ["done", "error"]
    assert "404" in topic["papers"][1]["error"]
    assert len(topic["synthesis"]["comparison"]) == 1  # only the paper that worked


def test_topic_errors_when_every_paper_fails(client, fake_arxiv):
    topic_id = _create(client, [_meta("9999.99999", "Missing Paper", "2019-01-01")])
    topic = client.get(f"/topics/{topic_id}").json()
    assert topic["status"] == "error"
    assert topic["synthesis"] is None


def test_a_paper_seen_before_is_reused_not_reparsed(client, fake_arxiv):
    first = client.get(
        f"/topics/{_create(client, [_meta('1111.00001', 'Widget Transformer', '2018-01-01')])}"
    ).json()
    second = client.get(
        f"/topics/{_create(client, [_meta('1111.00001', 'Widget Transformer', '2018-01-01')])}"
    ).json()

    assert fake_arxiv == ["1111.00001"]  # downloaded once, not twice
    assert first["papers"][0]["documentId"] == second["papers"][0]["documentId"]
    assert second["papers"][0]["brief"]["method"]  # brief carried over


def test_topic_input_validation_and_missing_topic(client):
    assert client.post("/topics", json={"query": "   "}).status_code == 400
    assert client.post("/topics/search", json={"query": ""}).status_code == 400
    assert client.get("/topics/does-not-exist").status_code == 404


def test_deleting_a_topic_removes_it_but_keeps_the_parsed_paper(client, fake_arxiv):
    topic_id = _create(client, [_meta("1111.00001", "Widget Transformer", "2018-01-01")])
    document_id = client.get(f"/topics/{topic_id}").json()["papers"][0]["documentId"]

    assert client.delete(f"/topics/{topic_id}").status_code == 204
    assert client.get(f"/topics/{topic_id}").status_code == 404
    assert client.delete(f"/topics/{topic_id}").status_code == 404
    assert client.get("/topics").json() == []
    # The parsed document stays: it is what lets another topic reuse the paper.
    assert client.get(f"/documents/{document_id}").status_code == 200


def test_refresh_recomputes_the_synthesis_from_current_briefs(client, fake_arxiv):
    topic_id = _create(
        client,
        [
            _meta("1111.00001", "Widget Transformer", "2018-01-01"),
            _meta("1111.00002", "Gadget Network", "2019-01-01"),
        ],
    )
    before = client.get(f"/topics/{topic_id}").json()
    assert before["status"] == "done"

    response = client.post(f"/topics/{topic_id}/refresh")
    assert response.status_code == 200

    after = client.get(f"/topics/{topic_id}").json()
    assert after["status"] == "done"
    assert [r["paperId"] for r in after["synthesis"]["comparison"]] == [
        r["paperId"] for r in before["synthesis"]["comparison"]
    ]
    assert fake_arxiv == ["1111.00001", "1111.00002"]  # nothing was downloaded again
    assert client.post("/topics/does-not-exist/refresh").status_code == 404

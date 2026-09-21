from app.services.claim_generation import ExtractiveProvider, extract_supporting_quote


def test_extractive_provider_pulls_leading_sentence_per_passage():
    provider = ExtractiveProvider()
    passages = [
        "The Transformer relies entirely on self-attention. It avoids recurrence.",
        "Multi-head attention lets the model jointly attend to information.",
    ]
    claims = provider.generate_claims("Model Architecture", passages, max_claims=5)

    assert len(claims) == 2
    assert claims[0].text.startswith("The Transformer relies entirely on self-attention")
    assert claims[0].source_chunk_index == 0
    assert claims[1].source_chunk_index == 1


def test_extractive_provider_stitches_short_leading_sentence():
    provider = ExtractiveProvider()
    passages = ["Yes. This is the real content that follows a short opener sentence."]
    claims = provider.generate_claims("Topic", passages, max_claims=5)

    assert len(claims) == 1
    assert claims[0].text.startswith("Yes. This is the real content")


def test_extractive_provider_respects_max_claims():
    provider = ExtractiveProvider()
    passages = [f"Sentence number {i}. More text follows." for i in range(10)]
    claims = provider.generate_claims("Topic", passages, max_claims=3)
    assert len(claims) == 3


def test_extract_supporting_quote_exact_substring():
    passage = "Self-attention connects all positions with a constant number of steps."
    claim = "connects all positions with a constant number of steps"
    assert extract_supporting_quote(claim, passage) == claim


def test_extract_supporting_quote_falls_back_to_best_sentence():
    passage = (
        "Recurrent models factor computation along symbol positions. "
        "Attention allows modeling dependencies without regard to distance."
    )
    claim = "Attention lets you model dependencies regardless of distance in the sequence."
    quote = extract_supporting_quote(claim, passage)
    assert "Attention allows modeling dependencies" in quote

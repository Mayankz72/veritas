from app.services.quiz_generation import ClozeQuizGenerator


def test_cloze_masks_first_number_in_eligible_sentence():
    generator = ClozeQuizGenerator()
    passages = ["We employ h = 8 parallel attention layers, or heads, in this architecture."]

    questions = generator.generate(passages, max_questions=5)

    assert len(questions) == 1
    q = questions[0]
    assert q.answer == "8"
    assert "____" in q.question
    assert "8" not in q.question
    assert q.supporting_sentence.startswith("We employ h = 8")
    assert q.source_chunk_index == 0


def test_cloze_skips_passages_without_a_number():
    generator = ClozeQuizGenerator()
    passages = ["Self-attention relates different positions of a single sequence."]
    questions = generator.generate(passages, max_questions=5)
    assert questions == []


def test_cloze_skips_short_sentences_but_finds_a_later_eligible_one():
    generator = ClozeQuizGenerator()
    passages = [
        "Yes. We trained the base model for a total of 100000 steps, which took about 12 hours."
    ]
    questions = generator.generate(passages, max_questions=5)
    assert len(questions) == 1
    assert questions[0].answer in ("100000", "12")


def test_cloze_respects_max_questions():
    generator = ClozeQuizGenerator()
    passages = [
        f"This sentence contains the number {i} for testing purposes here." for i in range(10)
    ]
    questions = generator.generate(passages, max_questions=4)
    assert len(questions) == 4

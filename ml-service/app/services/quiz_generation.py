"""Evidence-linked quiz generation.

Default strategy is cloze deletion (mask a number in a sentence, ask the
reader to recall it) - no LLM needed, and the "question" is by construction
a literal sentence from the source passage, so grounding verification
(app/services/grounding_verifier.py) always short-circuits it to
`supported`. This mirrors the ExtractiveProvider design in
claim_generation.py: the default path needs no API key and is trivially
checkable, real LLM providers are an upgrade for richer questions later.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GeneratedQuestion:
    question: str
    answer: str
    supporting_sentence: str
    source_chunk_index: int


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


class QuizGenerator(ABC):
    name: str

    @abstractmethod
    def generate(self, passages: list[str], max_questions: int) -> list[GeneratedQuestion]: ...


class ClozeQuizGenerator(QuizGenerator):
    """Masks the first number in the first eligible sentence of each passage."""

    name = "cloze"
    MIN_SENTENCE_LENGTH = 30

    def generate(self, passages: list[str], max_questions: int = 5) -> list[GeneratedQuestion]:
        questions: list[GeneratedQuestion] = []
        for idx, passage in enumerate(passages):
            if len(questions) >= max_questions:
                break
            question = self._question_from_passage(passage, idx)
            if question is not None:
                questions.append(question)
        return questions

    def _question_from_passage(self, passage: str, chunk_index: int) -> GeneratedQuestion | None:
        for sentence in _split_sentences(passage):
            if len(sentence) < self.MIN_SENTENCE_LENGTH:
                continue
            match = NUMBER_RE.search(sentence)
            if not match:
                continue
            answer = match.group(0)
            blanked = f"{sentence[: match.start()]}____{sentence[match.end() :]}"
            return GeneratedQuestion(
                question=blanked,
                answer=answer,
                supporting_sentence=sentence,
                source_chunk_index=chunk_index,
            )
        return None


def get_quiz_generator() -> QuizGenerator:
    return ClozeQuizGenerator()

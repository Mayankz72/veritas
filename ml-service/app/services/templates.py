"""Narrative templates: presets for the query box, not prompt-engineering
magic - each just pre-fills a good default query/topic for a common way of
reading a paper. Matches Trace's "Method Walkthrough" / "Results Briefing"
built-ins conceptually, reimplemented independently.
"""

from pydantic import BaseModel


class NarrativeTemplate(BaseModel):
    key: str
    label: str
    query: str


TEMPLATES: list[NarrativeTemplate] = [
    NarrativeTemplate(
        key="method_walkthrough",
        label="Method Walkthrough",
        query="Explain how the proposed method works, step by step.",
    ),
    NarrativeTemplate(
        key="results_briefing",
        label="Results Briefing",
        query="Summarize the key experimental results and what they show.",
    ),
    NarrativeTemplate(
        key="architecture_overview",
        label="Architecture Overview",
        query="Describe the overall model architecture and its main components.",
    ),
    NarrativeTemplate(
        key="limitations_and_future_work",
        label="Limitations & Future Work",
        query="What limitations or future research directions does the paper mention?",
    ),
]


def list_templates() -> list[NarrativeTemplate]:
    return TEMPLATES

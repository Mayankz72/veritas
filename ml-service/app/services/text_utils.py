import re

# Abbreviations whose trailing period must not end a sentence. Research-paper
# text is full of them ("et al.", "Fig. 3", "e.g."), and a naive
# split-on-period turns each into a bogus sentence fragment.
_ABBREVIATIONS = (
    "et al.",
    "e.g.",
    "i.e.",
    "cf.",
    "vs.",
    "Fig.",
    "Figs.",
    "Eq.",
    "Eqs.",
    "Sec.",
    "Tab.",
    "No.",
    "approx.",
    "resp.",
)
_PLACEHOLDER = ""
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])")


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", _PLACEHOLDER))
    parts = _SENTENCE_BOUNDARY_RE.split(protected)
    return [p.replace(_PLACEHOLDER, ".").strip() for p in parts if p.strip()]

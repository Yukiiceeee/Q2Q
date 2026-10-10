"""Query text sanitisation shared by the retrieval stack.

The retrieval stack assumes a ``raw_query`` that is nothing but the user's
question. In practice a query arrives wrapped in whatever scaffolding the
caller built around it: chat templates, answer-format instructions, date
conversion rules, few-shot examples. That scaffolding is addressed to the
*answer generator*, not to the retriever, and embedding it pollutes the
query vector -- "when is X planning to go camping?" and the same sentence
followed by 400 tokens of "CRITICAL DATE FORMAT RULES" are not the same
point in embedding space.

The rules here are dataset-agnostic: they key on the *shape* of prompt
scaffolding (a paragraph-introducing question label, a block-opening
heading, an answer cue), never on a specific dataset's wording beyond the
handful of synonymous labels that any QA template would use.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_query", "sanitize_short"]


# The question label must open a paragraph (\A or after a blank line), so a
# "Q:" occurring *inside* a question cannot truncate it.
_QUESTION_ANCHOR = re.compile(
    r"(?:\A|\n[ \t]*\n)[ \t]*(?:问题内容|问题|Question|Q)[ \t]*[:：][ \t]*",
    re.IGNORECASE,
)

# A leading paragraph that only sets up the question (names the memory
# source, states how to answer) rather than asking anything.
_PREAMBLE_PARAGRAPH = re.compile(
    r"^[ \t]*(?:Based on|Given|Refer(?:ring)? to|Using|Read|Consider"
    r"|请根据|请结合|根据|结合|以下是)",
    re.IGNORECASE,
)

# Block-opening markers: a heading or bracketed directive that introduces
# instructions rather than content. Anchored at line start for the same
# reason as the question anchor.
_TAIL_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "ANSWER FORMAT (CRITICAL):", "FORMAT REQUIREMENTS (CRITICAL - ...):",
    # "CRITICAL DATE FORMAT RULES:", "IMPORTANT:"
    re.compile(
        r"^[ \t]*(?:CRITICAL|IMPORTANT|NOTE)[^\n:]{0,60}:[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^[ \t]*(?:ANSWER|RESPONSE|OUTPUT|FORMAT)[^\n:]{0,60}:[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "【回答要求】", "[ANSWER REQUIREMENTS]"
    re.compile(r"^[ \t]*【[^】\n]{0,40}】", re.MULTILINE),
    re.compile(
        r"^[ \t]*\[(?:ANSWER|RESPONSE|OUTPUT|FORMAT)[^\]]{0,60}\]",
        re.MULTILINE | re.IGNORECASE,
    ),
)

# A paragraph that issues an answering directive rather than asking
# anything. Some templates place these between the question and the answer
# cue with no heading of their own ("Provide a concise and direct answer
# based on what you recall from the trajectory."). Only consulted from the
# second paragraph onwards, so a question that happens to open with one of
# these verbs is left alone.
_DIRECTIVE_PARAGRAPH = re.compile(
    r"^[ \t]*(?:Provide|Give|Respond|Reply|Do not|Do NOT"
    r"|Please[ \t]+(?:provide|give|answer|list|respond|keep)"
    r"|请(?:给出|回答|直接|简要|列出|结合|明确)|回答要求|注意)",
    re.IGNORECASE,
)

# Lines that are pure answer cue and carry no question content.
_TAIL_CUE_LINES: frozenset[str] = frozenset(
    {
        "answer:",
        "answer：",
        "答案:",
        "答案：",
        "答案",
        "answer",
        "your answer:",
        "final answer:",
    }
)

_MAX_TAIL_LINES = 40
_MAX_TAIL_CHARS = 2000


def _strip_leading_preamble(text: str) -> str:
    """Drop leading paragraphs that set up the question but do not ask it."""
    paragraphs = text.split("\n\n")
    while len(paragraphs) > 1 and _PREAMBLE_PARAGRAPH.match(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs)


def _cut_at_first_block(text: str) -> str:
    """Drop the first block-opening marker and everything after it."""
    cut = len(text)
    for pattern in _TAIL_BLOCK_PATTERNS:
        match = pattern.search(text)
        if match and match.start() < cut:
            cut = match.start()
    return text[:cut]


def _cut_at_directive_paragraph(text: str) -> str:
    """Drop a trailing directive paragraph that only says how to answer.

    Scanned from the second paragraph onwards: the question itself may sit
    in its own paragraph, but a directive never precedes it.
    """
    paragraphs = text.split("\n\n")
    for index in range(1, len(paragraphs)):
        if _DIRECTIVE_PARAGRAPH.match(paragraphs[index]):
            return "\n\n".join(paragraphs[:index])
    return text


def _cut_at_cue_line(text: str) -> str:
    """Drop a trailing answer-cue line and everything after it."""
    lines = text.split("\n")
    for index in range(1, len(lines)):
        if lines[index].strip().lower() in _TAIL_CUE_LINES:
            return "\n".join(lines[:index])
    return text


def _strip_edge_noise(text: str) -> str:
    """Trim surrounding blank lines, quotes and bullets."""
    return text.strip().strip("\"'“”").strip(" \t\n")


def sanitize_query(text: str) -> str:
    """Reduce a wrapped query to the question it carries.

    Applied in order, each step only ever shortens the text:
    1. keep everything after the last question label, if one opens a
       paragraph (the caller's template always puts the question last);
    2. drop leading set-up paragraphs;
    3. cut at the first block-opening instruction marker;
    4. cut at a trailing answer cue;
    5. strip edge noise and cap a runaway tail.

    Returns the input stripped when no rule fires, so an already-clean
    question round-trips untouched.
    """
    if not text:
        return ""

    original = text.strip()

    anchors = list(_QUESTION_ANCHOR.finditer(original))
    if anchors:
        candidate = original[anchors[-1].end():]
        # The anchor must lead somewhere; an empty tail means the pattern
        # matched a label rather than the question itself.
        if candidate.strip():
            original = candidate

    candidate = _strip_leading_preamble(original)
    candidate = _cut_at_first_block(candidate)
    candidate = _cut_at_directive_paragraph(candidate)
    candidate = _cut_at_cue_line(candidate)
    candidate = _strip_edge_noise(candidate)

    if not candidate:
        return text.strip()

    lines = candidate.split("\n")
    if len(lines) > _MAX_TAIL_LINES:
        candidate = "\n".join(lines[:_MAX_TAIL_LINES])
    if len(candidate) > _MAX_TAIL_CHARS:
        candidate = candidate[:_MAX_TAIL_CHARS].rstrip()

    return candidate


def sanitize_short(text: str, max_chars: int = 400) -> str:
    """Sanitise and bound the result to a compact question."""
    cleaned = sanitize_query(text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned

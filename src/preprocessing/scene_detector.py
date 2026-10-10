"""Content-driven scene detection for mode-specific memory construction.

Detection is based purely on structural signals in the text (turn markers,
step/action markers, density of enumerated records) — never on dataset
identity — so the same rules generalise to any new corpus with the same shape.
"""
from __future__ import annotations

import logging
import re

from src.scenes import SCENE_DIALOGUE, SCENE_DOCUMENT, SCENE_TRAJECTORY

logger = logging.getLogger(__name__)

# Agent execution traces: explicit step indices plus action/observation verbs.
_STEP_PATTERN = re.compile(r'(?:^|\n)\s*(?:step|Step|STEP)[ _]?(\d+)\s*[:.\)]', re.MULTILINE)

# Field labels that trajectory and record formats print with the very same
# "Label:" shape as a speaker tag. They name a structure slot, never a speaker,
# so they must stay out of the turn count that drives the dialogue rule.
_ACTION_LABELS = frozenset({"action", "observation", "thought", "tool_call", "function_call"})
_RECORD_LABELS = frozenset({"task", "date", "conversation", "result", "step"})
_STRUCTURAL_LABELS = _ACTION_LABELS | _RECORD_LABELS

_ACTION_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:' + "|".join(sorted(_ACTION_LABELS)) + r')\s*[:：]',
    re.MULTILINE | re.IGNORECASE,
)
# "Step 12" carries an ordinal, not a speaker identity.
_STEP_LABEL_PATTERN = re.compile(r'step\s*\d+', re.IGNORECASE)

# UI/accessibility-tree element references such as "click [96]" or "[178] link '...'"
_ELEMENT_REF_PATTERN = re.compile(r'\[\d{2,}\]')

# Conversational turns, in either convention:
#   "Name: text"                — speaker-tagged lines (chat logs, transcripts)
#   "Name said, \"text\""        — narrating transcript form (e.g. LoCoMo)
# One capture group so callers keep receiving speaker names, not tuples.
_TURN_PATTERN = re.compile(
    r'(?:^|\n)\s*([A-Z][\w .\'-]{0,28}|[一-龥]{1,10})\s*'
    r'(?:[:：]\s*\S|\ssaid\b\s*[,:])',
    re.MULTILINE,
)


def _is_speaker_label(label: str) -> bool:
    """Reject structure labels that merely look like a speaker tag."""
    normalized = label.strip().lower()
    if normalized in _STRUCTURAL_LABELS:
        return False
    return not _STEP_LABEL_PATTERN.fullmatch(normalized)


def detect_scene(text: str) -> str:
    """Classify text into a memory-construction scene mode.

    Returns one of SCENE_TRAJECTORY / SCENE_DIALOGUE / SCENE_DOCUMENT.
    """
    if not text or not text.strip():
        return SCENE_DOCUMENT

    sample = text[:200000]

    step_hits = len(_STEP_PATTERN.findall(sample))
    action_hits = len(_ACTION_PATTERN.findall(sample))
    element_hits = len(_ELEMENT_REF_PATTERN.findall(sample))

    # Trajectory needs an ordinal spine plus action/observation field labels,
    # or a dense UI element-reference tree on its own.
    # - The spine threshold is 1, not a larger margin: the two signals are
    #   independent structural families, so a single ordinal marker together
    #   with Action:/Observation: labels is already decisive. A bigger margin
    #   only rejected legitimate fragments, because one oversized observation
    #   (an accessibility tree or a subtitle dump) fills a whole chunk and
    #   leaves it with a lone step marker.
    # - The element branch stands alone deliberately: a single accessibility
    #   tree routinely exceeds 20 references while no dialogue or document
    #   corpus reaches even 1.
    if element_hits >= 20 or (step_hits >= 1 and action_hits >= 2):
        return SCENE_TRAJECTORY

    turn_matches = [
        m for m in _TURN_PATTERN.findall(sample) if _is_speaker_label(m)
    ]
    distinct_speakers = {m.strip() for m in turn_matches}
    if len(turn_matches) >= 5 and len(distinct_speakers) >= 2:
        return SCENE_DIALOGUE

    return SCENE_DOCUMENT


def extract_ordinal_span(text: str) -> tuple[int, int] | None:
    """Return (min, max) step index present in a trajectory, if any.

    Used to tell the extractor which ordinal range it must cover so that no
    step is silently dropped.
    """
    hits = [int(m) for m in _STEP_PATTERN.findall(text[:200000])]
    if not hits:
        return None
    return min(hits), max(hits)

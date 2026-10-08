#!/usr/bin/env python3
"""Cut a deceptive rollout at the point just before it states its conclusion."""
from __future__ import annotations

import re

THINK_CLOSE = "</think>"
SENT_RE = re.compile(r"[^.!?\n]+[.!?]+[\"')\]]*|\S[^.!?\n]*$")


def split_at_final_think_block(text: str, n_sentences: int) -> tuple[str, str] | None:
    """Split a completion's reasoning into (head, final `n_sentences`).

    The head becomes the steering prefix: the model has already committed by
    that point, but has not yet written the sentences that state the decision,
    so both arms resume inside the still-open think block. Returns None when the
    reasoning is too short to cut meaningfully.
    """
    cut = text.find(THINK_CLOSE)
    reasoning = text[:cut] if cut > 0 else text
    if len(reasoning.strip()) < 120:
        return None
    sents = [m.group(0) for m in SENT_RE.finditer(reasoning)]
    if len(sents) < n_sentences + 2:
        return None
    head = "".join(sents[:-n_sentences])
    tail = "".join(sents[-n_sentences:])
    if len(head.strip()) < 80 or len(tail.strip()) < 30:
        return None
    return head, tail

from __future__ import annotations

import re


def normalize_pae_text(text: str) -> str:
    """
    Minimal normalizer for PAE text used as model target.

    It only makes formatting stable for training.
    """
    text = text.strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Optional: enforce spaces around barlines if your source is inconsistent
    text = text.replace("|", " | ")
    text = re.sub(r"\s+", " ", text).strip()

    return text
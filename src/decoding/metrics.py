from __future__ import annotations

from typing import List, Tuple


def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Standard Levenshtein distance.
    """
    n = len(seq1)
    m = len(seq2)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    CER = edit distance over characters / number of reference characters
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    WER = edit distance over whitespace-separated tokens / number of reference tokens
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)


def corpus_cer(references: List[str], hypotheses: List[str]) -> float:
    total_dist = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        total_dist += edit_distance(list(ref), list(hyp))
        total_ref_len += len(ref)

    if total_ref_len == 0:
        return 0.0

    return total_dist / total_ref_len


def corpus_wer(references: List[str], hypotheses: List[str]) -> float:
    total_dist = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()

        total_dist += edit_distance(ref_words, hyp_words)
        total_ref_len += len(ref_words)

    if total_ref_len == 0:
        return 0.0

    return total_dist / total_ref_len
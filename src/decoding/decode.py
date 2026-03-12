from __future__ import annotations

from typing import List

import torch

from tokenization.pae_tokenizer import PAETokenizer


def greedy_decode_batch(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: PAETokenizer,
) -> List[str]:
    """
    Args:
        log_probs: [T, B, vocab_size]
        output_lengths: [B]

    Returns:
        List[str]: decoded normalized PAE strings
    """
    pred_ids = log_probs.argmax(dim=-1)   # [T, B]
    pred_ids = pred_ids.transpose(0, 1)   # [B, T]

    predictions: List[str] = []

    for b in range(pred_ids.size(0)):
        length = int(output_lengths[b].item())
        raw_ids = pred_ids[b, :length].tolist()
        text = tokenizer.ctc_collapse(raw_ids)
        predictions.append(text)

    return predictions
from __future__ import annotations

from typing import List

import torch


def pyctcdecode_batch(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    decoder,
) -> List[str]:
    """
    Args:
        log_probs: [T, B, vocab_size] log probabilities
        output_lengths: [B]
        decoder: pyctcdecode decoder

    Returns:
        List[str]
    """
    # pyctcdecode expects per-sample arrays shaped [time, vocab]
    probs = log_probs.detach().cpu().exp()   # convert log-probs -> probs
    probs = probs.transpose(0, 1)            # [B, T, vocab]

    predictions: List[str] = []

    for b in range(probs.size(0)):
        length = int(output_lengths[b].item())
        emissions = probs[b, :length].numpy()   # [T_b, vocab]
        text = decoder.decode(emissions)
        predictions.append(text)

    return predictions
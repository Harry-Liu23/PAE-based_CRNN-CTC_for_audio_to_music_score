from __future__ import annotations

from typing import Any, Dict, List

import torch


def ctc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads feature sequences and concatenates targets for CTCLoss.

    Returns:
        features:       [B, T_max, F]
        input_lengths:  [B]
        targets:        [sum(U_i)]
        target_lengths: [B]
    """
    batch_size = len(batch)
    feat_dim = batch[0]["features"].size(1)
    max_time = max(item["features"].size(0) for item in batch)

    padded_features = torch.zeros(batch_size, max_time, feat_dim, dtype=torch.float32)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)

    targets_list = []
    pae_texts = []
    audio_paths = []

    for i, item in enumerate(batch):
        feats = item["features"]
        tgt = item["targets"]

        T = feats.size(0)
        padded_features[i, :T] = feats
        input_lengths[i] = item["input_length"]
        target_lengths[i] = item["target_length"]

        targets_list.append(tgt)
        pae_texts.append(item["pae_text"])
        audio_paths.append(item["audio_path"])

    targets = torch.cat(targets_list, dim=0)

    return {
        "features": padded_features,      # [B, T_max, F]
        "input_lengths": input_lengths,   # [B]
        "targets": targets,               # [sum(U_i)]
        "target_lengths": target_lengths, # [B]
        "pae_texts": pae_texts,
        "audio_paths": audio_paths,
    }
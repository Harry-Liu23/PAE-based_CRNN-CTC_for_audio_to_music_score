from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from decoding.decode import greedy_decode_batch
from decoding.decode_pyctc import pyctcdecode_batch
from decoding.metrics import corpus_cer, corpus_wer


@dataclass
class TrainStepOutput:
    loss: float


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "features": batch["features"].to(device),
        "input_lengths": batch["input_lengths"].to(device),
        "targets": batch["targets"].to(device),
        "target_lengths": batch["target_lengths"].to(device),
        "pae_texts": batch["pae_texts"],
        "audio_paths": batch["audio_paths"],
    }


def greedy_decode(log_probs: torch.Tensor) -> List[List[int]]:
    """
    Args:
        log_probs: [T, B, vocab_size]

    Returns:
        list of token-id sequences, one per batch item, before CTC collapse
    """
    pred_ids = log_probs.argmax(dim=-1)  # [T, B]
    pred_ids = pred_ids.transpose(0, 1)  # [B, T]
    return [seq.tolist() for seq in pred_ids]


def train_one_step(
    model: nn.Module,
    batch: Dict,
    ctc_loss: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainStepOutput:
    model.train()
    batch = move_batch_to_device(batch, device)

    optimizer.zero_grad()

    output = model(
        features=batch["features"],
        input_lengths=batch["input_lengths"],
    )

    loss = ctc_loss(
        output.log_probs,           # [T, B, vocab]
        batch["targets"],           # [sum(target_lengths)]
        output.output_lengths,      # [B]
        batch["target_lengths"],    # [B]
    )

    loss.backward()
    optimizer.step()

    return TrainStepOutput(loss=float(loss.item()))


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        output = model(
            features=batch["features"],
            input_lengths=batch["input_lengths"],
        )

        loss = ctc_loss(
            output.log_probs,
            batch["targets"],
            output.output_lengths,
            batch["target_lengths"],
        )

        total_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0

    return total_loss / total_batches


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 20,
) -> float:
    total_loss = 0.0
    total_batches = 0

    for step, batch in enumerate(loader, start=1):
        out = train_one_step(
            model=model,
            batch=batch,
            ctc_loss=ctc_loss,
            optimizer=optimizer,
            device=device,
        )
        total_loss += out.loss
        total_batches += 1

        if step % log_every == 0:
            print(f"step={step} loss={out.loss:.4f}")

    if total_batches == 0:
        return 0.0

    return total_loss / total_batches


@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    ctc_loss,
    tokenizer,
    device,
    decoding: str = "greedy",
    pyctc_decoder=None,
):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    all_refs = []
    all_hyps = []

    for batch in loader:
        batch = {
            "features": batch["features"].to(device),
            "input_lengths": batch["input_lengths"].to(device),
            "targets": batch["targets"].to(device),
            "target_lengths": batch["target_lengths"].to(device),
            "pae_texts": batch["pae_texts"],
            "audio_paths": batch["audio_paths"],
        }

        output = model(
            features=batch["features"],
            input_lengths=batch["input_lengths"],
        )

        loss = ctc_loss(
            output.log_probs,
            batch["targets"],
            output.output_lengths,
            batch["target_lengths"],
        )

        if decoding == "greedy":
            hyps = greedy_decode_batch(
                log_probs=output.log_probs,
                output_lengths=output.output_lengths,
                tokenizer=tokenizer,
            )
        elif decoding == "pyctcdecode":
            if pyctc_decoder is None:
                raise ValueError("pyctc_decoder must be provided when decoding='pyctcdecode'")
            hyps = pyctcdecode_batch(
                log_probs=output.log_probs,
                output_lengths=output.output_lengths,
                decoder=pyctc_decoder,
            )
        else:
            raise ValueError(f"Unsupported decoding mode: {decoding}")

        refs = batch["pae_texts"]

        all_refs.extend(refs)
        all_hyps.extend(hyps)

        total_loss += float(loss.item())
        total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    cer = corpus_cer(all_refs, all_hyps)
    wer = corpus_wer(all_refs, all_hyps)

    return {
        "loss": avg_loss,
        "cer": cer,
        "wer": wer,
    }
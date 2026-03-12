from __future__ import annotations

from typing import List

from pyctcdecode import build_ctcdecoder

from pae_tokenizer import PAETokenizer


def build_labels_from_tokenizer(tokenizer: PAETokenizer) -> List[str]:
    """
    Build pyctcdecode labels in exact vocabulary-index order.

    Index i in this list must correspond to logits[..., i].
    """
    vocab_size = tokenizer.vocab.vocab_size
    labels = [tokenizer.vocab.itos[i] for i in range(vocab_size)]

    # pyctcdecode expects a blank-like CTC label in the label list.
    # Using "" is a practical choice for blank.
    if labels[tokenizer.vocab.blank_id] == tokenizer.blank_token:
        labels[tokenizer.vocab.blank_id] = ""

    return labels


def build_pae_ctc_decoder(tokenizer: PAETokenizer):
    """
    Plain CTC beam search decoder without a language model.
    """
    labels = build_labels_from_tokenizer(tokenizer)
    decoder = build_ctcdecoder(labels=labels)
    return decoder
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PAEVocab:
    string_to_index: Dict[str, int]
    index_to_string: Dict[int, str]
    blank_id: int

    @property
    def vocab_size(self) -> int:
        return len(self.string_to_index)


class PAETokenizer:
    """
    Character-level tokenizer for PAE-based output sequences.

    Notes:
    - blank_id is reserved for CTC and is NOT part of the text alphabet.
    """

    def __init__(self, alphabet: List[str]) -> None:
        unique = sorted(set(alphabet))
        self.blank_token = "<BLANK>"
        self.blank_id = 0

        string_to_index = {ch: i + 1 for i, ch in enumerate(unique)}
        index_to_string = {i + 1: ch for i, ch in enumerate(unique)}
        string_to_index[self.blank_token] = self.blank_id
        index_to_string[self.blank_id] = self.blank_token

        self.vocab = PAEVocab(string_to_index=string_to_index, index_to_string=index_to_string, blank_id=self.blank_id)

    @classmethod
    def from_texts(cls, texts: List[str]) -> "PAETokenizer":
        alphabet = set()
        for text in texts:
            alphabet.update(text)
        return cls(sorted(alphabet))

    def encode(self, text: str) -> List[int]:
        unknown = [ch for ch in text if ch not in self.vocab.string_to_index]
        if unknown:
            raise ValueError(f"Unknown characters in text: {sorted(set(unknown))}")
        return [self.vocab.string_to_index[ch] for ch in text]

    def decode_raw(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            if idx == self.blank_id:
                chars.append(self.blank_token)
            else:
                chars.append(self.vocab.index_to_string[idx])
        return "".join(chars)

    def ctc_collapse(self, ids: List[int]) -> str:
        """
        Collapse repeated tokens and remove blanks, as required by CTC decoding.
        """
        collapsed: List[int] = []
        prev = None
        for idx in ids:
            if idx != prev:
                collapsed.append(idx)
            prev = idx

        chars = [self.vocab.index_to_string[idx] for idx in collapsed if idx != self.blank_id]
        return "".join(chars)
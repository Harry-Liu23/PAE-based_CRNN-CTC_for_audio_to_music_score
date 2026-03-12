from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torchaudio
from torch.utils.data import Dataset

from tokenization.pae_preprocess import normalize_pae_text
from tokenization.pae_tokenizer import PAETokenizer
from audio_features import LogSTFTExtractor


@dataclass
class SampleItem:
    audio_path: str
    pae_text: str


class AudioToScoreDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: PAETokenizer,
        feature_extractor: LogSTFTExtractor,
        target_sample_rate: int = 22050,
    ) -> None:
        self.samples = [SampleItem(**s) for s in samples]
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.target_sample_rate = target_sample_rate

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)  # [channels, time]

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target sample rate
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate,
            )

        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Audio branch
        waveform = self._load_audio(sample.audio_path)
        features = self.feature_extractor(waveform)  # [T, F]

        # Target branch: normalize before tokenization
        pae_text = normalize_pae_text(sample.pae_text)
        target_ids = torch.tensor(self.tokenizer.encode(pae_text), dtype=torch.long)

        return {
            "features": features,                              # [T, F]
            "targets": target_ids,                            # [U]
            "input_length": torch.tensor(features.size(0), dtype=torch.long),
            "target_length": torch.tensor(target_ids.size(0), dtype=torch.long),
            "pae_text": pae_text,                             # normalized text
            "audio_path": sample.audio_path,
        }
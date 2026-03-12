from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CRNNOutput:
    log_probs: torch.Tensor      # [T_out, B, vocab_size]
    output_lengths: torch.Tensor # [B]


class ConvBlock(nn.Module):
    """
    Conv2D -> BatchNorm -> ReLU

    Input:
        [B, C_in, T, F]
    Output:
        [B, C_out, T_out, F_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CRNN(nn.Module):
    """
    Spectrogram-based CRNN for CTC transcription.

    Input:
        features: [B, T, F]
        input_lengths: [B]

    Output:
        log_probs: [T_out, B, vocab_size]
        output_lengths: [B]
    """

    def __init__(
        self,
        input_freq_dim: int,
        vocab_size: int,
        conv_channels: Tuple[int, int] = (16, 16),
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Two conv layers similar in spirit to the paper's spectrogram branch
        self.conv1 = ConvBlock(
            in_channels=1,
            out_channels=conv_channels[0],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv2 = ConvBlock(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # We reduce frequency dimension but keep time resolution intact.
        # This is useful for CTC because output length depends on time steps.
        self.freq_pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.freq_pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        reduced_freq_dim = input_freq_dim
        reduced_freq_dim = self._pool_out_dim(reduced_freq_dim, kernel=2, stride=2)
        reduced_freq_dim = self._pool_out_dim(reduced_freq_dim, kernel=2, stride=2)

        rnn_input_dim = conv_channels[1] * reduced_freq_dim

        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.rnn_bn = nn.BatchNorm1d(lstm_hidden_size * 2)
        self.classifier = nn.Linear(lstm_hidden_size * 2, vocab_size)

    @staticmethod
    def _pool_out_dim(size: int, kernel: int, stride: int, padding: int = 0) -> int:
        return ((size + 2 * padding - kernel) // stride) + 1

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Time dimension is preserved because pooling only acts on frequency.
        """
        return input_lengths.clone()

    def forward(
        self,
        features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> CRNNOutput:
        """
        Args:
            features: [B, T, F]
            input_lengths: [B]

        Returns:
            CRNNOutput with:
              log_probs: [T_out, B, vocab_size]
              output_lengths: [B]
        """
        # [B, T, F] -> [B, 1, T, F]
        x = features.unsqueeze(1)

        x = self.conv1(x)
        x = self.freq_pool1(x)

        x = self.conv2(x)
        x = self.freq_pool2(x)

        # x is now [B, C, T, F_reduced]
        B, C, T, F_red = x.shape

        # Rearrange for recurrent layers:
        # [B, C, T, F_red] -> [B, T, C * F_red]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F_red)

        output_lengths = self._compute_output_lengths(input_lengths)

        # RNN output: [B, T, 2*hidden]
        x, _ = self.rnn(x)

        # BatchNorm1d expects [B, C, T], so transpose
        x = x.transpose(1, 2)          # [B, 2H, T]
        x = self.rnn_bn(x)
        x = x.transpose(1, 2)          # [B, T, 2H]

        # Classifier: [B, T, vocab]
        logits = self.classifier(x)

        # CTC expects [T, B, vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1).contiguous()

        return CRNNOutput(
            log_probs=log_probs,
            output_lengths=output_lengths,
        )
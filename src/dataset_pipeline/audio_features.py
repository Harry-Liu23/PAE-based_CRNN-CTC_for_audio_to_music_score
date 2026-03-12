from __future__ import annotations

import torch


class LogSTFTExtractor(torch.nn.Module):
    """
    waveform -> STFT magnitude -> log-spaced filterbank -> log amplitude

    Output:
        [time_frames, n_log_bins]
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 65.40639132514966,  # C2
        bins_per_octave: int = 24,
        n_octaves: int = 6,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.n_log_bins = bins_per_octave * n_octaves
        self.eps = eps

        self.register_buffer("window", torch.hamming_window(n_fft), persistent=False)
        self.register_buffer("filterbank", self._build_log_filterbank(), persistent=False)

    def _build_log_filterbank(self) -> torch.Tensor:
        n_freq_bins = self.n_fft // 2 + 1
        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freq_bins)

        centers = []
        for k in range(self.n_log_bins):
            freq = self.f_min * (2.0 ** (k / self.bins_per_octave))
            centers.append(freq)
        centers = torch.tensor(centers, dtype=torch.float32)

        boundaries = torch.zeros(self.n_log_bins + 2, dtype=torch.float32)
        boundaries[1:-1] = centers
        boundaries[0] = centers[0] / (2.0 ** (1.0 / self.bins_per_octave))
        boundaries[-1] = centers[-1] * (2.0 ** (1.0 / self.bins_per_octave))

        fb = torch.zeros(n_freq_bins, self.n_log_bins, dtype=torch.float32)

        for i in range(self.n_log_bins):
            left = boundaries[i]
            center = boundaries[i + 1]
            right = boundaries[i + 2]

            up_slope = (fft_freqs - left) / max(center - left, 1e-12)
            down_slope = (right - fft_freqs) / max(right - center, 1e-12)
            tri = torch.minimum(up_slope, down_slope)
            tri = torch.clamp(tri, min=0.0)
            fb[:, i] = tri

        fb = fb / fb.sum(dim=0, keepdim=True).clamp_min(1e-12)
        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [num_samples] or [1, num_samples]

        Returns:
            features: [time_frames, n_log_bins]
        """
        if waveform.dim() == 2:
            if waveform.size(0) != 1:
                raise ValueError(f"Expected mono waveform [1, T], got {waveform.shape}")
            waveform = waveform.squeeze(0)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()  # [freq, time]

        features = mag.transpose(0, 1) @ self.filterbank  # [time, n_log_bins]
        features = torch.log(features + self.eps)

        return features
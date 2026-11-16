from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised in dependency checks
        raise RuntimeError("install the ML dependencies from requirements-ml.txt") from exc
    return torch, nn


@dataclass(frozen=True)
class BaselineCNNConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    channels: tuple[int, ...] = (32, 64, 128)
    embedding_dim: int = 256
    dropout: float = 0.2
    specaugment: bool = True
    max_frequency_mask_bins: int = 8


def _mfm(block, tensor):
    torch, _ = _require_torch()
    values = block(tensor)
    left, right = torch.chunk(values, 2, dim=1)
    return torch.maximum(left, right)


def _mfm_block(nn, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
    class MFMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels * 2, 3, stride=stride, padding=1)
            self.norm = nn.BatchNorm2d(out_channels * 2)

        def forward(self, tensor):
            values = self.norm(self.conv(tensor))
            left, right = values.chunk(2, dim=1)
            return left.maximum(right)

    return MFMBlock()


class ResidualBlock:
    """Factory for a small ResNet-18-compatible audio block."""

    @staticmethod
    def build(nn, in_channels: int, out_channels: int, stride: tuple[int, int]):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.skip = nn.Identity() if (in_channels == out_channels and stride == (1, 1)) else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

            def forward(self, tensor):
                residual = self.skip(tensor)
                tensor = torch.relu(self.bn1(self.conv1(tensor)))
                return torch.relu(self.bn2(self.conv2(tensor)) + residual)

        torch, _ = _require_torch()
        return Block()


class BaselineCNN:
    """STFT + LCNN/ResNet-style classifier with a shared ``(logits, embedding)`` output."""

    def __new__(cls, config: BaselineCNNConfig | None = None, backbone: str = "lcnn"):
        torch, nn = _require_torch()

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.config, self.backbone_name = config or BaselineCNNConfig(), backbone
                if self.backbone_name not in {"lcnn", "resnet18"}:
                    raise ValueError("backbone must be 'lcnn' or 'resnet18'")
                channels = self.config.channels
                layers = []
                in_channels = 1
                for index, out_channels in enumerate(channels):
                    stride = (1, 1) if index == 0 else (2, 2)
                    if self.backbone_name == "lcnn":
                        layers.append(_mfm_block(nn, in_channels, out_channels, stride))
                    else:
                        layers.extend([ResidualBlock.build(nn, in_channels, out_channels, stride), ResidualBlock.build(nn, out_channels, out_channels, (1, 1))])
                    in_channels = out_channels
                self.encoder = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, self.config.embedding_dim), nn.ReLU(), nn.Dropout(self.config.dropout))
                self.classifier = nn.Linear(self.config.embedding_dim, 2)

            def _features(self, waveform):
                window = torch.hann_window(self.config.win_length, device=waveform.device, dtype=waveform.dtype)
                spectrum = torch.stft(waveform, n_fft=self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length, window=window, return_complex=True, center=True)
                features = torch.log1p(spectrum.abs()).unsqueeze(1)
                mean = features.mean(dim=(2, 3), keepdim=True)
                std = features.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
                features = (features - mean) / std
                if self.training and self.config.specaugment and self.config.max_frequency_mask_bins > 0:
                    width = min(self.config.max_frequency_mask_bins, features.shape[2])
                    start = int(torch.randint(0, features.shape[2] - width + 1, ()).item())
                    features[:, :, start:start + width, :] = 0
                return features

            def forward(self, waveform):
                if waveform.ndim != 2: raise ValueError("waveform must have shape [batch, time]")
                features = self.encoder(self._features(waveform))
                embedding = self.embedding(self.pool(features))
                return self.classifier(embedding), embedding

        return Module()

from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the ML dependencies from requirements-ml.txt") from exc
    return torch, nn


@dataclass(frozen=True)
class XLSRAASISTConfig:
    model_name: str = "facebook/wav2vec2-xls-r-300m"
    sample_rate: int = 16_000
    truncate_layers: int | None = None
    freeze_feature_extractor: bool = True
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 2e-4
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1


class GraphAttentionBackend:
    """AASIST-style temporal graph head using self/neighbor attention and attentive pooling."""

    def __new__(cls, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        torch, nn = _require_torch()

        class Backend(nn.Module):
            def __init__(self):
                super().__init__(); self.projection=nn.Linear(input_dim, hidden_dim); self.attention=nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True); self.norm=nn.LayerNorm(hidden_dim); self.gate=nn.Linear(hidden_dim, 1); self.dropout=nn.Dropout(dropout); self.classifier=nn.Linear(hidden_dim,2)
            def forward(self, sequence):
                sequence=self.projection(sequence); previous=torch.roll(sequence,1,1); following=torch.roll(sequence,-1,1); neighborhood=(sequence+previous+following)/3; attended,_=self.attention(sequence,neighborhood,neighborhood,need_weights=False); sequence=self.norm(sequence+self.dropout(attended)); weights=torch.softmax(self.gate(sequence).squeeze(-1),dim=1).unsqueeze(-1); embedding=(sequence*weights).sum(1); return self.classifier(embedding), embedding
        return Backend()


class XLSRAASIST:
    """XLS-R encoder with learned sensitive-layer selection and graph-attention head."""

    def __new__(cls, config: XLSRAASISTConfig | None = None, encoder=None):
        torch, nn = _require_torch(); cfg = config or XLSRAASISTConfig()
        try:
            from transformers import AutoModel
        except ImportError as exc: raise RuntimeError("install transformers from requirements-ml.txt") from exc

        class Module(nn.Module):
            def __init__(self):
                super().__init__(); self.config=cfg; self.encoder=encoder if encoder is not None else AutoModel.from_pretrained(cfg.model_name, output_hidden_states=True); self.hidden_size=self.encoder.config.hidden_size; self.layer_weights=nn.Parameter(torch.zeros(self._layer_count())); self.head=GraphAttentionBackend(self.hidden_size,cfg.hidden_dim,cfg.num_heads,cfg.dropout)
                if cfg.freeze_feature_extractor and hasattr(self.encoder,"feature_extractor"):
                    for parameter in self.encoder.feature_extractor.parameters(): parameter.requires_grad=False
            def _layer_count(self):
                configured=getattr(self.encoder.config,"num_hidden_layers",12)+1; return min(configured,(self.config.truncate_layers+1) if self.config.truncate_layers else configured)
            def forward(self,waveform):
                if waveform.ndim != 2: raise ValueError("waveform must have shape [batch, time]")
                outputs=self.encoder(input_values=waveform, output_hidden_states=True); hidden=list(outputs.hidden_states[:self.layer_weights.numel()]); weights=torch.softmax(self.layer_weights[:len(hidden)],dim=0); selected=torch.stack(hidden,dim=0); selected=(selected*weights[:len(hidden)].view(-1,1,1,1)).sum(0); return self.head(selected)
            def parameter_groups(self):
                encoder_parameters=[p for p in self.encoder.parameters() if p.requires_grad]; encoder_ids={id(p) for p in encoder_parameters}; head_parameters=[p for p in self.parameters() if id(p) not in encoder_ids]; return [{"params":encoder_parameters,"lr":cfg.encoder_learning_rate},{"params":head_parameters,"lr":cfg.head_learning_rate}]
        return Module()

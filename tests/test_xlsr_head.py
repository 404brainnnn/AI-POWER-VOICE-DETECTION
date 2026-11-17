import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from models.neural.xlsr_aasist import GraphAttentionBackend, XLSRRAASIST, XLSRRAASISTConfig


def test_graph_head_output_shapes():
    head = GraphAttentionBackend(32, 16, 4)
    logits, embedding = head(torch.randn(2, 20, 32))
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 16)


def test_xlsr_wrapper_output_shapes_without_download():
    config = transformers.Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, conv_dim=(16, 16, 16, 16, 16, 16, 16), conv_stride=(2, 2, 2, 2, 2, 2, 2), conv_kernel=(5, 3, 3, 3, 3, 2, 2))
    encoder = transformers.Wav2Vec2Model(config)
    model = XLSRRAASIST(XLSRRAASISTConfig(truncate_layers=2, hidden_dim=16), encoder=encoder)
    logits, embedding = model(torch.randn(2, 16000))
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 16)

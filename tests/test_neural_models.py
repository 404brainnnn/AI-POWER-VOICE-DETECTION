import pytest

torch = pytest.importorskip("torch")

from models.neural.baseline_cnn import BaselineCNN, BaselineCNNConfig


def test_baseline_lcnn_output_shapes():
    model = BaselineCNN(BaselineCNNConfig(n_fft=64, win_length=32, hop_length=16, channels=(8, 16), embedding_dim=24), "lcnn")
    logits, embedding = model(torch.randn(2, 1024))
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 24)


def test_baseline_resnet_output_shapes():
    model = BaselineCNN(BaselineCNNConfig(n_fft=64, win_length=32, hop_length=16, channels=(8, 16), embedding_dim=24), "resnet18")
    logits, embedding = model(torch.randn(2, 1024))
    assert logits.shape == (2, 2)
    assert embedding.shape == (2, 24)


def test_baseline_can_overfit_tiny_batch():
    torch.manual_seed(7)
    model = BaselineCNN(BaselineCNNConfig(n_fft=32, win_length=16, hop_length=8, channels=(8, 16), embedding_dim=16, specaugment=False), "lcnn")
    inputs = torch.randn(4, 512)
    labels = torch.tensor([0, 1, 0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = None
    for _ in range(80):
        logits, _ = model(inputs); loss = torch.nn.functional.cross_entropy(logits, labels)
        initial = loss.item() if initial is None else initial
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert loss.item() < initial

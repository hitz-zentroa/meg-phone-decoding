# tests/test_nn_archs.py
"""
Smoke-tests for the tiny neural nets defined in `mpd.models.nn_archs`.

We only verify *tensor plumbing* (shapes, dtypes, activations).  Heavy-duty
training is covered in the higher-level `test_models_training.py` module.
"""

import pytest
import torch

from mpd.models import nn_archs as na

# Helpers & fixtures


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Force the models to run on *CPU* even if CUDA is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(name="random_on_interval")
def random_on_interval_fixture():
    """Return a helper that makes random tensors in a numeric interval."""

    def factory(shape, low=-1.0, high=1.0):
        return (high - low) * torch.rand(shape) + low

    return factory


# ANNModel


@pytest.mark.parametrize("hidden_layers", [None, [8, 4]])
@pytest.mark.parametrize("classes", [2, 3])
def test_ann_model_forward(classes, hidden_layers, random_on_interval):
    """Test the forward method of the ANN models."""
    model = na.ANNModel(num_features=12, classes=classes, hidden_layers=hidden_layers)

    x = random_on_interval((5, 12))  # (batch, features)
    out = model(x)

    assert out.shape == (5, classes)

    if classes == 2:  # sigmoid must be applied
        assert torch.all(out >= 0) and torch.all(out <= 1)
    else:  # raw logits – range unbounded
        assert out.dtype == torch.float32


# CNNModel


def test_cnn_model_forward(random_on_interval):
    """Test the forward method of the CNN models."""
    batch, channels, num_feat, classes = 4, 3, 60, 5
    model = na.CNNModel(
        num_features=num_feat,
        classes=classes,
        input_channels=channels,
        hidden_layers=None,  # single-layer head
        downsampling_factor=10,  # divides num_feat exactly (60 // 10 = 6)
    )

    x = random_on_interval((batch, channels, num_feat))
    out = model(x)

    assert out.shape == (batch, classes)  # (B, C)
    assert out.requires_grad


# DyslexNetTransformer


@pytest.mark.parametrize("classes", [2, 4])
def test_dyslexnet_tiny_forward(classes, random_on_interval):
    """Use small hyper-parameters so that the forward‐pass is fast and light."""
    seq_len, sensors = 10, 6
    model = na.DyslexNetTransformer(
        input_dim=sensors,
        seq_len=seq_len,
        num_classes=classes,
        hidden_size=32,  # tiny!
        emb_size=16,
        factor_size=8,
        num_heads=4,
        num_layers=2,
    )

    x = random_on_interval((2, sensors, seq_len))  # (batch, channels, time)
    out = model(x)

    assert out.shape == (2, classes)
    # Logits should be float tensor, no activation enforced by model
    assert out.dtype == torch.float32

"""Unit tests for the data-loader helpers in *mpd.models.utils*."""

import numpy as np
import pytest
import torch

from mpd.models.utils import encode_labels, prepare_data, prepare_data_fold


# encode_labels
def test_encode_labels_returns_consecutive_ints():
    """Tests the encode_labels function to return consecutive label IDs."""
    y = ["cat", "dog", "cat", "bird"]
    encoded, classes_ = encode_labels(y)

    # all elements are ints starting at 0 and contiguous
    assert encoded.dtype.kind in "iu"
    assert set(encoded) == {0, 1, 2}
    # classes_ preserves the order chosen by LabelEncoder (alphabetical)
    assert list(classes_) == ["bird", "cat", "dog"]
    # round-trip: original label == classes_[encoded]
    assert [classes_[i] for i in encoded] == y


# prepare_data
@pytest.mark.parametrize("n_samples, n_features", [(100, 10), (17, 3)])
@pytest.mark.parametrize("classes", [2, 4])
def test_prepare_data_split_and_types(n_samples, n_features, classes):
    """Verify the prepare_data function splits and types."""
    rng = np.random.default_rng(0)
    xdata = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, classes, size=n_samples)

    tr, vl = prepare_data(xdata, y, batch_size=8, split_ratio=0.25, classes=classes)

    # dataset sizes respected
    n_tr = len(tr.dataset)
    n_vl = len(vl.dataset)
    assert n_tr + n_vl == n_samples
    # 25 % (+/-1 due to rounding)
    assert abs(n_vl - 0.25 * n_samples) <= 1

    # dtypes: float32 inputs; label dtype depends on #classes
    xb, yb = next(iter(tr))
    assert xb.dtype == torch.float32
    if classes > 2:
        assert yb.dtype == torch.long
        assert yb.ndim == 1
    else:
        assert yb.dtype == torch.float32


# prepare_data_fold
def test_prepare_data_fold_preserves_exact_sizes():
    """Verify the prepare_data_fold function splits."""
    rng = np.random.default_rng(1)
    xdata_tr = rng.normal(size=(30, 5))
    xdata_vl = rng.normal(size=(12, 5))
    ytr = rng.integers(0, 3, size=30)
    yvl = rng.integers(0, 3, size=12)

    dl_tr, dl_vl = prepare_data_fold(
        xdata_tr, xdata_vl, ytr, yvl, batch_size=7, classes=3
    )

    assert len(dl_tr.dataset) == 30
    assert len(dl_vl.dataset) == 12

    _, yb = next(iter(dl_tr))
    # labels must be int64/long for CE loss in multi-class
    assert yb.dtype == torch.long

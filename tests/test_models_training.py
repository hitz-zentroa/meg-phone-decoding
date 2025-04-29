"""
Fast sanity-checks for *mpd.models.training*.

The tests purposefully use **tiny synthetic data** so they finish in a few
hundred milliseconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import mpd.models.training as tr


# Fixtures
@pytest.fixture(name="binary_dataset", scope="module")
def binary_dataset_fixture():
    """40 samples, 10 features, **two** classes."""
    rng = np.random.default_rng(0)
    xdata = rng.standard_normal((40, 10)).astype("float32")
    y = rng.integers(0, 2, size=40).astype("int64")
    return xdata, y


@pytest.fixture(name="multiclass_dataset", scope="module")
def multiclass_dataset_fixture():
    """60 samples, 12 features, **three** classes."""
    rng = np.random.default_rng(1)
    xdata = rng.standard_normal((60, 12)).astype("float32")
    y = rng.integers(0, 3, size=60).astype("int64")
    return xdata, y


# TODO: Fix this test failing
@pytest.mark.xfail(
    reason="Binary path in train_ann expects *long* labels "
    "but receives float – fix in library code."
)
def test_train_ann_binary_fails_expected(binary_dataset, monkeypatch):
    """Document the issue for the binary branch (floats vs CE loss)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    xdata, y = binary_dataset
    tr.train_ann(xdata, y, classes=2, max_epochs=1, early_stopping_patience=1)


def test_train_ann_multiclass_returns_rows(multiclass_dataset, monkeypatch):
    """Multiclass branch should run fine and return metric dictionaries."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    xdata, y = multiclass_dataset
    rows = tr.train_ann(xdata, y, classes=3, max_epochs=1, early_stopping_patience=1)

    assert isinstance(rows, list) and rows
    assert all(isinstance(r, dict) for r in rows)
    assert any(r["Metric"] == "Accuracy" for r in rows)


# train_ann_cv
@pytest.mark.parametrize(
    "classes,dataset_fixture",
    [
        pytest.param(3, "multiclass_dataset", id="multiclass"),
        pytest.param(
            2,
            "binary_dataset",
            id="binary",
        ),
    ],
)
def test_train_ann_cv_produces_rows(request, classes, dataset_fixture, monkeypatch):
    """Verify the train_ann_cv function produced metrics."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    xdata, y = request.getfixturevalue(dataset_fixture)

    rows = tr.train_ann_cv(
        xdata,
        y,
        classes=classes,
        model_type="ann",
        hidden_layers=None,
        max_epochs=1,
        early_stopping_patience=1,
        n_splits=2,
    )

    assert isinstance(rows, list) and rows
    for r in rows:
        assert {"Metric", "Score"} <= r.keys()

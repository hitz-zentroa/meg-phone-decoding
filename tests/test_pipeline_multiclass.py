"""Unit tests for mpd.pipelines.multiclass.train_model_multiclass."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import mpd.pipelines.multiclass as mc


# Dummy helpers
class _DummyEpochs:
    def __init__(self, phones):
        self.event_id = {p: i for i, p in enumerate(phones)}
        ids = np.tile(list(self.event_id.values()), 3)  # 6 trials
        self.events = np.column_stack([ids, ids, ids])
        self._data = np.zeros((len(ids), 2, 3))

    def equalize_event_counts(self, *_):
        """Equalize the number of trials in each condition.

        Returns
        -------
        epochs : instance of `Epochs`
            The modified instance. It is modified in-place.
        indices : `array` of `int`
            Indices from the original events list that were dropped.
        """

    def get_data(self):
        """Get all epochs as a 3D array.

        Returns
        -------
        data : `array` of shape (n_epochs, n_channels, n_times)
            The epochs data. Will be a copy when `copy=True` and will be a view
            when possible when `copy=False`.
        """
        return self._data


def _patch_dataset(monkeypatch, phones):
    monkeypatch.setattr(mc, "read_raw_with_annotations", lambda *_, **__: (None, None))
    monkeypatch.setattr(
        mc,
        "get_epochs",
        lambda *_, **__: (_DummyEpochs(phones), ["MEG0111", "MEG0121"]),
    )
    monkeypatch.setattr(mc, "wavelet_denoise", lambda x, *_: x)


def _make_args(model_type: str):
    return SimpleNamespace(
        data_path="/fake",
        subject="subj",
        task="self",
        phones=["a", "e"],
        limit=None,
        model_type=model_type,
        wavelet=False,
        meg=None,
        l_freq=None,
        h_freq=None,
        decim=10,
        lr=1e-4,
        hidden_layers=None,
        adam=False,
        dataset=None,
    )


# Tests
def test_ann_path_selects_neural_backend(monkeypatch):
    """ANN keyword -> train_ann_cv is called and its metrics propagated."""
    _patch_dataset(monkeypatch, ["a", "e"])

    # stub that mimics ANN backend behaviour
    called = MagicMock(return_value=[{"Metric": "Accuracy", "Score": 0.9}])
    monkeypatch.setattr(mc, "train_ann_cv", called)

    # linear backend should never be used in this branch
    monkeypatch.setattr(mc, "train_logistic_regression", lambda *_, **__: None)

    df, weights = mc.train_model_multiclass(**vars(_make_args("ann")))
    called.assert_called_once()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df["Metric"]) == {"Accuracy"}
    assert weights is None


def test_linear_path(monkeypatch):
    """Test that train_model_multiclass calls the machine learning models.

    Verify that *train_model_multiclass* dispatches to the **classical**
    (scikit-learn) backend when the requested model type does **not**
    contain the substrings `"ann"`, `"cnn"` or `"dyslexnet"`.
    """
    _patch_dataset(monkeypatch, ["a", "e"])
    called = MagicMock(return_value=([{"Metric": "Accuracy", "Score": 0.95}], None))
    monkeypatch.setattr(mc, "train_logistic_regression", called)
    monkeypatch.setattr(mc, "train_ann_cv", lambda *_, **__: None)
    df, _ = mc.train_model_multiclass(**vars(_make_args("elasticnet")))
    called.assert_called_once()
    assert not df.empty

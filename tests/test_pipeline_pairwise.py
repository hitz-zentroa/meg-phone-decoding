"""
Unit tests for mpd.pipelines.pairwise.train_model.

Heavy I/O and ML work is stubbed out so the test runs in milliseconds.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import mpd.pipelines.pairwise as pw


# Dataset stubs
class _DummyEpochs:
    """Mimic the bits of `mne.Epochs` used by the pipeline."""

    def __init__(self, phones, n_trials=4):
        self.event_id = {p: i for i, p in enumerate(phones)}
        ids = np.tile(list(self.event_id.values()), n_trials // len(phones))
        self.events = np.column_stack([ids, ids, ids])
        self._data = np.zeros((len(ids), 2, 3))  # (trials, channels, times)

    def __getitem__(self, _):
        return self  # allow `epochs[phone1, phone2]`

    def equalize_event_counts(self, *_):
        """Equalize the number of trials in each condition.

        Returns
        -------
        epochs : instance of `Epochs`
            The modified instance. It is modified in-place.
        indices : `array` of `int`
            Indices from the original events list that were dropped.
        """

    def get_data(self, *_, **__):
        """Get all epochs as a 3D array.

        Returns
        -------
        data : `array` of shape (n_epochs, n_channels, n_times)
            The epochs data. Will be a copy when `copy=True` and will be a view
            when possible when `copy=False`.
        """
        return self._data


def _patch_dataset(monkeypatch, phones):
    """Patch loaders, epoch extractor and denoiser."""
    monkeypatch.setattr(pw, "read_raw_with_annotations", lambda *_, **__: (None, None))
    monkeypatch.setattr(
        pw,
        "get_epochs",
        lambda *_, **__: (_DummyEpochs(phones), ["MEG0111", "MEG0121"]),
    )
    monkeypatch.setattr(pw, "wavelet_denoise", lambda x, *_: x)


# Helpers
def _make_args(model_type="elasticnet"):
    """Namespace with just the attributes accessed by `train_model`."""
    return SimpleNamespace(
        data_path="/fake",
        subject="subj",
        task="self",
        phones=["a", "e", "s"],  # 3 phones -> 3 unordered pairs
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
@pytest.mark.parametrize(
    "model_type, backend_attr",
    [
        ("ann", "train_ann_cv"),  # neural path
        ("elasticnet", "train_logistic_regression"),  # linear path
    ],
)
def test_pairwise_pipeline(monkeypatch, model_type, backend_attr):
    """Pipeline must call the correct backend and propagate its results."""
    phones = ["a", "e", "s"]
    _patch_dataset(monkeypatch, phones)

    # stub backends
    ann_return = [{"Metric": "Accuracy", "Score": 0.90}]
    lin_return = (
        [{"Metric": "Accuracy", "Score": 0.95}],
        [{"Fold": 0, "Channel": "MEG0111", "Weight": 0.1}],
    )

    ann_mock = MagicMock(return_value=ann_return)
    lin_mock = MagicMock(return_value=lin_return)

    monkeypatch.setattr(pw, "train_ann_cv", ann_mock)
    monkeypatch.setattr(pw, "train_logistic_regression", lin_mock)

    # run pipeline
    df_scores, df_weights = pw.train_model(**vars(_make_args(model_type)))

    # backend chosen exactly `C(3,2)=3` times (once per phone-pair)
    expected_calls = 3
    if backend_attr == "train_ann_cv":
        assert ann_mock.call_count == expected_calls
        assert lin_mock.call_count == 0
    else:
        assert lin_mock.call_count == expected_calls
        assert ann_mock.call_count == 0

    # basic sanity on outputs
    assert isinstance(df_scores, pd.DataFrame)
    assert len(df_scores) == expected_calls  # one metric row per pair
    assert {"phone1", "phone2"} <= set(df_scores.columns)

    # weights only when linear backend is used
    if backend_attr == "train_logistic_regression":
        assert isinstance(df_weights, pd.DataFrame)
        assert len(df_weights) == expected_calls  # row per pair (our stub)
    else:
        assert df_weights is None

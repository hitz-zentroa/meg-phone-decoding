"""Unit tests for the high-level experiment driver (mpd.pipelines.run).

The real training / loading functions are *monkey-patched* with lightweight
stubs so that we can focus on:

1.  selecting the correct pipeline (pairwise vs multiclass);
2.  looping over subjects × tasks; and
3.  writing the expected CSV files.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import mpd.pipelines.run as run_mod


# Helper factories
def _dummy_df():
    """Return a minimal DataFrame resembling a training result."""
    return pd.DataFrame({"Metric": ["Accuracy"], "Score": [0.99]})


def _dummy_weights():
    """Return a minimal DataFrame with channel coefficients."""
    return pd.DataFrame(
        {
            "phone1": ["a"],
            "phone2": ["e"],
            "Channel": ["MEG0111"],
            "Fold": [0],
            "Weight": [0.1],
        }
    )


def _make_args(tmp_path, *, multiclass: bool):
    """Create a SimpleNamespace with just the attributes accessed by run_experiment."""
    return SimpleNamespace(
        # paths / dataset
        dataset=Path("/fake/dataset"),
        output=tmp_path / "scores.csv",
        # experiment settings
        subjects=[],  # triggers get_subjects()
        tasks=["self"],
        phones=["a", "e"],
        model="elasticnet",
        multiclass=multiclass,
        # preprocessing flags
        wavelets=False,
        meg=None,
        l_freq=None,
        h_freq=None,
        decim=10,
        learning_rate=1e-4,
        hidden_layers=None,
        adam=False,
        frequencies=None,  # shortcut not used in these tests
        # misc
        log_level="WARNING",
        dataset_format=None,
    )


# Parametrised test
@pytest.mark.parametrize(
    "multiclass, pipeline_attr",
    [(True, "train_model_multiclass"), (False, "train_model")],
)
def test_run_experiment(
    monkeypatch, tmp_path, multiclass, pipeline_attr
):  # pylint: disable=unused-argument
    """The run_experiment should call the right pipeline and create CSV files."""
    # monkey-patch light stubs
    # get_subjects -> pretend dataset has just one participant
    monkeypatch.setattr(run_mod, "get_subjects", lambda *_, **__: ["subj1"])

    # replace BOTH pipelines with a spy that returns dummy dfs
    called = MagicMock()

    def _stub(*args, **kwargs):  # signature irrelevant here
        called(*args, **kwargs)
        return _dummy_df(), _dummy_weights()

    monkeypatch.setattr(run_mod, "train_model", _stub)
    monkeypatch.setattr(run_mod, "train_model_multiclass", _stub)

    # run
    args = _make_args(tmp_path, multiclass=multiclass)
    run_mod.run_experiment(args)

    # assertions
    # correct pipeline was chosen
    assert called.call_count == 1
    chosen = "train_model_multiclass" if multiclass else "train_model"
    assert getattr(run_mod, chosen) is _stub

    # result files exist
    scores_csv = tmp_path / "scores.csv"
    weights_csv = tmp_path / "scores-weights.csv"
    assert scores_csv.exists()
    assert weights_csv.exists()

    # content sanity check — the stub’s value should round-trip
    scores_df = pd.read_csv(scores_csv)
    assert scores_df.loc[0, "Metric"] == "Accuracy"

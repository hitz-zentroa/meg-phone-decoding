"""
Unit-tests for *mpd.models.classical.train_logistic_regression*.

The suite focuses on three aspects:

1. **Backend dispatch** – the helper must extract per-channel weights only
   when the underlying estimator is *really* a LogisticRegression;
2. **Metric table** – returned rows always contain at least *Accuracy*;
3. **Multi-class branch** – when `multi_class=True` the function must add
   OvO / OvR AUC metrics.
"""

import numpy as np
import pytest

from mpd.models.classical import train_logistic_regression


# Fixtures
def _make_dataset(n_samples=60, n_features=8, n_classes=2, seed=0):
    """Return a small synthetic (xdata, y, ch_names) tuple."""
    rng = np.random.default_rng(seed)
    xdata = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)
    ch_names = [f"ch{idx}" for idx in range(n_features)]
    return xdata, y, ch_names


# Tests
@pytest.mark.parametrize("model_type", ["logistic_regression", "elasticnet"])
def test_weights_returned_for_logistic_families(model_type):
    """Elastic-Net and plain LR expose per-channel coefficients."""
    xdata, y, ch = _make_dataset()
    rows, weights = train_logistic_regression(
        xdata, y, model_type=model_type, ch_names=ch, seed=42
    )

    # At least one metric row and one weight entry must exist
    assert any(r["Metric"] == "Accuracy" for r in rows)
    assert len(weights) > 0
    # Every weight dict has the expected keys and valid channel names
    for w in weights:
        assert {"Fold", "Channel", "Weight"} <= w.keys()
        assert w["Channel"] in ch


@pytest.mark.parametrize("model_type", ["svm", "ridge", "lda"])
def test_no_weights_for_non_logistic_backends(model_type):
    """SVM / Ridge / LDA do **not** yield coefficient maps."""
    xdata, y, ch = _make_dataset()
    rows, weights = train_logistic_regression(
        xdata, y, model_type=model_type, ch_names=ch, seed=42
    )

    assert any(r["Metric"] == "Accuracy" for r in rows)
    assert not weights or len(weights) == 0


def test_multiclass_branch_adds_ovo_auc():
    """With `multi_class=True` the rows must include OvO / OvR AUC."""
    xdata, y, ch = _make_dataset(n_samples=90, n_classes=3)
    rows, _ = train_logistic_regression(
        xdata,
        y,
        model_type="logistic_regression",
        ch_names=ch,
        multi_class=True,
        seed=42,
    )

    metrics = {r["Metric"] for r in rows}
    assert {"AUC (OvO)", "AUC (OvR)", "Accuracy"} <= metrics

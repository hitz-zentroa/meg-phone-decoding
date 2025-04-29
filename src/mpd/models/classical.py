"""Cross-validated training for **linear** Machine Learning models.

The helper is deliberately *generic*:

* takes a 2-D matrix `xdata` and label vector `y`;
* chooses the concrete estimator from the *string* given in `model_type`;
* wraps it in a :class:`sklearn.pipeline.Pipeline` with
  :class:`sklearn.preprocessing.StandardScaler`;
* evaluates **five-fold stratified CV** with a rich metric dictionary; and
* (for linear models) returns the per-channel coefficients so they can be
  visualised as sensor maps.

Neural-network models live in `mpd.models.training` – this file only handles
the classical scikit-learn branch.
"""

import logging
import re

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_logistic_regression(
    xdata, y, model_type, ch_names, multi_class=False, seed=None
):  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches  # noqa: E501
    """Trains a logistic regression model for binary classification.

    Parameters
    ----------
    xdata : numpy.array
        The input data in (n_trials, n_channels, n_times) format.
    y : numpy.array
        The values to predict in (n_trials, ) format.
    model_type : str
        The name of the model to train.
    ch_names : list
        List of channel names sorted.
    multi_class : bool
        Whether to train in multi-class classification task.
    seed : int
        Affects the ordering of the indices in the cross-validation.

    Returns
    -------
    list
        Prediction scores as a list of `dict`s with `"Metric"` and `"Score"`
        as keys.
    list
        The model weights with "Fold", "Channel" and "Weight" values.
    """
    args = {"multi_class": "multinomial"} if multi_class else {}
    if "lda" in model_type:
        logging.warning("Model: LDA (%s)", model_type)
        model = LinearDiscriminantAnalysis()
    elif "svm" in model_type:
        logging.warning("Model: SVM (%s)", model_type)
        model = SVC(probability=True)
    elif "elasticnet" in model_type or "l1" in model_type:
        logging.warning("Model: Elasticnet (%s)", model_type)
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, **args
        )
    elif "ridge" in model_type:
        logging.warning("Model: Ridge Classifier (%s)", model_type)
        model = RidgeClassifier(**args)
    elif "log_loss" in model_type:
        logging.warning("Model: Standard Logistic Regression (%s)", model_type)
        model = SGDClassifier(loss="log_loss", **args)
    else:
        logging.warning("Model: Logistic Regression (%s)", model_type)
        model = LogisticRegression(**args)  # basic classifier

    # It is extremely important to scale the data before running the classifier
    clf = make_pipeline(
        StandardScaler(), model  # scale the data, original values are very small
    )

    # Run cross-validation.
    n_splits = 5
    # Define multiple scoring metrics
    if multi_class:
        scoring_metrics = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, average="micro", zero_division=0),
            "recall": make_scorer(recall_score, average="micro", zero_division=0),
            "f1": make_scorer(f1_score, average="micro", zero_division=0),
            "roc_auc_ovo": make_scorer(
                roc_auc_score, multi_class="ovo", needs_proba=True
            ),
            "roc_auc_ovr": make_scorer(
                roc_auc_score, multi_class="ovr", needs_proba=True
            ),
        }
    else:
        scoring_metrics = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
    if seed is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_splits)

    # Adjust cross-validation to return the estimator with the weights
    scores = cross_validate(
        clf, X=xdata, y=y, cv=cv, scoring=scoring_metrics, return_estimator=True
    )

    rows = []
    for name, score_list in scores.items():
        match = re.search(r"test_(.*)", name)
        if match:
            name = match.group(1).replace("_", " ").capitalize()
            if name == "Roc auc":
                name = "AUC"
            elif name == "Roc auc ovo":
                name = "AUC (OvO)"
            elif name == "Roc auc ovr":
                name = "AUC (OvR)"
            for score in score_list:
                row = {"Metric": name, "Score": score}
                rows.append(row)

    # Gather weights from estimators
    weights = []
    for idx, estimator in enumerate(scores["estimator"]):
        if "logisticregression" in estimator.named_steps:
            coefs = estimator.named_steps["logisticregression"].coef_[0]
            for coef, channel in zip(coefs, ch_names):
                weights.append({"Fold": idx, "Channel": channel, "Weight": coef})

    return rows, weights

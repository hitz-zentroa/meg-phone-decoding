"""Pipeline to train one multi-class model per (subject, task).

This pipeline that trains **one multi-class model per (subject, task)** to
predict *which* phone (out of *n* phones) is being produced.

It is structurally similar to :pymod:`mpd.pipelines.pairwise`, but instead of
creating *C(n, 2)* binary classifiers it builds a **single classifier** whose
label space is the list given in `--phones`.  The concrete model—linear or
neural—is chosen at run-time from the `--model` flag.

Typical use (never call this directly, let
:pyfunc:`mpd.pipelines.run.run_experiment` dispatch it):

Example
-------
```python
>>> df_scores, _ = train_model_multiclass(
...     data_path="/data/meg",
...     subject="selfp03",
...     task="self",
...     phones=["a", "e", "s"],
...     model_type="elasticnet",
... )
```
"""

import logging

import numpy as np
import pandas as pd
from termcolor import colored

from mpd.data import read_raw_with_annotations
from mpd.data.epochs import get_epochs
from mpd.features.preprocessing import wavelet_denoise
from mpd.models.classical import train_logistic_regression
from mpd.models.training import train_ann_cv
from mpd.utils.display import print_df


def train_model_multiclass(
    data_path,
    subject,
    task,
    phones=["e", "a", "s", "o"],
    limit=None,
    model_type="logistic_regression",
    wavelet=False,
    meg=None,
    l_freq=None,
    h_freq=None,
    decim=10,
    lr=1e-4,
    hidden_layers=None,
    adam=False,
    dataset=None,
):  # pylint: disable=dangerous-default-value,too-many-arguments,too-many-positional-arguments,too-many-locals,unused-argument  # noqa: E501
    """Train **one multi-class phone decoder** for the given subject and task.

    Parameters
    ----------
    data_path : Path or str
        Root directory of the MEG dataset (the folder that contains `meg/`).
    subject : str
        Subject identifier, e.g. `"selfp03"`.
    task : str
        Experimental task, e.g. `"listen"`, `"playback"`, `"self"`.
    phones : list of str, default `("e","a","s","o")`
        Phones that will form the class set.  Order defines label indices.
    limit : int or None
        *Unused* in the current implementation; kept for API parity.
    model_type : {"logistic_regression", "elasticnet", "ann", "cnn",
                  "dyslexnet", ...}
        Which model family to instantiate.  Anything containing
        `"ann"/"cnn"/"dyslexnet"` triggers the neural-network path,
        otherwise a classical linear model is used.
    wavelet : bool
        If `True` apply Daubechies-4 wavelet denoising before training.
    meg, l_freq, h_freq, decim
        Passed straight to :pyfunc:`mpd.data.epochs.get_epochs` to pick sensor
        types and perform basic filtering / decimation.
    lr : float
        Learning-rate forwarded to the neural optimizers.
    hidden_layers : list[int] or None
        Sizes of the hidden layers when `model_type` is a neural network.
    adam : bool
        Use Adam (+weight-decay) instead of plain SGD for neural models.
    dataset : str
        The name of the dataset to load.

    Returns
    -------
    df_scores : pandas.DataFrame
        One row per cross-validation fold × metric.
        Columns: `["Metric", "Score"]`.
    None
        Placeholder for weights (not produced in the multiclass pipeline).

    Notes
    -----
    The heavy lifting—cross-validated training and metric computation—is
    delegated to:

    * :pyfunc:`mpd.models.training.train_ann_cv`   (neural nets)
    * :pyfunc:`mpd.models.classical.train_logistic_regression` (linear models)

    This wrapper merely prepares data, calls the trainer, pretty-prints a small
    summary and returns the raw DataFrame to the caller.
    """
    print(f"Subject: {subject}, Task: {task}")
    raw, annotations = read_raw_with_annotations(
        data_path,
        subject,
        task,
        dataset=dataset,
    )
    phone_epochs, ch_names = get_epochs(
        raw,
        annotations,
        "phones",
        meg=meg,
        l_freq=l_freq,
        h_freq=h_freq,
        decim=decim,
        selected_phones=phones,
    )

    # Balance the dataset labels
    phone_epochs.equalize_event_counts(phone_epochs.event_id)

    # Extract data and labels
    phone_event_id = {v: k for k, v in phone_epochs.event_id.items()}
    y = np.array(
        [
            phone_event_id[event]
            for event in phone_epochs.events[:, 2]
            if event in phone_event_id
        ]
    )
    data = phone_epochs.get_data()
    n_trials = data.shape[0]

    # Preprocessing and training setup
    if wavelet:
        # Denoise the data using Wavelets
        data = wavelet_denoise(data, "db4", 2)
        logging.info("Denoised data: %s", data.shape)

    # Run cross-validation
    model_type = model_type.lower()
    if "ann" in model_type:
        data = data.reshape(n_trials, -1)  # Reshape data for training
        scores = train_ann_cv(
            data,
            y,
            classes=len(phones),
            lr=lr,
            hidden_layers=hidden_layers,
            model_type=model_type,
            adam=adam,
        )
    elif "cnn" in model_type:
        scores = train_ann_cv(
            data,
            y,
            classes=len(phones),
            lr=lr,
            hidden_layers=hidden_layers,
            model_type=model_type,
            adam=adam,
        )
    elif "dyslexnet" in model_type:
        scores = train_ann_cv(
            data,
            y,
            classes=len(phones),
            lr=lr,
            hidden_layers=hidden_layers,
            model_type=model_type,
            adam=adam,
        )
    else:
        data = data.reshape(n_trials, -1)  # (n_trials, 9982)
        scores, _ = train_logistic_regression(
            data, y, model_type, ch_names, multi_class=True
        )
        # TODO: weights not used here

    df = pd.DataFrame(scores)
    df_mean = df.groupby("Metric")["Score"].agg(["mean", "std"])
    color = "green" if df_mean.loc["Accuracy", "mean"] > 0.60 else "white"
    print(f"Subject: {subject}, Task: {task}")
    print_df(colored("Mean:", color), df_mean)
    return df, None

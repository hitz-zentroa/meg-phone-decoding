"""Binary (pair-wise) phone-decoding pipeline.

For each `(subject, task)` pair and for **every unordered combination of two
phones** in the list supplied via `--phones`, this pipeline:

1.  extracts epochs that correspond only to those two phones;
2.  optionally denoises / filters the signal;
3.  trains a *binary* classifier with 5-fold cross-validation
    (linear model or NN, chosen at run-time);
4.  stores per-fold metrics and, when applicable, the channel-level weights of
    linear models.

The outer orchestration loop is handled by
:pyfunc:`mpd.pipelines.run.run_experiment`; this module is a *helper* that does
the heavy lifting for **one** subject and task.
"""

import itertools
import logging

import mne
import numpy as np
import pandas as pd
from termcolor import colored
from tqdm.auto import tqdm

from mpd.data import read_raw_with_annotations
from mpd.data.epochs import get_epochs
from mpd.features.preprocessing import wavelet_denoise
from mpd.models.classical import train_logistic_regression
from mpd.models.training import train_ann_cv
from mpd.utils.display import print_df


def train_model(
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
):  # pylint: disable=dangerous-default-value,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements  # noqa: E501
    """Train *one classifier per phone-pair* for the given subject and task.

    Parameters
    ----------
    data_path : Path or str
        Root directory of the dataset (folder that contains `meg/`).
    subject : str
        Subject identifier, e.g. `"selfp04"`.
    task : str
        Experiment condition (`"listen"`, `"playback"`, or `"self"`).
    phones : list of str
        Phones to consider.  All `C(len(phones), 2)` unordered pairs are
        evaluated independently.
    limit : int or None
        Optional maximum number of epochs per class.  If a phone has more than
        `limit` epochs, the excess is discarded **after** balancing.
    model_type : str
        Name that selects the model family.  Sub-strings `"ann"`, `"cnn"`,
        `"dyslexnet"` trigger the neural-network path, everything else falls
        back to a classical linear model
        (logistic / ridge / elastic-net / SVM, see code).
    wavelet : bool
        Apply Daubechies-4 wavelet denoising (level 2) on the epochs before
        training.
    meg, l_freq, h_freq, decim
        Passed straight to :pyfunc:`mpd.data.epochs.get_epochs`.
    lr : float
        Learning rate for neural optimisers.
    hidden_layers : list[int] or None
        Sizes of fully-connected hidden layers for the ANN/CNN models.
    adam : bool
        Use Adam (+weight-decay) instead of SGD for neural models.
    dataset : str
        The name of the dataset to load.

    Returns
    -------
    df_scores : pandas.DataFrame
        Row per *fold × metric × phone-pair*.  Columns:
        `["Metric", "Score", "phone1", "phone2"]`.
    df_weights : pandas.DataFrame or None
        If the underlying model exposes per-channel weights (e.g. Elastic-Net),
        this DataFrame contains them; otherwise `None`.
    """
    raw, annotations = read_raw_with_annotations(
        data_path,
        subject,
        task,
        dataset=dataset,
    )
    phone_epochs, ch_names = get_epochs(
        raw, annotations, "phones", meg=meg, l_freq=l_freq, h_freq=h_freq, decim=decim
    )

    # ## Multivariate Statistics (Decoding / MVPA)

    phones = list(phones)
    phone_pairs = list(itertools.combinations(phones, 2))

    dfs, dfs_weights = [], []
    for pair in tqdm(phone_pairs, desc="Train"):
        logging.info("PAIR: '%s' vs '%s' (%s: %s)", pair[0], pair[1], subject, task)
        epochs = phone_epochs[pair[0], pair[1]]
        epochs.equalize_event_counts(epochs.event_id)

        # Limit the number of examples per class to 100
        if limit is not None:
            if len(epochs[pair[0]]) < limit and len(epochs[pair[1]]) < limit:
                return None
            if len(epochs[pair[0]]) > limit:
                epochs = mne.concatenate_epochs(
                    [epochs[pair[0]][:limit], epochs[pair[1]]]
                )
            if len(epochs[pair[1]]) > limit:
                epochs = mne.concatenate_epochs(
                    [epochs[pair[0]], epochs[pair[1]][:limit]]
                )

        # Create an vector with length = no. of trials.
        y = np.empty(len(epochs.events), dtype=int)  # (888,)

        # Which trials are LEFT, which are RIGHT?
        idx_left = epochs.events[:, 2] == epochs.event_id[pair[0]]
        idx_right = epochs.events[:, 2] == epochs.event_id[pair[1]]

        # Encode: LEFT = 0, RIGHT = 1.
        y[idx_left] = 0
        y[idx_right] = 1

        data = epochs.get_data(
            picks=["meg"]
        )  # (n_epochs, n_channels, n_times) = (n_trials, 306, 31)

        logging.info("data: %s", data.shape)
        n_trials = data.shape[0]
        logging.info("n_trials: %s", n_trials)

        # Preprocessing
        if wavelet:
            # Denoise the data using Wavelets
            data = wavelet_denoise(data, "db4", 2)
            logging.info("denoised data: %s", data.shape)

        # Model selection
        model_type = model_type.lower()
        weights = None
        if "ann" in model_type:
            data = data.reshape(n_trials, -1)  # (n_trials, 9982)
            scores = train_ann_cv(
                data,
                y,
                classes=2,
                lr=lr,
                hidden_layers=hidden_layers,
                model_type=model_type,
                adam=adam,
            )
        elif "cnn" in model_type:
            scores = train_ann_cv(
                data,
                y,
                classes=2,
                lr=lr,
                hidden_layers=hidden_layers,
                model_type=model_type,
                adam=adam,
            )
        elif "dyslexnet" in model_type:
            scores = train_ann_cv(
                data,
                y,
                classes=2,
                lr=lr,
                hidden_layers=hidden_layers,
                model_type=model_type,
                adam=adam,
            )
        else:
            data = data.reshape(n_trials, -1)  # (n_trials, 9982)
            logging.info("X: %s", data.shape)
            logging.info("y: %s", y.shape)
            scores, weights = train_logistic_regression(data, y, model_type, ch_names)
        df = pd.DataFrame(scores)
        df["phone1"], df["phone2"] = pair
        dfs.append(df)
        df_grouped = df.groupby("Metric")["Score"].agg(["mean", "std"])
        print_df("Pair: " + ",".join(pair) + f" ({subject}-{task})", df_grouped)

        # Save the weights
        if weights is not None:
            df_weights = pd.DataFrame(weights)
            df["phone1"], df["phone2"] = pair
            df_weights["phone1"], df_weights["phone2"] = pair
            dfs_weights.append(df_weights)
        else:
            dfs_weights = None

    # Concatenate the results
    df = pd.concat(dfs)  # scores
    if dfs_weights is not None:
        df_weights = pd.concat(dfs_weights)  # weights
    else:
        df_weights = None
    # Get the average to report in the screen
    df_mean = df.groupby("Metric")["Score"].agg(["mean", "std"])
    color = "green" if df_mean.loc["Accuracy", "mean"] > 0.60 else "white"
    print_df(colored("Mean:", color), df_mean)
    return df, df_weights

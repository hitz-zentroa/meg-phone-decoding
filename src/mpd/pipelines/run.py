"""High-level experiment driver.

The sole purpose of this module is to **coordinate** an experiment:

1. obtain the list of subjects that exist in the dataset;
2. decide whether to run the *pairwise* or *multiclass* pipeline, based on the
   command-line arguments parsed in :pymeth:`mpd.cli._parse_args`;
3. loop over every `(subject, task)` combination, delegate the actual model
   training, gather the resulting data frames, and
4. write the aggregated results to the CSV paths requested by the user.

No machine-learning logic lives here; that work is handled by
`mpd.pipelines.pairwise` and `mpd.pipelines.multiclass`.
"""

import logging

import pandas as pd
from tqdm.auto import tqdm

from mpd.data import get_subjects
from mpd.features.frequency import get_frequency_band
from mpd.pipelines.multiclass import train_model_multiclass
from mpd.pipelines.pairwise import train_model


def run_experiment(args):
    """Execute a complete decoding experiment.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace returned by :pyfunc:`mpd.cli._parse_args`.  Important
        fields are documented in the CLI help; the ones accessed here include
        `dataset`, `subjects`, `tasks`, `model`, `multiclass`,
        `phones` and the various preprocessing flags.

    Notes
    -----
    *The function is run for its side-effects only.*
    It writes two CSV files:

    * **scores** – `args.output`
      Classification metrics for every subject × task × phone(-pair).
    * **weights** – same name with `-weights` suffix (only if available)
      Channel-level coefficients returned by linear models.

    The returned value is `None`.
    """
    # Get the subject list from the dataset:
    if len(args.subjects) == 0:
        subjects = get_subjects(args.dataset, dataset=args.dataset_format)
    else:
        subjects = args.subjects

    # Translate --frequencies shortcut into (l_freq, h_freq)
    if args.frequencies is not None:
        args.l_freq, args.h_freq = get_frequency_band(args.frequencies)

    # Select the train pipline depending on the task
    if args.multiclass:
        logging.info("Training task: multi-class")
        train_model_func = train_model_multiclass
    else:
        logging.info("Training task: phone-pairs")
        train_model_func = train_model

    # Run training for every (subject, task) combination
    dfs = []
    dfs_weights = []
    for subject in tqdm(subjects, desc="Subject"):
        for task in tqdm(args.tasks, desc="Task"):
            # Train logistic regression:
            df, df_weights = train_model_func(
                args.dataset,
                subject,
                task,
                args.phones,
                limit=None,
                model_type=args.model,
                wavelet=args.wavelets,
                meg=args.meg,
                l_freq=args.l_freq,
                h_freq=args.h_freq,
                decim=args.decim,
                lr=args.learning_rate,
                hidden_layers=args.hidden_layers,
                adam=args.adam,
                dataset=args.dataset_format,
            )
            df["Subject"] = subject
            df["Task"] = task
            if df_weights is not None:
                df_weights["Subject"] = subject
                df_weights["Task"] = task

            # Reorder columns for style
            column_order = ["Subject", "Task"] + [
                col for col in df.columns if col not in ["Subject", "Task"]
            ]
            df = df[column_order]
            if df_weights is not None:
                column_order_weights = [
                    "Subject",
                    "Task",
                    "phone1",
                    "phone2",
                    "Channel",
                    "Fold",
                ] + [
                    col
                    for col in df_weights.columns
                    if col
                    not in ["Subject", "Task", "phone1", "phone2", "Channel", "Fold"]
                ]

            # Save the results:
            dfs.append(df)

            if df_weights is not None and all(
                col in df_weights.columns for col in column_order_weights
            ):
                df_weights = df_weights[  # pylint: disable=unsubscriptable-object
                    column_order_weights
                ]
                dfs_weights.append(df_weights)

    # Persist results
    logging.info("Saving results to %s.", args.output)
    df = pd.concat(dfs)
    df.to_csv(args.output, index=False)
    weights_output = str(args.output).replace(".csv", "-weights.csv")
    logging.info("Saving weights to %s.", weights_output)

    # Save the weights if the model returned them
    if dfs_weights and len(dfs_weights) > 0:
        df_weights = pd.concat(dfs_weights)
        df_weights.to_csv(weights_output, index=False)

"""Command–line front-end for the MEG-phone-decoder package.

*   **Parses** all user-supplied flags (`_parse_args`).
*   Sets the global log level.
*   Hands the resulting namespace to
    :pyfunc:`mpd.pipelines.run.run_experiment`, which does the heavy work.

The module is installed as a console-script entry-point called
`mpd-train` (see *pyproject.toml*).  Example::

Example
-------
```shell
mpd-train ~/datasets/bourguignon2020 \
    --model elasticnet \
    --meg grad \
    --wavelets \
    --frequencies lpbeta \
    --output logs/bourguignon2020_elasticnet_grad_wavelets_lpbeta.csv
```
"""

import argparse
import logging
import os
from pathlib import Path

from .config import TASKS
from .pipelines.run import run_experiment


def _parse_args():
    """Build the argument parser and return the parsed `argparse.Namespace`.

    Returns
    -------
    argparse.Namespace
        The namespace populated with the argument values.

    Notes
    -----
    Call `mpd-train --help` for the full list and default values.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Use Logistic Regression to predict phones in "
            "BCBL speech production dataset"
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Root path of the dataset.",
    )
    parser.add_argument(
        "--phones",
        "-p",
        nargs="+",
        default=["a", "e", "i", "o", "u", "s", "n", "l", "ɾ", "t̪"],
        help="List of phones to analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.splitext(__file__)[0] + ".csv",
        help="Path to CSV file to store the scores.",
    )
    parser.add_argument(
        "--subjects",
        "-s",
        nargs="*",
        default=[],
        help="The list of subject to include.",
    )
    parser.add_argument(
        "--tasks", "-t", nargs="+", default=TASKS, help="The list of tasks to include."
    )
    parser.add_argument(
        "--decim",
        "-d",
        type=int,
        default=10,
        help="Factor by which to subsample the data.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="logistic_regression",
        help="The name of the model to train.",
    )
    parser.add_argument(
        "--wavelets",
        "--wavelet",
        "-w",
        action="store_true",
        help="Whether to use wavelets to denoise the signals.",
    )
    parser.add_argument(
        "--multiclass",
        "--multi",
        action="store_true",
        help=("Whether to train a multi-class model instead of phone-pairs task."),
    )
    parser.add_argument(
        "--meg",
        default=None,
        help=(
            "It can be ‘mag’, ‘grad’, ‘planar1’ or ‘planar2’ to select only "
            "magnetometers, all gradiometers, or a specific type of "
            "gradiometer."
        ),
    )
    parser.add_argument(
        "--l_freq",
        "--lfreq",
        "-lf",
        type=float,
        default=None,
        help=(
            "For FIR filters, the lower pass-band edge; for IIR filters, the "
            "lower cutoff frequency. If None the data are only low-passed."
        ),
    )
    parser.add_argument(
        "--h_freq",
        "--hfreq",
        "-hf",
        type=float,
        default=None,
        help=(
            "For FIR filters, the upper pass-band edge; for IIR filters, the "
            "upper cutoff frequency. If None the data are only high-passed."
        ),
    )
    parser.add_argument(
        "--frequencies",
        "--freq",
        "-f",
        type=str,
        default=None,
        help=(
            "Name of the frequency band to use: delta, theta, alpha, beta, "
            "gamma, HGA."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-4,
        help="Learning rate to use for the neural models.",
    )
    parser.add_argument(
        "--hidden_layers",
        "-hl",
        type=int,
        nargs="*",
        default=None,
        help="Hidden layers for the neural models.",
    )
    parser.add_argument(
        "--adam",
        action="store_true",
        help="Whether to use Adam optimizer with weight decay.",
    )
    parser.add_argument(
        "--dataset-format",
        "--ds",
        default="bourguignon2020",
        help="Which dataset adapter to use (see mpd/data/ for the list).",
    )
    levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    parser.add_argument("--log-level", "-l", default="WARNING", choices=levels)
    args = parser.parse_args()
    return args


def main() -> None:
    """Entrypoint installed as the `mpd-train` console command.

    1. Parse CLI arguments.
    2. Configure the root logger.
    3. Delegate to :pyfunc:`mpd.pipelines.run.run_experiment`.
    """
    args = _parse_args()
    logging.basicConfig(level=args.log_level)
    run_experiment(args)


if __name__ == "__main__":
    main()

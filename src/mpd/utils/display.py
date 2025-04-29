"""Small helpers for pretty-printing pandas DataFrames in the terminal.

Currently the module exposes a single convenience wrapper around
:pyfunc:`tabulate.tabulate`, used throughout the pipelines to give a quick,
human-readable summary of cross-validation metrics.

The helper is **not** meant for programmatic use (e.g. inside notebooks); its
sole purpose is CLI feedback.
"""

from tabulate import tabulate


def print_df(title, df):
    """Pretty-print a *pandas* DataFrame as an ASCII table.

    Parameters
    ----------
    title : str
        Heading printed above the table.
    df : pandas.DataFrame
        Table to display.  If the DataFrame contains the columns `"mean"` and
        `"std"`, the function merges them into a single column formatted as
        `"mean ± std"` (both multiplied by 100 and rounded to two decimals).

    Notes
    -----
    * Values are scaled to percentages (`×100`) before rounding.
    * The table style is *PostgreSQL* (`tablefmt="psql"`).
    * The function prints directly to stdout and returns `None`.
    """
    print(title)
    df = df.applymap(lambda x: round(x * 100, 2))
    if "mean" in df.columns and "std" in df.columns:
        df["mean"] = df.apply(
            lambda row: f'{row["mean"]:.2f} \u00b1 {row["std"]:.2f}', axis=1
        )
        df = df.drop("std", axis=1)
    print(tabulate(df, headers="keys", tablefmt="psql"))

"""Unit tests for mpd.utils.display.print_df.

•  Verifies the “mean ± std” merging (and ×100 scaling).
•  Verifies the fallback path when *std* is absent.
We capture **stdout** with pytest’s *capsys* fixture instead of trying to
parse the `tabulate` output structurally.
"""

import pandas as pd

import mpd.utils.display as disp


# Helpers
def _run_and_capture(df, capsys):
    """Call the helper and return the *stdout* string (no stderr expected)."""
    disp.print_df("Results", df)
    return capsys.readouterr().out


# Tests
def test_merge_mean_std_and_scaling(capsys):
    """Verify the mean and std format in the output.

    If the DataFrame contains *mean* and *std*, they must be scaled by 100,
    rounded to two decimals and merged into "mean ± std".
    """
    df = pd.DataFrame({"mean": [0.9234], "std": [0.0123]}, index=["Accuracy"])
    out = _run_and_capture(df, capsys)

    # heading printed first
    assert out.startswith("Results")

    # original metric label preserved
    assert "Accuracy" in out

    # 0.9234 ×100 -> 92.34  ; 0.0123 ×100 -> 1.23  (rounded to 2 decimals)
    assert "92.34 ± 1.23" in out


def test_single_score_column(capsys):
    """Verify what happens without std.

    When the DataFrame has no *std* column the helper must simply scale/round
    every numeric value.
    """
    df = pd.DataFrame({"Score": [0.81327]}, index=["F1"])
    out = _run_and_capture(df, capsys)

    assert "F1" in out
    # 0.81327 ×100 -> 81.33 (two-decimal rounding)
    assert "81.33" in out

"""Unit tests for mpd.data.annotations.find_nearest_meg_sample."""

import numpy as np
import pytest

from mpd.data.annotations import find_nearest_meg_sample


# Fixtures
@pytest.fixture(name="mapping")
def mapping_fixture():
    """A monotonically-increasing tds-vector.

    index  -> MEG sample, value -> WAV sample

    We keep it tiny for clarity:

        idx : 0   1   2   3   4   5
        val : 0  44  88 132 176 220
    """
    return np.arange(0, 6) * 44  # step = 44 Hz -> easier maths


# Tests
@pytest.mark.parametrize(
    "wav, expected",
    [
        (0, 0),  # exact first element
        (43, 1),  # nearest is index 1 (44)
        (45, 1),  # still index 1
        (88, 2),  # exact hit
        (150, 3),  # halfway between 132 and 176 -> 132 (idx 3) is closer
        (1000, 5),  # beyond end -> last index returned
    ],
)
def test_basic_behaviour(mapping, wav, expected):
    """Verify some basic examples in the find_nearest_meg_sample function."""
    assert find_nearest_meg_sample(wav, mapping) == expected


def test_start_parameter_speeds_up_search(mapping):
    """The find_nearest_meg_sample returns the correct index.

    When we pass the previous answer via `start`, the function must still
    return the correct index but scan only the tail of the array.
    """
    # first lookup
    first = find_nearest_meg_sample(90, mapping)  # ~ index 2
    assert first == 2

    # next WAV position is slightly later -> correct answer is index 3
    second = find_nearest_meg_sample(140, mapping, start=first)
    assert second == 3


def test_exact_match_within_scan(mapping):
    """If the WAV sample is exactly present, the corresponding index is returned."""
    assert find_nearest_meg_sample(176, mapping) == 4

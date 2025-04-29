"""Unit tests for mpd.features.frequency.get_frequency_band."""

import pytest

from mpd.features.frequency import get_frequency_band


@pytest.mark.parametrize(
    "band, expected",
    [
        ("delta", (0.2, 3.99)),
        ("theta", (4, 7.99)),
        ("alpha", (8, 13.99)),
        ("beta", (14, 31.99)),
        ("gamma", (32, 100)),
        ("hga", (60, 300)),
        ("lpTheta", (0.2, 7.99)),  # low-pass aliases, mixed-case
        ("lpAlpha", (0.2, 13.99)),
        ("lpBeta", (0.2, 31.99)),
        ("lpGamma", (0.2, 100)),
        ("lpHGA", (0.2, 300)),
    ],
)
def test_known_bands(band, expected):
    """All recognised band names return the documented (l_freq, h_freq)."""
    assert get_frequency_band(band) == expected


def test_case_insensitivity():
    """Names should be handled in a case-insensitive manner."""
    assert get_frequency_band("ALPHA") == get_frequency_band("alpha")


def test_unknown_band():
    """Unknown keys fall back to `(None, None)`."""
    assert get_frequency_band("foobar") == (None, None)

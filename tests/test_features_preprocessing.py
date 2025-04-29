"""Unit tests for mpd.features.preprocessing.wavelet_denoise."""

import numpy as np
import pytest
import pywt

from mpd.features.preprocessing import wavelet_denoise


# Helpers
def _manual_denoise(x, wavelet: str, level: int):
    """Reference implementation: keep only approximation coeffs."""
    coeffs = pywt.wavedec(x, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # zero details
    return pywt.waverec(coeffs, wavelet)[: x.size]  # trim padding


# Tests
@pytest.mark.parametrize("n_trials, n_channels, n_times", [(2, 3, 64), (1, 1, 33)])
@pytest.mark.parametrize("wavelet, level", [("db4", 2), ("db2", 1)])
def test_shape_and_immutability(n_trials, n_channels, n_times, wavelet, level):
    """Output must preserve the original shape and inputs stay untouched."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_trials, n_channels, n_times))
    orig = data.copy()

    den = wavelet_denoise(data, wavelet, level)

    # shape preserved
    assert den.shape == data.shape
    # function must NOT modify the input array in-place
    assert np.array_equal(data, orig)


def test_algorithm_matches_manual_reference():
    """For a small tensor the helper must equal the per-sample manual version."""
    wavelet, lvl = "db4", 2
    data = np.arange(16, dtype=float).reshape(1, 1, -1)  # shape (1,1,16)

    expected = _manual_denoise(data[0, 0], wavelet, lvl)
    got = wavelet_denoise(data, wavelet, lvl)[0, 0]

    assert np.allclose(got, expected, atol=1e-12)

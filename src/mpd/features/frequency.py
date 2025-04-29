"""Lookup utilities related to canonical EEG / MEG frequency bands.

The mapping is kept in **one place** so that both the CLI (`--freq alpha`)
and any analysis notebooks rely on the exact same numerical ranges.
"""


def get_frequency_band(band_name):
    """Get the frequency range (low and high freqs) for a given band name.

    Parameters
    ----------
    band_name : str
        The name of the frequency band. Accepted values are 'gamma', 'beta',
        'alpha', 'theta', 'delta', and 'hga' (High-Gamma Activity).

    Returns
    -------
    tuple
        A tuple containing the low frequency (l_freq) and high frequency
        (h_freq) of the specified band. Returns (None, None) if the band name
        is not recognized.

    Examples
    --------
    ```python
    >>> get_frequency_band("alpha")
    (8, 13)

    >>> get_frequency_band("hga")
    (60, 300)

    >>> get_frequency_band("beta")
    (14, 31)
    ```
    """
    bands = {
        "delta": (0.2, 3.99),
        "theta": (4, 7.99),
        "alpha": (8, 13.99),
        "beta": (14, 31.99),
        "gamma": (32, 100),
        "hga": (60, 300),
        "lptheta": (0.2, 7.99),  # < theta
        "lpalpha": (0.2, 13.99),  # < alpha
        "lpbeta": (0.2, 31.99),  # < beta
        "lpgamma": (0.2, 100),  # < gamma
        "lphga": (0.2, 300),  # < HGA
    }
    return bands.get(band_name.lower(), (None, None))

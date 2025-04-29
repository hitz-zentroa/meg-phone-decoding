"""Signal processing helpers like wavelet denoiser functions.

Reusable **signal-processing helpers** that operate on NumPy tensors and do not
depend on MNE or PyTorch. The functions here may be called from any pipeline,
any model, or even from external notebooks.

Currently the module offers:

* :pyfunc:`wavelet_denoise` – simple multi-level Daubechies denoising that
  keeps only the approximation coefficients (low-frequency content).
"""

import numpy as np
import pywt


def wavelet_denoise(data, wavelet, level):
    """Denoise data using wavelet decomposition and reconstruction.

    Parameters
    ----------
    data : numpy.array
        The MEG signal data with shape (n_trials, n_channels, n_times).
    wavelet : str
        The type of wavelet to use (e.g., 'db4' for Daubechies-4).
    level : int
        The level of decomposition to perform.

    Returns
    -------
    numpy.array
        The denoised data.

    Examples
    --------
    ```python
    # Denoise the data assuming 'data' is the MEG data array with shape
    # (n_trials, n_channels, n_times)
    denoised_data = wavelet_denoise(data, 'db4', 2)
    ```
    """
    # Output array
    denoised_data = np.zeros_like(data)

    for trial in range(data.shape[0]):
        for channel in range(data.shape[1]):
            # Decompose to get the coefficients
            coeffs = pywt.wavedec(data[trial, channel, :], wavelet, level=level)
            # Zero out all the detail coefficients
            coeffs[1:] = [np.zeros_like(detail) for detail in coeffs[1:]]
            # Reconstruct the signal using only the approximation coefficients
            reconstructed_signal = pywt.waverec(coeffs, wavelet)
            # Handle cases where the reconstructed signal might slightly exceed
            # the original dimensions
            if reconstructed_signal.size > data.shape[2]:
                reconstructed_signal = reconstructed_signal[: data.shape[2]]
            denoised_data[trial, channel, :] = reconstructed_signal

    return denoised_data

"""Tiny helper used during **audio-to-MEG alignment**.

The BCBL corpus ships a MATLAB structure where `tds` is an array whose
*index* is the MEG sample number (at 1 kHz after MaxFilter) and whose *value*
is the corresponding **44.1 kHz WAV sample**.  Given a timestamp expressed in
WAV samples, we need to locate the closest MEG sample so that the
:meth:`mne.Annotations` we create line up perfectly with the neural signal.

The lookup is done with a simple linear scan that starts from a user-supplied
`start` index—this makes successive searches **O(Δ)** instead of **O(N)** when
annotations are already roughly sorted in time.
"""


def find_nearest_meg_sample(wav_sample, array, start=0):
    """Locate the MEG sample that best aligns with a given WAV sample position.

    Parameters
    ----------
    wav_sample : int
        Target sample number on the 44 100 Hz audio timeline.
    array : Sequence[int]
        Mapping vector (`tds`) where **index = MEG sample**, **value = WAV
        sample**.
    start : int, default `0`
        Index from which to begin the search.  Supplying the previous match
        makes the scan almost instantaneous for monotonically increasing
        `wav_sample`s.

    Returns
    -------
    int
        Index of the MEG sample whose mapped WAV sample is *closest* to
        `wav_sample`.

    Notes
    -----
    The function performs a forward scan until the absolute difference starts
    increasing, then returns the previous index.  If the end of the array is
    reached, the last index is returned.
    """
    prev_diff = float("inf")

    for meg_sample in range(start, len(array)):
        wav_sample_in_array = array[meg_sample]
        diff = abs(wav_sample_in_array - wav_sample)

        if diff > prev_diff:
            # The difference started increasing, so the previous index was the
            # closest
            return meg_sample - 1

        prev_diff = diff

    # If we reached the end without an increasing difference, return the last
    # index
    return len(array) - 1

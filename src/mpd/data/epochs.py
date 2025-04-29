"""Generate `mne.Epochs` from *Raw* recording plus *Annotations*.

The helper applies **optional sensor-picking, filtering and decimation** before
epoching.  The exact parameters are driven by the CLI flags and remain
centralised here so that every pipeline calls the same logic.
"""

import logging

import mne


def get_epochs(
    raw,
    annotations,
    label,
    meg=None,
    l_freq=None,
    h_freq=None,
    decim=10,
    selected_phones=None,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Convert `raw` + `annotations` into an :class:`mne.Epochs` instance.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous MEG recording.
    annotations : dict[str, mne.Annotations]
        Dictionary created by :pyfunc:`mpd.data.read_raw_with_annotations`.
    label : str
        Key that selects which annotation tier to use (typically `"phones"`).
    meg : {"mag", "grad", "planar1", "planar2", True, False, None}
        Sensor selection forwarded to :pyfunc:`mne.pick_types`.  If `None`
        the full 306-channel set is used.  Passing `True` selects all MEG
        channels; the string options restrict the sensor type.
    l_freq, h_freq : float or None
        Cut-off frequencies (Hz) for an FIR/IIR band-pass applied **before**
        decimation.  If both are `None` the signal is only low-passed to
        avoid aliasing when `decim > 1`.
    decim : int
        Decimation factor applied inside :class:`mne.Epochs`.  A matching
        low-pass filter is inserted automatically when `decim > 1`.
    selected_phones : list[str] or None
        If provided, events whose label is *not* in this list are discarded.
        Useful for multi-class training with a restricted phone set.

    Returns
    -------
    phone_epochs : mne.Epochs
        Epoched data spanning –100 ms to +200 ms around every phone onset,
        baseline-corrected and optionally decimated.
    ch_names : list[str]
        Channel names in the same order as they appear in `phone_epochs`.

    Notes
    -----
    * `event_repeated="drop"` silently discards duplicate annotations.
    * When *manual* band-pass is not requested, the function designs a low-pass
      at `raw.info['sfreq'] / (2*decim)` to respect the Nyquist limit after
      decimation.
    """
    # Select only a type of sensors if required
    if meg is not None:
        if meg.lower() in ["yes", "true"]:
            meg = True
        channel_picks = mne.pick_types(raw.info, meg=meg)
        logging.warning("Picked up channels: %s", meg)
    else:
        channel_picks = None

    if l_freq is not None or h_freq is not None:
        logging.warning("Frequency filter: %s - %s Hz", l_freq, h_freq)
        raw = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    elif decim > 1:
        s_freq = raw.info["sfreq"]
        h_freq = s_freq / decim / 2
        logging.warning(
            "Frequency filter for antialiasing: %s from %s Hz", h_freq, s_freq
        )
        raw = raw.copy().filter(l_freq=None, h_freq=h_freq)

    raw_phones = raw.copy().set_annotations(annotations[label])
    phone_events, phone_event_id = mne.events_from_annotations(raw_phones)

    if selected_phones is not None:
        # Filter out the phone events not in the specified list
        phone_event_id = {
            phone: id
            for phone, id in phone_event_id.items()
            if phone in selected_phones
        }

    phone_epochs = mne.Epochs(
        raw,
        phone_events,
        event_id=phone_event_id,
        tmin=-0.1,
        tmax=0.2,  # 5 Hz
        decim=decim,
        baseline=(-0.1, 0.0),
        preload=True,
        event_repeated="drop",
        picks=channel_picks,
    )

    # Retrieve channel names sorted as in the data
    if channel_picks is not None:
        ch_names = [raw.info["ch_names"][i] for i in channel_picks]
    else:
        ch_names = raw.info["ch_names"]

    return phone_epochs, ch_names

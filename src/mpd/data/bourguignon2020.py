"""Low-level dataset I/O utilities for Bourguignon 2020 dataset.

* discovery of subject folders whose names follow the pattern `selfpNN`;
* loading a single `.fif` file with Maxwell-filtered MEG data;
* reading the accompanying **TextGrid** annotations and the author-provided
  realignment matrix (`*_realign.mat`);
* converting those annotations into an :class:`mne.Annotations` object that is
  sample-accurate with respect to the MEG timeline.

If you want to plug in another dataset, used this module as a base while
keeping the public API identical.
"""

import os
import re
from pathlib import Path

import mne
import scipy.io
import textgrid
from scipy.io.wavfile import read as read_wav
from tqdm.auto import tqdm

from .annotations import find_nearest_meg_sample


def get_subjects(data_path: Path) -> list:
    """Return the list of subject directories found under `<data_path>/meg/`.

    Parameters
    ----------
    data_path : Path or str
        Root folder of the corpus (the directory that contains `meg/`).

    Returns
    -------
    list[str]
        Directory names such as `"selfp01"`, `"selfp02"`, ..., **excluding**
        `"selfp17"` whose MEG recording is missing.

    Notes
    -----
    The function matches folder names with the regex `"selfp[0-9]+"` and sorts
    them lexicographically; no further validation is done.
    """
    meg_path = os.path.join(data_path, "meg")
    subjects = []
    for entry in os.listdir(meg_path):
        full_path = os.path.join(meg_path, entry)
        matches = re.match(r"selfp(\d+)", entry)
        if os.path.isdir(full_path) and matches is not None:
            subject = matches.group(0)
            subjects.append(subject)
    subjects.remove("selfp17")  # MEG recordings missing
    return subjects


def read_raw_with_annotations(
    data_path, subject, task
):  # pylint: disable=too-many-locals,too-many-statements
    """Load a *single* MEG recording together with phone-level annotations.

    Parameters
    ----------
    data_path : Path or str
        Dataset root folder.
    subject : str
        Subject identifier, e.g. `"selfp05"`.
    task : {"listen", "playback", "self"}
        Experimental condition.

    Returns
    -------
    raw : mne.io.Raw
        MEG recording (`preload=True`) from `*_tsss.fif`.
    annotations : dict[str, mne.Annotations]
        One entry per TextGrid tier.  Each :class:`mne.Annotations` contains
        *onset*, *duration* and *description* arrays that are aligned to the
        **MEG sample rate** (1 kHz) using the realignment matrix stored in
        `*_realign.mat`.

    Notes
    -----
    The alignment procedure follows the original authors' MATLAB code:

    1. audio timestamps (44.1 kHz) -> sample index in the alignment vector
       (`tds`);
    2. that index plus `dec` gives the corresponding MEG sample;
    3. finally converted to seconds via `raw.info['sfreq']`.

    Intervals whose label is `"spn"` are skipped.  Over-long durations that
    would overlap the next event are trimmed automatically.
    """
    raw_path = os.path.join(data_path, "meg", subject, f"{subject}_{task}_tsss.fif")
    raw = mne.io.read_raw_fif(raw_path, preload=True)

    wav_task = "self" if task == "playback" else task

    wav_path = os.path.join(data_path, "meg", subject, f"{subject}_{wav_task}_norm.wav")

    realign_path = os.path.join(
        data_path, "meg", subject, f"{subject}_{task}_realign.mat"
    )

    realign = scipy.io.loadmat(realign_path)

    textgrid_path = os.path.join(
        data_path, "meg", subject, f"{subject}_{wav_task}_norm.TextGrid"
    )

    tg = textgrid.TextGrid.fromFile(textgrid_path)
    for tier in tg[:5]:
        print("Tier name:", tier.name)
        for interval in tier[:10]:
            if interval.mark is None or len(interval.mark) == 0:
                continue
            print("  Interval:", interval.minTime, interval.maxTime, interval.mark)
        print("  ...")

    wav_sr, _ = read_wav(wav_path)  # 44100
    meg_sr = raw.info["sfreq"]
    annotations = {}

    tg = textgrid.TextGrid.fromFile(textgrid_path)
    for tier in tqdm(tg, desc="Tier"):
        skipped = 0
        total = 0
        prev_onset = 0
        onsets, durations, descriptions = [], [], []
        for interval in tqdm(tier, desc="Interval"):
            total += 1
            if interval.mark is None or len(interval.mark) == 0:
                continue
            if interval.mark == "spn":
                skipped += 1
                print(f"Skipping unknown label... ({tier.name}:{skipped})")
                continue

            # Extract event from the audio metadata:
            onset = interval.minTime
            offset = interval.maxTime
            description = interval.mark

            # Calculate the sample from event time:
            onset_wav = onset * wav_sr
            offset_wav = offset * wav_sr

            # Realign the event from the wav sample to the MEG sample:
            onset_meg = find_nearest_meg_sample(
                onset_wav, realign["tds"][0], prev_onset
            )
            # Optimization to avoid traversing from 0 for next samples:
            prev_onset = onset_meg
            offset_meg = find_nearest_meg_sample(
                offset_wav, realign["tds"][0], prev_onset
            )
            prev_onset = offset_meg

            # Calculation event time in seconds from sample number:
            dec = realign["dec"][0][0]
            onset = (onset_meg + dec) / meg_sr
            offset = (offset_meg + dec) / meg_sr
            duration = offset - onset
            # print("MEG: event:", description, "onset:", onset, "offset:", offset)

            # Save the event information in seconds:
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)

        # Fix durations if too long:
        assert len(onsets) == len(durations)
        for i in range(len(onsets) - 1):
            if onsets[i] + durations[i] > onsets[i + 1]:
                print("Fixing duration...")
                durations[i] = onsets[i + 1] - onsets[i]

        print(f"Total {tier.name}:{total}")
        annotations[tier.name] = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
        )
    return raw, annotations

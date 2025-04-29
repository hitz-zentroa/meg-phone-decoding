"""Dataset reading logic."""

from importlib import import_module

_DEFAULT = "bourguignon2020"

# Map short names -> Python module (dot-path)
_DATASETS = {
    "bourguignon2020": "mpd.data.bourguignon2020",
    # "mymeg": "my_project.mpd_adapter",        # user-added
}


def _get_backend(name=None):
    """Return the dataset-specific module, defaulting to bourguignon2020.

    Parameters
    ----------
    name : str
        The name of the dataset.
    """
    if name is None:
        name = _DEFAULT
    mod_name = _DATASETS.get(name, name)
    return import_module(mod_name)


def get_subjects(data_path, *, dataset=_DEFAULT):
    """Return the list of subject directories.

    Parameters
    ----------
    data_path : Path or str
        Root folder of the corpus.
    dataset : str (optional)
        The name of the dataset use to import the reading functions.

    Returns
    -------
    list[str]
        Directory names such as `"subj01"`, `"subj02"`, ...
    """
    return _get_backend(dataset).get_subjects(data_path)


def read_raw_with_annotations(data_path, subject, task, *, dataset=_DEFAULT):
    """Load a single MEG recording together with phone-level annotations.

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
        MEG recording (with `preload=True`).
    annotations : dict[str, mne.Annotations]
        One entry per TextGrid tier.  Each :class:`mne.Annotations` contains
        *onset*, *duration* and *description* arrays that are aligned to the
        **MEG sample rate** (1 kHz).
    dataset : str (optional)
        The name of the dataset use to import the reading functions.
    """
    mod = _get_backend(dataset)
    return mod.read_raw_with_annotations(data_path, subject, task)

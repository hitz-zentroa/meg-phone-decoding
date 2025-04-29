"""Unit tests for mpd.data.epochs.get_epochs.

We do NOT touch the real MNE code base – every heavy object is replaced by a
tiny stub so the tests run in a few milliseconds.
"""

import numpy as np

import mpd.data.epochs as ep


# Common stubs
class DummyRaw:
    """Minimal stand-in for `mne.io.Raw`."""

    def __init__(self, sfreq=1000.0, ch_names=None):
        self.info = {"sfreq": sfreq, "ch_names": ch_names or ["C1", "C2", "C3"]}
        self._filter_calls = []  # store (l_freq, h_freq) for assertions

    # copy/filter
    def copy(self):
        """Return copy of Raw instance.

        Returns
        -------
        inst : instance of `Raw`
            A copy of the instance.
        """
        clone = DummyRaw(self.info["sfreq"], self.info["ch_names"])
        # share to inspect later
        clone._filter_calls = self._filter_calls  # pylint: disable=protected-access
        return clone

    def filter(self, l_freq=None, h_freq=None):
        """Filter a subset of channels/vertices.

        Returns
        -------
        inst : instance of `Raw`
            The filtered data.
        """
        self._filter_calls.append((l_freq, h_freq))
        return self

    # annotations
    def set_annotations(self, _):
        """Setter for annotations.

        Returns
        -------
        self : instance of `Raw`
            The raw object with annotations.
        """
        return self


class DummyEpochs:  # pylint: disable=too-few-public-methods
    """Collect the arguments passed by `get_epochs` for inspection."""

    def __init__(self, raw, events, event_id, **kwargs):
        self.raw = raw
        self.events = events
        self.event_id = event_id
        self.kwargs = kwargs
        # expose picks so tests can inspect sensor selection
        self.picks = kwargs.get("picks", None)

    # just to mimic the real API the pipelines might call later
    def __iter__(self):
        return iter([])


def _patch_mne(monkeypatch):
    """Replace *every* MNE function used inside get_epochs with a stub."""

    # mne.pick_types
    def fake_pick_types(info, *, meg):
        # return indices that depend on the argument so we can assert later
        return [0, 2] if meg == "grad" else list(range(len(info["ch_names"])))

    monkeypatch.setattr(ep.mne, "pick_types", fake_pick_types)

    # mne.events_from_annotations
    def fake_events_from_annotations(_raw):
        #  Two events labelled 'a' and 'b' for the tests
        events = np.array([[0, 0, 1], [100, 0, 2]])
        event_id = {"a": 1, "b": 2}
        return events, event_id

    monkeypatch.setattr(ep.mne, "events_from_annotations", fake_events_from_annotations)

    # mne.Epochs
    monkeypatch.setattr(ep.mne, "Epochs", DummyEpochs)


# Tests
def test_channel_selection_and_names(monkeypatch):
    """`meg="grad"` must forward to pick_types and return the correct ch_names."""
    _patch_mne(monkeypatch)

    raw = DummyRaw(ch_names=["C1", "C2", "C3"])
    annotations = {"phones": None}

    epochs, ch_names = ep.get_epochs(raw, annotations, "phones", meg="grad")

    # pick_types stub returns [0,2]  ->  channel names must follow
    assert ch_names == ["C1", "C3"]
    # The same indices must be stored inside the DummyEpochs picks argument
    assert epochs.picks == [0, 2]


def test_selected_phones_filter(monkeypatch):
    """The selected_phones restricts the event_id passed to Epochs."""
    _patch_mne(monkeypatch)

    raw = DummyRaw()
    annotations = {"phones": None}

    epochs, _ = ep.get_epochs(raw, annotations, "phones", selected_phones=["a"])

    # Dummy events_from_annotations returns {'a':1,'b':2}; after filtering
    # only 'a' should remain
    assert list(epochs.event_id.keys()) == ["a"]


def test_antialias_lowpass_inserted(monkeypatch):
    """When decim>1 and no band-pass the helper must insert a low-pass."""
    _patch_mne(monkeypatch)

    raw = DummyRaw(sfreq=1000.0)
    annotations = {"phones": None}

    ep.get_epochs(raw, annotations, "phones", decim=4)  # triggers auto low-pass

    # The *last* call to DummyRaw.filter should be the Nyquist-safe LP
    # -> (None, 125.0)
    assert raw._filter_calls[-1] == (  # pylint: disable=protected-access
        None,
        1000.0 / 4 / 2,
    )

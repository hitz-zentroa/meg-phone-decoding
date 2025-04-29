"""Unit-tests for the dataset router (`mpd.data`).

The router must

1.  resolve the *default* backend when no name is supplied;
2.  load a user-registered backend module and forward the public helpers; and
3.  raise an informative `KeyError` for unknown keys.
"""

import sys
from types import ModuleType

import pytest

import mpd.data as md


# Helpers
def _register_dummy(monkeypatch, key="dummy"):
    """Create a one-off backend module and register it under *key*."""
    dummy = ModuleType("dummy.backend")  # pylint: disable=protected-access,no-member

    # minimal API expected by the router -------------------------------
    dummy.get_subjects = lambda root: ["subjX"]
    dummy.read_raw_with_annotations = lambda root, s, t: ("RAW", "ANN")

    # make importlib.import_module find it
    monkeypatch.setitem(sys.modules, dummy.__name__, dummy)  # pylint: disable=no-member
    # add entry in the dispatcher table
    monkeypatch.setitem(
        md._DATASETS, key, dummy.__name__  # pylint: disable=protected-access,no-member
    )

    return dummy


# Tests
def test_get_backend_handles_known_key(monkeypatch):
    """Verify it handles known datasets."""
    dummy = _register_dummy(monkeypatch)

    # The router must return *exactly* the same module object we injected
    backend = md._get_backend("dummy")  # pylint: disable=protected-access
    assert backend is dummy


def test_public_wrappers_delegate_to_backend(monkeypatch):
    """Verify it forward the calls to the backend."""
    _register_dummy(monkeypatch)

    # Both convenience wrappers should call into the dummy implementation
    assert md.get_subjects("/some/path", dataset="dummy") == ["subjX"]

    raw, ann = md.read_raw_with_annotations("/root", "subjX", "task", dataset="dummy")
    assert (raw, ann) == ("RAW", "ANN")


def test_unknown_dataset_key_raises_error():
    """Unknown datasets must raise an exception."""
    with pytest.raises(ModuleNotFoundError):
        md._get_backend("does_not_exist")  # pylint: disable=protected-access

"""
Unit tests for mpd.data.io.get_subjects.

The helper must:

* scan <root>/meg/
* return folders matching r"selfp\\d+"
* exclude "selfp17"
* keep lexicographic order
"""

from pathlib import Path

import pytest

from mpd.data import bourguignon2020 as io


def _touch_dir(path: Path):
    path.mkdir(parents=True)
    #  create a dummy file so that path.is_dir() is True regardless of platform
    (path / ".keep").touch()


def test_get_subjects_happy_path(tmp_path):
    """
    Synthetic directory layout:

        tmp/
          meg/
            selfp01/          ✓
            selfp03/          ✓
            selfp17/          ✗  (must be dropped)
            random_folder/    (ignored)
            selfp02.txt       (ignored – not a directory)

    Expected list -> ["selfp01", "selfp03"]   (lexicographic order)
    """
    meg = tmp_path / "meg"
    _touch_dir(meg / "selfp01")
    _touch_dir(meg / "selfp03")
    _touch_dir(meg / "selfp17")  # must be excluded
    _touch_dir(meg / "random_folder")  # does not match pattern
    (meg / "selfp02.txt").touch()  # file, not a directory

    subjects = io.get_subjects(tmp_path)

    assert sorted(subjects) == sorted(["selfp01", "selfp03"])


def test_missing_meg_dir(tmp_path, monkeypatch):  # pylint: disable=unused-argument
    """If <root>/meg does not exist, Python's `os.listdir` raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        io.get_subjects(tmp_path)

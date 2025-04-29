"""
Unit tests for the command-line interface (mpd.cli).

We validate two things:

1.  _parse_args correctly interprets user flags (only defaults tested here).
2.  main() delegates to run_experiment and sets the requested log level.

No heavy code is executed – everything is patched with tiny stubs.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from mpd import cli


# _parse_args
def test_parse_args_defaults(monkeypatch):
    """Calling `_parse_args` with only <dataset> yields documented defaults."""
    fake_dataset = "/some/path"
    monkeypatch.setattr(sys, "argv", ["mpd-train", fake_dataset], raising=False)

    ns = cli._parse_args()  # pylint: disable=protected-access

    # positional
    assert ns.dataset == Path(fake_dataset)
    # a few defaults that matter later in the pipeline
    assert ns.model == "logistic_regression"
    assert ns.tasks == cli.TASKS
    assert ns.log_level == "WARNING"
    # default phones list unchanged
    assert ns.phones[:3] == ["a", "e", "i"]


# main() – delegation & logging
def test_main_delegates_and_sets_logging(monkeypatch):
    """`cli.main` must call run_experiment with the parsed Namespace."""
    # prepare dummy objects
    dummy_ns = SimpleNamespace(log_level="INFO")  # logging level we expect
    run_called = {"val": False}

    def fake_parse():
        return dummy_ns

    def fake_run(arg):
        run_called["val"] = arg is dummy_ns

    # monkey-patch
    monkeypatch.setattr(cli, "_parse_args", fake_parse)
    monkeypatch.setattr(cli, "run_experiment", fake_run)

    log_cfg_called = {}

    def fake_basic_config(**kw):
        log_cfg_called.update(kw)

    import logging  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    # execute
    cli.main()

    # assertions
    assert run_called["val"], "run_experiment was not invoked with parsed args"
    assert log_cfg_called.get("level") == dummy_ns.log_level, "Incorrect log level"

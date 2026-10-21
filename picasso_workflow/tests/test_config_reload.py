#!/usr/bin/env python
"""Tests for picasso_workflow.reload_config (live config reload)."""

import copy
import tempfile
from pathlib import Path

import pytest

import picasso_workflow
from picasso_workflow import CONFIG, _load_layer, reload_config


@pytest.fixture()
def restore_config():
    """Snapshot and restore the process-global CONFIG around a test."""
    snapshot = copy.deepcopy(CONFIG)
    yield
    CONFIG.clear()
    CONFIG.update(snapshot)


def test_reload_config_mutates_in_place(monkeypatch, restore_config):
    before = CONFIG  # hold the reference the GUI imported
    monkeypatch.setattr(
        picasso_workflow,
        "load_config",
        lambda: {"Templates": {"X": "/x"}, "New": 1},
    )
    out = reload_config()
    # same object -> modules that imported CONFIG by reference see the change
    assert out is CONFIG
    assert CONFIG is before
    assert CONFIG["New"] == 1
    assert list(CONFIG["Templates"]) == ["X"]


def test_reload_config_drops_removed_keys(monkeypatch, restore_config):
    CONFIG["StaleKey"] = "gone-after-reload"
    monkeypatch.setattr(
        picasso_workflow, "load_config", lambda: {"Templates": {}}
    )
    reload_config()
    assert "StaleKey" not in CONFIG


def test_load_layer_degrades_instead_of_crashing():
    # A hand-edited config.yaml with a typo must not crash import/load_config;
    # _load_layer returns None (the layer is skipped) for bad/odd files.
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        malformed = base / "bad.yaml"
        malformed.write_text("a: [unclosed\n")
        assert _load_layer(malformed) is None  # YAMLError -> skipped

        not_a_mapping = base / "list.yaml"
        not_a_mapping.write_text("- a\n- b\n")
        assert _load_layer(not_a_mapping) is None  # non-dict -> skipped

        assert _load_layer(base / "missing.yaml") is None  # absent -> skipped

        good = base / "ok.yaml"
        good.write_text("Templates:\n  X: /p\n")
        assert _load_layer(good) == {"Templates": {"X": "/p"}}

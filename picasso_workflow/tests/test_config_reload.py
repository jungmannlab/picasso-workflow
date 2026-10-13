#!/usr/bin/env python
"""Tests for picasso_workflow.reload_config (live config reload)."""

import copy

import pytest

import picasso_workflow
from picasso_workflow import CONFIG, reload_config


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

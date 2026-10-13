#!/usr/bin/env python
"""Tests for the first-run "Getting started" onboarding dialog."""

import os
import pathlib
import tempfile
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

import yaml  # noqa: E402
from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow.gui import GettingStartedDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def fake_home():
    with tempfile.TemporaryDirectory() as d:
        home = pathlib.Path(d)
        with mock.patch.object(
            pathlib.Path, "home", classmethod(lambda cls: home)
        ):
            yield home


def test_paths_live_under_the_user_config_dir(qapp, fake_home):
    cfg = GettingStartedDialog.user_config_path()
    marker = GettingStartedDialog.marker_path()
    assert cfg == fake_home / ".config" / "picasso_workflow" / "config.yaml"
    assert marker.parent == cfg.parent


def test_open_user_config_seeds_from_template(qapp, fake_home):
    cfg = GettingStartedDialog.user_config_path()
    assert not cfg.exists()
    GettingStartedDialog.open_user_config(None)
    assert cfg.exists()
    # the seeded file is the bundled template, which carries Templates
    data = yaml.safe_load(cfg.read_text())
    assert "Templates" in data


def test_open_user_config_does_not_clobber_existing(qapp, fake_home):
    cfg = GettingStartedDialog.user_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("Templates:\n  Mine: /my/path\n")
    GettingStartedDialog.open_user_config(None)
    # unchanged -- existing user config must not be overwritten
    assert yaml.safe_load(cfg.read_text()) == {
        "Templates": {"Mine": "/my/path"}
    }


def test_accept_writes_marker_when_checkbox_checked(qapp, fake_home):
    d = GettingStartedDialog()
    assert d.dont_show.isChecked()  # default: show once
    assert not GettingStartedDialog.marker_path().exists()
    d.accept()
    assert GettingStartedDialog.marker_path().exists()


def test_accept_keeps_showing_when_unchecked(qapp, fake_home):
    d = GettingStartedDialog()
    d.dont_show.setChecked(False)
    d.accept()
    assert not GettingStartedDialog.marker_path().exists()

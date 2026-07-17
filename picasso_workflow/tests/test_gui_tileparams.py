#!/usr/bin/env python
"""GUI tests for per-channel / per-cell tile parameters.

Exercises the per-channel parameter capability of the aggregation/
investigation workflow builder: declaring parameter columns that differ
between channels, resolving their values per (dataset, channel) tile, and
flattening them into the ``single_dataset_tileparameters`` dict alongside
``#tags`` / ``filepath``. Also checks that the command dialog offers the new
columns as ``$$map`` keys and emits the optional default. Skips where a Qt
GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    yield win
    win.close()


def _seed_tree(win):
    """Two datasets x two channels with concrete file paths."""
    win.tree_data["datasets"] = ["cellA", "cellB"]
    win.tree_data["channels"] = ["ch0", "ch1"]
    win.tree_data["file_paths"] = {
        "cellA": {"ch0": "/d/a0.tif", "ch1": "/d/a1.tif"},
        "cellB": {"ch0": "/d/b0.tif", "ch1": "/d/b1.tif"},
    }
    win.tree_data["conditions"] = {}
    win.tree_data["tile_params"] = {}


def test_register_rejects_builtin_names(window):
    _seed_tree(window)
    for name in ("filepath", "#tags", ""):
        with pytest.raises(ValueError):
            window.register_tile_param(name, "channel", 1)


def test_channel_param_flatten(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    window.set_tile_param_value("min_locs", 8, "ch1")

    datasets = window._flatten_tree_to_datasets(None, with_conditions=False)

    assert datasets["#tags"] == [
        "cellA_ch0",
        "cellA_ch1",
        "cellB_ch0",
        "cellB_ch1",
    ]
    # channel value broadcast across both datasets, aligned to #tags
    assert datasets["min_locs"] == [5, 8, 5, 8]


def test_channel_param_default_fallback(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    # ch1 left unset -> falls back to the column default

    datasets = window._flatten_tree_to_datasets(None, with_conditions=False)
    assert datasets["min_locs"] == [5, 10, 5, 10]


def test_cell_param_precedence(window):
    _seed_tree(window)
    window.register_tile_param("radius", "cell", 4)
    window.set_tile_param_value("radius", 2, "ch0", dataset="cellA")

    datasets = window._flatten_tree_to_datasets(None, with_conditions=False)
    # only (cellA, ch0) overridden; the rest take the default
    assert datasets["radius"] == [2, 4, 4, 4]


def test_unset_falls_back_to_default(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    window.unset_tile_param_value("min_locs", "ch0")

    datasets = window._flatten_tree_to_datasets(None, with_conditions=False)
    assert datasets["min_locs"] == [10, 10, 10, 10]


def test_sync_prunes_removed_channel(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    window.set_tile_param_value("min_locs", 8, "ch1")

    # Drop a channel and sync (mirrors remove_channel + repopulate)
    window.tree_data["channels"] = ["ch0"]
    window._sync_tile_params()

    values = window.tree_data["tile_params"]["min_locs"]["values"]
    assert "ch1" not in values
    assert values["ch0"] == 5


def test_cmd_dialog_offers_columns_and_default(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)

    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window
    )
    try:
        keys = [
            dialog.map_combo.itemText(i)
            for i in range(dialog.map_combo.count())
        ]
        assert "filepath" in keys and "#tags" in keys
        assert "min_locs" in keys

        # Select the new column, start-of-stage timing, and a default.
        dialog.command_combo.setCurrentText("map")
        dialog.timing_start_radio.setChecked(True)
        dialog.map_combo.setCurrentText("min_locs")
        dialog.map_default.setText("10")
        assert dialog.get_command() == "('$$map', 'min_locs', 10)"

        # No default -> plain 2-tuple.
        dialog.map_default.setText("")
        assert dialog.get_command() == "('$$map', 'min_locs')"
    finally:
        dialog.close()

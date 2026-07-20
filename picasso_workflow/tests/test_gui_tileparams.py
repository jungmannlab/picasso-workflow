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


def test_cmd_dialog_shows_param_name(window):
    _seed_tree(window)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window, param_name="min_locs"
    )
    try:
        assert "min_locs" in dialog.windowTitle()
        dialog.command_combo.setCurrentText("map")
        assert "min_locs" in dialog.help_label.text()
    finally:
        dialog.close()


def test_cmd_dialog_offers_param_as_column_option(window):
    _seed_tree(window)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window, param_name="min_locs"
    )
    try:
        dialog.command_combo.setCurrentText("map")
        keys = [
            dialog.map_combo.itemText(i)
            for i in range(dialog.map_combo.count())
        ]
        assert "filepath" in keys and "#tags" in keys
        # the current parameter is offered as a ready-made column option
        assert "min_locs" in keys
        assert "min_locs" not in window.tree_data["tile_params"]

        # selecting it creates the per-channel column and reveals the table
        dialog.map_combo.setCurrentText("min_locs")
        assert "min_locs" in window.tree_data["tile_params"]
        assert dialog.map_values_table.rowCount() == 2
        dialog.timing_start_radio.setChecked(True)
        assert dialog.get_command() == "('$$map', 'min_locs')"
    finally:
        dialog.close()


def test_cmd_dialog_prunes_empty_autocreated_column_on_cancel(window):
    _seed_tree(window)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window, param_name="box_size"
    )
    try:
        dialog.command_combo.setCurrentText("map")
        dialog.map_combo.setCurrentText("box_size")
        assert "box_size" in window.tree_data["tile_params"]
    finally:
        dialog.reject()  # cancel with no values entered
    assert "box_size" not in window.tree_data["tile_params"]


def test_cmd_dialog_keeps_autocreated_column_with_values_on_cancel(window):
    _seed_tree(window)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window, param_name="box_size"
    )
    try:
        dialog.command_combo.setCurrentText("map")
        dialog.map_combo.setCurrentText("box_size")
        dialog.map_values_table.item(0, 1).setText("7")
        assert window.resolve_tile_param("box_size", None, "ch0") == 7
    finally:
        dialog.reject()
    # values were entered, so the column is preserved
    assert "box_size" in window.tree_data["tile_params"]


def test_cmd_dialog_inline_channel_values(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window, param_name="min_locs"
    )
    try:
        dialog.command_combo.setCurrentText("map")
        dialog.map_combo.setCurrentText("min_locs")

        # One editable value row per channel; blank cells show the default.
        assert dialog.map_values_table.rowCount() == 2
        assert dialog.map_values_table.item(0, 1).text() == "10"

        # Editing a cell writes through to the window's tile-param store.
        dialog.map_values_table.item(0, 1).setText("5")
        assert window.resolve_tile_param("min_locs", None, "ch0") == 5

        # Clearing a cell falls back to the default again.
        dialog.map_values_table.item(0, 1).setText("")
        assert window.resolve_tile_param("min_locs", None, "ch0") == 10

        # A built-in column has no per-channel value table.
        dialog.map_combo.setCurrentText("filepath")
        assert dialog.map_values_table.rowCount() == 0
    finally:
        dialog.close()


def test_cmd_dialog_prepopulates_existing_map(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")

    dialog = gui.ParameterCmdDialog(
        [],
        window.module_descriptor,
        0,
        parent=window,
        param_name="min_locs",
        current_value=("$$map", "min_locs", 10),
    )
    try:
        # Opens on the existing command, not the default filepath/map.
        assert dialog.command_combo.currentText() == "map"
        assert dialog.map_combo.currentText() == "min_locs"
        assert dialog.timing_start_radio.isChecked()
        assert dialog.map_default.text() == "10"
        # Stored per-channel values are shown (ch0 = 5, ch1 = default 10).
        assert dialog.map_values_table.rowCount() == 2
        assert dialog.map_values_table.item(0, 1).text() == "5"
        assert dialog.map_values_table.item(1, 1).text() == "10"
        assert dialog.get_command() == "('$$map', 'min_locs', 10)"
    finally:
        dialog.close()


def test_restore_dataset_table_roundtrip(window):
    import ast
    import pprint

    tree_data = {
        "datasets": ["cellA", "cellB"],
        "channels": ["GFP", "CD80"],
        "file_paths": {
            "cellA": {"GFP": "/d/a_gfp.tif", "CD80": "/d/a_cd80.tif"},
            "cellB": {"GFP": "/d/b_gfp.tif", "CD80": "/d/b_cd80.tif"},
        },
        "conditions": {},
        "tile_params": {
            "min_locs": {
                "scope": "channel",
                "default": 10,
                "values": {"GFP": 5, "CD80": 8},
            },
        },
    }
    gui_table = {"workflow_type": 1, "tree_data": tree_data}

    # The persisted block must be a valid Python literal that round-trips.
    dumped = pprint.pformat(gui_table, width=79, sort_dicts=False)
    assert ast.literal_eval(dumped) == gui_table

    window._restore_dataset_table(gui_table)
    assert window.tree_data["datasets"] == ["cellA", "cellB"]
    assert window.tree_data["channels"] == ["GFP", "CD80"]
    assert window.workflow_type.currentIndex() == 1

    # Per-channel values survive and resolve per (dataset, channel).
    assert window.resolve_tile_param("min_locs", "cellA", "GFP") == 5
    assert window.resolve_tile_param("min_locs", "cellB", "CD80") == 8

    # Flatten produces the aligned per-tile column again.
    datasets = window._flatten_tree_to_datasets(None, with_conditions=False)
    assert datasets["min_locs"] == [5, 8, 5, 8]


def test_restore_dataset_table_ignores_malformed(window):
    _seed_tree(window)
    before = window.tree_data["channels"]
    # Missing 'tree_data' -> no change, no crash.
    window._restore_dataset_table({"workflow_type": 1})
    assert window.tree_data["channels"] == before


def test_cmd_dialog_contextual_help(window):
    _seed_tree(window)
    dialog = gui.ParameterCmdDialog(
        [], window.module_descriptor, 0, parent=window
    )
    try:
        # map defaults to before-timing -> should warn to use $$
        dialog.command_combo.setCurrentText("map")
        dialog.timing_before_radio.setChecked(True)
        html = dialog.help_label.text()
        assert "Map" in html
        assert "start of stage" in html
        assert "&#9888;" in html  # warning glyph present on mismatch

        # switching to $$ clears the warning
        dialog.timing_start_radio.setChecked(True)
        assert "&#9888;" not in dialog.help_label.text()

        # a different command type shows different guidance
        dialog.command_combo.setCurrentText("Previous Module Result")
        assert "immediately before" in dialog.help_label.text()
    finally:
        dialog.close()


# --- per-channel summary shown below the parameter row -----------------

_PARAM_META = {
    "min_locs": {
        "type": "int",
        "description": "minimum number of localizations",
        "default": 10,
    },
    "radius": {"type": "float", "description": "radius", "default": 4.0},
}


def _seed_params(win):
    """Render the parameter rows of a small two-parameter module."""
    win._clear_parameter_layout()
    win.parameter_widgets = {}
    win._populate_parameter_widgets(_PARAM_META)
    return win.parameter_widgets


def _summary(win, param_name):
    return win.parameter_widgets[param_name].summary_label


def test_param_summary_hidden_without_command(window):
    _seed_tree(window)
    _seed_params(window)

    # isVisible() is always False under offscreen Qt with no shown
    # parent, so assert on the explicit hidden flag instead.
    label = _summary(window, "min_locs")
    assert label.isHidden()
    assert label.text() == ""


def test_param_summary_shows_channel_values(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    window.set_tile_param_value("min_locs", 10, "ch1")
    _seed_params(window)

    window._convert_widget_to_textbox("min_locs", "('$$map', 'min_locs', 10)")

    label = _summary(window, "min_locs")
    assert not label.isHidden()
    text = label.text()
    assert "per-channel" in text
    assert "ch0 = 5" in text
    assert "ch1 = 10" in text
    assert "default 10" in text
    # An unmapped parameter in the same module stays clean.
    assert _summary(window, "radius").text() == ""


def test_param_summary_uses_default_for_unset_channel(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    # ch1 left unset
    _seed_params(window)
    window._convert_widget_to_textbox("min_locs", "('$$map', 'min_locs', 10)")

    text = _summary(window, "min_locs").text()
    assert "ch0 = 5" in text
    assert "ch1 = 10" in text


def test_param_summary_hidden_for_builtin_column(window):
    _seed_tree(window)
    _seed_params(window)
    window._convert_widget_to_textbox("min_locs", "('$$map', 'filepath')")

    # filepath/#tags are per-channel by definition and live in the tree;
    # annotating them would be noise.
    assert _summary(window, "min_locs").isHidden()
    assert _summary(window, "min_locs").text() == ""


def test_param_summary_hidden_for_non_map_command(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    _seed_params(window)
    window._convert_widget_to_textbox(
        "min_locs", "('$get_prior_result', 'results, 00_x, y')"
    )

    assert _summary(window, "min_locs").text() == ""


def test_param_summary_follows_channel_rename(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "channel", 10)
    window.set_tile_param_value("min_locs", 5, "ch0")
    _seed_params(window)
    window._convert_widget_to_textbox("min_locs", "('$$map', 'min_locs', 10)")
    assert "ch0 = 5" in _summary(window, "min_locs").text()

    # Rename ch0 -> GFP, carrying its value across (what rename_channel
    # does to tree_data), then refresh.
    window.tree_data["channels"] = ["GFP", "ch1"]
    values = window.tree_data["tile_params"]["min_locs"]["values"]
    values["GFP"] = values.pop("ch0")
    window._refresh_param_summaries()

    text = _summary(window, "min_locs").text()
    assert "GFP = 5" in text
    assert "ch0" not in text


def test_param_summary_cell_scope_text(window):
    _seed_tree(window)
    window.register_tile_param("min_locs", "cell", 10)
    window.set_tile_param_value("min_locs", 5, "ch0", dataset="cellA")
    _seed_params(window)
    window._convert_widget_to_textbox("min_locs", "('$$map', 'min_locs', 10)")

    label = _summary(window, "min_locs")
    text = label.text()
    assert "per-cell (min_locs)" in text
    assert "1 of 4 cells set" in text
    # The full per-cell breakdown lives in the tooltip.
    assert "cellA/ch0 = 5" in label.toolTip()
    assert "cellB/ch1 = 10" in label.toolTip()


def test_param_summary_elides_many_channels(window):
    _seed_tree(window)
    window.tree_data["channels"] = ["c%d" % i for i in range(6)]
    window.register_tile_param("min_locs", "channel", 10)
    for i in range(6):
        window.set_tile_param_value("min_locs", i, "c%d" % i)
    _seed_params(window)
    window._convert_widget_to_textbox("min_locs", "('$$map', 'min_locs', 10)")

    label = _summary(window, "min_locs")
    text = label.text()
    assert "(+2 more)" in text
    assert "c5 = 5" not in text
    # ... but the tooltip lists every channel.
    tooltip = label.toolTip()
    for i in range(6):
        assert "c%d = %d" % (i, i) in tooltip


# --- commands on nested (dict) sub-parameters ---------------------------

_NESTED_META = {
    "sample_movie": {
        "type": "dict",
        "description": "sample movie settings",
        "required": True,
        "properties": {
            "n_sample": {
                "type": "int",
                "description": "number of sample frames",
                "default": 40,
            },
        },
    },
}


def _seed_nested_params(win):
    """Render a module with one required dict parameter."""
    win._clear_parameter_layout()
    win.parameter_widgets = {}
    win._populate_parameter_widgets(_NESTED_META)
    return win.parameter_widgets["sample_movie"].sub_parameters["n_sample"]


def test_nested_params_are_not_registered_top_level(window):
    """Pins the premise: only top-level names are in parameter_widgets."""
    _seed_nested_params(window)
    assert "sample_movie" in window.parameter_widgets
    assert "n_sample" not in window.parameter_widgets


def test_nested_cmd_conversion_uses_widget_info(window):
    _seed_tree(window)
    sub = _seed_nested_params(window)

    # Previously raised KeyError, since it looked "n_sample" up in
    # parameter_widgets where only top-level names live.
    window._convert_widget_to_textbox(
        "n_sample", "('$$map', 'n_sample', 40)", sub
    )

    assert isinstance(sub.widget, QtWidgets.QLineEdit)
    assert sub.widget.text() == "('$$map', 'n_sample', 40)"


def test_nested_cmd_button_passes_widget_info(window):
    sub = _seed_nested_params(window)
    seen = {}
    window._on_cmd_button_clicked = lambda pn, wi=None: seen.update(
        name=pn, info=wi
    )

    # emit() rather than click(): the parameters group box is disabled
    # until a module is selected, and click() is a no-op on a disabled
    # button. The connection is what matters here.
    sub.cmd_button.clicked.emit()

    assert seen["name"] == "n_sample"
    assert seen["info"] is sub


def test_nested_command_value_round_trips(window):
    _seed_nested_params(window)
    cmd = ("$get_prior_result", "results, 00_load_dataset_movie, filepath")

    window._populate_stored_parameters({"sample_movie": {"n_sample": cmd}})

    sub = window.parameter_widgets["sample_movie"].sub_parameters["n_sample"]
    # A spinbox would have silently swallowed the tuple.
    assert isinstance(sub.widget, QtWidgets.QLineEdit)

    top = window.parameter_widgets["sample_movie"]
    out = window._get_widget_value(top.widget, top.original_type, top)
    assert out["n_sample"] == cmd


def test_nested_param_summary_shows_channel_values(window):
    _seed_tree(window)
    window.register_tile_param("n_sample", "channel", 40)
    window.set_tile_param_value("n_sample", 20, "ch0")
    sub = _seed_nested_params(window)

    window._convert_widget_to_textbox(
        "n_sample", "('$$map', 'n_sample', 40)", sub
    )

    text = sub.summary_label.text()
    assert "ch0 = 20" in text
    assert "ch1 = 40" in text


# --- Remove Dataset acts on the selected dataset ------------------------


def _seed_agg_tree(win):
    """Seed the tree and render it in the aggregation tree widget.

    The workflow type must be set first: its change handler resets
    tree_data to defaults.
    """
    win.workflow_type.setCurrentIndex(1)
    _seed_tree(win)
    win._populate_tree_from_data()
    return win.files_tree_agg


def _select_dataset(tree, name):
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        if item.text(0) == name:
            tree.setCurrentItem(item)
            return item
    raise AssertionError(f"dataset {name} not in tree")


def _select_channel(tree, dataset, channel):
    ds_item = _select_dataset(tree, dataset)
    for i in range(ds_item.childCount()):
        child = ds_item.child(i)
        if child.text(1) == channel:
            tree.setCurrentItem(child)
            return child
    raise AssertionError(f"channel {channel} not under {dataset}")


@pytest.fixture
def confirm_yes(monkeypatch):
    """Auto-confirm question dialogs; record information dialogs."""
    shown = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda parent, title, text, *a, **k: shown.append((title, text)),
    )
    return shown


def test_remove_dataset_removes_selected(window, confirm_yes):
    tree = _seed_agg_tree(window)
    _select_dataset(tree, "cellA")

    window.remove_dataset()

    assert window.tree_data["datasets"] == ["cellB"]
    assert "cellA" not in window.tree_data["file_paths"]
    # Channels are untouched - they belong to the remaining dataset too.
    assert window.tree_data["channels"] == ["ch0", "ch1"]
    assert not confirm_yes  # no complaint dialog


def test_remove_dataset_rejects_channel_selection(window, confirm_yes):
    tree = _seed_agg_tree(window)
    _select_channel(tree, "cellA", "ch0")

    window.remove_dataset()

    # Previously this returned silently, removing nothing and saying
    # nothing.
    assert window.tree_data["datasets"] == ["cellA", "cellB"]
    assert len(confirm_yes) == 1
    assert "not a channel" in confirm_yes[0][1]


def test_remove_dataset_informs_when_nothing_selected(window, confirm_yes):
    tree = _seed_agg_tree(window)
    tree.clearSelection()
    tree.setCurrentItem(None)

    window.remove_dataset()

    assert window.tree_data["datasets"] == ["cellA", "cellB"]
    assert len(confirm_yes) == 1
    assert "select a dataset" in confirm_yes[0][1]


def test_remove_dataset_cancel_keeps_data(window, monkeypatch):
    tree = _seed_agg_tree(window)
    _select_dataset(tree, "cellA")
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No,
    )

    window.remove_dataset()

    assert window.tree_data["datasets"] == ["cellA", "cellB"]


def test_remove_dataset_prunes_cell_scope_values(window, confirm_yes):
    tree = _seed_agg_tree(window)
    window.register_tile_param("min_locs", "cell", 10)
    window.set_tile_param_value("min_locs", 5, "ch0", dataset="cellA")
    window.set_tile_param_value("min_locs", 8, "ch0", dataset="cellB")
    _select_dataset(tree, "cellA")

    window.remove_dataset()

    values = window.tree_data["tile_params"]["min_locs"]["values"]
    assert "cellA" not in values
    assert values["cellB"]["ch0"] == 8

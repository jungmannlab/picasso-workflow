#!/usr/bin/env python
"""GUI tests for scope-aware module palette and insert-after-selection.

Exercises the module-spec-driven behaviour added to the workflow builder:
the module dropdown lists only modules valid in the active workflow scope and
greys out those whose inputs are not yet available, and "Add module" inserts
after the selected workflow item. Skips where a Qt GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI palette test")

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


def _palette_items(win):
    cb = win.module_combobox
    return [cb.itemText(i) for i in range(cb.count())]


def _item_enabled(win, name):
    cb = win.module_combobox
    idx = cb.findText(name)
    assert idx >= 0, f"{name} not in palette"
    return cb.model().item(idx).isEnabled()


def _select_palette(win, name):
    cb = win.module_combobox
    idx = cb.findText(name)
    assert idx >= 0, f"{name} not in palette"
    cb.setCurrentIndex(idx)  # sets even if greyed (programmatic)


def test_module_descriptor_implements_all_modules():
    """ModuleDescriptor must implement every AbstractModuleCollection module.

    A missing GUI descriptor leaves an abstract method unimplemented, so
    ``ModuleDescriptor()`` (and hence ``gui.Window()``) raises TypeError at
    construction. The ``window`` fixture swallows that as a skip, so assert it
    directly here: the class must have no leftover abstract methods and must be
    constructible.
    """
    missing = sorted(gui.ModuleDescriptor.__abstractmethods__)
    assert not missing, f"ModuleDescriptor missing descriptors for: {missing}"
    md = gui.ModuleDescriptor()
    params_spec, results_spec = md.register_channels()
    assert isinstance(params_spec, dict) and isinstance(results_spec, dict)


def test_spline_calibration_visibility_follows_fitting_method(window):
    """localize's spline_calibration row is shown only for spline methods."""
    window.workflow_tabs.setCurrentIndex(0)
    window._refresh_module_palette()
    idx = window.module_combobox.findText("localize")
    assert idx >= 0, "localize not in the single-scope palette"
    # Selecting the module populates its parameter form.
    window.module_combobox.setCurrentText("localize")

    pw = window.parameter_widgets
    assert "fitting_method" in pw and "spline_calibration" in pw
    fitting_method = pw["fitting_method"].widget
    spline_row = pw["spline_calibration"].row_widget

    # Default is a gaussian fitter -> spline calibration hidden.
    fitting_method.setCurrentText("gausslq")
    assert spline_row.isHidden()
    # A spline method reveals it.
    fitting_method.setCurrentText("spline")
    assert not spline_row.isHidden()
    fitting_method.setCurrentText("spline-gpu")
    assert not spline_row.isHidden()
    # Back to a gaussian fitter -> hidden again.
    fitting_method.setCurrentText("gaussmle")
    assert spline_row.isHidden()


def test_palette_filtered_by_scope(window):
    # Single tab (index 0) is the default.
    window.workflow_tabs.setCurrentIndex(0)
    items = _palette_items(window)
    assert "load_dataset_movie" in items  # single-scope loader
    assert "load_datasets_to_aggregate" not in items  # aggregation-only
    assert "align_channels" not in items  # aggregation-only

    # Switch to the aggregation tab.
    window.workflow_tabs.setCurrentIndex(1)
    items = _palette_items(window)
    assert "load_datasets_to_aggregate" in items
    assert "load_dataset_movie" not in items  # single-only
    assert "undrift_rcc" not in items  # single-only


def test_palette_greys_unsatisfied_inputs(window):
    window.workflow_tabs.setCurrentIndex(0)
    window.single_workflow_modules.clear()
    window.single_workflow_list.clear()
    window._refresh_module_palette()

    # Nothing produced yet: a loader (no requires) is enabled; render
    # (requires locs_undrifted) is greyed out.
    assert _item_enabled(window, "load_dataset_movie")
    assert not _item_enabled(window, "render")


def test_add_module_inserts_after_selection(window):
    window.workflow_tabs.setCurrentIndex(0)
    window.single_workflow_modules.clear()
    window.single_workflow_list.clear()
    window._refresh_module_palette()

    # Add a loader at the end of the (empty) workflow.
    _select_palette(window, "load_dataset_movie")
    window.add_module()
    assert [m[0] for m in window.single_workflow_modules] == [
        "load_dataset_movie"
    ]

    # raw_movie is now available, so identify is addable; select row 0 and
    # add -> it must land at index 1 (after the selection), not the end.
    window.single_workflow_list.setCurrentRow(0)
    _select_palette(window, "identify")
    window.add_module()
    assert [m[0] for m in window.single_workflow_modules] == [
        "load_dataset_movie",
        "identify",
    ]

    # Select row 0 again and add another module: it inserts at index 1,
    # pushing the previously-added "identify" to index 2 (proves not append).
    window.single_workflow_list.setCurrentRow(0)
    _select_palette(window, "identify")
    window.add_module()
    names = [m[0] for m in window.single_workflow_modules]
    assert names == ["load_dataset_movie", "identify", "identify"]
    # The freshly inserted item is the new selection.
    assert window.single_workflow_list.currentRow() == 1


def test_button_is_save_when_editing_existing_module(window):
    window.workflow_tabs.setCurrentIndex(0)
    window.single_workflow_modules.clear()
    window.single_workflow_list.clear()
    window._refresh_module_palette()

    # No selection yet -> Add mode.
    assert window.add_module_button.text() == "Add module"

    # Add a module; it becomes the selection (an existing, editable item).
    _select_palette(window, "load_dataset_movie")
    window.add_module()
    assert window.editing_workflow_index == 0
    assert window.add_module_button.text() == "Save module"

    # Clicking the button in this state must SAVE (no duplicate), not add.
    window._on_module_button()
    assert [m[0] for m in window.single_workflow_modules] == [
        "load_dataset_movie"
    ]


def test_manual_module_pick_resets_to_add_mode(window):
    window.workflow_tabs.setCurrentIndex(0)
    window.single_workflow_modules.clear()
    window.single_workflow_list.clear()
    window._refresh_module_palette()

    _select_palette(window, "load_dataset_movie")
    window.add_module()
    assert window.add_module_button.text() == "Save module"

    # Manually choosing a module from the palette means "compose new" -> Add.
    cb = window.module_combobox
    idx = cb.findText("dummy_module")
    assert idx >= 0
    cb.setCurrentText("dummy_module")  # fires on_module_changed (unblocked)
    assert window.editing_workflow_index == -1
    assert window.add_module_button.text() == "Add module"

    # And now the button adds rather than saves.
    window.add_module()
    assert [m[0] for m in window.single_workflow_modules] == [
        "load_dataset_movie",
        "dummy_module",
    ]


def test_selecting_existing_row_enters_save_mode(window):
    window.workflow_tabs.setCurrentIndex(0)
    window.single_workflow_modules.clear()
    window.single_workflow_list.clear()
    window._refresh_module_palette()

    _select_palette(window, "load_dataset_movie")
    window.add_module()
    _select_palette(window, "dummy_module")
    window.add_module()  # -> [movie, dummy], two items

    # Re-select the first row: editing an existing item -> Save mode.
    window.single_workflow_list.setCurrentRow(0)
    assert window.editing_workflow_index == 0
    assert window.add_module_button.text() == "Save module"


def test_reference_remap_helper_updates_both_workflows(window):
    """_remap_references_after_change rewrites cross-workflow refs in place."""
    from picasso_workflow import workflow_references as wfref

    window.single_workflow_modules[:] = [
        ("identify", {}),
        ("save_single_dataset", {}),
    ]
    window.aggregation_workflow_modules[:] = [
        (
            "load_to_aggregate",
            {
                "fp": (
                    "$$get_prior_result",
                    "all_results, single_dataset, $$all, "
                    "01_save_single_dataset, filepath",
                )
            },
        ),
    ]
    sgl_id = id(window.single_workflow_modules)
    agg_id = id(window.aggregation_workflow_modules)

    # Insert a single-workflow module at index 0: save moves 01 -> 02.
    changes = window._remap_references_after_change(
        wfref.SINGLE, wfref.insertion_index_map(0, 2)
    )

    # List identities are preserved (slice-assign, not reassignment).
    assert id(window.single_workflow_modules) == sgl_id
    assert id(window.aggregation_workflow_modules) == agg_id
    # The aggregation -> single cross-reference was bumped.
    assert (
        "02_save_single_dataset"
        in window.aggregation_workflow_modules[0][1]["fp"][1]
    )
    assert len(changes) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

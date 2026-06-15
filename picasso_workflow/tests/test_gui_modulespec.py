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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

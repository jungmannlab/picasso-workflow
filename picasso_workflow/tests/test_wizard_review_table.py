#!/usr/bin/env python
"""Tests for the wizard review table's long-word wrapping delegate."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtGui, QtWidgets  # noqa: E402

from picasso_workflow.wizard_manual_to_workflow import (  # noqa: E402
    ManualToWorkflowWizard,
    _WrapAnywhereDelegate,
)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_table():
    table = QtWidgets.QTableWidget(1, 1)
    table.setColumnWidth(0, 60)  # deliberately narrow
    delegate = _WrapAnywhereDelegate(table, table)
    table.setItemDelegate(delegate)
    return table, delegate


def _height(table, delegate, text):
    table.setItem(0, 0, QtWidgets.QTableWidgetItem(text))
    opt = QtWidgets.QStyleOptionViewItem()
    index = table.model().index(0, 0)
    return delegate.sizeHint(opt, index).height()


def test_long_spaceless_token_wraps_to_multiple_lines(qapp):
    # A long token with no whitespace (a path / parameter string) must wrap
    # inside the narrow column -- i.e. the row grows well beyond one text line.
    table, delegate = _make_table()
    line = QtGui.QFontMetrics(table.font()).height()

    short = _height(table, delegate, "x")
    long_token = _height(
        table, delegate, "/Volumes/pool/users/x_" + "a" * 80 + ".hdf5"
    )

    assert short <= 2 * line  # a short label stays ~one line
    assert long_token >= 4 * line  # the long token wrapped onto many lines


def test_wizard_opens_without_qt_parent_but_keeps_main_window(qapp):
    # Opened with NO Qt parent (so it can be stacked behind the main GUI) yet
    # still retains the main window to apply results into.
    main = QtWidgets.QWidget()
    wiz = ManualToWorkflowWizard(parent=None, main_window=main)
    assert wiz.parent() is None
    assert wiz._main_window is main
    wiz.deleteLater()
    main.deleteLater()


def test_wizard_main_window_falls_back_to_parent(qapp):
    # Back-compat: the old positional ManualToWorkflowWizard(window) form still
    # applies results to that window.
    parent = QtWidgets.QWidget()
    wiz = ManualToWorkflowWizard(parent)
    assert wiz._main_window is parent
    wiz.deleteLater()
    parent.deleteLater()


def test_delegate_uses_wrap_anywhere_mode(qapp):
    table, delegate = _make_table()
    table.setItem(0, 0, QtWidgets.QTableWidgetItem("abc"))
    opt = QtWidgets.QStyleOptionViewItem()
    opt.font = table.font()
    doc = delegate._document(opt, table.model().index(0, 0))
    assert (
        doc.defaultTextOption().wrapMode()
        == QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
    )


def test_delegate_preserves_confidence_foreground_color(qapp):
    # The delegate paints with the item's own foreground (confidence tint); it
    # must not crash and must read the ForegroundRole. Paint into a pixmap.
    table, delegate = _make_table()
    item = QtWidgets.QTableWidgetItem("undrift_from_picked")
    item.setForeground(QtGui.QColor("#b36b00"))
    table.setItem(0, 0, item)

    pixmap = QtGui.QPixmap(60, 40)
    pixmap.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(pixmap)
    opt = QtWidgets.QStyleOptionViewItem()
    opt.rect = pixmap.rect()
    opt.font = table.font()
    delegate.paint(painter, opt, table.model().index(0, 0))
    painter.end()  # no exception == the paint path is sound


def test_delete_single_module_remaps_aggregation_reference(qapp):
    # Removing a single-dataset module must renumber the aggregation's
    # $$get_prior_result cross-reference to save_single_dataset (the main GUI
    # does this; the wizard previously did not -> stale index -> runtime
    # PriorResultError).
    from picasso_workflow.wizard_manual_to_workflow import _ReviewPage

    wiz = _ReviewPage(parent=None)
    try:
        wiz._single = [
            ("load_dataset_localizations", {}),
            ("picasso_undrift_aim", {}),
            ("find_similar", {}),
            ("undrift_from_picked", {}),
            ("render", {}),  # index 4 -- the removable module
            ("save_single_dataset", {"filename": ("$$map", "#tags")}),  # idx 5
        ]
        wiz._single_meta = [{} for _ in wiz._single]
        wiz._agg = [
            ("analysis_documentation", {}),
            (
                "load_datasets_to_aggregate",
                {
                    "filepaths": (
                        "$$get_prior_result",
                        "all_results, single_dataset, $$all, "
                        "05_save_single_dataset, filepath",
                    ),
                    "tags": ("$$map", "#tags"),
                },
            ),
            ("save_datasets_aggregated", {}),
        ]
        wiz._agg_meta = [{} for _ in wiz._agg]
        # avoid touching the Qt table / validation plumbing
        wiz._rebuild_table = lambda: None
        wiz._revalidate = lambda: None

        assert wiz._delete_row("single", 4) is True
        # save_single_dataset shifted 05 -> 04; the aggregation ref must follow
        ref = wiz._agg[1][1]["filepaths"][1]
        assert "04_save_single_dataset" in ref
        assert "05_save_single_dataset" not in ref
        # and the single list lost exactly the render module
        assert [m[0] for m in wiz._single] == [
            "load_dataset_localizations",
            "picasso_undrift_aim",
            "find_similar",
            "undrift_from_picked",
            "save_single_dataset",
        ]
    finally:
        wiz.deleteLater()

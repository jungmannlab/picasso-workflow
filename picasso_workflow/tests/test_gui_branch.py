#!/usr/bin/env python
"""GUI tests for the ``branch`` module editor.

Exercises the workflow-builder support for branch modules:

* rendering ``branch_modules`` / ``join_modules`` as indented sub-rows in the
  workflow list (with node descriptors so row operations resolve to the owning
  module);
* selecting a sub-row to edit that sub-module's own parameters;
* the branch-id selector and per-branch ``("$branch", [...])`` overrides
  (display, per-branch editing, and the shared<->branch-specific toggle);
* structural edits: reorder within a branch, move a sub-module out to the top
  level, move a top-level module into a branch, and add/remove within a branch;
* the guard that editing a branch module's own parameters never clobbers its
  inline-managed sub-workflows.

Skips where a Qt GUI cannot be constructed.
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


def _branch_workflow():
    """Aggregation workflow: a module, a branch (3 subs + 1 join), a module."""
    return [
        ("align_channels", {}),
        (
            "branch",
            {
                "branch_type": "explicit",
                "n_branches": 3,
                "branch_labels": ["cell0", "cell1", "cell2"],
                "branch_modules": [
                    (
                        "create_mask2",
                        {
                            "nth_largest_cell": ("$branch", [1, 2, 3]),
                            "select_cell": True,
                        },
                    ),
                    ("nneighbor", {"nth_NN": 4}),
                    ("fit_csr", {}),
                ],
                "join_modules": [("dummy_module", {})],
            },
        ),
        ("save_datasets_aggregated", {}),
    ]


def _seed(win, modules=None):
    """Load a branch workflow into the aggregation tab, editing state clean."""
    modules = modules if modules is not None else _branch_workflow()
    win.aggregation_workflow_modules = modules
    win.workflow_tabs.setCurrentIndex(1)
    # A real workflow load resets editing state; mirror that here.
    win.editing_workflow_index = -1
    win.editing_workflow_tab = -1
    win.editing_workflow_subnode = None
    win._clear_branch_context()
    win._refresh_workflow_list(win.aggregation_workflow_list, modules)
    return modules, win.aggregation_workflow_list


def _select(win, lw, top, section=None, sub=None):
    """Select the row for a top module, or a branch sub-module."""
    for row in range(lw.count()):
        node = win._row_node(lw, row)
        if node is None:
            continue
        if section is None:
            if node["kind"] == "module" and node["top"] == top:
                lw.setCurrentRow(row)
                return
        elif (
            node["kind"] == "sub"
            and node["top"] == top
            and node["section"] == section
            and node["sub"] == sub
        ):
            lw.setCurrentRow(row)
            return
    raise AssertionError(f"row not found: {top} {section} {sub}")


def _branch_module_names(modules):
    return [s[0] for s in modules[1][1].get("branch_modules", [])]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def test_branch_submodules_render_indented(window):
    modules, lw = _seed(window)
    # 3 top modules + 3 branch subs + 1 join sub = 7 rows
    assert lw.count() == 7
    kinds = [window._row_node(lw, r)["kind"] for r in range(lw.count())]
    assert kinds.count("sub") == 4
    # a sub-row resolves to the owning branch (top index 1)
    _select(window, lw, 1, "branch", 1)
    assert window._selected_module_index(lw) == 1
    # the trailing module keeps its true index despite the indented subs
    assert window._module_item_row(lw, 2) == lw.count() - 1


# --------------------------------------------------------------------------
# Editing a sub-module's own parameters
# --------------------------------------------------------------------------
def test_edit_submodule_params(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 1)  # nneighbor
    assert window.editing_workflow_subnode == {"section": "branch", "sub": 1}
    window.parameter_widgets["nth_NN"].widget.setValue(9)
    window._update_editing_workflow_item()
    sub_name, sub_params = modules[1][1]["branch_modules"][1]
    assert sub_name == "nneighbor"
    assert sub_params["nth_NN"] == 9
    # sibling untouched
    assert modules[1][1]["branch_modules"][0][0] == "create_mask2"


# --------------------------------------------------------------------------
# Branch-id selector + per-branch overrides
# --------------------------------------------------------------------------
def test_branch_id_override_display_and_edit(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 0)  # create_mask2 with $branch override
    assert window.editing_branch_context["n_branches"] == 3
    assert "nth_largest_cell" in window._branch_override_params
    wi = window.parameter_widgets["nth_largest_cell"]
    # branch 0 -> 1, branch 2 -> 3
    assert window._get_widget_value(wi.widget, wi.original_type, wi) == 1
    window.branch_id_combobox.setCurrentIndex(2)
    wi = window.parameter_widgets["nth_largest_cell"]
    assert window._get_widget_value(wi.widget, wi.original_type, wi) == 3
    # edit branch 2 -> 9; other branches preserved
    wi.widget.setValue(9)
    window._update_editing_workflow_item()
    assert modules[1][1]["branch_modules"][0][1]["nth_largest_cell"] == (
        "$branch",
        [1, 2, 9],
    )


def test_per_branch_toggle(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 1)  # nneighbor, shared nth_NN
    wi = window.parameter_widgets["nth_NN"]
    # make branch-specific
    wi.per_branch_checkbox.setChecked(True)
    stored = modules[1][1]["branch_modules"][1][1]["nth_NN"]
    assert stored == ("$branch", [4, 4, 4])
    # edit branch 2 then toggle back to shared -> adopts branch 2's value
    window.branch_id_combobox.setCurrentIndex(2)
    window.parameter_widgets["nth_NN"].widget.setValue(7)
    window._update_editing_workflow_item()
    assert modules[1][1]["branch_modules"][1][1]["nth_NN"] == (
        "$branch",
        [4, 4, 7],
    )
    window.parameter_widgets["nth_NN"].per_branch_checkbox.setChecked(False)
    assert modules[1][1]["branch_modules"][1][1]["nth_NN"] == 7


# --------------------------------------------------------------------------
# Structural edits (Step 4)
# --------------------------------------------------------------------------
def test_reorder_within_branch(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 1)  # nneighbor
    window._move_selected(-1)  # up: swap with create_mask2
    assert _branch_module_names(modules) == [
        "nneighbor",
        "create_mask2",
        "fit_csr",
    ]


def test_move_sub_out_of_branch(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 0)
    window._move_selected(-1)  # first sub up -> top-level before branch
    names = [n for n, _ in modules]
    assert names == [
        "align_channels",
        "create_mask2",
        "branch",
        "save_datasets_aggregated",
    ]
    # branch now holds the remaining two
    branch = next(m for m in modules if m[0] == "branch")
    assert [s[0] for s in branch[1]["branch_modules"]] == [
        "nneighbor",
        "fit_csr",
    ]


def test_move_top_module_into_branch(window):
    modules, lw = _seed(window)
    _select(window, lw, 0)  # align_channels, above the branch
    window._move_selected(1)  # down -> becomes first branch sub-module
    branch = next(m for m in modules if m[0] == "branch")
    assert [s[0] for s in branch[1]["branch_modules"]] == [
        "align_channels",
        "create_mask2",
        "nneighbor",
        "fit_csr",
    ]


def test_move_join_sub_out_and_crossings(window):
    # workflow whose branch has both a branch and a join section
    modules = [
        ("align_channels", {}),
        (
            "branch",
            {
                "branch_type": "explicit",
                "n_branches": 2,
                "branch_modules": [("create_mask2", {}), ("nneighbor", {})],
                "join_modules": [("dummy_module", {}), ("fit_csr", {})],
            },
        ),
        ("save_datasets_aggregated", {}),
    ]
    _, lw = _seed(window, modules)

    def secs():
        b = next(m for m in modules if m[0] == "branch")[1]
        return (
            [s[0] for s in b["branch_modules"]],
            [s[0] for s in b["join_modules"]],
        )

    # last join sub, down -> out to top-level below the branch
    _select(window, lw, 1, "join", 1)  # fit_csr (last join)
    window._move_selected(1)
    assert [n for n, _ in modules] == [
        "align_channels",
        "branch",
        "fit_csr",
        "save_datasets_aggregated",
    ]
    assert secs() == (["create_mask2", "nneighbor"], ["dummy_module"])


def test_join_module_toggle(window):
    # a branch with no join modules; make a sub a join module and back
    modules = [
        ("align_channels", {}),
        (
            "branch",
            {
                "branch_type": "explicit",
                "n_branches": 2,
                "branch_modules": [
                    (
                        "create_mask2",
                        {"nth_largest_cell": ("$branch", [1, 2])},
                    ),
                    ("nneighbor", {}),
                ],
                "join_modules": [],
            },
        ),
    ]
    _, lw = _seed(window, modules)

    def secs():
        b = modules[1][1]
        return (
            [s[0] for s in b.get("branch_modules", [])],
            [s[0] for s in b.get("join_modules", [])],
        )

    _select(window, lw, 1, "branch", 0)  # create_mask2 (has $branch override)
    window.join_module_checkbox.setChecked(True)
    assert secs() == (["nneighbor"], ["create_mask2"])
    # per-branch override collapsed to a concrete value in the join module
    assert modules[1][1]["join_modules"][0][1]["nth_largest_cell"] == 1
    # selecting the join sub: toggle checked, branch-id selector hidden
    _select(window, lw, 1, "join", 0)
    assert window.join_module_checkbox.isChecked()
    assert window.editing_branch_context is None
    # move it back to the branch section
    window.join_module_checkbox.setChecked(False)
    assert secs() == (["nneighbor", "create_mask2"], [])


def test_remove_within_branch(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 1)  # nneighbor
    window.remove_selected()
    assert _branch_module_names(modules) == ["create_mask2", "fit_csr"]
    # the branch module and its neighbours survive
    assert [n for n, _ in modules] == [
        "align_channels",
        "branch",
        "save_datasets_aggregated",
    ]


def test_add_into_branch(window):
    modules, lw = _seed(window)
    _select(window, lw, 1, "branch", 0)  # create_mask2
    # picking a module clears editing state but leaves the list selection
    window.module_combobox.setCurrentText("nneighbor")
    window.add_module()
    assert _branch_module_names(modules) == [
        "create_mask2",
        "nneighbor",
        "nneighbor",
        "fit_csr",
    ]


def test_editing_branch_module_preserves_submodules(window):
    """Editing the branch module's own params must not wipe its sub-modules."""
    modules, lw = _seed(window)
    _select(window, lw, 1)  # the branch module itself
    window.parameter_widgets["n_branches"].widget.setValue(5)
    window._update_editing_workflow_item()
    assert modules[1][1]["n_branches"] == 5
    assert _branch_module_names(modules) == [
        "create_mask2",
        "nneighbor",
        "fit_csr",
    ]

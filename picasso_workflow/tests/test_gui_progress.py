#!/usr/bin/env python
"""GUI test for the Run tab's multi-stage progress monitor logic.

Exercises the pure aggregation/labelling helpers on the ``Window`` class
without constructing the full Qt window (they touch no widgets), so it runs
wherever PyQt6 imports. Skips gracefully where PyQt6 is unavailable.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI test")

from picasso_workflow import gui  # noqa: E402


@pytest.fixture
def win():
    # bypass __init__ (no QApplication / widgets needed for these helpers)
    return gui.Window.__new__(gui.Window)


def _single(idx, state, modules):
    return {
        "kind": "single",
        "report_name": f"myrun_sgl_{idx:02d}_cell{idx}_240101-1200",
        "state": state,
        "total": len(modules),
        "current": next(
            (i for i, m in enumerate(modules) if m[1] == "running"),
            None,
        ),
        "modules": [
            {"i": i, "name": n, "status": s, "fraction": f}
            for i, (n, s, f) in enumerate(modules)
        ],
    }


def _agg(dataset_states):
    return {
        "kind": "aggregation",
        "report_name": "myrun_240101-1200",
        "state": "running",
        "datasets": [
            {"i": i, "tag": f"cell{i}", "state": s}
            for i, s in enumerate(dataset_states)
        ],
    }


def test_stage_label(win):
    assert win._stage_label(_single(2, "running", [])) == (
        "[single 02] myrun_sgl_02_cell2"
    )
    agg_stage = {"report_name": "myrun_aggregation_240101-1200"}
    assert win._stage_label(agg_stage) == (
        "[aggregation stage] myrun_aggregation"
    )


def test_sorted_singles_orders_datasets_then_aggregation(win):
    s0 = _single(0, "done", [])
    s1 = _single(1, "running", [])
    agg_stage = {
        "kind": "single",
        "report_name": "myrun_aggregation_240101-1200",
        "state": "pending",
        "modules": [],
    }
    ordered = win._sorted_singles([agg_stage, s1, s0])
    names = [win._stage_label(s) for s in ordered]
    assert names[0].startswith("[single 00]")
    assert names[1].startswith("[single 01]")
    assert names[2].startswith("[aggregation stage]")


def test_overall_progress_combines_singles_and_agg_stage(win):
    # 2 datasets: #0 done (1.0), #1 running at 0.5; no agg stage yet
    agg = _agg(["done", "running"])
    s0 = _single(0, "done", [("a", "done", 1.0)])
    s1 = _single(1, "running", [("a", "running", 0.5)])
    overall = win._overall_progress(agg, [s0, s1], [agg, s0, s1])
    # (1.0 + 0.5) / 2 datasets = 0.75
    assert abs(overall - 0.75) < 1e-6


def test_overall_progress_weights_aggregation_stage(win):
    # both datasets done, aggregation stage half done -> (2 + 0.5)/(2+1)
    agg = _agg(["done", "done"])
    s0 = _single(0, "done", [("a", "done", 1.0)])
    s1 = _single(1, "done", [("a", "done", 1.0)])
    agg_stage = {
        "kind": "single",
        "report_name": "myrun_aggregation_240101-1200",
        "state": "running",
        "total": 2,
        "current": 1,
        "modules": [
            {"i": 0, "name": "a", "status": "done", "fraction": 1.0},
            {"i": 1, "name": "b", "status": "running", "fraction": 0.0},
        ],
    }
    overall = win._overall_progress(
        agg, [s0, s1, agg_stage], [agg, s0, s1, agg_stage]
    )
    assert abs(overall - (2.5 / 3)) < 1e-6


def test_overall_progress_single_run(win):
    s = _single(0, "running", [("a", "done", 1.0), ("b", "running", 0.0)])
    # no aggregation state -> mean of module fractions = (1 + 0)/2 = 0.5
    assert abs(win._overall_progress(None, [s], [s]) - 0.5) < 1e-6


def test_active_stage_module_label_points_at_running(win):
    s0 = _single(0, "done", [("a", "done", 1.0)])
    s1 = _single(1, "running", [("a", "done", 1.0), ("b", "running", 0.3)])
    label = win._active_stage_module_label([_agg(["done", "running"]), s0, s1])
    assert "[single 01]" in label
    assert "module 2/2 b" in label


def test_scope_states_filters_by_job_id(win):
    """Only stages carrying the current job id are kept; a still-pending
    resubmission (no matching stages) yields nothing, not the old run."""
    old = {
        "kind": "aggregation",
        "report_name": "myrun_5834387",
        "state": "done",
    }
    old_sgl = {
        "kind": "single",
        "report_name": "myrun_5834387_sgl_00_cell0_260901-1740",
        "state": "done",
    }
    new_sgl = {
        "kind": "single",
        "report_name": "myrun_5837262_sgl_00_cell0_260902-1021",
        "state": "running",
    }
    states = [old, old_sgl, new_sgl]

    # current job -> only its stages
    scoped = win._scope_states_to_current_run(states, "5837262")
    assert scoped == [new_sgl]

    # a pending resubmission whose stages have not been written yet -> empty
    assert win._scope_states_to_current_run([old, old_sgl], "5837262") == []

    # no job id (local monitoring) -> unchanged
    assert win._scope_states_to_current_run(states, "") == states

    # job id must match as a whole token, not a substring of a longer id
    assert win._scope_states_to_current_run(states, "372") == []


@pytest.fixture(scope="module")
def full_window():
    """A real Window (widgets constructed), for display-rendering tests."""
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        w = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    yield w
    w.close()
    del app


def test_completed_but_unfinished_is_flagged(full_window):
    """SLURM COMPLETED while progress < 100% is flagged, not shown green."""
    slurm = {"success": True, "status": "COMPLETED", "details": {}}
    # a single run stuck mid-way: module b still running at 30%
    s = _single(0, "running", [("a", "done", 1.0), ("b", "running", 0.3)])
    full_window._update_monitor_display(slurm, [s])
    text = full_window.monitor_state_label.text()
    assert "unfinished" in text
    assert "stopped at" in text
    assert "#f9a825" in full_window.monitor_state_label.styleSheet()


def test_completed_and_finished_shows_plain_completed(full_window):
    """A genuinely finished run keeps the plain green COMPLETED chip."""
    slurm = {"success": True, "status": "COMPLETED", "details": {}}
    s = _single(0, "done", [("a", "done", 1.0), ("b", "done", 1.0)])
    full_window._update_monitor_display(slurm, [s])
    text = full_window.monitor_state_label.text()
    assert text == "Job state: COMPLETED"
    assert "unfinished" not in text


def test_pending_job_with_no_states_shows_pending_empty(full_window):
    """A PENDING resubmission (its stages scoped out) shows PENDING and an
    empty tree, not a previous run's progress."""
    slurm = {"success": True, "status": "PENDING", "details": {}}
    full_window._update_monitor_display(slurm, [])
    assert full_window.monitor_state_label.text() == "Job state: PENDING"
    assert full_window.module_tree.topLevelItemCount() == 0
    assert full_window.overall_progress_bar.value() == 0


def test_tree_expansion_survives_refresh(full_window):
    """The user's expand/collapse of a node persists across refreshes, so
    investigating details is not undone on every poll."""
    agg = _agg(["running", "pending"])
    s0 = _single(0, "running", [("a", "done", 1.0), ("b", "running", 0.3)])
    full_window._update_monitor_display(None, [agg, s0])

    # set a known expansion on the now-attached items
    root = full_window.module_tree.topLevelItem(0)
    root.setExpanded(True)
    root.child(0).setExpanded(True)  # the ds:0 stage

    # a poll with advanced progress rebuilds the tree -> expansion kept
    s0b = _single(0, "running", [("a", "done", 1.0), ("b", "running", 0.8)])
    full_window._update_monitor_display(None, [agg, s0b])
    root = full_window.module_tree.topLevelItem(0)
    assert root.isExpanded()
    assert root.child(0).isExpanded()

    # collapsing likewise persists across the next refresh
    root.setExpanded(False)
    root.child(0).setExpanded(False)
    full_window._update_monitor_display(None, [agg, s0b])
    root = full_window.module_tree.topLevelItem(0)
    assert not root.isExpanded()
    assert not root.child(0).isExpanded()

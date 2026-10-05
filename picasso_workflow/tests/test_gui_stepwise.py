#!/usr/bin/env python
"""GUI tests for stepwise (module-by-module) workflow development.

The stepwise controls must bake a ``stop_after`` argument into the
generated ``start_workflow.py``, force resume semantics while stepping,
and be unavailable for Investigation workflows.

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


def _populate(window, workflow_type):
    """Set the workflow type and give both workflows some modules."""
    window.workflow_type.setCurrentIndex(workflow_type)
    window.single_workflow_modules = [
        ("load_dataset_movie", {}),
        ("identify", {}),
    ]
    window.aggregation_workflow_modules = [("load_datasets_to_aggregate", {})]
    window._refresh_stepwise_targets()


def _generate(window, tmp_path):
    window.results_folder_display.setText(str(tmp_path))
    path = window.create_python_script("local", "localhost")
    assert path is not None
    with open(path) as f:
        return f.read()


def test_stepwise_single_bakes_stop_after(window, tmp_path):
    _populate(window, 0)
    window.files_mode_combo.setCurrentIndex(2)  # no input files
    window.continue_previous.setChecked(False)
    window.stepwise_enable.setChecked(True)
    window._set_stepwise_target("single", 1)

    content = _generate(window, tmp_path)
    assert "stop_after=1," in content
    # stepping implies resuming
    assert "continue_previous_runners=True" in content


def test_no_stepwise_no_stop_after(window, tmp_path):
    _populate(window, 0)
    window.files_mode_combo.setCurrentIndex(2)

    content = _generate(window, tmp_path)
    assert "stop_after" not in content


def test_stepwise_aggregation_bakes_phase_tuple(window, tmp_path):
    _populate(window, 1)
    window.stepwise_enable.setChecked(True)
    window._set_stepwise_target("aggregation", 0)

    content = _generate(window, tmp_path)
    assert "stop_after=('aggregation', 0)," in content

    window._set_stepwise_target("single", 1)
    content = _generate(window, tmp_path)
    assert "stop_after=('single', 1)," in content


def test_stepwise_forces_resume_and_restores(window):
    _populate(window, 0)
    window.continue_previous.setChecked(False)

    window.stepwise_enable.setChecked(True)
    assert window.continue_previous.isChecked()
    assert not window.continue_previous.isEnabled()

    window.stepwise_enable.setChecked(False)
    assert window.continue_previous.isEnabled()
    assert not window.continue_previous.isChecked()


def test_stepwise_target_combo_lists_both_phases(window):
    _populate(window, 1)
    combo = window.stepwise_target
    data = [combo.itemData(row) for row in range(combo.count())]
    assert ("single", 0) in data
    assert ("single", 1) in data
    assert ("aggregation", 0) in data


def test_stepwise_unsupported_for_investigation(window):
    _populate(window, 0)
    window.stepwise_enable.setChecked(True)

    _populate(window, 2)
    assert not window.stepwise_enable.isEnabled()
    assert not window.stepwise_enable.isChecked()

#!/usr/bin/env python
"""GUI test for the "Continue previous run (resume)" option.

The checkbox state must be baked into the generated ``start_workflow.py``
as the ``continue_previous_runners`` argument of the coordinator's
``run_analysis`` call.

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


def _generate(window, tmp_path):
    window.results_folder_display.setText(str(tmp_path))
    window.workflow_type.setCurrentIndex(0)  # Single Workflow
    window.files_mode_combo.setCurrentIndex(2)  # no input files
    path = window.create_python_script("local", "localhost")
    assert path is not None
    with open(path) as f:
        return f.read()


def test_resume_checkbox_bakes_flag_into_script(window, tmp_path):
    window.continue_previous.setChecked(True)
    content = _generate(window, tmp_path)
    assert "continue_previous_runners=True" in content

    window.continue_previous.setChecked(False)
    content = _generate(window, tmp_path)
    assert "continue_previous_runners=False" in content

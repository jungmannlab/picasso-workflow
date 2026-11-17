#!/usr/bin/env python
"""GUI test: Investigation-workflow script generation is refused.

The GUI cannot currently generate a runnable Investigation script (the emitted
driver is out of sync with ``InvestigationWorkflowCoordinator``), so
``create_python_script`` must refuse with a warning and return ``None`` instead
of writing a broken script.

Skips where a Qt GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from unittest.mock import patch  # noqa: E402

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


def test_investigation_script_generation_refused(window, tmp_path):
    """Investigation returns None (no script written) and warns the user."""
    window.results_folder_display.setText(str(tmp_path))
    window.workflow_type.setCurrentIndex(2)  # Investigation Workflow

    with patch.object(QtWidgets.QMessageBox, "warning") as mock_warn:
        path = window.create_python_script("local", "localhost")

    assert path is None
    assert mock_warn.called
    assert not os.path.exists(os.path.join(str(tmp_path), "start_workflow.py"))


def test_single_workflow_still_generates(window, tmp_path):
    """The guard does not affect a normal Single workflow."""
    window.results_folder_display.setText(str(tmp_path))
    window.workflow_type.setCurrentIndex(0)  # Single Workflow
    window.files_mode_combo.setCurrentIndex(2)  # no input files

    path = window.create_python_script("local", "localhost")
    assert path is not None
    assert os.path.exists(path)

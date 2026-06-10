#!/usr/bin/env python
"""Integration test for the GUI Results/report tab.

Drives the Results tab headlessly: point it at a saved run folder, generate
the HTML report and verify the report + per-module status populate. Skips
gracefully where a Qt GUI cannot be constructed (no display / PyQt6).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tab test")

import yaml  # noqa: E402
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


def _write_run(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "WorkflowRunner.yaml").write_text(
        yaml.dump(
            {
                "results": {
                    "00_dummy_module": {
                        "start time": "t0",
                        "end time": "t1",
                        "duration": 1.0,
                        "success": True,
                    }
                },
                "reporter_config": {"report_name": "run"},
                "analysis_config": {},
                "workflow_modules": [("dummy_module", {})],
            }
        )
    )


def test_results_tab_present_and_enabled(window):
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Results" in titles
    assert window.tabs.isTabEnabled(titles.index("Results"))
    assert hasattr(window, "report_view")
    assert hasattr(window, "run_combo")


def test_documentation_toggles_default_state(window):
    # Confluence on by default (preserves prior behaviour), HTML off
    assert window.document_confluence_checkbox.isChecked() is True
    assert window.document_html_checkbox.isChecked() is False


def test_run_dropdown_lists_runs_under_results_folder(window, tmp_path):
    base = tmp_path / "results"
    _write_run(base / "run_a_240101-1200")
    _write_run(base / "run_b_240101-1300")
    (base / "not_a_run").mkdir(parents=True, exist_ok=True)

    # setting the results folder auto-rescans the run dropdown
    window.results_folder_display.setText(str(base))

    paths = {
        window.run_combo.itemData(i) for i in range(window.run_combo.count())
    }
    assert str(base / "run_a_240101-1200") in paths
    assert str(base / "run_b_240101-1300") in paths
    # a plain folder without runner state is not listed
    assert str(base / "not_a_run") not in paths


def test_run_dropdown_finds_runs_nested_in_wrapper(window, tmp_path):
    # mirrors the real layout: runs live inside an AnalysisResults-* wrapper
    base = tmp_path / "260601_t"
    wrapper = base / "AnalysisResults-260601_t"
    _write_run(wrapper / "260601_t_260602-0945")
    _write_run(wrapper / "260601_t_260603-0919")
    (base / "logs").mkdir(parents=True, exist_ok=True)

    window.results_folder_display.setText(str(base))

    labels = {
        window.run_combo.itemText(i) for i in range(window.run_combo.count())
    }
    paths = {
        window.run_combo.itemData(i) for i in range(window.run_combo.count())
    }
    # found despite the intermediate wrapper folder
    assert str(wrapper / "260601_t_260602-0945") in paths
    assert str(wrapper / "260601_t_260603-0919") in paths
    # unique basenames -> short labels
    assert "260601_t_260602-0945" in labels


def test_results_tab_generates_and_shows_report(window, tmp_path):
    base = tmp_path / "results"
    folder = base / "run_240101-1200"
    _write_run(folder)

    window.results_folder_display.setText(str(base))
    idx = window.run_combo.findData(str(folder))
    assert idx >= 0
    window.run_combo.setCurrentIndex(idx)
    window._results_refresh()

    # the report was generated and recorded
    assert window._report_path == str(folder / "report.html")
    assert os.path.isfile(window._report_path)
    assert "Dummy Module" in open(window._report_path, encoding="utf-8").read()

    # the per-module status overview is populated
    assert window.report_status.topLevelItemCount() == 1
    item = window.report_status.topLevelItem(0)
    assert item.text(0) == "00_dummy_module"
    assert item.text(1) == "OK"


def test_results_tab_handles_no_selection(window, tmp_path, monkeypatch):
    # silence the (otherwise modal) warning dialog so the test can't block
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    # results folder with no runs -> empty dropdown -> refresh is a no-op
    base = tmp_path / "empty"
    base.mkdir()
    window.results_folder_display.setText(str(base))
    assert window.run_combo.count() == 0
    window._results_refresh()  # must not raise

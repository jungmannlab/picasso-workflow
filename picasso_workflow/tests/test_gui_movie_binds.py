#!/usr/bin/env python
"""GUI tests for binding input-data directories into the Apptainer run.

A cluster run executes ``start_workflow.py`` inside an Apptainer image, which
only sees host paths that are explicitly ``--bind``-ed. The results folder is
always bound, but the raw acquisition folders holding the input movies -- and
any other data file named in a module parameter -- live elsewhere. Without
binding them, the containerized module fails with ``FileNotFoundError``. These
tests pin ``_input_data_bind_dirs`` and its ``_dataset_movie_paths`` helper.
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


def _seed_aggregation(win, file_paths):
    win.workflow_type.setCurrentIndex(1)  # Aggregation Workflow
    win.tree_data["datasets"] = ["origami"]
    win.tree_data["channels"] = list(file_paths)
    win.tree_data["file_paths"] = {"origami": dict(file_paths)}
    win.tree_data["conditions"] = {}
    win.tree_data["tile_params"] = {}
    win.single_workflow_modules = []
    win.aggregation_workflow_modules = []


def test_aggregation_binds_movie_parent_dirs(window):
    """Each channel's movie dir is bound once, de-duplicated per folder."""
    _seed_aggregation(
        window,
        {
            "R1": "/data/exp/raw/acq_R1_1/acq_R1_NDTiffStack.tif",
            "R3": "/data/exp/raw/acq_R3_1/acq_R3_NDTiffStack.tif",
        },
    )

    dirs = window._input_data_bind_dirs(None)

    assert dirs == [
        "/data/exp/raw/acq_R1_1",
        "/data/exp/raw/acq_R3_1",
    ]


def test_aggregation_dedupes_shared_folder(window):
    """Two channels whose movies share one folder yield a single bind."""
    _seed_aggregation(
        window,
        {
            "R1": "/data/exp/raw/acq_1/mov_R1.tif",
            "R3": "/data/exp/raw/acq_1/mov_R3.tif",
        },
    )

    dirs = window._input_data_bind_dirs(None)

    assert dirs == ["/data/exp/raw/acq_1"]


def test_single_autodetect_binds_nothing(window):
    """Auto-detect mode has no explicit input files to bind."""
    window.workflow_type.setCurrentIndex(0)  # Single Workflow
    window.files_mode_combo.setCurrentIndex(1)  # auto-detect raw files
    window.single_workflow_modules = []
    window.aggregation_workflow_modules = []

    assert window._input_data_bind_dirs(None) == []


def test_module_param_input_file_is_bound_when_present(tmp_path, window):
    """A real data file named in a module parameter gets its dir bound."""
    calib = tmp_path / "calib" / "z_calibration.yaml"
    calib.parent.mkdir(parents=True)
    calib.write_text("mock: calibration\n")

    _seed_aggregation(window, {"R1": "/data/exp/raw/acq_1/mov_R1.tif"})
    window.single_workflow_modules = [
        ("localize", {"calib_file": str(calib)}),
    ]

    dirs = window._input_data_bind_dirs(None)

    assert "/data/exp/raw/acq_1" in dirs
    assert str(calib.parent) in dirs


def test_module_param_nonexistent_path_is_skipped(window):
    """A path-like parameter that is not a real local file is not bound."""
    _seed_aggregation(window, {"R1": "/data/exp/raw/acq_1/mov_R1.tif"})
    # A bare output filename (endswith .hdf5 -> looks path-like) and an
    # absolute input that does not exist: neither should be bound.
    window.single_workflow_modules = [
        ("save", {"filename": "locs.hdf5"}),
        ("mask", {"mask_file": "/nope/does_not_exist/mask.hdf5"}),
    ]

    dirs = window._input_data_bind_dirs(None)

    assert dirs == ["/data/exp/raw/acq_1"]

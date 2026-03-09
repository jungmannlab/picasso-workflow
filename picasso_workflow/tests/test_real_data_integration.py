#!/usr/bin/env python
"""
Module Name: test_real_data_integration.py
Author: Heinrich Grabmayr
Description: Integration tests that require real acquired datasets from the
    lab network volumes.

All tests here carry both the `integration` and `real_data` markers and use
the `network_test_data` fixture (defined in conftest.py).  They are skipped
automatically when the data directory is not mounted.

Run on a lab machine with:

    export PW_TEST_DATA_DIR=/Volumes/pool-miblab1/users/<you>/test-datasets
    pytest -m "integration and real_data"

Or add the following to ~/.config/picasso_workflow/config.yaml and run
pytest -m "integration and real_data" without the environment variable:

    TestData:
      directory: /Volumes/pool-miblab1/users/<you>/test-datasets

Expected layout under PW_TEST_DATA_DIR
---------------------------------------
The directory may contain any structure; tests discover files automatically.
Specific sub-tests document their requirements in their docstrings.

    <PW_TEST_DATA_DIR>/
        some_acquisition/
            some_acquisition_MMStack_Pos0.ome.tif   # 2D movie
        3d_acquisition/
            3d_acquisition_MMStack_Pos0.ome.tif     # 3D movie (for zfit)
"""
import importlib.util
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("picasso")

from picasso_workflow.workflow import WorkflowRunner  # noqa: E402
from picasso_workflow.tests.conftest import (  # noqa: E402
    analysis_config as _analysis_config,
    dummy_reporter_config as _dummy_reporter_config,
    _try_call_get_workflow,
)


logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.real_data]

_MOVIE_EXTENSIONS = (".ome.tif", ".tif", ".tiff", ".czi", ".raw")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _find_movies(base_dir, max_files=5):
    """Return up to max_files movie file paths found under base_dir."""
    found = []
    for root, _, files in os.walk(base_dir):
        for fname in sorted(files):
            if any(fname.endswith(ext) for ext in _MOVIE_EXTENSIONS):
                found.append(os.path.join(root, fname))
                if len(found) >= max_files:
                    return found
    return found


# ---------------------------------------------------------------------------
# load_picassoconfig
# ---------------------------------------------------------------------------

def test_load_picassoconfig(tmp_path):
    """Verify that the picasso config referenced in config.yaml is accessible
    and can be loaded by the load_picassoconfig module.

    Confirms both that the pool-miblab5 volume is mounted and that the config
    file itself is well-formed.  Does not require PW_TEST_DATA_DIR.
    """
    from picasso_workflow import CONFIG

    config_path = CONFIG.get("PicassoParameters", {}).get("config")
    if not config_path or not os.path.isfile(config_path):
        pytest.skip(
            f"Picasso config not accessible at {config_path!r}. "
            "Is the pool-miblab5 volume mounted?"
        )

    workflow_modules = [
        ("load_picassoconfig", {"fp_config": config_path}),
    ]

    with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock):
        wr = WorkflowRunner.config_from_dicts(
            _dummy_reporter_config("test_rd_load_picassoconfig"),
            _analysis_config(str(tmp_path)),
            workflow_modules,
        )
        wr.run()

    result = wr.results.get("00_load_picassoconfig", {})
    assert result.get("success"), "load_picassoconfig did not succeed"


# ---------------------------------------------------------------------------
# Minimal 2D pipeline on real movies
# ---------------------------------------------------------------------------

def test_minimal_pipeline_on_real_data(network_test_data, tmp_path):
    """Run load → identify (auto net_gradient) → localize on real movies.

    Exercises the auto_netgrad heuristic on genuine photon-count data and
    confirms that picasso's identify/localize chain works on actual
    acquisitions.  Tests up to 3 movies found under PW_TEST_DATA_DIR.

    Each movie is processed independently; a per-movie sub-directory under
    tmp_path is used as the result location so runs don't interfere.
    """
    movies = _find_movies(network_test_data, max_files=3)
    if not movies:
        pytest.skip(f"No movie files found under {network_test_data}")

    for movie_path in movies:
        label = os.path.relpath(movie_path, network_test_data).replace(
            os.sep, "_"
        )
        result_dir = tmp_path / label
        result_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running minimal pipeline on {movie_path}")

        workflow_modules = [
            (
                "load_dataset_movie",
                {
                    "filename": movie_path,
                    "sample_movie": {
                        "filename": "selected_frames.mp4",
                        "n_sample": 40,
                        "max_quantile": 0.9998,
                        "fps": 2,
                    },
                },
            ),
            (
                "identify",
                {
                    "auto_netgrad": {
                        "filename": "ng_histogram.png",
                        "frame_numbers": (
                            "$get_previous_module_result",  # get from prior results
                            "sample_movie, sample_frame_idx",
                        ),
                        "box_size": 7,
                        "start_ng": -3000,
                        "zscore": 5,
                    },
                    "ids_vs_frame": {"filename": "ids_vs_frame.png"},
                    "box_size": 7,
                },
            ),
            (
                "localize",
                {"fit_method": "lsq", "box_size": 7, "fit_parallel": False},
            ),
        ]

        with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock):
            wr = WorkflowRunner.config_from_dicts(
                _dummy_reporter_config(f"test_rd_minimal_{label}"),
                _analysis_config(str(result_dir)),
                workflow_modules,
            )
            wr.run()

        result = wr.results.get("02_localize", {})
        assert result.get("success"), (
            f"localize did not succeed for {movie_path}"
        )
        n_locs = result.get("n_locs", 0)
        assert n_locs > 0, (
            f"No localisations found in {movie_path}; "
            "check that the file is a real DNA-PAINT acquisition"
        )
        logger.info(f"  → {n_locs} localisations found")


# ---------------------------------------------------------------------------
# Full pipeline with undrift on real data
# ---------------------------------------------------------------------------

def test_full_pipeline_undrift_on_real_data(network_test_data, tmp_path):
    """Run load → identify → localize → undrift_rcc → save on the first
    movie found under PW_TEST_DATA_DIR.

    Requires a movie with enough frames for RCC (>= 3 × segmentation = 1500).
    Skips if no suitable movie is found.
    """
    movies = _find_movies(network_test_data, max_files=1)
    if not movies:
        pytest.skip(f"No movie files found under {network_test_data}")

    movie_path = movies[0]
    logger.info(f"Running full pipeline on {movie_path}")

    workflow_modules = [
        (
            "load_dataset_movie",
            {
                "filename": movie_path,
                "sample_movie": {
                    "filename": "selected_frames.mp4",
                    "n_sample": 40,
                    "max_quantile": 0.9998,
                    "fps": 2,
                },
            },
        ),
        (
            "identify",
            {
                "auto_netgrad": {
                    "filename": "ng_histogram.png",
                    "frame_numbers": (
                        "$get_previous_module_result",  # get from prior results
                        "sample_movie, sample_frame_idx",
                    ),
                    "box_size": 7,
                    "start_ng": -3000,
                    "zscore": 5,
                },
                "ids_vs_frame": {"filename": "ids_vs_frame.png"},
                "box_size": 7,
            },
        ),
        (
            "localize",
            {"fit_method": "lsq", "box_size": 7, "fit_parallel": False},
        ),
        (
            "undrift_rcc",
            {
                "segmentation": 500,
                "max_iter_segmentations": 4,
                "filename": "drift.csv",
                "save_locs": {"filename": "locs_undrift.hdf5"},
            },
        ),
        ("save_single_dataset", {"filename": "locs.hdf5"}),
    ]

    with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock):
        wr = WorkflowRunner.config_from_dicts(
            _dummy_reporter_config("test_rd_full_undrift"),
            _analysis_config(str(tmp_path)),
            workflow_modules,
        )
        wr.run()

    assert wr.results.get("04_save_single_dataset", {}).get("success"), (
        "Full pipeline did not complete successfully"
    )


# ---------------------------------------------------------------------------
# Discovered start_workflow.py tests
# ---------------------------------------------------------------------------

def _import_workflow_script(script_path):
    """Import a start_workflow.py file and return the module object."""
    name = f"start_workflow_{Path(script_path).parent.name}"
    spec = importlib.util.spec_from_file_location(name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_metaworkflow_params(workflow_modules):
    """Return True if any module parameter uses '$$' metaworkflow syntax.

    Parameters prefixed with '$$' (e.g. ('$$map', 'filepath'),
    ('$$get_prior_result', ...)) require InvestigationCoordinator to resolve
    and cannot be run as standalone WorkflowRunner tests.
    """
    def _check(v):
        if isinstance(v, (list, tuple)):
            if len(v) >= 1 and isinstance(v[0], str) and v[0].startswith("$$"):
                return True
            return any(_check(x) for x in v)
        if isinstance(v, dict):
            return any(_check(x) for x in v.values())
        return False

    for _name, params in workflow_modules:
        if isinstance(params, dict) and any(_check(v) for v in params.values()):
            return True
    return False


def test_run_discovered_workflow(workflow_script, tmp_path):
    """Run a start_workflow.py discovered under TestData.directory.

    Each script found by the pytest_generate_tests hook in conftest.py becomes
    a separate parametrized test case, identified by its parent directory name.

    Module extraction order:
      1. Module-level workflow_modules_sgl / workflow_modules
      2. get_workflow(dummy_datasets)["single_dataset_modules"]

    Scripts whose single-dataset modules use '$$' metaworkflow parameters
    ($$map, $$get_prior_result, $$index) are skipped — those workflows require
    InvestigationCoordinator and cannot run as standalone tests.

    Confluence reporting is replaced by MagicMock so no credentials or network
    access are needed.  Results are written to pytest's tmp_path.
    """
    script_path = Path(workflow_script)

    try:
        mod = _import_workflow_script(script_path)
    except Exception as exc:
        pytest.skip(f"Could not import {script_path.name}: {exc}")

    # 1. Try module-level variable
    workflow_modules = getattr(mod, "workflow_modules_sgl", None) or getattr(
        mod, "workflow_modules", None
    )

    # 2. Fall back to get_workflow(dummy)
    if workflow_modules is None:
        result = _try_call_get_workflow(mod)
        if isinstance(result, dict):
            workflow_modules = result.get("single_dataset_modules")
        elif isinstance(result, list):
            workflow_modules = result

    if workflow_modules is None:
        pytest.skip(
            f"{script_path.parent.name}: no workflow_modules_sgl found and "
            "get_workflow() is absent or raised"
        )

    # Skip scripts that need metaworkflow ($$) parameter resolution
    if _has_metaworkflow_params(workflow_modules):
        pytest.skip(
            f"{script_path.parent.name}: single-dataset modules use metaworkflow "
            "parameters ($$map / $$get_prior_result / $$index) — requires "
            "InvestigationCoordinator, not runnable as a standalone test"
        )

    result_dir = tmp_path / script_path.parent.name
    result_dir.mkdir(parents=True, exist_ok=True)

    with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock), patch(
        "picasso_workflow.workflow.ConfluenceInterface", MagicMock
    ):
        wr = WorkflowRunner.config_from_dicts(
            _dummy_reporter_config(script_path.parent.name),
            _analysis_config(str(result_dir)),
            workflow_modules,
        )
        wr.run()

    failed = [
        k for k, v in wr.results.items()
        if isinstance(v, dict) and v.get("success") is False
    ]
    assert not failed, (
        f"{script_path.parent.name}: workflow modules reported failure: {failed}"
    )

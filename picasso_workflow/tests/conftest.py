#!/usr/bin/env python
"""
conftest.py — session-scoped pytest fixtures shared across the test suite.

Synthetic data fixtures
-----------------------
synthetic_movie_5k
    5 000-frame 128×128 OME-TIFF with ~20 persistent Gaussian emitters on a
    Poisson background.  Sufficient for the full identify → localize →
    undrift_rcc pipeline (segmentation=500 → 10 frame segments).

synthetic_locs_10k
    10 000 synthetic localisations in picasso HDF5 format, pre-clustered into
    ~50 groups.  Suitable for clustering, density analysis, neighbour-distance,
    CSR fitting, and Ripley's K tests.

Network data fixture
--------------------
network_test_data
    Path to a directory of real acquired datasets on the pool volumes.  Tests
    that request this fixture are automatically skipped when the path is not
    accessible.  Configure via:
      - Environment variable  PW_TEST_DATA_DIR
      - User config.yaml      TestData → directory

Shared workflow helpers
-----------------------
analysis_config(results_folder) → dict
dummy_reporter_config(report_name) → dict
    Available as plain functions (not fixtures) so that test modules can call
    them directly without going through pytest's fixture injection.
"""
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared workflow helpers (plain functions, not fixtures)
# ---------------------------------------------------------------------------

_CAMERA_INFO = {
    "Gain": 1,
    "Sensitivity": 0.45,
    "Baseline": 100,
    "Qe": 0.82,
    "Pixelsize": 130,  # nm
}


def analysis_config(results_folder):
    """Minimal analysis config dict for WorkflowRunner tests."""
    return {
        "result_location": str(results_folder),
        "camera_info": _CAMERA_INFO,
        "gpufit_installed": False,
    }


def dummy_reporter_config(report_name):
    """reporter_config that satisfies WorkflowRunner without real Confluence.

    WorkflowRunner requires a 'ConfluenceReporter' sub-dict to initialise
    self.confluencereporter.  When ConfluenceReporter is patched with a
    MagicMock the values here are never used for real network calls.
    """
    return {
        "report_name": report_name,
        "ConfluenceReporter": {
            "base_url": "http://mock-confluence",
            "username": "mock-user",
            "space_key": "MOCK",
            "parent_page_title": "mock-parent",
            "token": "mock-token",
        },
    }


# ---------------------------------------------------------------------------
# Synthetic movie fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_movie_5k(tmp_path_factory):
    """Return path to a 5 000-frame 128×128 OME-TIFF.

    Contains ~20 persistent Gaussian emitters (sigma=1.5 px, ~1 000 photons
    each) on a Poisson background (~100 counts).  Emitter positions are fixed
    across all frames (no drift), so undrift_rcc should converge to ~zero
    drift, which is a valid and reproducible test outcome.

    The file is written once per session and shared between all tests that
    request it.
    """
    tifffile = pytest.importorskip(
        "tifffile", reason="tifffile required for synthetic_movie_5k fixture"
    )

    rng = np.random.default_rng(42)
    n_frames, height, width = 5000, 128, 128
    n_emitters = 20
    photons_mean = 1000
    bg_mean = 100
    sigma = 1.5
    on_fraction = 0.85

    # Fixed emitter positions
    ey = rng.uniform(10, height - 10, n_emitters)
    ex = rng.uniform(10, width - 10, n_emitters)

    # Precompute Gaussian kernels (n_emitters, H, W) — float32 to save memory
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    kernels = np.exp(
        -(
            (xx[None] - ex[:, None, None]) ** 2
            + (yy[None] - ey[:, None, None]) ** 2
        )
        / (2 * sigma**2)
    )

    path = tmp_path_factory.mktemp("synthetic") / "movie_5k.ome.tif"

    # Pre-allocate uint16 stack; picasso's io.load_movie requires a
    # contiguous multi-page TIFF written in a single imwrite call so that
    # the stored shape metadata matches the actual data.
    movie = np.zeros((n_frames, height, width), dtype=np.uint16)
    for i in range(n_frames):
        frame = rng.poisson(bg_mean, (height, width)).astype(np.float32)
        active = rng.random(n_emitters) < on_fraction
        if active.any():
            photons = rng.poisson(photons_mean, n_emitters).astype(
                np.float32
            )
            frame += (
                photons[active, None, None] * kernels[active]
            ).sum(axis=0)
        np.clip(frame, 0, 65535, out=frame)
        movie[i] = frame.astype(np.uint16)

    tifffile.imwrite(str(path), movie)
    return path


# ---------------------------------------------------------------------------
# Synthetic localisation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_locs_10k(tmp_path_factory):
    """Return path to an HDF5 file with 10 000 synthetic localisations.

    The locs are pre-clustered into ~50 groups (sigma=0.3 px within each
    cluster) spread over a 128×128 px field.  Suitable for clustering
    (DBSCAN, HDBSCAN), density analysis, neighbour-distance histograms,
    CSR fitting, and Ripley's K tests.

    Requires picassosr to be installed (uses picasso.io.save_locs).
    """
    pytest.importorskip("picasso", reason="picassosr required for synthetic_locs_10k")
    from picasso import io as picasso_io

    rng = np.random.default_rng(42)
    n_locs = 10_000
    n_clusters = 50
    n_frames = 5000

    centres = rng.uniform(5, 123, (n_clusters, 2))
    cluster_idx = rng.integers(0, n_clusters, n_locs)

    locs = np.zeros(
        n_locs,
        dtype=[
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("ellipticity", "f4"),
            ("net_gradient", "f4"),
            ("n_id", "u4"),
        ],
    )
    locs["frame"] = rng.integers(0, n_frames, n_locs, dtype=np.uint32)
    locs["x"] = (centres[cluster_idx, 0] + rng.normal(0, 0.3, n_locs)).astype(
        np.float32
    )
    locs["y"] = (centres[cluster_idx, 1] + rng.normal(0, 0.3, n_locs)).astype(
        np.float32
    )
    locs["photons"] = rng.poisson(1000, n_locs).astype(np.float32)
    locs["sx"] = locs["sy"] = np.float32(1.5)
    locs["lpx"] = locs["lpy"] = np.float32(0.02)
    locs["net_gradient"] = np.float32(3000.0)

    path = tmp_path_factory.mktemp("synthetic") / "locs_10k.hdf5"
    info = [
        {
            "Frames": n_frames,
            "Width": 128,
            "Height": 128,
            "Pixelsize": 130,
            "Generated by": "picasso-workflow test fixture (synthetic_locs_10k)",
        }
    ]
    picasso_io.save_locs(str(path), locs, info)
    return path


# ---------------------------------------------------------------------------
# Network data fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def network_test_data():
    """Return path to a directory of real acquired datasets on the pool volumes.

    Path resolution priority:
      1. Environment variable   PW_TEST_DATA_DIR
      2. User config.yaml       TestData → directory

    Tests that use this fixture are skipped automatically when the path is
    not set or the directory is not mounted.  To run real-data tests on a
    lab machine:

        export PW_TEST_DATA_DIR=/Volumes/pool-miblab1/users/<you>/test-datasets
        pytest -m "integration and real_data"

    Or add to ~/.config/picasso_workflow/config.yaml:

        TestData:
          directory: /Volumes/pool-miblab1/users/<you>/test-datasets
    """
    path = os.getenv("PW_TEST_DATA_DIR")
    if not path:
        try:
            from picasso_workflow import CONFIG
            path = CONFIG.get("TestData", {}).get("directory")
        except Exception:
            pass

    if not path or not os.path.isdir(path):
        pytest.skip(
            "Real acquired test data not available. "
            "Set PW_TEST_DATA_DIR or add TestData.directory to "
            "~/.config/picasso_workflow/config.yaml."
        )
    return path

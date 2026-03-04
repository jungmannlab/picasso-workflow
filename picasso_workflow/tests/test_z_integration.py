#!/usr/bin/env python
"""
Module Name: test_z_integration.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the integration of the package with picasso: run complete
    analysis workflows on minimal datasets.

Two test classes plus one parametrized function:

  Test_A_PicassoIntegration
      Runs the real picasso pipeline against the bundled minimal OME-TIFF
      test datasets. Confluence is replaced by MagicMocks so no network
      access or credentials are required.  Skipped automatically when
      picassosr is not installed.

      Mocking note: WorkflowRunner uses ConfluenceReporter; AggregationWork-
      flowRunner uses ConfluenceInterface directly.  Both are patched where
      needed.

  Test_B_ConfluenceIntegration
      Runs the same workflow with a live ConfluenceReporter.  Skipped
      unless the TEST_CONFLUENCE_* environment variables are all set.

  test_template_smoke (parametrized)
      For each snapshotted template in TestData/templates/ runs the first
      safe modules against the bundled test data.  Silently skipped when no
      templates are present.  Populate via: python tools/snapshot_templates.py
"""
import importlib.util
import logging
import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Skip the entire module when picassosr is not installed.
pytest.importorskip("picasso")

from picasso_workflow.workflow import (  # noqa: E402
    AggregationWorkflowRunner,
    WorkflowRunner,
)
import picasso_workflow.standard_singledataset_workflows as ssw  # noqa: E402
import picasso_workflow.standard_aggregation_workflows as saw  # noqa: E402


logger = logging.getLogger(__name__)

# Every test in this module carries the "integration" mark so it can be
# selected or deselected with  pytest -m integration / -m "not integration".
pytestmark = pytest.mark.integration

# Absolute path to the bundled minimal test datasets.
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "TestData",
    "integration",
)

# Absolute path to the snapshotted production template start_workflow.py files.
_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "TestData",
    "templates",
)

# Modules that need more frames or pre-existing results than the test data
# provides — skip them when constructing a smoke-test workflow from a template.
_SMOKE_SKIP_MODULES = {
    "undrift_rcc",
    "undrift_aim",
    "undrift_rsso",
    "undrift_from_picked",
    "save_single_dataset",
    "save_datasets_aggregated",
}

# Standard bundled test file used as substitute for template data paths.
_SMOKE_TEST_FILE = os.path.join(
    _DATA_DIR,
    "3C_30px_1kframes_1",
    "3C_30px_1kframes_MMStack_Pos0.ome.tif",
)

_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
)

# Names of the environment variables required for live Confluence tests.
_CONFLUENCE_ENV_VARS = (
    "TEST_CONFLUENCE_URL",
    "TEST_CONFLUENCE_TOKEN",
    "TEST_CONFLUENCE_SPACE",
    "TEST_CONFLUENCE_PAGE",
    "TEST_CONFLUENCE_USERNAME",
)
_CONFLUENCE_AVAILABLE = all(os.getenv(v) for v in _CONFLUENCE_ENV_VARS)

# Shared helpers live in conftest.py so that test_real_data_integration.py
# can also use them without importing this module.
from picasso_workflow.tests.conftest import (  # noqa: E402
    analysis_config as _analysis_config,
    dummy_reporter_config as _dummy_reporter_config,
)


class Test_A_PicassoIntegration(unittest.TestCase):
    """Run complete workflows against the minimal test datasets, exercising
    the real picasso pipeline.  Confluence is replaced by a MagicMock so no
    credentials or network access are needed.
    """

    def setUp(self):
        os.makedirs(_RESULTS_DIR, exist_ok=True)

    def tearDown(self):
        try:
            shutil.rmtree(_RESULTS_DIR)
        except FileNotFoundError:
            pass

    def test_01_minimal_singledataset(self):
        """load → identify → localize on the bundled 30 px / 1k-frame stack.

        undrift_rcc and save_single_dataset are omitted: RCC needs more
        frames than the test dataset reliably provides.
        """
        workflow_modules = ssw.minimal(
            filepath=os.path.join(
                _DATA_DIR,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            )
        )
        # Drop undrift_rcc (index 3) and save_single_dataset (index 4).
        workflow_modules = workflow_modules[:-2]

        with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock):
            wr = WorkflowRunner.config_from_dicts(
                _dummy_reporter_config("test_a01_minimal_singledataset"),
                _analysis_config(_RESULTS_DIR),
                workflow_modules,
            )
            wr.run()

    def test_02_minimal_channel_align(self):
        """load → identify → localize → save → align on the two bundled stacks.

        undrift_rcc is omitted for the same reason as test_01; the filepaths
        reference in the aggregation step is updated to match the renumbered
        single-dataset modules.
        """
        filepaths = [
            os.path.join(
                _DATA_DIR,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            ),
            os.path.join(
                _DATA_DIR,
                "3C_30px_1kframes_shifted_1",
                "3C_30px_1kframes_shifted_MMStack_Pos0.ome.tif",
            ),
        ]
        agg_workflow = saw.minimal_channel_align(filepaths=filepaths)

        # Remove undrift_rcc (index 3) from single-dataset modules.
        # save_single_dataset moves from index 4 → index 3.
        agg_workflow["single_dataset_modules"] = [
            agg_workflow["single_dataset_modules"][i] for i in [0, 1, 2, 4]
        ]

        # Build the correct $$get_prior_result locator.  The standard workflow
        # generator writes "$all" (single $), but the aggregation
        # ParameterCommandExecutor uses command_sign="$$", so "$$all" is
        # required for the wildcard to be recognised.  After removing
        # undrift_rcc the save step is now module 3, not 4.
        agg_workflow["aggregation_modules"][0][1]["filepaths"] = (
            "$$get_prior_result",
            "all_results, single_dataset, $$all, "
            "03_save_single_dataset, filepath",
        )

        with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock), \
                patch("picasso_workflow.workflow.ConfluenceInterface", MagicMock):
            awr = AggregationWorkflowRunner.config_from_dicts(
                _dummy_reporter_config("test_a02_minimal_channel_align"),
                _analysis_config(_RESULTS_DIR),
                agg_workflow,
            )
            awr.run()


@pytest.mark.integration
def test_03_undrift_rcc(synthetic_movie_5k, tmp_path):
    """Full pipeline — load → identify → localize → undrift_rcc → save —
    on a 5 000-frame synthetic stack.

    Uses a fixed net_gradient threshold (300 ADU) to avoid the auto_netgrad
    heuristic, which is tuned for real photon-count data.  The synthetic
    emitters produce consistent localisations across all 5 000 frames, so
    RCC converges to ~zero drift, confirming the algorithm runs end-to-end.
    """
    workflow_modules = [
        (
            "load_dataset_movie",
            {"filename": str(synthetic_movie_5k)},
        ),
        (
            "identify",
            {
                "min_gradient": 300,
                "box_size": 7,
                "ids_vs_frame": {"filename": "ids_vs_frame.png"},
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
            _dummy_reporter_config("test_03_undrift_rcc"),
            _analysis_config(str(tmp_path)),
            workflow_modules,
        )
        wr.run()


@pytest.mark.skipif(
    not _CONFLUENCE_AVAILABLE,
    reason=(
        "Confluence env vars not set "
        f"({', '.join(_CONFLUENCE_ENV_VARS)})"
    ),
)
class Test_B_ConfluenceIntegration(unittest.TestCase):
    """Run a minimal workflow with a live ConfluenceReporter.

    Requires all TEST_CONFLUENCE_* environment variables to be set.
    """

    def setUp(self):
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        self._wr = None
        self._reporter_config_base = {
            "report_name": "",
            "ConfluenceReporter": {
                "base_url": os.getenv("TEST_CONFLUENCE_URL"),
                "username": os.getenv("TEST_CONFLUENCE_USERNAME"),
                "space_key": os.getenv("TEST_CONFLUENCE_SPACE"),
                "parent_page_title": os.getenv("TEST_CONFLUENCE_PAGE"),
                "token": os.getenv("TEST_CONFLUENCE_TOKEN"),
            },
        }

    def tearDown(self):
        try:
            shutil.rmtree(_RESULTS_DIR)
        except FileNotFoundError:
            pass
        if self._wr is not None:
            try:
                self._wr.confluencereporter.ci.delete_page(
                    self._wr.reporter_config["report_name"]
                )
            except Exception:
                pass

    def test_01_minimal_singledataset_with_confluence(self):
        reporter_config = {
            **self._reporter_config_base,
            "report_name": "test_b01_minimal_singledataset",
        }
        workflow_modules = ssw.minimal(
            filepath=os.path.join(
                _DATA_DIR,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            )
        )
        workflow_modules = workflow_modules[:-2]

        self._wr = WorkflowRunner.config_from_dicts(
            reporter_config,
            _analysis_config(_RESULTS_DIR),
            workflow_modules,
        )
        self._wr.run()


# ---------------------------------------------------------------------------
# Level B helpers — template smoke tests
# ---------------------------------------------------------------------------

def _import_template(path):
    """Dynamically import a start_workflow.py and return the module object."""
    spec = importlib.util.spec_from_file_location("_start_workflow", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _smoke_workflow_from_template(path):
    """Load a template's single-dataset workflow and return the first 3 modules
    that are safe to run against the tiny bundled test dataset.

    Returns None when the template is not suitable for a smoke test (e.g. it
    starts with load_dataset_localizations rather than load_dataset_movie, or
    its workflow_modules_sgl is not a module-level list).
    """
    try:
        mod = _import_template(path)
    except Exception:
        return None

    sgl = getattr(mod, "workflow_modules_sgl", None)
    if not isinstance(sgl, list) or not sgl:
        return None

    # Only smoke-test templates whose first step loads a raw movie.
    first_name = sgl[0][0] if isinstance(sgl[0], (tuple, list)) else None
    if first_name != "load_dataset_movie":
        return None

    # Collect up to 3 "safe" modules, replacing the filename in the first.
    smoke_modules = []
    for module_name, params in sgl:
        if module_name in _SMOKE_SKIP_MODULES:
            continue
        if module_name == "load_dataset_movie":
            params = dict(params)
            params["filename"] = _SMOKE_TEST_FILE
            # Drop optional sub-keys that write files we don't need.
            params.pop("sample_movie", None)
        smoke_modules.append((module_name, params))
        if len(smoke_modules) == 3:
            break

    return smoke_modules if smoke_modules else None


def _smoke_template_cases():
    """Return (name, smoke_modules) pairs for all suitable snapshotted templates."""
    if not os.path.isdir(_TEMPLATES_DIR):
        return []
    cases = []
    for name in sorted(os.listdir(_TEMPLATES_DIR)):
        path = os.path.join(_TEMPLATES_DIR, name, "start_workflow.py")
        if not os.path.isfile(path):
            continue
        modules = _smoke_workflow_from_template(path)
        if modules is not None:
            cases.append((name, modules))
    return cases


@pytest.mark.parametrize(
    "template_name,smoke_modules",
    _smoke_template_cases(),
)
@pytest.mark.integration
def test_template_smoke(template_name, smoke_modules, tmp_path):
    """Run the first safe modules from a snapshotted template against the
    bundled test data, exercising the real picasso pipeline.

    Catches regressions where a code change breaks a production workflow
    without breaking the unit tests.
    """
    # Patch both ConfluenceReporter (used by WorkflowRunner) and
    # ConfluenceInterface (used directly by AggregationWorkflowRunner).
    with patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock), \
            patch("picasso_workflow.workflow.ConfluenceInterface", MagicMock):
        wr = WorkflowRunner.config_from_dicts(
            _dummy_reporter_config(f"test_c_template_{template_name}"),
            _analysis_config(str(tmp_path)),
            smoke_modules,
        )
        wr.run()

#!/usr/bin/env python
"""
Module Name: test_wp6_preprocess.py
Author: Heinrich Grabmayr
Initial Date: October 21, 2026
Description: WP-6 -- the preprocess recipe, its live-locs reuse seam, and the
    best-effort per-module registry logging wired into WorkflowRunner.

    Hermetic: no Qt, no network, no GPU. AutoPicasso / reporters are mocked,
    and the registry is exercised through the in-memory
    ``picasso_registry.testing.mock_registry`` (a FastAPI TestClient over a
    fresh in-memory SQLite db) or a tiny local double -- never a real service.
"""

import os
import shutil
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from picasso_workflow import standard_singledataset_workflows as ssw
from picasso_workflow.live_locs import (
    LiveLocsDescriptor,
    frame_range_checksum,
    reuse_live_locs,
)
from picasso_workflow.registry_client import (
    RegistryLogger,
    registry_logger_from_config,
)
from picasso_workflow.workflow import (
    WorkflowRunner,
    _module_artifacts,
    _module_metrics,
)

# The one test using picasso-registry's server-backed mock_registry (FastAPI
# TestClient) skips when picasso-registry is absent (it's not a dep —
# picasso-workflow stays wheel-only/hermetic), so it runs in the full dev
# container but not on the hosted CI runner. Everything else here uses the
# local in-process doubles and always runs.
try:
    from picasso_registry.testing import mock_registry as _mock_registry

    _ = _mock_registry
    HAS_REGISTRY = True
except Exception:
    HAS_REGISTRY = False

requires_registry = unittest.skipUnless(
    HAS_REGISTRY, "picasso-registry not installed (hosted wheel-only CI)"
)


def _movie(n_frames=5, h=4, w=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**16, size=(n_frames, h, w), dtype=np.uint16)


# ---------------------------------------------------------------------------
# The preprocess recipe: validates, has no undrift, reuses live locs.
# ---------------------------------------------------------------------------
class TestPreprocessRecipe(unittest.TestCase):
    def test_default_recipe_relocalizes_and_validates(self):
        wm = ssw.preprocess("movie.tif")
        names = [m for m, _ in wm]
        self.assertEqual(
            names,
            [
                "load_dataset_movie",
                "identify",
                "localize",
                "render",
                "save_single_dataset",
            ],
        )
        self.assertEqual([], ssw.validate_preprocess(wm))

    def test_recipe_has_no_undrift(self):
        # The cluster does drift correction; the recipe must not.
        wm = ssw.preprocess("movie.tif")
        names = {m for m, _ in wm}
        self.assertNotIn("undrift_rcc", names)
        self.assertNotIn("undrift_aim", names)
        self.assertFalse(any(n.startswith("undrift") for n in names))

    def test_reuse_valid_lossless_live_locs_skips_relocalization(self):
        movie = _movie()
        cs = frame_range_checksum(movie, 0, len(movie))
        desc = {
            "locs_path": "live.hdf5",
            "frame_start": 0,
            "frame_stop": len(movie),
            "checksum": cs,
            "lossless": True,
        }
        wm = ssw.preprocess("movie.tif", live_locs=desc, movie=movie)
        names = [m for m, _ in wm]
        # identify + localize replaced by a single load of the live locs.
        self.assertEqual(
            names,
            ["load_dataset_localizations", "render", "save_single_dataset"],
        )
        self.assertNotIn("localize", names)
        self.assertEqual(wm[0][1]["filename"], "live.hdf5")
        self.assertEqual([], ssw.validate_preprocess(wm))

    def test_invalid_checksum_falls_back_to_relocalization(self):
        movie = _movie()
        desc = {
            "locs_path": "live.hdf5",
            "frame_start": 0,
            "frame_stop": len(movie),
            "checksum": "deadbeef",  # wrong
            "lossless": True,
        }
        wm = ssw.preprocess("movie.tif", live_locs=desc, movie=movie)
        self.assertIn("localize", [m for m, _ in wm])

    def test_non_lossless_live_locs_relocalizes(self):
        movie = _movie()
        cs = frame_range_checksum(movie, 0, len(movie))
        desc = {
            "locs_path": "live.hdf5",
            "frame_start": 0,
            "frame_stop": len(movie),
            "checksum": cs,
            "lossless": False,  # lossy live pass
        }
        wm = ssw.preprocess("movie.tif", live_locs=desc, movie=movie)
        self.assertIn("localize", [m for m, _ in wm])

    def test_partial_frame_coverage_relocalizes(self):
        movie = _movie()
        cs = frame_range_checksum(movie, 0, len(movie) - 1)
        desc = {
            "locs_path": "live.hdf5",
            "frame_start": 0,
            "frame_stop": len(movie) - 1,  # does not cover the whole movie
            "checksum": cs,
            "lossless": True,
        }
        wm = ssw.preprocess("movie.tif", live_locs=desc, movie=movie)
        self.assertIn("localize", [m for m, _ in wm])

    def test_absent_live_locs_relocalizes(self):
        wm = ssw.preprocess("movie.tif", live_locs=None, movie=_movie())
        self.assertIn("localize", [m for m, _ in wm])

    def test_live_locs_without_movie_cannot_be_validated(self):
        # No movie to checksum against -> safe default is to re-localize.
        desc = {
            "locs_path": "live.hdf5",
            "frame_start": 0,
            "frame_stop": 5,
            "checksum": "x",
            "lossless": True,
        }
        wm = ssw.preprocess("movie.tif", live_locs=desc, movie=None)
        self.assertIn("localize", [m for m, _ in wm])


# ---------------------------------------------------------------------------
# The live-locs reuse decision + checksum, in isolation.
# ---------------------------------------------------------------------------
class TestLiveLocsReuse(unittest.TestCase):
    def test_checksum_is_stable_and_range_sensitive(self):
        movie = _movie(seed=7)
        a = frame_range_checksum(movie, 0, 5)
        b = frame_range_checksum(movie, 0, 5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, frame_range_checksum(movie, 0, 4))

    def test_checksum_detects_changed_frames(self):
        m1 = _movie(seed=1)
        m2 = m1.copy()
        m2[2, 0, 0] ^= 1  # flip one bit in one frame
        self.assertNotEqual(
            frame_range_checksum(m1, 0, 5), frame_range_checksum(m2, 0, 5)
        )

    def test_reuse_returns_descriptor_when_valid(self):
        movie = _movie(seed=3)
        desc = LiveLocsDescriptor(
            locs_path="live.hdf5",
            frame_start=0,
            frame_stop=len(movie),
            checksum=frame_range_checksum(movie, 0, len(movie)),
            lossless=True,
        )
        out = reuse_live_locs(movie, desc)
        self.assertIsInstance(out, LiveLocsDescriptor)
        self.assertEqual(out.locs_path, "live.hdf5")

    def test_from_mapping_rejects_malformed(self):
        self.assertIsNone(LiveLocsDescriptor.from_mapping(None))
        self.assertIsNone(LiveLocsDescriptor.from_mapping("nope"))
        self.assertIsNone(LiveLocsDescriptor.from_mapping({"locs_path": "x"}))


# ---------------------------------------------------------------------------
# RegistryLogger: best-effort / non-blocking / injectable / keyed by run_id.
# ---------------------------------------------------------------------------
class _RecordingDouble:
    """A lightweight in-memory registry double (no FastAPI needed).

    Records every log call; mirrors the subset of the
    ``picasso_registry`` client surface the wrapper uses.
    """

    def __init__(self):
        self.metrics = []
        self.artifacts = []

    def log_metrics(self, **fields):
        self.metrics.append(fields)
        return {"id": f"m{len(self.metrics)}", **fields}

    def create(self, resource, **fields):
        if resource == "artifact":
            self.artifacts.append(fields)
        return {
            "id": f"a{len(self.artifacts)}",
            "resource": resource,
            **fields,
        }


class _BoomDouble:
    """A double whose every call raises -- stands in for a registry outage."""

    def log_metrics(self, **fields):
        raise RuntimeError("registry down")

    def create(self, resource, **fields):
        raise RuntimeError("registry down")


class TestRegistryLogger(unittest.TestCase):
    def test_disabled_without_run_id(self):
        rl = RegistryLogger(None, client=_RecordingDouble())
        self.assertFalse(rl.enabled)
        self.assertIsNone(rl.log_module_metrics("localize", 0, {"n_locs": 1}))

    def test_disabled_without_client_or_factory(self):
        rl = RegistryLogger("run1")
        self.assertFalse(rl.enabled)
        self.assertIsNone(rl.log_module_metrics("localize", 0, {}))

    def test_logs_metrics_keyed_by_run_id(self):
        dbl = _RecordingDouble()
        rl = RegistryLogger("run7", client=dbl)
        rl.log_module_metrics("localize", 2, {"n_locs": 99})
        self.assertEqual(len(dbl.metrics), 1)
        rec = dbl.metrics[0]
        self.assertEqual(rec["analysis_run_id"], "run7")
        self.assertEqual(rec["module"], "localize")
        self.assertEqual(rec["module_index"], 2)
        self.assertEqual(rec["n_locs"], 99)

    def test_logs_artifact_keyed_by_run_id(self):
        dbl = _RecordingDouble()
        rl = RegistryLogger("run7", client=dbl)
        rl.log_artifact("save_single_dataset", "locs_hdf5", "/p/locs.hdf5")
        self.assertEqual(len(dbl.artifacts), 1)
        art = dbl.artifacts[0]
        self.assertEqual(art["analysis_run_id"], "run7")
        self.assertEqual(art["kind"], "locs_hdf5")
        self.assertEqual(art["uri"], "/p/locs.hdf5")

    def test_outage_is_swallowed_never_raises(self):
        rl = RegistryLogger("run7", client=_BoomDouble())
        # Must return None, not raise -- a registry outage cannot break a run.
        self.assertIsNone(rl.log_module_metrics("m", 0, {"n_locs": 1}))
        self.assertIsNone(rl.log_artifact("m", "locs_hdf5", "/x"))

    def test_factory_error_disables_logging(self):
        def boom_factory():
            raise RuntimeError("cannot build client")

        rl = RegistryLogger("run7", client_factory=boom_factory)
        self.assertTrue(rl.enabled)  # enabled until the factory is tried
        self.assertIsNone(rl.log_module_metrics("m", 0, {}))
        self.assertFalse(rl.enabled)  # flipped off after the factory failed

    def test_from_config_empty_is_disabled(self):
        self.assertFalse(registry_logger_from_config(None).enabled)
        self.assertFalse(registry_logger_from_config({}).enabled)
        self.assertFalse(registry_logger_from_config({"registry": {}}).enabled)

    def test_from_config_with_injected_client(self):
        dbl = _RecordingDouble()
        rl = registry_logger_from_config(
            {"registry": {"run_id": "r1", "client": dbl}}
        )
        self.assertTrue(rl.enabled)
        rl.log_module_metrics("localize", 0, {"n_locs": 5})
        self.assertEqual(dbl.metrics[0]["analysis_run_id"], "r1")


# ---------------------------------------------------------------------------
# RegistryLogger against the real in-memory mock_registry (FastAPI TestClient).
# ---------------------------------------------------------------------------
class TestRegistryLoggerAgainstMockRegistry(unittest.TestCase):
    @requires_registry
    def test_metrics_land_in_the_registry_keyed_by_run_id(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.log_analysis(id="runZ", kind="preprocess", status="running")
            rl = RegistryLogger("runZ", client=reg)
            rl.log_module_metrics("localize", 2, {"n_locs": 1234})
            rows = reg.list("metrics", analysis_run_id="runZ")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["analysis_run_id"], "runZ")
            self.assertEqual(rows[0]["n_locs"], 1234)


# ---------------------------------------------------------------------------
# Metric / artifact extraction from a module's results dict.
# ---------------------------------------------------------------------------
class TestResultExtraction(unittest.TestCase):
    def test_metrics_extracted_and_renamed(self):
        res = {"nlocs": 500, "duration": 12.5, "success": True}
        self.assertEqual(
            _module_metrics(res), {"n_locs": 500, "compute_seconds": 12.5}
        )

    def test_non_numeric_and_bool_skipped(self):
        res = {"nlocs": "lots", "success": True, "duration": True}
        self.assertEqual(_module_metrics(res), {})

    def test_artifacts_extracted(self):
        res = {"filepath": "/p/locs.hdf5"}
        self.assertEqual(
            _module_artifacts(res), [("locs_hdf5", "/p/locs.hdf5")]
        )

    def test_no_artifacts_when_absent(self):
        self.assertEqual(_module_artifacts({"nlocs": 1}), [])


# ---------------------------------------------------------------------------
# End-to-end: running modules through WorkflowRunner logs exactly one metrics
# record per module to an injected registry, each carrying the run_id.
# ---------------------------------------------------------------------------
class TestWorkflowRunnerRegistryWiring(unittest.TestCase):
    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def _runner(self, registry_cfg):
        reporter_config = {
            "report_name": "wp6report",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        if registry_cfg is not None:
            analysis_config["registry"] = registry_cfg
        return WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, []
        )

    def test_one_metrics_record_per_module_with_run_id(self):
        dbl = _RecordingDouble()
        wr = self._runner({"run_id": "runABC", "client": dbl})
        try:
            # Three modules, each returning a results dict (as a real module
            # would via module_decorator).
            modules = {
                "load_dataset_movie": {"success": True},
                "localize": {"success": True, "nlocs": 321},
                "save_single_dataset": {
                    "success": True,
                    "filepath": "/p/locs.hdf5",
                },
            }
            for idx, (name, result) in enumerate(modules.items()):
                setattr(
                    wr.autopicasso,
                    name,
                    (lambda r: (lambda i, p: (p, dict(r))))(result),
                )
                wr.call_module(name, idx, {})

            # Exactly one metrics record per module, all keyed by run_id.
            self.assertEqual(len(dbl.metrics), len(modules))
            self.assertEqual(
                [m["analysis_run_id"] for m in dbl.metrics],
                ["runABC"] * len(modules),
            )
            self.assertEqual([m["module"] for m in dbl.metrics], list(modules))
            # localize carried its n_locs; save_single_dataset logged the
            # locs artifact pointer.
            loc = next(m for m in dbl.metrics if m["module"] == "localize")
            self.assertEqual(loc["n_locs"], 321)
            self.assertEqual(len(dbl.artifacts), 1)
            self.assertEqual(dbl.artifacts[0]["kind"], "locs_hdf5")
        finally:
            shutil.rmtree(wr.result_folder, ignore_errors=True)

    def test_scope_defaults_single_and_follows_registry_scope(self):
        # Metrics are tagged "single" by default; the aggregation runner sets
        # registry_scope="aggregation" on its sub-runner, so aggregation
        # metrics are not mislabeled as single-scope.
        dbl = _RecordingDouble()
        wr = self._runner({"run_id": "runS", "client": dbl})
        try:
            wr.autopicasso.m = lambda i, p: (p, {"success": True})
            wr.call_module("m", 0, {})
            self.assertEqual(dbl.metrics[-1]["scope"], "single")
            wr.registry_scope = "aggregation"
            wr.call_module("m", 1, {})
            self.assertEqual(dbl.metrics[-1]["scope"], "aggregation")
        finally:
            shutil.rmtree(wr.result_folder, ignore_errors=True)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_no_registry_config_is_a_no_op(self):
        # Without a registry config, call_module must run exactly as before.
        reporter_config = {
            "report_name": "wp6report2",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, []
        )
        try:
            wr.autopicasso.m = lambda i, p: (p, {"success": True})
            # Should neither raise nor require any registry.
            self.assertTrue(wr.call_module("m", 0, {}))
            self.assertFalse(wr._registry_logger().enabled)
        finally:
            shutil.rmtree(wr.result_folder, ignore_errors=True)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_registry_outage_does_not_break_a_module(self):
        wr = self._runner({"run_id": "runX", "client": _BoomDouble()})
        try:
            wr.autopicasso.m = lambda i, p: (p, {"success": True, "nlocs": 7})
            # The outage must be swallowed: the module still succeeds.
            self.assertTrue(wr.call_module("m", 0, {}))
        finally:
            shutil.rmtree(wr.result_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

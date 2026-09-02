#!/usr/bin/env python
"""
Module Name: test_progress.py
Author: Heinrich Grabmayr
Initial Date: September 1, 2026
Description: Unit tests for picasso_workflow.progress.

These tests exercise the progress emitter, its JSON sink, the consumer-side
helpers and the picasso progress proxy. They need neither picasso nor any
data, so they run in the fast unit tier.
"""

import json
import os
import shutil
import tempfile
import unittest

from picasso_workflow import progress as p


class TestProgressManager(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_start_initialises_modules(self):
        pm = p.ProgressManager(self.folder, sinks=[])
        pm.start(["load", "identify", "localize"])
        self.assertEqual(pm.state["state"], p.RUNNING)
        self.assertEqual(pm.state["total"], 3)
        self.assertEqual(len(pm.state["modules"]), 3)
        self.assertTrue(
            all(m["status"] == p.PENDING for m in pm.state["modules"])
        )

    def test_module_lifecycle_and_timing(self):
        pm = p.ProgressManager(self.folder, sinks=[])
        pm.start(["load", "identify"])
        pm.module_start(1)
        self.assertEqual(pm.state["modules"][1]["status"], p.RUNNING)
        self.assertEqual(pm.state["current"], 1)
        pm.module_end(1, p.DONE)
        m = pm.state["modules"][1]
        self.assertEqual(m["status"], p.DONE)
        self.assertEqual(m["fraction"], 1.0)
        self.assertIsInstance(m["elapsed"], float)

    def test_module_skipped(self):
        pm = p.ProgressManager(self.folder, sinks=[])
        pm.start(["a", "b"])
        pm.module_skipped(0)
        self.assertEqual(pm.state["modules"][0]["status"], p.SKIPPED)
        self.assertEqual(pm.state["modules"][0]["fraction"], 1.0)

    def test_progress_clamped(self):
        pm = p.ProgressManager(self.folder, sinks=[])
        pm.start(["a"])
        pm.module_start(0)
        pm.module_progress(0, 5.0)  # out of range -> clamp to 1.0
        self.assertEqual(pm.state["modules"][0]["fraction"], 1.0)
        pm.module_progress(0, -1.0)
        self.assertEqual(pm.state["modules"][0]["fraction"], 0.0)

    def test_progress_throttled_emits(self):
        # message changes always emit; fraction-only within throttle do not
        emitted = []
        pm = p.ProgressManager(
            self.folder,
            sinks=[lambda s: emitted.append(s["updated"])],
            throttle=1000.0,
        )
        pm.start(["a"])
        emitted.clear()
        pm.module_start(0)
        n_after_start = len(emitted)
        pm.module_progress(0, 0.1)  # throttled, no message change -> no emit
        self.assertEqual(len(emitted), n_after_start)
        pm.module_progress(0, 0.2, msg="frame 2")  # message change -> emit
        self.assertEqual(len(emitted), n_after_start + 1)

    def test_finish(self):
        pm = p.ProgressManager(self.folder, sinks=[])
        pm.start(["a"])
        pm.finish(p.FAILED)
        self.assertEqual(pm.state["state"], p.FAILED)

    def test_sink_failure_is_non_fatal(self):
        def boom(state):
            raise RuntimeError("sink is broken")

        pm = p.ProgressManager(self.folder, sinks=[boom])
        # must not raise
        pm.start(["a"])
        pm.module_start(0)
        pm.module_end(0, p.DONE)


class TestJsonSink(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_json_written_and_readable(self):
        pm = p.ProgressManager(self.folder)  # default sinks (json + log)
        pm.start(["load", "identify"])
        pm.module_start(0)
        pm.module_end(0, p.DONE)
        path = os.path.join(self.folder, p.PROGRESS_FILENAME)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["total"], 2)
        # and read_progress returns the same
        self.assertEqual(p.read_progress(self.folder)["total"], 2)

    def test_worker_rank_uses_separate_file(self):
        pm = p.ProgressManager(self.folder, rank=2, size=4)
        pm.start(["a"])
        self.assertFalse(
            os.path.exists(os.path.join(self.folder, p.PROGRESS_FILENAME))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.folder, "progress.rank2.json"))
        )

    def test_read_latest_progress_walks_tree(self):
        sub = os.path.join(self.folder, "run_240101-1200")
        os.makedirs(sub)
        pm = p.ProgressManager(sub)
        pm.start(["a", "b"])
        found = p.read_latest_progress(self.folder)
        self.assertIsNotNone(found)
        self.assertEqual(found["total"], 2)

    def test_read_all_progress_collects_stages(self):
        # simulate an aggregation run: top-level + two singles + agg stage
        top = os.path.join(self.folder, "myrun_240101-1200")
        os.makedirs(top)
        p.ProgressManager(top, kind="aggregation").datasets_init(["c1", "c2"])
        for sub in (
            "myrun_sgl_00_c1_240101-1200",
            "myrun_sgl_01_c2_240101-1200",
            "myrun_aggregation_240101-1200",
        ):
            d = os.path.join(top, sub)
            os.makedirs(d)
            p.ProgressManager(d, report_name=sub).start(["a"])
        states = p.read_all_progress(self.folder)
        self.assertEqual(len(states), 4)
        kinds = [s["kind"] for s in states]
        self.assertEqual(kinds.count("aggregation"), 1)
        self.assertEqual(kinds.count("single"), 3)


class TestStageHelpers(unittest.TestCase):
    def test_stage_name_strips_postfix(self):
        self.assertEqual(
            p.stage_name({"report_name": "myrun_sgl_02_cell_240101-1200"}),
            "myrun_sgl_02_cell",
        )
        self.assertEqual(p.stage_name({}), "(run)")

    def test_is_aggregation_stage(self):
        self.assertTrue(
            p.is_aggregation_stage(
                {"report_name": "myrun_aggregation_240101-1200"}
            )
        )
        self.assertFalse(
            p.is_aggregation_stage(
                {"report_name": "myrun_sgl_00_x_240101-1200"}
            )
        )

    def test_dataset_index(self):
        self.assertEqual(
            p.dataset_index({"report_name": "myrun_sgl_07_x_240101-1200"}), 7
        )
        self.assertIsNone(
            p.dataset_index({"report_name": "myrun_aggregation_240101-1200"})
        )


class TestOverallFraction(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(p.overall_fraction(None), 0.0)
        self.assertEqual(p.overall_fraction({}), 0.0)

    def test_modules(self):
        state = {
            "modules": [
                {"status": p.DONE},
                {"status": p.SKIPPED},
                {"status": p.RUNNING, "fraction": 0.5},
                {"status": p.PENDING},
            ]
        }
        self.assertAlmostEqual(p.overall_fraction(state), 2.5 / 4)

    def test_datasets(self):
        state = {
            "modules": [],
            "datasets": [
                {"state": p.DONE},
                {"state": p.PENDING},
            ],
        }
        self.assertAlmostEqual(p.overall_fraction(state), 0.5)


class TestPicassoProgressProxy(unittest.TestCase):
    def test_callable_and_dialog_conventions(self):
        seen = []
        proxy = p.PicassoProgressProxy(
            lambda frac, msg=None: seen.append((frac, msg)),
            total=100,
            phase="identify",
        )
        proxy(50)  # plain-callable convention
        proxy.set_value(75)  # dialog convention
        self.assertEqual(seen[0], (0.5, "identify"))
        self.assertEqual(seen[1], (0.75, "identify"))

    def test_setmaximum_updates_scale(self):
        seen = []
        proxy = p.PicassoProgressProxy(
            lambda frac, msg=None: seen.append(frac), total=None
        )
        proxy(10)  # no max -> fraction None
        self.assertIsNone(seen[-1])
        proxy.setMaximum(20)
        proxy(10)
        self.assertEqual(seen[-1], 0.5)

    def test_list_argument(self):
        seen = []
        proxy = p.PicassoProgressProxy(
            lambda frac, msg=None: seen.append(frac), total=10
        )
        proxy([3])  # identify passes a list-like current
        self.assertEqual(seen[-1], 0.3)

    def test_callback_errors_are_swallowed(self):
        def boom(frac, msg=None):
            raise ValueError("callback broke")

        proxy = p.PicassoProgressProxy(boom, total=10)
        proxy(5)  # must not raise
        proxy.set_value(5)

    def test_dialog_noops_do_not_raise(self):
        proxy = p.PicassoProgressProxy(lambda f, m=None: None, total=10)
        proxy.zero_progress("phase")
        proxy.init()
        proxy.update()
        proxy.close()
        proxy.closeEvent()
        proxy.setLabelText("x")
        proxy.play_sound_notification()
        self.assertEqual(list(proxy.get_iterator(0, 3)), [0, 1, 2])
        self.assertEqual(proxy.maximum(), 10)


class TestAbortFlag(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_request_and_clear(self):
        self.assertFalse(p.abort_requested(self.folder))
        p.request_abort(self.folder)
        self.assertTrue(p.abort_requested(self.folder))
        p.clear_abort(self.folder)
        self.assertFalse(p.abort_requested(self.folder))

    def test_clear_missing_is_noop(self):
        p.clear_abort(self.folder)  # must not raise


if __name__ == "__main__":
    unittest.main()

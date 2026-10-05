#!/usr/bin/env python
"""
Module Name: test_workflow.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the module workflow.py
    Mock as many intra-package dependencies as possible,
    this is only about the module itself. For the interaction
    of the different modules, see test_integration.py
"""

import os
import shutil
import logging
import unittest
from unittest.mock import patch, MagicMock

import yaml

from picasso_workflow.analyse import AutoPicassoError
from picasso_workflow.workflow import (
    WorkflowRunner,
    AggregationWorkflowRunner,
    _checkpoint_from_module_results,
    _find_previous_runner_postfix,
    _module_parameters_changed,
    _module_parameters_changed_legacy,
    _strip_runstamp,
)

logger = logging.getLogger(__name__)


# @unittest.skip("")
class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )

    def tearDown(self):
        pass

    @patch("picasso_workflow.workflow.ParameterCommandExecutor")
    def test_a01_WorkflowRunner_init(self, mock_pce):
        wr = WorkflowRunner()
        assert wr.results == {}

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a02_WorkflowRunner_from_config(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        assert wr.results == {}
        logger.debug(wr.autopicasso)
        logger.debug(wr.confluencereporter)

        # created a folder upon initialization. remove it.
        shutil.rmtree(wr.result_folder)

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a03_WorkflowRunner_save_load(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        wr.save(self.results_folder)

        wr2 = WorkflowRunner.load(self.results_folder)

        # clean up
        # shutil.rmtree(wr.result_folder)
        shutil.rmtree(wr2.result_folder)
        os.remove(os.path.join(self.results_folder, "WorkflowRunner.yaml"))

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a04_WorkflowRunner_call_module(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        wr.autopicasso.my_module = lambda i, p: ({}, {"success": True})

        wr.call_module("my_module", 0, {"parameter0": 1})

        shutil.rmtree(wr.result_folder)

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a04b_call_module_surfaces_analyse_error(self):
        """When the analysis step raises, the original exception must
        propagate from call_module() -- not a KeyError from looking up
        self.results[key] when reporting the (non-existent) success
        result to Confluence. Regression test for workflow.py:791.
        """
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )

        boom = RuntimeError("kaboom")

        def failing_module(i, parameters):
            raise boom

        wr.autopicasso.my_module = failing_module
        # Replace the per-module success reporter with a strict mock so
        # we can assert it is NOT invoked when the analysis step failed.
        wr.confluencereporter.my_module = MagicMock()

        with self.assertRaises(RuntimeError) as cm:
            wr.call_module("my_module", 0, {"parameter0": 1})
        assert "kaboom" in str(cm.exception)

        # Confluence error path was used; success-path reporter was not.
        wr.confluencereporter.report_error.assert_called_once()
        wr.confluencereporter.my_module.assert_not_called()

        shutil.rmtree(wr.result_folder)

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a05_WorkflowRunner_run(self, mock_call_module):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = [("load_dataset_movie", {"b": 3})]

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )

        wr.run()

        shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a05b_run_writes_progress_json(self, mock_call_module):
        """A successful run emits a progress.json marking every module done."""
        from picasso_workflow import progress as pwprogress

        mock_call_module.return_value = True
        reporter_config = {"report_name": "progressreport"}
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = [
            ("load_dataset_movie", {"b": 3}),
            ("identify", {"min_gradient": 1}),
        ]
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        wr.run()

        state = pwprogress.read_progress(wr.result_folder)
        self.assertIsNotNone(state)
        self.assertEqual(state["state"], "done")
        self.assertEqual(state["total"], 2)
        self.assertTrue(all(m["status"] == "done" for m in state["modules"]))
        self.assertEqual(pwprogress.overall_fraction(state), 1.0)
        shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a05d_run_accepts_picasso_set_modules(self, mock_call_module):
        """picasso-set modules pass the runner's available-modules check."""
        mock_call_module.return_value = True
        reporter_config = {"report_name": "picassosetreport"}
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = [
            ("load_dataset_localizations", {"filename": "a.hdf5"}),
            ("picasso_density", {"radius": 1.5}),
        ]
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        wr.run()

        self.assertEqual(2, mock_call_module.call_count)
        self.assertEqual(
            "picasso_density", mock_call_module.call_args_list[1][0][0]
        )
        shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_a05c_abort_flag_stops_run(self, mock_call_module):
        """An abort flag stops the run before the next module, state aborted."""
        from picasso_workflow import progress as pwprogress

        mock_call_module.return_value = True
        reporter_config = {"report_name": "abortreport"}
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = [
            ("load_dataset_movie", {"b": 3}),
            ("identify", {"min_gradient": 1}),
        ]
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        # request abort before running: no module should execute
        pwprogress.request_abort(wr.result_folder)
        success = wr.run()

        self.assertFalse(success)
        self.assertEqual(mock_call_module.call_count, 0)
        state = pwprogress.read_progress(wr.result_folder)
        self.assertEqual(state["state"], "aborted")
        shutil.rmtree(wr.result_folder)

    def test_b01_AggregationWR_init(self):
        awr = AggregationWorkflowRunner()
        assert awr.sgl_workflow_locations == []

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceInterface", MagicMock)
    @patch("picasso_workflow.workflow.WorkflowRunner", MagicMock)
    @patch("picasso_workflow.workflow.ParameterTiler")
    def test_b01_AggregationWR_fromdicts(self, mock_parameter_tiler):
        mock_parameter_tiler = MagicMock()
        mock_parameter_tiler.ntiles = 3
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {
                "base_url": "",
                "username": "",
                "space_key": "",
                "parent_page_title": "",
            },
        }
        analysis_config = {"result_location": self.results_folder}
        aggregation_workflow = {
            "single_dataset_tileparameters": {},
            "single_dataset_modules": [("load_dataset", {"b": 3})],
            "aggregation_modules": [],
        }

        awr = AggregationWorkflowRunner().config_from_dicts(
            reporter_config, analysis_config, aggregation_workflow
        )
        assert awr.sgl_workflow_locations == []

        shutil.rmtree(awr.result_folder)

    # @unittest.skip('')
    @patch("picasso_workflow.workflow.ConfluenceInterface")
    @patch("picasso_workflow.workflow.WorkflowRunner")
    @patch("picasso_workflow.workflow.ParameterTiler")
    def test_b02_AggregationWR_save_load(
        self, mock_parameter_tiler, mock_WR, mock_ci
    ):
        # create_page returns the new page's id (a string); the runner now
        # stores it in the reporter config, so the mock must return a
        # serializable value rather than a bare MagicMock.
        mock_ci.return_value.create_page.return_value = "12345"
        mock_parameter_tiler = MagicMock()
        mock_parameter_tiler.ntiles = 3
        mock_parameter_tiler.return_value = {"the_parameters": [0, 1, 2]}
        mock_WR = MagicMock()
        mock_WR.results = {}
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {
                "base_url": "",
                "username": "",
                "space_key": "",
                "parent_page_title": "",
            },
        }
        analysis_config = {"result_location": self.results_folder}
        aggregation_workflow = {
            "single_dataset_tileparameters": {},
            "single_dataset_modules": [("load_dataset", {"b": 3})],
            "aggregation_modules": [],
        }

        awr = AggregationWorkflowRunner().config_from_dicts(
            reporter_config, analysis_config, aggregation_workflow
        )
        awr.all_results["single_dataset"] = [
            {"load_results": {"filename": "a.tiff"}},
            {"load_results": {"filename": "b.tiff"}},
        ]
        awr.all_results["aggregation"] = []

        awr.save(self.results_folder)
        logger.debug("Saved AggregationWorkflowRunner successfully.")

        awr2 = AggregationWorkflowRunner.load(self.results_folder)
        logger.debug("Loaded AggregationWorkflowRunner successfully.")

        shutil.rmtree(awr2.result_folder)
        os.remove(
            os.path.join(self.results_folder, "AggregationWorkflowRunner.yaml")
        )


class Test_D_WorkflowRunnerErrorRecording(unittest.TestCase):
    """A failed module must leave a trace on disk and a rich report."""

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "TestData", "results"
        )
        os.makedirs(self.results_folder, exist_ok=True)

    def _runner(self):
        return WorkflowRunner.config_from_dicts(
            {"report_name": "myreport", "ConfluenceReporter": {"a": 0}},
            {"result_location": self.results_folder},
            [("dummy_module", {"parameter0": 1})],
        )

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_report_error_receives_index_and_parameters(self):
        wr = self._runner()

        def failing_module(i, parameters):
            raise RuntimeError("kaboom")

        wr.autopicasso.dummy_module = failing_module
        wr.confluencereporter.dummy_module = MagicMock()

        with self.assertRaises(RuntimeError):
            wr.call_module("dummy_module", 0, {"parameter0": 1})

        kwargs = wr.confluencereporter.report_error.call_args[1]
        assert kwargs["i"] == 0
        assert kwargs["parameters"] == {"parameter0": 1}

        shutil.rmtree(wr.result_folder)

    # ParameterCommandExecutor is left real here so the recorded
    # parameters are the genuine resolved dict.
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    def test_failed_module_is_recorded_in_yaml(self):
        """A plain exception used to escape run() before save(), so the
        failing module was absent from WorkflowRunner.yaml entirely."""
        wr = self._runner()

        def failing_module(i, parameters):
            raise RuntimeError("kaboom")

        wr.autopicasso.dummy_module = failing_module
        wr.confluencereporter.dummy_module = MagicMock()

        with self.assertRaises(RuntimeError):
            wr.run()

        fp = os.path.join(wr.result_folder, "WorkflowRunner.yaml")
        assert os.path.exists(fp)
        with open(fp, "r") as f:
            data = yaml.unsafe_load(f)
        entry = data["results"]["00_dummy_module"]
        assert entry["success"] is False
        assert entry["error"]["type"] == "RuntimeError"
        assert "kaboom" in entry["error"]["message"]
        assert entry["error"]["index"] == 0
        assert entry["parameters"] == {"parameter0": 1}

        shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_autopicasso_error_on_first_module_returns_false(self):
        """Used to raise UnboundLocalError on 'success' instead."""
        wr = self._runner()

        def failing_module(i, parameters):
            raise AutoPicassoError("nope")

        wr.autopicasso.dummy_module = failing_module
        wr.confluencereporter.dummy_module = MagicMock()

        assert wr.run() is False

        shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_reraised_error_keeps_original_traceback(self):
        """copy.copy() dropped __traceback__, so the exception escaping
        call_module() used to stop at the re-raise instead of pointing at
        the code that actually failed."""
        import traceback as _tb

        wr = self._runner()

        def deep_failure():
            raise RuntimeError("kaboom")

        wr.autopicasso.dummy_module = lambda i, p: deep_failure()
        wr.confluencereporter.dummy_module = MagicMock()

        # Not assertRaises: it stores the exception via
        # with_traceback(None), which would strip exactly what is under
        # test here.
        text = None
        try:
            wr.call_module("dummy_module", 0, {"parameter0": 1})
        except RuntimeError as exc:
            assert exc.__traceback__ is not None
            text = "".join(
                _tb.format_exception(type(exc), exc, exc.__traceback__)
            )
        assert text is not None, "call_module did not raise"
        assert "in deep_failure" in text

        shutil.rmtree(wr.result_folder)


class Test_E_AggregationFailureTraceability(unittest.TestCase):
    """A skipped aggregation must name which datasets failed, and why."""

    def _awr(self, single_results):
        awr = AggregationWorkflowRunner.__new__(AggregationWorkflowRunner)
        awr.all_results = {"single_dataset": single_results}
        return awr

    def test_describes_recorded_error(self):
        awr = self._awr(
            [
                {
                    "12_nneighbor": {"success": True},
                    "13_fit_csr": {
                        "success": False,
                        "error": {
                            "type": "ValueError",
                            "message": "max_dist=300.0 leaves 0 of 99",
                        },
                    },
                }
            ]
        )
        desc = awr._describe_single_failure(0)
        assert "13_fit_csr" in desc
        assert "ValueError" in desc
        assert "leaves 0 of 99" in desc

    def test_falls_back_to_last_module_when_no_error_recorded(self):
        """Results written before failures were recorded stop silently."""
        awr = self._awr([{"12_nneighbor": {"success": True}}])
        desc = awr._describe_single_failure(0)
        assert "12_nneighbor" in desc
        assert "no error recorded" in desc

    def test_handles_missing_results(self):
        awr = self._awr([None])
        assert "no results recorded" in awr._describe_single_failure(0)
        awr2 = self._awr([])
        assert "no results recorded" in awr2._describe_single_failure(3)

    def test_collects_only_failed_datasets(self):
        awr = self._awr(
            [
                {"00_a": {"success": True}},
                {
                    "00_a": {
                        "success": False,
                        "error": {"type": "ValueError", "message": "boom"},
                    }
                },
                {"00_a": {"success": True}},
            ]
        )
        failures = awr._failed_single_datasets(
            [True, False, True],
            ["/f0", "/f1", "/f2"],
            ["tagA", "tagB", "tagC"],
        )
        assert len(failures) == 1
        idx, tag, folder, desc = failures[0]
        assert (idx, tag, folder) == (1, "tagB", "/f1")
        assert "ValueError: boom" in desc

    def test_abort_body_names_every_failure(self):
        from picasso_workflow.confluence import aggregation_abort_body

        body = aggregation_abort_body(
            [
                (11, "PDL1-H9_PDL1-Mb", "/res/11", "fit_csr: ValueError"),
                (15, "PDL1-H12_PDL1-Mb", "/res/15", "fit_csr: ValueError"),
            ],
            16,
        )
        assert "2 of 16" in body
        assert "PDL1-H9_PDL1-Mb" in body
        assert "PDL1-H12_PDL1-Mb" in body
        assert "/res/11" in body and "/res/15" in body
        assert "<h2>" in body

    def test_report_abort_is_best_effort(self):
        """A reporting failure must not mask the analysis failure."""
        awr = self._awr([])
        awr.ci = MagicMock()
        awr.ci.update_page_content.side_effect = RuntimeError("down")
        awr.reporter_config = {
            "report_name": "r",
            "ConfluenceReporter": {"parent_page_id": "42"},
        }
        # must not raise
        awr._report_aggregation_abort([(0, "t", "/f", "d")], 1)

    def test_report_abort_noop_without_page_id(self):
        awr = self._awr([])
        awr.ci = MagicMock()
        awr.reporter_config = {"report_name": "r", "ConfluenceReporter": {}}
        awr._report_aggregation_abort([(0, "t", "/f", "d")], 1)
        awr.ci.update_page_content.assert_not_called()


# --- checkpoint-aware resume -------------------------------------------------


class Test_F_CheckpointResume(unittest.TestCase):
    """Resume planning: frontier, checkpoint restore, param-change re-run."""

    MODULES = [
        ("load_dataset_localizations", {"filename": "in.hdf5"}),
        ("dbscan", {"radius": 2, "min_density": 10}),
        ("nneighbor", {"dims": ["x", "y"]}),
        ("save_single_dataset", {"filename": "out.hdf5"}),
    ]

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )

    def _make_runner(self, workflow_modules, succeeded, warm=False):
        """Build a runner simulating a resumed run.

        succeeded: number of leading modules recorded as previously
        succeeded (results entry + module folder). warm: keep the
        MagicMock autopicasso attributes (in-memory state present).
        """
        reporter_config = {"report_name": "resumereport"}
        analysis_config = {"result_location": self.results_folder}
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        # the class-level patch specs the mock to the constructor args;
        # use a plain MagicMock whose attributes are freely accessible
        wr.autopicasso = MagicMock()
        if not warm:
            # simulate the fresh AutoPicasso of a loaded runner
            wr.autopicasso.locs = None
            wr.autopicasso.movie = None
            wr.autopicasso.identifications = None
            wr.autopicasso.channel_locs = None
        wr.results = {}
        for i in range(succeeded):
            module_name = workflow_modules[i][0]
            module_id = f"{i:02d}_{module_name}"
            folder = os.path.join(wr.result_folder, module_id)
            os.makedirs(folder, exist_ok=True)
            wr.results[module_id] = {"success": True, "folder": folder}
        return wr

    def _touch(self, *path):
        fp = os.path.join(*path)
        with open(fp, "w") as f:
            f.write("x")
        return fp

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_restore_from_checkpoint(self, mock_call_module):
        """Cold resume restores the latest checkpoint and re-runs after it."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=3)
        # module 1 saved locs (save_locs / always_save auto-save)
        ckpt_fp = self._touch(wr.results["01_dbscan"]["folder"], "locs.hdf5")
        try:
            success = wr.run()
            self.assertTrue(success)
            wr.autopicasso.load_checkpoint.assert_called_once()
            descriptor = wr.autopicasso.load_checkpoint.call_args[0][0]
            self.assertEqual(ckpt_fp, descriptor["single"]["filepath"])
            self.assertEqual("01_dbscan", descriptor["module_id"])
            # module 0 and 1 skipped, 2 and 3 executed
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([2, 3], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_no_checkpoint_reruns_from_scratch(self, mock_call_module):
        """Cold resume without any saved locs re-runs everything."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=3)
        try:
            wr.run()
            wr.autopicasso.load_checkpoint.assert_not_called()
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([0, 1, 2, 3], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_all_succeeded_skips_everything(self, mock_call_module):
        """A fully-successful previous run skips all modules, no restore."""
        wr = self._make_runner(self.MODULES, succeeded=len(self.MODULES))
        try:
            success = wr.run()
            self.assertTrue(success)
            self.assertEqual(0, mock_call_module.call_count)
            wr.autopicasso.load_checkpoint.assert_not_called()
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_warm_memory_skips_to_frontier(self, mock_call_module):
        """A live re-run (state in memory) skips straight to the frontier."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=3, warm=True)
        try:
            wr.run()
            wr.autopicasso.load_checkpoint.assert_not_called()
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([3], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_hard_memory_conflict_rejects_checkpoint(self, mock_call_module):
        """A checkpoint stranding a memory-only dependency is rejected."""
        mock_call_module.return_value = True
        modules = [
            ("load_dataset_movie", {"filename": "a.tiff"}),
            ("identify", {"min_gradient": 5000}),
            ("localize", {"fit_method": "lsq"}),
        ]
        wr = self._make_runner(modules, succeeded=2)
        # a checkpoint at module 0 exists, but identify/localize need the
        # memory-only raw_movie produced by module 0 -> must run scratch
        self._touch(wr.results["00_load_dataset_movie"]["folder"], "locs.hdf5")
        try:
            wr.run()
            wr.autopicasso.load_checkpoint.assert_not_called()
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([0, 1, 2], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_restore_failure_falls_back_to_scratch(self, mock_call_module):
        """An unreadable checkpoint file downgrades to a scratch re-run."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=3)
        self._touch(wr.results["01_dbscan"]["folder"], "locs.hdf5")
        wr.autopicasso.load_checkpoint.side_effect = OSError("corrupt")
        try:
            wr.run()
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([0, 1, 2, 3], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_changed_parameter_moves_frontier(self, mock_call_module):
        """A changed argument on a succeeded module forces re-run from it."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=len(self.MODULES))
        ckpt_fp = self._touch(
            wr.results["00_load_dataset_localizations"]["folder"],
            "locs.hdf5",
        )
        previous = [(name, dict(params)) for name, params in self.MODULES]
        previous[1] = ("dbscan", {"radius": 5, "min_density": 10})
        wr._previous_workflow_modules = previous
        try:
            wr.run()
            descriptor = wr.autopicasso.load_checkpoint.call_args[0][0]
            self.assertEqual(ckpt_fp, descriptor["single"]["filepath"])
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([1, 2, 3], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_adopt_workflow_modules(self):
        """Edited parameters are adopted; a changed sequence is not."""
        wr = self._make_runner(self.MODULES, succeeded=0)
        try:
            edited = [(name, dict(params)) for name, params in self.MODULES]
            edited[1] = ("dbscan", {"radius": 7, "min_density": 10})
            wr.adopt_workflow_modules(edited)
            self.assertEqual(edited, wr.workflow_modules)
            self.assertEqual(
                dict(self.MODULES[1][1]),
                dict(wr._previous_workflow_modules[1][1]),
            )
            # sequence mismatch: keep the previous modules
            wr.adopt_workflow_modules([("dbscan", {})])
            self.assertEqual(edited, wr.workflow_modules)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_adopt_compares_against_pristine_parameters(self):
        """Module write-backs into their parameters (persisted to yaml) must
        not read as user edits: comparison uses the pristine snapshot."""
        modules = [(name, dict(params)) for name, params in self.MODULES]
        fresh = [(name, dict(params)) for name, params in self.MODULES]
        wr = self._make_runner(modules, succeeded=0)
        try:
            # simulate a module writing back into its own parameters (e.g.
            # identify storing the estimated min_gradient)
            wr.workflow_modules[1][1]["added_by_module"] = 42
            wr.adopt_workflow_modules(fresh)
            self.assertNotIn(
                "added_by_module", wr._previous_workflow_modules[1][1]
            )
            self.assertFalse(
                any(
                    _module_parameters_changed(prev[1], new[1])
                    for prev, new in zip(wr._previous_workflow_modules, fresh)
                )
            )
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_file_mediated_resume_continues_at_frontier(
        self, mock_call_module
    ):
        """Without a checkpoint, a cold resume still continues at the
        frontier when the remaining modules need no in-memory state (e.g.
        continuation after a manual step)."""
        mock_call_module.return_value = True
        modules = [
            ("manual", {"filename": "a.hdf5"}),
            ("load_dataset_localizations", {"filename": "a.hdf5"}),
            ("dbscan", {"radius": 2, "min_density": 10}),
        ]
        wr = self._make_runner(modules, succeeded=1)
        try:
            wr.run()
            wr.autopicasso.load_checkpoint.assert_not_called()
            ran = [c.args[1] for c in mock_call_module.call_args_list]
            self.assertEqual([1, 2], ran)
        finally:
            shutil.rmtree(wr.result_folder)

    @patch("picasso_workflow.workflow.WorkflowRunner.call_module")
    @patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
    @patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
    @patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
    def test_stale_param_diff_does_not_refire_on_second_run(
        self, mock_call_module
    ):
        """run() consumes the previous-run diff so a second in-process run
        does not re-run from a stale parameter-change frontier."""
        mock_call_module.return_value = True
        wr = self._make_runner(self.MODULES, succeeded=len(self.MODULES))
        self._touch(
            wr.results["00_load_dataset_localizations"]["folder"],
            "locs.hdf5",
        )
        previous = [(name, dict(params)) for name, params in self.MODULES]
        previous[1] = ("dbscan", {"radius": 5, "min_density": 10})
        wr._previous_workflow_modules = previous
        try:
            wr.run()
            self.assertIsNone(wr._previous_workflow_modules)
        finally:
            shutil.rmtree(wr.result_folder)


def test_module_parameters_changed_detects_value_change():
    prev = {"radius": 2, "nested": {"a": [1, 2]}}
    new = {"radius": 2, "nested": {"a": [1, 2]}}
    assert not _module_parameters_changed(prev, new)
    assert _module_parameters_changed(prev, {**new, "radius": 3})


def test_module_parameters_changed_legacy():
    """Legacy yamls (no pristine snapshot) hold mutated parameters: module
    write-backs are ignored, resolved commands compare equal to their raw
    form, and a real user edit is still detected."""
    prev = {
        "n_plot_structures": 20,
        "dimensions": ["x", "y"],  # written back by the module, not a user key
        "filepath": "/resolved/file.hdf5",
        "filepath_originalnocmd": (
            "get_prior_result",  # stored sign-stripped by the PCE
            "results, 00_x, filepath",
        ),
        "nested": {"a": 1, "added_by_module": 2},
    }
    new_same = {
        "n_plot_structures": 20,
        "filepath": ("$get_prior_result", "results, 00_x, filepath"),
        "nested": {"a": 1},
    }
    assert not _module_parameters_changed_legacy(prev, new_same)
    assert _module_parameters_changed_legacy(
        prev, {**new_same, "n_plot_structures": 21}
    )
    assert _module_parameters_changed_legacy(
        prev, {**new_same, "brand_new_key": 1}
    )
    changed_cmd = dict(new_same)
    changed_cmd["filepath"] = (
        "$get_prior_result",
        "results, 01_other, filepath",
    )
    assert _module_parameters_changed_legacy(prev, changed_cmd)


def test_module_parameters_changed_legacy_ignores_write_backs():
    """Values a module overwrites in place (registered in
    RUNTIME_PARAMETER_WRITE_BACKS) must not read as user edits -- here
    identify's estimated min_gradient and the absolutized auto_netgrad
    filename -- while a real edit in the same dicts is still detected."""
    prev = {
        "box_size": 7,
        "min_gradient": 4123.7,  # overwritten with the estimate
        "auto_netgrad": {
            # overwritten with the absolute result-folder path
            "filename": "/results/01_identify/auto_identification.png",
            "frame_numbers": [0, 10],
        },
    }
    new_same = {
        "box_size": 7,
        "min_gradient": 5000,
        "auto_netgrad": {
            "filename": "auto_identification.png",
            "frame_numbers": [0, 10],
        },
    }
    assert not _module_parameters_changed_legacy(prev, new_same, "identify")
    # without the module's ignore list, the write-backs would be flagged
    assert _module_parameters_changed_legacy(prev, new_same)
    # a genuine edit next to the ignored keys is still detected
    edited = {
        **new_same,
        "auto_netgrad": {**new_same["auto_netgrad"], "frame_numbers": [0, 5]},
    }
    assert _module_parameters_changed_legacy(prev, edited, "identify")
    assert _module_parameters_changed_legacy(
        prev, {**new_same, "box_size": 9}, "identify"
    )


@patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
@patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
@patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
def test_resume_from_pre_snapshot_yaml_detects_changed_param(tmp_path):
    """Resuming a run recorded before the pristine snapshot existed falls
    back to the legacy comparison and still detects an edited argument."""
    modules = [("pick_origami", {"n_plot_structures": 20})]
    wr = WorkflowRunner.config_from_dicts(
        {"report_name": "rep"}, {"result_location": str(tmp_path)}, modules
    )
    wr.save(wr.result_folder)
    # strip the snapshot, as a yaml written by an older version
    fp = os.path.join(wr.result_folder, "WorkflowRunner.yaml")
    with open(fp) as f:
        data = yaml.safe_load(f)
    del data["workflow_modules_pristine"]
    with open(fp, "w") as f:
        yaml.dump(data, f)

    edited = [("pick_origami", {"n_plot_structures": 21})]
    wr2 = WorkflowRunner.config_from_dicts(
        {"report_name": "rep_991231-2359"},
        {"result_location": str(tmp_path)},
        edited,
        continue_previous_runner=True,
    )
    assert wr2._previous_modules_are_legacy
    assert wr2.workflow_modules == edited
    assert _module_parameters_changed_legacy(
        wr2._previous_workflow_modules[0][1], edited[0][1]
    )


def test_module_parameters_changed_on_pristine_commands():
    """Pristine snapshots hold raw commands on both sides, so identical
    $-command parameters compare equal and a changed one is detected."""
    prev = {"filepath": ("$get_prior_result", "results, 00_load, filepath")}
    new = {"filepath": ("$get_prior_result", "results, 00_load, filepath")}
    assert not _module_parameters_changed(prev, new)
    changed = {
        "filepath": ("$get_prior_result", "results, 01_other, filepath")
    }
    assert _module_parameters_changed(prev, changed)


def test_checkpoint_detection_explicit_descriptor(tmp_path):
    fp = tmp_path / "locs.hdf5"
    fp.write_text("x")
    results = {"checkpoint": {"single": {"filepath": str(fp)}}}
    assert _checkpoint_from_module_results("any_module", results) == {
        "single": {"filepath": str(fp)}
    }
    # missing file: rejected
    results = {"checkpoint": {"single": {"filepath": str(tmp_path / "n")}}}
    assert _checkpoint_from_module_results("any_module", results) is None


def test_checkpoint_detection_legacy_channels(tmp_path):
    fps = []
    for tag in ("ch_a", "ch_b"):
        fp = tmp_path / f"{tag}.hdf5"
        fp.write_text("x")
        fps.append(str(fp))
    results = {"filepaths": fps}
    ckpt = _checkpoint_from_module_results("save_datasets_aggregated", results)
    assert ckpt == {"channels": {"filepaths": fps, "tags": ["ch_a", "ch_b"]}}
    # recorded tags win over derived ones
    results = {"filepaths": fps, "tags": ["t1", "t2"]}
    ckpt = _checkpoint_from_module_results(
        "load_datasets_to_aggregate", results
    )
    assert ckpt["channels"]["tags"] == ["t1", "t2"]


def test_checkpoint_detection_legacy_filepath_is_name_gated(tmp_path):
    fp = tmp_path / "out.hdf5"
    fp.write_text("x")
    results = {"filepath": str(fp)}
    assert _checkpoint_from_module_results("save_single_dataset", results) == {
        "single": {"filepath": str(fp)}
    }
    # the manual module records an unrelated filepath: not a checkpoint
    assert _checkpoint_from_module_results("manual", results) is None


def test_checkpoint_detection_folder_autosave(tmp_path):
    folder = tmp_path / "03_dbscan"
    folder.mkdir()
    results = {"folder": str(folder)}
    assert _checkpoint_from_module_results("dbscan", results) is None
    (folder / "locs.hdf5").write_text("x")
    assert _checkpoint_from_module_results("dbscan", results) == {
        "single": {"filepath": str(folder / "locs.hdf5")}
    }


def test_checkpoint_detection_tuple_tags_and_mismatch(tmp_path):
    """Tuple tags (yaml python/tuple round-trip) are accepted; a
    length-mismatched tags entry falls back to filename-derived tags."""
    fps = []
    for tag in ("ch_a", "ch_b"):
        fp = tmp_path / f"{tag}.hdf5"
        fp.write_text("x")
        fps.append(str(fp))
    results = {"filepaths": fps, "tags": ("t1", "t2")}
    ckpt = _checkpoint_from_module_results(
        "load_datasets_to_aggregate", results
    )
    assert ckpt["channels"]["tags"] == ["t1", "t2"]
    results = {"filepaths": fps, "tags": ["only_one"]}
    ckpt = _checkpoint_from_module_results(
        "load_datasets_to_aggregate", results
    )
    assert ckpt["channels"]["tags"] == ["ch_a", "ch_b"]


def test_strip_runstamp():
    assert _strip_runstamp("myreport_260929-1015") == "myreport"
    assert _strip_runstamp("myreport") == "myreport"
    # only a trailing runstamp is stripped
    assert (
        _strip_runstamp("myreport_260929-1015_x") == "myreport_260929-1015_x"
    )


def test_find_previous_runner_postfix_handles_tokens(tmp_path):
    """Folders may carry a per-run token between base name and runstamp;
    the latest trailing runstamp wins and the full postfix is returned."""
    (tmp_path / "tag_p1a2b3_240310-1130").mkdir()
    (tmp_path / "tag_240311-0900").mkdir()
    (tmp_path / "tag_notarun").mkdir()
    (tmp_path / "othertag_240312-0900").mkdir()
    assert _find_previous_runner_postfix(str(tmp_path), "tag") == "240311-0900"
    # the token-bearing folder wins once it is the latest
    (tmp_path / "tag_p9z8y7_240315-1000").mkdir()
    assert (
        _find_previous_runner_postfix(str(tmp_path), "tag")
        == "p9z8y7_240315-1000"
    )
    assert _find_previous_runner_postfix(str(tmp_path), "nomatch") is None
    assert _find_previous_runner_postfix(str(tmp_path / "gone"), "t") is None


@patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
@patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
@patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
def test_resume_finds_previous_run_despite_fresh_runstamp(tmp_path):
    """The coordinators stamp report names per launch; resume must still
    find the earlier run's folder by the stamp-free base name."""
    modules = [("dbscan", {"radius": 2})]
    reporter_config = {"report_name": "myreport"}
    analysis_config = {"result_location": str(tmp_path)}
    wr = WorkflowRunner.config_from_dicts(
        reporter_config, analysis_config, modules
    )
    wr.results = {"00_dbscan": {"success": True}}
    wr.save(wr.result_folder)

    edited = [("dbscan", {"radius": 3})]
    wr2 = WorkflowRunner.config_from_dicts(
        {"report_name": "myreport_991231-2359"},
        {"result_location": str(tmp_path)},
        edited,
        continue_previous_runner=True,
    )
    assert wr2.results == {"00_dbscan": {"success": True}}
    assert wr2.workflow_modules == edited
    assert wr2._previous_workflow_modules == modules


@patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
@patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
@patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
def test_resume_with_yamlless_folder_starts_fresh(tmp_path):
    """A previous run killed before its first save leaves a folder without
    WorkflowRunner.yaml; resume must start fresh instead of crashing."""
    (tmp_path / "myreport_240310-1130").mkdir()
    wr = WorkflowRunner.config_from_dicts(
        {"report_name": "myreport"},
        {"result_location": str(tmp_path)},
        [("dbscan", {"radius": 2})],
        continue_previous_runner=True,
    )
    assert wr.results == {}


@patch("picasso_workflow.workflow.ConfluenceReporter", MagicMock)
@patch("picasso_workflow.workflow.AutoPicasso", MagicMock)
@patch("picasso_workflow.workflow.ParameterCommandExecutor", MagicMock)
def test_pristine_modules_round_trip_save_load(tmp_path):
    """The pristine parameter snapshot survives save/load untouched by
    in-place mutations of the live workflow_modules."""
    modules = [("dbscan", {"radius": 2})]
    wr = WorkflowRunner.config_from_dicts(
        {"report_name": "roundtrip"},
        {"result_location": str(tmp_path)},
        modules,
    )
    # simulate $-resolution / module write-back mutating the live params
    wr.workflow_modules[0][1]["radius"] = 99
    wr.workflow_modules[0][1]["added"] = "abc"
    wr.save(wr.result_folder)
    wr2 = WorkflowRunner.load(wr.result_folder)
    assert wr2.workflow_modules_pristine == [("dbscan", {"radius": 2})]


# --- dynamic single-dataset scheduling (atomic filesystem claims) -----------


def _bare_runner(rank=0, result_folder="."):
    """An AggregationWorkflowRunner with just the fields the claim helpers
    touch, bypassing __init__ (no config needed)."""
    r = AggregationWorkflowRunner.__new__(AggregationWorkflowRunner)
    r.rank = rank
    r.result_folder = result_folder
    return r


def test_claim_dataset_is_exclusive_and_covers_all(tmp_path):
    """Two ranks racing over the same datasets: each dataset is claimed by
    exactly one rank, and every dataset is claimed."""
    claim_dir = str(tmp_path / "claims")
    os.makedirs(claim_dir)
    r0, r1 = _bare_runner(rank=0), _bare_runner(rank=1)

    claimed0, claimed1 = [], []
    for i in range(6):
        # both ranks reach dataset i; the atomic mkdir lets exactly one win
        if r0._claim_dataset(claim_dir, i):
            claimed0.append(i)
        if r1._claim_dataset(claim_dir, i):
            claimed1.append(i)

    assert sorted(claimed0 + claimed1) == list(range(6))  # all covered
    assert set(claimed0) & set(claimed1) == set()  # none double-claimed
    # re-claiming an owned dataset fails (idempotent skip)
    assert r1._claim_dataset(claim_dir, claimed0[0]) is False


def test_claim_dir_is_job_scoped(tmp_path, monkeypatch):
    """The claim dir is keyed by SLURM job id, so a new launch (new job) gets
    a fresh dir and stale claims never block a rerun; off-cluster -> 'local'.
    """
    r = _bare_runner(result_folder=str(tmp_path))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    d1 = r._claim_dir()
    monkeypatch.setenv("SLURM_JOB_ID", "67890")
    d2 = r._claim_dir()
    assert d1 != d2 and "12345" in d1 and "67890" in d2

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert r._claim_dir().endswith(os.path.join("_pwf_claims", "local"))


def test_claim_dataset_survives_fs_error(tmp_path):
    """An unexpected filesystem error claims the dataset locally (run it) so it
    is never silently dropped."""
    r = _bare_runner()
    # claim dir does not exist -> os.mkdir raises FileNotFoundError, which is
    # not FileExistsError, so the dataset is run here rather than skipped
    missing = str(tmp_path / "does_not_exist")
    assert r._claim_dataset(missing, 0) is True


def test_wait_returns_when_all_marked(tmp_path):
    """The barrier returns immediately once every folder has a marker, and
    never reclaims a dataset that already finished."""
    r = _bare_runner(result_folder=str(tmp_path))
    f0, f1 = str(tmp_path / "d0"), str(tmp_path / "d1")
    r._write_single_marker(f0, True)
    r._write_single_marker(f1, True)
    reclaimed = []
    r._wait_for_single_markers(
        [f0, f1], reclaim=lambda i: reclaimed.append(i), poll=0
    )
    assert reclaimed == []


def test_wait_reclaims_orphaned_dataset(tmp_path):
    """A dataset with no marker and no advancing progress.json (its rank died)
    is re-run on rank 0 instead of hanging the barrier until timeout."""
    r = _bare_runner(result_folder=str(tmp_path))
    f0, f1 = str(tmp_path / "d0"), str(tmp_path / "d1")
    r._write_single_marker(f0, True)  # d0 already done
    # d1 is orphaned: no marker, no progress.json
    reclaimed = []

    def reclaim(i):
        reclaimed.append(i)
        r._write_single_marker(
            [f0, f1][i], True
        )  # completing it ends the wait

    # stale_grace=-1 makes an unchanged-progress dataset orphaned on the first
    # poll, so the reclaim path is exercised deterministically (no sleep).
    r._wait_for_single_markers(
        [f0, f1], reclaim=reclaim, poll=0, stale_grace=-1
    )
    assert reclaimed == [1]

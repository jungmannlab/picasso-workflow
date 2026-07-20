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
from picasso_workflow.workflow import WorkflowRunner, AggregationWorkflowRunner

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

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

from picasso_workflow.workflow import WorkflowRunner, AggregationWorkflowRunner


logger = logging.getLogger(__name__)


@unittest.skip("")
class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "TestData", "workflow"
        )

    def tearDown(self):
        pass

    @patch("picasso_workflow.workflow.ParameterCommandExecutor")
    def test_a01_WorkflowRunner_init(self, mock_pce):
        wr = WorkflowRunner()
        assert wr.results == {}

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
    def test_a04_WorkflowRunner_save_load(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
        }
        analysis_config = {"result_location": self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )
        wr.autopicasso.my_module = lambda i, p: ({}, {})

        wr.call_module("my_module", 0, {"parameter0": 1})

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
        workflow_modules = [("load_dataset", {"b": 3})]

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )

        wr.run()

        shutil.rmtree(wr.result_folder)

    def test_b01_AggregationWR_init(self):
        awr = AggregationWorkflowRunner()
        assert awr.sgl_workflow_locations == []

    @patch("picasso_workflow.workflow.WorkflowRunner", MagicMock)
    @patch("picasso_workflow.workflow.ParameterTiler")
    def test_b01_AggregationWR_fromdicts(self, mock_parameter_tiler):
        mock_parameter_tiler = MagicMock()
        mock_parameter_tiler.ntiles = 3
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
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
    @patch("picasso_workflow.workflow.WorkflowRunner")
    @patch("picasso_workflow.workflow.ParameterTiler")
    def test_b02_AggregationWR_save_load(self, mock_parameter_tiler, mock_WR):
        mock_parameter_tiler = MagicMock()
        mock_parameter_tiler.ntiles = 3
        mock_parameter_tiler.return_value = {"the_parameters": [0, 1, 2]}
        mock_WR = MagicMock()
        mock_WR.results = {}
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {"a": 0},
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

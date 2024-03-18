#!/usr/bin/env python
"""
Module Name: test_integration.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the integration of the package: run complete
    analysis workflows on minimal data.
"""
import os
import shutil
import logging
import unittest


from picasso_workflow.workflow import WorkflowRunner, AggregationWorkflowRunner


logger = logging.getLogger(__name__)


# @unittest.skip('')
class Test_A_PackageIntegration(unittest.TestCase):
    """Test the interplay of the different modules in the package."""

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "TestData",
            "integration",
        )
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

    def tearDown(self):
        pass

    def test_a01_WorkflowRunner(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {
                "base_url": self.confluence_url,
                "space_key": self.confluence_space,
                "parent_page_title": self.confluence_page,
                "token": self.confluence_token,
            },
        }
        analysis_config = {
            "result_location": self.results_folder,
            "camera_info": {
                "gain": 1,
                "sensitivity": 0.45,
                "baseline": 100,
                "qe": 0.82,
                "pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        workflow_modules = [
            (
                "manual",
                {
                    "prompt": "Please manually undrift.",
                    "filename": "locs_undrift.hdf5",
                },
            )
        ]

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )

        wr.run()

        shutil.rmtree(wr.result_folder)

    def test_b01_AggregationWorkflowRunner(self):
        reporter_config = {
            "report_name": "myreport",
            "ConfluenceReporter": {
                "base_url": self.confluence_url,
                "space_key": self.confluence_space,
                "parent_page_title": self.confluence_page,
                "token": self.confluence_token,
            },
        }
        analysis_config = {
            "result_location": self.results_folder,
            "camera_info": {
                "gain": 1,
                "sensitivity": 0.45,
                "baseline": 100,
                "qe": 0.82,
                "pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        single_dataset_modules = [
            (
                "manual",
                {
                    "prompt": "Please manually undrift.",
                    "additional_parameter": ("$map", "mypar"),
                    "filename": "locs_undrift.hdf5",
                },
            )
        ]
        aggregation_workflow = {
            "single_dataset_tileparameters": {"mypar": [0, 1]},
            "single_dataset_modules": single_dataset_modules,
            "aggregation_modules": [],
        }

        wr = AggregationWorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, aggregation_workflow
        )

        wr.run()

        shutil.rmtree(wr.result_folder)


@unittest.skip("")
class Test_B_CompleteIntegration(unittest.TestCase):
    """Test the interplay of the package with its dependencies
    (i.e. picasso), and the complete analysis of minimal datasets.
    """

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "TestData",
            "integration",
        )

    def tearDown(self):
        pass

    def test_01_WorkflowRunner(self):
        assert False

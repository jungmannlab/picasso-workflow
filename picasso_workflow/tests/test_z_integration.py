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


from picasso_workflow.workflow import (
    WorkflowRunner,
    AggregationWorkflowRunner,
    WorkflowError,
)
from picasso_workflow.confluence import ConfluenceInterface
import picasso_workflow.standard_singledataset_workflows as ssw
import picasso_workflow.standard_aggregation_workflows as saw


logger = logging.getLogger(__name__)


# @unittest.skip('')
class Test_A_PackageIntegration(unittest.TestCase):
    """Test the interplay of the different modules in the package."""

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

    def tearDown(self):
        pass

    def test_a01_WorkflowRunner(self):
        reporter_config = {
            "report_name": "test_a01_WorkflowRunner",
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

        # clean up
        shutil.rmtree(wr.result_folder)
        wr.confluencereporter.ci.delete_page(wr.reporter_config["report_name"])

    # @unittest.skip('')
    def test_b01_AggregationWorkflowRunner(self):
        reporter_config = {
            "report_name": "test_b01_AggregationWorkflowRunner",
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
                    "additional_parameter": ("$$map", "mypar"),
                    "filename": "locs_undrift.hdf5",
                },
            )
        ]
        aggregation_workflow = {
            "single_dataset_tileparameters": {"mypar": [0, 1]},
            "single_dataset_modules": single_dataset_modules,
            "aggregation_modules": [],
        }

        awr = AggregationWorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, aggregation_workflow
        )

        with self.assertRaises(WorkflowError):
            awr.run()

        # clean up
        shutil.rmtree(awr.result_folder)
        ci = ConfluenceInterface(
            base_url=self.confluence_url,
            space_key=self.confluence_space,
            parent_page_title=self.confluence_page,
            token=self.confluence_token,
        )
        for repname in awr.cpage_names:
            ci.delete_page(repname)


# @unittest.skip("")
class Test_B_CompleteIntegration(unittest.TestCase):
    """Test the interplay of the package with its dependencies
    (i.e. picasso), and the complete analysis of minimal datasets.
    """

    def setUp(self):
        self.data_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "TestData",
            "integration",
        )
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

    def tearDown(self):
        pass

    def test_01_WorkflowRunner_ssw_minimal(self):
        reporter_config = {
            "report_name": "test_01_WorkflowRunner_ssw_minimal",
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
        workflow_modules = ssw.minimal(
            filepath=os.path.join(
                self.data_folder,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            )
        )
        # remove rcc, as the test dataset is too small
        workflow_modules = workflow_modules[:-2]

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules
        )

        wr.run()

        # clean up
        shutil.rmtree(wr.result_folder)
        wr.confluencereporter.ci.delete_page(wr.reporter_config["report_name"])

    def test_02_AggregationWorkflowRunner_saw_align_channels(self):
        reporter_config = {
            "report_name": "test_02_AWR_saw_align_channels",
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
        filepaths = [
            os.path.join(
                self.data_folder,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            ),
            os.path.join(
                self.data_folder,
                "3C_30px_1kframes_shifted_1",
                "3C_30px_1kframes_shifted_MMStack_Pos0.ome.tif",
            ),
        ]
        agg_workflow = saw.minimal_channel_align(filepaths=filepaths)
        # remove rcc, as the test dataset is too small
        agg_workflow["single_dataset_modules"] = [
            agg_workflow["single_dataset_modules"][i] for i in [0, 1, 2, 4]
        ]
        # agg_workflow['single_dataset_modules'][0][1][
        #     "load_camera_info"] = True
        agg_workflow["aggregation_modules"][0][1]["filepaths"] = (
            "$$get_prior_result",
            "all_results, single_dataset, $$all, "
            + "03_save_single_dataset, "
            + "filepath",
        )

        awr = AggregationWorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, agg_workflow
        )

        awr.run()

        # clean up
        shutil.rmtree(awr.result_folder)
        ci = ConfluenceInterface(
            base_url=self.confluence_url,
            space_key=self.confluence_space,
            parent_page_title=self.confluence_page,
            token=self.confluence_token,
        )
        for repname in awr.cpage_names:
            ci.delete_page(repname)

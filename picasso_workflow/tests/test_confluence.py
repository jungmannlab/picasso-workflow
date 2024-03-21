#!/usr/bin/env python
"""
Module Name: test_confluence.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: Test the module confluence.py
"""
import logging
import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np

from picasso_workflow import confluence


logger = logging.getLogger(__name__)


class Test_A_ConfluenceInterface(unittest.TestCase):

    def setUp(self):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

        self.testpgtitle = "mytestpage"
        self.bodytxt = "mybodytext"

    def tearDown(self):
        pass

    def instantiate_confluence_interface(self):
        return confluence.ConfluenceInterface(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            self.confluence_token,
        )

    # @unittest.skip("")
    def test_01_interface_01_all(self):
        logger.debug("testing all inferface")
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.confluence_page)
        assert pgtitle == self.confluence_page

        pgv = ci.get_page_version(self.confluence_page)
        logger.debug(f"page version: {pgv}")

        pgbdy = ci.create_page(self.testpgtitle, self.bodytxt)
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)

        pgbdy = ci.get_page_body(self.testpgtitle)
        assert pgbdy == self.bodytxt

        att_id = ci.upload_attachment(
            pgid,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testimg.png",
            ),
        )
        ci.update_page_content_with_image_attachment(
            pgtitle, pgid, "testimg.png"
        )
        logger.debug(f"successfully uploaded attachment with id {att_id}")

        ci.update_page_content(pgtitle, pgid, "body update")

        att_id = ci.upload_attachment(
            pgid,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testmov.mp4",
            ),
        )
        ci.update_page_content_with_movie_attachment(
            pgtitle, pgid, "testmov.mp4"
        )
        logger.debug(f"successfully uploaded attachment with id {att_id}")

        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        logger.debug(f"Deleting page {pgid}, {pgtitle}")
        ci.delete_page(pgid)


# @unittest.skip('')
class Test_B_ConfluenceReporter(unittest.TestCase):

    @patch("picasso_workflow.confluence.ConfluenceInterface")
    def setUp(self, mock_cfi):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

        report_name = "my test report"

        # Mock the ConfluenceInterface to avoid Confluence interaction
        mock_instance = MagicMock()
        mock_instance.create_page.return_value = 534
        mock_instance.update_page_content.return_value = None
        mock_instance.get_page_properties.return_value = 123, "titleofhtepage"
        mock_cfi.return_value = mock_instance

        self.cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_token,
        )
        # self.cr.ci.upda

    def tearDown(self):
        pass

    # @unittest.skip("")
    def test_01_load_dataset_movie(self):

        pars_load = {
            "filename": "my test file location",
            "save_directory": "my test save directory",
        }
        results_load = {
            "start time": "now",
            "picasso version": "0.0.0",
            "movie.shape": (40000, 2048, 1024),
            "duration": 10.2,
            "sample_movie": {
                "sample_frame_idx": [1, 6, 11],
                "filename": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "TestData",
                    "confluence",
                    "testmov.mp4",
                ),
            },
        }
        self.cr.load_dataset_movie(0, pars_load, results_load)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def test_02_identify(self):
        parameters = {
            "min_gradient": 10000,
            "box_size": 7,
        }
        results = {
            "start time": "now",
            "duration": 16.4,
            "num_identifications": 23237,
        }
        self.cr.identify(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def test_03_localize(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 16.4,
            "locs_columns": ("x", "y", "photons"),
        }
        self.cr.localize(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def test_04_undrift_rcc(self):
        parameters = {
            "dimensions": ["x", "y"],
            "segmentation": 1000,
        }
        results = {
            "start time": "now",
            "message": "This is my message to you.",
            "duration": 27.4,
        }
        self.cr.undrift_rcc(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def test_05_manual(self):
        parameters = {
            "prompt": "Do something.",
            "filename": "abc.png",
            "success": False,
        }
        results = {
            "start time": "now",
            "message": "This is my message to you.",
        }
        self.cr.manual(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def test_06_summarize_dataset(self):
        parameters = {"methods": {"nena": {"inputpar": "a"}}}
        results = {"nena": {"best_vals": (3, 5, 7), "res": 1.23}}
        self.cr.summarize_dataset(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def test_07_save_single_dataset(self):
        parameters = {}
        results = {"start time": "now", "filepath": "/path/to/my/file"}
        self.cr.save_single_dataset(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def test_08_load_datasets_to_aggregate(self):
        parameters = {}
        results = {
            "start time": "now",
            "filepaths": ["/path/to/my/file", "/and/the/other"],
            "tags": ["a", "b"],
        }
        self.cr.load_datasets_to_aggregate(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def test_09_align_channels(self):
        parameters = {}
        results = {
            "start time": "now",
            "shifts": np.array([[3, 4], [2, 3], [1, 2]]),
        }
        self.cr.align_channels(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)


# @unittest.skip('')
class Test_C_ConfluenceReporter(Test_B_ConfluenceReporter):
    """This time really use the ConfluenceInterface, no mocking."""

    def setUp(self):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")

        report_name = "my test report"

        self.cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_token,
        )

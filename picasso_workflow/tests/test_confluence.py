#!/usr/bin/env python
"""
Module Name: test_confluence.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: Test the module confluence.py
"""
import logging
import unittest
import os

from picasso_workflow import confluence


logger = logging.getLogger(__name__)


class TestConfluence(unittest.TestCase):

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

    @unittest.skip("")
    def test_01_interface_01_all(self):
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
        ci.delete_page(pgid)

    @unittest.skip("")
    def test_02_reporter_01_all(self):
        report_name = "my test report"
        cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_token,
        )

        pars_load = {
            "filename": "my test file location",
            "save_directory": "my test save directory",
        }
        results_load = {
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
        cr.load_dataset(0, pars_load, results_load)

        # pgid, pgtitle = ci.get_page_properties(cr.report_page_name)

        # cr.ci.delete_page(pgid)

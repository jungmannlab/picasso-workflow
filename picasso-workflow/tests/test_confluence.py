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
        self.confluence_url = os.getenv('TEST_CONFLUENCE_URL')
        self.confluence_token = os.getenv('TEST_CONFLUENCE_TOKEN')
        self.confluence_space = os.getenv('TEST_CONFLUENCE_SPACE')
        self.confluence_page = os.getenv('TEST_CONFLUENCE_PAGE')

        self.testpgtitle = 'mytestpage'
        self.bodytxt = 'mybodytext'

    def tearDown(self):
        pass

    def instantiate_confluence_interface(self):
        return confluence.ConfluenceInterface(
            self.confluence_url, self.confluence_space,
            self.confluence_page, self.confluence_token)

    def test_01_interface_01_Instatiation(self):
        ci = self.instantiate_confluence_interface()

    def test_01_interface_02_get_page_info(self):
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.confluence_page)
        assert pgtitle == self.confluence_page

    def test_01_interface_03_get_page_version(self):
        ci = self.instantiate_confluence_interface()
        pgv = ci.get_page_version(self.confluence_page)

    def test_01_interface_04_create_page(self):
        ci = self.instantiate_confluence_interface()
        pgbdy = ci.create_page(self.testpgtitle, self.bodytxt)

    def test_01_interface_05_get_page_body(self):
        ci = self.instantiate_confluence_interface()
        pgbdy = ci.get_page_body(self.testpgtitle)
        assert pgbdy == self.bodytxt

    def test_01_interface_06_upload_attachment(self):
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        att_id = ci.upload_attachment(
            pgid, os.path.join('TestData', 'Confluence', 'testimg.png'))

    def test_01_interface_07_update_page_content(self):
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        ci.update_page_content(pgid, pgtitle, 'body update')
        
    def test_01_interface_07_update_page_content_with_movie_attachment(self):
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        att_id = ci.upload_attachment(
            pgid, pgtitle, os.path.join('TestData', 'Confluence', 'testmov.mp4'))

    def test_01_interface_08_delete_page(self):
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        ci.delete_page(pgid)

    def test_02_reporter_01_all(self):
        report_name = 'my test report'
        cr = confluence.ConfluenceReporter(
            self.confluence_url, self.confluence_space,
            self.confluence_page, report_name, self.confluence_token)

        pars_load = {
            'filename': 'my test file location',
            'save_directory': 'my test save directory',
        }
        results_load = {
            'picasso version': '0.0.0',
            'movie.shape': (40000, 2048, 1024),
            'duration': 10.2,
            'sample_movie': {
                'sample_frame_idx': [1, 6, 11],
                'filename': os.path.join('TestData', 'Confluence', 'testmov.mp4')
            }
        }
        cr.load_dataset(pars_load, results_load)

        pgid, pgtitle = ci.get_page_properties(cr.report_page_name)

        cr.ci.delete_page(pgid)


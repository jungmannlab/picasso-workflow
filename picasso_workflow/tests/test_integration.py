#!/usr/bin/env python
"""
Module Name: test_integration.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the integration of the package: run complete
    analysis workflows on minimal data.
"""
import os
import logging
import unittest


logger = logging.getLogger(__name__)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'TestData', 'integration')

    def tearDown(self):
        pass

    def test_01(self):
        assert False

#!/usr/bin/env python
"""
Module Name: test_workflow.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the module workflow.py
"""
import os
import shutil
import logging
import unittest
from unittest.mock import patch

from picasso_workflow.workflow import (
    WorkflowRunner, AggregationWorkflowRunner)


logger = logging.getLogger(__name__)


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'TestData', 'workflow')

    def tearDown(self):
        pass

    @patch('picasso_workflow.workflow.ParameterCommandExecutor')
    def test_01_WorkflowRunner_init(self, mock_pce):
        wr = WorkflowRunner()
        assert wr.results == {}

    @patch('picasso_workflow.workflow.ConfluenceReporter')
    @patch('picasso_workflow.workflow.AutoPicasso')
    @patch('picasso_workflow.workflow.ParameterCommandExecutor')
    def test_02_WorkflowRunner_from_config(self, mock_pce, mock_ap, mock_cr):
        reporter_config = {'report_name': 'myreport'}
        analysis_config = {'result_location': self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules)
        assert wr.results == {}

        # created a folder upon initialization. remove it.
        shutil.rmtree(wr.result_folder)

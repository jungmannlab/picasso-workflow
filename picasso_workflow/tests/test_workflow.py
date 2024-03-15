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
from unittest.mock import patch, MagicMock

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

    @patch('picasso_workflow.workflow.ConfluenceReporter', MagicMock)
    @patch('picasso_workflow.workflow.AutoPicasso', MagicMock)
    @patch('picasso_workflow.workflow.ParameterCommandExecutor', MagicMock)
    def test_02_WorkflowRunner_from_config(self):
        reporter_config = {
            'report_name': 'myreport',
            'ConfluenceReporter': {'a': 0}}
        analysis_config = {'result_location': self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules)
        assert wr.results == {}
        logger.debug(wr.autopicasso)
        logger.debug(wr.confluencereporter)

        # created a folder upon initialization. remove it.
        shutil.rmtree(wr.result_folder)

    @patch('picasso_workflow.workflow.ConfluenceReporter', MagicMock)
    @patch('picasso_workflow.workflow.AutoPicasso', MagicMock)
    @patch('picasso_workflow.workflow.ParameterCommandExecutor', MagicMock)
    def test_03_WorkflowRunner_save_load(self):
        reporter_config = {
            'report_name': 'myreport',
            'ConfluenceReporter': {'a': 0}}
        analysis_config = {'result_location': self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules)
        wr.save(self.results_folder)

        wr2 = WorkflowRunner.load(self.results_folder)

        # clean up
        # shutil.rmtree(wr.result_folder)
        shutil.rmtree(wr2.result_folder)
        os.remove(os.path.join(self.results_folder, 'WorkflowRunner.yaml'))

    # @unittest.skip('')
    @patch('picasso_workflow.workflow.ConfluenceReporter', MagicMock)
    @patch('picasso_workflow.workflow.AutoPicasso', MagicMock)
    @patch('picasso_workflow.workflow.ParameterCommandExecutor', MagicMock)
    def test_04_WorkflowRunner_save_load(self):
        reporter_config = {
            'report_name': 'myreport',
            'ConfluenceReporter': {'a': 0}}
        analysis_config = {'result_location': self.results_folder}
        workflow_modules = []

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules)
        wr.autopicasso.my_module = lambda i, p: ({}, {})

        wr.call_module('my_module', 0, {'parameter0': 1})

        shutil.rmtree(wr.result_folder)

    # @unittest.skip('')
    @patch('picasso_workflow.workflow.WorkflowRunner.call_module')
    @patch('picasso_workflow.workflow.ConfluenceReporter', MagicMock)
    @patch('picasso_workflow.workflow.AutoPicasso', MagicMock)
    @patch('picasso_workflow.workflow.ParameterCommandExecutor', MagicMock)
    def test_05_WorkflowRunner_run(self, mock_call_module):
        reporter_config = {
            'report_name': 'myreport',
            'ConfluenceReporter': {'a': 0}}
        analysis_config = {'result_location': self.results_folder}
        workflow_modules = [('load_dataset', {'b': 3})]

        wr = WorkflowRunner.config_from_dicts(
            reporter_config, analysis_config, workflow_modules)

        wr.run()

        shutil.rmtree(wr.result_folder)

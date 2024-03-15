#!/usr/bin/env python
"""
Module Name: workflow.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This module implements the class ReportingAnalyzer,
    which orchestrates picasso analysis and confluence reporting
"""
import os
from datetime import datetime
import logging
import inspect
import yaml

from picasso_workflow.analyse import AutoPicasso, AutoPicassoError
from picasso_workflow.confluence import ConfluenceReporter
from picasso_workflow.util import (
    AbstractModuleCollection, correct_path_separators,
    ParameterCommandExecutor, ParameterTiler)


logger = logging.getLogger(__name__)


class AggregationWorkflowRunner():
    """Often, workflows have to be separated into separate
    'sub' workflows, e.g. when multiple DNA-PAINT datasets
    are to be evaluated, and then aggregated. This is what this
    class aims to do
    """

    def __init__(self, use_prefix=True):
        if use_prefix:
            self.prefix = datetime.now().strftime('%y%m%d-%H%M')
        else:
            self.prefix = ''
        self.continue_workflow = False
        self.sgl_workflow_locations = []

    @classmethod
    def config_from_dicts(
            cls, reporter_config, analysis_config, aggregation_workflow,
            use_prefix=True):
        """To keep flexibility for initialization methods, this is not
        done in __init__. This way in the future, we can instantiate
        by providing config file names, retrieving config and parameters
        via a web API, or such.
        Args:
            reporter_config : dict
                configuration of the reporter, for now Confluence reporter
            analysis_config : dict
                general analysis configuration
            aggregation_workflow : dict
                the workflow modules to run, which need to be separated into
                individual runs. Keys:
                single_dataset_tileparameters : dict
                    describes the parameters that need to be adjusted for every
                    individual single data set analysis
                single_dataset_modules : list of tuples
                    (workflow_modules of WorkflowRunner)
                    describes the modules run for the analysis of the
                    individual datasets
                aggregation_modules : list of tuples
                    (workflow_modules of WorkflowRunner)
                    describes the modules run for the aggregation analysis
                    (e.g. labeling efficiency, RESI, ..)
            use_prefix : bool
                whether to add the date-time tag to the results folder.
        """
        if (sgltilepars := aggregation_workflow.get(
                'single_dataset_tileparameters')) is None:
            raise KeyError(
                '''aggregation_workflow missing
                "single_dataset_tileparameters".''')
        instance = cls(use_prefix)
        instance.parameter_tiler = ParameterTiler(instance, sgltilepars)
        instance.all_results = {
            'single_dataset': [None] * instance.parameter_tiler.ntiles,
            'aggregation': None}
        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        # set date and time to report name
        if instance.use_prefix:
            report_name = (
                reporter_config['report_name'] + '_' + instance.prefix[:-1])
        else:
            report_name = reporter_config['report_name']
        # reporter_config['report_name'] = report_name
        # create analysis result directory
        instance.result_folder = os.path.join(
            analysis_config['result_location'], report_name)
        try:
            os.mkdir(instance.result_folder)
        except FileExistsError:
            pass

        instance.aggregation_workflow = aggregation_workflow
        return instance

    def run(self):
        """individualize the aggregation workflow and run.
        """
        # First, run the individual analysis
        sgl_ds_workflow_parameters = self.aggregation_workflow[
            'single_dataset_modules']
        individual_parametersets, tags = self.parameter_tiler.run(
            sgl_ds_workflow_parameters)
        report_name = self.reporter_config['report_name']
        sgl_wkfl_reporter_config = self.reporter_config.copy()
        sgl_wkfl_analysis_config = self.analysis_config.copy()
        for i, (parameter_set, tag) in enumerate(
                zip(individual_parametersets, tags)):
            sgl_wkfl_reporter_config['report_name'] = report_name + f'_{i:02d}'
            if tag:
                sgl_wkfl_reporter_config['report_name'] += f'_{tag}'
            # sgl_wkfl_analysis_config['result_location'] = os.path.join(
            #     self.result_folder, sgl_wkfl_reporter_config['report_name'])
            if self.continue_workflow:
                try:
                    wr = WorkflowRunner.load(os.path.join(
                        self.result_folder,
                        sgl_wkfl_reporter_config['report_name']))
                except:
                    wr = WorkflowRunner.config_from_dicts(
                        sgl_wkfl_reporter_config, sgl_wkfl_analysis_config,
                        parameter_set, use_prefix=False)
            else:
                wr = WorkflowRunner.config_from_dicts(
                    sgl_wkfl_reporter_config, self.analysis_config.copy(),
                    parameter_set, use_prefix=False)
            wr.run(contd=self.continue_workflow)
            self.all_results['single_dataset'][i] = wr.results
            self.sgl_workflow_locations.append(wr.result_folder)
            self.save_results(self.result_folder)

        # Then, run the aggregation workflow
        pce = ParameterCommandExecutor(self)
        parameters = pce.run(self.aggregation_workflow['aggregation_modules'])
        reporter_config = self.reporter_config.copy()
        reporter_config['report_name'] = (
            reporter_config['report_name'] + '_aggregation')
        wr = WorkflowRunner.config_from_dicts(
            reporter_config, self.analysis_config, parameters)
        wr.run()
        self.all_results['aggregation'] = wr.results

    def save(self, dirn='.'):
        """Save the current config and results into
        'WorkflowRunnerResults.yaml' in the given directory.
        Args:
            dirn : str
                the directory to save into
        """
        fn = os.path.join(dirn, 'WorkflowRunnerResults.yaml')
        data = {
            'sgl_workflow_locations': self.sgl_workflow_locations,
            'all_results': self.all_results,
            'prefix': self.prefix,
            'reporter_config': self.reporter_config,
            'analysis_config': self.analysis_config,
            'aggregation_workflow': self.aggregation_workflow,
        }
        with open(fn, 'w') as f:
            yaml.dump(data, f)

    def load(self, dirn='.'):
        """Load instance from a "WorkflowRunnerResults.yaml" file.
        Args:
            dirn : str
                the directory to load from
        """
        fn = os.path.join(dirn, 'WorkflowRunnerResults.yaml')
        with open(fn, 'w') as f:
            data = yaml.loads(f)
        self.prefix = data['prefix']
        self.all_results = data['all_results']
        self.reporter_config = data['reporter_config']
        self.analysis_config = data['analysis_config']
        self.aggregation_workflow = data['aggregation_workflow']
        self.sgl_workflow_locations = data['sgl_workflow_locations']
        self.continue_workflow = True


class WorkflowRunner():
    """Runs a workflow, defined as a sequence of modules, which
    are worked on and their results published on Confluence.

    Currently supported usage:
    ra, ac, wm = {}, {}, {}
    wr = WorkflowRunner.config_from_dicts(rc, ac, wm)
    wr.run()
    """

    def __init__(self, use_prefix=True):
        self.use_prefix = use_prefix
        if use_prefix:
            self.prefix = datetime.now().strftime('%y%m%d-%H%M')
        else:
            self.prefix = ''

        self.parameter_command_executor = ParameterCommandExecutor(self)
        self.results = {}

    @classmethod
    def config_from_dicts(
            cls, reporter_config, analysis_config, workflow_modules,
            use_prefix=True):
        """To keep flexibility for initialization methods, this is not
        done in __init__. This way in the future, we can instantiate
        by providing config file names, retrieving config and parameters
        via a web API, or such.
        Args:
            reporter_config : dict
                configuration of the reporter, for now Confluence reporter
            analysis_config : dict
                general analysis configuration
            workflow_modules : list of tuples
                the workflow modules to run
            use_prefix : bool
                whether to add the date-time tag to the results folder.
        """
        instance = cls(use_prefix)
        # set date and time to report name
        if instance.use_prefix:
            report_name = (
                reporter_config['report_name'] + '_' + instance.prefix)
        else:
            report_name = reporter_config['report_name']
        reporter_config['report_name'] = report_name

        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        instance._initialize_reporter(reporter_config)
        instance._initialize_analysis(analysis_config, report_name)
        instance.workflow_modules = workflow_modules
        return instance

    def _initialize_analysis(self, analysis_config, report_name):
        """Initializes the Analysis worker.
        """
        logger.debug('Initializing Analysis.')
        # create analysis result directory
        self.result_folder = os.path.join(
            analysis_config.pop('result_location'), report_name)
        try:
            os.mkdir(self.result_folder)
        except FileExistsError:
            pass

        self.autopicasso = AutoPicasso(
            self.result_folder, analysis_config)

    def _initialize_reporter(self, reporter_config):
        """Initializes the reporter, documenting the analysis.
        """
        logger.debug('Initializing Reporter.')
        self.report_name = reporter_config['report_name']
        if init_kwargs := reporter_config.get('ConfluenceReporter'):
            self.confluencereporter = ConfluenceReporter(**init_kwargs)

    def run(self, contd=False):
        """
        Args:
            contd : bool
                set this True after loading results from file and continuing
                with after the last previously executed module
        """
        # first, check whether all modules are actually implemented
        available_modules = inspect.getmembers(AbstractModuleCollection)
        available_modules = [
            name for name, _ in available_modules
            if inspect.ismethod(_) or inspect.isfunction(_)]
        logger.debug(f'Available modules: {str(available_modules)}')
        for module_name, module_parameters in self.workflow_modules:
            if module_name not in available_modules:
                raise NotImplementedError(
                    f'Requested module {module_name} not implemented.')

        # now, run the modules
        for i, (module_name, module_parameters
                ) in enumerate(self.workflow_modules):
            # all modules are called with iteration and parameter dict
            # as arguments
            module_parameters = self.parameter_command_executor.run(
                module_parameters)
            self.call_module(module_name, i, module_parameters)
            self.save()

    ##########################################################################
    # UTIL FUNCTIONS
    ##########################################################################

    def get_prefixed_filename(self, filename):
        """
        """
        return os.path.join(self.savedir, self.prefix + filename)

    def save(self, dirn='.'):
        """Save the current results into 'WorkflowRunnerResults.yaml' in the
        given directory.
        Args:
            dirn : str
                the directory to save into
        """
        filepath = os.path.join(dirn, 'WorkflowRunnerResults.yaml')
        data = {
            'results': self.results,
            'reporter_config': self.reporter_config,
            'analysis_config': self.analysis_config,
            'workflow_modules': self.workflow_modules
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, dirn='.'):
        """Load the results from a "WorkflowRunnerResults.yaml" file.
        Args:
            dirn : str
                the directory to load from
        """
        filepath = os.path.join(dirn, 'WorkflowRunnerResults.yaml')
        with open(filepath, 'w') as f:
            data = yaml.loads(f)
        instance = cls()
        instance.results = data['results']
        instance.reporter_config = data['reporter_config']
        instance.analysis_config = data['analysis_config']
        instance.workflow_modules = data['workflow_modules']
        report_name = instance.reporter_config['report_nape']
        instance._initialize_analysis(instance.analysis_config, report_name)
        instance._initialize_reporter(instance.reporter_config)
        return instance

    ##########################################################################
    # MODULES
    ##########################################################################

    def call_module(self, fun_name, i, parameters):
        """At the level of the WorkflowRunner, all modules are processed the
        same way:
        first, the analysis is performed (by calling the module in
        autopicasso), and then the results are reported (by calling the module
        in confluencereporter). Therefore, this unified call_module function
        can do the job, instead of writing separate methods here for all
        modules.

        Args:
            fun_name : str
                the function (module) name.
            i : int
                the index of the module in the workflow
            parameters : dict
                the module parameters
        """
        key = f'{i:02d}_{fun_name}'
        logger.debug(f'Working on {key}')
        fun_ap = getattr(self.autopicasso, fun_name)
        try:
            parameters, self.results[key] = fun_ap(i, parameters)
        except AutoPicassoError as e:
            logger.error(e)
        fun_cr = getattr(self.confluencereporter, fun_name)
        fun_cr(i, parameters, self.results[key])


def get_ra():
    # fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack_1.tif'
    fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack.tif'
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'test page'
    report_name = 'analysis_report'
    return ReportingAnalyzer(fn, base_url, space_key, parent_page_title, report_name)

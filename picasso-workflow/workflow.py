#!/usr/bin/env python
"""
Module Name: workflow.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This module implements the class ReportingAnalyzer, which orchestrates
    picasso analysis and confluence reporting
"""
import os
import re
from datetime import datetime
import logging
import inspect

from picasso_workflow.analyse import AutoPicasso
from picasso_workflow.confluence import ConfluenceReporter
from picasso_workflow.util import AbstractModuleCollection, correct_path_separators


logger = logging.getLogger(__name__)


class WorkflowRunner(AbstractModuleCollection):
    """Runs a workflow, defined as a sequence of modules, which
    are worked on and their results published on Confluence.
    Inherits AbstractModuleCollection to ensure all implemented 
    modules are supported. 

    Currently supported usage:
    ra, ac, wm = {}, {}, {}
    wr = WorkflowRunner.config_from_dicts(rc, ac, wm)
    wr.run()
    """
    def __init__(self):
        self.prefix = datetime.now().strftime('%y%m%d-%H%M_')

    @classmethod
    def config_from_dicts(cls, reporter_config, analysis_config, workflow_modules):
        """To keep flexibility for initialization methods, this is not
        done in __init__. This way in the future, we can instantiate
        by providing config file names, retrieving config and parameters
        via a web API, or such.
        """
        instance = cls()
        # set date and time to report name
        report_name = reporter_config['report_name'] + '_' + self.prefix[:-1]
        reporter_config['report_name'] = report_name

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
        os.mkdir(self.result_folder)

        self.autopicasso = AutoPicasso(
            self.result_folder, analysis_config)

    def _initialize_reporter(self, reporter_config):
        """Initializes the reporter, documenting the analysis.
        """
        logger.debug('Initializing Reporter.')
        self.report_name = report_name
        if init_kwargs := reporter_config.get('ConfluenceReporter'):
            self.confluencereporter = ConfluenceReporter(**init_kwargs)

    def run(self):
        # first, check whether all modules are actually implemented
        available_modules = members = inspect.getmembers(AbstractModuleCollection)
        availbable_modules = [
            name for name, _ in available_modules
            if inspect.ismethod(_) or inspect.isfunction(_)]
        logger.debug(f'Available modules: {str(available_modules)}')
        for module_name, module_parameters in self.workflow_modules:
            if module_name not in available_modules:
                raise NotImplementedError(f'Requested module {module_name} not implemented.')

        # now, run the modules
        for module_name, module_parameters in self.workflow_modules:
            module_fun = getattr(self, module_name)
            logger.debug(f'Running module {module_name}.')
            # all modules are called with one dict as argument
            module_parameters = self.execute_parameter_commands(module_parameters)
            module_fun(module_parameters)

    #################################################################################
    #### UTIL FUNCTIONS
    #################################################################################

    def get_prefixed_filename(self, filename):
        """
        """
        return os.path.join(self.savedir, self.prefix + filename)

    def execute_parameter_commands(self, parameters):
        """Scan a parameter set for commands to execute prior to module
        execution.
        commands: '$get_prior_result'
        Args:
            parameters : dict
                the parameters for a module
        """
        return scan_dict(parameters)

    def scan_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = scan_dict(v)
            elif isinstance(v, list):
                d[k] = scan_list(v)
            elif isinstance(v, tuple):
                d[k] = scan_tuple(v)
        return d

    def scan_list(self, l):
        for i, it in enumerate(l):
            if isinstance(it, dict):
                l[i] = scan_dict(it)
            elif isinstance(it, list):
                l[i] = scan_list(it)
            elif isinstance(it, tuple):
                l[i] = scan_tuple(it)
        return l

    def scan_tuple(self, t):
        if len(t) == 2 and isinstance(t[0], str) and t[0][0]=='$':
            # this is a parameter command
            if t[0] == '$get_prior_result':
                logger.debug(f'Getting prior result from {t[1]}.')
                res = self.get_prior_result(t[1])
                logger.debug(f'Prior result is {res}.')
            # elif add more parameter commands
            return res
        else:
            # it's just a normal tuple
            tout = []
            for i, it in enumerate(t):
                if isinstance(it, dict):
                    tout[i] = scan_dict(it)
                elif isinstance(it, list):
                    tout[i] = scan_list(it)
                elif isinstance(it, tuple):
                    tout[i] = scan_tuple(it)
            return tuple(tout)


    def get_prior_result(self, locator):
        """In some cases, input parameters for a module should be taken from
        prior results. This is performed here
        Args:
            locator : str
                the chain of attributes for finding the prior result, comma separated.
                They all need to be obtainable with getattr, starting from this class
                e.g. "results_load, sample_movie, sample_frame_idx"
                obtains self.results_load['sample_movie']['sample_frame_idx']
        Returns:
            the last attribute in the chain.
        """
        root_att = self
        for att_name in locator.split(','):
            try:
                root_att = getattr(root_att, att_name.strip())
            except AttributeError as e:
                logger.error(f'Could not get attribute {att_name} from {str(root_att)}.')
                raise e
        return root_att


    #################################################################################
    #### MODULES
    #################################################################################

    def load(self, parameters):
        parameters, self.results_load = self.autopicasso.load(parameters)
        self.confluencereporter.load(parameters, self.results_load)
        logger.debug('Loaded DNA-PAINT image data.')

    def identify(self, parameters):
        parameters, self.results_identify = self.autopicasso.identify(parameters)
        self.confluencereporter.identify(parameters, self.results_identify)
        logger.debug('Identified spots.')

    def localize(self, parameters):
        parameters, self.results_localize = self.autopicasso.localize(parameters)
        self.confluencereporter.localize(parameters, self.results_localize)
        logger.debug('Localized spots')

    def undrift_rcc(self, parameters):
        try:
            parameters, self.results_undrift_rcc = self.autopicasso.undrift_rcc(parameters)
            self.confluencereporter.undrift_rcc(parameters, self.results_undrift_rcc)
            logger.debug('undrifted dataset')
        except UndriftError:
            max_segmentation = pars_undrift['segmentation']
            # initial segmentation
            pars_undrift['segmentation'] = int(
                pars_undrift['segmentation'] / 2**pars_undrift['max_iter_segmentations'])
            res_undrift = {
                'message': f'''
                    Undrifting did not work in {pars_undrift['max_iter_segmentations']} iterations
                    up to a segmentation of {max_segmentation}.''',
            }
            self.confluencereporter.undrift_rcc(pars_undrift, res_undrift)
            logger.error('Error in dataset undrifting')

    def describe(self, parameters):
        parameters, self.results_describe = self.autopicasso.describe(parameters)
        self.confluencereporter.describe(parameters, self.results_describe)
        logger.debug('Described dataset.')

    def save(self, parameters):
        self.autopicasso.save_locs()
        self.confluencereporter.save()


def get_ra():
    # fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack_1.tif'
    fn = 'R1to6_3C-20nm-10nm_231208-1547\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_1\\R1to6_3C-20nm-10nm_231208-1547_23-12-08_1547_prtclstep1_round_0-R1_NDTiffStack.tif'
    base_url = 'https://mibwiki.biochem.mpg.de'
    space_key = "~hgrabmayr"
    parent_page_title = 'test page'
    report_name = 'analysis_report'
    return ReportingAnalyzer(fn, base_url, space_key, parent_page_title, report_name)

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
import copy

from picasso_workflow.analyse import AutoPicasso, AutoPicassoError
from picasso_workflow.confluence import (
    ConfluenceReporter,
    ConfluenceInterface,
    ConfluenceInterfaceError,
)
from picasso_workflow.util import (
    AbstractModuleCollection,
    ParameterCommandExecutor,
    ParameterTiler,
    DictSimpleTyper,
)


# For loading yaml files
# Custom constructor for handling 'tag:yaml.org,2002:python/tuple'
def python_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


# # Register the custom constructors
yaml.constructor.SafeConstructor.add_constructor(
    "tag:yaml.org,2002:python/tuple", python_tuple_constructor
)


logger = logging.getLogger(__name__)


class AggregationWorkflowRunner:
    """Often, workflows have to be separated into separate
    'sub' workflows, e.g. when multiple DNA-PAINT datasets
    are to be evaluated, and then aggregated. This is what this
    class aims to do
    """

    def __init__(self, postfix=None):
        """
        Args:
            postfix : str, default None
                The postfix to use (for loading prior analyses).
                Format: %y%m%d-%H%M.
                If None, a new postfix is generated
        """
        if postfix:
            self.postfix = postfix
        else:
            self.postfix = datetime.now().strftime("%y%m%d-%H%M")
        self.continue_workflow = False
        self.sgl_workflow_locations = []
        self.cpage_names = []

    @classmethod
    def config_from_dicts(
        cls,
        reporter_config,
        analysis_config,
        aggregation_workflow,
        postfix=None,
        continue_previous_runner=False,
    ):
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
            postfix : str
                The postfix to use (for loading prior analyses).
                Format: %y%m%d-%H%M.
                If None, a new postfix is generated
            continue_previous_runner : bool, default False
                continue a previous analysis that aborted (e.g. because of a
                manual step). If no previous analysis exists in that folder,
                create a new one.
        """
        if continue_previous_runner:
            folder = analysis_config["result_location"]
            report_name = reporter_config["report_name"]
            postfix = cls._check_previous_runner(folder, report_name)
            logger.debug(f"Found postfix: {postfix}")
            if postfix is not None:
                report_name = report_name + "_" + postfix
                runner_folder = os.path.join(folder, report_name)
                instance = cls.load(runner_folder)
                return instance
        if (
            sgltilepars := aggregation_workflow.get(
                "single_dataset_tileparameters"
            )
        ) is None:
            raise KeyError(
                """aggregation_workflow missing
                "single_dataset_tileparameters"."""
            )
        instance = cls(postfix)
        instance.parameter_tiler = ParameterTiler(instance, sgltilepars)
        instance.all_results = {
            "single_dataset": [None] * instance.parameter_tiler.ntiles,
            "aggregation": None,
        }
        # set date and time to report name
        if instance.postfix:
            report_name = (
                reporter_config["report_name"] + "_" + instance.postfix
            )
        else:
            report_name = reporter_config["report_name"]
        if confluence_config := reporter_config.get("ConfluenceReporter"):
            instance._initialize_confluence_interface(**confluence_config)
            body_text = """<b>Aggregation analysis reslts</b>"""
            try:
                instance.ci.create_page(report_name, body_text)
            except ConfluenceInterfaceError:
                logger.debug(
                    "Error creating page, it already exists. Continuing"
                )
            reporter_config["ConfluenceReporter"][
                "parent_page_title"
            ] = report_name
            instance.cpage_names.append(report_name)

        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        # reporter_config['report_name'] = report_name
        # create analysis result directory
        instance.result_folder = os.path.join(
            analysis_config["result_location"], report_name
        )
        try:
            os.mkdir(instance.result_folder)
        except FileExistsError:
            pass

        instance.aggregation_workflow = aggregation_workflow
        return instance

    @classmethod
    def _check_previous_runner(cls, folder, report_name):
        """Check for a previous runner instance in the given location
        Args:
            folder : str
                the folder to look in
            report_name : str
                the name of the report
        Returns:
            postfix : str
                the postfix of the latest previous runner in that location
                if none are found, postfix is None
        """
        dirs = [
            it
            for it in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, it))
        ]
        dirs = [it for it in dirs if report_name in it]
        # find the latest runner
        latest_datetime = None
        latest_postfix = None
        for d in dirs:
            try:
                # cut out the postfix
                postfix_start = len(report_name) + 1
                postfix = d[postfix_start:]
                dt = datetime.strptime(postfix, "%y%m%d-%H%M")
            except Exception:
                continue
            if latest_datetime is None or latest_datetime < dt:
                latest_datetime = dt
                latest_postfix = postfix
        return latest_postfix

    def _initialize_confluence_interface(
        self, base_url, space_key, parent_page_title, token=None
    ):
        self.ci = ConfluenceInterface(
            base_url=base_url,
            space_key=space_key,
            parent_page_title=parent_page_title,
            token=token,
        )

    def run(self):
        """individualize the aggregation workflow and run."""
        # First, run the individual analysis
        sgl_ds_workflow_parameters = self.aggregation_workflow[
            "single_dataset_modules"
        ]
        individual_parametersets, tags = self.parameter_tiler.run(
            sgl_ds_workflow_parameters
        )
        report_name = self.reporter_config["report_name"]
        sgl_wkfl_reporter_config = copy.deepcopy(self.reporter_config)
        sgl_wkfl_analysis_config = copy.deepcopy(self.analysis_config)

        sgl_dataset_success = [None] * len(tags)
        for i, (parameter_set, tag) in enumerate(
            zip(individual_parametersets, tags)
        ):
            sgl_wkfl_reporter_config["report_name"] = (
                report_name + f"_sgl_{i:02d}"
            )
            if tag:
                sgl_wkfl_reporter_config["report_name"] += f"_{tag}"
            sgl_wkfl_analysis_config["result_location"] = self.result_folder
            # sgl_wkfl_analysis_config['result_location'] = os.path.join(
            #     self.result_folder, sgl_wkfl_reporter_config['report_name'])
            if self.continue_workflow:
                try:
                    logger.debug(
                        "loading WorkflowRunner from "
                        + os.path.join(
                            self.result_folder,
                            sgl_wkfl_reporter_config["report_name"]
                            + "_"
                            + self.postfix,
                        )
                    )
                    wr = WorkflowRunner.load(
                        os.path.join(
                            self.result_folder,
                            sgl_wkfl_reporter_config["report_name"]
                            + "_"
                            + self.postfix,
                        )
                    )
                except Exception:
                    logger.debug("loading did not work. creating from dict.")
                    wr = WorkflowRunner.config_from_dicts(
                        sgl_wkfl_reporter_config,
                        sgl_wkfl_analysis_config,
                        parameter_set,
                        postfix=self.postfix,
                    )
            else:
                logger.debug("not dontinuing workflow.starting new.")
                wr = WorkflowRunner.config_from_dicts(
                    sgl_wkfl_reporter_config,
                    sgl_wkfl_analysis_config,
                    parameter_set,
                    postfix=self.postfix,
                )
            self.cpage_names.append(wr.reporter_config["report_name"])
            sgl_dataset_success[i] = wr.run()
            self.all_results["single_dataset"][i] = wr.results
            self.sgl_workflow_locations.append(wr.result_folder)
            self.save(self.result_folder)

        if not all(sgl_dataset_success):
            msg = (
                "Not all single datasets were analysed successfully. "
                + "Therefore, no aggregation analysis is started."
            )
            logger.error(msg)
            raise WorkflowError(msg)

        # Then, run the aggregation workflow
        pce = ParameterCommandExecutor(
            self,
            map_dict=self.aggregation_workflow.get(
                "single_dataset_tileparameters"
            ),
            command_sign="$$",
        )
        parameters = pce.run(self.aggregation_workflow["aggregation_modules"])
        agg_reporter_config = copy.deepcopy(self.reporter_config)
        agg_reporter_config["report_name"] = (
            agg_reporter_config["report_name"] + "_aggregation"
        )
        agg_analysis_config = copy.deepcopy(self.analysis_config)
        agg_analysis_config["result_location"] = self.result_folder
        # try loading
        if self.continue_workflow:
            try:
                logger.debug(
                    "loading WorkflowRunner from "
                    + os.path.join(
                        self.result_folder,
                        agg_reporter_config["report_name"]
                        + "_"
                        + self.postfix,
                    )
                )
                wr = WorkflowRunner.load(
                    os.path.join(
                        self.result_folder,
                        agg_reporter_config["report_name"]
                        + "_"
                        + self.postfix,
                    )
                )
            except Exception:
                logger.debug("loading did not work. creating from dict.")
                wr = WorkflowRunner.config_from_dicts(
                    agg_reporter_config,
                    agg_analysis_config,
                    parameters,
                    postfix=self.postfix,
                )
        else:
            logger.debug("not dontinuing workflow.starting new.")
            wr = WorkflowRunner.config_from_dicts(
                agg_reporter_config,
                agg_analysis_config,
                parameters,
                postfix=self.postfix,
            )
        self.cpage_names.append(wr.reporter_config["report_name"])
        wr.run()
        self.all_results["aggregation"] = wr.results

    def save(self, dirn="."):
        """Save the current config and results into
        'WorkflowRunnerResults.yaml' in the given directory.
        Args:
            dirn : str
                the directory to save into
        """
        fp = os.path.join(dirn, "AggregationWorkflowRunner.yaml")
        data = {
            "sgl_workflow_locations": self.sgl_workflow_locations,
            "all_results": self.all_results,
            "postfix": self.postfix,
            "reporter_config": self.reporter_config,
            "analysis_config": self.analysis_config,
            "aggregation_workflow": self.aggregation_workflow,
        }
        with open(fp, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, dirn="."):
        """Load instance from a "WorkflowRunnerResults.yaml" file.
        Args:
            dirn : str
                the directory to load from
        """
        fp = os.path.join(dirn, "AggregationWorkflowRunner.yaml")
        with open(fp, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        instance = cls.config_from_dicts(
            data["reporter_config"],
            data["analysis_config"],
            data["aggregation_workflow"],
            data["postfix"],
        )
        instance.all_results = data["all_results"]
        instance.sgl_workflow_locations = data["sgl_workflow_locations"]
        instance.continue_workflow = True
        return instance


class WorkflowError(Exception):
    pass


class WorkflowRunner:
    """Runs a workflow, defined as a sequence of modules, which
    are worked on and their results published on Confluence.

    Currently supported usage:
    ra, ac, wm = {}, {}, {}
    wr = WorkflowRunner.config_from_dicts(rc, ac, wm)
    wr.run()
    """

    def __init__(self, postfix=None):
        """
        Args:
            postfix : str, default None
                The postfix to use (for loading prior analyses).
                Format: %y%m%d-%H%M.
                If None, a new postfix is generated
        """
        if postfix:
            self.postfix = postfix
        else:
            self.postfix = datetime.now().strftime("%y%m%d-%H%M")

        self.parameter_command_executor = ParameterCommandExecutor(self)
        self.results = {}

    @classmethod
    def config_from_dicts(
        cls,
        reporter_config,
        analysis_config,
        workflow_modules,
        postfix=None,
        continue_previous_runner=False,
    ):
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
            postfix : str, default None
                The postfix to use (for loading prior analyses).
                Format: %y%m%d-%H%M.
                If None, a new postfix is generated
            continue_previous_runner : bool, default False
                continue a previous analysis that aborted (e.g. because of a
                manual step). If no previous analysis exists in that folder,
                create a new one.
        """
        if continue_previous_runner:
            folder = analysis_config["result_location"]
            report_name = reporter_config["report_name"]
            postfix = cls._check_previous_runner(folder, report_name)
            if postfix is not None:
                report_name = report_name + "_" + postfix
                runner_folder = os.path.join(folder, report_name)
                instance = cls.load(runner_folder)
                return instance

        instance = cls(postfix)
        # set date and time to report name
        report_name = reporter_config["report_name"] + "_" + instance.postfix
        reporter_config["report_name"] = report_name

        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        instance._initialize_reporter(reporter_config)
        instance._initialize_analysis(analysis_config, report_name)
        instance.workflow_modules = workflow_modules
        return instance

    @classmethod
    def _check_previous_runner(cls, folder, report_name):
        """Check for a previous runner instance in the given location
        Args:
            folder : str
                the folder to look in
            report_name : str
                the name of the report
        Returns:
            postfix : str
                the postfix of the latest previous runner in that location
                if none are found, postfix is None
        """
        dirs = [
            it
            for it in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, it))
        ]
        dirs = [it for it in dirs if report_name in it]
        # find the latest runner
        latest_datetime = None
        latest_postfix = None
        for d in dirs:
            try:
                # cut out the postfix
                postfix_start = len(report_name) + 1
                postfix = d[postfix_start:]
                dt = datetime.strptime(postfix, "%y%m%d-%H%M")
            except Exception:
                continue
            if latest_datetime is None or latest_datetime < dt:
                latest_datetime = dt
                latest_postfix = postfix
        return latest_postfix

    def _initialize_analysis(self, analysis_config, report_name):
        """Initializes the Analysis worker."""
        logger.debug("Initializing Analysis.")
        # create analysis result directory
        self.result_folder = os.path.join(
            analysis_config.pop("result_location"), report_name
        )
        try:
            os.mkdir(self.result_folder)
        except FileExistsError:
            pass

        self.autopicasso = AutoPicasso(self.result_folder, analysis_config)

    def _initialize_reporter(self, reporter_config):
        """Initializes the reporter, documenting the analysis."""
        logger.debug("Initializing Reporter.")
        self.report_name = reporter_config["report_name"]
        if init_kwargs := reporter_config.get("ConfluenceReporter"):
            init_kwargs["report_name"] = self.report_name
            logger.debug(init_kwargs)
            self.confluencereporter = ConfluenceReporter(**init_kwargs)

    def run(self):
        """Run the analysis of the worfklow modules.
        Returns:
            success : bool
                whether all modulles ran through successfully.
        """
        # first, check whether all modules are actually implemented
        available_modules = inspect.getmembers(AbstractModuleCollection)
        available_modules = [
            name
            for name, _ in available_modules
            if inspect.ismethod(_) or inspect.isfunction(_)
        ]
        available_modules = [
            name for name in available_modules if name != "__init__"
        ]
        logger.debug(f"Available modules: {str(available_modules)}")
        for module_name, module_parameters in self.workflow_modules:
            if module_name not in available_modules:
                raise NotImplementedError(
                    f"Requested module {module_name} not implemented."
                )

        # now, run the modules
        all_previously_succeeded = True
        for i, (module_name, module_parameters) in enumerate(
            self.workflow_modules
        ):
            # # check whether the next module has been analysed already
            # if self.module_previously_analyzed(i + 1):
            #     # if it has, skip this. This way an aborted module
            #     # will be re-analyzed.
            #     logger.debug(
            #         f"""Module {i}, {module_name} has been previously
            #         analyzed. Skipping."""
            #     )
            #     continue
            if (
                all_previously_succeeded
                and self.module_previously_succeeded(i, module_name)
            ) and self.module_previously_analyzed(i):
                # if it has, skip this. This way an aborted module
                # will be re-analyzed.
                logger.debug(
                    f"""Module {i}, {module_name} has been previously
                    analyzed. Skipping."""
                )
                continue
            else:
                all_previously_succeeded = False
            # all modules are called with iteration and parameter dict
            # as arguments
            module_parameters = self.parameter_command_executor.run(
                module_parameters, curr_rootidx=i
            )
            success = self.call_module(module_name, i, module_parameters)
            if not success:
                break
            self.save(self.result_folder)
        else:
            success = True
        return success

    ##########################################################################
    # UTIL FUNCTIONS
    ##########################################################################

    def get_postfixed_filename(self, filename):
        """ """
        return os.path.join(self.savedir, self.postfix + filename)

    def save(self, dirn="."):
        """Save the current results into 'WorkflowRunnerResults.yaml' in the
        given directory.
        Args:
            dirn : str
                the directory to save into
        """
        pce = DictSimpleTyper(to_simple_type=True)
        filepath = os.path.join(dirn, "WorkflowRunner.yaml")
        data = {
            "results": pce.run(self.results),
            "reporter_config": pce.run(self.reporter_config),
            "analysis_config": pce.run(self.analysis_config),
            "workflow_modules": pce.run(self.workflow_modules),
        }
        logger.debug("saving data:")
        logger.debug(str(data))
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, dirn="."):
        """Load the results from a "WorkflowRunner.yaml" file.
        Args:
            dirn : str
                the directory to load from
        """
        filepath = os.path.join(dirn, "WorkflowRunner.yaml")
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        instance = cls()
        instance.results = data["results"]
        instance.reporter_config = data["reporter_config"]
        instance.analysis_config = data["analysis_config"]
        instance.analysis_config["result_location"] = os.path.join(dirn, "..")
        instance.workflow_modules = data["workflow_modules"]
        report_name = instance.reporter_config["report_name"]
        instance._initialize_analysis(instance.analysis_config, report_name)
        instance._initialize_reporter(instance.reporter_config)
        return instance

    def module_previously_analyzed(self, i):
        """Checks whether the module with index i has been analysed previously.
        If it has, a folder with its index prefixed has been created.
        Args:
            i : int
                the module index
        Returns:
            module_found : bool
                whether the folder corresponding to the module index was found
        """
        # via created directories:
        dirs = os.listdir(self.result_folder)
        dirs = [
            d
            for d in dirs
            if os.path.isdir(os.path.join(self.result_folder, d))
        ]
        prefix = f"{i:02d}_"
        module_found = any([d.startswith(prefix) for d in dirs])
        return module_found

    def module_previously_succeeded(self, i, module_name):
        """Check whether a module has previously succeeded, which must be
        saved in the results.
        Args:
            i : int
                the module index
            module_name : str
                the module name
        Returns:
            module_found : bool
                whether a previous module evaluation has succeeded
        """
        module_id = f"{i:02d}_{module_name}"
        logger.debug("looking for previous " + module_id)
        logger.debug(str(self.results.get(module_id, {})))
        logger.debug(
            str(self.results.get(module_id, {}).get("success", False))
        )
        return self.results.get(module_id, {}).get("success", False)

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
        Returns:
            success : bool
                whether the module ended successfully
        """
        key = f"{i:02d}_{fun_name}"
        logger.debug(f"Working on {key}")
        fun_ap = getattr(self.autopicasso, fun_name)
        try:
            parameters, self.results[key] = fun_ap(i, parameters)
        except AutoPicassoError as e:
            self.results[key]["success"] = False
            logger.error(e)
            raise e
        logger.debug(f"RESULTS: {self.results[key]}")
        fun_cr = getattr(self.confluencereporter, fun_name)
        try:
            fun_cr(i, parameters, self.results[key])
        except ConfluenceInterfaceError as e:
            logger.error(e)
        return self.results[key]["success"]

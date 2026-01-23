#!/usr/bin/env python
"""
Module Name: metaworkflow.py
Author: Heinrich Grabmayr
Initial Date: January 29, 2025
Description: This module implements functionality to do
    higher level analyses, e.g. running workflows on
    multiple conditions and cells, and aggregating over these.
"""
import os
from picasso_workflow import ON_CLUSTER
from picasso_workflow import AggregationWorkflowRunner, WorkflowRunner
from picasso_workflow import confluence, util
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd

# import logging
from loguru import logger
import time
from datetime import datetime
import hashlib
from picasso import io
import pathlib
from functools import reduce
import textwrap
import platform
import re
import abc
from picasso_workflow import CONFIG


# if ON_CLUSTER:
#     from mpi4py import MPI


# logger = logging.getLogger(__name__)


class PathParser:
    """A class to parse paths for multi-level picasso-workflow analysis
    The paths come in a dictionary with keys that have four underscores
    specifying the different analysis levels, and values that are the
    paths to the corresponding data. The goal of the PathParser is
    to re-structure the data in multilevel dicts, for the different
    analysis levels, and to adjust the paths to the current machine.
    """

    def __init__(self):
        # as loaded in picasso_workflow.__init__ from .env file

        # drivepaths = os.environ["DRIVEPATHS"]
        # self.drive_paths = {}
        # for machinepaths in drivepaths.split(";"):
        #     try:
        #         machine, paths = machinepaths.strip().split("::")
        #     except (IndexError, ValueError):
        #         continue
        #     paths = [p.strip() for p in paths.split(",")]
        #     self.drive_paths[machine] = paths

        self.drive_paths = CONFIG["Drivepaths"]
        logger.debug(f"drivepaths: {self.drive_paths}")

    def windows_path_to_curr_os(self, winpath, drive_map):
        """Convert a windows path to a path in the format for the current
        os. Replace the windows drive (e.g. 'X:' by a given drive)
        Args:
            winpath : str
                the windows-style path to convert
            drive_map : dict
                the map from windows drive (e.g. "W:") to posix drive
                (e.g. "/Volumes/pool-miblab4")
        """
        winpath = pathlib.PureWindowsPath(winpath)
        if winpath.drive:
            # absolute path
            drive = drive_map[winpath.drive]
            currospath = pathlib.Path(drive, *winpath.parts[1:])
        else:
            # relative path
            currospath = pathlib.Path(*winpath.parts)
        return str(currospath)

    def posix_path_to_curr_os(self, posixpath, drive_map):
        """Convert a windows path to a path in the format for the current
        os. Replace the windows drive (e.g. 'X:' by a given drive)
        Args:
            posixpath : str
                the posix-style path to convert
            drive_map : dict
                the map from windows drive (e.g. "W:") to posix drive
                (e.g. "/Volumes/pool-miblab4")
        """
        posixpath = pathlib.PurePosixPath(posixpath)
        if posixpath.drive:
            # absolute path
            drive = drive_map[posixpath.drive]
            # print('old drive: ', posixpath.drive)
            # print('new drive:' , drive)
            currospath = pathlib.Path(drive, *posixpath.parts[1:])
        else:
            # print('no drive')
            drivelen = len(pathlib.Path(list(drive_map.keys())[0]).parts)
            pseudodrive = pathlib.Path(*posixpath.parts[:drivelen])
            # print('pseudodrive:', pseudodrive)
            # print(drive_map)
            if drive := drive_map.get(str(pseudodrive)):
                # print('found drive:', drive)
                # found the drive after all
                currospath = pathlib.Path(drive, *posixpath.parts[drivelen:])
            else:
                # print("did not find a drive")
                # relative path
                currospath = pathlib.Path(*posixpath.parts)
        # print('new path:', currospath)
        return str(currospath)

    def check_path_style(self, path):
        """Check whether a path is windows or posix style by
        comparing the number of / and \
        """
        num_fwd = path.count("/")
        num_bwd = path.count("\\")
        is_posix = num_fwd > num_bwd
        return is_posix

    def check_machine(self, machine, pattern):
        """Checks a machine against a machine pattern, for example
        'mymachine6234' against 'mymachine6XXX', with X being a digit
        """
        regex = pattern.replace("X", r"\S")
        regex = f"^{regex}$"
        # print('machine', machine, 'pattern', pattern, 'regex', regex)
        return re.fullmatch(regex, machine) is not None

    def get_machine_drivepaths(self, machine):
        # print('drive paths', self.drive_paths)
        for pattern, paths in self.drive_paths.items():
            if self.check_machine(machine, pattern):
                return paths
        else:
            return None

    def convert_path(self, src_path, dest_machine):
        """Convert a path from a source machine style to a dest
        machine style and volume notation
        """
        # if dest_machine not in self.drive_paths.keys():
        #     raise ValueError(
        #         f"Machine {dest_machine} not defined in .env! \
        #         ({self.drive_paths.keys()})"
        #     )
        # print(CONFIG)
        # find current machine key
        if dest_machine is None:
            for dest_machine in self.drive_paths.keys():
                if self.check_machine(platform.node(), dest_machine):
                    break
        dest_paths = self.get_machine_drivepaths(dest_machine)
        # print('dest_machine', dest_machine, 'paths', dest_paths)

        for src_machine, drivepaths in self.drive_paths.items():
            src_on_machine = any([p in src_path for p in drivepaths])
            if src_on_machine:
                # src_machine has all drive paths defined
                break

        logger.debug(f"found src machine: {src_machine}")
        logger.debug(f"dest machine: {dest_machine}")

        drive_map = {}
        # for src_p, dest_p in zip(
        #     self.drive_paths[src_machine], self.drive_paths[dest_machine]
        # ):
        for src_p, dest_p in zip(self.drive_paths[src_machine], dest_paths):
            drive_map[src_p] = dest_p

        logger.debug(f"drive map is {drive_map}")

        # check whether input path is windows or posix style
        is_posix = self.check_path_style(src_path)
        if is_posix:
            dest_path = self.posix_path_to_curr_os(
                src_path, drive_map=drive_map
            )
        else:
            dest_path = self.windows_path_to_curr_os(
                src_path, drive_map=drive_map
            )
        return dest_path

    def parse_source(
        self, src_loc, receptors, dest_machine=None, underscores=[1, 0, 0]
    ):
        """
        Args:
            src_loc : string
                filepath to a yaml file defining a list (len 1) of dict:
                keys: strings with four "_"
                    (A_B_C_D: cell type/condition defined by A_B,
                     C: cell/img position ID, D: imaging target)
                values: paths to data
            receptors : list of str or int
                The targets to analyse, if they are the same in all datasets
                Or the number of targets to analyse, if their identities differ
            underscores : list of int
                the number of underscores in the different levels.
        """
        src_dict = io.load_info(src_loc)[0]
        # if dest_machine not in self.drive_paths.keys():
        #     raise ValueError(
        #         f"Machine {dest_machine} not defined in .env! \
        #         ({self.drive_paths.keys()})"
        #     )

        # find current machine key
        if dest_machine is None:
            for dest_machine in self.drive_paths.keys():
                if dest_machine in platform.node():
                    break
        dest_paths = self.get_machine_drivepaths(dest_machine)

        for src_machine, drivepaths in self.drive_paths.items():
            src_on_machine = []
            for srcp in src_dict.values():
                src_on_machine.append(any([p in srcp for p in drivepaths]))
            if all(src_on_machine):
                # src_machine has all drive paths defined
                break

        logger.debug(f"found src machine: {src_machine}")
        logger.debug(f"dest machine: {dest_machine}")

        drive_map = {}
        for src_p, dest_p in zip(self.drive_paths[src_machine], dest_paths):
            drive_map[src_p] = dest_p

        logger.debug(f"drive map is {drive_map}")

        # keys: [dataset]_[treatment]_[cell#]_[receptor]
        # turn into hierarchical dict
        # * lvl1 key: [dataset_treatment]
        # * lvl2 key: cell#
        # * values: list of receptors, according to list order above
        filepaths = {}
        targets = {}
        for k, v in src_dict.items():
            k_items = k.split("_")
            if len(k_items) != sum(underscores) + len(underscores):
                raise KeyError(
                    f"Dataset id {k} not valid for the number of underscores "
                    + f"in levels of {underscores}, separated by underscores."
                )
            k_lvls = []
            jcurr = 0
            if isinstance(receptors, list):
                n_targets = len(receptors)
            else:
                n_targets = receptors
            for i, nunder in enumerate(underscores):
                k_lvls.append("_".join(k_items[jcurr : jcurr + 1 + nunder]))
                jcurr += 1 + nunder
            dict_query = filepaths
            dict_query2 = targets
            for i, k_lvl in enumerate(k_lvls[:-1]):
                if k_lvl not in dict_query.keys():
                    if i < len(k_lvls) - 2:
                        dict_query[k_lvl] = {}
                        dict_query2[k_lvl] = {}
                    else:
                        dict_query[k_lvl] = [None] * n_targets
                        dict_query2[k_lvl] = [None] * n_targets
                dict_query = dict_query[k_lvl]
                dict_query2 = dict_query2[k_lvl]
            logger.debug(f"created filepaths {filepaths}")
            logger.debug(f"created tags {targets}")
            # second to last, pre-target level:
            # if k_lvls[-2] not in dict_query.keys():
            #     dict_query[k_lvls[-1]] = [None] * n_targets
            #     dict_query2[k_lvls[-1]] = [None] * n_targets
            # logger.debug(f'created filepaths {filepaths}')
            # logger.debug(f"created tags {targets}")
            # logger.debug(f"dict_query is {dict_query}")
            # logger.debug(f"dict_query2 is {dict_query2}")

            if isinstance(receptors, list):
                rcp_idx = receptors.index(k_lvls[-1])
                try:
                    dict_query2[rcp_idx] = k_lvls[-1]
                except Exception:
                    pass
            else:
                rcp_idx = sum([v is not None for v in dict_query2])
                try:
                    dict_query2[rcp_idx] = k_lvls[-1]
                except Exception:
                    pass
            # check whether input path is windows or posix style
            is_posix = self.check_path_style(v)
            if is_posix:
                dict_query[rcp_idx] = self.posix_path_to_curr_os(
                    v, drive_map=drive_map
                )
            else:
                dict_query[rcp_idx] = self.windows_path_to_curr_os(
                    v, drive_map=drive_map
                )
            logger.debug(f"updated filepaths {filepaths}")
            logger.debug(f"updated tags {targets}")
        return filepaths, targets


def find_dnapaint_raw(working_folder):
    datasets = {}
    for root, dirs, files in os.walk(working_folder):
        for file in files:
            match = re.search(r"_MMStack_Pos(\d+).ome.tif", file)
            if match is not None:
                key = os.path.split(root)[-1]
                datasets[key] = os.path.join(root, file)
                continue
            match = re.search(r"_NDTiffStack.tif", file)
            if match is not None:
                key = os.path.split(root)[-1]
                datasets[key] = os.path.join(root, file)
                continue
    dest_file = os.path.join(working_folder, "src_loc.yaml")
    io.save_info(dest_file, [datasets])
    return datasets, dest_file


class AbstractWorkflowCoordinator(abc.ABC):
    """Abstract Base Class for coordination of an analysis.
    This wraps the WorkflowRunner and AggregationWorkflowRunner
    from picasso_workflow.workflow for convenient usability.
    """

    def __init__(
        self,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        investigation_description="",
        dest_machine=None,
        always_save=False,
        profile_space=None,
        profile_basepage=None,
    ):
        self.root_folder = os.path.join(
            working_folder, f"AnalysisResults-{analysis_name}"
        )
        self.root_page = self.analysis_name

        self.confluence_url = confluence_url
        self.confluence_space = confluence_space
        self.confluence_token = confluence_token

        self.always_save = always_save

        if ON_CLUSTER:
            # comm = MPI.COMM_WORLD
            # self.rank = comm.Get_rank()  # Get the rank of the process
            # self.size = comm.Get_size()  # Get the total number of processes
            self.rank = int(os.getenv("SLURM_PROCID"))
            self.size = int(os.getenv("SLURM_NTASKS"))
            logger.debug(f"Assigned rank {self.rank}, size {self.size}.")
        else:
            self.rank = 0
            self.size = 1

        if self.rank == 0:
            ci = confluence.ConfluenceInterface(
                self.confluence_url,
                self.confluence_space,
                base_page,
                token=self.confluence_token,
            )
            try:
                if investigation_description == "":
                    investigation_description = f"""
                    <p><strong>Analysis file location</strong>

                    The files created during the analysis run can be found in
                    {self.root_folder}.</p>
                    """
                ci.create_page(self.root_page, investigation_description)
            except confluence.ConfluenceInterfaceError:
                pass
        else:
            # ensure rank 0 has created the root page
            time.sleep(2)
        self.ci = confluence.ConfluenceInterface(
            self.confluence_url,
            self.confluence_space,
            self.root_page,
            token=self.confluence_token,
        )

        if profile_space is not None:
            self.profiler = PerformanceProfiler(
                self.confluence_url,
                profile_space,
                profile_basepage,
                self.confluence_token,
            )
            self.profiler.init_profile_page()
        else:
            self.profiler = None

    @classmethod
    def hash_hex(cls, s, length=6):
        """Hashes a string and digests it to a hex string of a given length.
        Args:
            s : str
                string to hash
            length : int (even)
                length to digest to
        Returns:
            string of len length
        """
        return hashlib.shake_128(s.encode("utf-8")).hexdigest(int(length / 2))

    def get_configs(
        self,
        report_name,
        root_folder,
        cell_type=None,
        cell_name=None,
        camera_info=None,
    ):
        if cell_type is None:
            result_location = root_folder
        elif cell_name is None:
            result_location = os.path.join(root_folder, cell_type)
        else:
            result_location = os.path.join(root_folder, cell_type, cell_name)
        os.makedirs(result_location, exist_ok=True)

        reporter_config = {
            "report_name": report_name,
            "ConfluenceReporter": {
                "base_url": self.confluence_url,
                "space_key": self.confluence_space,
                "parent_page_title": report_name,
                "token": self.confluence_token,
            },
        }

        # if camera_info is None:
        #     camera_info = {
        #         "Gain": 1,
        #         "Sensitivity": 0.22,  # Artemis
        #         "Baseline": 100,
        #         "Qe": 1,
        #         "Pixelsize": 130,  # nm
        #     }
        analysis_config = {
            "result_location": result_location,
            "camera_info": camera_info,
            "gpufit_installed": False,
            "always_save": self.always_save,
        }
        return reporter_config, analysis_config


class SingleWorkflowCoordinator(AbstractWorkflowCoordinator):
    """A class to coordinate the analysis of a single measurement, e.g.
    one Exchange round.
    """

    def __init__(
        self,
        src_loc_file,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        dest_machine=None,
        always_save=False,
        profile_space=None,
        profile_basepage=None,
    ):
        self.dataset_filepaths = io.load_info(src_loc_file)[0]
        self.tile_entries = {
            "filepath": list(self.dataset_filepaths.values()),
            "#tags": list(self.dataset_filepaths.keys()),
        }
        self.analysis_name = os.path.split(working_folder)[-1]

        super().__init__(
            analysis_name,
            working_folder,
            confluence_url,
            confluence_space,
            confluence_token,
            base_page,
            dest_machine,
            always_save,
            profile_space,
            profile_basepage,
        )

    def prepare_analysis(self, workflow_modules):
        """
        Args:
            workflow_modules:
                list of tuple, defining modules to run
        Returns:
            run_wr_kwargs
        """
        # make sure different nodes query confluence at different times
        time.sleep(3 * self.rank)

        run_wr_kwargs = []
        execution_item = -1

        tiler = util.ParameterTiler(None, self.tile_entries)
        all_workflow_module_sets, tags = tiler.run(workflow_modules)
        print(all_workflow_module_sets)
        print(tags)
        for wkfl_mods, tag in zip(all_workflow_module_sets, tags):
            execution_item += 1
            if execution_item % self.size != self.rank:
                continue
            report_name = tag + "_" + datetime.now().strftime("%y%m%d-%H%M")
            text = f"""
                Worker of rank {self.rank} working on {report_name}
                (execution item {execution_item})"""
            text = textwrap.fill(textwrap.dedent(text), width=70)
            print(text)
            logger.debug(text)
            try:
                confpagid_cn = self.ci.create_page(report_name, "")
            except confluence.ConfluenceInterfaceError:
                confpagid_cn, _ = self.ci.get_page_properties(
                    page_title=report_name
                )
            # print('confluence page id cell name', confpagid_cn)

            reporter_config, analysis_config = self.get_configs(
                report_name, self.root_folder
            )

            wr = WorkflowRunner.config_from_dicts(
                reporter_config,
                analysis_config,
                wkfl_mods,
                continue_previous_runner=True,
                postfix="",
            )
            run_wr_kwargs.append(
                {
                    "wr": wr,
                    "dataset_name": tag,
                }
            )
        return run_wr_kwargs

    def run_analysis(self, workflow_modules):
        run_wr_kwargs = self.prepare_analysis(workflow_modules)

        # print(f'rank {self.rank}, size {self.size}: running {run_awr_kwargs}')

        for kwargs in run_wr_kwargs:
            self.run_wr(**kwargs)
            # try:
            #     self.run_wr(**kwargs)
            # except Exception as e:
            #     logger.error(e)
            #     pass

            if self.profiler is not None:
                self.profiler.append_profile_page(kwargs["wr"].all_results)

    def run_wr(self, wr, dataset_name):
        logger.debug(f"starting to analyse {dataset_name}")
        wr.run()
        logger.debug(f"finished analysing {dataset_name}")
        plt.close("all")


class AggregationWorkflowCoordinator(AbstractWorkflowCoordinator):
    """A class to coordinate the analysis of a measurement of one FOV with multiple
    targets, e.g. labeling efficiency of one cell.
    """

    def __init__(
        self,
        src_loc_file,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        dest_machine="hpcl8001",
        investigation_description="",
        always_save=False,
        profile_space=None,
        profile_basepage=None,
    ):
        # src_loc_file is a picasso-info like yaml file, representing a list of
        # dicts, with keys "#tags" and #filepath
        self.dataset_filepaths = io.load_info(src_loc_file)
        # if not, it may be a dict of key (tag): filepath pairs
        if "#tags" not in self.dataset_filepaths[0].keys():
            # self.dataset_filepaths = io.load_info(src_loc_file)[0]
            self.tile_entries = {
                "filepath": list(self.dataset_filepaths[0].values()),
                "#tags": list(self.dataset_filepaths[0].keys()),
            }

        self.analysis_name = analysis_name

        super().__init__(
            analysis_name,
            working_folder,
            confluence_url,
            confluence_space,
            confluence_token,
            base_page,
            dest_machine,
            always_save,
            profile_space,
            profile_basepage,
        )

    def prepare_analysis(self, workflow_modules_sgl, workflow_modules_agg):
        """
        Args:
            workflow_modules_multi : dict of
                single_dataset_tileparameters
                single_dataset_modules:
                    list of tuple, defining modules to run in the first stage
                    of evaluation of the individual rounds
                aggregation_modules:
                    list of tuple, defining modules to run in the second strage
                    of the evaluation, across the varous rounds
        Returns:
            run_wr_kwargs
        """
        # make sure different nodes query confluence at different times
        time.sleep(3 * self.rank)

        run_awr_kwargs = []

        execution_item = -1
        for datasets in self.dataset_filepaths:
            execution_item += 1
            if execution_item % self.size != self.rank:
                continue

            if rname := datasets.get("report_name"):
                report_name = (
                    rname + "_" + datetime.now().strftime("%y%m%d-%H%M")
                )
                # create the corresponding confluence page
                try:
                    ci = confluence.ConfluenceInterface(
                        self.confluence_url,
                        self.confluence_space,
                        self.root_page,
                        token=self.confluence_token,
                    )
                    ci.create_page(report_name, "")
                except Exception:
                    pass
            else:
                report_name = self.analysis_name

            reporter_config, analysis_config = self.get_configs(
                report_name, self.root_folder
            )

            workflow_modules_multi = {
                "single_dataset_tileparameters": datasets,
                "single_dataset_modules": workflow_modules_sgl,
                "aggregation_modules": workflow_modules_agg,
            }

            awr = AggregationWorkflowRunner.config_from_dicts(
                reporter_config,
                analysis_config,
                workflow_modules_multi,
                continue_previous_runner=True,
                single_workflow_parallel=False,
                postfix="",
            )
            run_awr_kwargs.append(
                {
                    "awr": awr,
                    "report_name": report_name,
                }
            )
        return run_awr_kwargs

    def run_analysis(self, workflow_modules_sgl, workflow_modules_agg):
        run_awr_kwargs = self.prepare_analysis(
            workflow_modules_sgl, workflow_modules_agg
        )

        print(f"rank {self.rank}, size {self.size}: running {run_awr_kwargs}")

        for kwargs in run_awr_kwargs:
            self.run_awr(**kwargs)

            if self.profiler is not None:
                self.profiler.append_profile_page(kwargs["awr"].all_results)

    def run_awr(self, awr, report_name):
        logger.debug(f"starting to analyse {report_name}")
        awr.run()
        logger.debug(f"finished analysing {report_name}")
        plt.close("all")


class InvestigationCoordinator(AbstractWorkflowCoordinator):
    """A class to coordinate the analysis of a measurement series, i.e.
    multiple conditions / cell types, cells, target molecules.
    """

    def __init__(
        self,
        src_loc,
        receptors,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        dest_machine="hpcl8",
        investigation_description="",
        always_save=False,
        iterations=1,
        underscores=[1, 0, 0, 0],
        profile_space=None,
        profile_basepage=None,
    ):
        """
        Args:
            iterations : int
                potentially call the workflow on the same data multple times
                with an iterator (e.g. to select different cells)
        """
        self.dataset_filepaths, self.tags = PathParser().parse_source(
            src_loc,
            receptors,
            dest_machine,
            underscores=underscores,
        )

        self.cells = {
            k: list(v.keys()) for k, v in self.dataset_filepaths.items()
        }
        self.receptors = receptors
        self.iterations = iterations

        self.analysis_name = analysis_name

        super().__init__(
            analysis_name,
            working_folder,
            confluence_url,
            confluence_space,
            confluence_token,
            base_page,
            investigation_description,
            dest_machine,
            always_save,
            profile_space,
            profile_basepage,
        )

    def prepare_sglcell_analysis(self, get_workflow_modules):
        """
        Args:
            get_workflow_modules:
                Callable taking arguments cell_type, report_name, datasets
                and returning workflow_modules_multi
        Returns:
            run_awr_kwargs
        """
        # make sure different nodes query confluence at different times
        time.sleep(3 * self.rank)

        queue = multiprocessing.Queue()
        lock = multiprocessing.Lock()
        analysis_name_hash = self.hash_hex(self.analysis_name, 6)
        fp_dfa2 = f"workflow_{self.analysis_name}_sglcell_successes.xlsx"
        try:
            df = pd.read_excel(fp_dfa2, index_col=0, header=0)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "cell_type",
                    "cell_name",
                    "module",
                    "success",
                    "start time",
                    "duration",
                ]
            )

        run_awr_kwargs = []
        execution_item = -1

        for cell_type, type_dict in self.dataset_filepaths.items():
            try:
                confpagid_ct = self.ci.create_page(
                    f"{cell_type}-{analysis_name_hash}", ""
                )
            except confluence.ConfluenceInterfaceError:
                confpagid_ct, _ = self.ci.get_page_properties(
                    page_title=f"{cell_type}-{analysis_name_hash}"
                )
            # print('confluence page id cell type', confpagid_ct)
            analysis_cell_hash = self.hash_hex(
                f"{cell_type}-{self.analysis_name}", 6
            )

            for cell_name, fps in type_dict.items():
                receptors = self.tags[cell_type][cell_name]
                for i in range(self.iterations):
                    execution_item += 1
                    if execution_item % self.size != self.rank:
                        continue
                    if self.iterations > 1:
                        cell_name_i = f"{cell_name}-it{i}"
                    else:
                        cell_name_i = cell_name
                    report_name = f"{cell_name_i}-{analysis_cell_hash}"
                    text = f"""
                        Worker of rank {self.rank} working on {report_name}
                        (execution item {execution_item})"""
                    text = textwrap.fill(textwrap.dedent(text), width=70)
                    print(text)
                    try:
                        confpagid_cn = self.ci.create_page(
                            report_name, "", parent_id=confpagid_ct
                        )
                    except confluence.ConfluenceInterfaceError:
                        confpagid_cn, _ = self.ci.get_page_properties(
                            page_title=report_name
                        )
                    # print('confluence page id cell name', confpagid_cn)

                    reporter_config, analysis_config = self.get_configs(
                        report_name, self.root_folder, cell_type, cell_name_i
                    )
                    datasets = {"#tags": receptors, "filepath": fps}
                    workflow_modules_multi = get_workflow_modules(
                        cell_type, report_name, datasets, i=i
                    )
                    # analyse if the cell has not been analysed yet
                    n_modules = len(
                        workflow_modules_multi["aggregation_modules"]
                    )
                    n_success = df.loc[
                        (df["cell_type"] == cell_type)
                        & (df["cell_name"] == cell_name_i),
                        "success",
                    ].sum()
                    if n_modules == n_success:
                        # logger.debug(
                        #     f"""skipping {cell_type}, {cell_name}
                        #     because it was analysed already."""
                        # )
                        # continue
                        pass

                    awr = AggregationWorkflowRunner.config_from_dicts(
                        reporter_config,
                        analysis_config,
                        workflow_modules_multi,
                        continue_previous_runner=True,
                        single_workflow_parallel=False,
                    )
                    run_awr_kwargs.append(
                        {
                            "awr": awr,
                            "cell_type": cell_type,
                            "cell_name": cell_name_i,
                            "queue": queue,
                            "lock": lock,
                            "fp_dfa2": fp_dfa2,
                        }
                    )
        return run_awr_kwargs

    def prepare_mergecell_analysis(self, get_mergecell_workflow_modules):
        """
        Args:
            get_mergecell_workflow_modules:
                Callable taking arguments cell_type, report_name, datasets
                and returning workflow_modules_multi
        Returns:
            run_awr_kwargs
        """
        # make sure different nodes query confluence at different times
        time.sleep(3 * self.rank)

        queue = multiprocessing.Queue()
        lock = multiprocessing.Lock()
        fp_dfa2 = f"workflow_{self.analysis_name}_mergecell_successes.xlsx"
        # try:
        #     df = pd.read_excel(fp_dfa2, index_col=0, header=0)
        # except FileNotFoundError:
        #     df = pd.DataFrame(columns=[
        #         "cell_type", "cell_name", "module", "success",
        #         "start time", "duration"])

        run_awr_kwargs = []
        execution_item = -1

        analysis_name_hash = self.hash_hex(self.analysis_name, 6)
        for cell_type, cell_names in self.cells.items():
            execution_item += 1
            if execution_item % self.size != self.rank:
                continue
            try:
                confpagid_ct = self.ci.create_page(
                    f"{cell_type}-{analysis_name_hash}", ""
                )
            except confluence.ConfluenceInterfaceError:
                confpagid_ct, _ = self.ci.get_page_properties(
                    page_title=f"{cell_type}-{analysis_name_hash}"
                )
            # print('confluence page id cell type', confpagid_ct)
            analysis_cell_hash = self.hash_hex(
                f"{cell_type}-{self.analysis_name}", 6
            )

            report_names = [
                f"{cell_name}-{analysis_cell_hash}" for cell_name in cell_names
            ]
            fp_workflows = [
                os.path.join(self.root_folder, cell_type, cell_name)
                for cell_name in cell_names
            ]
            report_name = f"{cell_type}-{analysis_name_hash}"
            datasets = {"#tags": ["dummmy"], "filepath": ["dummy"]}

            reporter_config, analysis_config = self.get_configs(
                report_name, self.root_folder, cell_type
            )

            workflow_modules_multi = get_mergecell_workflow_modules(
                cell_type, fp_workflows, report_names, datasets
            )

            awr = AggregationWorkflowRunner.config_from_dicts(
                reporter_config,
                analysis_config,
                workflow_modules_multi,
                continue_previous_runner=True,
            )
            run_awr_kwargs.append(
                {
                    "awr": awr,
                    "cell_type": cell_type,
                    "cell_name": "",
                    "queue": queue,
                    "lock": lock,
                    "fp_dfa2": fp_dfa2,
                }
            )
        return run_awr_kwargs

    def run_awr(self, awr, cell_type, cell_name, queue, lock, fp_dfa2):
        logger.debug(f"starting to analyse {cell_type}, {cell_name}")
        awr.run()
        logger.debug(f"finished analysing {cell_type}, {cell_name}")
        plt.close("all")
        agg_results = awr.all_results["aggregation"]
        successes = {}
        for mod, res in agg_results.items():
            successes[mod] = res.get("success", False)
            logger.debug(
                f"""{cell_type}, {cell_name}, {mod}:
                {res.get('success', False)}"""
            )
        return_val = (cell_type, cell_name, successes)
        queue.put(return_val)
        with lock:
            try:
                df = pd.read_excel(fp_dfa2, index_col=0, header=0)
            except FileNotFoundError:
                df = pd.DataFrame(
                    columns=[
                        "cell_type",
                        "cell_name",
                        "module",
                        "success",
                        "start time",
                        "duration",
                    ]
                )
            for mod, suc in successes.items():
                idx = len(df.index)
                df.loc[idx, "cell_type"] = cell_type
                df.loc[idx, "cell_name"] = cell_name
                df.loc[idx, "module"] = mod
                df.loc[idx, "success"] = suc
                df.loc[idx, "start time"] = res.get("start time", "")
                df.loc[idx, "duration"] = res.get("duration", 0)
            df.to_excel(fp_dfa2)
        return return_val

    def initialize_summary_pages(
        self,
        ci,
        summary_page_title,
        summary_columns,
    ):
        # rcode = generate_random_code(6)
        # summary_page_title = f"{summary_page_title}-{rcode}"
        text = f"<b>{summary_page_title}</b>"
        text += "<table>"
        text += "<tr>"
        text += "<td><b>Dataset</b></td>"
        for col in summary_columns:
            text += f"<td><b>{col}</b></td>"
        text += "</tr>"
        text += "</table>"
        try:
            ci.create_page(summary_page_title, text)
        except confluence.ConfluenceInterfaceError:
            # pagid, pagtit = ci.get_page_properties(summary_page_title)
            # ci.delete_page(pagid)
            # ci.create_page(summary_page_title, text)
            logger.debug("Could not create summary page. Probably exists")
        return summary_page_title

    def extract_fpfig_from_results(self, awr, figloc):
        """From the allresults attribute of the mergecell
        AggregationWorkflowRunner, extract filepaths to figures to put
        onto the summary page
        Args:
            awr : AggregationWorkflowRunner
                the Runner to extract results from
            figloc : list of list of str
                outer list: For each column in the summary page
                inner list: location within results. e.g.
                    ["aggregation", "00_ripleysk_average2", "fp_figmeanvals"]
        """
        fp_figs = [
            reduce(lambda d, key: d[key], loc, awr.all_results)
            for loc in figloc
        ]
        return fp_figs

    def add_to_summary_page(
        self,
        summary_page_title,
        dataset,
        data_list,
        data_types="img",
    ):
        """
        Args:
            data_list : list
                list of data to enter into the summary table
            data_types : str or list of str of list of dict
                the data types to enter.
                "img": image (file path). Image is uploaded and displayed
                any other: values are directly entered
                if str, all values are the same type
            data_fmt : dict, optional
                for each data_type, formatting options can be given.
                default values are
        """
        pagid, pagtit = self.ci.get_page_properties(summary_page_title)

        data_fmt_default = {
            "img": {"height": 350},
            "float": {"precision": 6, "unit": "", "factor": 1},
            "str": {},
        }
        # add default values to data_fmt
        if isinstance(data_types, str):
            data_types = [data_types] * len(data_list)
        for i, entry in enumerate(data_types):
            if isinstance(entry, str):
                data_types[i] = {"type": entry}
                for def_k, def_v in data_fmt_default[entry].items():
                    data_types[i][def_k] = def_v
            elif isinstance(entry, dict):
                # key "type" is required
                tp = entry["type"]
                if tp not in data_fmt_default.keys():
                    continue
                for def_k, def_v in data_fmt_default[tp].items():
                    if def_k not in entry.keys():
                        data_types[i][def_k] = def_v

        fn_figs = []
        for fp_fig, data_def in zip(data_list, data_types):
            if data_def["type"] == "img":
                # rcode = generate_random_code(6)
                rt, fn_fig = os.path.split(fp_fig)
                # fn_fig = f"{dataset}_{rcode}_{fn_fig}"
                # fp_figu = os.path.join(rt, fn_fig)
                # shutil.copy(fp_fig, fp_figu)
                fp_figu = fp_fig
                self.ci.upload_attachment(pagid, fp_figu)
                fn_figs.append(fn_fig)
            else:
                fn_figs.append(fp_fig)

        text = self.ci.get_page_body(summary_page_title)
        text, postfix = text.split("</table>")
        text += f"""
            <tr>
            <td>
                  <b>{dataset}</b>
            </td>
            """
        for fnf, data_def in zip(fn_figs, data_types):
            if data_def["type"] == "img":
                image_height = data_def["height"]
                text += f"""
                <td>
                      <ac:image ac:height="{image_height}">
                      <ri:attachment ri:filename="{fnf}" />
                      </ac:image>
                </td>
                """
            elif data_def["type"] == "float":
                precision = data_def["precision"]
                unit = data_def["unit"]
                factor = data_def["factor"]
                value = f"{factor * fnf:.{precision}f} {unit}"
                text += f"<td>{value}</td>"
            else:
                text += f"<td>{fnf}</td>"
        text += "</tr>"
        text += "</table>" + postfix
        self.ci.update_page_content(
            summary_page_title, pagid, text, replace=True
        )

    def run_sglcell_analysis(
        self,
        get_sglcell_workflow_modules,
        summary_page_title=None,
        summary_columns=None,
        figloc=None,
        data_types={"type": "img"},
    ):
        try:
            summary_page_title = f"{summary_page_title} - {self.analysis_name}"
            if self.rank == 0:
                summary_page_title = self.initialize_summary_pages(
                    self.ci,
                    summary_page_title,
                    summary_columns,
                )
        except Exception:
            pass

        run_awr_kwargs = self.prepare_sglcell_analysis(
            get_sglcell_workflow_modules
        )

        # print(f'rank {self.rank}, size {self.size}: running {run_awr_kwargs}')

        for kwargs in run_awr_kwargs:
            self.run_awr(**kwargs)

            fp_figs = self.extract_fpfig_from_results(kwargs["awr"], figloc)
            self.add_to_summary_page(
                summary_page_title,
                kwargs["cell_type"] + "-" + kwargs["cell_name"],
                data_list=fp_figs,
                data_types=data_types,
            )
            if self.profiler is not None:
                self.profiler.append_profile_page(kwargs["awr"].all_results)

    def run_mergecell_analysis(
        self,
        get_mergecell_workflow_modules,
        summary_page_title,
        summary_columns,
        figloc,
        data_types="img",
    ):
        summary_page_title = f"{summary_page_title} - {self.analysis_name}"
        if self.rank == 0:
            summary_page_title = self.initialize_summary_pages(
                self.ci,
                summary_page_title,
                summary_columns,
            )

        run_awr_kwargs = self.prepare_mergecell_analysis(
            get_mergecell_workflow_modules
        )

        # print(f'rank {self.rank}, size {self.size}: running {run_awr_kwargs}')

        for kwargs in run_awr_kwargs:
            self.run_awr(**kwargs)

            fp_figs = self.extract_fpfig_from_results(kwargs["awr"], figloc)
            self.add_to_summary_page(
                summary_page_title,
                kwargs["cell_type"],
                data_list=fp_figs,
                data_types=data_types,
            )
            if self.profiler is not None:
                self.profiler.append_profile_page(kwargs["awr"].all_results)


class PerformanceProfiler:
    def __init__(
        self, confluence_url, confluence_space, base_page, confluence_token
    ):
        self.ci = confluence.ConfluenceInterface(
            confluence_url,
            confluence_space,
            base_page,
            token=confluence_token,
        )

    def init_profile_page(
        self,
        page_title="Cluster Performance Profiling",
    ):
        self.page_title = page_title
        self.gen_columns = ["cluster", "module", "Confluence page"]
        self.data_columns = [
            "end time",
            "peak_cpu_cores",
            "peak_cpu_usage",
            "mean_cpu_usage",
            "peak_memory_gb",
            "peak_memory_per_locs",
            "nlocs",
            "duration",
        ]
        # rcode = generate_random_code(6)
        # summary_page_title = f"{summary_page_title}-{rcode}"
        text = f"<b>{page_title}</b>"
        text += "<table>"
        text += "<tr>"
        for col in self.gen_columns + self.data_columns:
            text += f"<td><b>{col}</b></td>"
        text += "</tr>"
        text += "</table>"
        try:
            self.ci.create_page(page_title, text)
        except confluence.ConfluenceInterfaceError:
            # pagid, pagtit = ci.get_page_properties(page_title)
            # ci.delete_page(pagid)
            # ci.create_page(page_title, text)
            pass
        return page_title

    def append_profile_page(self, all_results):
        pagid, pagtit = self.ci.get_page_properties(self.page_title)

        entry_rows = []
        for stage_key, stage in all_results.items():
            if not isinstance(stage, list):
                stage = [stage]
            for stage_item in stage:
                for module, results in stage_item.items():
                    row = [platform.node(), module, ""]
                    for col in self.data_columns:
                        row.append(results.get(col, 0))
                    entry_rows.append(row)

        text = self.ci.get_page_body(self.page_title)
        text, postfix = text.split("</table>")
        for row in entry_rows:
            text += "<tr>"
            for data in row:
                text += "<td>"
                text += f"{data}"
                text += "</td>"
            text += "</tr>"
        text += "</table>" + postfix
        self.ci.update_page_content(self.page_title, pagid, text, replace=True)

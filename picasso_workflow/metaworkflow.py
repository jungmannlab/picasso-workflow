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
from picasso_workflow import AggregationWorkflowRunner
from picasso_workflow import confluence
import matplotlib.pyplot as plt
import multiprocessing
from mpi4py import MPI
import pandas as pd
import logging
import time
import hashlib
from picasso import io
import pathlib
from functools import reduce


logger = logging.getLogger(__name__)


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
        drivepaths = os.environ["DRIVEPATHS"]
        self.drive_paths = {}
        for machinepaths in drivepaths.split(";"):
            try:
                machine, paths = machinepaths.strip().split("::")
            except (IndexError, ValueError):
                continue
            paths = [p.strip() for p in paths.split(",")]
            self.drive_paths[machine] = paths
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

    def parse_source(self, src_loc, receptors, dest_machine="hpcl8001"):
        """
        Args:
            src_loc : string
                filepath to a yaml file defining a list (len 1) of dict:
                keys: strings with four "_"
                    (A_B_C_D: cell type/condition defined by A_B,
                     C: cell/img position ID, D: imaging target)
                values: paths to data
        """
        src_dict = io.load_info(src_loc)[0]
        if dest_machine not in self.drive_paths.keys():
            raise ValueError(f"Machine {dest_machine} not defined in .env!")

        for src_machine, drivepaths in self.drive_paths.items():
            src_on_machine = []
            for srcp in src_dict.values():
                src_on_machine.append(any([p in srcp for p in drivepaths]))
            if all(src_on_machine):
                # src_machine has all drive paths defined
                break

        drive_map = {}
        for src_p, dest_p in zip(
            self.drive_paths[src_machine], self.drive_paths[dest_machine]
        ):
            drive_map[src_p] = dest_p

        # keys: [dataset]_[treatment]_[cell#]_[receptor]
        # turn into hierarchical dict
        # * lvl1 key: [dataset_treatment]
        # * lvl2 key: cell#
        # * values: list of receptors, according to list order above
        filepaths = {}
        for k, v in src_dict.items():
            k_items = k.split("_")
            k1 = f"{k_items[0]}_{k_items[1]}"
            k2 = k_items[2]
            rcp = k_items[3]
            rcp_idx = receptors.index(rcp)
            if k1 not in filepaths.keys():
                filepaths[k1] = {}
            if k2 not in filepaths[k1].keys():
                filepaths[k1][k2] = [None] * len(receptors)
            # check whether input path is windows or posix style
            is_posix = self.check_path_style(v)
            if is_posix:
                filepaths[k1][k2][rcp_idx] = self.posix_path_to_curr_os(
                    v, drive_map=drive_map
                )
            else:
                filepaths[k1][k2][rcp_idx] = self.windows_path_to_curr_os(
                    v, drive_map=drive_map
                )
        return filepaths


class InvestigationCoordinator:
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
        dest_machine="hpcl8001",
        investigation_description="",
    ):
        self.dataset_filepaths = PathParser().parse_source(
            src_loc, receptors, dest_machine
        )

        self.cells = {
            k: list(v.keys()) for k, v in self.dataset_filepaths.items()
        }
        self.receptors = receptors

        self.analysis_name = analysis_name
        self.root_folder = os.path.join(
            working_folder, f"InvestigationResults-{analysis_name}"
        )
        self.root_page = analysis_name

        self.confluence_url = confluence_url
        self.confluence_space = confluence_space
        self.confluence_token = confluence_token

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()  # Get the rank of the process
        self.size = comm.Get_size()  # Get the total number of processes

        if self.rank == 0:
            ci = confluence.ConfluenceInterface(
                self.confluence_url,
                self.confluence_space,
                base_page,
                token=self.confluence_token,
            )
            try:
                investigation_description += f"""
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
                execution_item += 1
                if execution_item % self.size != self.rank:
                    continue
                report_name = f"{cell_name}-{analysis_cell_hash}"
                print(
                    f"""
                    Worker of rank {self.rank} working on {report_name}
                    (execution item {execution_item})"""
                )
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
                    report_name, self.root_folder, cell_type, cell_name
                )
                datasets = {"#tags": self.receptors, "filepath": fps}
                workflow_modules_multi = get_workflow_modules(
                    cell_type, report_name, datasets
                )
                # analyse if the cell has not been analysed yet
                n_modules = len(workflow_modules_multi["aggregation_modules"])
                n_success = df.loc[
                    (df["cell_type"] == cell_type)
                    & (df["cell_name"] == cell_name),
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
                        "cell_name": cell_name,
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

    def get_configs(
        self,
        report_name,
        root_folder,
        cell_type,
        cell_name=None,
        camera_info=None,
    ):
        if cell_name is None:
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

        if camera_info is None:
            camera_info = {
                "Gain": 1,
                "Sensitivity": 0.22,  # Artemis
                "Baseline": 100,
                "Qe": 1,
                "Pixelsize": 130,  # nm
            }
        analysis_config = {
            "result_location": result_location,
            "camera_info": camera_info,
            "gpufit_installed": False,
            "always_save": False,
        }
        return reporter_config, analysis_config

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
            pagid, pagtit = ci.get_page_properties(summary_page_title)
            ci.delete_page(pagid)
            ci.create_page(summary_page_title, text)
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

    def add_to_summary_page(self, summary_page_title, dataset, fp_figs):
        """
        fp_figs : len 4
        """
        pagid, pagtit = self.ci.get_page_properties(summary_page_title)

        fn_figs = []
        for fp_fig in fp_figs:
            # rcode = generate_random_code(6)
            rt, fn_fig = os.path.split(fp_fig)
            # fn_fig = f"{dataset}_{rcode}_{fn_fig}"
            # fp_figu = os.path.join(rt, fn_fig)
            # shutil.copy(fp_fig, fp_figu)
            fp_figu = fp_fig
            self.ci.upload_attachment(pagid, fp_figu)
            fn_figs.append(fn_fig)

        text = self.ci.get_page_body(summary_page_title)
        text, postfix = text.split("</table>")
        text += f"""
            <tr>
            <td>
                  <b>{dataset}</b>
            </td>
            """
        for fnf in fn_figs:
            text += f"""
            <td>
                  <ac:image ac:height="350">
                  <ri:attachment ri:filename="{fnf}" />
                  </ac:image>
            </td>
            """
        text += "</tr>"
        text += "</table>" + postfix
        self.ci.update_page_content(
            summary_page_title, pagid, text, replace=True
        )

    def run_sglcell_analysis(self, get_sglcell_workflow_modules):
        run_awr_kwargs = self.prepare_sglcell_analysis(
            get_sglcell_workflow_modules
        )

        # print(f'rank {self.rank}, size {self.size}: running {run_awr_kwargs}')

        for kwargs in run_awr_kwargs:
            self.run_awr(**kwargs)

    def run_mergecell_analysis(
        self,
        get_mergecell_workflow_modules,
        summary_page_title,
        summary_columns,
        figloc,
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
                fp_figs,
            )

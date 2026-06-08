#!/usr/bin/env python
"""Higher-level analysis coordination across conditions and cells.

Provides path parsing across machines (:class:`PathParser`) and a family of
workflow coordinators that run picasso-workflow analyses over multiple
conditions, cells and targets and aggregate over them.

Author: Heinrich Grabmayr
Initial date: January 29, 2025
"""

from __future__ import annotations

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
    """Parse and re-map data paths for multi-level picasso-workflow analysis.

    Paths arrive in a dictionary whose keys encode the analysis levels (via
    underscore-separated fields) and whose values are paths to the
    corresponding data. The parser restructures these into nested per-level
    dicts and rewrites the paths for the current machine's drive layout.
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

    def windows_path_to_curr_os(self, winpath: str, drive_map: dict) -> str:
        """Convert a Windows-style path to the current OS's drive layout.

        Replaces the Windows drive (e.g. ``X:``) with the mapped drive.

        Parameters
        ----------
        winpath : str
            The Windows-style path to convert.
        drive_map : dict
            Map from Windows drive (e.g. ``"W:"``) to the target drive
            (e.g. ``"/Volumes/pool-miblab4"``).

        Returns
        -------
        str
            The converted path.
        """
        winpath = pathlib.PureWindowsPath(winpath)
        if winpath.drive:
            # absolute path
            drive = drive_map[winpath.drive]
            # currospath = pathlib.Path(drive, *winpath.parts[1:])

            # change logic to not convert to current os but new drive type os
            if ":" in drive:  # converted path is a windows path
                currospath = pathlib.PureWindowsPath(drive, *winpath.parts[1:])
            else:  # converted path is a Posix path
                currospath = pathlib.PurePosixPath(drive, *winpath.parts[1:])
        else:
            # relative path
            # currospath = pathlib.Path(*winpath.parts)

            # change logic to not convert to current os but new drive type os
            if (
                ":" in list(drive_map.values())[0][:3]
            ):  # converted path is a windows path
                # currospath = pathlib.PureWindowsPath(drive, *winpath.parts[1:])
                currospath = pathlib.PureWindowsPath(winpath)
            else:  # converted path is a Posix path
                # currospath = pathlib.PurePosixPath(drive, *winpath.parts[1:])
                currospath = pathlib.PurePosixPath(winpath)
        return str(currospath)

    def posix_path_to_curr_os(self, posixpath: str, drive_map: dict) -> str:
        """Convert a posix-style path to the current OS's drive layout.

        Replaces the source drive root with the mapped drive.

        Parameters
        ----------
        posixpath : str
            The posix-style path to convert.
        drive_map : dict
            Map from source drive (e.g. ``"W:"``) to the target drive
            (e.g. ``"/Volumes/pool-miblab4"``).

        Returns
        -------
        str
            The converted path.
        """
        # Backslashes are only ever path separators in these data paths,
        # never filename characters. Normalise them so a mixed-separator
        # path (e.g. "U:/a/b\\c\\d") splits into every component, instead
        # of leaving a backslash-joined tail unconverted (PurePosixPath
        # only ever splits on "/").
        posixpath = pathlib.PurePosixPath(str(posixpath).replace("\\", "/"))
        if posixpath.drive:
            # absolute path
            drive = drive_map[posixpath.drive]
            logger.debug(f"old drive: {posixpath.drive}")
            logger.debug(f"new drive: {drive}")
            # currospath = pathlib.Path(drive, *posixpath.parts[1:])

            # change logic to not convert to current os but new drive type os
            if ":" in drive:  # converted path is a windows path
                currospath = pathlib.PureWindowsPath(
                    drive, *posixpath.parts[1:]
                )
            else:  # converted path is a Posix path
                currospath = pathlib.PurePosixPath(drive, *posixpath.parts[1:])
        else:
            # print('no drive')
            drivelen = len(pathlib.Path(list(drive_map.keys())[0]).parts)
            pseudodrive = pathlib.PurePosixPath(*posixpath.parts[:drivelen])
            # print('pseudodrive:', pseudodrive)
            # print(drive_map)
            if drive := drive_map.get(str(pseudodrive)):
                # print('found drive:', drive)
                # found the drive after all
                # currospath = pathlib.Path(drive, *posixpath.parts[drivelen:])

                # change logic to not convert to current os but new drive type os
                if ":" in drive:  # converted path is a windows path
                    currospath = pathlib.PureWindowsPath(
                        drive, *posixpath.parts[drivelen:]
                    )
                else:  # converted path is a Posix path
                    currospath = pathlib.PurePosixPath(
                        drive, *posixpath.parts[drivelen:]
                    )
            else:
                # print("did not find a drive")
                # relative path
                # currospath = pathlib.Path(*posixpath.parts)

                # change logic to not convert to current os but new drive type os
                if (
                    ":" in list(drive_map.values())[0]
                ):  # converted path is a windows path
                    currospath = pathlib.PureWindowsPath(
                        drive, *posixpath.parts[1:]
                    )
                else:  # converted path is a Posix path
                    currospath = pathlib.PurePosixPath(
                        drive, *posixpath.parts[drivelen:]
                    )
        # print('new path:', currospath)
        return str(currospath)

    def check_path_style(self, path: str) -> bool:
        """Check whether a path is windows- or posix-style.

        A leading drive letter (e.g. ``"U:"``) or UNC prefix
        (``"\\\\server"``) unambiguously marks a Windows path; a leading
        ``"/"`` marks a posix path. These take precedence over separator
        counting: a Windows path may well contain forward slashes (e.g.
        ``"U:/users/foo\\bar"``), so counting ``"/"`` against ``"\\"`` would
        misclassify it as posix and lose the backslash-joined parts. Only
        relative paths (no such prefix) fall back to comparing the number of
        ``"/"`` and ``"\\"`` separators.

        Parameters
        ----------
        path : str
            The path to classify.

        Returns
        -------
        bool
            True if the path is posix-style, False if it is Windows-style.
        """
        path = str(path)
        if re.match(r"[A-Za-z]:", path) or path.startswith("\\\\"):
            return False  # windows
        if path.startswith("/"):
            return True  # posix
        num_fwd = path.count("/")
        num_bwd = path.count("\\")
        return num_fwd > num_bwd

    def check_machine(self, machine: str, pattern: str) -> bool:
        """Check a machine name against a machine pattern.

        For example, ``"mymachine6234"`` against ``"mymachine6XXX"``, with
        ``X`` standing for any non-whitespace character.

        Parameters
        ----------
        machine : str
            The concrete machine name.
        pattern : str
            The pattern, with ``X`` as a wildcard character.

        Returns
        -------
        bool
            Whether the machine name matches the pattern.
        """
        regex = pattern.replace("X", r"\S")
        regex = f"^{regex}$"
        # print('machine', machine, 'pattern', pattern, 'regex', regex)
        return re.fullmatch(regex, machine) is not None

    def get_machine_drivepaths(self, machine: str) -> list | None:
        """Return the configured drive paths for a machine, or None.

        Parameters
        ----------
        machine : str
            The machine name to resolve against the ``Drivepaths`` config.

        Returns
        -------
        list or None
            The drive paths of the first matching machine pattern, or None if
            no pattern matches.
        """
        # print('drive paths', self.drive_paths)
        for pattern, paths in self.drive_paths.items():
            if self.check_machine(machine, pattern):
                return paths
        else:
            return None

    def convert_path(self, src_path: str, dest_machine: str | None) -> str:
        """Convert a path from a source machine to a destination machine.

        Rewrites both the path style and the volume notation. If the
        destination machine cannot be resolved (it is not listed in the
        ``Drivepaths`` config), or the source path is not located under any
        known drive root, the path is returned unchanged.

        Parameters
        ----------
        src_path : str
            The source path to convert.
        dest_machine : str or None
            The destination machine name. If None, the current host is used.

        Returns
        -------
        str
            The converted path, or ``src_path`` unchanged if it could not be
            resolved.
        """
        # find current machine key
        if dest_machine is None:
            for machine in self.drive_paths.keys():
                if self.check_machine(platform.node(), machine):
                    dest_machine = machine
                    break
            if dest_machine is None:
                logger.warning(
                    f"Current machine {platform.node()} is not listed "
                    "in the Drivepaths config; path not converted."
                )
                return src_path
        dest_paths = self.get_machine_drivepaths(dest_machine)
        if dest_paths is None:
            logger.warning(
                f"Destination machine {dest_machine} is not listed in "
                "the Drivepaths config; path not converted."
            )
            return src_path

        # find the source machine: the one with drive roots that the
        # source path is located under
        src_machine = None
        for machine, drivepaths in self.drive_paths.items():
            if any([p in src_path for p in drivepaths]):
                src_machine = machine
                break
        if src_machine is None:
            logger.debug(
                f"Path {src_path} is not under any known drive root; "
                "path not converted."
            )
            return src_path

        logger.debug(f"found src machine: {src_machine}")
        logger.debug(f"dest machine: {dest_machine}")

        drive_map = {}
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
        self,
        src_loc: str,
        receptors: list | int,
        dest_machine: str | None = None,
        underscores: list[int] = [1, 0, 0],
    ) -> tuple[dict, dict]:
        """Parse a source-location YAML into nested filepath and target dicts.

        Parameters
        ----------
        src_loc : str
            Filepath to a YAML file defining a length-1 list of dict whose
            keys encode the analysis levels (e.g. ``A_B_C_D``: cell
            type/condition ``A_B``, cell/image position ID ``C``, imaging
            target ``D``) and whose values are paths to data.
        receptors : list of str or int
            The targets to analyse if identical across datasets, or the number
            of targets if their identities differ between datasets.
        dest_machine : str, optional
            The destination machine name. If None, the current host is used.
        underscores : list of int, optional
            The number of underscores in each analysis level. Default is
            ``[1, 0, 0]``.

        Returns
        -------
        filepaths : dict
            Nested dict of per-level data paths, rewritten for the current
            machine.
        targets : dict
            Nested dict of per-level target identities.
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


def find_dnapaint_raw(working_folder: str) -> tuple[dict, str]:
    """Discover raw DNA-PAINT movies and write a source-location file.

    Parameters
    ----------
    working_folder : str
        Folder to search for raw movies.

    Returns
    -------
    datasets : dict
        Mapping of dataset tags to discovered raw-movie paths.
    dest_file : str
        Path to the written ``src_loc.yaml`` describing the datasets.
    """
    from picasso_workflow import util
    from picasso import io

    datasets = util.find_raw_movies(working_folder)

    dest_file = os.path.join(working_folder, "src_loc.yaml")
    io.save_info(dest_file, [datasets])
    return datasets, dest_file


class AbstractWorkflowCoordinator(abc.ABC):
    """Abstract base class for coordinating an analysis.

    Wraps :class:`~picasso_workflow.workflow.WorkflowRunner` and
    :class:`~picasso_workflow.workflow.AggregationWorkflowRunner` for
    convenient usability, including Confluence page setup and SLURM rank
    handling.
    """

    def __init__(
        self,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        confluence_username=None,
        base_page="base_page",
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
        self.confluence_username = confluence_username

        logger.debug(f"confluence_url: {confluence_url}")
        logger.debug(f"confluence_space: {confluence_space}")
        logger.debug(f"confluence_token: {confluence_token}")
        logger.debug(f"confluence_username: {confluence_username}")

        self.always_save = always_save

        if ON_CLUSTER:
            # comm = MPI.COMM_WORLD
            # self.rank = comm.Get_rank()  # Get the rank of the process
            # self.size = comm.Get_size()  # Get the total number of processes
            self.rank = int(os.getenv("SLURM_PROCID"))
            self.size = int(os.getenv("SLURM_NTASKS"))
            logger.debug(
                f"Assigned this node rank {self.rank}, size {self.size}."
            )
        else:
            self.rank = 0
            self.size = 1
            logger.debug(
                f"No SLRUM env vars found. Assigned this node rank {self.rank}, size {self.size}."
            )

        if self.rank == 0:
            ci = confluence.ConfluenceInterface(
                self.confluence_url,
                self.confluence_space,
                base_page,
                username=self.confluence_username,
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
            self.confluence_username,
            token=self.confluence_token,
        )

        if profile_space is not None:
            self.profiler = PerformanceProfiler(
                self.confluence_url,
                profile_space,
                profile_basepage,
                self.confluence_username,
                self.confluence_token,
            )
            self.profiler.init_profile_page()
        else:
            self.profiler = None

    @classmethod
    def hash_hex(cls, s: str, length: int = 6) -> str:
        """Hash a string and digest it to a hex string of a given length.

        Parameters
        ----------
        s : str
            String to hash.
        length : int, optional
            Length (even) to digest to. Default is 6.

        Returns
        -------
        str
            Hex digest of length ``length``.
        """
        return hashlib.shake_128(s.encode("utf-8")).hexdigest(int(length / 2))

    def _shared_runstamp(self) -> str:
        """Return a run timestamp (``%y%m%d-%H%M``) identical across ranks.

        Multi-node aggregation needs every rank to agree on report names
        (hence result folders) so they can cooperate on one aggregation.
        Rank 0 writes the stamp to a shared file in the root folder; worker
        ranks read it (waiting briefly for it to appear).

        Returns
        -------
        str
            The shared run timestamp.

        Raises
        ------
        RuntimeError
            On a worker rank, if the stamp does not appear within the
            wait window.
        """
        stamp_file = os.path.join(self.root_folder, ".pwf_runstamp")
        if self.rank == 0:
            stamp = datetime.now().strftime("%y%m%d-%H%M")
            os.makedirs(self.root_folder, exist_ok=True)
            with open(stamp_file, "w") as f:
                f.write(stamp)
            return stamp
        # worker ranks: wait for rank 0 to publish the stamp
        for _ in range(600):
            try:
                with open(stamp_file) as f:
                    stamp = f.read().strip()
                if stamp:
                    return stamp
            except FileNotFoundError:
                pass
            time.sleep(1)
        raise RuntimeError(
            "Timed out waiting for the shared run timestamp from rank 0."
        )

    def get_configs(
        self,
        report_name: str,
        root_folder: str,
        cell_type: str | None = None,
        cell_name: str | None = None,
        camera_info: dict | None = None,
    ) -> tuple[dict, dict]:
        """Build reporter and analysis config dicts for one workflow run.

        The result location is nested under ``root_folder`` by ``cell_type``
        and ``cell_name`` when those are provided.

        Parameters
        ----------
        report_name : str
            Name of the report (and Confluence parent page).
        root_folder : str
            Root result folder for the analysis.
        cell_type, cell_name : str, optional
            Optional nesting levels for the result location.
        camera_info : dict, optional
            Camera metadata passed through to the analysis config.

        Returns
        -------
        reporter_config : dict
            Reporter configuration including Confluence settings.
        analysis_config : dict
            General analysis configuration.
        """
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
                "username": self.confluence_username,
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

    def build_overview_body(
        self,
        title: str,
        result_folder: str,
        workflow_config: dict | None = None,
        ntiles: int | None = None,
        intro_html: str = "",
    ) -> str:
        """Assemble a run-overview page body.

        Combines run metadata and an optional collapsible config snapshot via
        :func:`confluence.overview_body`. Gathers SLURM job/node, host,
        software versions and folder locations so the page is self-describing.
        Shared across coordinators so single and aggregation overview pages
        stay consistent; the actual storage-format rendering lives in
        :func:`confluence.overview_body`.

        Parameters
        ----------
        title : str
            Page heading.
        result_folder : str
            Full path to the run's results and script folder.
        workflow_config : dict, optional
            Workflow definition, rendered as a collapsible YAML snapshot.
        ntiles : int, optional
            Number of single datasets (shown only when provided).
        intro_html : str, optional
            Optional intro paragraph(s) as valid storage-format HTML.

        Returns
        -------
        str
            Confluence storage-format HTML body.
        """
        env = os.environ
        gpus = (
            env.get("SLURM_GPUS_ON_NODE")
            or env.get("SLURM_JOB_GPUS")
            or env.get("CUDA_VISIBLE_DEVICES")
            or "none"
        )
        try:
            from picasso_workflow import __version__ as pw_version
        except Exception:
            pw_version = "unknown"
        try:
            from picasso import __version__ as picasso_version
        except Exception:
            picasso_version = "unknown"

        rows = [
            ("Run timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        if ntiles is not None:
            rows.append(("# single datasets aggregated", ntiles))
        rows += [
            ("Results / script folder", result_folder),
            ("SLURM log folder", os.path.join(result_folder, "logs")),
            ("SLURM job ID", env.get("SLURM_JOB_ID", "N/A (not a SLURM job)")),
            ("SLURM job name", env.get("SLURM_JOB_NAME", "N/A")),
            ("SLURM partition", env.get("SLURM_JOB_PARTITION", "N/A")),
            (
                "SLURM node(s)",
                env.get(
                    "SLURM_JOB_NODELIST", env.get("SLURMD_NODENAME", "N/A")
                ),
            ),
            ("SLURM submit dir", env.get("SLURM_SUBMIT_DIR", "N/A")),
            ("CPUs per task", env.get("SLURM_CPUS_PER_TASK", "N/A")),
            ("GPUs", gpus),
            ("Host", platform.node()),
            ("picasso-workflow version", pw_version),
            ("picasso version", picasso_version),
        ]
        return confluence.overview_body(
            title, rows, intro_html=intro_html, config=workflow_config
        )


class SingleWorkflowCoordinator(AbstractWorkflowCoordinator):
    """Coordinate the analysis of a single measurement (e.g. one Exchange
    round)."""

    def __init__(
        self,
        src_loc_file,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        confluence_username=None,
        base_page="base_page",
        dest_machine=None,
        always_save=False,
        profile_space=None,
        profile_basepage=None,
    ):
        if src_loc_file is None:
            # "no input files" mode: run the workflow exactly once, with
            # no dataset filepath mapping. Useful for workflows whose
            # modules load data themselves (e.g. spinna_batch reads its
            # source files from a config csv).
            self.dataset_filepaths = {}
            self.tile_entries = {"#tags": [analysis_name]}
        else:
            self.dataset_filepaths = io.load_info(src_loc_file)[0]
            self.tile_entries = {
                "filepath": list(self.dataset_filepaths.values()),
                "#tags": list(self.dataset_filepaths.keys()),
            }
        self.analysis_name = os.path.split(working_folder)[-1]

        super().__init__(
            analysis_name=analysis_name,
            working_folder=working_folder,
            confluence_url=confluence_url,
            confluence_space=confluence_space,
            confluence_token=confluence_token,
            confluence_username=confluence_username,
            base_page=base_page,
            dest_machine=dest_machine,
            always_save=always_save,
            profile_space=profile_space,
            profile_basepage=profile_basepage,
        )

    def prepare_analysis(
        self,
        workflow_modules: list[tuple],
        continue_previous_runners: bool = False,
    ) -> list[dict]:
        """Tile the workflow over datasets and build per-dataset runners.

        Parameters
        ----------
        workflow_modules : list of tuple
            The modules to run, as ``(module_name, parameters)``.
        continue_previous_runners : bool, optional
            Whether to continue previously aborted runners. Default is False.

        Returns
        -------
        list of dict
            One ``{"wr": WorkflowRunner, "dataset_name": str}`` entry per
            dataset handled by this rank.
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
                continue_previous_runner=continue_previous_runners,
                postfix="",
            )
            run_wr_kwargs.append(
                {
                    "wr": wr,
                    "dataset_name": tag,
                }
            )
        return run_wr_kwargs

    def run_analysis(
        self,
        workflow_modules: list[tuple],
        continue_previous_runners: bool = False,
    ) -> None:
        """Prepare and run the per-dataset workflows handled by this rank.

        Parameters
        ----------
        workflow_modules : list of tuple
            The modules to run, as ``(module_name, parameters)``.
        continue_previous_runners : bool, optional
            Whether to continue previously aborted runners. Default is False.
        """
        run_wr_kwargs = self.prepare_analysis(
            workflow_modules, continue_previous_runners
        )

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

    def run_wr(self, wr: WorkflowRunner, dataset_name: str) -> None:
        """Run a single :class:`WorkflowRunner` and close figures afterwards.

        Parameters
        ----------
        wr : WorkflowRunner
            The configured runner to execute.
        dataset_name : str
            Dataset tag, used only for logging.
        """
        logger.debug(f"starting to analyse {dataset_name}")
        wr.run()
        logger.debug(f"finished analysing {dataset_name}")
        plt.close("all")


class AggregationWorkflowCoordinator(AbstractWorkflowCoordinator):
    """Coordinate the analysis of one FOV with multiple targets.

    For example, the labeling efficiency of one cell across several imaging
    targets.
    """

    def __init__(
        self,
        src_loc_file,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        confluence_username=None,
        base_page="base_page",
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
            analysis_name=analysis_name,
            working_folder=working_folder,
            confluence_url=confluence_url,
            confluence_space=confluence_space,
            confluence_token=confluence_token,
            confluence_username=confluence_username,
            base_page=base_page,
            investigation_description=investigation_description,
            dest_machine=dest_machine,
            always_save=always_save,
            profile_space=profile_space,
            profile_basepage=profile_basepage,
        )

    def prepare_analysis(
        self,
        workflow_modules_sgl: list[tuple],
        workflow_modules_agg: list[tuple],
        continue_previous_runners: bool = False,
    ) -> list[dict]:
        """Build the per-group aggregation runners for this rank.

        Distributes whole aggregation groups across SLURM ranks when there are
        at least as many groups as ranks; otherwise all ranks cooperate on
        each group (single workflows are distributed inside
        :meth:`AggregationWorkflowRunner.run`).

        Parameters
        ----------
        workflow_modules_sgl : list of tuple
            Modules run in the first stage (per individual round).
        workflow_modules_agg : list of tuple
            Modules run in the second stage (aggregation across rounds).
        continue_previous_runners : bool, optional
            Whether to continue previously aborted runners. Default is False.

        Returns
        -------
        list of dict
            One ``{"awr": AggregationWorkflowRunner, "report_name": str}``
            entry per aggregation group handled by this rank.
        """
        # make sure different nodes query confluence at different times
        time.sleep(3 * self.rank)

        run_awr_kwargs = []

        # Two-level parallelism across SLURM ranks:
        #   * When there are at least as many aggregation groups as ranks,
        #     distribute whole groups across ranks (each rank runs its groups
        #     end to end; no intra-group split). This avoids idle ranks for
        #     many-groups/few-singles runs.
        #   * Otherwise (e.g. a single multicolor aggregation), all ranks
        #     cooperate on each group and the single workflows within it are
        #     distributed across ranks inside AggregationWorkflowRunner.run().
        # A shared timestamp keeps report names (and result folders)
        # identical across ranks for the cooperative case.
        runstamp = self._shared_runstamp()
        n_groups = len(self.dataset_filepaths)
        distribute_groups = n_groups >= self.size

        execution_item = -1
        for datasets in self.dataset_filepaths:
            execution_item += 1
            if distribute_groups and execution_item % self.size != self.rank:
                continue
            # Rank that owns the group's Confluence side effects: the single
            # owning rank in group-distribution mode, else rank 0.
            owns_page = distribute_groups or self.rank == 0
            # rank/size handed to the runner for intra-group single
            # distribution (disabled when whole groups are distributed).
            if distribute_groups:
                runner_rank, runner_size = 0, 1
            else:
                runner_rank, runner_size = self.rank, self.size

            if rname := datasets.get("report_name"):
                report_name = rname + "_" + runstamp
                dedicated_page = True
                # create the corresponding confluence page (page owner only)
                if owns_page:
                    try:
                        ci = confluence.ConfluenceInterface(
                            self.confluence_url,
                            self.confluence_space,
                            self.root_page,
                            username=self.confluence_username,
                            token=self.confluence_token,
                        )
                        ci.create_page(report_name, "")
                    except Exception:
                        pass
            else:
                report_name = self.analysis_name
                dedicated_page = False

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
                continue_previous_runner=continue_previous_runners,
                single_workflow_parallel=False,
                postfix="",
                rank=runner_rank,
                size=runner_size,
            )

            # Write the run overview onto the dedicated aggregation page
            # (page owner only). Skipped when the run reuses the root page,
            # i.e. no per-dataset report_name, to avoid clobbering the
            # investigation page.
            if dedicated_page and owns_page:
                try:
                    ntiles = getattr(
                        getattr(awr, "parameter_tiler", None), "ntiles", None
                    )
                    body = self.build_overview_body(
                        "Aggregation analysis results",
                        getattr(awr, "result_folder", self.root_folder),
                        workflow_config=workflow_modules_multi,
                        ntiles=ntiles,
                        intro_html=(
                            "<p>Overview page of an aggregation run. The "
                            "child pages below contain the individual "
                            "single-dataset workflows and their "
                            "aggregation.</p>"
                        ),
                    )
                    pgid, _ = self.ci.get_page_properties(report_name)
                    self.ci.update_page_content(
                        report_name, pgid, body, replace=True
                    )
                except Exception as e:
                    logger.debug(f"Could not write aggregation overview: {e}")

            run_awr_kwargs.append(
                {
                    "awr": awr,
                    "report_name": report_name,
                }
            )
        return run_awr_kwargs

    def run_analysis(
        self,
        workflow_modules_sgl: list[tuple],
        workflow_modules_agg: list[tuple],
        continue_previous_runners: bool = False,
    ) -> None:
        """Prepare and run the aggregation workflows handled by this rank.

        Parameters
        ----------
        workflow_modules_sgl : list of tuple
            Modules run in the first stage (per individual round).
        workflow_modules_agg : list of tuple
            Modules run in the second stage (aggregation across rounds).
        continue_previous_runners : bool, optional
            Whether to continue previously aborted runners. Default is False.
        """
        run_awr_kwargs = self.prepare_analysis(
            workflow_modules_sgl,
            workflow_modules_agg,
            continue_previous_runners,
        )

        print(f"rank {self.rank}, size {self.size}: running {run_awr_kwargs}")

        for kwargs in run_awr_kwargs:
            self.run_awr(**kwargs)

            if self.profiler is not None:
                self.profiler.append_profile_page(kwargs["awr"].all_results)

    def run_awr(
        self, awr: AggregationWorkflowRunner, report_name: str
    ) -> None:
        """Run one :class:`AggregationWorkflowRunner` and close figures.

        Parameters
        ----------
        awr : AggregationWorkflowRunner
            The configured aggregation runner to execute.
        report_name : str
            Report name, used only for logging.
        """
        logger.debug(f"starting to analyse {report_name}")
        awr.run()
        logger.debug(f"finished analysing {report_name}")
        plt.close("all")


class InvestigationCoordinator(AbstractWorkflowCoordinator):
    """Coordinate the analysis of a measurement series.

    Covers multiple conditions / cell types, cells and target molecules.
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
        confluence_username=None,
        base_page="base_page",
        dest_machine="hpcl8",
        investigation_description="",
        always_save=False,
        iterations=1,
        underscores=[1, 0, 0, 0],
        profile_space=None,
        profile_basepage=None,
    ):
        """Initialize the investigation coordinator.

        Parameters
        ----------
        iterations : int, optional
            Number of times to call the workflow on the same data with an
            iterator (e.g. to select different cells). Default is 1.

        Notes
        -----
        Remaining parameters mirror those of
        :class:`AbstractWorkflowCoordinator` plus the source location
        (``src_loc``), the ``receptors`` to analyse and the per-level
        ``underscores`` used to parse the source.
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
            analysis_name=analysis_name,
            working_folder=working_folder,
            confluence_url=confluence_url,
            confluence_space=confluence_space,
            confluence_token=confluence_token,
            confluence_username=confluence_username,
            base_page=base_page,
            investigation_description=investigation_description,
            dest_machine=dest_machine,
            always_save=always_save,
            profile_space=profile_space,
            profile_basepage=profile_basepage,
        )

    def prepare_sglcell_analysis(self, get_workflow_modules) -> list[dict]:
        """Build per-cell aggregation runners for this rank.

        Parameters
        ----------
        get_workflow_modules : callable
            Callable taking ``cell_type, report_name, datasets`` (and ``i``)
            and returning the ``workflow_modules_multi`` dict.

        Returns
        -------
        list of dict
            One runner kwargs entry per cell handled by this rank.
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

    def prepare_mergecell_analysis(
        self, get_mergecell_workflow_modules
    ) -> list[dict]:
        """Build per-cell-type "merge cell" aggregation runners for this rank.

        Parameters
        ----------
        get_mergecell_workflow_modules : callable
            Callable taking ``cell_type, fp_workflows, report_names, datasets``
            and returning the ``workflow_modules_multi`` dict.

        Returns
        -------
        list of dict
            One runner kwargs entry per cell type handled by this rank.
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

    def run_awr(
        self,
        awr: AggregationWorkflowRunner,
        cell_type: str,
        cell_name: str,
        queue,
        lock,
        fp_dfa2: str,
    ) -> tuple:
        """Run one cell's aggregation and record per-module success.

        Runs the aggregation workflow, then appends a success row per module
        to the shared ``fp_dfa2`` spreadsheet (guarded by ``lock``) and pushes
        the result onto ``queue``.

        Parameters
        ----------
        awr : AggregationWorkflowRunner
            The configured aggregation runner to execute.
        cell_type, cell_name : str
            Identifiers used for logging and the success spreadsheet.
        queue : multiprocessing.Queue
            Queue the ``(cell_type, cell_name, successes)`` result is put on.
        lock : multiprocessing.Lock
            Lock serialising writes to the shared spreadsheet.
        fp_dfa2 : str
            Path to the per-module success spreadsheet (``.xlsx``).

        Returns
        -------
        tuple
            ``(cell_type, cell_name, successes)`` where ``successes`` maps
            module name to its success flag.
        """
        logger.debug(f"starting to analyse {cell_type}, {cell_name}")
        awr.run()
        logger.debug(f"finished analysing {cell_type}, {cell_name}")
        plt.close("all")
        agg_results = awr.all_results["aggregation"]
        successes = {}
        for mod, res in agg_results.items():
            successes[mod] = res.get("success", False)
            logger.debug(f"""{cell_type}, {cell_name}, {mod}:
                {res.get('success', False)}""")
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
        summary_page_title: str,
        summary_columns: list[str],
    ) -> str:
        """Create a Confluence summary page with a header table.

        Parameters
        ----------
        ci : confluence.ConfluenceInterface
            Interface used to create the page.
        summary_page_title : str
            Title of the summary page.
        summary_columns : list of str
            Column headers (besides the leading ``Dataset`` column).

        Returns
        -------
        str
            The summary page title (unchanged), for convenient chaining.
        """
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

    def extract_fpfig_from_results(
        self, awr: AggregationWorkflowRunner, figloc: list[list[str]]
    ) -> list[str]:
        """Extract figure filepaths from a runner's ``all_results``.

        Parameters
        ----------
        awr : AggregationWorkflowRunner
            The runner to extract results from.
        figloc : list of list of str
            One entry per summary-page column; each inner list is the key
            path within ``all_results``, e.g.
            ``["aggregation", "00_ripleysk_average2", "fp_figmeanvals"]``.

        Returns
        -------
        list of str
            The extracted figure filepaths, one per column.
        """
        fp_figs = [
            reduce(lambda d, key: d[key], loc, awr.all_results)
            for loc in figloc
        ]
        return fp_figs

    def add_to_summary_page(
        self,
        summary_page_title: str,
        dataset: str,
        data_list: list,
        data_types: str | list = "img",
    ) -> None:
        """Append a dataset row to a Confluence summary table.

        Parameters
        ----------
        summary_page_title : str
            Title of the summary page to append to.
        dataset : str
            Row label for the dataset.
        data_list : list
            Data to enter into the summary table, one item per column.
        data_types : str or list, optional
            The data type(s) of the entries. ``"img"`` uploads and displays an
            image from a file path; any other type is entered directly. If a
            single str, it applies to all entries; a list may also contain
            per-entry dicts with a required ``"type"`` key and optional
            formatting keys. Default is ``"img"``.
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
        summary_page_title: str | None = None,
        summary_columns: list[str] | None = None,
        figloc: list[list[str]] | None = None,
        data_types: dict | str = {"type": "img"},
    ) -> None:
        """Run the per-cell analyses and populate the summary page.

        Parameters
        ----------
        get_sglcell_workflow_modules : callable
            Callable returning the ``workflow_modules_multi`` for a cell.
        summary_page_title : str, optional
            Base title of the summary page; the analysis name is appended.
        summary_columns : list of str, optional
            Column headers for the summary table.
        figloc : list of list of str, optional
            Key paths into each runner's ``all_results`` for the figures to
            display, one per column (see :meth:`extract_fpfig_from_results`).
        data_types : dict or str, optional
            Data type spec for the summary entries. Default
            ``{"type": "img"}``.
        """
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
        summary_page_title: str,
        summary_columns: list[str],
        figloc: list[list[str]],
        data_types: str | list = "img",
    ) -> None:
        """Run the per-cell-type "merge cell" analyses and fill the summary.

        Parameters
        ----------
        get_mergecell_workflow_modules : callable
            Callable returning the ``workflow_modules_multi`` for a cell type.
        summary_page_title : str
            Base title of the summary page; the analysis name is appended.
        summary_columns : list of str
            Column headers for the summary table.
        figloc : list of list of str
            Key paths into each runner's ``all_results`` for the figures to
            display, one per column.
        data_types : str or list, optional
            Data type spec for the summary entries. Default is ``"img"``.
        """
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
    """Collect per-module resource-usage metrics onto a Confluence page."""

    def __init__(
        self,
        confluence_url: str,
        confluence_space: str,
        base_page: str,
        confluence_username: str | None,
        confluence_token: str | None,
    ):
        """Initialize the profiler's Confluence interface.

        Parameters
        ----------
        confluence_url : str
            Base URL of the Confluence instance.
        confluence_space : str
            Key of the Confluence space.
        base_page : str
            Title of the parent page for the profiling page.
        confluence_username, confluence_token : str or None
            Confluence credentials.
        """
        self.ci = confluence.ConfluenceInterface(
            confluence_url,
            confluence_space,
            base_page,
            username=confluence_username,
            token=confluence_token,
        )

    def init_profile_page(
        self,
        page_title: str = "Cluster Performance Profiling",
    ) -> str:
        """Create the profiling page with its header table.

        Parameters
        ----------
        page_title : str, optional
            Title of the profiling page. Default is
            ``"Cluster Performance Profiling"``.

        Returns
        -------
        str
            The page title (unchanged), for convenient chaining.
        """
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

    def append_profile_page(self, all_results: dict) -> None:
        """Append resource-usage rows for one run to the profiling page.

        Parameters
        ----------
        all_results : dict
            A runner's ``all_results``; each module's metrics are read from
            the configured data columns and appended as a table row.
        """
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

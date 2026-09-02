#!/usr/bin/env python
"""Orchestrate picasso analysis and Confluence reporting.

Implements :class:`WorkflowRunner`, which runs a single-dataset workflow as a
sequence of modules and publishes each module's results to Confluence, and
:class:`AggregationWorkflowRunner`, which splits the work into per-dataset
sub-workflows (optionally across SLURM ranks) and then aggregates them.

Author: Heinrich Grabmayr
Initial date: March 7, 2024
"""

from __future__ import annotations

import os
import time
from datetime import datetime

# import logging
from loguru import logger
import inspect
import yaml
import copy
import re
import traceback

from picasso_workflow.analyse import AutoPicasso, AutoPicassoError
from picasso_workflow.confluence import (
    ConfluenceReporter,
    ConfluenceInterface,
    ConfluenceInterfaceError,
    aggregation_abort_body,
    _PARAM_BLACKLIST,
)
from picasso_workflow.html_reporter import (
    HTMLReporter,
    write_aggregation_index,
)
from picasso_workflow.modulespec import Scope, validate_workflow
from picasso_workflow.util import (
    AbstractModuleCollection,
    ParameterCommandExecutor,
    ParameterTiler,
    DictSimpleTyper,
)
from picasso_workflow import progress as pwprogress
from picasso_workflow.progress import (
    ProgressManager,
    RUNNING,
    DONE,
    FAILED,
    ABORTED,
)


# For loading yaml files
def python_tuple_constructor(loader, node):
    """Construct a tuple from a ``tag:yaml.org,2002:python/tuple`` YAML node.

    Registered on the safe loader so workflow configs that serialise tuples
    round-trip back to tuples instead of failing to load.
    """
    return tuple(loader.construct_sequence(node))


# # Register the custom constructors
yaml.constructor.SafeConstructor.add_constructor(
    "tag:yaml.org,2002:python/tuple", python_tuple_constructor
)


def _log_workflow_validation(steps, scope, label):
    """Run pre-flight workflow validation and log findings (warn-only).

    Phase-2 integration of :func:`picasso_workflow.modulespec.validate_workflow`:
    issues are logged as warnings but never block execution. Wrapped so a bug
    in the (still-maturing) annotation layer can never abort a real run.

    Parameters
    ----------
    steps : iterable
        Workflow steps as ``(module_name, parameters)`` tuples.
    scope : Scope
        The workflow scope to validate against.
    label : str
        Human-readable label for the log messages (which workflow this is).
    """
    try:
        errors = validate_workflow(steps, scope)
    except Exception as e:  # never let validation break a run
        logger.debug(f"{label}: workflow validation skipped ({e!r}).")
        return
    if errors:
        logger.warning(
            f"{label}: pre-flight validation found {len(errors)} issue(s) "
            "(warn-only, not blocking):"
        )
        for err in errors:
            logger.warning(f"  {err}")
    else:
        logger.debug(f"{label}: pre-flight validation passed.")


# logger = logging.getLogger(__name__)


class AggregationWorkflowRunner:
    """Run several single-dataset workflows and aggregate their results.

    Many analyses split into per-dataset sub-workflows (e.g. when multiple
    DNA-PAINT datasets are evaluated individually) whose results are then
    combined in an aggregation step. This class coordinates that pattern,
    optionally distributing the single-dataset workflows across SLURM ranks.
    """

    def __init__(self, postfix: str | None = None):
        """Initialize the runner.

        Parameters
        ----------
        postfix : str, optional
            Postfix used to load prior analyses, formatted ``%y%m%d-%H%M``.
            If None, a new postfix is generated from the current time.
        """
        if postfix:
            self.postfix = postfix
        else:
            self.postfix = datetime.now().strftime("%y%m%d-%H%M")
        self.continue_workflow = False
        self.single_workflow_parallel = False
        self.sgl_workflow_locations = []
        self.cpage_names = []
        self._html_reporting = False
        self._agg_report_folder = None
        # SLURM task identity for multi-node parallelism of the single
        # workflows. Defaults to a single (rank 0) process off-cluster.
        self.rank = int(os.getenv("SLURM_PROCID") or 0)
        self.size = int(os.getenv("SLURM_NTASKS") or 1)

    @classmethod
    def config_from_dicts(
        cls,
        reporter_config: dict,
        analysis_config: dict,
        aggregation_workflow: dict,
        postfix: str | None = None,
        continue_previous_runner: bool = False,
        single_workflow_parallel: bool = False,
        rank: int | None = None,
        size: int | None = None,
    ) -> "AggregationWorkflowRunner":
        """Build a configured runner from plain config dicts.

        Initialization is kept out of ``__init__`` to preserve flexibility for
        alternative entry points in the future (config file names, a web API,
        etc.).

        Parameters
        ----------
        reporter_config : dict
            Configuration of the reporter (currently the Confluence reporter).
        analysis_config : dict
            General analysis configuration.
        aggregation_workflow : dict
            The workflow modules to run, split into individual runs, with keys:

            ``single_dataset_tileparameters`` : dict
                Parameters that must be adjusted for every individual
                single-dataset analysis.
            ``single_dataset_modules`` : list of tuple
                ``workflow_modules`` of :class:`WorkflowRunner` describing the
                per-dataset analysis.
            ``aggregation_modules`` : list of tuple
                ``workflow_modules`` of :class:`WorkflowRunner` describing the
                aggregation analysis (e.g. labeling efficiency, RESI).
        postfix : str, optional
            Postfix used to load prior analyses, formatted ``%y%m%d-%H%M``.
            If None, a new postfix is generated.
        continue_previous_runner : bool, optional
            Continue a previous analysis that aborted (e.g. at a manual step).
            If no previous analysis exists in that folder, a new one is
            created. Default is False.
        single_workflow_parallel : bool, optional
            Whether the single-dataset workflows run in parallel. Default is
            False.
        rank, size : int, optional
            SLURM task identity overriding the environment-derived values, used
            to control how single workflows are distributed across ranks.

        Returns
        -------
        AggregationWorkflowRunner
            The configured runner instance.
        """
        # check whether the report_name has a postfix-format already
        # Check if report_name already has a postfix pattern
        report_name = reporter_config["report_name"]
        postfix_pattern = r"_(\d{6}-\d{4})$"
        match = re.search(postfix_pattern, report_name)

        extracted_postfix = postfix
        if match:
            # Extract existing postfix and validate format
            existing_postfix = match.group(1)
            try:
                # datetime.strptime(existing_postfix, "%y%m%d-%H%M")
                # Valid postfix found, separate base name from postfix
                base_report_name = report_name[: match.start()]
                reporter_config["report_name"] = base_report_name
                extracted_postfix = existing_postfix
            except ValueError:
                # Invalid postfix format, treat as part of the name
                pass

        if continue_previous_runner:
            folder = analysis_config["result_location"]
            report_name = reporter_config["report_name"]
            # Use extracted postfix if available, otherwise
            # check for previous runner
            if extracted_postfix is not None:
                postfix = extracted_postfix
            else:
                postfix = cls._check_previous_runner(folder, report_name)
            logger.debug(f"Found postfix: {postfix}")
            if postfix is not None:
                report_name = report_name + "_" + postfix
                runner_folder = os.path.join(folder, report_name)
                try:
                    instance = cls.load(runner_folder)
                    return instance
                except FileNotFoundError:
                    logger.debug(f"Could not load runner from {runner_folder}")

        # If we have an extracted postfix but aren't continuing, use it
        if extracted_postfix is not None and not continue_previous_runner:
            postfix = extracted_postfix

        if (
            sgltilepars := aggregation_workflow.get(
                "single_dataset_tileparameters"
            )
        ) is None:
            raise KeyError("""aggregation_workflow missing
                "single_dataset_tileparameters".""")
        instance = cls(postfix)
        # The coordinator may override the SLURM-derived rank/size to control
        # how the single workflows are distributed (e.g. run all locally when
        # whole aggregation groups are already distributed across ranks).
        if rank is not None:
            instance.rank = rank
        if size is not None:
            instance.size = size
        instance.single_workflow_parallel = single_workflow_parallel
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
        # analysis result directory; computed before the Confluence page so
        # its location can be documented on the overview page.
        instance.result_folder = os.path.join(
            analysis_config["result_location"], report_name
        )
        if confluence_config := reporter_config.get("ConfluenceReporter"):
            instance._initialize_confluence_interface(**confluence_config)
            # The overview content (run metadata + config snapshot) is
            # written by the AggregationWorkflowCoordinator, which has the
            # orchestration context. Here we only ensure the page exists.
            # On multi-node runs only rank 0 creates it; worker ranks still
            # set parent_page_title so their child pages nest correctly.
            # Resolve the parent page by id (returned by create_page, or
            # already supplied by the coordinator) rather than by title:
            # Confluence Cloud's title search index lags page creation, so a
            # title lookup in a child workflow can fail to find a page that
            # was just created on this (or another) rank.
            parent_page_id = confluence_config.get("parent_page_id")
            if instance.rank == 0:
                try:
                    created_id = instance.ci.create_page(report_name, "")
                    parent_page_id = parent_page_id or created_id
                except ConfluenceInterfaceError:
                    logger.debug(
                        "Error creating page, it already exists. Continuing"
                    )
            reporter_config["ConfluenceReporter"][
                "parent_page_title"
            ] = report_name
            reporter_config["ConfluenceReporter"][
                "parent_page_id"
            ] = parent_page_id
            instance.cpage_names.append(report_name)

        # HTML reporting: every child WorkflowRunner writes its own
        # report.html into its own result subfolder (the HTMLReporter config
        # propagates via reporter_config). A fixed report_dir would make the
        # children collide, so it is dropped here; the aggregation overview
        # index always goes into the aggregation result folder.
        if (html_cfg := reporter_config.get("HTMLReporter")) is not None:
            instance._html_reporting = True
            if isinstance(html_cfg, dict) and html_cfg.get("report_dir"):
                logger.debug(
                    "Ignoring HTMLReporter.report_dir for the aggregation "
                    "run; child reports use per-dataset folders."
                )
                html_cfg.pop("report_dir", None)

        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        # reporter_config['report_name'] = report_name
        # create analysis result directory
        try:
            os.mkdir(instance.result_folder)
        except FileExistsError:
            pass

        instance.aggregation_workflow = aggregation_workflow
        return instance

    @classmethod
    def _check_previous_runner(
        cls, folder: str, report_name: str
    ) -> str | None:
        """Find the postfix of the latest previous runner in a location.

        Parameters
        ----------
        folder : str
            The folder to look in.
        report_name : str
            The name of the report.

        Returns
        -------
        str or None
            The postfix of the latest previous runner in that location, or
            None if none are found.
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
        self,
        base_url: str,
        space_key: str,
        parent_page_title: str,
        username: str | None = None,
        token: str | None = None,
        parent_page_id: str | None = None,
    ) -> None:
        """Create the Confluence interface used to publish the overview page.

        Parameters
        ----------
        base_url : str
            Base URL of the Confluence instance.
        space_key : str
            Key of the Confluence space to write into.
        parent_page_title : str
            Title of the page under which new pages are nested.
        username, token : str, optional
            Confluence credentials.
        parent_page_id : str, optional
            Id of the parent page; when given, skips the title-based lookup
            (see :class:`~picasso_workflow.confluence.ConfluenceInterface`).
        """
        self.ci = ConfluenceInterface(
            base_url=base_url,
            space_key=space_key,
            parent_page_title=parent_page_title,
            username=username,
            token=token,
            parent_page_id=parent_page_id,
        )

    def run(self) -> bool | None:
        """Individualize the aggregation workflow and run it.

        Runs every single-dataset workflow (distributed across ranks when
        running under SLURM), then aggregates on rank 0.

        Returns
        -------
        bool or None
            On worker ranks, whether this rank's single datasets all
            succeeded. Rank 0 (and single-task runs) return None after the
            aggregation step; failures are raised as :class:`WorkflowError`.
        """
        # pre-flight: validate both sub-workflows (warn-only, non-blocking).
        # single_dataset_modules run per dataset (single scope);
        # aggregation_modules run on the pooled results (aggregation scope).
        _log_workflow_validation(
            self.aggregation_workflow.get("single_dataset_modules", []),
            Scope.SINGLE,
            "aggregation: single-dataset modules",
        )
        _log_workflow_validation(
            self.aggregation_workflow.get("aggregation_modules", []),
            Scope.AGGREGATION,
            "aggregation: aggregation modules",
        )

        # Only rank 0 persists the (shared) aggregation state; worker ranks
        # would race on the same AggregationWorkflowRunner.yaml.
        if self.rank == 0:
            self.save(self.result_folder)
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

        n_sgl = len(tags)
        sgl_dataset_success = [None] * n_sgl
        sgl_folders = [None] * n_sgl

        # progress tracking at the aggregation level: one entry per single
        # dataset. Rank 0 owns the shared progress.json; worker ranks write a
        # per-rank file (see ProgressManager). Each single WorkflowRunner also
        # writes its own progress.json in its subfolder.
        self.progress = ProgressManager(
            self.result_folder,
            kind="aggregation",
            rank=self.rank,
            size=self.size,
            report_name=report_name,
        )
        self.progress.datasets_init(tags)
        self.progress.mark_running()

        # Multi-node parallelism: distribute the single-dataset workflows
        # across the SLURM ranks by dynamic self-scheduling. Each rank walks
        # the dataset list and races to claim the next one (an atomic mkdir on
        # the shared filesystem); the winner runs it and only then advances.
        # A rank that finishes a light dataset immediately claims the next
        # unclaimed one, so both nodes stay busy even when the heavy
        # (spot-rich) datasets happen to cluster - which a static i %% size
        # split cannot avoid. Results and a completion marker go to the shared
        # result folder; rank 0 then waits for every marker, loads the results
        # produced by other ranks from disk, and runs the aggregation. With a
        # single task (off-cluster) every dataset runs here.
        claim_dir = self._claim_dir()
        if self.size > 1:
            os.makedirs(claim_dir, exist_ok=True)
        logger.debug(
            f"Aggregation runner rank {self.rank}/{self.size} claiming "
            f"single datasets dynamically ({n_sgl} total)."
        )

        for i, (parameter_set, tag) in enumerate(
            zip(individual_parametersets, tags)
        ):
            sgl_name = report_name + f"_sgl_{i:02d}"
            if tag:
                sgl_name += f"_{tag}"
            sgl_folders[i] = os.path.join(
                self.result_folder, sgl_name + "_" + self.postfix
            )
            if self.size > 1 and not self._claim_dataset(claim_dir, i):
                continue  # claimed by another rank

            sgl_wkfl_reporter_config["report_name"] = sgl_name
            sgl_wkfl_analysis_config["result_location"] = self.result_folder
            if self.continue_workflow:
                try:
                    logger.debug(
                        f"loading WorkflowRunner from {sgl_folders[i]}"
                    )
                    wr = WorkflowRunner.load(sgl_folders[i])
                except Exception:
                    logger.debug("loading did not work. creating from dict.")
                    wr = WorkflowRunner.config_from_dicts(
                        copy.deepcopy(sgl_wkfl_reporter_config),
                        copy.deepcopy(sgl_wkfl_analysis_config),
                        parameter_set,
                        postfix=self.postfix,
                    )
            else:
                logger.debug("not continuing workflow. starting new.")
                wr = WorkflowRunner.config_from_dicts(
                    copy.deepcopy(sgl_wkfl_reporter_config),
                    copy.deepcopy(sgl_wkfl_analysis_config),
                    parameter_set,
                    postfix=self.postfix,
                )
            self.cpage_names.append(wr.reporter_config["report_name"])
            self.progress.dataset_update(i, RUNNING)
            # Never let an unhandled error escape before the completion
            # marker is written - otherwise rank 0 would wait on the barrier
            # until timeout. A failed single still marks the run as failed,
            # which aborts the aggregation below.
            try:
                success = wr.run()
            except Exception as e:
                logger.error(f"Single dataset {i} ({tag}) failed: {e}")
                logger.error(traceback.format_exc())
                success = False
            self.progress.dataset_update(i, DONE if success else FAILED)
            sgl_dataset_success[i] = success
            self.all_results["single_dataset"][i] = getattr(
                wr, "results", None
            )
            if self.rank == 0:
                self.save(self.result_folder)
            if self.size > 1:
                self._write_single_marker(sgl_folders[i], success)

        # Worker ranks are done once their share is finished and marked;
        # the aggregation is performed by rank 0 only.
        if self.size > 1 and self.rank != 0:
            owned = [s for s in sgl_dataset_success if s is not None]
            logger.debug(
                f"Rank {self.rank} finished its single datasets "
                f"({sum(bool(s) for s in owned)}/{len(owned)} ok); "
                "leaving aggregation to rank 0."
            )
            rank_ok = all(owned) if owned else True
            self.progress.finish(DONE if rank_ok else FAILED)
            return rank_ok

        # Rank 0 (or a single-task run): wait for the single datasets handled
        # by other ranks and load their results from disk.
        if self.size > 1:
            self._wait_for_single_markers(sgl_folders)
            for i in range(n_sgl):
                if sgl_dataset_success[i] is not None:
                    continue  # ran on this rank, already in memory
                status = self._read_single_marker(sgl_folders[i])
                sgl_dataset_success[i] = status == "success"
                try:
                    self.all_results["single_dataset"][i] = (
                        self._load_single_results(sgl_folders[i])
                    )
                except Exception as e:
                    logger.error(
                        "Could not load single-dataset results from "
                        f"{sgl_folders[i]}: {e}"
                    )
                    sgl_dataset_success[i] = False
        self.sgl_workflow_locations = sgl_folders
        self.save(self.result_folder)

        failures = self._failed_single_datasets(
            sgl_dataset_success, sgl_folders, tags
        )

        # Write the HTML overview already now (even when not all singles
        # succeed) so a partial run still has a navigable local index.
        self._write_html_overview(sgl_folders, failures=failures)

        if failures:
            # Name the offenders: hunting for which of N datasets failed,
            # and why, used to mean opening every result folder by hand.
            detail = "\n".join(
                f"  [{i:02d}] {tag or '(no tag)'}: {description}\n"
                f"         {folder}"
                for i, tag, folder, description in failures
            )
            msg = (
                f"{len(failures)} of {n_sgl} single datasets failed, so no "
                f"aggregation analysis is started:\n{detail}"
            )
            logger.error(msg)
            self.progress.finish(FAILED)
            self._report_aggregation_abort(failures, n_sgl)
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
            logger.debug("not continuing workflow.starting new.")
            wr = WorkflowRunner.config_from_dicts(
                agg_reporter_config,
                agg_analysis_config,
                parameters,
                postfix=self.postfix,
            )
        self.cpage_names.append(wr.reporter_config["report_name"])
        self._agg_report_folder = wr.result_folder
        agg_success = wr.run()
        self.all_results["aggregation"] = wr.results
        self.save(self.result_folder)
        self.progress.finish(DONE if agg_success else FAILED)

        # Refresh the HTML overview now that the aggregation report exists.
        self._write_html_overview(sgl_folders, self._agg_report_folder)

    def _write_html_overview(
        self,
        sgl_folders: list,
        agg_folder: str | None = None,
        failures: list | None = None,
    ) -> None:
        """Write the top-level ``index.html`` linking the child reports.

        No-op unless HTML reporting is configured. Links to each
        single-dataset ``report.html`` and (when available) the aggregation
        ``report.html``, relative to the aggregation result folder.

        Parameters
        ----------
        sgl_folders : list of str
            The single-dataset result folders (each holds a ``report.html``).
        agg_folder : str, optional
            The aggregation result folder, if the aggregation step has run.
        failures : list of tuple, optional
            ``(index, tag, folder, description)`` per failed single
            dataset, listed in the overview table so a partial run shows
            what went wrong.
        """
        if not self._html_reporting:
            return

        child_reports = []
        for i, folder in enumerate(sgl_folders):
            if not folder:
                continue
            report = os.path.join(folder, "report.html")
            href = os.path.relpath(report, self.result_folder)
            label = os.path.basename(folder.rstrip(os.sep)) or f"dataset {i}"
            child_reports.append((label, href, os.path.isfile(report)))
        if agg_folder:
            report = os.path.join(agg_folder, "report.html")
            href = os.path.relpath(report, self.result_folder)
            child_reports.append(("Aggregation", href, os.path.isfile(report)))

        rows = [
            ("Report", self.reporter_config.get("report_name", "")),
            ("Result folder", self.result_folder),
            ("Single datasets", len(sgl_folders)),
            ("Reports found", sum(1 for _, _, ok in child_reports if ok)),
        ]
        if failures:
            rows.append(
                ("Failed datasets", f"{len(failures)} of {len(sgl_folders)}")
            )
            for idx, tag, _folder, description in failures:
                rows.append(
                    (f"Failed [{idx:02d}] {tag or '(no tag)'}", description)
                )
        try:
            write_aggregation_index(
                self.result_folder,
                self.reporter_config.get("report_name", "Aggregation report"),
                rows,
                child_reports,
                config=self.aggregation_workflow,
            )
        except Exception as e:  # reporting must never abort the analysis
            logger.error(f"Could not write HTML aggregation index: {e}")

    def _report_aggregation_abort(self, failures: list, n_total: int) -> None:
        """Post the skipped-aggregation summary to the run's parent page.

        Best effort: a reporting problem must not replace the analysis
        failure that is about to be raised.

        Parameters
        ----------
        failures : list of tuple
            ``(index, tag, folder, description)`` per failed dataset.
        n_total : int
            Total number of single datasets.
        """
        ci = getattr(self, "ci", None)
        if ci is None:
            return
        try:
            confluence_config = self.reporter_config.get(
                "ConfluenceReporter", {}
            )
            page_id = confluence_config.get("parent_page_id")
            page_name = self.reporter_config.get("report_name", "")
            if not page_id:
                logger.debug(
                    "No parent page id; skipping the Confluence summary of "
                    "the skipped aggregation."
                )
                return
            ci.update_page_content(
                page_name,
                page_id,
                aggregation_abort_body(failures, n_total),
            )
        except Exception as e:
            logger.error(
                f"Could not report the skipped aggregation to Confluence: {e}"
            )

    def _describe_single_failure(self, i: int) -> str:
        """Explain why single dataset ``i`` failed, from its results.

        Parameters
        ----------
        i : int
            Index of the single dataset.

        Returns
        -------
        str
            The failing module and exception, e.g.
            ``"fit_csr: ValueError: min_dist=50.0, max_dist=300.0 leaves
            0 of 99 ..."``. Falls back to the last module reached when no
            error was recorded (results written before failures were
            recorded), or a plain note when there are no results at all.
        """
        try:
            results = self.all_results["single_dataset"][i]
        except (KeyError, IndexError, TypeError):
            results = None
        if not results:
            return "no results recorded"

        for key, res in reversed(list(results.items())):
            if not isinstance(res, dict):
                continue
            error = res.get("error")
            if error:
                return (
                    f"{key}: {error.get('type', 'Error')}: "
                    f"{error.get('message', '')}"
                )
            if res.get("success") is False:
                return f"{key}: failed (no exception recorded)"

        last = list(results)[-1] if results else None
        return f"no error recorded; last module reached was {last}"

    def _failed_single_datasets(
        self, sgl_dataset_success: list, sgl_folders: list, tags: list
    ) -> list:
        """Collect a description of every failed single dataset.

        Parameters
        ----------
        sgl_dataset_success : list of bool
            Per-dataset success flags.
        sgl_folders : list of str
            Per-dataset result folders.
        tags : list of str
            Per-dataset tags.

        Returns
        -------
        list of tuple
            ``(index, tag, folder, description)`` for each failure.
        """
        failures = []
        for i, ok in enumerate(sgl_dataset_success):
            if ok:
                continue
            tag = tags[i] if i < len(tags) else ""
            folder = sgl_folders[i] if i < len(sgl_folders) else ""
            failures.append((i, tag, folder, self._describe_single_failure(i)))
        return failures

    @staticmethod
    def _single_marker_path(folder: str) -> str:
        """Return the completion-marker path for a single-dataset folder."""
        return os.path.join(folder, "_pwf_single_done.txt")

    def _write_single_marker(self, folder: str, success: bool) -> None:
        """Drop a completion marker for a finished single dataset.

        Lets rank 0 know the dataset is finished and whether it succeeded.
        Written atomically via a rank-specific temp file + ``os.replace``.

        Parameters
        ----------
        folder : str
            The single-dataset result folder to write the marker into.
        success : bool
            Whether the single-dataset workflow succeeded.
        """
        try:
            os.makedirs(folder, exist_ok=True)
            marker = self._single_marker_path(folder)
            tmp = f"{marker}.{self.rank}.tmp"
            with open(tmp, "w") as f:
                f.write("success" if success else "failed")
            os.replace(tmp, marker)
        except Exception as e:
            logger.error(
                f"Could not write single-dataset marker in {folder}: {e}"
            )

    def _read_single_marker(self, folder: str) -> str | None:
        """Return a single dataset's marker contents, or None if absent.

        Parameters
        ----------
        folder : str
            The single-dataset result folder to read the marker from.

        Returns
        -------
        str or None
            ``"success"`` / ``"failed"`` if the marker exists, else None.
        """
        try:
            with open(self._single_marker_path(folder)) as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    def _claim_dir(self) -> str:
        """Per-launch directory of dataset claims for dynamic scheduling.

        Scoped by SLURM job id so a claim means "a rank is running this
        dataset in *this* launch". A stale claim left by a crashed earlier
        attempt (same result folder, new job) then never blocks a rerun, while
        the persistent per-folder completion marker records "finished" across
        launches. Off-cluster (no SLURM_JOB_ID) it falls back to ``local``,
        but claiming is skipped entirely for a single task anyway.

        Returns
        -------
        str
            The claim directory for this launch.
        """
        job = os.getenv("SLURM_JOB_ID") or "local"
        return os.path.join(self.result_folder, "_pwf_claims", str(job))

    def _claim_dataset(self, claim_dir: str, i: int) -> bool:
        """Atomically claim single dataset ``i`` for this rank.

        Creating a directory is atomic on a shared (NFS) filesystem - the
        server serialises the MKDIR - so exactly one rank wins the race for
        each dataset. This turns each rank's ordered pass into greedy
        "next-available" self-scheduling: a rank advances to the next dataset
        only after finishing its current one, so a free rank always grabs the
        next unclaimed dataset and no rank idles while work remains.

        Parameters
        ----------
        claim_dir : str
            The per-launch claim directory (see :meth:`_claim_dir`).
        i : int
            Index of the single dataset to claim.

        Returns
        -------
        bool
            True if this rank claimed the dataset, False if another rank
            already owns it. On an unexpected filesystem error it returns True
            (run it here): a redundant run only wastes time and the last marker
            and results win, whereas skipping could drop the dataset and hang
            rank 0's barrier.
        """
        try:
            os.mkdir(os.path.join(claim_dir, f"{i:04d}"))
            return True
        except FileExistsError:
            return False
        except Exception as e:
            logger.warning(
                f"Claim for single dataset {i} could not be created ({e}); "
                f"running it on rank {self.rank} to be safe."
            )
            return True

    def _wait_for_single_markers(
        self,
        folders: list[str],
        timeout: float = 7 * 24 * 3600,
        poll: float = 15,
    ) -> None:
        """Block until every single-dataset folder has a completion marker.

        Used by rank 0 before aggregating, to gather the datasets handled by
        other ranks via the shared filesystem. SLURM enforces the real wall
        time; the timeout here is only a safety net against an unrecoverable
        hang (e.g. a worker that died without writing a marker).

        Parameters
        ----------
        folders : list of str
            The single-dataset result folders to wait on.
        timeout : float, optional
            Maximum seconds to wait before raising. Default is one week.
        poll : float, optional
            Seconds between polls of the shared filesystem. Default is 15.

        Raises
        ------
        WorkflowError
            If the timeout elapses before all markers appear.
        """
        start = time.time()
        pending = set(range(len(folders)))
        while pending:
            pending = {
                i
                for i in pending
                if self._read_single_marker(folders[i]) is None
            }
            if not pending:
                break
            if time.time() - start > timeout:
                raise WorkflowError(
                    "Timed out waiting for single-dataset workflows on "
                    f"other ranks: {[folders[i] for i in sorted(pending)]}"
                )
            logger.debug(
                f"Rank 0 waiting for {len(pending)} single dataset(s) to "
                "finish on other ranks."
            )
            time.sleep(poll)

    @staticmethod
    def _load_single_results(folder: str) -> dict:
        """Load only the results dict a single WorkflowRunner saved.

        Avoids re-initializing the Confluence reporter that
        :meth:`WorkflowRunner.load` would set up.

        Parameters
        ----------
        folder : str
            The single-dataset result folder, holding
            ``WorkflowRunner.yaml``.

        Returns
        -------
        dict
            The saved ``results`` dictionary.
        """
        fp = os.path.join(folder, "WorkflowRunner.yaml")
        with open(fp, "r") as f:
            data = yaml.safe_load(f)
        return data["results"]

    def save(self, dirn: str = ".") -> None:
        """Save the current config and results to the given directory.

        Writes ``AggregationWorkflowRunner.yaml``.

        Parameters
        ----------
        dirn : str, optional
            The directory to save into. Default is the current directory.
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
    def load(cls, dirn: str = ".") -> "AggregationWorkflowRunner":
        """Load an instance from an ``AggregationWorkflowRunner.yaml`` file.

        Parameters
        ----------
        dirn : str, optional
            The directory to load from. Default is the current directory.

        Returns
        -------
        AggregationWorkflowRunner
            The reconstructed runner, marked to continue a previous run.
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
    """Raised when a workflow cannot complete (e.g. a failed dataset)."""


class WorkflowRunner:
    """Run a workflow and publish its results to Confluence.

    A workflow is a sequence of modules that are run in order, each module's
    results being reported to Confluence.

    Examples
    --------
    >>> rc, ac, wm = {}, {}, {}
    >>> wr = WorkflowRunner.config_from_dicts(rc, ac, wm)
    >>> wr.run()
    """

    def __init__(self, postfix: str | None = None):
        """Initialize the runner.

        Parameters
        ----------
        postfix : str, optional
            Postfix used to load prior analyses, formatted ``%y%m%d-%H%M``.
            If None, a new postfix is generated from the current time.
        """
        if postfix:
            self.postfix = postfix
        else:
            self.postfix = datetime.now().strftime("%y%m%d-%H%M")

        self.parameter_command_executor = ParameterCommandExecutor(self)
        self.results = {}
        # Progress tracking. ``progress`` is built lazily in run() (needs the
        # result folder and module list); ``_abort_requested`` supports a
        # cooperative in-process stop, complementing the on-disk abort flag.
        self.progress = None
        self._abort_requested = False

    @classmethod
    def config_from_dicts(
        cls,
        reporter_config: dict,
        analysis_config: dict,
        workflow_modules: list[tuple],
        postfix: str | None = None,
        continue_previous_runner: bool = False,
    ) -> "WorkflowRunner":
        """Build a configured runner from plain config dicts.

        Initialization is kept out of ``__init__`` to preserve flexibility for
        alternative entry points in the future (config file names, a web API,
        etc.).

        Parameters
        ----------
        reporter_config : dict
            Configuration of the reporter (currently the Confluence reporter).
        analysis_config : dict
            General analysis configuration.
        workflow_modules : list of tuple
            The workflow modules to run, as ``(module_name, parameters)``.
        postfix : str, optional
            Postfix used to load prior analyses, formatted ``%y%m%d-%H%M``.
            If None, a new postfix is generated.
        continue_previous_runner : bool, optional
            Continue a previous analysis that aborted (e.g. at a manual step).
            If no previous analysis exists in that folder, a new one is
            created. Default is False.

        Returns
        -------
        WorkflowRunner
            The configured runner instance.
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
    def _check_previous_runner(
        cls, folder: str, report_name: str
    ) -> str | None:
        """Find the postfix of the latest previous runner in a location.

        Parameters
        ----------
        folder : str
            The folder to look in.
        report_name : str
            The name of the report.

        Returns
        -------
        str or None
            The postfix of the latest previous runner in that location, or
            None if none are found.
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

    def _initialize_analysis(
        self, analysis_config: dict, report_name: str
    ) -> None:
        """Initialize the analysis worker and its result directory.

        Parameters
        ----------
        analysis_config : dict
            General analysis configuration; ``result_location`` is popped to
            build the result folder.
        report_name : str
            Name of the report, used as the result subfolder name.
        """
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

    def _initialize_reporter(self, reporter_config: dict) -> None:
        """Initialize the reporter(s) that document the analysis.

        Supports two reporter backends, either or both of which may be
        configured under ``reporter_config``:

        - ``ConfluenceReporter`` -- live Confluence reporting (its sub-dict
          holds the connection kwargs).
        - ``HTMLReporter`` -- a local navigable ``report.html`` (no Confluence
          connection or credentials). Its sub-dict may set ``report_dir``;
          otherwise the report is written into the run's result folder.

        Configured reporters are collected in ``self.reporters`` and invoked
        in turn for every module. ``self.confluencereporter`` is retained as
        an alias for the Confluence reporter when present.

        Parameters
        ----------
        reporter_config : dict
            Reporter configuration.
        """
        logger.debug("Initializing Reporter.")
        self.report_name = reporter_config["report_name"]
        self.reporters = []
        self.confluencereporter = None
        if init_kwargs := reporter_config.get("ConfluenceReporter"):
            init_kwargs["report_name"] = self.report_name
            # logger.debug(init_kwargs)
            self.confluencereporter = ConfluenceReporter(**init_kwargs)
            self.reporters.append(self.confluencereporter)
        if (html_kwargs := reporter_config.get("HTMLReporter")) is not None:
            report_dir = html_kwargs.get("report_dir")
            if not report_dir:
                # _initialize_analysis sets result_folder but pops
                # result_location, and the two entry points call them in
                # different orders -- prefer whichever is available.
                if getattr(self, "result_folder", None):
                    report_dir = self.result_folder
                else:
                    report_dir = os.path.join(
                        self.analysis_config["result_location"],
                        self.report_name,
                    )
            self.htmlreporter = HTMLReporter(report_dir, self.report_name)
            self.reporters.append(self.htmlreporter)

    def run(self) -> bool:
        """Run the analysis of the workflow modules in order.

        Already-succeeded modules from a previous run are skipped; execution
        stops at the first module that fails.

        Returns
        -------
        bool
            Whether all modules ran through successfully.
        """
        # pre-flight: validate dependencies/scope (warn-only, non-blocking)
        _log_workflow_validation(
            self.workflow_modules, Scope.SINGLE, "single-dataset workflow"
        )

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

        # progress tracking: build the emitter and announce the module list
        progress = self._ensure_progress()
        progress.start([name for name, _ in self.workflow_modules])

        # now, run the modules
        all_previously_succeeded = True
        # Bind up front: a module raising on the very first iteration used
        # to leave this unbound and fail with UnboundLocalError below,
        # masking the real error.
        success = False
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
                logger.debug(f"""Module {i}, {module_name} has been previously
                    analyzed. Skipping.""")
                progress.module_skipped(i)
                continue
            else:
                all_previously_succeeded = False

            # cooperative abort: stop cleanly at the next module boundary if
            # an abort was requested (in-process or via the on-disk flag).
            if self._abort_callback():
                logger.warning(
                    f"Abort requested; stopping before module {i} "
                    f"({module_name})."
                )
                success = False
                progress.finish(ABORTED)
                return success
            # all modules are called with iteration and parameter dict
            # as arguments
            progress.module_start(i)
            try:
                # Resolve the $-commands (e.g. $get_prior_result) inside the
                # try: a failure here -- such as referencing a module that
                # does not exist in this workflow -- must be recorded as a
                # module failure and finalize the run's progress, rather than
                # escaping with the progress state stuck at RUNNING (which
                # leaves the live monitor showing the dataset as still running
                # long after the rank has moved on).
                module_parameters = self.parameter_command_executor.run(
                    module_parameters, curr_rootidx=i
                )
                success = self.call_module(module_name, i, module_parameters)
            except AutoPicassoError:
                success = False
                progress.module_end(i, FAILED)
            except Exception:
                # Any other exception used to escape before save(), so the
                # failing module never reached WorkflowRunner.yaml. Record
                # it, then let it propagate as before.
                success = False
                progress.module_end(i, FAILED)
                progress.finish(FAILED)
                self.save(self.result_folder)
                raise
            else:
                progress.module_end(i, DONE if success else FAILED)

            self.save(self.result_folder)
            if not success:
                break
        else:
            success = True

        if progress.state["state"] == RUNNING:
            progress.finish(DONE if success else FAILED)
        return success

    def _ensure_progress(self) -> ProgressManager:
        """Return the run's :class:`ProgressManager`, building it if needed.

        Built lazily because it needs both the result folder (set at
        initialization) and, conceptually, the module list (only meaningful
        at ``run`` time). May be pre-set by a coordinator to inject sinks.

        Returns
        -------
        ProgressManager
        """
        if self.progress is None:
            self.progress = ProgressManager(
                self.result_folder,
                kind="single",
                report_name=getattr(self, "report_name", None),
            )
        # Wire the analysis worker so long picasso calls can report
        # intra-module progress and honour aborts.
        if getattr(self, "autopicasso", None) is not None:
            self.autopicasso._abort_callback = self._abort_callback
        return self.progress

    def _abort_callback(self) -> bool:
        """Whether the run should abort (in-process flag or on-disk flag).

        Passed to picasso's long-running calls and checked between modules,
        so a GUI/operator can stop a run gracefully at the next checkpoint.

        Returns
        -------
        bool
        """
        if self._abort_requested:
            return True
        try:
            return pwprogress.abort_requested(self.result_folder)
        except Exception:
            return False

    ##########################################################################
    # UTIL FUNCTIONS
    ##########################################################################

    def get_postfixed_filename(self, filename: str) -> str:
        """Return ``filename`` prefixed with the runner's postfix.

        Parameters
        ----------
        filename : str
            The base filename.

        Returns
        -------
        str
            The postfixed path under ``self.savedir``.
        """
        return os.path.join(self.savedir, self.postfix + filename)

    def save(self, dirn: str = ".") -> None:
        """Save the current results to the given directory.

        Writes ``WorkflowRunner.yaml``.

        Parameters
        ----------
        dirn : str, optional
            The directory to save into. Default is the current directory.
        """
        pce = DictSimpleTyper(to_simple_type=True)
        filepath = os.path.join(dirn, "WorkflowRunner.yaml")
        data = {
            "results": pce.run(self.results),
            "reporter_config": pce.run(self.reporter_config),
            "analysis_config": pce.run(self.analysis_config),
            "workflow_modules": pce.run(self.workflow_modules),
        }
        # logger.debug("saving data:")
        # logger.debug(str(data))
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(cls, dirn: str = ".") -> "WorkflowRunner":
        """Load the results from a ``WorkflowRunner.yaml`` file.

        Parameters
        ----------
        dirn : str, optional
            The directory to load from. Default is the current directory.

        Returns
        -------
        WorkflowRunner
            The reconstructed runner with analysis and reporter initialized.
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

    def module_previously_analyzed(self, i: int) -> bool:
        """Check whether the module with index ``i`` was analysed previously.

        If it was, a folder prefixed with its index exists in the result
        folder.

        Parameters
        ----------
        i : int
            The module index.

        Returns
        -------
        bool
            Whether the folder corresponding to the module index was found.
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

    def module_previously_succeeded(self, i: int, module_name: str) -> bool:
        """Check whether a module previously succeeded, per the saved results.

        Parameters
        ----------
        i : int
            The module index.
        module_name : str
            The module name.

        Returns
        -------
        bool
            Whether a previous evaluation of the module succeeded.
        """
        module_id = f"{i:02d}_{module_name}"
        logger.debug("looking for previous " + module_id)
        # logger.debug(str(self.results.get(module_id, {})))
        logger.debug(
            str(self.results.get(module_id, {}).get("success", False))
        )
        return self.results.get(module_id, {}).get("success", False)

    def _report_module_error(self, e, fun_name, i, parameters, key):
        """Log, report and record a module failure.

        Posts a detailed error section to every reporter and records the
        failure in ``self.results`` so the next ``save()`` writes it to
        ``WorkflowRunner.yaml`` -- otherwise the failing module leaves no
        trace on disk at all.

        Parameters
        ----------
        e : Exception
            The exception raised by the module.
        fun_name : str
            Name of the failed module.
        i : int
            Index of the module in the workflow.
        parameters : dict
            The module's resolved parameters.
        key : str
            Results key of the module, ``f"{i:02d}_{fun_name}"``.
        """
        logger.error(e)
        logger.error(traceback.format_exc())

        partial = getattr(e, "_pwf_partial_results", None) or {}
        # The preceding module's results are the inputs this one worked
        # from; self.results is insertion-ordered.
        previous_results = next(reversed(list(self.results.values())), None)

        if e.__traceback__ is not None:
            tb_text = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
        else:
            tb_text = traceback.format_exc()

        for reporter in self.reporters:
            try:
                reporter.report_error(
                    e,
                    fun_name,
                    i=i,
                    parameters=parameters,
                    result_folder=partial.get("folder"),
                    previous_results=previous_results,
                )
            except Exception as report_exc:
                logger.error(f"Could not report the error: {report_exc}")
                logger.error(traceback.format_exc())

        self.results[key] = {
            **partial,
            "success": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": tb_text,
                "module": fun_name,
                "index": i,
            },
            # Filtered: an injected live object would be yaml.dump'd as an
            # !!python/object tag and break the reload path.
            "parameters": {
                k: v
                for k, v in parameters.items()
                if k not in _PARAM_BLACKLIST
            },
        }

    def call_module(self, fun_name: str, i: int, parameters: dict) -> bool:
        """Run one workflow module: analyse, then report.

        At the :class:`WorkflowRunner` level every module is processed the same
        way -- the analysis is performed by calling the module on
        ``autopicasso``, then its results are reported by calling the module on
        ``confluencereporter`` -- so this single method handles all modules
        instead of one method per module.

        Parameters
        ----------
        fun_name : str
            The function (module) name.
        i : int
            The index of the module in the workflow.
        parameters : dict
            The module parameters.

        Returns
        -------
        bool
            Whether the module ended successfully.

        Raises
        ------
        AutoPicassoError
            Re-raised if the analysis step failed (after reporting the error
            to Confluence).
        """
        key = f"{i:02d}_{fun_name}"
        logger.debug(f"Working on {key}")

        # Wire intra-module progress: long picasso calls forward their frame/
        # spot/segment counts through this callback, which the analysis worker
        # converts to a 0..1 fraction for module ``i``.
        if self.progress is not None:
            self.autopicasso._progress_callback = (
                lambda fraction, msg=None, _i=i: self.progress.module_progress(
                    _i, fraction, msg
                )
            )
            self.autopicasso._abort_callback = self._abort_callback

        # For conditional_branch module, inject the parameter_command_executor
        # so it can resolve sub-module parameters
        if fun_name == "conditional_branch":
            parameters["parameter_command_executor"] = (
                self.parameter_command_executor
            )

        fun_ap = getattr(self.autopicasso, fun_name)
        analyse_error = None
        try:
            parameters, self.results[key] = fun_ap(i, parameters)
        except AutoPicassoError as e:
            # Bind the exception itself, not a copy: copy.copy() goes
            # through __reduce__ and drops __traceback__, so the re-raise
            # below used to surface with a stack that stopped here.
            analyse_error = e
            self._report_module_error(e, fun_name, i, parameters, key)
        except Exception as e:
            analyse_error = e
            self._report_module_error(e, fun_name, i, parameters, key)

        # If the analysis step crashed, self.results[key] was never
        # written; skip the per-module success-path Confluence reporter
        # (which would crash with KeyError and mask analyse_error) and
        # re-raise the real cause. The error has already been posted to
        # Confluence via report_error(...) above.
        if analyse_error is not None:
            raise analyse_error

        # logger.debug(f"RESULTS: {self.results[key]}")
        for reporter in self.reporters:
            try:
                getattr(reporter, fun_name)(i, parameters, self.results[key])
            except ConfluenceInterfaceError as e:
                logger.error(e)
                logger.error(traceback.format_exc())

        return self.results[key]["success"]

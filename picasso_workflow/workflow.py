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
from dataclasses import dataclass
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
from picasso_workflow.modulespec import (
    CHANNEL_LOCS_CAPABILITIES,
    LOCS_STATE_CAPABILITIES,
    MEMORY_ONLY_CAPABILITIES,
    MODULE_REGISTRY,
    RUNTIME_PARAMETER_WRITE_BACKS,
    SINGLE_LOCS_CAPABILITIES,
    Scope,
    restart_conflicts,
    validate_workflow,
)
from picasso_workflow.util import (
    AbstractModuleCollection,
    ParameterCommandExecutor,
    ParameterTiler,
    DictSimpleTyper,
    convert_filepath_for_machine,
)
from picasso_workflow import progress as pwprogress
from picasso_workflow.progress import (
    ProgressManager,
    RUNNING,
    DONE,
    FAILED,
    ABORTED,
    PAUSED,
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


@dataclass(frozen=True)
class ResumePlan:
    """How a (possibly resumed) run starts.

    Parameters
    ----------
    start_index : int
        The first module to execute; modules before it are skipped.
    frontier : int
        The first module that must re-run: either it did not previously
        succeed, or its parameters changed since the previous run. Modules
        in ``[start_index, frontier)`` previously succeeded but re-run
        because their in-memory effects are not covered by the checkpoint.
    checkpoint : dict or None
        The checkpoint descriptor to restore before running (with added
        ``module_index``/``module_id`` bookkeeping), or None for a scratch
        run / a run needing no restore.
    description : str
        One human-readable line describing the decision, for the log.
    """

    start_index: int
    frontier: int
    checkpoint: dict | None
    description: str


# Modules whose legacy results keys reference restorable locs files, for
# resuming runs recorded before the explicit "checkpoint" descriptor
# existed. Name-gated because other modules record unrelated "filepath"
# keys (e.g. the manual module's user-provided file).
_LEGACY_SINGLE_CHECKPOINT_MODULES = {
    "save_single_dataset",
    "load_dataset_localizations",
}
_LEGACY_CHANNEL_CHECKPOINT_MODULES = {
    "load_datasets_to_aggregate",
    "save_datasets_aggregated",
}


def _existing_file(fp) -> str | None:
    """Resolve a persisted file path against the current machine.

    Paths in ``WorkflowRunner.yaml`` were written on the machine that ran
    the analysis; :func:`~picasso_workflow.util.convert_filepath_for_machine`
    translates known drive roots (``Drivepaths`` config) so a run can be
    resumed on another machine.

    Parameters
    ----------
    fp : object
        The persisted path (any type; non-strings are rejected).

    Returns
    -------
    str or None
        The (possibly translated) path if the file exists, else None.
    """
    if not isinstance(fp, str) or not fp:
        return None
    fp = convert_filepath_for_machine(fp)
    return fp if os.path.isfile(fp) else None


def _valid_single_part(part) -> dict | None:
    """Validate the ``single`` part of a checkpoint descriptor.

    Parameters
    ----------
    part : object
        The candidate part.

    Returns
    -------
    dict or None
        The part with a machine-resolved file path, or None if malformed
        or the file is missing.
    """
    if not isinstance(part, dict):
        return None
    fp = _existing_file(part.get("filepath"))
    return {"filepath": fp} if fp is not None else None


def _valid_channels_part(part) -> dict | None:
    """Validate the ``channels`` part of a checkpoint descriptor.

    Parameters
    ----------
    part : object
        The candidate part.

    Returns
    -------
    dict or None
        The part with machine-resolved file paths (and tags derived from
        the filenames when absent or malformed), or None if malformed or
        any referenced file is missing.
    """
    if not isinstance(part, dict):
        return None
    fps = part.get("filepaths")
    if not isinstance(fps, (list, tuple)) or not fps:
        return None
    resolved = [_existing_file(fp) for fp in fps]
    if any(fp is None for fp in resolved):
        return None
    tags = part.get("tags")
    if not isinstance(tags, (list, tuple)) or len(tags) != len(resolved):
        # files are named {tag}.hdf5 by _save_datasets_agg
        tags = [os.path.splitext(os.path.basename(fp))[0] for fp in resolved]
    return {"filepaths": resolved, "tags": list(tags)}


def _checkpoint_from_module_results(
    module_name: str, module_results: dict
) -> dict | None:
    """Extract a restorable checkpoint from a module's saved results.

    Parameters
    ----------
    module_name : str
        The module's name (without index prefix).
    module_results : dict
        The module's entry in the runner's results (from
        ``WorkflowRunner.yaml``).

    Returns
    -------
    dict or None
        A checkpoint descriptor (``single`` and/or ``channels`` parts, see
        :meth:`AutoPicasso.load_checkpoint`), or None if the module results
        reference no restorable locs on disk. A descriptor may cover only
        part of the recorded state (e.g. its channel files were deleted);
        the resume planner checks whether the unrestored state is actually
        needed before accepting it (see ``_checkpoint_lost_capabilities``).
    """
    # 1) explicit descriptor recorded by the module / module_decorator
    explicit = module_results.get("checkpoint")
    if isinstance(explicit, dict):
        valid = {}
        if (single := _valid_single_part(explicit.get("single"))) is not None:
            valid["single"] = single
        channels = _valid_channels_part(explicit.get("channels"))
        if channels is not None:
            valid["channels"] = channels
        if valid:
            if len(valid) < len(explicit.keys() & {"single", "channels"}):
                logger.warning(
                    "Checkpoint descriptor is only partially restorable "
                    "(some files are missing); the resume planner checks "
                    "whether the missing state is needed."
                )
            return valid
    # 2) legacy channel keys (runs recorded before the explicit descriptor)
    if module_name in _LEGACY_CHANNEL_CHECKPOINT_MODULES:
        part = _valid_channels_part(
            {
                "filepaths": module_results.get("filepaths"),
                "tags": module_results.get("tags"),
            }
        )
        if part is not None:
            return {"channels": part}
    # 3) legacy single-locs key
    if module_name in _LEGACY_SINGLE_CHECKPOINT_MODULES:
        fp = module_results.get("filepath")
        if isinstance(fp, str) and fp.endswith(".hdf5"):
            resolved = _existing_file(fp)
            if resolved is not None:
                return {"single": {"filepath": resolved}}
    # 4) legacy module_decorator auto-save (save_locs / always_save)
    folder = module_results.get("folder")
    if isinstance(folder, str):
        fp = _existing_file(os.path.join(folder, "locs.hdf5"))
        if fp is not None:
            return {"single": {"filepath": fp}}
    return None


def _checkpoint_lost_capabilities(checkpoint: dict | None) -> frozenset[str]:
    """The capabilities still lost after restoring a checkpoint.

    Parameters
    ----------
    checkpoint : dict or None
        A checkpoint descriptor, or None for no restore at all.

    Returns
    -------
    frozenset[str]
        Capability tokens the restart must not depend on: the memory-only
        set plus whatever locs state the checkpoint does not cover.
    """
    restored: frozenset[str] = frozenset()
    if checkpoint:
        if "single" in checkpoint:
            restored |= SINGLE_LOCS_CAPABILITIES
        if "channels" in checkpoint:
            restored |= CHANNEL_LOCS_CAPABILITIES
    return MEMORY_ONLY_CAPABILITIES | (LOCS_STATE_CAPABILITIES - restored)


def _module_parameters_changed(prev_params: dict, new_params: dict) -> bool:
    """Whether a module's parameters changed between two runs.

    Compares the previous run's *pristine* parameters (snapshotted before
    any ``$``-command resolution or module write-back, see
    ``WorkflowRunner.workflow_modules_pristine``) against the caller's.
    Both sides are simple-typed the same way ``WorkflowRunner.save``
    serializes to yaml, so numpy/tuple round-trips do not read as changes.
    A false positive only costs a safe extra re-run.

    Parameters
    ----------
    prev_params : dict
        The module's pristine parameters from the previous run.
    new_params : dict
        The module's parameters as configured now.

    Returns
    -------
    bool
    """
    typer = DictSimpleTyper(to_simple_type=True)
    prev = typer.run(copy.deepcopy(prev_params))
    new = typer.run(copy.deepcopy(new_params))
    return prev != new


_ORIGINALNOCMD_SUFFIX = "_originalnocmd"


def _substitute_companions(params: dict) -> None:
    """Replace resolved command values by their ``*_originalnocmd`` originals.

    ``ParameterCommandExecutor`` resolves command tuples in place but keeps
    the raw command (sign-stripped, truncated to two elements) under a
    companion key. Substituting the originals back, on both sides of a
    comparison, makes resolved and unresolved parameter sets comparable.

    Parameters
    ----------
    params : dict
        A (deep-copied) parameter dict; modified in place.
    """
    for key in [
        k
        for k in params
        if isinstance(k, str) and k.endswith(_ORIGINALNOCMD_SUFFIX)
    ]:
        params[key[: -len(_ORIGINALNOCMD_SUFFIX)]] = params.pop(key)


def _normalize_command(value):
    """Normalize a raw ``$``-command tuple to the companion-key form.

    The ``*_originalnocmd`` companions store commands sign-stripped and
    truncated to two elements; normalize raw commands the same way so the
    two forms compare equal.

    Parameters
    ----------
    value : object
        A parameter value.

    Returns
    -------
    object
        The normalized value (unchanged for non-command values).
    """
    if (
        isinstance(value, tuple)
        and value
        and isinstance(value[0], str)
        and value[0].startswith("$")
    ):
        return (value[0].lstrip("$"),) + tuple(value[1:2])
    return value


def _legacy_values_differ(
    prev_value, new_value, ignore: frozenset = frozenset(), path: str = ""
) -> bool:
    """Conservatively compare a legacy parameter value against a new one.

    Parameters
    ----------
    prev_value : object
        Value from a legacy yaml (resolved commands, module write-backs).
    new_value : object
        Value as configured now.
    ignore : frozenset[str], optional
        Dotted key paths to skip (keys the module overwrites at run time).
    path : str, optional
        The dotted path of the value being compared (for ``ignore``).

    Returns
    -------
    bool
        Whether the new value differs from the previous one. Keys present
        only on the previous side are ignored (module write-backs).
    """
    if isinstance(new_value, dict) and isinstance(prev_value, dict):
        _substitute_companions(prev_value)
        _substitute_companions(new_value)
        for key, value in new_value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key_path in ignore:
                continue
            if key not in prev_value or _legacy_values_differ(
                prev_value[key], value, ignore, key_path
            ):
                return True
        return False
    if isinstance(new_value, list) and isinstance(prev_value, list):
        if len(new_value) != len(prev_value):
            return True
        return any(
            _legacy_values_differ(p, n, ignore, path)
            for p, n in zip(prev_value, new_value)
        )
    return _normalize_command(new_value) != _normalize_command(prev_value)


def _module_parameters_changed_legacy(
    prev_params: dict, new_params: dict, module_name: str = ""
) -> bool:
    """Best-effort parameter comparison against a pre-snapshot yaml.

    Runs recorded before the pristine snapshot existed only persisted the
    *mutated* parameters (``$``-commands resolved in place, values written
    back by the modules themselves). Comparison is therefore conservative:
    commands are normalized to the ``*_originalnocmd`` companion form,
    only keys present in the new parameters are compared, and keys the
    module is known to overwrite at run time
    (:data:`~picasso_workflow.modulespec.RUNTIME_PARAMETER_WRITE_BACKS`,
    e.g. the auto-estimated ``min_gradient``) are skipped. Remaining
    limitation: *removing* a parameter is not detected.

    Parameters
    ----------
    prev_params : dict
        The module's parameters from the legacy yaml.
    new_params : dict
        The module's parameters as configured now.
    module_name : str, optional
        The module's name, to look up its run-time write-back keys.

    Returns
    -------
    bool
    """
    typer = DictSimpleTyper(to_simple_type=True)
    prev = typer.run(copy.deepcopy(prev_params))
    new = typer.run(copy.deepcopy(new_params))
    ignore = RUNTIME_PARAMETER_WRITE_BACKS.get(module_name, frozenset())
    return _legacy_values_differ(prev, new, ignore)


_RUNSTAMP_RE = re.compile(r"_(\d{6}-\d{4})$")


def _strip_runstamp(report_name: str) -> str:
    """Strip a trailing ``_%y%m%d-%H%M`` runstamp from a report name.

    The coordinators stamp report names per launch; for finding a previous
    run, the stable, stamp-free base name is what matters.

    Parameters
    ----------
    report_name : str
        The (possibly stamped) report name.

    Returns
    -------
    str
        The base name without a trailing runstamp.
    """
    return _RUNSTAMP_RE.sub("", report_name)


def _find_previous_runner_postfix(folder: str, report_name: str) -> str | None:
    """Find the postfix of the latest previous runner folder.

    Runner folders are named ``{report_name}_{postfix}``, where the postfix
    ends in a ``%y%m%d-%H%M`` timestamp and may carry a per-run token in
    front of it (e.g. ``p1a2b3_240310-1130``, woven in by the
    coordinators). The folder with the latest trailing timestamp wins.

    Parameters
    ----------
    folder : str
        The folder to look in.
    report_name : str
        The stable (stamp-free) base report name.

    Returns
    -------
    str or None
        The postfix of the latest previous runner, or None if none found.
    """
    try:
        dirs = [
            it
            for it in os.listdir(folder)
            if it.startswith(report_name + "_")
            and os.path.isdir(os.path.join(folder, it))
        ]
    except FileNotFoundError:
        return None
    latest_datetime = None
    latest_postfix = None
    for d in dirs:
        postfix = d[len(report_name) + 1 :]
        try:
            dt = datetime.strptime(postfix.rsplit("_", 1)[-1], "%y%m%d-%H%M")
        except ValueError:
            continue
        if latest_datetime is None or latest_datetime < dt:
            latest_datetime = dt
            latest_postfix = postfix
    return latest_postfix


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
        # Stepwise development runs: ``("single", i)`` stops every
        # single-dataset workflow after module i and skips the aggregation;
        # ``("aggregation", i)`` runs the single-dataset phase fully and
        # stops the aggregation workflow after module i. A per-launch
        # directive, not persisted to AggregationWorkflowRunner.yaml.
        self.stop_after = None
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
        stop_after: tuple | None = None,
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
        stop_after : tuple, optional
            Stepwise development boundary, as ``(phase, index)`` with phase
            ``"single"`` or ``"aggregation"`` (see the ``stop_after``
            attribute). Default is None (run everything).

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
            # Candidate postfixes: an explicitly passed/extracted one first
            # (it may name the exact run to adopt), then the latest previous
            # run found on disk. The extracted postfix may merely be THIS
            # launch's timestamp (the coordinators stamp the report name per
            # launch), in which case its folder does not exist and the
            # discovery takes over.
            candidates = []
            if extracted_postfix is not None:
                candidates.append(extracted_postfix)
            discovered = cls._check_previous_runner(folder, report_name)
            if discovered is not None and discovered not in candidates:
                candidates.append(discovered)
            logger.debug(f"Previous-runner postfix candidates: {candidates}")
            for candidate in candidates:
                runner_folder = os.path.join(
                    folder, report_name + "_" + candidate
                )
                try:
                    # adopt the caller's reporter choice: the persisted one
                    # may reference reporters disabled since (see load())
                    instance = cls.load(
                        runner_folder, reporter_config=reporter_config
                    )
                except FileNotFoundError:
                    logger.debug(f"Could not load runner from {runner_folder}")
                    continue
                # take over the caller's (possibly fixed) parameters;
                # change detection happens per WorkflowRunner
                instance._adopt_aggregation_workflow(aggregation_workflow)
                instance.stop_after = stop_after
                return instance

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
        instance.stop_after = stop_after
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
        return _find_previous_runner_postfix(folder, report_name)

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
        # Link the aggregation overview page for the GUI monitor (best-effort).
        cfg = (self.reporter_config or {}).get("ConfluenceReporter") or {}
        if getattr(self, "ci", None) is not None and cfg.get("parent_page_id"):
            try:
                self.progress.set_report_url(
                    self.ci.page_url(cfg["parent_page_id"])
                )
            except Exception as e:
                logger.debug(f"Could not record aggregation report URL: {e}")

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
        # stepwise development boundary, split into phase and module index
        stop_phase, stop_index = self.stop_after or (None, None)

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
            # Compute the folder for every dataset (even ones another rank
            # claims) so rank 0 can find their markers/results afterwards.
            sgl_folders[i] = os.path.join(
                self.result_folder,
                self._single_dataset_name(report_name, i, tag)
                + "_"
                + self.postfix,
            )
            if self.size > 1 and not self._claim_dataset(claim_dir, i):
                continue  # claimed by another rank
            self._run_single_dataset(
                i,
                parameter_set,
                tag,
                report_name,
                sgl_folders,
                sgl_dataset_success,
                sgl_wkfl_reporter_config,
                sgl_wkfl_analysis_config,
            )

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
            if rank_ok and stop_phase == "single":
                self.progress.finish(PAUSED)
            else:
                self.progress.finish(DONE if rank_ok else FAILED)
            return rank_ok

        # Rank 0 (or a single-task run): wait for the single datasets handled
        # by other ranks and load their results from disk. Re-run (here) any
        # dataset whose owning rank died mid-run, so one lost worker cannot
        # hang the barrier until timeout.
        if self.size > 1:
            self._wait_for_single_markers(
                sgl_folders,
                reclaim=lambda i: self._run_single_dataset(
                    i,
                    individual_parametersets[i],
                    tags[i],
                    report_name,
                    sgl_folders,
                    sgl_dataset_success,
                    sgl_wkfl_reporter_config,
                    sgl_wkfl_analysis_config,
                ),
            )
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

        # Stepwise boundary in the single-dataset phase: every dataset
        # stopped cleanly at its boundary, so skip the aggregation workflow.
        if stop_phase == "single":
            logger.info(
                "Stepwise boundary in the single-dataset phase reached; "
                "skipping the aggregation workflow."
            )
            self.progress.finish(PAUSED)
            return None

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
                    ),
                    reporter_config=copy.deepcopy(agg_reporter_config),
                )
                wr.adopt_workflow_modules(parameters)
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
        if stop_phase == "aggregation":
            wr.stop_after = stop_index
        self.cpage_names.append(wr.reporter_config["report_name"])
        self._agg_report_folder = wr.result_folder
        agg_success = wr.run()
        self.all_results["aggregation"] = wr.results
        self.save(self.result_folder)
        if agg_success and wr.paused:
            self.progress.finish(PAUSED)
        else:
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
    def _single_dataset_name(report_name: str, i: int, tag: str) -> str:
        """Return single dataset ``i``'s report name (``..._sgl_NN[_tag]``).

        Parameters
        ----------
        report_name : str
            The aggregation run's report name.
        i : int
            Index of the single dataset.
        tag : str
            The dataset's tag (per-channel/condition label), possibly empty.

        Returns
        -------
        str
            The single-dataset report name.
        """
        sgl_name = report_name + f"_sgl_{i:02d}"
        if tag:
            sgl_name += f"_{tag}"
        return sgl_name

    def _run_single_dataset(
        self,
        i: int,
        parameter_set: dict,
        tag: str,
        report_name: str,
        sgl_folders: list,
        sgl_dataset_success: list,
        sgl_wkfl_reporter_config: dict,
        sgl_wkfl_analysis_config: dict,
    ) -> bool:
        """Run one single-dataset workflow and record its result + marker.

        Shared by the main self-scheduling loop and rank 0's recovery of a
        dataset orphaned by a crashed worker (see
        :meth:`_wait_for_single_markers`). Never lets an error escape before
        the completion marker is written, so rank 0's barrier cannot hang on
        it.

        Parameters
        ----------
        i : int
            Index of the single dataset.
        parameter_set : dict
            The tiled parameter set for this dataset.
        tag : str
            The dataset's tag (per-channel/condition label), possibly empty.
        report_name : str
            The aggregation run's report name (single names derive from it).
        sgl_folders : list of str
            Per-dataset result folders; ``sgl_folders[i]`` is read here.
        sgl_dataset_success : list
            Per-dataset success flags; ``sgl_dataset_success[i]`` is set here.
        sgl_wkfl_reporter_config : dict
            The single-workflow reporter config template (mutated per dataset).
        sgl_wkfl_analysis_config : dict
            The single-workflow analysis config template (mutated per dataset).

        Returns
        -------
        bool
            Whether the single-dataset workflow succeeded.
        """
        sgl_name = self._single_dataset_name(report_name, i, tag)
        sgl_wkfl_reporter_config["report_name"] = sgl_name
        sgl_wkfl_analysis_config["result_location"] = self.result_folder
        if self.continue_workflow:
            try:
                logger.debug(f"loading WorkflowRunner from {sgl_folders[i]}")
                wr = WorkflowRunner.load(
                    sgl_folders[i],
                    reporter_config=copy.deepcopy(sgl_wkfl_reporter_config),
                )
                wr.adopt_workflow_modules(parameter_set)
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
        # stepwise development: a "single"-phase boundary applies to every
        # per-dataset workflow (set here to also cover the loaded-runner path)
        if self.stop_after is not None and self.stop_after[0] == "single":
            wr.stop_after = self.stop_after[1]
        self.cpage_names.append(wr.reporter_config["report_name"])
        self.progress.dataset_update(i, RUNNING)
        # Never let an unhandled error escape before the completion marker is
        # written - otherwise rank 0 would wait on the barrier until timeout. A
        # failed single still marks the run as failed, which aborts the
        # aggregation below.
        try:
            success = wr.run()
        except Exception as e:
            logger.error(f"Single dataset {i} ({tag}) failed: {e}")
            logger.error(traceback.format_exc())
            success = False
        if success and wr.paused:
            self.progress.dataset_update(i, PAUSED)
        else:
            self.progress.dataset_update(i, DONE if success else FAILED)
        sgl_dataset_success[i] = success
        self.all_results["single_dataset"][i] = getattr(wr, "results", None)
        if self.rank == 0:
            self.save(self.result_folder)
        if self.size > 1:
            self._write_single_marker(sgl_folders[i], success)
        return success

    @staticmethod
    def _single_marker_path(folder: str) -> str:
        """Return the completion-marker path for a single-dataset folder."""
        return os.path.join(folder, "_pwf_single_done.txt")

    def _launch_token(self) -> str:
        """Token identifying this launch (SLURM job id, else ``local``).

        Shared by :meth:`_claim_dir` and the completion markers so both are
        scoped to a single launch.
        """
        return os.getenv("SLURM_JOB_ID") or "local"

    def _write_single_marker(self, folder: str, success: bool) -> None:
        """Drop a completion marker for a finished single dataset.

        Lets rank 0 know the dataset is finished and whether it succeeded.
        Written atomically via a rank-specific temp file + ``os.replace``.
        The marker is stamped with this launch's token (see
        :meth:`_launch_token`) so a stale marker from a previous run in the
        same (reused) result folder -- e.g. a stepwise step that paused its
        datasets -- is not mistaken for this launch's completion.

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
                status = "success" if success else "failed"
                f.write(f"{status} {self._launch_token()}")
            os.replace(tmp, marker)
        except Exception as e:
            logger.error(
                f"Could not write single-dataset marker in {folder}: {e}"
            )

    def _read_single_marker(self, folder: str) -> str | None:
        """Return a single dataset's marker status for *this* launch, or None.

        A marker stamped with a different launch token (a stale marker from a
        previous run in the reused result folder) is treated as absent, so
        rank 0's barrier waits for this launch's workers instead of adopting
        the previous run's result.

        Parameters
        ----------
        folder : str
            The single-dataset result folder to read the marker from.

        Returns
        -------
        str or None
            ``"success"`` / ``"failed"`` if a marker for this launch exists,
            else None (absent, stale, or legacy/unstamped).
        """
        try:
            with open(self._single_marker_path(folder)) as f:
                content = f.read().strip()
        except FileNotFoundError:
            return None
        parts = content.split()
        if len(parts) == 2 and parts[1] == self._launch_token():
            return parts[0]
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
        return os.path.join(
            self.result_folder, "_pwf_claims", self._launch_token()
        )

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

    @staticmethod
    def _single_progress_mtime(folder: str) -> float | None:
        """Return the mtime of a single dataset's ``progress.json``, or None.

        A dataset actively running keeps rewriting its ``progress.json``, so a
        stalled (or missing) mtime is a liveness signal that the owning rank
        died. See :meth:`_wait_for_single_markers`.

        Parameters
        ----------
        folder : str
            The single-dataset result folder.

        Returns
        -------
        float or None
            The modification time, or None if the file is absent/unreadable.
        """
        try:
            return os.path.getmtime(os.path.join(folder, "progress.json"))
        except OSError:
            return None

    def _wait_for_single_markers(
        self,
        folders: list[str],
        reclaim=None,
        timeout: float = 7 * 24 * 3600,
        poll: float = 15,
        stale_grace: float = 1800,
    ) -> None:
        """Block until every single-dataset folder has a completion marker.

        Used by rank 0 before aggregating, to gather the datasets handled by
        other ranks via the shared filesystem.

        A worker that dies after claiming a dataset leaves its claim dir behind
        (so no other rank retries it) and never writes a marker, which would
        otherwise hang this barrier until ``timeout``. To recover, a dataset
        whose marker is absent *and* whose ``progress.json`` has not advanced
        for ``stale_grace`` seconds is treated as orphaned and re-run here via
        ``reclaim(i)``. Re-running a merely-slow (still-live) dataset is safe -
        the last marker/results win - so a false positive only wastes compute.
        SLURM still enforces the real wall time; ``timeout`` is a final safety
        net (e.g. if ``reclaim`` is not provided).

        Parameters
        ----------
        folders : list of str
            The single-dataset result folders to wait on.
        reclaim : callable, optional
            ``reclaim(i)`` re-runs orphaned dataset ``i`` on this rank and
            writes its marker. If None, orphans are only waited on (legacy
            pure-wait behaviour) until ``timeout``.
        timeout : float, optional
            Maximum seconds to wait before raising. Default is one week.
        poll : float, optional
            Seconds between polls of the shared filesystem. Default is 15.
        stale_grace : float, optional
            Seconds a pending dataset's ``progress.json`` may stay unchanged
            before it is considered orphaned and reclaimed. Default is 1800.

        Raises
        ------
        WorkflowError
            If the timeout elapses before all markers appear.
        """
        start = time.time()
        last_mtime: dict[int, float | None] = {}
        stall_since: dict[int, float] = {}
        while True:
            pending = [
                i
                for i in range(len(folders))
                if self._read_single_marker(folders[i]) is None
            ]
            if not pending:
                return
            now = time.time()
            # Refresh each pending dataset's liveness: reset its stall timer
            # whenever its progress.json advances.
            for i in pending:
                mtime = self._single_progress_mtime(folders[i])
                if i not in last_mtime or mtime != last_mtime[i]:
                    last_mtime[i] = mtime
                    stall_since[i] = now
            if reclaim is not None:
                orphaned = [
                    i
                    for i in pending
                    if now - stall_since.get(i, now) > stale_grace
                ]
                for i in orphaned:
                    logger.warning(
                        f"Single dataset {i} ({folders[i]}) has no completion "
                        f"marker and no progress for {stale_grace:.0f}s; "
                        "assuming its rank died and re-running it on rank 0."
                    )
                    reclaim(i)  # runs the dataset and writes its marker
                    stall_since.pop(i, None)
                    last_mtime.pop(i, None)
                if orphaned:
                    continue  # re-poll immediately after reclaiming
            if now - start > timeout:
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
    def load(
        cls, dirn: str = ".", reporter_config: dict | None = None
    ) -> "AggregationWorkflowRunner":
        """Load an instance from an ``AggregationWorkflowRunner.yaml`` file.

        Parameters
        ----------
        dirn : str, optional
            The directory to load from. Default is the current directory.
        reporter_config : dict, optional
            The caller's reporter configuration for a continued run: the
            reporter backends follow it while the persisted run's
            ``report_name`` is kept (see :meth:`WorkflowRunner.load`).

        Returns
        -------
        AggregationWorkflowRunner
            The reconstructed runner, marked to continue a previous run.
        """
        fp = os.path.join(dirn, "AggregationWorkflowRunner.yaml")
        with open(fp, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        persisted_reporter_config = data["reporter_config"]
        if reporter_config is not None:
            adopted = copy.deepcopy(reporter_config)
            adopted["report_name"] = persisted_reporter_config["report_name"]
        else:
            adopted = persisted_reporter_config
        instance = cls.config_from_dicts(
            adopted,
            data["analysis_config"],
            data["aggregation_workflow"],
            data["postfix"],
        )
        instance.all_results = data["all_results"]
        instance.sgl_workflow_locations = data["sgl_workflow_locations"]
        instance.continue_workflow = True
        return instance

    def _adopt_aggregation_workflow(
        self, new_aggregation_workflow: dict
    ) -> None:
        """Adopt an edited aggregation workflow on resume.

        Counterpart to :meth:`WorkflowRunner.adopt_workflow_modules` at the
        aggregation level: takes over the caller's (possibly fixed)
        parameters so they reach the tiled per-dataset and aggregation-stage
        runners, where the per-module change detection happens. Only adopted
        if the module-name sequences of both stages and the number of
        dataset tiles are unchanged; otherwise the previous run's workflow
        is kept (warned). Best-effort: an error here must not break the
        resume, only disable adoption.

        Parameters
        ----------
        new_aggregation_workflow : dict
            The aggregation workflow as configured now (with
            ``single_dataset_tileparameters``, ``single_dataset_modules``
            and ``aggregation_modules``).
        """
        try:
            self._do_adopt_aggregation_workflow(new_aggregation_workflow)
        except Exception as e:
            logger.warning(
                f"Could not adopt the edited aggregation workflow on "
                f"resume ({e!r}); keeping the previous run's workflow."
            )

    def _do_adopt_aggregation_workflow(
        self, new_aggregation_workflow: dict
    ) -> None:
        """Implementation of :meth:`_adopt_aggregation_workflow`."""
        if not new_aggregation_workflow:
            return

        def _names(workflow, key):
            return [name for name, _ in (workflow.get(key) or [])]

        for key in ("single_dataset_modules", "aggregation_modules"):
            if _names(self.aggregation_workflow, key) != _names(
                new_aggregation_workflow, key
            ):
                logger.warning(
                    "Not adopting the edited aggregation workflow on "
                    f"resume: the {key} module sequence changed; keeping "
                    "the previous run's workflow."
                )
                return
        tilepars = new_aggregation_workflow.get(
            "single_dataset_tileparameters"
        )
        if not tilepars:
            logger.warning(
                "Not adopting the edited aggregation workflow on resume: "
                "single_dataset_tileparameters missing."
            )
            return
        new_tiler = ParameterTiler(self, tilepars)
        if new_tiler.ntiles != self.parameter_tiler.ntiles:
            logger.warning(
                "Not adopting the edited aggregation workflow on resume: "
                f"the number of datasets changed ({self.parameter_tiler.ntiles}"
                f" -> {new_tiler.ntiles}); keeping the previous run's "
                "workflow."
            )
            return
        self.aggregation_workflow = new_aggregation_workflow
        self.parameter_tiler = new_tiler


def _progress_error_text(e: BaseException) -> str:
    """Compact "type: message" plus traceback for a progress entry.

    Trimmed to fit :meth:`ProgressManager.module_end`'s 2000-character cap,
    keeping the traceback *tail* (where the raised error is) when the full
    text is too long.

    Parameters
    ----------
    e : BaseException
        The exception that failed the module.

    Returns
    -------
    str
    """
    header = f"{type(e).__name__}: {e}"
    if e.__traceback__ is not None:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    else:
        tb = traceback.format_exc()
    budget = 2000 - len(header) - 10
    if budget > 0 and len(tb) > budget:
        tb = "...\n" + tb[-budget:]
    return f"{header}\n{tb}"


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
        # Pristine snapshot of the workflow modules as configured, taken
        # before any $-command resolution or module write-back mutates the
        # parameter dicts (both happen in place). Persisted to
        # WorkflowRunner.yaml so a resume can compare the caller's
        # parameters against what the previous run was configured with.
        self.workflow_modules_pristine = None
        # The previous run's modules (set by adopt_workflow_modules on
        # resume); lets _plan_resume detect changed parameters. Consumed
        # by run(). The legacy flag marks pre-snapshot yamls, whose
        # mutated parameters need the conservative legacy comparison.
        self._previous_workflow_modules = None
        self._previous_modules_are_legacy = False
        # Progress tracking. ``progress`` is built lazily in run() (needs the
        # result folder and module list); ``_abort_requested`` supports a
        # cooperative in-process stop, complementing the on-disk abort flag.
        self.progress = None
        self._abort_requested = False
        # Stepwise development runs: stop cleanly after this module index
        # (None = run to the end). A per-launch directive, not persisted to
        # WorkflowRunner.yaml; the next step is a resume with a later (or no)
        # boundary. ``paused`` records that the last run() stopped at the
        # boundary rather than completing.
        self.stop_after = None
        self.paused = False

    @classmethod
    def config_from_dicts(
        cls,
        reporter_config: dict,
        analysis_config: dict,
        workflow_modules: list[tuple],
        postfix: str | None = None,
        continue_previous_runner: bool = False,
        stop_after: int | None = None,
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
        stop_after : int, optional
            Stop cleanly after the module with this index (stepwise
            development runs). The boundary module saves its localizations
            as a checkpoint, so the next step (a resume with a later
            boundary) continues from there. Default is None (run to the
            end).

        Returns
        -------
        WorkflowRunner
            The configured runner instance.
        """
        if continue_previous_runner:
            folder = analysis_config["result_location"]
            # the coordinators stamp the report name per launch; a previous
            # run is found by the stable, stamp-free base name
            base_name = _strip_runstamp(reporter_config["report_name"])
            found_postfix = cls._check_previous_runner(folder, base_name)
            if found_postfix is not None:
                runner_folder = os.path.join(
                    folder, base_name + "_" + found_postfix
                )
                try:
                    # adopt the caller's reporter choice: the persisted one
                    # may reference reporters disabled since (see load())
                    instance = cls.load(
                        runner_folder, reporter_config=reporter_config
                    )
                except FileNotFoundError:
                    # e.g. the previous run died before its first save():
                    # the folder exists but holds no WorkflowRunner.yaml.
                    # Fall through to creating a fresh runner.
                    logger.debug(f"Could not load runner from {runner_folder}")
                else:
                    # take over the caller's (possibly fixed) parameters;
                    # the previous run's pristine parameters are kept for
                    # change detection on resume
                    instance.adopt_workflow_modules(workflow_modules)
                    instance.stop_after = stop_after
                    return instance

        instance = cls(postfix)
        instance.stop_after = stop_after
        # set date and time to report name
        report_name = reporter_config["report_name"] + "_" + instance.postfix
        reporter_config["report_name"] = report_name

        instance.reporter_config = reporter_config
        instance.analysis_config = analysis_config
        instance._initialize_reporter(reporter_config)
        instance._initialize_analysis(analysis_config, report_name)
        instance.workflow_modules = workflow_modules
        instance.workflow_modules_pristine = copy.deepcopy(workflow_modules)
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
        return _find_previous_runner_postfix(folder, report_name)

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
        stops at the first module that fails. With ``stop_after`` set
        (stepwise development), execution also stops -- cleanly, with
        ``paused`` set -- after that module.

        Returns
        -------
        bool
            Whether all modules run so far succeeded (all of them, or, on a
            stepwise run, all up to the ``stop_after`` boundary).
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
        # picasso-set modules are not part of the AbstractModuleCollection
        # contract; they are registered via modulespec instead.
        available_modules += [
            name
            for name, spec in MODULE_REGISTRY.items()
            if spec.module_set == "picasso"
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

        # now, run the modules. On a resumed run, skip up to the last
        # checkpointed module, restore its saved locs, and re-run from there
        # (see _plan_resume).
        plan = self._plan_resume()
        logger.info(f"Resume plan: {plan.description}")
        # consume the previous-run diff: a second in-process run() must not
        # re-fire a stale parameter-change frontier against post-run state
        self._previous_workflow_modules = None
        if plan.checkpoint is not None:
            try:
                self.autopicasso.load_checkpoint(plan.checkpoint)
            except Exception as e:
                logger.warning(
                    "Could not restore checkpoint "
                    f"{plan.checkpoint.get('module_id')}: {e}; "
                    "re-running from scratch instead."
                )
                plan = ResumePlan(
                    0, plan.frontier, None, "checkpoint restore failed"
                )
        # Bind up front: a module raising on the very first iteration used
        # to leave this unbound and fail with UnboundLocalError below,
        # masking the real error.
        success = False
        self.paused = False
        for i, (module_name, module_parameters) in enumerate(
            self.workflow_modules
        ):
            # Stepwise development: stop cleanly at the stop-after boundary.
            # Checked first so modules beyond the boundary are neither run
            # nor marked skipped, whatever the resume plan says. Reaching
            # this point means nothing before the boundary failed (a failure
            # breaks the loop below), so the partial run counts as a success.
            if self.stop_after is not None and i > self.stop_after:
                logger.info(
                    f"Stepwise boundary: stopping before module {i:02d} "
                    f"({module_name}); modules up to {self.stop_after:02d} "
                    "are complete."
                )
                success = True
                self.paused = True
                break
            if i < plan.start_index:
                logger.debug(f"""Module {i}, {module_name} has been previously
                    analyzed. Skipping.""")
                progress.module_skipped(i)
                continue
            if i < plan.frontier:
                logger.info(
                    f"Re-running previously succeeded module {i:02d}_"
                    f"{module_name}: its in-memory effects were lost on "
                    "resume and are not covered by the checkpoint."
                )

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
                # The branch module owns resolution of its sub-workflow payload
                # (branch_modules/join_modules reference branch-local and $$map
                # results that do not exist yet at this point), so pass its
                # config through raw; branch() resolves each piece itself.
                if module_name != "branch":
                    module_parameters = self.parameter_command_executor.run(
                        module_parameters, curr_rootidx=i
                    )
                success = self.call_module(module_name, i, module_parameters)
            except AutoPicassoError as e:
                success = False
                # record the error with the progress entry, so the monitor
                # (local or over SSH) can show the traceback without access
                # to WorkflowRunner.yaml or the logs
                progress.module_end(i, FAILED, error=_progress_error_text(e))
            except Exception as e:
                # Any other exception used to escape before save(), so the
                # failing module never reached WorkflowRunner.yaml. Record
                # it, then let it propagate as before.
                success = False
                progress.module_end(i, FAILED, error=_progress_error_text(e))
                progress.finish(FAILED)
                self.save(self.result_folder)
                raise
            else:
                # a module may fail without raising (success=False in its
                # results); surface its recorded error/message with the
                # progress entry too, so the monitor can display it
                err_text = None
                if not success:
                    err_text = self._module_failure_text(
                        f"{i:02d}_{module_name}"
                    )
                progress.module_end(
                    i, DONE if success else FAILED, error=err_text
                )

            # Stepwise boundary: ensure the boundary module leaves a resume
            # checkpoint, so the next step continues from here instead of an
            # earlier incidental checkpoint. Done at the runner level (not via
            # a module-specific ``save_locs`` parameter, whose meaning differs
            # per module -- e.g. ``localize`` reads it as a dict). Only when
            # the module ran this session; a boundary skipped on resume keeps
            # whatever checkpoint the previous run recorded.
            if success and self.stop_after == i:
                self.autopicasso.save_locs_checkpoint(
                    self.results[f"{i:02d}_{module_name}"]
                )

            self.save(self.result_folder)
            if not success:
                break
        else:
            success = True

        if progress.state["state"] == RUNNING:
            if self.paused:
                progress.finish(PAUSED)
            else:
                progress.finish(DONE if success else FAILED)
        return success

    def _module_failure_text(self, key: str) -> str | None:
        """Best-effort failure description from a module's recorded results.

        Used for modules that fail without raising (``success: False`` in
        their results): their ``error`` may be the structured dict written
        by :meth:`_report_module_error`, a plain string set by the module,
        or absent (then ``message`` is tried).

        Parameters
        ----------
        key : str
            Results key of the module, ``f"{i:02d}_{fun_name}"``.

        Returns
        -------
        str or None
        """
        res = self.results.get(key) or {}
        err = res.get("error")
        if isinstance(err, dict):
            return err.get("traceback") or (
                f"{err.get('type', 'Error')}: {err.get('message', '')}"
            )
        if err:
            return str(err)
        msg = res.get("message")
        return str(msg) if msg else None

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
        # Record the Confluence report-page URL so the GUI monitor can link
        # to it (best-effort; never let a reporter quirk abort the run).
        reporter = getattr(self, "confluencereporter", None)
        if reporter is not None:
            try:
                self.progress.set_report_url(reporter.report_page_url)
            except Exception as e:
                logger.debug(f"Could not record Confluence report URL: {e}")
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
            "workflow_modules_pristine": pce.run(
                self.workflow_modules_pristine
            ),
        }
        # logger.debug("saving data:")
        # logger.debug(str(data))
        with open(filepath, "w") as f:
            yaml.dump(data, f)

    @classmethod
    def load(
        cls, dirn: str = ".", reporter_config: dict | None = None
    ) -> "WorkflowRunner":
        """Load the results from a ``WorkflowRunner.yaml`` file.

        Parameters
        ----------
        dirn : str, optional
            The directory to load from. Default is the current directory.
        reporter_config : dict, optional
            The caller's reporter configuration for a continued run. When
            given, the reporter *backends* (Confluence / HTML and their
            settings) follow this configuration instead of the persisted
            one, while the loaded run's ``report_name`` (its identity) is
            kept. This keeps a resume from resurrecting a reporter the
            caller has since disabled: building a ConfluenceReporter
            contacts the server, so a stale persisted config can kill the
            resume (e.g. with a 403) before any module runs.

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
        persisted_reporter_config = data["reporter_config"]
        if reporter_config is not None:
            adopted = copy.deepcopy(reporter_config)
            adopted["report_name"] = persisted_reporter_config["report_name"]
            instance.reporter_config = adopted
        else:
            instance.reporter_config = persisted_reporter_config
        instance.analysis_config = data["analysis_config"]
        instance.analysis_config["result_location"] = os.path.join(dirn, "..")
        instance.workflow_modules = data["workflow_modules"]
        # absent in yamls written before the pristine snapshot existed
        instance.workflow_modules_pristine = data.get(
            "workflow_modules_pristine"
        )
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

    def adopt_workflow_modules(self, new_modules: list) -> None:
        """Adopt an edited module list on resume, keeping the previous one.

        The typical resume scenario is "fix a parameter and re-run": the
        caller's (edited) modules must replace the previous run's, whose
        *pristine* parameter snapshot (recorded before ``$``-command
        resolution and module write-back mutated them) is kept in
        ``self._previous_workflow_modules`` so that :meth:`_plan_resume`
        can re-run from the first changed module. The edited list is only
        adopted if its module-name sequence matches the previous run's;
        otherwise the previous modules are kept (warned). Best-effort: an
        error here must not break the resume, only disable adoption.

        Parameters
        ----------
        new_modules : list of tuple
            The workflow modules as configured now, as
            ``(module_name, parameters)``.
        """
        try:
            if not new_modules:
                return
            prev_pristine = self.workflow_modules_pristine
            prev_names = [name for name, _ in self.workflow_modules]
            new_names = [name for name, _ in new_modules]
            if prev_names != new_names:
                logger.warning(
                    "Not adopting the edited workflow modules on resume: "
                    f"the module sequence changed ({prev_names} -> "
                    f"{new_names}); keeping the previous run's modules."
                )
                return
            prev_modules = self.workflow_modules
            self.workflow_modules = new_modules
            self.workflow_modules_pristine = copy.deepcopy(new_modules)
            if prev_pristine is not None:
                # exact comparison against the pristine snapshot
                self._previous_workflow_modules = prev_pristine
                self._previous_modules_are_legacy = False
            else:
                # yamls recorded before the pristine snapshot existed only
                # hold the mutated parameters (resolved $-commands, module
                # write-backs); fall back to the conservative legacy
                # comparison against those
                logger.info(
                    "The previous run has no pristine parameter snapshot "
                    "(recorded by an older version); using a best-effort "
                    "comparison -- module-estimated values may trigger "
                    "extra re-runs, and a removed parameter is not "
                    "detected."
                )
                self._previous_workflow_modules = prev_modules
                self._previous_modules_are_legacy = True
        except Exception as e:
            logger.warning(
                f"Could not adopt the edited workflow modules on resume "
                f"({e!r}); keeping the previous run's modules."
            )

    def _memory_is_warm(self) -> bool:
        """Whether the analysis worker still holds in-memory dataset state.

        True for a live in-process re-run (the previous modules' effects are
        still present); False for a resumed run built from disk, where a
        checkpoint restore is needed.

        Returns
        -------
        bool
        """
        ap = getattr(self, "autopicasso", None)
        if ap is None:
            return False
        return any(
            getattr(ap, attr, None) is not None
            for attr in ("locs", "movie", "identifications", "channel_locs")
        )

    def _restart_conflicts(
        self, restart_index: int, checkpoint: dict | None
    ) -> tuple[list, list]:
        """Conflicts of restarting at ``restart_index`` after a restore.

        Wraps :func:`~picasso_workflow.modulespec.restart_conflicts` with
        the capabilities actually lost given the (possibly partial)
        checkpoint, and never lets the best-effort annotation layer break
        a run.

        Parameters
        ----------
        restart_index : int
            Index of the first module that would execute.
        checkpoint : dict or None
            The checkpoint that would be restored, or None.

        Returns
        -------
        hard : list of str
        soft : list of str
        """
        try:
            return restart_conflicts(
                self.workflow_modules,
                restart_index,
                lost_capabilities=_checkpoint_lost_capabilities(checkpoint),
            )
        except Exception as e:  # never let the spec layer break a run
            logger.debug(f"restart_conflicts skipped ({e!r}).")
            return [], []

    def _plan_resume(self) -> ResumePlan:
        """Decide where to start this run and what state to restore.

        The *frontier* is the first module that must re-run: the first one
        that did not previously succeed (per saved results + module folder),
        or the first one whose parameters changed since the previous run,
        whichever comes first. If in-memory state is still warm (live
        re-run with unchanged parameters), execution skips straight to the
        frontier as before. Otherwise the latest previously-succeeded
        module with a restorable locs checkpoint on disk (that does not
        strand a needed in-memory dependency, see
        :func:`~picasso_workflow.modulespec.restart_conflicts`) determines
        the start: its locs are restored and the modules after it re-run.
        With no checkpoint, execution still continues at the frontier if
        the remaining modules need no lost in-memory state (file-mediated
        workflows, e.g. after a manual step); otherwise it starts from
        scratch.

        Returns
        -------
        ResumePlan
        """
        n = len(self.workflow_modules)
        frontier = n
        # one listdir instead of one per module: which module indices have
        # a result folder ("previously analyzed")
        try:
            found_prefixes = {
                d.split("_", 1)[0]
                for d in os.listdir(self.result_folder)
                if "_" in d
                and os.path.isdir(os.path.join(self.result_folder, d))
            }
        except FileNotFoundError:
            found_prefixes = set()
        for i, (module_name, _) in enumerate(self.workflow_modules):
            if not (
                self.module_previously_succeeded(i, module_name)
                and f"{i:02d}" in found_prefixes
            ):
                frontier = i
                break
        params_changed = False
        if self._previous_workflow_modules is not None:
            for i, (module_name, module_parameters) in enumerate(
                self.workflow_modules[:frontier]
            ):
                try:
                    if self._previous_modules_are_legacy:
                        changed = _module_parameters_changed_legacy(
                            self._previous_workflow_modules[i][1],
                            module_parameters,
                            module_name,
                        )
                    else:
                        changed = _module_parameters_changed(
                            self._previous_workflow_modules[i][1],
                            module_parameters,
                        )
                except Exception as e:
                    logger.debug(
                        f"Parameter comparison failed for module {i} "
                        f"({module_name}): {e}; treating as changed."
                    )
                    changed = True
                if changed:
                    logger.info(
                        f"Parameters of module {i:02d}_{module_name} "
                        "changed since the previous run; re-running from "
                        "there."
                    )
                    frontier = i
                    params_changed = True
                    break
        if frontier == 0:
            return ResumePlan(
                0,
                0,
                None,
                "no previous results to skip; running all "
                "modules from scratch",
            )
        if frontier == n:
            return ResumePlan(
                n,
                n,
                None,
                f"all {n} modules previously succeeded (unchanged "
                "parameters); skipping everything",
            )
        # A live in-process re-run may continue on its warm state -- but
        # not when a parameter change pulled the frontier back: the warm
        # state then post-dates the module to re-run, so restore from a
        # checkpoint (or re-run from scratch) instead.
        if not params_changed and self._memory_is_warm():
            return ResumePlan(
                frontier,
                frontier,
                None,
                "in-memory state still present; skipping to module "
                f"{frontier} without restore",
            )
        for k in range(frontier - 1, -1, -1):
            module_name = self.workflow_modules[k][0]
            module_id = f"{k:02d}_{module_name}"
            checkpoint = _checkpoint_from_module_results(
                module_name, self.results.get(module_id, {})
            )
            if checkpoint is None:
                continue
            hard, soft = self._restart_conflicts(k + 1, checkpoint)
            if hard:
                for msg in hard:
                    logger.info(f"Checkpoint at {module_id} not viable: {msg}")
                continue
            for msg in soft:
                logger.warning(f"Resume advisory: {msg}")
            checkpoint = dict(checkpoint)
            checkpoint["module_index"] = k
            checkpoint["module_id"] = module_id
            return ResumePlan(
                k + 1,
                frontier,
                checkpoint,
                f"restoring locs saved by {module_id}, re-running "
                f"modules {k + 1} to {n - 1}",
            )
        # No checkpoint. Continuing at the frontier without a restore is
        # still valid when none of the remaining modules needs in-memory
        # state from before it (file-mediated workflows) -- the behavior
        # resumes relied on before checkpoints existed.
        hard, soft = self._restart_conflicts(frontier, None)
        if not hard:
            for msg in soft:
                logger.warning(f"Resume advisory: {msg}")
            return ResumePlan(
                frontier,
                frontier,
                None,
                f"no locs checkpoint found, but modules {frontier} onward "
                "need no in-memory state from before; continuing at "
                f"module {frontier}",
            )
        for msg in hard:
            logger.info(f"Continuing at module {frontier} not viable: {msg}")
        return ResumePlan(
            0,
            frontier,
            None,
            f"module {frontier} must re-run but no restorable locs "
            "checkpoint was found before it; re-running from scratch",
        )

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
        # Expose the progress manager + this module's index so a module that
        # runs sub-modules (e.g. branch) can report a nested progress tree.
        self.autopicasso._progress_manager = self.progress
        self.autopicasso._module_index = i
        # Expose the reporters so the branch module can stream each branch's
        # sub-module reports to a live child page as they finish (reporters
        # without the hook, e.g. HTML, are ignored and report in one batch).
        self.autopicasso._branch_live_reporters = self.reporters

        # For the conditional_branch and branch modules, inject the
        # parameter_command_executor so they can resolve sub-module parameters.
        # Inject into a shallow copy so the live executor is never persisted
        # into workflow_modules/results by the following save().
        if fun_name in ("conditional_branch", "branch"):
            parameters = {
                **parameters,
                "parameter_command_executor": self.parameter_command_executor,
            }

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

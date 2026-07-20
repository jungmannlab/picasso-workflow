#!/usr/bin/env python
"""Interaction with Confluence.

Defines :class:`ConfluenceReporter`, which documents each analysis module's
parameters and results on Confluence (mirroring the
:class:`~picasso_workflow.util.AbstractModuleCollection` contract), and
:class:`ConfluenceInterface`, a thin wrapper around the Atlassian Confluence
API.

Author: Heinrich Grabmayr
Initial date: March 7, 2024
"""

from __future__ import annotations

import html

# import logging
from loguru import logger
import os
import time
import traceback

import numpy as np
import yaml
from atlassian import Confluence as con
from requests.exceptions import ConnectionError, HTTPError

from picasso_workflow.util import AbstractModuleCollection

# logger = logging.getLogger(__name__)


def _yaml_safe(value):
    """Recursively convert tuples to lists for ``yaml.safe_dump``.

    ``yaml.safe_dump`` has no Python-tuple representer; module parameters
    often hold command tuples like ``('$map', 'filepath')`` that would
    otherwise raise.

    Parameters
    ----------
    value : object
        The structure to sanitize.

    Returns
    -------
    object
        The structure with all tuples replaced by lists.
    """
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value


# Parameter keys that must never be rendered or serialized: the workflow
# runner injects a live ParameterCommandExecutor under this name.
_PARAM_BLACKLIST = ("parameter_command_executor",)


def _format_val(v):
    """Render a parameter/result value, unwrapping numpy scalars."""
    if type(v).__module__ == "numpy" and hasattr(v, "item"):
        try:
            return str(v.item())
        except Exception:
            pass
    return str(v)


def _expand_macro(title, mapping, skip_keys=()):
    """Build a collapsible Confluence 'expand' macro from a mapping.

    Parameters
    ----------
    title : str
        The macro title, e.g. ``"Parameters"``.
    mapping : dict
        Key/value pairs rendered as a bullet list.
    skip_keys : iterable of str, optional
        Keys to omit, e.g. non-serializable injected objects.

    Returns
    -------
    str
        The storage-format macro.
    """
    text = (
        '<ac:structured-macro ac:name="expand" ac:schema-version="1">'
        f'<ac:parameter ac:name="title">{html.escape(str(title))}'
        "</ac:parameter>"
        "<ac:rich-text-body>"
        "<ul>"
    )
    for k, v in mapping.items():
        if k in skip_keys:
            continue
        text += (
            f"<li>{html.escape(str(k))}: "
            f"{html.escape(_format_val(v))}</li>"
        )
    text += "</ul></ac:rich-text-body></ac:structured-macro>"
    return text


def _code_macro(text, language=None):
    """Wrap text in a Confluence code macro, preserving its formatting.

    Parameters
    ----------
    text : str
        The literal text, e.g. a traceback. Newlines are preserved.
    language : str, optional
        Syntax-highlighting language, e.g. ``"python"``.

    Returns
    -------
    str
        The storage-format macro.
    """
    # CDATA cannot contain the literal "]]>"; split it if present.
    text = str(text).replace("]]>", "]]]]><![CDATA[>")
    lang = (
        f'<ac:parameter ac:name="language">{html.escape(language)}'
        "</ac:parameter>"
        if language
        else ""
    )
    return (
        '<ac:structured-macro ac:name="code" ac:schema-version="1">'
        f"{lang}"
        "<ac:plain-text-body>"
        f"<![CDATA[{text}]]>"
        "</ac:plain-text-body>"
        "</ac:structured-macro>"
    )


def _innermost_project_frame(exc):
    """Describe the deepest traceback frame inside picasso_workflow.

    A traceback often ends deep in scipy/numpy, where the failing line says
    nothing about the workflow. This picks out the last frame that belongs
    to this package, which is the one worth showing first.

    Parameters
    ----------
    exc : BaseException
        The exception to inspect.

    Returns
    -------
    str or None
        e.g. ``"picasso_outpost.py:2625 in nndistribution_from_csr"``, with
        the source line appended when available. None if no project frame
        is present or the traceback cannot be walked.
    """
    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        frames = traceback.extract_tb(exc.__traceback__)
        for frame in reversed(frames):
            if os.path.abspath(frame.filename).startswith(pkg_dir):
                where = (
                    f"{os.path.basename(frame.filename)}:{frame.lineno} "
                    f"in {frame.name}"
                )
                if frame.line:
                    where += f" -- {frame.line.strip()}"
                return where
    except Exception:  # never let reporting mask the real error
        pass
    return None


def config_snapshot_macro(
    config, title="Workflow configuration (YAML snapshot)"
):
    """Build a collapsible Confluence 'expand' macro with a YAML config.

    Renders ``config`` as a YAML code block inside an expand macro, for
    reproducibility.

    Parameters
    ----------
    config : object
        The configuration to serialize.
    title : str, optional
        The macro title. Default is ``"Workflow configuration (YAML
        snapshot)"``.

    Returns
    -------
    str
        The storage-format macro, or an empty string if the config cannot be
        serialized (so a snapshot problem never blocks page creation).
    """
    try:
        yaml_text = yaml.safe_dump(
            _yaml_safe(config),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    except Exception as e:  # never block page creation on a dump issue
        logger.debug(f"Could not serialize config snapshot: {e}")
        return ""
    return (
        '<ac:structured-macro ac:name="expand" ac:schema-version="1">'
        f'<ac:parameter ac:name="title">{html.escape(title)}</ac:parameter>'
        "<ac:rich-text-body>"
        f"{_code_macro(yaml_text, language='yaml')}"
        "</ac:rich-text-body>"
        "</ac:structured-macro>"
    )


def overview_body(title, rows, intro_html="", config=None):
    """Build a Confluence storage-format overview page body.

    A reusable page body used for run overview pages (e.g. the aggregation
    main page): a heading, an optional intro paragraph, a metadata table and
    an optional collapsible YAML snapshot of the run configuration.

    Parameters
    ----------
    title : str
        Page heading.
    rows : list of tuple
        ``(label, value)`` metadata rendered as a two-column table.
    intro_html : str, optional
        Intro paragraph(s); must already be valid storage-format HTML.
    config : dict or None, optional
        Run configuration rendered as a collapsible YAML snapshot.

    Returns
    -------
    str
        Confluence storage-format HTML body.
    """
    table_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    snapshot = config_snapshot_macro(config) if config is not None else ""
    return (
        f"<h1>{html.escape(title)}</h1>"
        f"{intro_html}"
        f"<table><tbody>{table_rows}</tbody></table>"
        f"{snapshot}"
    )


def module_decorator(method):
    """Wrap a reporter module to render its parameters and results.

    Builds collapsible "Parameters" and "Results" expand macros from the
    module's ``parameters`` and ``results`` dicts and passes them to the
    wrapped method as ``parameter_text`` and ``result_text``.

    Parameters
    ----------
    method : callable
        The reporter method to wrap.

    Returns
    -------
    callable
        The wrapped method.
    """

    def module_wrapper(self, i, parameters, results, postpone_report=False):
        # create parameter and results documentation
        parameter_text = _expand_macro(
            "Parameters", parameters, skip_keys=_PARAM_BLACKLIST
        )
        result_text = _expand_macro("Results", results)

        # call the module
        retval = method(
            self,
            i,
            parameters,
            results,
            parameter_text,
            result_text,
            postpone_report=postpone_report,
        )
        return retval

    return module_wrapper


# ---------------------------------------------------------------------------
# Unified Confluence credential resolution
# ---------------------------------------------------------------------------

# Two credential profiles. Non-secret fields come from a ``config.yaml``
# section (overridable by an env var); the token comes ONLY from an env var
# and is never read from config. ``parent_page_title`` maps to the config's
# ``DefaultPage`` key.
_CONFLUENCE_PROFILES = {
    "Confluence": {
        "section": "Confluence",
        "fields": {
            # field: (config key, env override)
            "base_url": ("URL", "CONFLUENCE_URL"),
            "space_key": ("Space", "CONFLUENCE_SPACE"),
            "parent_page_title": ("DefaultPage", "CONFLUENCE_BASE_PAGE"),
            "username": ("Username", "CONFLUENCE_USERNAME"),
        },
        # token env vars, in priority order (legacy alias last)
        "token_env": ("CONFLUENCE_TOKEN", "CONFLUENCE_BEARER"),
    },
    "ConfluenceTest": {
        "section": "ConfluenceTest",
        "fields": {
            "base_url": ("URL", "TEST_CONFLUENCE_URL"),
            "space_key": ("Space", "TEST_CONFLUENCE_SPACE"),
            "parent_page_title": ("DefaultPage", "TEST_CONFLUENCE_PAGE"),
            "username": ("Username", "TEST_CONFLUENCE_USERNAME"),
        },
        "token_env": ("TEST_CONFLUENCE_TOKEN",),
    },
}


def _strip_surrounding_quotes(value):
    """Strip a single pair of matching surrounding quotes from a string.

    The CI runner appends ``.env`` lines verbatim to ``$GITHUB_ENV``, which
    leaves values wrapped in quotes; stripping them here keeps the rest of
    the code quote-agnostic.
    """
    if (
        isinstance(value, str)
        and len(value) >= 2
        and value[0] == value[-1]
        and value[0] in "\"'"
    ):
        return value[1:-1]
    return value


def resolve_confluence_credentials(profile="Confluence", config=None):
    """Resolve Confluence credentials for a profile.

    Non-secret fields (``base_url``, ``space_key``, ``parent_page_title``,
    ``username``) come from the profile's ``config.yaml`` section, each
    overridable by its environment variable (env wins). The token comes only
    from an environment variable -- never from config -- and is ``None`` if
    unset. Surrounding quotes are stripped from env values.

    Parameters
    ----------
    profile : str, optional
        ``"Confluence"`` (operational) or ``"ConfluenceTest"`` (tests).
        Default is ``"Confluence"``.
    config : dict, optional
        The merged configuration. Defaults to the package-wide ``CONFIG``
        (imported lazily to avoid an import cycle).

    Returns
    -------
    dict
        ``{base_url, space_key, parent_page_title, username, token}``.
    """
    spec = _CONFLUENCE_PROFILES[profile]
    if config is None:
        from picasso_workflow import CONFIG as config
    section = (config or {}).get(spec["section"], {}) or {}

    creds = {}
    for field, (config_key, env_name) in spec["fields"].items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            creds[field] = _strip_surrounding_quotes(env_value)
        else:
            creds[field] = section.get(config_key)

    token = None
    for env_name in spec["token_env"]:
        raw = os.environ.get(env_name)
        if raw:
            token = _strip_surrounding_quotes(raw)
            break
    creds["token"] = token
    return creds


class ConfluenceReporter(AbstractModuleCollection):
    """Upload reports of automated picasso evaluations to Confluence.

    Implements the reporting side of the
    :class:`~picasso_workflow.util.AbstractModuleCollection` contract: for
    each analysis module there is a matching method that documents that
    module's parameters and results on the report's Confluence page.
    """

    def __init__(
        self,
        base_url: str,
        space_key: str,
        parent_page_title: str,
        report_name: str,
        username: str | None = None,
        token: str | None = None,
        parent_page_id: str | None = None,
    ):
        """Initialize the reporter and create (or reuse) its report page.

        Parameters
        ----------
        base_url : str
            Base URL of the Confluence instance.
        space_key : str
            Key of the Confluence space.
        parent_page_title : str
            Title of the parent page under which the report nests.
        report_name : str
            Title of the report page to create or reuse.
        username, token : str, optional
            Confluence credentials.
        parent_page_id : str, optional
            Id of the parent page. When given, the title-based parent lookup
            is skipped -- used on multi-rank runs where a cooperating rank
            already created the parent and shared its id, avoiding the
            eventually-consistent title lookup that would otherwise race
            page creation.
        """
        logger.debug("Initializing ConfluenceReporter.")

        self.ci = ConfluenceInterface(
            base_url,
            space_key,
            parent_page_title,
            username,
            token,
            parent_page_id=parent_page_id,
        )

        # create page
        self.report_page_name = report_name

        try:
            self.report_page_id = self.ci.create_page(
                self.report_page_name, body_text=""
            )
            logger.debug(f"Created page {self.report_page_name}")
        except ConfluenceInterfaceError:
            self.report_page_id, pgname = self.ci.get_page_properties(
                self.report_page_name
            )
            logger.debug(f"""Failed to create page {self.report_page_name}.
                Continuing on the pre-existing page""")

    def report_error(
        self,
        e,
        module,
        i=None,
        parameters=None,
        result_folder=None,
        previous_results=None,
    ):
        """Report an analysis error to Confluence.

        Appends a page section documenting an error that occurred during
        workflow execution: which module failed, what it was called with,
        and the full traceback in a code block so its formatting survives.

        Parameters
        ----------
        e : Exception
            The exception that occurred during analysis.
        module : str
            Name of the module where the error occurred.
        i : int, optional
            Index of the module in the workflow, used in the heading.
        parameters : dict, optional
            The module's (fully resolved) parameters. Rendered as a
            collapsible list; this is usually where the cause is visible.
        result_folder : str, optional
            The module's result folder, created before the failure.
        previous_results : dict, optional
            Results of the preceding module, giving the inputs this module
            was working from.
        """
        try:
            text = self._error_report_text(
                e, module, i, parameters, result_folder, previous_results
            )
        except Exception as report_error_exc:
            # A bug in reporting must never replace the real diagnosis.
            logger.error(
                f"Could not build the error report: {report_error_exc}"
            )
            text = (
                "<ac:layout><ac:layout-section ac:type='single'>"
                "<ac:layout-cell>"
                "<p><strong>ERROR OCCURRED</strong></p>"
                f"During analysis of {html.escape(str(module))}, an error "
                "occurred."
                f"{html.escape(str(e))}"
                f"{html.escape(traceback.format_exc())}"
                "</ac:layout-cell></ac:layout-section></ac:layout>"
            )
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def _error_report_text(
        self,
        e,
        module,
        i=None,
        parameters=None,
        result_folder=None,
        previous_results=None,
    ):
        """Build the storage-format body of an error report.

        See :meth:`report_error` for the parameters.

        Returns
        -------
        str
            The storage-format section.
        """
        if i is None:
            heading = f"Error in {html.escape(str(module))}"
        else:
            heading = (
                f"Module {i:02d}: {html.escape(str(module))} &mdash; FAILED"
            )

        facts = [
            f"<li>Exception type: {html.escape(type(e).__name__)}</li>",
            f"<li>Message: {html.escape(str(e))}</li>",
        ]
        frame = _innermost_project_frame(e)
        if frame:
            facts.append(
                f"<li>Failing picasso-workflow frame: "
                f"{html.escape(frame)}</li>"
            )
        if result_folder:
            facts.append(
                f"<li>Result folder: {html.escape(str(result_folder))}</li>"
            )

        # format_exception, not format_exc: the latter reads the ambient
        # sys.exc_info() and yields "NoneType: None" outside an except block.
        if e.__traceback__ is not None:
            tb_text = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
        else:
            tb_text = traceback.format_exc()

        text = (
            '<ac:layout><ac:layout-section ac:type="single">'
            "<ac:layout-cell>"
            f"<h2>{heading}</h2>"
            f"<ul>{''.join(facts)}</ul>"
            f"{_code_macro(tb_text, language='python')}"
        )
        if parameters is not None:
            text += _expand_macro(
                "Parameters", parameters, skip_keys=_PARAM_BLACKLIST
            )
        if previous_results is not None:
            text += _expand_macro("Preceding module results", previous_results)
        text += "</ac:layout-cell></ac:layout-section></ac:layout>"
        return text

    def dummy_module(self, i, parameters, results, postpone_report=False):
        """Report the placeholder ``dummy_module``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Uses no keys.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        postpone_report : bool, optional
            If True, build the report text but defer posting it. Default is
            False.
        """
        logger.debug("dummy_module.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Dummy Module</strong></p>
        Only for debugging purposes. Remove when workflow works.
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def conditional_branch(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``conditional_branch`` module to Confluence.

        Documents which branch was taken and the executed sub-modules.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, build the report text but defer posting it. Default is
            False.
        """
        logger.debug(f"Reporting conditional_branch module {i:02d}")

        # Extract condition information
        condition = results.get("condition", {})
        condition_result = results.get("condition_result", False)
        branch_taken = results.get("branch_taken", "unknown")
        branch_results = results.get("if_branch", {})
        skipped_branch = results.get("skipped_branch", "unknown")
        skipped_modules = results.get("skipped_modules", [])

        # Format condition for display
        if "left" in condition and "operator" in condition:
            left_val = condition.get("left", "?")
            operator = condition.get("operator", "?")
            right_val = condition.get("right", "?")
            condition_str = f"{left_val} {operator} {right_val}"
        else:
            condition_str = str(condition)

        # Create the main conditional module section
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Conditional Branch</strong></p>
        <h4>Condition Evaluation</h4>
        <ul>
        <li><strong>Condition:</strong> {html.escape(condition_str)}</li>
        <li><strong>Result:</strong> <span style="background-color: {'#c3e6cb' if condition_result else '#f5c6cb'}; padding: 2px 6px; border-radius: 3px;">{condition_result}</span></li>
        <li><strong>Branch Taken:</strong> <strong>{html.escape(branch_taken)}</strong></li>
        <li><strong>Skipped Branch:</strong> {html.escape(skipped_branch)}</li>
        <li><strong>Skipped Modules:</strong> {html.escape(', '.join(skipped_modules)) if skipped_modules else 'None'}</li>
        <li><strong>Start Time:</strong> {html.escape(str(results.get('start time', 'N/A')))}</li>
        <li><strong>Total Duration:</strong> {results.get("duration", 0) // 60:.0f} min {(results.get("duration", 0) % 60):.02f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # Create collapsible section for executed sub-modules
        text += f"""
        <ac:structured-macro ac:name="expand" ac:schema-version="1">
        <ac:parameter ac:name="title">Executed Sub-Modules ({len(branch_results)} modules in {branch_taken} branch)</ac:parameter>
        <ac:rich-text-body>
        """

        # Report on each executed sub-module
        for sub_key in sorted(branch_results.keys()):
            sub_results = branch_results[sub_key]
            # Extract module name from key (format: "00_module_name")
            module_name = "_".join(sub_key.split("_")[1:])

            text += f"""
            <div style="margin-left: 20px; border-left: 3px solid #4a90e2; padding-left: 10px; margin-bottom: 15px;">
            <h5>Sub-Module {html.escape(sub_key)}: {html.escape(module_name)}</h5>
            """

            # Check if module execution was successful
            if not sub_results.get("success", True):
                # text += f"""
                # <p style="color: #d9534f;"><strong>⚠ Module Failed</strong></p>
                # <p>Error: {html.escape(str(sub_results.get('error', 'Unknown error')))}</p>
                # """
                pass
            else:
                # Try to call the specific reporter for this sub-module
                if hasattr(self, module_name):
                    try:
                        logger.debug(
                            f"Calling reporter for sub-module: {module_name}"
                        )

                        # # Close all open tags to create valid HTML before intermediate update
                        # text += """
                        # </div>
                        # </ac:rich-text-body>
                        # </ac:structured-macro>
                        # </ac:layout-cell></ac:layout-section></ac:layout>
                        # """
                        # # Update page with content so far
                        # self.ci.update_page_content(
                        #     self.report_page_name, self.report_page_id, text
                        # )

                        # Call the sub-module reporter
                        reporter_method = getattr(self, module_name)
                        # Extract the sub-module index from the key
                        sub_idx = int(sub_key.split("_")[0])

                        # Get the sub-module parameters from the branch parameters
                        sub_module_params = {}
                        if branch_taken == "if_true":
                            branch_modules = parameters.get("if_true", [])
                        else:
                            branch_modules = parameters.get("if_false", [])

                        for idx, (mod_name, mod_params) in enumerate(
                            branch_modules
                        ):
                            if idx == sub_idx and mod_name == module_name:
                                sub_module_params = mod_params
                                break

                        # Call the reporter method
                        logger.debug("Now performing the call.")
                        module_text = reporter_method(
                            sub_idx,
                            sub_module_params,
                            sub_results,
                            postpone_report=True,
                        )
                        # logger.debug(f"not calling {str(reporter_method)}")

                        # # Start a new text section after the sub-module report
                        # # Reopen the layout and expand macro structure
                        # text = f"""
                        # <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
                        # <ac:structured-macro ac:name="expand" ac:schema-version="1">
                        # <ac:parameter ac:name="title">Executed Sub-Modules ({len(branch_results)} modules in {html.escape(branch_taken)} branch)</ac:parameter>
                        # <ac:rich-text-body>
                        # <div style="margin-left: 20px; border-left: 3px solid #4a90e2; padding-left: 10px; margin-bottom: 15px;">
                        # """
                        text += module_text

                    except Exception as e:
                        logger.error(
                            f"Error calling reporter for {module_name}: {e}"
                        )
                        raise e
                        text += f"""
                        <p style="color: #f0ad4e;">⚠ Could not generate detailed report: {html.escape(str(e))}</p>
                        """
                else:
                    logger.warning(
                        f"No reporter found for sub-module: {module_name}"
                    )
                    # Show basic results information
                    text += """
                    <ac:structured-macro ac:name="expand" ac:schema-version="1">
                    <ac:parameter ac:name="title">Results</ac:parameter>
                    <ac:rich-text-body>
                    <ul>
                    """
                    for k, v in sub_results.items():
                        if k not in ["folder", "start time", "end time"]:
                            text += f"<li>{k}: {html.escape(str(v))}</li>"
                    text += """
                    </ul>
                    </ac:rich-text-body>
                    </ac:structured-macro>
                    """

            text += "</div>"

        # Close the collapsible section
        text += """
        </ac:rich-text-body>
        </ac:structured-macro>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        logger.debug(f"updating page with text: {text}")

        # Update the page
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def analysis_documentation(
        self, i, parameters, results, postpone_report=False
    ):
        """Report the ``analysis_documentation`` module to Confluence.

        Tabulates the recorded hardware/software metadata.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting analysis_documentation.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Analysis Hard- and Software</strong></p>
        <ac:structured-macro ac:name="expand" ac:schema-version="1">
        <ac:parameter ac:name="title">Results</ac:parameter>
        <ac:rich-text-body>
        <ul>
        """
        for k, v in results.items():
            text += f"<li>{html.escape(str(k))}: {html.escape(str(v))}</li>"
        text += """
        </ul>
        </ac:rich-text-body>
        </ac:structured-macro>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    ##########################################################################
    # Single dataset modules
    ##########################################################################

    def convert_zeiss_movie(
        self, i, parameters, results, postpone_report=False
    ):
        """Report the ``convert_zeiss_movie`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting convert_zeiss_movie.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Converting Movie from .czi into .raw</strong></p>
        <p>Converted the file {parameters["filepath"]} to
        {results["filepath_raw"]} in {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s.</p>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def load_dataset_movie(
        self, i, pars_load, results_load, postpone_report=False
    ):
        """Report the ``load_dataset_movie`` module to Confluence.

        Documents the loaded movie, its size and (if created) the subsampled
        sample movie.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        pars_load, results_load : dict
            The ``load_dataset_movie`` module's parameters and results (see
            the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting a loaded dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Load Movie</strong></p>
        <ul>
        <li>Picasso Version: {results_load['picasso version']}</li>
        <li>Movie Location: {pars_load['filename']}</li>
        <li>Movie Size: Frames: {results_load['movie.shape'][0]},
        Width: {results_load['movie.shape'][1]},
        Height: {results_load['movie.shape'][2]}</li>
        <li>Start Time: {results_load['start time']}</li>
        <li>Duration: {results_load["duration"] // 60:.0f} min
        {(results_load["duration"] % 60):.02f} s</li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        if (sample_mov_res := results_load.get("sample_movie")) is not None:
            text += f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p>Subsampled Frames</p>
            <ul>
            <li> {len(sample_mov_res['sample_frame_idx'])} frames:
             {str(sample_mov_res['sample_frame_idx'])}</li>
            </ul>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """
            # Upload movie attachment immediately
            logger.debug("Uploading movie of subsampled images.")
            self.ci.upload_attachment(
                self.report_page_id, sample_mov_res["filename"]
            )
            # Add movie reference to text
            movie_filename = os.path.split(sample_mov_res["filename"])[1]
            text += f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p>
            <ac:structured-macro ac:name="multimedia" ac:schema-version="1">
            <ac:parameter ac:name="name"><ri:attachment ri:filename="{movie_filename}"/></ac:parameter>
            </ac:structured-macro>
            </p>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """

        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def load_dataset_localizations(
        self, i, parameters, results, postpone_report=False
    ):
        """Report the ``load_dataset_localizations`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting a loaded dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Load localizations</strong></p>
        <ul>
        <li>Picasso Version: {results['picasso version']}</li>
        <li>Localizations Location: {parameters['filename']}</li>
        <li>Number of localizations: {results['nlocs']}</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def identify(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``identify`` module to Confluence.

        Documents the identification settings and counts, and uploads the
        auto-netgrad and identifications-vs-frame plots if present.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting Identification.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Identify</strong></p>
        <ul>
        <li>Min Net Gradient: {parameters['min_gradient']:,.0f}</li>
        <li>Box Size: {parameters['box_size']} px</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Identifications found: {results['num_identifications']:,}
        </li>
        </ul>
        {parameter_text}
        {result_text}
        """

        fig_fps = []
        titles = []

        # if (res_autonetgrad := results.get("auto_netgrad")) is not None:
        #     logger.debug("Uploading graph for auto_netgrad.")
        #     self.ci.upload_attachment(
        #         self.report_page_id, res_autonetgrad["filename"]
        #     )
        #     self.ci.update_page_content_with_image_attachment(
        #         self.report_page_name,
        #         self.report_page_id,
        #         os.path.split(res_autonetgrad["filename"])[1],
        #     )
        # if (res := results.get("ids_vs_frame")) is not None:
        #     logger.debug("uploading graph for identifications vs frame.")
        #     self.ci.upload_attachment(self.report_page_id, res["filename"])
        #     self.ci.update_page_content_with_image_attachment(
        #         self.report_page_name,
        #         self.report_page_id,
        #         os.path.split(res["filename"])[1],
        #     )

        if fp_fig := results.get("auto_netgrad", {}).get("filename"):
            fig_fps.append(fp_fig)
            titles.append("Automatic min_grad detection")

        if fp_fig := results.get("ids_vs_frame", {}).get("filename"):
            fig_fps.append(fp_fig)
            titles.append("#identifications vs frame")

        if len(fig_fps) > 0:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                            <ac:image ac:height="350">
                            <ri:attachment ri:filename="{fn}" />
                            </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def localize(self, i, parameters, results, postpone_report=False):
        """Report the ``localize`` module to Confluence.

        Documents the localization run and uploads the locs-vs-frame plot if
        present.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting Localization of spots.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Localize</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Locs Column names: {results['locs_columns']}</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

        if (res := results.get("locs_vs_frame")) is not None:
            # print('uploading graph')
            self.ci.upload_attachment(self.report_page_id, res["filename"])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res["filename"])[1],
            )

    @module_decorator
    def zfit(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``zfit`` module to Confluence.

        Documents the z calibration used and uploads the calibration and
        z-histogram figures if present.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting zfit.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: fit z coordinate to localizations</strong></p>
        z fitting using astigmatism.
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Used z calibration from: {results['fp_calibration']}</li>
        <li>Magnification Factor: {parameters['magnification_factor']}</li></ul>
        {parameter_text}
        {result_text}
        """

        fig_fps = []
        titles = []

        if fp_fig := results.get("fp_calibration_fig"):
            fig_fps.append(fp_fig)
            titles.append("Calibration graphs")

        if fp_fig := results.get("fp_fig_zhist"):
            fig_fps.append(fp_fig)
            titles.append("z histogram")

        if len(fig_fps) > 0:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                            <ac:image ac:height="350">
                            <ri:attachment ri:filename="{fn}" />
                            </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def load_picassoconfig(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``load_picassoconfig`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting load_picassoconfig.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Load picasso CONFIG</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Loaded new configuration from: {parameters['fp_config']}</li>
        <li>saved config for documentation: {results['fp_config']}</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def export_brightfield(
        self, i, parameters, results, postpone_report=False
    ):
        """Report the ``export_brightfield`` module to Confluence.

        Uploads each exported brightfield PNG to the report page.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting export_brightfield.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Exporting Brightfield</strong></p>
        <ul>
        """
        text += f"""
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        # Accumulate images
        for label, fp in results.get("labeled filepaths", {}).items():
            text += f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p><strong>{label}</strong></p>
            """
            # Upload image immediately
            self.ci.upload_attachment(self.report_page_id, fp)
            # Add image reference to text
            filename = os.path.split(fp)[1]
            text += f"""
            <p>
            <ac:image><ri:attachment ri:filename="{filename}"/></ac:image>
            </p>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """

        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def render(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``render`` module to Confluence.

        Uploads the full-FOV, center-of-mass-zoom and tiled renderings.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting render.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Rendering Localizations</strong></p>
        Summary:
        <ul>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        generate_active_rois = parameters.get("generate_active_rois", True)
        rois = results.get("fp_scene_rois", [])

        text += "<table><tr>"

        # Left column: Field of View Overviews
        text += "<td style='vertical-align: middle; text-align: center; padding-right: 20px;'>"
        if fp_fullfov := results.get("fp_scene_fullfov"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fullfov)
            except ConfluenceInterfaceError:
                pass
            fn_fullfov = os.path.split(fp_fullfov)[1]

            fp_unmarked = results.get("fp_scene_fullfov_unmarked")
            if fp_unmarked:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp_unmarked)
                except ConfluenceInterfaceError:
                    pass
                fn_unmarked = os.path.split(fp_unmarked)[1]

                text += f"""
                    <p><b>Overview Images</b></p>
                    <p>
                        <ac:image ac:height="450">
                            <ri:attachment ri:filename="{fn_unmarked}" />
                        </ac:image>
                        &nbsp;&nbsp;
                        <ac:image ac:height="450">
                            <ri:attachment ri:filename="{fn_fullfov}" />
                        </ac:image>
                    </p>
                """
            else:
                text += f"""
                    <p><b>Overview Image</b></p>
                    <ac:image ac:height="450">
                        <ri:attachment ri:filename="{fn_fullfov}" />
                    </ac:image>
                """
        text += "</td>"

        # Right column: Either active site images OR the Zoom-in image depending on selection
        text += "<td style='vertical-align: middle; text-align: center;'>"
        if generate_active_rois and rois:
            text += "<p><b>Zoom-In Images (Density-Driven)</b></p>"
            text += "<table style='border-collapse: collapse; border: none; margin-left: auto; margin-right: auto;'>"
            for idx in range(0, len(rois), 2):
                text += "<tr>"
                for sub_idx in range(idx, min(idx + 2, len(rois))):
                    fp_roi = rois[sub_idx]
                    try:
                        self.ci.upload_attachment(self.report_page_id, fp_roi)
                    except ConfluenceInterfaceError:
                        pass
                    fn_roi = os.path.split(fp_roi)[1]
                    text += f"""
                        <td style='border: 1px solid #ddd; padding: 6px; text-align: center;'>
                            <ac:image ac:height="200">
                                <ri:attachment ri:filename="{fn_roi}" />
                            </ac:image>
                            <br/><b>Site {sub_idx + 1}</b>
                        </td>"""
                # If odd number of ROIs and this is the last row, add an empty cell for layout alignment
                if len(rois) % 2 != 0 and idx + 1 >= len(rois):
                    text += "<td style='border: none;'></td>"
                text += "</tr>"
            text += "</table>"
        else:
            if fp_ctrmass := results.get("fp_scene_ctrmass"):
                try:
                    self.ci.upload_attachment(self.report_page_id, fp_ctrmass)
                except ConfluenceInterfaceError:
                    pass
                fn_ctrmass = os.path.split(fp_ctrmass)[1]
                text += f"""
                    <p><b>Zoom-In Image</b></p>
                    <ac:image ac:height="450">
                        <ri:attachment ri:filename="{fn_ctrmass}" />
                    </ac:image>
                """
        text += "</td>"

        text += "</tr></table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def undrift_rcc(self, i, parameters, results, postpone_report=False):
        """Report the ``undrift_rcc`` module to Confluence.

        Documents the RCC settings and uploads the drift plot if present.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting undrifting via RCC.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Undrifting via RCC</strong></p>
        <ul><li>Dimensions: {parameters.get('dimensions')}</li>
        <li>Segmentation: {parameters.get('segmentation')}</li>
        """
        if msg := results.get("message"):
            text += f"""<li>Note: {msg}</li>"""
        text += f"""
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

        if driftimg_fn := results.get("filepath_plot"):
            self.ci.upload_attachment(self.report_page_id, driftimg_fn)
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(driftimg_fn)[1],
            )

    @module_decorator
    def undrift_rsso(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``undrift_rsso`` module to Confluence.

        Documents the iterative RSSO drift correction (magnitudes, confidence
        intervals, iterations) and uploads the drift plots.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting undrift_rsso.")

        drift_mag_x = results.get("drift_magnitude_x", np.nan)
        drift_mag_y = results.get("drift_magnitude_y", np.nan)
        total_drift = results.get("total_drift", np.nan)
        drift_quality = results.get("mean_drift_quality", np.nan)

        # Confidence interval metrics
        mean_uncertainty_x = results.get("mean_uncertainty_x", np.nan)
        mean_uncertainty_y = results.get("mean_uncertainty_y", np.nan)

        # Get iterative RSSO specific metrics
        n_iterations = results.get("n_iterations", 1)
        converged = results.get("converged", False)
        convergence_rms = results.get("convergence_rms", np.nan)
        subsampling_fraction = results.get("subsampling_fraction", 1.0)
        mean_uncertainty_x_new = results.get(
            "uncertainty_x-mean", mean_uncertainty_x
        )
        mean_uncertainty_y_new = results.get(
            "uncertainty_y-mean", mean_uncertainty_y
        )

        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Undrifting via Iterative RSSO</strong></p>
        Summary:
        <ul>
        <li>t<sub>on</sub>: {parameters.get('ton')} frames</li>
        <li>t<sub>off</sub>: {parameters.get('toff')} frames</li>
        <li>Max shift per frame: {parameters.get('max_shift')} pixels</li>
        <li>Processing chunk size: {parameters.get('chunk_size', 100)} frames</li>
        <li>Min locs per frame: {parameters.get('min_locs_per_frame', 10)}</li>
        <li>Max iterations: {parameters.get('max_iterations', 5)}</li>
        <li>Subsampling fraction: {subsampling_fraction:.1%}</li>
        <li>Numba optimization: {parameters.get('enable_numba_optimization', True)}</li>
        </ul>

        <p><strong>Convergence Results:</strong></p>
        <ul>
        <li><strong>Iterations performed:</strong> {n_iterations}</li>
        <li><strong>Converged:</strong> {'Yes' if converged else 'No'}</li>
        <li><strong>Final RMS change:</strong> {convergence_rms:.3f} nm</li>
        </ul>

        <p><strong>Drift Results:</strong></p>
        <ul>
        <li><strong>Total drift X:</strong> {drift_mag_x:.2f} nm</li>
        <li><strong>Total drift Y:</strong> {drift_mag_y:.2f} nm</li>
        <li><strong>Total drift magnitude:</strong> {total_drift:.2f} nm</li>
        <li>Mean drift quality: {drift_quality:.1f} measurements/frame</li>
        </ul>

        <p><strong>Uncertainty Analysis:</strong></p>
        <ul>
        <li><strong>Mean uncertainty X:</strong> {mean_uncertainty_x_new:.3f} nm</li>
        <li><strong>Mean uncertainty Y:</strong> {mean_uncertainty_y_new:.3f} nm</li>
        </ul>

        <ul>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # Add drift plot if available
        # Try new key first, fall back to legacy key for compatibility
        drift_plot = results.get("drift_plot") or results.get("fp_fig")
        if drift_plot:
            try:
                self.ci.upload_attachment(self.report_page_id, drift_plot)
            except ConfluenceInterfaceError:
                pass
            _, drift_plot_name = os.path.split(drift_plot)
            text += f"""
            <p><strong>Drift Trajectory:</strong></p>
            <ac:image ac:align="center" ac:layout="center">
            <ri:attachment ri:filename="{drift_plot_name}" />
            </ac:image>
            """

        # Add convergence plot if available (for iterative RSSO)
        if convergence_plot := results.get("convergence_plot"):
            try:
                self.ci.upload_attachment(
                    self.report_page_id, convergence_plot
                )
            except ConfluenceInterfaceError:
                pass
            _, convergence_plot_name = os.path.split(convergence_plot)
            text += f"""
            <p><strong>Convergence Analysis:</strong></p>
            <ac:image ac:align="center" ac:layout="center">
            <ri:attachment ri:filename="{convergence_plot_name}" />
            </ac:image>
            """

        # Add robustness plot if available (for iterative RSSO)
        if robustness_plot := results.get("robustness_plot"):
            try:
                self.ci.upload_attachment(self.report_page_id, robustness_plot)
            except ConfluenceInterfaceError:
                pass
            _, robustness_plot_name = os.path.split(robustness_plot)
            text += f"""
            <p><strong>Robustness Assessment:</strong></p>
            <ac:image ac:align="center" ac:layout="center">
            <ri:attachment ri:filename="{robustness_plot_name}" />
            </ac:image>
            """

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        if postpone_report:
            return text
        else:
            try:
                self.ci.update_page_content(
                    self.report_page_name, self.report_page_id, text
                )
            except Exception as e:
                logger.error(
                    f"Error updating Confluence page {self.report_page_name}, {self.report_page_id}"
                )
                logger.debug(text)
                raise e

    @module_decorator
    def undrift_aim(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``undrift_aim`` module to Confluence.

        Documents the AIM settings and uploads the drift plot if present.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting undrift_aim.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Undrifting via AIM</strong></p>
        Summary:
        <ul>
        <li>Dimensions: {parameters.get('dimensions')}</li>
        <li>Segmentation: {parameters.get('segmentation')} frames</li>
        <li>Intersect distance: {parameters.get('intersect_d')} nm</li>
        <li>Local search region radius: {parameters.get('roi_r')} nm</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def manual(self, i, parameters, results, postpone_report=False):
        """ """
        logger.debug("Reporting manual step")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Manual step</strong></p>
        <ul><li>prompt: {parameters.get('prompt')}</li>
        <li>filename: {parameters.get('filename')}</li>
        <li>file present: {results.get('success')}</li>
        <li>Start Time: {results['start time']}</li>
        </ul>"""
        if not results.get("success"):
            text += "<p>" + results["message"] + "</p>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def summarize_dataset(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        logger.debug("Reporting summarize dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Summarize Dataset</strong></p>"""
        for meth, meth_pars in parameters["methods"].items():
            if meth.lower() == "nena":
                meth_res = results["nena"]
                if meth_res.get("NeNa") is not None:
                    nenastr = meth_res.get("NeNa")
                else:
                    nenastr = "None"
                if meth_res.get("chisqr") is not None:
                    chisqstr = f"{meth_res.get('chisqr'):.1f}"
                else:
                    chisqstr = "None"
                text += f"""
                    <p>NeNa</p>
                    <ul>
                    <li>NeNa value: {nenastr}</li>
                    <li>Chi Square: {chisqstr}</li>
                    </ul>"""
                if fp_nena := meth_res.get("filepath_plot"):
                    self.ci.upload_attachment(self.report_page_id, fp_nena)
                    _, fn_nena = os.path.split(fp_nena)
                    text += (
                        "<ul><ac:image><ri:attachment "
                        + f'ri:filename="{fn_nena}" />'
                        + "</ac:image></ul>"
                    )
            elif meth.lower() == "median-loc-precision":
                meth_res = results["median-loc-precision"]
                if meth_res.get("median_lp-px") is not None:
                    lppxstr = f"{meth_res.get('median_lp-px'):.4f} px"
                else:
                    lppxstr = "None"
                if meth_res.get("median_lp-nm") is not None:
                    lpnmstr = f"{meth_res.get('median_lp-nm'):.2f} nm"
                else:
                    lpnmstr = "None"
                text += f"""
                    <p>Median Localization Precision</p>
                    <ul>
                    <li>med loc prec [px]: {lppxstr}</li>
                    <li>med loc prec [nm]: {lpnmstr}</li>
                    </ul>"""
        text += parameter_text
        text += result_text
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    # def aggregate_cluster(self, i, parameters, results):
    #     logger.debug("Reporting aggregate_cluster.")
    #     text = f"""
    #     <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
    #     <p><strong>aggregate_cluster</strong></p>
    #     <ul><li>Start Time: {results['start time']}</li>
    #     <li>Duration: {results["duration"] // 60:.0f} min
    #     {(results["duration"] % 60):.02f} s</li>
    #     <li>Number of locs after aggregating: {results.get('nlocs')}</li>
    #     </ul>"""

    #     text += """
    #     </ac:layout-cell></ac:layout-section></ac:layout>
    #     """
    #     self.ci.update_page_content(
    #         self.report_page_name, self.report_page_id, text
    #     )

    def density(self, i, parameters, results, postpone_report=False):
        """Report the ``density`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting density.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Local density computation</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Radius: {parameters.get('radius')}</li>
        </ul>"""

        text += """
        <b>TODO: generate plot for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def dbscan(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``dbscan`` module to Confluence.

        Documents the DBSCAN settings and uploads the cluster-size figure.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting dbscan.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: dbscan clustering</strong></p>
        Summary:
        <ul>
        <li>Radius: {parameters.get('radius'):.2f} nm</li>
        <li>min_samples: {parameters.get('min_samples')}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        if fp_fig := results.get("fp_fig_clustersizes"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def hdbscan(self, i, parameters, results, postpone_report=False):
        """Report the ``hdbscan`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting hdbscan.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: hdbscan clustering</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>min_cluster: {parameters.get('min_cluster')}</li>
        <li>min_sample: {parameters.get('min_sample')}</li>
        </ul>"""

        text += """
        <b>TODO: generate plot for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def binding_event_analysis(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``binding_event_analysis`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: binding_event_analysis</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        text += """
        <b>TODO: show plots for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def resolution_analysis(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``resolution_analysis`` module to Confluence.

        Documents the autocorrelation resolution metrics and uploads the
        resolution and radial-profile plots.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """

        resolution = results.get("resolution", np.nan)
        sigma_x = results.get("sigma_x", np.nan)
        sigma_y = results.get("sigma_y", np.nan)
        fit_quality = results.get("fit_quality", np.nan)

        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Resolution Analysis
        (Point Pattern Autocorrelation)</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Resolution: {resolution:.1f} nm (FWHM)</li>
        <li>σ<sub>x</sub>: {sigma_x:.1f} nm</li>
        <li>σ<sub>y</sub>: {sigma_y:.1f} nm</li>
        <li>Fit Quality (R²): {fit_quality:.3f}</li>
        <li>Grid spacing (Δr): {parameters.get('delta_r', 5.0):.1f} nm</li>
        <li>Max radius: {parameters.get('r_max', 100.0):.1f} nm</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # Add resolution plot if available
        if fig_resolution := results.get("fig_resolution"):
            text += f"""
            <p><strong>Resolution Analysis Plot:</strong></p>
            <ac:image ac:align="center" ac:layout="center"
                ac:original-height="400">
            <ri:attachment ri:filename="{os.path.basename(fig_resolution)}" />
            </ac:image>
            """
            self.ci.upload_attachment(self.report_page_id, fig_resolution)

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def resolution_frc_spatial(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``resolution_frc_spatial`` module to Confluence.

        Documents the spatial-FRC resolution metrics and uploads the FRC plot.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """

        resolution = results.get("resolution_frc_spatial", [np.nan])[0]
        resolution_unsmoothed = results.get("resolution_unsmoothed", np.nan)
        if isinstance(resolution_unsmoothed, tuple):
            resolution_unsmoothed = resolution_unsmoothed[0]
        resolution_std = results.get("resolution_std", np.nan)
        n_regions = results.get("n_regions", 0)
        n_regions_total = results.get("n_regions_total", 0)
        threshold = results.get("threshold", 1.0 / 7.0)

        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Resolution Analysis
        (Spatial FRC)</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Resolution (smoothed): {resolution:.1f} nm</li>
        <li>Resolution (unsmoothed): {resolution_unsmoothed:.1f} nm</li>
        <li>Resolution std (across regions): {resolution_std:.1f} nm</li>
        <li>Valid regions: {n_regions} / {n_regions_total}</li>
        <li>FRC threshold: {threshold:.3f} (1/7)</li>
        <li>Render pixel size: {parameters.get('pixelsize_render', 5.0):.1f} nm</li>
        <li>Region size: {parameters.get('region_size', 10.0):.1f} µm</li>
        <li>Min locs per region: {parameters.get('min_locs_per_region', 500)}</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # Add FRC plot if available
        if fig_frc := results.get("fig_frc"):
            text += f"""
            <p><strong>Spatial FRC Curve:</strong></p>
            <ac:image ac:align="center" ac:layout="center"
                ac:height="400">
            <ri:attachment ri:filename="{os.path.basename(fig_frc)}" />
            </ac:image>
            """
            self.ci.upload_attachment(self.report_page_id, fig_frc)

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def smlm_clusterer(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: smlm_clusterer clustering</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>radius: {parameters.get('radius'):.2f} nm</li>
        <li>min_locs: {parameters.get('min_locs')}</li>
        <li>basic_fa: {parameters.get('basic_fa')}</li>
        <li>radius_z: {parameters.get('radius_z')}</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        text += """
        <b>TODO: generate plot for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def gaussian_mixture_cluster(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        logger.debug("Reporting gaussian_mixture_cluster.")

        pct_discarded = (
            (results["n_locs_in"] - results["n_locs_clustered"])
            / results["n_locs_in"]
            * 100
        )
        locspctr = results["n_locs_clustered"] / results["n_centers"]
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}:
        Gaussian Mixture Model clustering</strong></p>
        Keeping centers as new locs.
        Summary:
        <ul>
        <li>Locs discarded: {pct_discarded:.1f} %</li>
        <li>Mean number of locs per center: {locspctr:.1f}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_fig_clustersizes"):
            fig_fps.append(fp_fig)
            titles.append("Cluster Size Distribution")
        if fp_fig := results.get("fp_fig_subclustering"):
            fig_fps.append(fp_fig)
            titles.append("Subcluster-test: sparse vs dense regions")

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def nneighbor(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        logger.debug("Reporting nneighbor.")
        d = len(parameters["dims"])
        density_rdf = results["density_rdf"]
        if isinstance(density_rdf, list):
            density_text = "; ".join(
                [f"{dens * 1e3**d:.02f} µm^{-d}" for dens in density_rdf]
            )
        else:
            density_text = f"{density_rdf * 1e3**d:.02f} µm^{-d}"
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Nearest Neighbor analysis</strong></p>
        Radial Distribution Function (RDF) and Nearest Neighbor Distributions.
        The RDF shows the density of spots in an annulus of a given radius
        r and thickness delta r, averaged over all spots. If the RDF deviates
        from the overall density, it means there is structure at that
        lengthscale in the data. E.g. the RDF is low at small distances due to
        finite resoltion.
        Summary:
        <ul>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Dimensions taken into account: {parameters['dims']}</li>
        <li>Bin size is the median of the first NN, divided by:
        {parameters['subsample_1stNN']}</li>
        <li>Displayed RDF up to nearest neighbor #: {parameters['nth_rdf']}
        </li>
        <li>Saved numpy txt file as: {results["nneighbors"]}</li>
        <li>Density from RDF: {density_text}
        </li>
        </ul>
        {parameter_text}
        {result_text}
        """
        if warnings := results.get("warnings"):
            warning_items = "".join(
                f"<li>{html.escape(str(w))}</li>" for w in warnings
            )
            text += f"""
            <ac:structured-macro ac:name="warning">
            <ac:rich-text-body>
            <p><strong>Warnings</strong></p>
            <ul>{warning_items}</ul>
            </ac:rich-text-body>
            </ac:structured-macro>
            """
        if fp_fig := results.get("fp_fig"):
            # try:
            #     self.ci.upload_attachment(self.report_page_id, fp_fig)
            # except ConfluenceInterfaceError:
            #     pass
            # _, fp_fig = os.path.split(fp_fig)
            # text += (
            #     "<ul><ac:image><ri:attachment "
            #     + f'ri:filename="{fp_fig}" />'
            #     + "</ac:image></ul>"
            # )

            if isinstance(fp_fig, str):
                fig_fps = [fp_fig]
                titles = ["locs"]
            elif isinstance(fp_fig, list):
                fig_fps = fp_fig
                titles = [""] * len(fp_fig)

            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def fit_csr(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``fit_csr`` module to Confluence.

        Documents the CSR fit (density, goodness-of-fit metrics) and uploads
        the fit figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting fit_csr.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Completely Spatially Random Distribution
        Fit</strong></p>
        The distance distributions of the first N neighbors in the data are
        fitted to the analytical CSR distributions simultaneously, using a
        maximum likelihood esitmator.
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>"""
        d = parameters["dimensionality"]
        if isinstance(results["density"], list):
            text += "<li>Density fitted:"
            for density in results["density"]:
                text += f"{density * 1e6} µm^(-{d}), "
            text += "</li>"
        else:
            text += f"""<li>Density fitted:
             {results['density'] * 1e6} µm^(-{d})</li>
            """

        # Add goodness-of-fit documentation
        # text += """</ul>
        # <p><strong>Goodness-of-Fit Assessment</strong></p>
        # <p>The quality of the CSR model fit is evaluated using two
        # complementary approaches:</p>
        # <ul>
        # <li><strong>Wasserstein Distance:</strong> Measures the
        # distributional difference between observed and theoretical CSR
        # nearest neighbor distances. Lower values indicate better fit
        # (typical range: 0.01-1.0 nm).</li>
        # <li><strong>Kolmogorov-Smirnov Tests:</strong> Statistical tests
        # for each k-th nearest neighbor order. Higher p-values (greater 0.05)
        # suggest good agreement with CSR, while lower p-values (smaller than
        # 0.05) indicate significant deviation from spatial randomness.</li>
        # </ul>
        # """
        text += """</ul>
        <p><strong>Goodness-of-Fit Assessment</strong></p>
        The quality of the CSR model fit is evaluated using
        <strong>Wasserstein Distance:</strong> Measures the distributional
        difference between observed and theoretical CSR nearest neighbor
        distances.
        Lower values indicate better fit (typical range: 0.01-1.0 nm).
        """

        # Add Wasserstein distances
        if mean_wasserstein_dist := results.get("mean_wasserstein_distance"):
            if isinstance(mean_wasserstein_dist, list):
                text += (
                    "<p><strong>Mean Wasserstein Distances:</strong></p><ul>"
                )
                for i_tag, dist in enumerate(mean_wasserstein_dist):
                    tag_name = (
                        f"Dataset {i_tag+1}"
                        if len(mean_wasserstein_dist) > 1
                        else "Dataset"
                    )
                    text += f"<li>{tag_name}: {dist:.3f} nm</li>"
                text += "</ul>"
            else:
                text += (
                    f"<p><strong>Mean Wasserstein Distance:</strong> "
                    f"{mean_wasserstein_dist:.3f} nm</p>"
                )

        # # Add KS test p-values
        # if ks_pvalues := results.get("ks_pvalues_per_k"):
        #     text += (
        #         "<p><strong>Kolmogorov-Smirnov Test Results "
        #         "(p-values):</strong></p>"
        #     )
        #     if isinstance(ks_pvalues[0], list) if ks_pvalues else False:
        #         # Multiple datasets
        #         for i_tag, pvalues_list in enumerate(ks_pvalues):
        #             tag_name = (
        #                 f"Dataset {i_tag+1}"
        #                 if len(ks_pvalues) > 1
        #                 else "Dataset"
        #             )
        #             text += f"<p><em>{tag_name}:</em></p><ul>"
        #             for k_idx, pvalue in enumerate(pvalues_list):
        #                 k = k_idx + parameters.get("kmin", 1)
        #                 text += (
        #                     f"<li style='margin-left: 20px;'>k={k}: "
        #                     f"p = {pvalue:.2e}</li>"
        #                 )
        #             text += "</ul>"
        #     else:
        #         # Single dataset
        #         text += "<ul>"
        #         for k_idx, pvalue in enumerate(ks_pvalues):
        #             k = k_idx + parameters.get("kmin", 1)
        #             text += (
        #                 f"<li style='margin-left: 20px;'>k={k}: "
        #                 f"p = {pvalue:.3f}</li>"
        #             )
        #         text += "</ul>"

        text += f"""
        {parameter_text}
        {result_text}
        """
        if fp_fig := results.get("fp_fig"):
            # try:
            #     self.ci.upload_attachment(self.report_page_id, fp_fig)
            # except ConfluenceInterfaceError:
            #     pass
            # _, fp_fig = os.path.split(fp_fig)
            # text += (
            #     "<ul><ac:image><ri:attachment "
            #     + f'ri:filename="{fp_fig}" />'
            #     + "</ac:image></ul>"
            # )

            if isinstance(fp_fig, str):
                fig_fps = [fp_fig]
                titles = ["locs"]
            elif isinstance(fp_fig, list):
                fig_fps = fp_fig
                titles = [""] * len(fp_fig)

            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def save_single_dataset(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("Reporting dataset saving.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Saving Resulting Dataset</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>filepath: {results.get('filepath')}</li>
        <li>saved number of locs: {results.get('nlocs')}</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    def load_datasets_to_aggregate(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("Reporting load_datasets_to_aggregate.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Loading Datasets to aggregate</strong></p>
        <ul><li>filepaths: {results.get('filepaths')}</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>tags: {results.get('tags')}</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def align_channels(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``align_channels`` module to Confluence.

        Documents the per-channel shifts (and RSSO uncertainties, if any) and
        uploads the before/after and shift-plot figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting align_channels.")
        shifttxt = f"""
        <li>Shifts in x [px]: {results.get('shifts')[0, :]}</li>
        <li>Shifts in y [px]: {results.get('shifts')[1, :]}</li>
        """
        try:
            shifttxt += f"""
                <li>Shifts in z [px]: {results.get('shifts')[2, :]}</li>"""
        except TypeError:
            pass
        except IndexError:
            pass

        # Add confidence analysis information if available
        shift_uncertainties = results.get("shift_uncertainties", {})
        confidence_txt = ""
        if (
            shift_uncertainties
            and results.get("alignment_algorithm") == "RSSO"
        ):
            mean_x_uncertainty = shift_uncertainties.get(
                "mean_x_uncertainty", np.nan
            )
            mean_y_uncertainty = shift_uncertainties.get(
                "mean_y_uncertainty", np.nan
            )
            max_x_uncertainty = shift_uncertainties.get(
                "max_x_uncertainty", np.nan
            )
            max_y_uncertainty = shift_uncertainties.get(
                "max_y_uncertainty", np.nan
            )

            if not np.isnan(mean_x_uncertainty):
                confidence_txt = f"""
                <li><strong>Confidence Analysis (RSSO method):</strong></li>
                <ul>
                <li>Mean X shift uncertainty: {mean_x_uncertainty:.3f} px</li>
                <li>Mean Y shift uncertainty: {mean_y_uncertainty:.3f} px</li>
                <li>Max X shift uncertainty: {max_x_uncertainty:.3f} px</li>
                <li>Max Y shift uncertainty: {max_y_uncertainty:.3f} px</li>
                <li>95% confidence intervals:
                ±{1.96*np.mean([mean_x_uncertainty, mean_y_uncertainty]):.3f}
                px</li>
                </ul>
                """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Align Channels</strong></p>
        <p>Channels are aligned via RCC if no fiducials are given, and via
        picked localizations if (picked, e.g. from find_gold) fiducials
        are given.</p>

        Summary:
        <ul>
        {shifttxt}
        {confidence_txt}
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_scene_locs_before"):
            fig_fps.append(fp_fig)
            titles.append("Localizations before alignment")
        if fp_fig := results.get("fp_scene_locs_after"):
            fig_fps.append(fp_fig)
            titles.append("Localizations after alignment")
        if fp_fig := results.get("fp_scene_fids_before"):
            fig_fps.append(fp_fig)
            titles.append("Fiducials before alignment")
        if fp_fig := results.get("fp_scene_fids_after"):
            fig_fps.append(fp_fig)
            titles.append("Fiducials after alignment")
        if fp_figs := results.get("fp_figs"):
            fig_fps += fp_figs
            titles += [f"Shift plot {i}" for i in range(len(fp_figs))]
        if fp_fig := results.get("fig_confidence_filepath"):
            fig_fps.append(fp_fig)
            titles.append("Channel shifts with confidence intervals")

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def combine_channels(self, i, parameters, results, postpone_report=False):
        """Report the ``combine_channels`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting combine_channels.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Combine Channels</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Combine map: {results["combine_map"]}</li>
        </ul>"""
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def save_datasets_aggregated(
        self, i, parameters, results, postpone_report=False
    ):
        """Report the ``save_datasets_aggregated`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting save_datasets_aggregated.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Saving Datasets aggregated</strong></p>
        <ul><li>filepaths: {results.get('filepaths')}</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>tags: {results.get('tags')}</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    # def spinna_manual(self, i, parameters, results, postpone_report=False):
    #     """ """
    #     logger.debug("Reporting spinna_manual.")
    #     text = f"""
    #     <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
    #     <p><strong>Module {i:02d}: SPINNA-Manual</strong></p>
    #     <ul><li>file present: {results.get('success')}</li>
    #     <li>Start Time: {results['start time']}</li>
    #     <li>Duration: {results["duration"] // 60:.0f} min
    #     {(results["duration"] % 60):.02f} s</li>
    #     """
    #     if not results["success"]:
    #         text += "<li>" + results["message"] + "</li>"
    #     else:
    #         text += f"<li>Result folder: {results['result_dir']}</li>"
    #         summary = pd.read_csv(results["fp_summary"])
    #         for i, row in summary.iterrows():
    #             text += f"<p><strong> Row {i} </strong></p><ul>"
    #             for col, val in row.items():
    #                 text += f"<li>{col}: {str(val)}</li>"
    #             text += "</ul>"
    #     text += """</ul>
    #     </ac:layout-cell></ac:layout-section></ac:layout>
    #     """
    #     if postpone_report:
    #         return text
    #     else:
    #         self.ci.update_page_content(
    #             self.report_page_name, self.report_page_id, text
    #         )
    #     if results["success"]:
    #         for fp in results["fp_fig"]:
    #             self.ci.upload_attachment(self.report_page_id, fp)
    #             self.ci.update_page_content_with_image_attachment(
    #                 self.report_page_name,
    #                 self.report_page_id,
    #                 os.path.split(fp)[1],
    #             )

    @module_decorator
    def spinna(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """ """
        logger.debug("Reporting spinna_manual.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: SPINNA</strong></p>
        Summary:
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li># simulated structures: {parameters["n_simulate"]}</li>
        <li># simulation repeats: {parameters["sim_repeats"]}</li>
        <li>Nearest Neighbors to evaluate: {parameters["n_nearest_neighbors"]}
        </li>
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = results.get("fp_figs", [])
        titles = ["" for _ in range(len(fig_fps))]

        if len(fig_fps) > 0:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def spinna_batch(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the SPINNA batch analysis module to Confluence.

        Writes the module summary, parameters and results, and embeds
        the NND figures (``results['fp_figs']``) as a table.
        """
        logger.debug("Reporting spinna_batch.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: SPINNA batch analysis</strong></p>
        Summary:
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = results.get("fp_figs", [])
        titles = ["" for _ in range(len(fig_fps))]

        if len(fig_fps) > 0:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def ripleysk(self, i, parameters, results, postpone_report=False):
        logger.debug("Reporting ripleysk.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Ripley's K Analysis</strong></p>
        <ul>
        <p>Ripley's K analyis investigates pair-wise clustering or dispersing
        organization between different channels. It is currently implemented
        in two different modes: "Ripleys"-mode is the analysis based on
        Ripley's K curves. To correct for finite-size and border effects,
        Ripley's K curves are normalized to mean and variance of
        completely spatially random simulations. "RDF"-mode is inspired by
        the above but calculates the radial distribution function (i.e.
        density at annulus of radius r instead of whole circle of radius r),
        and normalizes to a randomized version of the original data: for the
        evaluation of each radius r, each spot in the original data is moved by
        a random vector in a circle around it with radius r, to level out
        density fluctuations during normalization, in addition to the border
        effects. RDF is not Ripleys, it just uses the same infrastructure for
        testing.</p>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Type of analysis:
        {str(parameters["atype"])}</li>
        <li>Integral significance threshold:
        {parameters["ripleys_threshold"]}</li>
        <li>Ripleys Integrals location: {results["fp_ripleys_meanval"]}</li>
        <li>Significantly interacting pairs:
        {str(results["ripleys_significant"])}</li>
        </ul>"""

        if fp_fig := results.get("fp_fig_normalized"):
            text += "<ul><table>"
            text += "<tr><td><b>Normalized Curves</b></td>"
            text += "<td><b>Un-normalized Curves</b></td></tr>"
            text += "<tr><td>"
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="750"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_unnormalized")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="750"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td></tr></table></ul>"
        if fp_fig := results.get("fp_fig_ripleys_meanval"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
            text += (
                "The Ripley's mean value is the Ripley's K integral"
                + ", divided by the maximum integration distance."
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    # @module_decorator
    # def ripleysk_rafal(
    #     self, i, parameters, results, parameter_text, result_text
    # ):
    #     logger.debug("Reporting ripleysk_rafal.")
    #     text = f"""
    #     <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
    #     <p><strong>Module {i:02d}: Rafal's Ripley's K Analysis</strong></p>
    #     Summary:
    #     <ul>
    #     <li>Duration: {results["duration"] // 60:.0f} min
    #     {(results["duration"] % 60):.02f} s</li>
    #     </ul>
    #     {parameter_text}
    #     {result_text}
    #     """

    #     fig_fps = []
    #     titles = []
    #     if fp_fig := results.get("fp_fig_raw_binary"):
    #         fig_fps.append(fp_fig)
    #         titles.append("Raw Matrix_binary")
    #     if fp_fig := results.get("fp_fig_postprocessed_binary"):
    #         fig_fps.append(fp_fig)
    #         titles.append("Postprocessed Matrix_binary")
    #     if fp_fig := results.get("fp_fig_unnormalized_binary"):
    #         fig_fps.append(fp_fig)
    #         titles.append("raw Ripley's K curves_binary")
    #     if fp_fig := results.get("fp_fig_mask_binary"):
    #         fig_fps.append(fp_fig)
    #         titles.append("mask used_binary")
    #     if fp_fig := results.get("fp_fig_normalized_binary"):
    #         fig_fps.append(fp_fig)
    #         titles.append("normalized Ripley's K curves_binary")

    #     if len(fig_fps) > 1:
    #         fn_figs = []
    #         for fp in fig_fps:
    #             try:
    #                 self.ci.upload_attachment(self.report_page_id, fp)
    #             except ConfluenceInterfaceError:
    #                 pass
    #             fn_figs.append(os.path.split(fp)[1])

    #         text += "<table><tr>"
    #         for tit in titles:
    #             text += f"<td><b>{tit}</b></td>"
    #         text += "</tr>"
    #         text += "<tr>"
    #         for fn in fn_figs:
    #             text += f"""
    #                 <td>
    #                       <ac:image ac:height="350">
    #                       <ri:attachment ri:filename="{fn}" />
    #                       </ac:image>
    #                 </td>"""
    #         text += "</tr>"
    #         text += "</table>"

    #     fig_fps = []
    #     titles = []
    #     if fp_fig := results.get("fp_fig_raw_density"):
    #         fig_fps.append(fp_fig)
    #         titles.append("Raw Matrix_density")
    #     if fp_fig := results.get("fp_fig_postprocessed_density"):
    #         fig_fps.append(fp_fig)
    #         titles.append("Postprocessed Matrix_density")
    #     if fp_fig := results.get("fp_fig_unnormalized_density"):
    #         fig_fps.append(fp_fig)
    #         titles.append("raw Ripley's K curves_density")
    #     if fp_fig := results.get("fp_fig_mask_density"):
    #         fig_fps.append(fp_fig)
    #         titles.append("mask used_density")
    #     if fp_fig := results.get("fp_fig_normalized_density"):
    #         fig_fps.append(fp_fig)
    #         titles.append("normalized Ripley's K curves_density")

    #     if len(fig_fps) > 1:
    #         fn_figs = []
    #         for fp in fig_fps:
    #             try:
    #                 self.ci.upload_attachment(self.report_page_id, fp)
    #             except ConfluenceInterfaceError:
    #                 pass
    #             fn_figs.append(os.path.split(fp)[1])

    #         text += "<table><tr>"
    #         for tit in titles:
    #             text += f"<td><b>{tit}</b></td>"
    #         text += "</tr>"
    #         text += "<tr>"
    #         for fn in fn_figs:
    #             text += f"""
    #                 <td>
    #                       <ac:image ac:height="350">
    #                       <ri:attachment ri:filename="{fn}" />
    #                       </ac:image>
    #                 </td>"""
    #         text += "</tr>"
    #         text += "</table>"

    #     text += """
    #     </ac:layout-cell></ac:layout-section></ac:layout>
    #     """
    #     self.ci.update_page_content(
    #         self.report_page_name, self.report_page_id, text
    #     )

    @module_decorator
    def ripleysk2(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        logger.debug("Reporting ripleysk2.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Ripley's K Analysis (2)</strong></p>
        <p>Ripley's K analyis investigates pair-wise clustering or dispersing
        organization between different channels.
        </p>
        Summary:
        <ul>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Metric:
        {str(parameters["metric"])}</li>
        <li>Control Type:
        {str(parameters.get("controltype"))}</li>
        <li>z-score significance threshold:
        {parameters.get("ripleys_threshold")}</li>
        <li>Significantly interacting pairs:
        {str(results.get("ripleys_significant"))}</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        if fp_fig := results.get("fp_fig_normalized"):
            text += "<ul><table>"
            text += "<tr><td><b>Mean Value</b></td>"
            text += "<td><b>Normalized Curves</b></td>"
            text += "<td><b>Un-normalized Curves</b></td></tr>"
            text += "<tr><td>"
            fp_fig = results.get("fp_fig_ripleys_meanval")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ul><ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image></ul>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_normalized")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_unnormalized")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td></tr></table></ul>"

        text += (
            "The Ripley's mean value is the Ripley's K integral"
            + ", divided by the maximum integration distance."
        )
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def ripleysk_average(self, i, parameters, results, postpone_report=False):
        logger.debug("Reporting ripleysk_average.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Averaging of Repley's K Integrals</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Integral significance threshold:
        {parameters["ripleys_threshold"]}</li>
        <li>Loaded from workflows:
        {parameters["report_names"]}</li>
        <li>in folders:
        {parameters["fp_workflows"]}</li>
        <li>Folders to save significant pairs:
        {results["output_folders"]}</li>
        <li>Ripleys Integrals location:
        {results["fp_ripleys_significant"]}</li>
        <li>Significantly interacting pairs:
        {str(results["ripleys_significant"])}</li>
        </ul>"""

        if fp_fig := results.get("fp_figmeanvals"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def ripleysk_average2(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        logger.debug("Reporting ripleysk_average2.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Averaging of Ripley's K Integrals
        </strong></p>
        Summary:
        <ul>
        <li>Loaded from workflows:
        {parameters["report_names"]}</li>
        <li>in folders:
        {parameters["fp_workflows"]}</li>
        <li>Ripleys Integrals location:
        {results["fp_ripleys_significant"]}</li>
        <li>Significantly interacting pairs:
        {str(results["ripleys_significant"])}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        if fp_fig := results.get("fp_fig_normalized"):
            text += "<ul><table>"
            text += "<tr><td><b>Mean and std of mean values</b></td>"
            text += "<td><b>Normalized Curves of all datasets</b></td>"
            text += "<td><b>Un-normalized Curves of all datasets</b></td></tr>"
            text += "<tr><td>"
            fp_fig = results.get("fp_figmeanvals")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ul><ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image></ul>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_normalized")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_unnormalized")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="500"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td></tr></table></ul>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def protein_interactions(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("protein_interactions.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Direct Protein Interaction Analysis</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Interaction pairs analyzed:
        {parameters["interaction_pairs"]}</li>
        </ul>"""

        if fp_fig := results.get("fp_fig_imap"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        if props := results.get("Interaction proportions"):
            text += "<table>"
            text += "<tr>"
            for c in ["", "A", "AA", "B", "BB", "AB", "AABB"]:
                text += f"<td><b>{c}</b></td>"
            text += "</tr>"
            for pair, p in props.items():
                text += "<tr>"
                a, b = pair.split(",")
                text += f"<td><p>A: <b>{a}</b></p><p>B: <b>{b}</b></p></td>"
                if a == b:
                    p_disp = [
                        f"{c:.2f} %" if i < 2 else "NA"
                        for i, c in enumerate(p)
                    ]
                else:
                    p_disp = [f"{c:.2f} %" for i, c in enumerate(p)]
                for c in p_disp:
                    text += f"<td>{c}</td>"
                text += "</tr>"
            text += "</table>"

        if fp_fig := results.get("fp_allfigs"):
            text += "<table>"
            for i, fp_pairs in enumerate(fp_fig):
                text += "<tr>"
                for j, fp_combi in enumerate(fp_pairs):
                    try:
                        self.ci.upload_attachment(
                            self.report_page_id, fp_combi
                        )
                    except ConfluenceInterfaceError:
                        # aid = self.ci.get_attachment_id(
                        #     self.report_page_id, fp_combi)
                        # self.ci.delete_attachment(self.report_page_id, aid)
                        # self.ci.upload_attachment(
                        #     self.report_page_id, fp_combi
                        # )
                        pass
                    _, fp_combi = os.path.split(fp_combi)
                    text += "<td>"
                    text += f"""
                      <ac:image ac:height="150">
                      <ri:attachment ri:filename="{fp_combi}" />
                      </ac:image>"""
                    text += "</td>"
                text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def protein_interactions_average(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("protein_interactions_average.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Direct Protein Interaction Analysis
        Average</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""

        if fp_fig := results.get("fp_fig_imap"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def create_mask(self, i, parameters, results, postpone_report=False):
        """Report the ``create_mask`` module to Confluence.

        Uploads the blur and mask figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting create_mask.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Create Density Mask</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""
        if fp_fig_mask := results.get("fp_fig_mask"):
            fp_fig_blur = results["fp_fig_blur"]
            for fp in [fp_fig_blur, fp_fig_mask]:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
            fp_fig_mask = os.path.split(fp_fig_mask)[1]
            fp_fig_blur = os.path.split(fp_fig_blur)[1]

            text += "<table>"
            text += """
                <tr>
                <td><b>Blurred Combined Data</b></td>
                <td><b>Final Mask</b></td>
                </tr>"""
            text += f"""
                <tr>
                <td>
                      <ac:image ac:height="350">
                      <ri:attachment ri:filename="{fp_fig_blur}" />
                      </ac:image>
                </td>
                <td>
                      <ac:image ac:height="350">
                      <ri:attachment ri:filename="{fp_fig_mask}" />
                      </ac:image>
                </td>
                </tr>"""
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def create_mask2(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``create_mask2`` module to Confluence.

        Documents the cell area and uploads the localization and mask figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting create_mask2.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Create Density Mask</strong></p>
        Summary:
        <ul>
        <li>Area: {results["area"]} µm^2</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_scene_locs_before"):
            fig_fps.append(fp_fig)
            titles.append("Localizations for creating the mask")
        if fp_fig := results.get("fp_fig_mask_binary"):
            fig_fps.append(fp_fig)
            titles.append("Binary Mask")
        if fp_fig := results.get("fp_fig_mask_density"):
            fig_fps.append(fp_fig)
            titles.append("Density Mask")
        if fp_fig := results.get("fp_scene_locs_after"):
            fig_fps.append(fp_fig)
            titles.append("Localizations after applying the mask")

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def refine_mask_by_density(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``refine_mask_by_density`` module to Confluence.

        Uploads the density-histogram and refined-mask figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting refine_mask_by_density.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Refine Mask by Density</strong></p>
        Summary:
        <ul>
        <li>Area: {results["area_um^2"]} µm^2</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_density_hist_before"):
            fig_fps.append(fp_fig)
            titles.append("Density histogram before selection")
        if fp_fig := results.get("fp_fig_mask_density"):
            fig_fps.append(fp_fig)
            titles.append("Density Mask after selection")
        if fp_fig := results.get("fp_fig_mask_binary"):
            fig_fps.append(fp_fig)
            titles.append("Binary Mask after selection")
        if fp_fig := results.get("fp_density_hist_after"):
            fig_fps.append(fp_fig)
            titles.append("Density histogram after selection")
        if fp_fig := results.get("fp_scene_locs_before"):
            fig_fps.append(fp_fig)
            titles.append("Localizations before applying the mask")
        if fp_fig := results.get("fp_scene_locs_after"):
            fig_fps.append(fp_fig)
            titles.append("Localizations after applying the mask")

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def dbscan_molint(self, i, parameters, results, postpone_report=False):
        """Report the ``dbscan_molint`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting dbscan_molint.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: DBSCAN - Molecular Interaction version
        </strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def CSR_sim_in_mask(self, i, parameters, results, postpone_report=False):
        """Report the ``CSR_sim_in_mask`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting CSR_sim_in_mask.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: CSR simulation in density mask</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def dbscan_merge_cells(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("dbscan_merge_cells.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Merge DBSCAN results over multiple
        cells</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def dbscan_merge_stimulations(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("dbscan_merge_stimulations.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Merge DBSCAN results over multiple
        stimulations</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def binary_barcodes(self, i, parameters, results, postpone_report=False):
        logger.debug("binary_barcodes.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Analyse and plot binary barcodes</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def plot_densities(self, i, parameters, results, postpone_report=False):
        logger.debug("plot_densities.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Show Densities</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""
        if fp_fig := results.get("fp_fig_density"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig := results.get("fp_fig_area"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def find_cluster_motifs(
        self, i, parameters, results, postpone_report=False
    ):
        logger.debug("find_cluster_motifs.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Analyse and plot Cluster Motifs</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Threshold Cluster Population:
        {100 * parameters["population_threshold"]:.1f}%</li>
        <li>Threshold Exp Cells have barcode at least once:
        {100 * parameters["cellfraction_threshold"]:.1f}%</li>
        <li>t-Test threshold p-value:
        {parameters["ttest_pvalue_max"]:.3f}</li>
        <li>Significant Barcodes: {results["significant_barcodes"]}</li>
        </ul>"""
        if fp_fig := results.get("fp_fig_degreeofclustering"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig := results.get("fp_fig_fracdegreeofclustering"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig := results.get("fp_fig_nbarcodesbox"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig := results.get("fp_fig_abarcodesbox"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        if fp_fig_list := results.get("fp_fig_ntargets"):
            for fp_fig in fp_fig_list:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp_fig)
                except ConfluenceInterfaceError:
                    pass
                _, fp_fig = os.path.split(fp_fig)
                text += (
                    "<ul><ac:image><ri:attachment "
                    + f'ri:filename="{fp_fig}" />'
                    + "</ac:image></ul>"
                )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def interaction_graph(self, i, parameters, results, postpone_report=False):
        """Report the ``interaction_graph`` module to Confluence.

        Uploads the target-interaction graph figure.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting interaction_graph.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Interaction Graph</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        </ul>"""
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def find_gold(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``find_gold`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting find_gold.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Find Gold Beads</strong></p>
        Summary:
        <ul>
        <li># Gold Beads found: {results["n_gold"]}</li>
        <li># Gold Bead locs saved at: {results["fp_gold"]}</li>
        <li># Non-gold Bead locs saved at: {results["fp_nogold"]}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def find_similar(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``find_similar`` module to Confluence.

        Uploads the phase-space and picked-locs figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting find_structures.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Find Similar</strong></p>
        Summary:
        <ul>
        <li># Picks found: {results["n_picks"]}</li>
        <li># Locs picked: {results["n_picked_locs"]} of total
        {results["n_locs"]}
        ({(100 * results["n_picked_locs"] / results["n_locs"]):.1f} %)</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_phasespace_hexbin"):
            fig_fps.append(fp_fig)
            titles.append("Phase Space")
        if fp_fig := results.get("fp_phasespace"):
            fig_fps.append(fp_fig)
            titles.append("Phase Space Selection")
        if fp_fig := results.get("fp_picked_fullfov"):
            fig_fps.append(fp_fig)
            titles.append("Picked Locs")

        if len(fig_fps) > 0:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        # 2D table
        nrows = parameters.get("n_plot_structures")
        if nrows is not None:
            fig_fps = results.get("fp_renderings")  # list (col) of list of fps
            col_titles = [f"Example {i + 1}" for i in range(nrows)]
            if fig_fps is not None:
                row_titles = [
                    f"Structure Cluster {i}" for i in range(len(fig_fps))
                ]
            else:
                row_titles = []

            if fig_fps is not None and len(fig_fps) > 0:
                fn_figs = []
                for row_fps in fig_fps:
                    row_fns = []
                    for fp in row_fps:
                        try:
                            self.ci.upload_attachment(self.report_page_id, fp)
                        except ConfluenceInterfaceError:
                            pass
                        row_fns.append(os.path.split(fp)[1])
                    fn_figs.append(row_fns)

                text += "<table><tr><td></td>"
                for tit in col_titles:
                    text += f"<td><b>{tit}</b></td>"
                text += "</tr>"
                for row_tit, row_fns in zip(row_titles, fn_figs):
                    text += "<tr>"
                    text += f"<td><b>{row_tit}</b></td>"
                    for fn in row_fns:
                        text += f"""
                            <td>
                                  <ac:image ac:height="350">
                                  <ri:attachment ri:filename="{fn}" />
                                  </ac:image>
                            </td>"""
                    text += "</tr>"
                text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def find_structures(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``find_structures`` module to Confluence.

        Uploads the raw-cluster and pick-similar-cluster figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting find_structures.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Find Structures</strong></p>
        Summary:
        <ul>
        <li># Types of structures found: {results["n_clusters"]}</li>
        <li># Picks found for types: {results["n_picks"]}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_rawcluster"):
            fig_fps.append(fp_fig)
            titles.append("Raw clustering")
        if fp_fig := results.get("fp_picksimcluster"):
            fig_fps.append(fp_fig)
            titles.append("Pick similar clustering")

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"

        # 2D table
        nrows = parameters.get("n_plot_structures")
        if nrows is not None:
            fig_fps = results.get("fp_renderings")  # list (col) of list of fps
            col_titles = [f"Example {i + 1}" for i in range(nrows)]
            if fig_fps is not None:
                row_titles = [
                    f"Structure Cluster {i}" for i in range(len(fig_fps))
                ]
            else:
                row_titles = []

            if fig_fps is not None and len(fig_fps) > 0:
                fn_figs = []
                for row_fps in fig_fps:
                    row_fns = []
                    for fp in row_fps:
                        try:
                            self.ci.upload_attachment(self.report_page_id, fp)
                        except ConfluenceInterfaceError:
                            pass
                        row_fns.append(os.path.split(fp)[1])
                    fn_figs.append(row_fns)

                text += "<table><tr><td></td>"
                for tit in col_titles:
                    text += f"<td><b>{tit}</b></td>"
                text += "</tr>"
                for row_tit, row_fns in zip(row_titles, fn_figs):
                    text += "<tr>"
                    text += f"<td><b>{row_tit}</b></td>"
                    for fn in row_fns:
                        text += f"""
                            <td>
                                  <ac:image ac:height="350">
                                  <ri:attachment ri:filename="{fn}" />
                                  </ac:image>
                            </td>"""
                    text += "</tr>"
                text += "</table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def undrift_from_picked(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``undrift_from_picked`` module to Confluence.

        Uploads the drift figure.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting undrift_from_picked.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Undrift from picked</strong></p>
        Summary:
        <ul>
        <li># based on picked locs at: {parameters["fp_picked_locs"]}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += (
                "<ul><ac:image><ri:attachment "
                + f'ri:filename="{fp_fig}" />'
                + "</ac:image></ul>"
            )
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def filter_locs(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``filter_locs`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting filter_locs.")

        if isinstance(parameters["field"], str):
            fields = [parameters["field"]]
            minvals = [parameters.get("minval")]
            maxvals = [parameters.get("maxval")]
        else:
            fields = parameters["field"]
            minvals = parameters.get("minval")
            if minvals is None:
                minvals = [None] * len(fields)
            maxvals = parameters.get("maxval")
            if maxvals is None:
                maxvals = [None] * len(fields)
        txtfilt = ""
        for field, minval, maxval in zip(fields, minvals, maxvals):
            txtfilt += f"<li>{field}: {minval} - {maxval}</li>"
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Filter localizations</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Locs filtered from: {results["nlocs_before"]} to
        {results["nlocs_after"]} (down
        {(results["nlocs_before"] - results["nlocs_after"])
         / results["nlocs_before"] * 100:.1f}%)
        </li>
        <li>Fields filtered:<ul>{txtfilt}</ul></li>
        </ul>
        {parameter_text}
        {result_text}"""

        if fp_fig := results.get("fp_fig_before"):
            text += "<table>"
            text += "<tr><td><b>Before Filtering</b></td>"
            text += "<td><b>After Filtering</b></td></tr>"
            text += "<tr><td>"
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="350"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_after")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="350"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td></tr></table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def filter_transient_binding(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``filter_transient_binding`` module to Confluence.

        Documents the filter ranges and uploads the before/after histograms.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting filter_transient_binding.")

        fields = results["fields_filtered"]
        minvals = results.get("all_xmin")
        maxvals = results.get("all_xmax")
        txtfilt = ""
        for field, minval, maxval in zip(fields, minvals, maxvals):
            txtfilt += f"<li>{field}: {minval} - {maxval}</li>"
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}:
        Filter localizations for transient binding</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Locs filtered from: {results["nlocs_before"]} to
        {results["nlocs_after"]} (down
        {(results["nlocs_before"] - results["nlocs_after"])
         / results["nlocs_before"] * 100:.1f}%)
        </li>
        <li>Fields filtered:<ul>{txtfilt}</ul></li>
        </ul>
        {parameter_text}
        {result_text}
        """

        if fp_fig := results.get("fp_fig_before"):
            text += "<table>"
            text += "<tr><td><b>Before Filtering</b></td>"
            text += "<td><b>After Filtering</b></td></tr>"
            text += "<tr><td>"
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="350"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td><td>"
            fp_fig = results.get("fp_fig_after")
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig = os.path.split(fp_fig)
            text += f"""
                <ac:image ac:width="350"><ri:attachment
                ri:filename="{fp_fig}" />
                </ac:image>"""
            text += "</td></tr></table>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def link_locs(self, i, parameters, results, postpone_report=False):
        """Report the ``link_locs`` module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting link_locs.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Link localizations</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Maximum Distance [px]: {parameters["d_max"]}</li>
        <li>Maximum transient dark time: {parameters["tolerance"]}</li>
        </ul>"""
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    def insert_image(self, fp_fig, postpone_report=False):
        try:
            self.ci.upload_attachment(self.report_page_id, fp_fig)
        except ConfluenceInterfaceError:
            pass
        _, fn_fig = os.path.split(fp_fig)
        text = f"""
            <ac:image ac:width="500"><ri:attachment
            ri:filename="{fn_fig}" />
            </ac:image>"""
        return text

    @module_decorator
    def pairwise_module_executor(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``pairwise_module_executor`` module to Confluence.

        Documents the sub-module run for all channel pairs and uploads the
        result heatmap(s) and figures.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """
        logger.debug("Reporting pairwise_module_executor.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Pairwise execution of a submodule
        </strong></p>
        Summary:
        <ul>
        <li>Submodule executed: {parameters["module_name"]}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # add matrix figure
        if fp_fig := results.get("fp_fig_matrix"):
            text += "<ul>"
            text += self.insert_image(fp_fig)
            text += "</ul>"

        # add other figures
        if fp_figs := results.get("fp_figs"):
            fig_keys = parameters.get("result_fpfig")
            if not isinstance(fig_keys, list):
                fig_keys = [fig_keys]
            # first layer: different types of figures
            for matrix_figs, fig_key in zip(fp_figs, fig_keys):
                text += f"<p><b>{fig_key}</b></p>"
                text += "<ul><table>"
                for ir, fp_fig_row in enumerate(matrix_figs):
                    text += "<tr>"
                    for ic, fp_fig in enumerate(fp_fig_row):
                        text += "<td>"
                        text += self.insert_image(fp_fig)
                        text += "</td>"
                    text += "</tr>"
                text += "</table></ul>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )

    @module_decorator
    def random_val(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``random_val`` debugging module to Confluence.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """

    @module_decorator
    def labeling_efficiency_analysis(
        self,
        i,
        parameters,
        results,
        parameter_text,
        result_text,
        postpone_report=False,
    ):
        """Report the ``labeling_efficiency_analysis`` module to Confluence.

        Documents the derived labeling efficiencies (and uncertainties) and
        uploads the SPINNA NND figures. See the matching
        :class:`~picasso_workflow.util.AbstractModuleCollection` method for the
        model derivation.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters, results : dict
            The module's parameters and results (see the matching
            :class:`~picasso_workflow.util.AbstractModuleCollection` method).
        parameter_text, result_text : str
            Pre-rendered parameter/result macros from the decorator.
        postpone_report : bool, optional
            If True, return the report text instead of posting it. Default is
            False.
        """

        def show_dict_percentages(d):
            txt_out = "<ul>"
            for k, v in d.items():
                txt_out += f"<li>{k}: {100 * v:.2f} %</li>"
            txt_out += "</ul>"
            return txt_out

        logger.debug("Reporting labeling_efficiency_analysis.")

        le_std_txt = ""
        if lestd := results.get("labeling_efficiency_std", {}):
            if not all([v == 0 for v in lestd.values()]):
                le_std_txt = f"<li>Labeling efficiency std: {show_dict_percentages(lestd)}</li>"
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Labeling Efficiency Evaluation</strong></p>
        Summary:
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Labeling efficiency:
            {show_dict_percentages(results["labeling_efficiency"])}</li>
        {le_std_txt}
        </ul>
        {parameter_text}
        {result_text}
        """
        fig_fps = results.get("fp_fig", [])
        titles = ["" for i in range(len(fig_fps))]

        if len(fig_fps) > 1:
            fn_figs = []
            for fp in fig_fps:
                try:
                    self.ci.upload_attachment(self.report_page_id, fp)
                except ConfluenceInterfaceError:
                    pass
                fn_figs.append(os.path.split(fp)[1])

            text += "<table><tr>"
            for tit in titles:
                text += f"<td><b>{tit}</b></td>"
            text += "</tr>"
            text += "<tr>"
            for fn in fn_figs:
                text += f"""
                    <td>
                          <ac:image ac:height="350">
                          <ri:attachment ri:filename="{fn}" />
                          </ac:image>
                    </td>"""
            text += "</tr>"
            text += "</table>"
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        if postpone_report:
            return text
        else:
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )


class UndriftError(Exception):
    """Raised when an undrift step fails."""


# Number of times to retry a Confluence write that fails with an
# optimistic-locking (StaleState) version conflict, and the base backoff.
_STALE_STATE_RETRIES = 5
_STALE_STATE_BACKOFF = 0.5

# Confluence is eventually consistent: a just-created page is not immediately
# queryable by title/id. Retry a page lookup this many times (with backoff)
# before concluding the page really does not exist.
_PAGE_LOOKUP_RETRIES = 6
_PAGE_LOOKUP_BACKOFF = 0.5

# Confluence intermittently returns transient server-side failures (HTTP 5xx)
# or rate-limits (HTTP 429), e.g. while a page create/request is briefly
# delayed. These are not the caller's fault and usually succeed on retry.
_TRANSIENT_RETRIES = 5
_TRANSIENT_BACKOFF = 0.5

# The Confluence connection is occasionally dropped mid-request (the server
# closes the keep-alive socket -> RemoteDisconnected/ConnectionError).
# Reconnect and retry with backoff rather than failing the whole workflow.
_CONNECTION_RETRIES = 5
_CONNECTION_BACKOFF = 0.5


def _is_transient_error(error):
    """Return True for a retryable transient Confluence server error.

    Covers HTTP 429 (rate limited) and 5xx (server-side) responses, which
    are transient and typically succeed when the call is retried after a
    short backoff. Distinct from :func:`_is_stale_state_conflict`, which
    handles the 409 optimistic-locking case.

    Parameters
    ----------
    error : Exception
        The exception raised by the Confluence API call.

    Returns
    -------
    bool
        Whether the error is a retryable transient server error.
    """
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return status == 429 or 500 <= status < 600
    text = str(error).lower()
    return "too many requests" in text or "internal server error" in text


def _is_stale_state_conflict(error):
    """Return True for a Confluence optimistic-locking version conflict.

    Confluence rejects a page update whose cached version is stale with an
    HTTP 409 wrapping a Hibernate ``StaleStateException`` /
    ``ConflictException``. The update matched zero rows, so it did *not*
    take effect and is safe to retry after re-fetching the page version.

    Parameters
    ----------
    error : Exception
        The exception raised by the Confluence API call.

    Returns
    -------
    bool
        Whether the error is a retryable version conflict.
    """
    text = str(error)
    if "StaleStateException" in text or "ConflictException" in text:
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 409


def confluence_call(method):
    """Retry a Confluence API call on transient connection/version failures.

    The Confluence connection is sometimes lost; wrapping an interface method
    with this decorator reconnects and calls it again in that case. It also
    retries (with exponential backoff) on optimistic-locking version
    conflicts: each retry re-issues the call, which re-fetches the page's
    current version before writing.

    Parameters
    ----------
    method : callable
        The :class:`ConfluenceInterface` method to wrap.

    Returns
    -------
    callable
        The wrapped method.
    """

    def confluence_call_wrapper(self, *args, **kwargs):
        attempt = 0
        transient_attempt = 0
        conn_attempt = 0
        while True:
            try:
                # call the confluence api
                return method(self, *args, **kwargs)
            except ConnectionError as e:
                if conn_attempt < _CONNECTION_RETRIES:
                    conn_attempt += 1
                    wait = min(
                        _CONNECTION_BACKOFF * 2 ** (conn_attempt - 1), 8.0
                    )
                    logger.warning(
                        f"{method.__name__} lost the Confluence connection "
                        f"(attempt {conn_attempt}/{_CONNECTION_RETRIES}); "
                        f"reconnecting and retrying in {wait:.1f}s. ({str(e)})"
                    )
                    time.sleep(wait)
                    self.connect()
                    continue
                raise ConfluenceInterfaceError(
                    f"Calling {method.__name__} failed after "
                    f"{_CONNECTION_RETRIES} reconnect attempts: {str(e)}"
                )
            except HTTPError as e:
                if "unauthorized" in str(e).lower():
                    raise e
                if (
                    _is_stale_state_conflict(e)
                    and attempt < _STALE_STATE_RETRIES
                ):
                    attempt += 1
                    wait = min(_STALE_STATE_BACKOFF * 2 ** (attempt - 1), 8.0)
                    logger.warning(
                        f"{method.__name__} hit a Confluence version "
                        f"conflict (attempt {attempt}/{_STALE_STATE_RETRIES}"
                        f"); refetching page version and retrying in "
                        f"{wait:.1f}s."
                    )
                    time.sleep(wait)
                    continue
                if (
                    _is_transient_error(e)
                    and transient_attempt < _TRANSIENT_RETRIES
                ):
                    transient_attempt += 1
                    wait = min(
                        _TRANSIENT_BACKOFF * 2 ** (transient_attempt - 1), 8.0
                    )
                    logger.warning(
                        f"{method.__name__} hit a transient Confluence error "
                        f"(attempt {transient_attempt}/{_TRANSIENT_RETRIES}); "
                        f"retrying in {wait:.1f}s. ({str(e)})"
                    )
                    time.sleep(wait)
                    continue
                raise ConfluenceInterfaceError(
                    f"Calling {method.__name__} failed with HTTPError: "
                    f"{str(e)}"
                )

    return confluence_call_wrapper


class ConfluenceInterface:
    """Interface to the Confluence API.

    For API access, create an API token in Confluence and store it as the
    environment variable ``CONFLUENCE_BEARER`` (e.g.
    ``setx CONFLUENCE_BEARER "your_confluence_api_token"``); otherwise pass it
    at initialization.

    Parameters
    ----------
    base_url : str
        The Confluence URL to connect to.
    space_key : str
        The Confluence space key to work in.
    parent_page_title : str
        The (already existing) parent page to create the reports under.
    username : str, optional
        Username to authenticate with. If given, authentication uses url +
        username + password; if None or ``""``, url + token (Confluence
        Server needs token-based authentication).
    token : str, optional
        The password (for username-based auth) or token. If None, the
        ``CONFLUENCE_BEARER`` environment variable is used.
    """

    def __init__(
        self,
        base_url,
        space_key,
        parent_page_title,
        username=None,
        token=None,
        parent_page_id=None,
    ):
        """ """
        if token is None:
            self.bearer_token = self.get_bearer_token()
        else:
            self.bearer_token = token
        self.base_url = base_url
        if username != "":
            self.username = username
        else:
            self.username = None
        self.space_key = space_key

        logger.debug(f"confluence_url: {self.base_url}")
        logger.debug(f"confluence_space: {self.space_key}")
        # never log the token value itself
        logger.debug(f"confluence_token set: {bool(self.bearer_token)}")
        logger.debug(f"confluence_username: {self.username}")

        self.connect()

        if parent_page_id:
            # The caller already knows the parent page id (e.g. a cooperating
            # SLURM rank that created the page). Use it directly and skip the
            # title-based lookup, which is eventually consistent and can
            # transiently fail to find a just-created page (or race the rank
            # that creates it).
            self.parent_page_id = parent_page_id
        else:
            self.parent_page_id, _ = self.get_page_properties(
                parent_page_title
            )

    def connect(self):
        """Connect to confluence (cloud or server) by authentification depending
        on the settings stored at initialization.
        """
        if self.username is not None:
            self.confluence = con(
                url=self.base_url,
                username=self.username,
                password=self.bearer_token,
            )
        else:
            self.confluence = con(url=self.base_url, token=self.bearer_token)

    def get_bearer_token(self):
        """Return the operational Confluence token from the environment.

        Reads ``CONFLUENCE_TOKEN`` (the canonical name), falling back to the
        legacy ``CONFLUENCE_BEARER`` for backwards compatibility. The token is
        generated in the personal details of Confluence and is only ever
        supplied via the environment, never stored in config files.
        """
        return os.environ.get("CONFLUENCE_TOKEN") or os.environ.get(
            "CONFLUENCE_BEARER"
        )

    @confluence_call
    def get_page_properties(self, page_title="", page_id=""):
        """Get a page's id and title.

        Parameters
        ----------
        page_title, page_id : str, optional
            Look up the page by title or by id (exactly one must be given).

        Returns
        -------
        id : str
            The page id.
        title : str
            The page title.
        """
        if page_title == "" and page_id == "":
            logger.error("One of page_title and page_id must be given.")
            raise ConfluenceInterfaceError(
                "Cannot get page properties. "
                + "One of page_title and page_id must be given."
            )

        # A just-created page may not be queryable yet (eventual
        # consistency); retry the lookup with backoff before giving up.
        ident = page_title or page_id
        for attempt in range(_PAGE_LOOKUP_RETRIES + 1):
            if page_title != "":
                page = self.confluence.get_page_by_title(
                    space=self.space_key, title=page_title
                )
            else:
                page = self.confluence.get_page_by_id(page_id=page_id)

            if page is not None:
                return page["id"], page["title"]

            if attempt < _PAGE_LOOKUP_RETRIES:
                wait = min(_PAGE_LOOKUP_BACKOFF * 2**attempt, 8.0)
                logger.warning(
                    f"Page '{ident}' not found yet (attempt "
                    f"{attempt + 1}/{_PAGE_LOOKUP_RETRIES + 1}); it may be "
                    f"newly created. Retrying in {wait:.1f}s."
                )
                time.sleep(wait)

        raise ConfluenceInterfaceError(
            f"Page '{ident}' not found on {self.base_url}."
        )

    @confluence_call
    def get_page_version(self, page_title="", page_id=""):
        """Get a page's version number.

        Parameters
        ----------
        page_title, page_id : str, optional
            Look up the page by title or by id (exactly one must be given).

        Returns
        -------
        int
            The page version number.
        """
        if page_title != "":
            page = self.confluence.get_page_by_title(
                space=self.space_key, title=page_title, expand="version"
            )
        elif page_id != "":
            page = self.confluence.get_page_by_id(
                page_id=page_id, expand="body.version"
            )
        else:
            logger.exception("One of page_title and page_id must be given.")

        return page["version"]["number"]

    @confluence_call
    def get_page_body(self, page_title="", page_id=""):
        """Get a page's storage-format body.

        Parameters
        ----------
        page_title, page_id : str, optional
            Look up the page by title or by id (exactly one must be given).

        Returns
        -------
        str
            The page body in Confluence storage format.
        """
        if page_title != "":
            page = self.confluence.get_page_by_title(
                space=self.space_key, title=page_title, expand="body.storage"
            )
        elif page_id != "":
            page = self.confluence.get_page_by_id(
                page_id=page_id, expand="body.storage"
            )
        else:
            logger.exception("One of page_title and page_id must be given.")

        return page["body"]["storage"]["value"]

    @confluence_call
    def create_page(self, page_title, body_text, parent_id="rootparent"):
        """Create a Confluence page.

        Parameters
        ----------
        page_title : str
            The title of the page to create.
        body_text : str
            The page content (Confluence storage-format HTML).
        parent_id : str, optional
            The id of the parent page. If ``'rootparent'`` (the default), this
            interface's ``parent_page_id`` is used.

        Returns
        -------
        page_id : str
            The id of the newly created page.
        """
        if parent_id == "rootparent":
            parent_id = self.parent_page_id
        page = self.confluence.create_page(
            space=self.space_key,
            title=page_title,
            body=body_text,
            parent_id=parent_id,
            type="page",
            representation="storage",
            editor="v2",
            full_width=True,
        )
        return page["id"]

    @confluence_call
    def delete_page(self, page_id, recursive=False):
        # allow the page name to be used instead of page_id
        if isinstance(page_id, str) and not page_id.isnumeric():
            page_id, pgname = self.get_page_properties(page_id)
        self.confluence.remove_page(page_id, status=None, recursive=recursive)
        # implement logger

    @confluence_call
    def upload_attachment(self, page_id, filename):
        """Upload an attachment to a page.

        Parameters
        ----------
        page_id : str
            The page id to attach the file to.
        filename : str
            The local filename of the file to attach.

        Returns
        -------
        attachment_id : str
            The id of the attachment.
        """
        self.confluence.attach_file(
            filename=filename, page_id=page_id, space=self.space_key
        )

        target_name = os.path.basename(str(filename))
        attachments_container = self.confluence.get_attachments_from_content(
            page_id=page_id, start=0, limit=500
        )
        attachment_id = None
        for attachment in attachments_container["results"]:
            if attachment["title"] == target_name:
                attachment_id = attachment["id"]
                break

        if attachment_id is None:
            logger.warning(
                f"Uploaded '{target_name}' to page {page_id}, but no "
                "matching attachment was found when querying the page."
            )

        return attachment_id

    @confluence_call
    def get_attachment_id(self, page_id, filename):
        """Get the id of an attachment on a page.

        Parameters
        ----------
        page_id : str
            The page id holding the attachment.
        filename : str
            The filename of the attachment to look up.

        Returns
        -------
        attachment_id : str
            The id of the attachment.
        """
        attachments_container = self.confluence.get_attachments_from_content(
            page_id, start=0, limit=500
        )
        attachments = attachments_container["results"]
        for attachment in attachments:
            if attachment["title"].lower() == filename.lower():
                attachment_id = attachment["id"]
                break

        return attachment_id

    @confluence_call
    def delete_attachment(self, page_id, attachment_id):
        """Delete an attachment from a page.

        Parameters
        ----------
        page_id : str
            The page id holding the attachment.
        attachment_id : str
            The id of the attachment to delete.
        """
        self.confluence.delete_attachment(page_id, attachment_id, version=None)

    @confluence_call
    def update_page_content(
        self, page_name, page_id, body_update, replace=False
    ):
        if not replace:
            status = self.confluence.append_page(
                parent_id=None,
                page_id=page_id,
                title=page_name,
                append_body=body_update,
            )
        else:
            status = self.confluence.update_page(
                page_id=page_id,
                title=page_name,
                body=body_update,
            )
        return status

    @confluence_call
    def update_page_content_with_movie_attachment(
        self, page_name, page_id, filename
    ):
        body_update = f"""
            <ac:structured-macro ac:name="multimedia" ac:schema-version="1">
            <ac:parameter ac:name="autoplay">false</ac:parameter>
            <ac:parameter ac:name="name"><ri:attachment
            ri:filename=\"{filename}\" /></ac:parameter>
            <ac:parameter ac:name="loop">false</ac:parameter>
            <ac:parameter ac:name="width">30%</ac:parameter>
            <ac:parameter ac:name="height">30%</ac:parameter>
            </ac:structured-macro>
            """
        self.confluence.append_page(
            page_id,
            page_name,
            append_body=body_update,
            parent_id=None,
            type="page",
            representation="storage",
            minor_edit=False,
        )

    @confluence_call
    def update_page_content_with_image_attachment(
        self, page_name, page_id, filename
    ):
        body_update = (
            f'<ac:image ac:height="350"><ri:attachment ri:filename="{filename}" />'
            + "</ac:image>"
        )
        self.confluence.append_page(
            page_id,
            page_name,
            append_body=body_update,
            parent_id=None,
            type="page",
            representation="storage",
            minor_edit=False,
        )


class ConfluenceInterfaceError(Exception):
    """Raised when a :class:`ConfluenceInterface` operation fails."""


class NullConfluenceInterface:
    """No-op stand-in for :class:`ConfluenceInterface`.

    Mirrors the interface's method surface but performs no network calls, so
    coordinator code that talks to ``self.ci`` can run unchanged when
    Confluence documentation is disabled. Methods return benign values.
    """

    def __init__(self, *args, **kwargs):
        pass

    def connect(self):
        pass

    def get_page_properties(self, page_title="", page_id=""):
        return "local", page_title or page_id

    def get_page_version(self, page_title="", page_id=""):
        return 1

    def get_page_body(self, page_title="", page_id=""):
        return ""

    def create_page(self, page_title, body_text="", parent_id="rootparent"):
        return "local"

    def delete_page(self, page_id, recursive=False):
        pass

    def upload_attachment(self, page_id, filename):
        return os.path.basename(str(filename))

    def get_attachment_id(self, page_id, filename):
        return os.path.basename(str(filename))

    def delete_attachment(self, page_id, attachment_id):
        pass

    def update_page_content(
        self, page_name, page_id, body_update, replace=False
    ):
        return None

    def update_page_content_with_movie_attachment(
        self, page_name, page_id, filename
    ):
        return None

    def update_page_content_with_image_attachment(
        self, page_name, page_id, filename
    ):
        return None

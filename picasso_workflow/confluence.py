#!/usr/bin/env python
"""
Module Name: confluence.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Interaction with Confluence
"""
import html

# import logging
from loguru import logger
import os
import traceback

import numpy as np
import pandas as pd
import yaml
from atlassian import Confluence as con
from requests.exceptions import ConnectionError, HTTPError

from picasso_workflow.util import AbstractModuleCollection

# logger = logging.getLogger(__name__)


def _yaml_safe(value):
    """Recursively convert tuples to lists so a structure can be serialized
    with yaml.safe_dump (which has no Python-tuple representer). Module
    parameters often hold command tuples like ``('$map', 'filepath')`` that
    would otherwise raise.
    """
    if isinstance(value, dict):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value


def config_snapshot_macro(
    config, title="Workflow configuration (YAML snapshot)"
):
    """Build a collapsible Confluence 'expand' macro containing ``config``
    as a YAML code block, for reproducibility.

    Returns an empty string if the config cannot be serialized, so a
    snapshot problem never prevents a page from being created.
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
    # CDATA cannot contain the literal "]]>"; split it if present.
    yaml_text = yaml_text.replace("]]>", "]]]]><![CDATA[>")
    return (
        '<ac:structured-macro ac:name="expand" ac:schema-version="1">'
        f'<ac:parameter ac:name="title">{html.escape(title)}</ac:parameter>'
        "<ac:rich-text-body>"
        '<ac:structured-macro ac:name="code" ac:schema-version="1">'
        '<ac:parameter ac:name="language">yaml</ac:parameter>'
        "<ac:plain-text-body>"
        f"<![CDATA[{yaml_text}]]>"
        "</ac:plain-text-body>"
        "</ac:structured-macro>"
        "</ac:rich-text-body>"
        "</ac:structured-macro>"
    )


def overview_body(title, rows, intro_html="", config=None):
    """Build a Confluence storage-format overview page body.

    A reusable page body used for run overview pages (e.g. the aggregation
    main page): a heading, an optional intro paragraph, a metadata table,
    and an optional collapsible YAML snapshot of the run configuration.

    Args:
        title : str
            page heading
        rows : list of (label, value) tuples
            metadata rendered as a two-column table
        intro_html : str, optional
            intro paragraph(s); must already be valid storage-format HTML
        config : dict or None, optional
            run configuration rendered as a collapsible YAML snapshot
    Returns:
        str : Confluence storage-format HTML body
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
    def module_wrapper(self, i, parameters, results, postpone_report=False):
        # create parameter and results documentation
        parameter_text = """
            <ac:structured-macro ac:name="expand" ac:schema-version="1">
            <ac:parameter ac:name="title">Parameters</ac:parameter>
            <ac:rich-text-body>
            <ul>
            """

        def _format_val(v):
            if type(v).__module__ == "numpy" and hasattr(v, "item"):
                try:
                    return str(v.item())
                except Exception:
                    pass
            return str(v)

        for k, v in parameters.items():
            parameter_text += f"<li>{html.escape(str(k))}: {html.escape(_format_val(v))}</li>"

        parameter_text += """
        </ul>
        </ac:rich-text-body>
        </ac:structured-macro>
        """

        result_text = """
            <ac:structured-macro ac:name="expand" ac:schema-version="1">
            <ac:parameter ac:name="title">Results</ac:parameter>
            <ac:rich-text-body>
            <ul>
            """
        for k, v in results.items():
            result_text += f"<li>{html.escape(str(k))}: {html.escape(_format_val(v))}</li>"

        result_text += """
        </ul>
        </ac:rich-text-body>
        </ac:structured-macro>
        """

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


class ConfluenceReporter(AbstractModuleCollection):
    """A class to upload reports of automated picasso evaluations
    to confluence
    """

    def __init__(
        self,
        base_url,
        space_key,
        parent_page_title,
        report_name,
        username=None,
        token=None,
    ):
        logger.debug("Initializing ConfluenceReporter.")

        self.ci = ConfluenceInterface(
            base_url, space_key, parent_page_title, username, token
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
            logger.debug(
                f"""Failed to create page {self.report_page_name}.
                Continuing on the pre-existing page"""
            )

    def report_error(self, e, module):
        """Report errors that occur during analysis to Confluence.

        Creates a Confluence page section documenting errors that occurred
        during workflow execution, including exception details and traceback.

        Args:
            e : Exception
                The exception that occurred during analysis
            module : str
                Name or identifier of the module where the error occurred

        Returns:
            None
        """
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>ERROR OCCURRED</strong></p>
        During analysis of {module}, an error occurred.
        """
        text += html.escape(str(e))
        text += html.escape(traceback.format_exc())
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def dummy_module(self, i, parameters, results, postpone_report=False):
        """A module that does nothing, for quickly removing
        modules in a workflow without having to renumber the
        following result idcs. Only for workflow debugging,
        remove when done.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results (unchanged)
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
        """Execute different sub-module sequences based on a condition.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    condition : dict
                        condition dictionary with keys:
                            - "left": value or parameter command tuple
                            - "operator": str (>, <, >=, <=, ==, !=)
                            - "right": value or parameter command tuple
                        or logical condition with "and"/"or" keys
                    if_true : list of tuples
                        list of (module_name, module_parameters) tuples
                        to execute if condition is True
                    if_false : list of tuples
                        list of (module_name, module_parameters) tuples
                        to execute if condition is False
                optional keys:
                    parameter_command_executor : ParameterCommandExecutor
                        if provided, will be used for resolving parameter
                        commands in condition values
            results : dict
                the results this function generates

        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results including:
                    - condition_result : bool
                    - branch_taken : str ("if_true" or "if_false")
                    - if_branch : dict of sub-module results
                    - branch_modules : dict of flat-indexed results
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
        """This module documents where and how analysis is being performed
        Args:
            parameters : dict
                This module does not use any parameters
        Returns:
            parameters : dict
                as input, unchanged
            results : dict
                the analysis results, updated with:
                    picasso version : str
                        version of picasso library used
                    picasso-workflow version : str
                        version of picasso-workflow
                    Architecture : str
                        machine architecture
                    OS : str
                        operating system
                    host : str
                        hostname of machine
                    processor : str
                        processor information
                    CPU Frequency [MHz] : float
                        current CPU frequency
                    CPU cores : int
                        number of CPU cores
                    Memory total [GB] : int
                        total system memory in GB
                    Memory available [GB] : int
                        available system memory in GB
                    GPU : str
                        GPU name or "N/A"
                    GPU memory [GB] : int
                        GPU memory in GB or 0 if no GPU
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
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.
        Args:
            parameters : dict
                necessary items:
                    filepath : str
                        the czi file name to load.
                optional items:
                    filename_raw : str
                        the raw file name to write to
                    info : dict, information as used by picasso
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    filepath_raw : str
                        full path to the output raw file
                    filename_raw : str
                        name of the output raw file
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
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads DNA-PAINT movie data and metadata into memory for subsequent
        analysis. Optionally creates sample movies and loads camera
        configuration. The data is saved in self.movie and self.info.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filename : str
                        Path to the movie file to load
                Optional keys:
                    sample_movie : dict
                        Parameters for creating a subsampled movie
                    load_camera_info : bool
                        Whether to load camera configuration from
                        picasso.CONFIG
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    picasso version : str
                        Version of picasso library used
                    movie.shape : tuple
                        Movie dimensions (frames, width, height)
                    sample_movie : dict
                        Results from subsampled movie creation (if requested)

        Returns:
            parameters : dict
                Input parameters, potentially modified (sample_movie paths
                updated)
            results : dict
                Input results with added movie information and metadata
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
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        The data is saved in
            self.locs
            self.info
        Args:
            parameters : dict
                necessary items:
                    filename : str
                        the (main) file name to load. This can be image files,
                        or hdf5.
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    picasso version : str
                        version of picasso library used
                    nlocs : int
                        number of localizations loaded
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
        """Identifies localizations in a loaded dataset.

        Identifies potential localization sites in the loaded movie using
        net gradient thresholding. Optionally performs automatic net gradient
        detection and creates identification vs frame plots.
        The data is saved in self.identifications.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    box_size : int
                        Size of the detection box in pixels
                    min_gradient : float
                        Minimum net gradient threshold for detection
                        (required unless auto_netgrad is provided)
                Optional keys:
                    auto_netgrad : dict
                        Parameters for automatic net gradient detection:
                            box_size : int
                                Box size for auto detection
                            frame_numbers : list or int
                                Frame range for analysis
                            filename : str
                                Output filename for auto-detection plot
                            start_ng : float
                                Starting net gradient value
                            zscore : float
                                Z-score threshold for detection
                            bins : int
                                Number of histogram bins
                    ids_vs_frame : dict
                        Parameters for plotting identifications vs time:
                            filename : str
                                Output filename for plot
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    num_identifications : int
                        Total number of identifications found
                    auto_netgrad : dict
                        Results from automatic net gradient detection (if
                        requested)
                    ids_vs_frame : dict
                        Results from identifications vs frame analysis (if
                        requested)

        Returns:
            parameters : dict
                Input parameters, potentially with updated min_gradient
            results : dict
                Input results with identification statistics and optional plots
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
        """Localizes Spots previously identified.
        The data is saved in
            self.locs
        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    box_size : as always
                    fit_parallel : bool
                        whether to fit on multiple cores
                optional items:
                    locs_vs_frame : dict
                        for plotting locs vs time
                        items correspond to arguments of _plot_locs_vs_frame
                    save_locs : dict
                        if saving localizations is requested.
                        Items correpsond to arguments of save_locs
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    locs_vs_frame : dict
                        plot results if locs_vs_frame parameter was provided
                    locs_columns : list
                        list of column names in the localizations array
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
        """Fits z positions to previously localized spots.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    magnification_factor : float
                        the magnification factor for z calibration
                optional items:
                    fp_calibration : str
                        filepath to the 3D calibration yaml file
                        if not given
                    save_locs : dict
                        if saving localizations is requested.
                        Items correpsond to arguments of save_locs
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
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
        """
        Loads a specific picasso configuration file, as opposed to the default
        version residing in the picasso installation folder.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    fp_config : str
                        filepath to a config file.
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
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
        """Opens a single-plane tiff image and saves it to png with
        contrast adjustment.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    filepath : str or list of str or dict
                        the tiff file(s) to load. The converted file(s) will
                        have the same name, but with .png extension
                        if dict: keys are labels
                optional items:
                    min_quantile : float, default: 0
                        the quantile below which pixels are shown black
                    max_quantile : float, default: 1
                        the quantile above which pixels are shown white
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    labeled filepaths : dict
                        keys : labels
                        values : filepaths
                    success : bool
                        whether the export was successful
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
        """Renders localizations on the whole field of view, and on
        a zoom in around the center of mass of localizations.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                optional items:
                    ctrmass_fov_nm : Field of view of the zoom in rendering
                        around the center of mass in nm
                    fullfov_pixelsize : The rendered pixel size [nm] of the
                        full FOV rendering
                    ctrmass_pixelsize : The rendered pixel size [nm] of the
                        zoom in rendering around the center of mass
                    ctrmass_blur_method : Blur method
                    ctrmass_min_blur_width : min blur with
                    ctrmass_ang : angle
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    fp_scene_fullfov : str
                        filepath to full FOV rendering
                    fp_scene_ctrmass : str
                        filepath to center of mass zoom rendering (conditional, only if ctrmass_fov_nm provided)
                    fp_scene_tiles : list of lists of str
                        filepaths to the 5x5 tiled renderings
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
        """Undrifts localized data using redundant cross correlation.
        drift is saved in
        self.drift

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    segmentation : int
                        the number of frames segmented for RCC
                optional items:
                    max_iter_segmentations : int, default: 3
                        maximum number of iterations to adaptively increase segmentation if RCC fails
                    filename : str
                        the drift txt file name
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
                Note: dimensions parameter is set to ['x', 'y'] by this module
            results : dict
                the analysis results, updated with:
                    success : bool
                        whether undrifting was successful
                    message : str
                        error or warning messages if any
                    filepath_driftfile : str
                        filepath to drift txt file (conditional, only if undrifting succeeded)
                    filepath_plot : str
                        filepath to drift plot png (conditional, only if undrifting succeeded)
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
        """Undrift localized data using iterative RSSO-based drift correction

        This method applies an iterative RSSO (Redundant Spot Shift
        Overrepresentation) approach where each frame is compared against
        the whole dataset to compute total drift for that frame. The process
        is repeated iteratively with the undrifted dataset to improve accuracy.
        Includes uncertainty analysis, confidence evaluation, windowing and
        outlier detection.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    ton : float
                        Half-life of localization in frames (how long a spot
                        stays visible)
                    toff : float
                        Time in frames for a spot to reappear after
                        disappearing
                    max_shift : float
                        Maximum expected drift per frame in pixels
                optional items:
                    min_locs_per_frame : int
                        Minimum localizations per frame for reliable drift
                        estimation (default: 10)
                    max_iterations : int
                        Maximum number of iterative refinement rounds (default: 5)
                    convergence_threshold : float
                        RMS drift change threshold for convergence in nm (default: 0.1)
                    plot_drift : bool
                        Whether to save drift plots (default: True)
                    save_locs : bool
                        Whether to save undrifted localizations (default: True)
                    n_processes : int or None
                        Number of processes for parallel computation (default: auto)
                    confidence_threshold : float
                        Confidence threshold for windowing analysis (default: 0.8)
                    outlier_detection_enabled : bool
                        Enable RSSO failure and outlier detection (default: True)
                    outlier_z_threshold : float
                        Z-score threshold for temporal outlier detection (default: 3.5)
                    min_signal_to_noise : float
                        Minimum signal-to-noise ratio for drift measurements (default: 0.5)
                    windowing_enabled : bool
                        Enable adaptive windowing for low-confidence frames (default: True)
                    window_size_range : tuple
                        Min and max window sizes for adaptive windowing (default: (3, 20))

        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results including:
                    success : bool
                        whether drift correction succeeded
                    drift_x, drift_y : ndarray
                        total drift trajectories in nm for each frame
                    uncertainty_x, uncertainty_y : ndarray
                        uncertainty estimates for drift measurements
                    drift_quality : ndarray
                        quality/confidence metrics per frame
                    n_iterations : int
                        number of iterations performed
                    convergence_rms : float
                        final RMS change indicating convergence
                    drift_plots : str
                        path to drift visualization plots
        """
        logger.debug("Reporting undrift_rsso.")

        drift_mag_x = results.get("drift_magnitude_x", np.nan)
        drift_mag_y = results.get("drift_magnitude_y", np.nan)
        total_drift = results.get("total_drift", np.nan)
        drift_quality = results.get("mean_drift_quality", np.nan)

        coarse_drift_x = results.get("coarse_drift_magnitude_x", np.nan)
        coarse_drift_y = results.get("coarse_drift_magnitude_y", np.nan)
        fine_drift_x = results.get("fine_drift_magnitude_x", np.nan)
        fine_drift_y = results.get("fine_drift_magnitude_y", np.nan)

        # Confidence interval metrics
        mean_uncertainty_x = results.get("mean_uncertainty_x", np.nan)
        mean_uncertainty_y = results.get("mean_uncertainty_y", np.nan)
        confidence_95_x = results.get("confidence_95_x", np.nan)
        confidence_95_y = results.get("confidence_95_y", np.nan)
        max_uncertainty_x = results.get("max_uncertainty_x", np.nan)
        max_uncertainty_y = results.get("max_uncertainty_y", np.nan)

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
        """Unrift localized data using the AIM algorithm
        drift is saved in
        self.drift

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    segmentation : int
                        the number of frames segmented
                    intersect_d : float
                        Intersect distance in nanometers.
                    roi_r : float
                        Radius of the local search region in nanometers.
                        Should be larger than the maximum expected drift wihtin
                        segmentation.
                    dimensions : list of str
                        the dimensions undrifted, typically ['x', 'y'].
                optional items:
                    progress : callback function
                        progress callback for status updates
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    success : bool
                        whether undrifting was successful
                    fp_driftfile : str
                        filepath to drift txt file
                    fp_fig : str
                        filepath to drift plot png
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
        """Calculate local localization density
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    radius : float
                        the radius for calculating local density
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Perform clustering using dbscan.

        Applies DBSCAN clustering algorithm to localizations, optionally
        replacing localizations with cluster centers for subsequent analysis.
        After this module, the standard locs will be the cluster centers.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        The DBSCAN radius parameter in nm
                    min_samples : int
                        Minimum number of samples required for a cluster
                    continue_with_centers : bool
                        Whether to replace localizations with cluster centers
                Optional keys:
                    save_locs : bool
                        Whether to save clustered localization data to results
                        folder
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    fp_fig_clustersizes : str
                        Filepath to cluster size distribution figure
                    fp_centers : str
                        Filepath to cluster centers file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with clustering outputs and file paths
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
        """Perform hdbscan clustering. After this module, the standard
        locs will be the cluster centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    min_cluster : float
                        the hdbscan min_cluster
                    min_samples : float
                        the hdbscan min_sample
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Perform resolution analysis using point pattern autocorrelation

        This method calculates the spatial resolution of localizations
        by computing a 2D autocorrelation function and fitting a Gaussian to
        extract resolution metrics. The analysis includes 2D Gaussian fitting,
        radial profile computation, and 1D Gaussian fitting to the radial profile.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                with optional keys:
                    delta_r : float
                        grid spacing for autocorrelation (default: 5 nm)
                    r_max : float
                        maximum radius for autocorrelation (default: 100 nm)
                    batch_size : int or None
                        number of data points per batch for chunking (auto-calculated if None)
                    n_processes : int or None
                        number of parallel processes (auto-detected if None, capped at 4)
                    use_chunking : bool
                        enable memory-efficient chunking for large datasets (default: True)
                    use_sparse : bool
                        use sparse matrices for very large grids (default: False)

        Results:
            resolution : float
                average resolution in nm (FWHM)
            sigma_x, sigma_y : float
                Gaussian standard deviations in x,y directions
            fwhm_x, fwhm_y : float
                Full-width half-maximum in x,y directions
            fit_quality : float
                R-squared goodness of fit
            autocorr_map : ndarray
                2D autocorrelation intensity map
            radial_profile : ndarray
                radial profile of autocorrelation
            radial_distances : ndarray
                distance values for radial profile
            resolution_radial : float
                resolution from radial Gaussian fit (FWHM)
            resolution_dblradial : float
                resolution from double Gaussian fit (FWHM)
            fig_resolution : str
                path to resolution plot
            fig_radial : str
                path to radial profile plot
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
        """Calculate resolution using spatial FRC approach

        This method divides the FOV into spatial regions, computes FRC for each
        region independently, and averages the results. Benefits:
        - Lower memory usage (smaller images per region)
        - Better statistics through spatial averaging
        - Efficient multiprocessing (fully independent regions)
        - Preserves high spatial frequencies

        Args:
            i : int
                the index of the module
            parameters: dict
                with optional keys:
                    pixelsize_render : float
                        pixel size for rendered images in nm (default: 5 nm)
                    smoothing_sigma : float or None
                        Gaussian smoothing sigma in pixels (default: None)
                    threshold : float
                        FRC threshold for resolution cutoff (default: 1/7 ≈ 0.143)
                    region_size : float
                        size of each spatial region in micrometers (default: 10.0 µm)
                    min_locs_per_region : int
                        minimum localizations per region to process (default: 500)
                    max_frc_range_nm : float or None
                        maximum FRC range in nm (default: None = full range)
                    n_processes : int
                        number of parallel processes (default: 4)
                    smoothing_window : float
                        moving average window size for FRC smoothing in 1/nm
                        (default: 0.005)

        Results:
            resolution_frc_spatial : float
                mean FRC-based resolution in nm
            resolution_std : float
                standard deviation across regions
            n_regions : int
                number of valid regions processed
            cutoff_frequency : float
                mean spatial frequency at resolution cutoff (1/nm)
            frc_curve_mean : ndarray
                mean FRC curve across regions
            frc_curve_std : ndarray
                std of FRC curves
            spatial_frequencies : ndarray
                spatial frequency values (1/nm)
            threshold : float
                threshold used
            fig_frc : str
                path to FRC curve plot
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
        """Fit a Completely Spatially Random Distribution to nearest neighbors.

        Fits CSR model to nearest neighbor distance distributions and evaluates
        goodness-of-fit using statistical measures and visualization.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    nneighbors : str or numpy.ndarray or list
                        If str: filepath to nearest neighbor data file
                        If array: 2D array (N, k) of kth nearest neighbor
                        distances
                        If list: multiple datasets or file paths
                    dimensionality : int
                        Spatial dimensionality (2 or 3) for CSR model
                Optional keys:
                    kmin : int
                        Minimum k-th nearest neighbor order to fit (default: 1)
                    min_dist : float
                        Minimum observable distance in nm due to technical
                        limits
                    max_dist : float
                        Maximum distance for filtering analysis
                    bkg_fraction : float
                        Background fraction for fitting
                    fit_bkg : bool
                        Whether to fit background (default: False)
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    density : float or list
                        Fitted spatial density value(s) in units^(-d)
                    bkg_fraction : list
                        Background fraction values
                    fp_fig : str or list
                        Filepath(s) to CSR fit visualization figure(s)
                    wasserstein_distances_per_k : list
                        Wasserstein distances for each k-th nearest neighbor
                        order
                    mean_wasserstein_distance : float or list
                        Mean Wasserstein distance across all k orders
                    ks_pvalues_per_k : list
                        Kolmogorov-Smirnov p-values for each k-th NN order

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with CSR fitting results and goodness-of-fit
                metrics
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
        """Aligns multiple channels to each other (part of an aggregation
        workflow)
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    filepaths : list of str
                        the previously saved hdf5 files to be loaded and
                        aligned. if not given, the last processed data is used
                    align_pars : dict
                        kwargs of picasso_outpost.align_channels
                            max_iterations, convergence
                    fp_fiducials : list of str
                        the previously saved hdf5 files of fiducial markers
                        to be loaded and aligned.
                    fig_filename : str
                        the location to save the drift figure to
                    crop_boundaries : bool
                        whether to crop the localizations according to the
                        image boundaries (after shifting)
                    fp_co_shift_channel_locs : list of str
                        hdf5 files not in the 'main workflow' that should
                        be shifted as well. This could e.g. be clustered
                        localizations when the workflow has continued with
                        cluster centers
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Combines multiple channels into one dataset. This is relevant
        e.g. for RESI.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    tag : str
                        the tag / name of the combined dataset
                    combine_col : str
                        the column name for the IDs to the different datasets
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Save data of multiple single-dataset workflows from one
        aggregation workflow.

        Saves all channel localization data and metadata from the aggregated
        workflow to individual files in the results folder.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                The results dictionary, updated with:
                    filepaths : list
                        List of all saved file paths from the aggregated
                        datasets

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Updated results dictionary with saved file paths
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
        """
        This is Susanne's implementation of calculating a cell mask,
        written (ni part?) for the initial version of the DC-Atlas.
        May be obsolete with create_mask2, but kept for backwards
        compatibility. To be deprecated on the long run.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    fp_combined_locs : str
                        filepath to the locs combined in 'combine_channels'
                        module
                    margin : float
                        Size of the added empty margin to the FOV, in nm
                    binsize : float
                        Size o fthe 2D histogram bins of the first step, in nm
                    sigma_mask_blur : int
                        parameter of the gaussian blur in binsize units
                    mask_resolution : float
                        Controls the digital resolution of the mask, in nm
                    combine_col : str
                        the name of the combine column, e.g. 'combine_id'
                        or 'protein'. Same as used in 'combine_channels' module
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """
        This is Rafal's implementation of cell masking, written for the
        3rd version of the DC Atlas. It is (mostly?) identical with an
        implementation of it in spinna, which will be integrated into
        picasso soon. Evaluate deprecation (or moving source from
        outpost_modules/ripleys to picasso/spinna) at that time.

        the locs must be protein positions at this stage.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    binsize : float
                        the bin size in nanometers. A good value is 20
                    blursize : float
                        the gaussian blur to apply in nanometers.
                        A good value is 400
                    mask_pixel_size : float
                        the pixelsize of the final mask, in nanometers.
                        Often used: 10
                    threshold : float
                        the threshold value below which the mask is set
                        to zero. For example 1 / 3
                    binary : boolean
                        whether to create a binary or density mask
                    select_cell : boolean
                        whether to select the largest connected component,
                        assumed to be the cell of interest.
                    fill_holes : boolean
                        whether to fill holes in the cell mask
                    dilate_nm : float
                        the nanometers to dilate the mask (useful if a large
                        threshold has been used)
                    apply_to_locs : boolean
                        whether to drop all localizations outside the area
                and optional keys:
                    fp_combined_locs : str default: None or ''
                        filepath to the locs combined in 'combine_channels'
                        module. If None or '', loaded channel_locs is used
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    combine_col : str
                        the name of the combine column, e.g. 'combine_id'
                        or 'protein'. Same as used in 'combine_channels' module
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """
        This module analyses and refines a previously created mask.
        Particularly, the density histogram of the mask bins are plotted,
        and an area of homogeneous density can be selected

        the locs must be protein positions at this stage.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_mask : str
                        the file path to the mask
                    min_density, max_density : float
                        the density range to select
                and optional keys:
                    nbins : int
                        the number of bins for plotting
                    nth_largest : int
                        select the nth largest area in density range.
                        1-based: set 1 for largest.
                    apply_to_locs : bool
                        whether to apply the created mask to the locs
                    smoothe_nm : float
                        the number of nanometers to dilate and erode
                        the mask. This can be useful to remove excessive
                        holes and ragging in the mask due to the
                        density thresholding
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    epsilon_nm : float
                        dbscan epsilon in nm
                    minpts : int
                        minimum number of points
                    sigma_linker : float
                        ... in nm
                    fp_merge_mask : str
                        filepath to the merge mask (generated in module
                        'create_mask')
                    thresh_type : str
                        ...
                    cell_name : str
                        the name of the cell currently analyzed
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """TO BE CLEANED UP
        simulate CSR within a density mask, and perform dbscan as well
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    fp_mask_dict : str
                        filepath to the mask_dict.pkl file generated in
                        the 'create_mask' module
                    N_repeats : int
                        number of simulation repeats
                    epsilon_nm : float
                        dbscan epsilon in nm
                    minpts : int
                        minimum number of points
                    sigma_linker : float
                        ... in nm
                    fp_merge_mask : str
                        filepath to the merge mask (generated in module
                        'create_mask')
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Plot the interaction graph, displaying the different targets
        and their interactions in a graph. The node sizes denote the
        density, and the ripley interaction matrix is represented in the
        edges.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_workflows : list of str
                        the paths to the folders of separate workflows
                        where the separate ripleys analyses have been done
                    report_names : list of str
                        the report names of those worklfows
                    swkfl_protint_key : str
                        the results key of the protein_interactions module.
                        e.g. '09_protein_interactions'
                    fp_density : str
                        fp to the denfsities of the channels.
                    fp_ripleys_meanvals : str
                        the filepath to the interaction matrix
                    edge_factor : float
                        factor to display useful sizes
                    node_factor : float
                        factor to display useful sizes
                    channel_colors : list of str
                        colors to describe the receptors with
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Find localizations stemming from gold beads based on blinking
        kinetics.
        The metrics used are number of locs and rms deviation from mean
        frame
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    remove_gold : bool
                        if present and set to True, the gold locs
                        are discarded and self.locs is set to the
                        nongold-locs
                    diameter : float
                        the pick similar diameter for identifying gold
                    std_range, mean_rmsd : float
                        the pick similar parameters identifying gold
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """pick similar in nlocs/rmsd space (with specified limits in
        that space).
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    diameter : float
                        the pick similar diameter for identifying gold
                and optional keys:
                    min_n_locs_per_frame : float, range 0-1
                        the min percentage of frames with events in the pick
                        region to pick. default: 0.01
                    max_n_locs_per_frame : float, range 0-1
                        the max percentage of frames with events in the pick
                        region to pick. default: 0.01
                    min_rmsd : float
                        the minimum root mean square distance from pick center
                        to pick
                    max_rmsd : float
                        the maximum root mean square distance from pick center
                        to pick
                    n_plot_structures : int
                        the number of structures to plot
                    display_pixelsize : float
                        the pixelsize for display in nm, default: 1
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """pick similar on clusters in nlocs/rmsd space.
        This may be useful for automated picking of origamis, and may
        help for defining parameters for finding gold
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    diameter : float
                        the pick similar diameter for identifying gold
                and optional keys:
                    min_n_locs_per_frame : float
                        the percentage of frames with events in the pick
                        region below which there is noise. default: 0.01
                    n_plot_structures : int
                        the number of structures to plot
                    display_pixelsize : float
                        the pixelsize for display in nm, default: 1
                    xi : float
                        the xi parameter for clustering. default 0.05
                    min_cluster_size : float
                        the minimun cluster size (fract). default .05
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Performs undrift from piced locs.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_picked_locs : str
                        filepath to the picked locs to undrift from
                        (.hdf5 file of list of locs, with 'group' column
                         to describe picks)
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Filter localizations to lie within a min-max range of a metric.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    field : str or list of str
                        the field(s) to filter on
                and optional keys:
                    minval : dtype of field (or list of it)
                        the minimum value(s) to accept
                    maxval : dtype of field (or list of it)
                        the maximum value(s) to accept
                    mode : str
                        the mode of threshold application:
                         - absolute: minval and maxval are values
                            in units of the field
                         - zscore: minval and maxval are in units of
                            standard deviations from the mean
                            (-2, 2 means cut off at 2*std from mean)
                         - quantile: minval and maxval are quantiles
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Filter molecule positions (after clustering or Gaussian Mixture)
        for those who show transient binding. Specifically, the mean frame
        should not be at extreme positions
        (default, 0.1 > mean frame / nframes > 0.9), and std of frames
        (default: 0.3 > std frame).
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    meanframe_cutoff : float (0-1, default .1)
                        filter out positions at more extreme temporal positions
                    stdframe_cutoff : float
                        filter out positions with lower std than .16
                    fp_locs : str
                        the filepath to the underlying localizations
                        (self.locs are centers). If given, these are filtered
                        as well and saved with the same filename in the current
                        results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Link localizations.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    d_max : int
                        maximum distance to link [px]
                    tolerance : int
                        maximum transient dark time [frames]
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Calls another module (as a sub-module) for all pairs in the
        channel_locs
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    module_name : str
                        the module to call
                    param_target1 : str
                        parameter name of the first target to set for the
                        module
                    param_target2 : str
                        parameter name of the second target to set for the
                        module
                    module_kwargs : dict
                        the other arguments to the module
                and optional keys:
                    result_scalar : str
                        the key to display in a heatmap as main result
                    scalar_threshold : float
                        the saturation value in the heatmap
                    scalar_minval : float
                        the minimum value for color in the heatmap
                    result_fpfig : str or list of str
                        the key to the filepath of one or more figures
                        generated to display for documentation
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        """Generate random values and plot for debugging and testing the
        pairwise module.

        Creates a random value and generates a test plot with random data
        for debugging purposes in pairwise module workflows.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    xlabel : str
                        Label for the x-axis of the test plot
                    ylabel : str
                        Label for the y-axis of the test plot
                Optional keys: (none)
            results : dict
                The results dictionary, updated with:
                    random_val : float
                        A random value between 0 and 1
                    fp_fig : str
                        Filepath to the generated test figure

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Updated results dictionary with random value and figure path
        """
        pass

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
        """Analyse for labeling efficiency.
        Perform 3 component SPINNA analysis for monomers and heterodimers
        of target (A) and reference (B). For the analysis, we enter a
        labeling efficiency of 1, yielding proportions of monomers and
        dimers as seen in the data. The real labeling efficiency is then

        Model:
        Binders A and B bind to an engineered construct A*-anchor-B*.
            A <-> A*-anchor-B* <-> B
        There are four possible configurations:
            A_only: AA*-anchor-B*
            AB: AA*-anchor-B*B
            B_only: A*-anchor-B*B
            None (invisible in data): A*-anchor-B*
        Number of total constructs with A, or B, respectively:
            #A_tot = #A_only + #AB
            #B_tot = #B_only + #AB

        Proportions can be given in terms of #structures, or in terms
        of #molecules, e.g.
        with proportions given in terms of #structures
         10 monomers, 10 dimers (20molecules in dimers) -> p_m = 50%, p_d=50%

        with proportions given in terms of #molecules
         10 monomers, 10 dimers (20molecules in dimers) -> p_m = 33%, p_d=66%

        in terms of #structures
        prop_A^S = #A_only / (#A_only + #B_only + #AB)
        prop_B^S = #B_only / (#A_only + #B_only + #AB)
        prop_AB^S = #AB / (#A_only + #B_only + #AB)
        in terms of #molecules
        prop_A^S = #A_only / (#A_only + #B_only + 2 #AB)
        prop_B^S = #B_only / (#A_only + #B_only + 2 #AB)
        prop_AB^S = 2 #AB / (#A_only + #B_only + 2 #AB)

        #AB = #anchor * LE_A * LE_B
        #A_tot = #anchor * LE_A
        #B_tot = #anchor * LE_B
        #A_only = #A_tot - #AB = #anchor * LE_A * (1 - LE_B)
        #B_only = #B_tot - #AB = #anchor * LE_B * (1 - LE_A)

        THUS, finally, the labeling efficiency can be calculated by

        with proportions given in terms of #structures
        LE_A = prop(AB) / (prop(B) + prop(AB))
        LE_B = prop(AB) / (prop(A) + prop(AB))

        with proportions given in terms of #molecules
        LE_A = prop(AB) / (2 * prop(B) + prop(AB))
        LE_B = prop(AB) / (2 * prop(A) + prop(AB))

        SPINNA outputs propportions in terms of #molecules, so the last
        formulae are used below.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    reference_name : str
                        the channgel_tag of the reference
                    target_name : str
                        the channel_tag of the target queried for LE
                    pair_distance: 10 # real distance of pair of tags in nm
                    labeling_uncertainty : dict, channel tag to float
                        labeling uncertainty [nm]; good value is e.g. 5
                    n_simulate : int
                        number of target molecules to be simulated;
                        good value is e.g. 50000
                    density : dict, channel tag to float
                        density to simulate [nm^2 or nm^3];
                        area density if 2D; volume density if 3D
                    granularity : float
                        the spinna res_factor
                    sim_repeats : int
                        number of simulation repeats, for noise reduction
                and optional keys:
                    nn_nth : int
                        number of nearest neighbors to analyse
                        default: 1
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
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
        le_std = ""
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
    pass


def confluence_call(method):
    """When calling the confluence API, sometimes the connection is lost.
    Therefore, wrap the functions using this decorator in order to
    re-connect and call again in that case.
    """

    def confluence_call_wrapper(self, *args, **kwargs):
        try:
            # call the confluence api
            status = method(self, *args, **kwargs)
        except ConnectionError as e:
            logger.exception(e)
            self.connect()
            status = method(self, *args, **kwargs)
        except HTTPError as e:
            if "unauthorized" in str(e).lower():
                raise e
            raise ConfluenceInterfaceError(
                f"Calling {method.__name__} failed with HTTPError: {str(e)}"
            )

        return status

    return confluence_call_wrapper


class ConfluenceInterface:
    """A Interface class to access Confluence

    For access to the Confluence API, create an API token in confluence,
    and store it as an environment variable:
    $ setx CONFLUENCE_BEARER "your_confluence_api_token"
    If the token is not stored as an environment variable, specify it
    here at initialization.

    Args:
        base_url : str
            the confluence url to connect to.
        space_key : str
            the confluence space key to work in
        parent_page_title : str
            the (already existing) parent page to crate the reports under
        username : str, default None
            the username to authenticate with.
            If given, authentiation is performed with url, username and
            password.
            If None or "", authentication is performed with url and token.
            For Confluence Server, token-based authentication is needed.
        token : str, default None
            the password (if username-based authentication), or token.
            If None, the environment variable CONFLUENCE_BEARER is polled.
    """

    def __init__(
        self, base_url, space_key, parent_page_title, username=None, token=None
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
        logger.debug(f"confluence_token: {self.bearer_token}")
        logger.debug(f"confluence_username: {self.username}")

        self.connect()

        self.parent_page_id, _ = self.get_page_properties(parent_page_title)

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
        """Set this by setting the environment variable in the windows command
        line on the server:
        $ setx CONFLUENCE_BEARER <your_confluence_api_token>
        The confluence api token can be generated and copied in the personal
        details of confluence.
        """
        return os.environ.get("CONFLUENCE_TOKEN")

    @confluence_call
    def get_page_properties(self, page_title="", page_id=""):
        """
        Returns:
            id : str
                the page id
            title : str
                the page title
        """
        if page_title != "":
            page = self.confluence.get_page_by_title(
                space=self.space_key, title=page_title
            )
        elif page_id != "":
            page = self.confluence.get_page_by_id(page_id=page_id)
        else:
            logger.error("One of page_title and page_id must be given.")
            raise ConfluenceInterfaceError(
                "Cannot get page properties. "
                + "One of page_title and page_id must be given."
            )

        return page["id"], page["title"]

    @confluence_call
    def get_page_version(self, page_title="", page_id=""):
        """
        Returns:
            data : dict
                results
                    id, title, version
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
        """
        Returns:
            data : dict
                results
                    id, title, version
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
        """
        Args:
            page_title : str
                the title of the page to be created
            body_text : str
                the content of the page, with the confuence markdown / html
            parent_id : str
                the id of the parent page. If 'rootparent', the parent_page_id
                of this ConfluenceInterface is used
        Returns:
            page_id : str
                the id of the newly created page
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
        """Uploads an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            filename : str
                the local filename of the file to attach
        Returns:
            attachment_id : str
                the id of the attachment
        """
        self.confluence.attach_file(
            filename=filename, page_id=page_id, space=self.space_key
        )

        attachments_container = self.confluence.get_attachments_from_content(
            page_id=page_id, start=0, limit=500
        )
        for attachment in attachments_container["results"]:
            attachment_id = attachment["id"]
            break

        return attachment_id

    @confluence_call
    def get_attachment_id(self, page_id, filename):
        """Get the id of an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            filename : str
                the local filename of the file to retreive
        Returns:
            attachment_id : str
                the id of the attachment
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
        """Deletes an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            attachment_id : str
                the id of the attachment
        Returns:
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
    pass

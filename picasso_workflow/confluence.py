#!/usr/bin/env python
"""
Module Name: confluence.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Interaction with Confluence
"""
import logging
import traceback
import os
import numpy as np
import pandas as pd
from atlassian import Confluence as con
from requests.exceptions import ConnectionError, HTTPError
import html

from picasso_workflow.util import AbstractModuleCollection


logger = logging.getLogger(__name__)


def module_decorator(method):
    def module_wrapper(self, i, parameters, results):
        # create parameter and results documentation
        parameter_text = """
            <ac:structured-macro ac:name="expand" ac:schema-version="1">
            <ac:parameter ac:name="title">Parameters</ac:parameter>
            <ac:rich-text-body>
            <ul>
            """
        for k, v in parameters.items():
            parameter_text += f"<li>{k}: {v}</li>"

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
            result_text += f"<li>{k}: {v}</li>"

        result_text += """
        </ul>
        </ac:rich-text-body>
        </ac:structured-macro>
        """

        # call the module
        method(self, i, parameters, results, parameter_text, result_text)

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

    def dummy_module(self, i, parameters, results):
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
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds

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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def analysis_documentation(self, i, parameters, results):
        """This module documents where and how analysis is being performed"""
        logger.debug("Reporting analysis_documentation.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Analysis Hard- and Software</strong></p>
        <ul>
        """
        for k, v in results.items():
            text += f"<li>{k}: {v}</li>"
        text += """
        </ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    ##########################################################################
    # Single dataset modules
    ##########################################################################

    def convert_zeiss_movie(self, i, parameters, results):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.

        Converts Zeiss .czi movie files to picasso-compatible .raw format
        for subsequent analysis steps in the workflow.
        Generates Confluence documentation for the conversion process.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filepath : str
                        Path to the input Zeiss .czi file
                Optional keys:
                    output_filepath : str
                        Custom output path for the .raw file
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    filepath_raw : str
                        Path to the output .raw file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with added file path information
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def load_dataset_movie(self, i, pars_load, results_load):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads DNA-PAINT movie data and metadata into memory for subsequent
        analysis. Optionally creates sample movies and loads camera configuration.
        Generates Confluence documentation for movie loading process.

        Args:
            i : int
                The index of the module in the workflow
            pars_load : dict
                Required keys:
                    filename : str
                        Path to the movie file to load
                Optional keys:
                    sample_movie : dict
                        Parameters for creating a subsampled movie
                    load_camera_info : bool
                        Whether to load camera configuration from picasso.CONFIG
            results_load : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
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
            pars_load : dict
                Input parameters, potentially modified (sample_movie paths updated)
            results_load : dict
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )
        if (sample_mov_res := results_load.get("sample_movie")) is not None:
            text = f"""
            <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
            <p>Subsampled Frames</p>
            <ul>
            <li> {len(sample_mov_res['sample_frame_idx'])} frames:
             {str(sample_mov_res['sample_frame_idx'])}</li>
            </ul>
            </ac:layout-cell></ac:layout-section></ac:layout>
            """
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )
            logger.debug("Uploading movie of subsampled images.")
            self.ci.upload_attachment(
                self.report_page_id, sample_mov_res["filename"]
            )
            self.ci.update_page_content_with_movie_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(sample_mov_res["filename"])[1],
            )

    def load_dataset_localizations(self, i, parameters, results):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads pre-computed localization data from file for analysis workflows
        that skip the identification and localization steps.
        Generates Confluence documentation for loading localization data.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filename : str
                        Path to the localization file to load
                Optional keys:
                    additional_info : dict
                        Additional metadata to include
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    picasso version : str
                        Version of picasso library used
                    nlocs : int
                        Number of localizations loaded

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with added localization information
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

    def identify(self, i, parameters, results):
        """Identifies localizations in a loaded dataset.

        Identifies potential localization sites in the loaded movie using
        net gradient thresholding. Optionally performs automatic net gradient
        detection and creates identification vs frame plots.
        Generates Confluence documentation for the identification process.

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
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for generated files
                Results updated with:
                    num_identifications : int
                        Total number of identifications found
                    auto_netgrad : dict
                        Results from automatic net gradient detection (if requested)
                    ids_vs_frame : dict
                        Results from identifications vs frame analysis (if requested)

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
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )
        if (res_autonetgrad := results.get("auto_netgrad")) is not None:
            logger.debug("Uploading graph for auto_netgrad.")
            self.ci.upload_attachment(
                self.report_page_id, res_autonetgrad["filename"]
            )
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res_autonetgrad["filename"])[1],
            )
        if (res := results.get("ids_vs_frame")) is not None:
            logger.debug("uploading graph for identifications vs frame.")
            self.ci.upload_attachment(self.report_page_id, res["filename"])
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(res["filename"])[1],
            )

    def localize(self, i, parameters, results):
        """Describes the Localize section of picasso
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
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

    def export_brightfield(self, i, parameters, results):
        """Describes the export_brightfield section of picasso
        Args:
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

        for label, fp in results.get("labeled filepaths", {}).items():
            text = f"""<p><strong>{label}</strong></p>"""
            self.ci.update_page_content(
                self.report_page_name, self.report_page_id, text
            )
            self.ci.upload_attachment(self.report_page_id, fp)
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(fp)[1],
            )

    @module_decorator
    def render(self, i, parameters, results, parameter_text, result_text):
        """Renders localizations
        Args:
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
        fig_fps = []
        titles = []
        if fp_fig := results.get("fp_scene_fullfov"):
            fig_fps.append(fp_fig)
            titles.append("Localizations in whole Field of View")
        if fp_fig := results.get("fp_scene_ctrmass"):
            fig_fps.append(fp_fig)
            titles.append("Localizations in Zoom-In")

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

    def undrift_rcc(self, i, parameters, results):
        """Describes the Localize section of picasso
        Args:
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """Report RSSO undrifting results to Confluence"""
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

        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: Undrifting via RSSO
        (2-stage with confidence analysis)</strong></p>
        Summary:
        <ul>
        <li>t<sub>on</sub>: {parameters.get('ton')} frames</li>
        <li>t<sub>off</sub>: {parameters.get('toff')} frames</li>
        <li>Max shift per frame: {parameters.get('max_shift')} pixels</li>
        <li>Processing chunk size:
            {parameters.get('processing_chunk_size', 100)} frames</li>
        <li>Min locs per frame: {parameters.get('min_locs_per_frame', 10)}</li>
        <li>Min locs per block:
            {parameters.get('min_locs_per_block', 100)}</li>
        </ul>

        <p><strong>Drift Results:</strong></p>
        <ul>
        <li><strong>Coarse drift (t<sub>off</sub>-scale):</strong>
        X={coarse_drift_x:.2f} nm, Y={coarse_drift_y:.2f} nm</li>
        <li><strong>Fine drift (frame-scale):</strong>
        X={fine_drift_x:.2f} nm, Y={fine_drift_y:.2f} nm</li>
        <li><strong>Total drift:</strong>
        X={drift_mag_x:.2f} nm, Y={drift_mag_y:.2f} nm</li>
        <li><strong>Total drift magnitude:</strong> {total_drift:.2f} nm</li>
        <li>Mean drift quality: {drift_quality:.1f} measurements/frame</li>
        </ul>

        <p><strong>Confidence Analysis:</strong></p>
        <ul>
        <li><strong>Mean uncertainty:</strong>
        X={mean_uncertainty_x:.2f} nm, Y={mean_uncertainty_y:.2f} nm</li>
        <li><strong>Max uncertainty:</strong>
        X={max_uncertainty_x:.2f} nm, Y={max_uncertainty_y:.2f} nm</li>
        <li><strong>95% confidence interval:</strong>
        ±{confidence_95_x:.2f} nm (X), ±{confidence_95_y:.2f} nm (Y)</li>
        <li><strong>Relative precision:</strong>
        {(mean_uncertainty_x/drift_mag_x*100 if drift_mag_x > 0 else 0):.1f}%
        (X),
        {(mean_uncertainty_y/drift_mag_y*100 if drift_mag_y > 0 else 0):.1f}%
        (Y)</li>
        </ul>

        <ul>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.2f} s</li>
        </ul>
        {parameter_text}
        {result_text}
        """

        # Add drift plot if available
        if fp_fig := results.get("fp_fig"):
            try:
                self.ci.upload_attachment(self.report_page_id, fp_fig)
            except ConfluenceInterfaceError:
                pass
            _, fp_fig_name = os.path.split(fp_fig)
            text += f"""
            <p><strong>Drift Trajectory:</strong></p>
            <ac:image ac:align="center" ac:layout="center">
            <ri:attachment ri:filename="{fp_fig_name}" />
            </ac:image>
            """

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """

        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def undrift_aim(self, i, parameters, results, parameter_text, result_text):
        """Describes the AIM undrifting
        Args:
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def manual(self, i, parameters, results):
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
        self, i, parameters, results, parameter_text, result_text
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

    def density(self, i, parameters, results):
        """Document local density computation analysis.

        Generates Confluence documentation for local localization density
        calculation, including computation parameters and timing information.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        Radius for local density calculation in nm
                Optional keys:
                    save_locs : bool
                        Whether to save density-annotated localizations
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                    nlocs : int
                        Number of localizations processed
                Optional keys:
                    density_stats : dict
                        Statistical summary of density calculations

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results (unchanged)
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def dbscan(self, i, parameters, results, parameter_text, result_text):
        """Perform clustering using dbscan.

        Applies DBSCAN clustering algorithm to localizations, optionally
        replacing localizations with cluster centers for subsequent analysis.
        Generates Confluence documentation for the DBSCAN clustering analysis.

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
                        Whether to save clustered localization data to results folder
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for generated files
                Results updated with:
                    fp_fig_clustersizes : str
                        Filepath to cluster size distribution figure
                    fp_centers : str
                        Filepath to cluster centers file
            parameter_text : str
                Pre-formatted parameter documentation text
            result_text : str
                Pre-formatted results documentation text

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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def hdbscan(self, i, parameters, results):
        """Perform clustering using hdbscan.

        Applies HDBSCAN (Hierarchical DBSCAN) clustering algorithm to localizations.
        Generates Confluence documentation for HDBSCAN clustering analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    min_cluster : int
                        Minimum cluster size for HDBSCAN
                    min_sample : int
                        Minimum samples parameter for HDBSCAN
                Optional keys:
                    continue_with_centers : bool
                        Whether to use cluster centers for subsequent analysis
                    save_locs : bool
                        Whether to save clustered localization data
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Optional keys:
                    n_clusters : int
                        Number of clusters identified
                    cluster_centers : numpy.ndarray
                        Coordinates of cluster centers

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results (unchanged)
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def binding_event_analysis(
        self, i, parameters, results, parameter_text, result_text
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """Report resolution analysis results to Confluence"""

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
    def smlm_clusterer(
        self, i, parameters, results, parameter_text, result_text
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
        self, i, parameters, results, parameter_text, result_text
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
    def nneighbor(self, i, parameters, results, parameter_text, result_text):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def fit_csr(self, i, parameters, results, parameter_text, result_text):
        """Fit a Completely Spatially Random Distribution to nearest neighbors.

        Fits CSR model to nearest neighbor distance distributions and evaluates
        goodness-of-fit using statistical measures and visualization.
        Generates Confluence documentation for CSR distribution fitting analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    nneighbors : str or numpy.ndarray or list
                        If str: filepath to nearest neighbor data file
                        If array: 2D array (N, k) of kth nearest neighbor distances
                        If list: multiple datasets or file paths
                    dimensionality : int
                        Spatial dimensionality (2 or 3) for CSR model
                Optional keys:
                    kmin : int
                        Minimum k-th nearest neighbor order to fit (default: 1)
                    min_dist : float
                        Minimum observable distance in nm due to technical limits
                    max_dist : float
                        Maximum distance for filtering analysis
                    bkg_fraction : float
                        Background fraction for fitting
                    fit_bkg : bool
                        Whether to fit background (default: False)
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
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
                        Wasserstein distances for each k-th nearest neighbor order
                    mean_wasserstein_distance : float or list
                        Mean Wasserstein distance across all k orders
                    ks_pvalues_per_k : list
                        Kolmogorov-Smirnov p-values for each k-th NN order
            parameter_text : str
                Pre-formatted parameter documentation text
            result_text : str
                Pre-formatted results documentation text

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with CSR fitting results and goodness-of-fit metrics
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def save_single_dataset(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    def load_datasets_to_aggregate(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def align_channels(
        self, i, parameters, results, parameter_text, result_text
    ):
        """Describes the align_channels module
        Args:
            parameters : dict
                filenames : the net gradient used
            results : dict
                required:
                    shifts
                optional:
                    fig_filepath
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

    def combine_channels(self, i, parameters, results):
        """Describes the combine_channels module
        Args:
            parameters : dict
                filenames : the net gradient used
            results : dict
                required:
                optional:
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def save_datasets_aggregated(self, i, parameters, results):
        """save data of multiple single-dataset workflows from one
        aggregation workflow."""
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def spinna_manual(self, i, parameters, results):
        """ """
        logger.debug("Reporting spinna_manual.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Module {i:02d}: SPINNA-Manual</strong></p>
        <ul><li>file present: {results.get('success')}</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        """
        if not results["success"]:
            text += "<li>" + results["message"] + "</li>"
        else:
            text += f"<li>Result folder: {results['result_dir']}</li>"
            summary = pd.read_csv(results["fp_summary"])
            for i, row in summary.iterrows():
                text += f"<p><strong> Row {i} </strong></p><ul>"
                for col, val in row.items():
                    text += f"<li>{col}: {str(val)}</li>"
                text += "</ul>"
        text += """</ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )
        if results["success"]:
            for fp in results["fp_fig"]:
                self.ci.upload_attachment(self.report_page_id, fp)
                self.ci.update_page_content_with_image_attachment(
                    self.report_page_name,
                    self.report_page_id,
                    os.path.split(fp)[1],
                )

    @module_decorator
    def spinna(self, i, parameters, results, parameter_text, result_text):
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
        titles = ["" for i in range(len(fig_fps))]

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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def ripleysk(self, i, parameters, results):
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
    def ripleysk2(self, i, parameters, results, parameter_text, result_text):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def ripleysk_average(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def ripleysk_average2(
        self, i, parameters, results, parameter_text, result_text
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

    def protein_interactions(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def protein_interactions_average(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def create_mask(self, i, parameters, results):
        """Create a density mask"""
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def create_mask2(
        self, i, parameters, results, parameter_text, result_text
    ):
        """Create a density mask"""
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """Create a density mask"""
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

    def dbscan_molint(self, i, parameters, results):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def CSR_sim_in_mask(self, i, parameters, results):
        """TO BE CLEANED UP
        simulate CSR within a density mask
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def dbscan_merge_cells(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def dbscan_merge_stimulations(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def binary_barcodes(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def plot_densities(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def find_cluster_motifs(self, i, parameters, results):
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def interaction_graph(self, i, parameters, results):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def find_gold(self, i, parameters, results, parameter_text, result_text):
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """pick similar on clusters in nlocs/rmsd space
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """pick similar on clusters in nlocs/rmsd space
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """Performs undrift from piced locs.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def filter_locs(self, i, parameters, results, parameter_text, result_text):
        """Filter localizations to lie within a min-max range of a metric.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
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
            text += "<ul><table>"
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
            text += "</td></tr></table></ul>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def filter_transient_binding(
        self, i, parameters, results, parameter_text, result_text
    ):
        """Filter molecule positions (after clustering or Gaussian Mixture)
        for those who show transient binding. Specifically, the mean frame
        should not be at extreme positions
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
            text += "<ul><table>"
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
            text += "</td></tr></table></ul>"

        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def link_locs(self, i, parameters, results):
        """Link localizations.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def insert_image(self, fp_fig):
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
        self, i, parameters, results, parameter_text, result_text
    ):
        """Calls another module (as a sub-module) for all pairs in the
        channel_locs
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
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    @module_decorator
    def random_val(self, i, parameters, results, parameter_text, result_text):
        """This is just for debugging"""
        pass

    @module_decorator
    def labeling_efficiency_analysis(
        self, i, parameters, results, parameter_text, result_text
    ):
        """Analyse for labeling efficiency.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
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
        <li>Labeling efficiency std:
            {show_dict_percentages(results["labeling_efficiency_std"])}</li>
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
            f'<ac:image><ri:attachment ri:filename="{filename}" />'
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

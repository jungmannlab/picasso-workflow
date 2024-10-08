#!/usr/bin/env python
"""
Module Name: confluence.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Interaction with Confluence
"""
import logging
import os
import requests
import pandas as pd
from atlassian import Confluence as con

from picasso_workflow.util import AbstractModuleCollection


logger = logging.getLogger(__name__)


class ConfluenceReporter(AbstractModuleCollection):
    """A class to upload reports of automated picasso evaluations
    to confluence
    """

    def __init__(
        self, base_url, username, space_key, parent_page_title, report_name, token
    ):
        logger.debug("Initializing ConfluenceReporter.")
        
        self.ci = ConfluenceInterface(
            base_url, username, space_key, parent_page_title, token
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

    def dummy_module(self, i, parameters, results):
        """A module that does nothing, for quickly removing
        modules in a workflow without having to renumber the
        following result idcs. Only for workflow debugging,
        remove when done.
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
        text = """
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Analysis Hard- and Software</strong></p>
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
        """Descries converting from Zeiss."""
        logger.debug("Reporting convert_zeiss_movie.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Converting Movie from .czi into .raw</strong></p>
        <p>Converted the file {parameters["filepath"]} to
        {results["filepath_raw"]} in {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s.</p>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def load_dataset_movie(self, i, pars_load, results_load):
        """Describes the loading
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
        """
        logger.debug("Reporting a loaded dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Load Movie</strong></p>
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
        """Describes the loading
        Args:
            i : int
            parameters : dict
            results : dict
        """
        logger.debug("Reporting a loaded dataset.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Load localizations</strong></p>
        <ul>
        <li>Picasso Version: {results['picasso version']}</li>
        <li>Movie Location: {parameters['filename']}</li>
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
        """Describes the identify step
        Args:
            localize_params : dict
                net_gradient : the net gradient used
                frames : the number of frames
            fn_movie : str
                the filename to the movie generated
            fn_hist : str
                the filename to the histogram plot generated
        """
        logger.debug("Reporting Identification.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Identify</strong></p>
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
        <p><strong>Localize</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Locs Column names: {results['locs_columns']}</li></ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        # text = "<p><strong>Localize</strong></p>"
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
        text = """
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Exporting Brightfield</strong></p>
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

    def undrift_rcc(self, i, parameters, results):
        """Describes the Localize section of picasso
        Args:
        """
        logger.debug("Reporting undrifting via RCC.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Undrifting via RCC</strong></p>
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

    def undrift_aim(self, i, parameters, results):
        """Describes the AIM undrifting
        Args:
        """
        logger.debug("Reporting undrift_aim.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Undrifting via AIM</strong></p>
        <ul><li>Dimensions: {parameters.get('dimensions')}</li>
        <li>Segmentation: {parameters.get('segmentation')} frames</li>
        <li>Intersect distance: {parameters.get('intersect_d')} nm</li>
        <li>Local search region radius: {parameters.get('roi_r')} nm</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li></ul>
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
        <p><strong>Manual step</strong></p>
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

    def summarize_dataset(self, i, parameters, results):
        logger.debug("Reporting dataset description.")
        text = """
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Descriptive Statistics</strong></p>"""
        for meth, meth_pars in parameters["methods"].items():
            if meth.lower() == "nena":
                meth_res = results["nena"]
                text += f"""
                    <p>NeNa</p>
                    <ul>
                    <li>NeNa value: {str(meth_res.get('NeNa'))}</li>
                    <li>Best Fit Values: {str(meth_res.get('res'))}</li>
                    <li>Chi Square: {str(meth_res.get('chisqr'))}</li>
                    </ul>"""
                if fp_nena := meth_res.get("filepath_plot"):
                    self.ci.upload_attachment(self.report_page_id, fp_nena)
                    _, fn_nena = os.path.split(fp_nena)
                    text += (
                        "<ul><ac:image><ri:attachment "
                        + f'ri:filename="{fn_nena}" />'
                        + "</ac:image></ul>"
                    )
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        logger.debug("description text: " + text)
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
        logger.debug("Reporting density.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Local density computation</strong></p>
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

    def dbscan(self, i, parameters, results):
        logger.debug("Reporting dbscan.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>dbscan clustering</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <ul><li>Duration: {results['duration']} s</li>
        <li>Radius: {parameters.get('radius')}</li>
        <li>min_density: {parameters.get('min_density')}</li>
        </ul>"""

        text += """
        <b>TODO: generate plot for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def hdbscan(self, i, parameters, results):
        logger.debug("Reporting hdbscan.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>dbscan clustering</strong></p>
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

    def smlm_clusterer(self, i, parameters, results):
        logger.debug("Reporting smlm_clusterer.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>smlm_clusterer clustering</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>radius: {parameters.get('radius')}</li>
        <li>min_locs: {parameters.get('min_locs')}</li>
        <li>basic_fa: {parameters.get('basic_fa')}</li>
        <li>radius_z: {parameters.get('radius_z')}</li>
        </ul>"""

        text += """
        <b>TODO: generate plot for reporting</b>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def nneighbor(self, i, parameters, results):
        logger.debug("Reporting nneighbor.")
        d = len(parameters["dims"])
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Nearest Neighbor analysis</strong></p>
        Radial Distribution Function (RDF) and Nearest Neighbor Distributions.
        The RDF shows the density of spots in an annulus of a given radius
        r and thickness delta r, averaged over all spots. If the RDF deviates
        from the overall density, it means there is structure at that
        lengthscale in the data. E.g. the RDF is low at small distances due to
        finite resoltion.
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Dimensions taken into account: {parameters['dims']}</li>
        <li>Bin size is the median of the first NN, divided by:
        {parameters['subsample_1stNN']}</li>
        <li>Displayed NN up to nearest neighbor #: {parameters['nth_NN']}</li>
        <li>Displayed RDF up to nearest neighbor #: {parameters['nth_rdf']}
        </li>
        <li>Saved numpy txt file as: {results["nneighbors"]}</li>
        <li>Density from RDF: {results['density_rdf'] * 1e3**d:.02f} µm^{d}
        </li>
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

    def fit_csr(self, i, parameters, results):
        logger.debug("Reporting fit_csr.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Completely Spatially Random Distribution Fit</strong></p>
        The distance distributions of the first N neighbors in the data are
        fitted to the analytical CSR distributions simultaneously, using a
        maximum likelihood esitmator.
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Dimensionality of analytical CSR:
         {parameters['dimensionality']}</li>"""
        if isinstance(parameters["nneighbors"], str):
            text += f"""<li>Experimental Nearest neighbor distances loaded
             from: {parameters['nneighbors']}</li>"""
        else:
            text += f"""<li>Experimental Nearest neighbor distances:
             {parameters['nneighbors'].shape[0]} spots,
              {parameters['nneighbors'].shape[1]} neighbors</li>"""
        text += f"""<li>Density fitted:
         {results['density']} nm^(-{parameters['dimensionality']})</li>
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

    def save_single_dataset(self, i, parameters, results):
        logger.debug("Reporting dataset saving.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Saving Resulting Dataset</strong></p>
        <ul><li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>filepath: {results.get('filepath')}</li>
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
        <p><strong>Loading Datasets to aggregate</strong></p>
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

    def align_channels(self, i, parameters, results):
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
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Align Channels via RCC</strong></p>
        <ul><li>Shifts in x [px]: {results.get('shifts')[0, :]}</li>
        <li>Shifts in y [px]: {results.get('shifts')[1, :]}</li>
        <li>Shifts in z [px]: {results.get('shifts')[2, :]}</li>
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

        if driftimg_fn := results.get("fig_filepath"):
            self.ci.upload_attachment(self.report_page_id, driftimg_fn)
            self.ci.update_page_content_with_image_attachment(
                self.report_page_name,
                self.report_page_id,
                os.path.split(driftimg_fn)[1],
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
        <p><strong>Combine Channels</strong></p>
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
        <p><strong>Saving Datasets aggregated</strong></p>
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
        <p><strong>SPINNA-Manual</strong></p>
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

    def spinna(self, i, parameters, results):
        """ """
        logger.debug("Reporting spinna_manual.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>SPINNA-Manual</strong></p>
        <ul><li>file present: {results.get('success')}</li>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Labeling Efficiency: {parameters["labeling_efficiency"]} %</li>
        <li>Labeling Uncertainty: {parameters["labeling_uncertainty"]} nm</li>
        <li># simulated structures: {parameters["n_simulate"]}</li>
        <li>Nearest Neighbors to evaluate: {parameters["n_nearest_neighbors"]}
        </li>
        <li>Using Mask: {parameters.get("fp_mask_dict") is not None}</li>
        <li>Density: {parameters["density"]} [1/nm^d]</li>
        <li>Random Rotation Mode: {parameters["random_rot_mode"]}</li>
        <li># simulation repeats: {parameters["sim_repeats"]}</li>
        <li>Histogram Bin Size: {parameters["fit_NND_bin"]}</li>
        <li>Histogram Max value: {parameters["fit_NND_maxdist"]}</li>
        """
        if fp_figs := results.get("fp_figs"):
            for fp_fig in fp_figs:
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
        text += """</ul>
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def ripleysk(self, i, parameters, results):
        logger.debug("Reporting ripleysk.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Ripley's K Analysis</strong></p>
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

    def ripleysk_average(self, i, parameters, results):
        logger.debug("Reporting ripleysk_average.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Averaging of Repley's K Integrals</strong></p>
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

    def protein_interactions(self, i, parameters, results):
        logger.debug("protein_interactions.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Direct Protein Interaction Analysis</strong></p>
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
        <p><strong>Direct Protein Interaction Analysis Average</strong></p>
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
        <p><strong>Create Density Mask</strong></p>
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

    def dbscan_molint(self, i, parameters, results):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow
        """
        logger.debug("Reporting dbscan_molint.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>DBSCAN - Molecular Interaction version</strong></p>
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
        <p><strong>CSR simulation in density mask</strong></p>
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
        <p><strong>Merge DBSCAN results over multiple cells</strong></p>
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
        <p><strong>Merge DBSCAN results over multiple stimulations</strong></p>
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
        <p><strong>Analyse and plot binary barcodes</strong></p>
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
        <p><strong>Show Densities</strong></p>
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
        <p><strong>Analyse and plot Cluster Motifs</strong></p>
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
        <p><strong>Interaction Graph</strong></p>
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

    def find_gold(self, i, parameters, results):
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
        <p><strong>Find Gold Beads</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li># Gold Beads found: {results["n_gold"]}</li>
        <li># Gold Bead locs saved at: {results["fp_gold"]}</li>
        <li># Non-gold Bead locs saved at: {results["fp_nogold"]}</li>
        </ul>"""
        text += """
        </ac:layout-cell></ac:layout-section></ac:layout>
        """
        self.ci.update_page_content(
            self.report_page_name, self.report_page_id, text
        )

    def undrift_from_picked(self, i, parameters, results):
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
        <p><strong>Undrift from picked</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li># based on piced locs at: {parameters["fp_picked_locs"]}</li>
        <li>saved undrifted locs to: {results["fp_locs"]}</li>
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

    def filter_locs(self, i, parameters, results):
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
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Filter localizations</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Field to filter on: {parameters["field"]}</li>
        <li>Range to accept (inclusive):
        {parameters["minval"]} - {parameters["maxval"]}</li>
        </ul>"""
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
        <p><strong>Link localizations</strong></p>
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

    def labeling_efficiency_analysis(self, i, parameters, results):
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
        logger.debug("Reporting labeling_efficiency_analysis.")
        text = f"""
        <ac:layout><ac:layout-section ac:type="single"><ac:layout-cell>
        <p><strong>Labeling Efficiency Evaluation</strong></p>
        <ul>
        <li>Start Time: {results['start time']}</li>
        <li>Duration: {results["duration"] // 60:.0f} min
        {(results["duration"] % 60):.02f} s</li>
        <li>Labeling efficiency: {results["labeling_efficiency"]}</li>
        </ul>"""
        if fp_figs := results.get("fp_fig"):
            for fp_fig in fp_figs:
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


class UndriftError(Exception):
    pass


class ConfluenceInterface:
    """A Interface class to access Confluence

    For access to the Confluence API, create an API token in confluence,
    and store it as an environment variable:
    $ setx CONFLUENCE_BEARER "your_confluence_api_token"
    """

    def __init__(self, base_url, username, space_key, parent_page_title, token):
        print("HERE0", base_url, username, token)
        self.confluence = con(
            url=base_url,
            username=username,
            password=token)
        
        if token is None:
            self.bearer_token = self.get_bearer_token()
        else:
            self.bearer_token = token
        self.base_url = base_url
        self.username = username
        self.space_key = space_key
        print("HERE1:", parent_page_title)
        self.parent_page_id, _ = self.get_page_properties(parent_page_title)


    def get_bearer_token(self):
        """Set this by setting the environment variable in the windows command
        line on the server:
        $ setx CONFLUENCE_BEARER <your_confluence_api_token>
        The confluence api token can be generated and copied in the personal
        details of confluence.
        """
        return os.environ.get("CONFLUENCE_TOKEN")

    def get_page_properties(self, page_title="", page_id=""):
        """
        Returns:
            id : str
                the page id
            title : str
                the page title
        """
        print("HERE:", page_title, self.space_key, page_title)
        if page_title != "":
            page = self.confluence.get_page_by_title(space = self.space_key, title = page_title)
        elif page_id != "":
            page = self.confluence.get_page_by_id(page_id=page_id)
        else:
            logger.error("One of page_title and page_id must be given.")
            raise ConfluenceInterfaceError(
                "Cannot get page properties. "
                + "One of page_title and page_id must be given."
            )
        # Needs exception for  raise ConfluenceInterfaceError("Failed to get page content.") + logger.error
        return page["id"], page["title"]

    def get_page_version(self, page_title="", page_id=""):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != "":
            page = self.confluence.get_page_by_title(space = self.space_key, title = page_title, expand='version')
        elif page_id != "":
            page = self.confluence.get_page_by_id(page_id=page_id, expand='body.version')
        else:
            logger.error("One of page_title and page_id must be given.")
        
        # Needs exception for    raise ConfluenceInterfaceError("Failed to get page content.") + logger.error
        return page['version']["number"]

    def get_page_body(self, page_title="", page_id=""):
        """
        Returns:
            data : dict
                results
                    id, title, version
        """
        if page_title != "":
            page = self.confluence.get_page_by_title(space = self.space_key, title = page_title, expand='body.storage')
        elif page_id != "":
            page = self.confluence.get_page_by_id(page_id=page_id, expand='body.storage')
        else:
            logger.error("One of page_title and page_id must be given.")
        
        # Needs exception for    raise ConfluenceInterfaceError("Failed to get page content.") + logger.error
        return page['body']['storage']['value']

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
        page = self.confluence.create_page(space=self.space_key, title = page_title, body = body_text, parent_id=parent_id, type='page', representation='storage', editor='v2', full_width=True)
        # Needs exception for    raise ConfluenceInterfaceError("Failed to get page content.") + logger.error
        return page["id"]
    
    def delete_page(self, page_id):
        # allow the page name to be used instead of page_id
        if isinstance(page_id, str) and not page_id.isnumeric():
            page_id, pgname = self.get_page_properties(page_id)
        self.confluence.remove_page(page_id, status=None, recursive=False)
        # implement logger 

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
        self.confluence.attach_file(filename=filename,  page_id = page_id, space=self.space_key)
        # Needs exception for    raise ConfluenceInterfaceError("Failed to upload attachment.") + logger.error

        attachments_container = self.confluence.get_attachments_from_content(page_id=page_id, start=0, limit=500)
        for attachment in attachments_container["results"]:
            attachment_id = attachment["id"]
            break
        
        return attachment_id

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
        attachments_container = self.confluence.get_attachments_from_content(page_id, start=0, limit=500)
        attachments = attachments_container['results']
        for attachment in attachments:
            if attachment["title"].lower() == filename.lower():
                attachment_id = attachment["id"]
                break

        return attachment_id

    def delete_attachment(self, page_id, attachment_id):
        """Deletes an attachment to a page
        Args:
            page_id : str
                the page id the attachment should be saved to.
            attachment_id : str
                the id of the attachment
        Returns:
        """
        self.confluence.delete_attachment('page_id', 'attachment_id', version=None)

    def update_page_content(self, page_name, page_id, body_update):
       status = self.confluence.update_page(
           parent_id=None,
           page_id=page_id,
           title=page_name,
           body=body_update,
           )

    def update_page_content_with_movie_attachment(
        self, page_name, page_id, filename
    ):
        self.confluence.append_page(page_id, page_name, filename, parent_id=None, type='page', representation='storage', minor_edit=False)


    def update_page_content_with_image_attachment(
        self, page_name, page_id, filename
    ):
       self.confluence.append_page(page_id, page_name, filename, parent_id=None, type='page', representation='storage', minor_edit=False)


class ConfluenceInterfaceError(Exception):
    pass

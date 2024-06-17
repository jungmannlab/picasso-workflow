#!/usr/bin/env python
"""
Module Name: analyse.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This is the picasso interface of picasso-workflow
"""
from picasso import lib, io, localize, gausslq, postprocess, clusterer
from picasso import __version__ as picassoversion
from picasso import CONFIG as pCONFIG
import os
import time
from concurrent import futures as _futures
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
from datetime import datetime
import yaml
import pickle

from picasso_workflow.util import AbstractModuleCollection
from picasso_workflow import process_brightfield
from picasso_workflow import picasso_outpost
from picasso_workflow.ripleys_analysis import run_ripleysAnalysis


logger = logging.getLogger(__name__)


def module_decorator(method):
    def module_wrapper(self, i, parameters):
        # create the results direcotry
        # method_name = get_caller_name(2)
        method_name = method.__name__
        module_result_dir = os.path.join(
            self.results_folder, f"{i:02d}_" + method_name
        )
        try:
            os.mkdir(module_result_dir)
        except FileExistsError:
            pass

        results = {
            "folder": module_result_dir,
            "start time": datetime.now().strftime("%y-%m-%d %H:%M:%S"),
        }

        # call the module
        parameters, results = method(self, i, parameters, results)

        # post-actions
        # modules only need to specifically set an error.
        if results.get("success") is None:
            results["success"] = True
            # save locs if desired
            if parameters.get("save_locs") is True or self.analysis_config.get(
                "always_save"
            ):
                if hasattr(self, "locs") and self.locs is not None:
                    self._save_locs(
                        os.path.join(results["folder"], "locs.hdf5")
                    )
                if (
                    hasattr(self, "channel_locs")
                    and self.channel_locs is not None
                ):
                    self._save_datasets_agg(results["folder"])
        results["end time"] = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        td = datetime.strptime(
            results["end time"], "%y-%m-%d %H:%M:%S"
        ) - datetime.strptime(results["start time"], "%y-%m-%d %H:%M:%S")
        results["duration"] = td.total_seconds()
        logger.debug(f"RESULTS: {results}")
        return parameters, results

    return module_wrapper


class AutoPicasso(AbstractModuleCollection):
    """A class to automatically evaluate datasets.
    Each module that runs saves their results into a separate folder.
    """

    # for single-dataset analysis
    movie = None
    info = []
    identifications = None
    locs = None
    drift = None

    # for multi-dataset analysis (aggregation)
    channel_locs = None
    channel_info = None
    channel_tags = None

    def __init__(self, results_folder, analysis_config):
        """
        Args:
            results_folder : str
                the folder all analysis modules save their respective results
                to.
            analysis_config : dict
                the general configuration. necessary items:
                    gpufit_installed : bool
                        whether the machine has gpufit installed
                (potentially) optional items
                    camera_info : dict
                        as used by picasso. Only necessary if not loaded by
                        module load_dataset
                    always_save : bool
                        whether every module should end in saving the current
                        locs
        """
        self.results_folder = results_folder
        self.analysis_config = analysis_config

    @module_decorator
    def dummy_module(self, i, parameters, results):
        """A module that does nothing, for quickly removing
        modules in a workflow without having to renumber the
        following result idcs. Only for workflow debugging,
        remove when done.
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
        return parameters, results

    ##########################################################################
    # Single dataset modules
    ##########################################################################

    @module_decorator
    def convert_zeiss_movie(self, i, parameters, results):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.
        Args:
            parameters : dict
                necessary items:
                    filepath : str
                        the czi file name to load.
                optional items:
                    filepath_raw : str
                        the file name to write to
                    info : dict, information as used by picasso
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        filepath_czi = parameters["filepath"]
        filename_raw = parameters.get("filename_raw")
        if filename_raw is None:
            filename_raw = os.path.split(
                (os.path.splitext(filepath_czi)[0] + ".raw")
            )[1]
        filepath_raw = os.path.join(results["folder"], filename_raw)
        picasso_outpost.convert_zeiss_file(
            filepath_czi, filepath_raw, parameters.get("info")
        )

        results["filepath_raw"] = filepath_raw
        results["filename_raw"] = filename_raw
        return parameters, results

    @module_decorator
    def load_dataset_movie(self, i, parameters, results):
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        The data is saved in
            self.movie
            self.info
        Args:
            parameters : dict
                necessary items:
                    filename : str
                        the (main) file name to load.
                optional items:
                    sample_movie : dict, used for creating a subsampled movie
                        keywords as used in method create_sample_movie
                    load_camera_info : bool
                        whether to load the camera information for the camera
                        used from picasso.CONFIG
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        results["picasso version"] = picassoversion
        self.movie, self.info = io.load_movie(parameters["filename"])
        results["movie.shape"] = self.movie.shape

        if parameters.get("load_camera_info"):
            cam_name = self.info[0]["Camera"]
            if cam_config := pCONFIG.get("Cameras", {}).get(cam_name):
                # # find quantum efficiency
                # filter_name = cam_config.get("Channel Device", {}).get(
                #     "Name")
                # filter_used = self.info.get(filter_name)
                # emission_wavelength = cam_config.get(
                #     "Channel Device", {}).get(
                #     "Emission Wavelengths", {}).get(filter_used)
                # qe = cam_config.get("Quantum Efficiency", {}).get(
                #     emission_wavelength, 1)

                # find camera sensitivity
                sensitivity = cam_config.get("Sensitivity")
                # sensitivity starts being a dict, and ends as a value
                cat_vals = ""
                for category in cam_config.get("Sensitivity Categories"):
                    category_value = self.info[0].get(f"{cam_name}-{category}")
                    cat_vals += f"{category}: {category_value}; "
                    sensitivity = sensitivity.get(category_value, {})
                if isinstance(sensitivity, dict):
                    raise PicassoConfigError(
                        f"""Could not find sensitivity value for camera
                        {cam_name} with category values {cat_vals} in picasso
                        CONFIG."""
                    )

                camera_info = {
                    "baseline": cam_config["Baseline"],
                    "gain": cam_config.get("Gain", 1),
                    "sensitivity": sensitivity,
                    "qe": 1,  # relevant are detected, not incident photons
                    "pixelsize": cam_config["Pixelsize"],
                }
                self.analysis_config["camera_info"] = camera_info
            else:
                raise PicassoConfigError(
                    f"Cannot load camera {cam_name} from picasso CONFIG."
                )

        # create sample movie
        if (samplemov_pars := parameters.get("sample_movie")) is not None:
            samplemov_pars["filename"] = os.path.join(
                results["folder"], samplemov_pars["filename"]
            )
            res = self._create_sample_movie(**samplemov_pars)
            results["sample_movie"] = res

        return parameters, results

    def _create_sample_movie(
        self, filename, n_sample=30, min_quantile=0, max_quantile=0.9998, fps=1
    ):
        """Create a subsampled movie of the movie loaded. The movie is saved
        to disk and referenced by filename.
        Args:
            filename : str
                the file name to save the subsamled movie as (.mp4)
            rest: as in save_movie
        """
        results = {}
        if len(self.movie) < n_sample:
            n_sample = len(self.movie)

        dn = int(len(self.movie) / (n_sample - 1))
        frame_numbers = np.arange(0, len(self.movie), dn)
        results["sample_frame_idx"] = frame_numbers

        subsampled_frames = np.array([self.movie[i] for i in frame_numbers])
        process_brightfield.save_movie(
            filename,
            subsampled_frames,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
            fps=fps,
        )
        results["filename"] = filename
        return results

    @module_decorator
    def load_dataset_localizations(self, i, parameters, results):
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
                the analysis results
        """
        results["picasso version"] = picassoversion
        self.locs, self.info = io.load_locs(parameters["filename"])
        results["nlocs"] = len(self.locs)

        return parameters, results

    def _auto_min_netgrad(
        self,
        box_size,
        frame_numbers,
        filename=None,
        start_ng=-3000,
        zscore=5,
        bins=None,
    ):
        """Calculate histograms of the net gradient at local maxima of n
        frames. For the automatic calculation of a threshold net_gradient for
        localizations, assume the background (of random local maxima without a
        localization signal) to be Gaussian distributed. Assume the background
        peak in the histogram is the highest value. The threshold net_gradient
        will be determined as zscore bacground standard deviations above the
        peak.

        Args:
            box_size : int
                the box size for evaluation
            frame_numbers : list
                the frame indexes to analyze
            filename : str, default None
                the file name of the plot to be created
                no plot generated if None
            start_ng : float
                the minimum net gradient to accept for the histogram.
                this should be below zero, to capture all net gradient
                values that exist in the data
            zscore : float
                the number of sigmas above the background net gradient peak
                to set as the estimated min net gradient threshold
            bins : None, int or array
                specify the bins of the histogram
        Returns:
            results : dict; with
                filename : string
                    filename of the generated plot
                estd_net_grad : float
                    the estimated min net gradient
        """
        results = {}
        identifications = []

        for frame_number in frame_numbers:
            identifications.append(
                localize.identify_by_frame_number(
                    self.movie, start_ng, box_size, frame_number
                )
            )
        # id_list = identifications
        identifications = np.hstack(identifications).view(np.recarray)
        identifications.sort(kind="mergesort", order="frame")

        # calculate histogram
        if bins is None:
            hi = np.quantile(identifications["net_gradient"], 0.9995)
            bins = np.linspace(start_ng, hi, num=500)
        hist, edges = np.histogram(
            identifications["net_gradient"], bins=bins, density=True
        )

        # find the background peak, assume it to be Gaussian and the
        # highest peak in the histogram: find max and FWHM
        # FWHM as the most robust detection for peak width
        # only use the lower half for FWHM calculation, as the higher
        # tail is confounded by non-background spots
        bkg_peak_height, bkg_peak_pos = np.max(hist), np.argmax(hist)
        bkg_half_lo = np.argsort(
            np.abs(hist[:bkg_peak_pos] - bkg_peak_height / 2)
        )
        bkg_fwhm = 2 * np.abs(bkg_peak_pos - bkg_half_lo[0])
        bkg_sigma = bkg_fwhm / np.sqrt(4 * np.log(2))
        # threshold at zscore * bkg_sigma
        ng_est_idx = int(zscore * bkg_sigma) + bkg_peak_pos
        if ng_est_idx >= len(edges):
            ng_est_idx = len(edges) - 1
        results["estd_net_grad"] = edges[ng_est_idx]
        bkg_peak = edges[bkg_peak_pos]
        lo_idx = int(bkg_peak_pos - bkg_sigma)
        if lo_idx < 0:
            lo_idx = 0
        bkg_sigma = bkg_peak - edges[lo_idx]

        # plot results
        if filename:
            fig, ax = plt.subplots(nrows=2)
            ax[0].plot(edges[:-1], hist, color="b", label="combined histogram")
            # for i, frame_number in enumerate(frame_numbers):
            #     hi, ed = np.histogram(
            #         id_list[i]['net_gradient'], bins=bins, density=True)
            #     ax.plot(ed[:-1], hi, color='gray')
            ylims = ax[0].get_ylim()
            ax[0].set_title("Net Gradient histogram of subsampled frames")
            ax[0].set_xlabel("net gradient")
            ax[0].set_yscale("log")
            ax[0].plot(
                [results["estd_net_grad"], results["estd_net_grad"]],
                ylims,
                color="r",
                label="estimated min net gradient: {:.0f}".format(
                    results["estd_net_grad"]
                ),
            )
            ax[0].plot(
                [edges[bkg_peak_pos], edges[bkg_peak_pos]],
                ylims,
                color="gray",
                label=f"background: {bkg_peak:.0f}+/-{bkg_sigma:.0f}",
            )
            ax[0].legend()
            # plt.show()

            sample_spots, ng_start, ng_end = self._draw_sample_spots(
                identifications, results["estd_net_grad"], box_size
            )
            ax[1].imshow(sample_spots, cmap="gray", interpolation="nearest")
            ax[1].grid(visible=False)
            ax[1].tick_params(bottom=False, left=False)
            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_title(
                "spots with net_gradient " + f"{ng_start:.0f} to {ng_end:.0f}"
            )

            results["filename"] = filename
            plt.tight_layout()
            fig.savefig(results["filename"])
        return results

    def _draw_sample_spots(
        self,
        identifications,
        estd_net_grad,
        box_size,
        sample_spots_rows=4,
        sample_spots_cols=12,
    ):
        """Pull up example spots at the threshold net_grad for
        visualizing the automatically found min net gradient.
        Args:
            identifications : np recarray
                identifications from subsampled frames with
                very low min net gradient.
            estd_net_grad : float
                the estimated min net gradient
            box_size : uneven int
                the box size to display
            sample_spots_rows : int
                the number of rows of spots to display
            sample_spots_cols : int
                the number of cols of spots to display
        Returns:
            canvas : 2D array
                the canvas with spots to display
            ng_start : float
                the lowest net gradient shown (upper left spot)
            ng_end : float
                the highest net gradient shown (lower right spot)
        """
        n_spots = sample_spots_cols * sample_spots_rows
        sample_idxs = np.argsort(
            np.abs(identifications["net_gradient"] - estd_net_grad)
        )[:n_spots]
        sample_identifications = identifications[sample_idxs]
        sample_identifications = sample_identifications[
            np.argsort(sample_identifications["net_gradient"])
        ]

        # sample_spots = localize.get_spots(
        sample_spots = picasso_outpost.get_spots(
            self.movie,
            sample_identifications,
            box_size,
            self.analysis_config["camera_info"],
        )
        ng_start = np.min(sample_identifications["net_gradient"])
        ng_end = np.max(sample_identifications["net_gradient"])

        border_width = 2
        canvas_size = (
            box_size * sample_spots_rows
            + border_width * (sample_spots_rows - 1),
            box_size * sample_spots_cols
            + border_width * (sample_spots_cols - 1),
        )

        canvas = np.zeros(canvas_size, dtype=np.uint8)
        for i, spot in enumerate(sample_spots):
            ix, iy = i // sample_spots_cols, i % sample_spots_cols
            pix = ix * (box_size + border_width)
            piy = iy * (box_size + border_width)
            logger.debug(f"drawing spot {i} at ({pix}, {piy}: {str(spot)}")
            canvas[pix : pix + box_size, piy : piy + box_size] = (
                picasso_outpost.normalize_spot(spot)
            )
        return canvas, ng_start, ng_end

    @module_decorator
    def identify(self, i, parameters, results):
        """Identifies localizations in a loaded dataset.
        The data is saved in
            self.identifications
        Args:
            parameters : dict
                necessary items:
                    box_size : as always
                    min_gradient : only if not auto_netgrad
                optional items:
                    auto_netgrad : dict, in case the min net_gradient
                        shall be automatically detected.
                        Items correspond to arguments of _auto_min_netgrad
                    ids_vs_frame : dict
                        for plotting identifications vs time
                        items correspond to arguments of _plot_ids_vs_frame
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        # auto-detect net grad if required:
        if (autograd_pars := parameters.get("auto_netgrad")) is not None:
            if "filename" in autograd_pars.keys():
                autograd_pars["filename"] = os.path.join(
                    results["folder"], autograd_pars["filename"]
                )
            res = self._auto_min_netgrad(**autograd_pars)
            results["auto_netgrad"] = res
            parameters["min_gradient"] = res["estd_net_grad"]

        curr, futures = localize.identify_async(
            self.movie,
            parameters["min_gradient"],
            parameters["box_size"],
            roi=None,
        )
        self.identifications = localize.identifications_from_futures(futures)
        results["num_identifications"] = len(self.identifications)

        if (pars := parameters.get("ids_vs_frame")) is not None:
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            results["ids_vs_frame"] = self._plot_ids_vs_frame(**pars)

        # add info
        new_info = {
            "Generated by": "picasso-workflow : identify",
            "Box Size": parameters["box_size"],
            "Min. Net Gradient": float(parameters["min_gradient"]),
            # "parameters": parameters,
        }
        self.info = self.info + [new_info]

        return parameters, results

    def _plot_ids_vs_frame(self, filename):
        """Plot identifications vs frame index"""
        results = {}
        frames = np.arange(len(self.movie))
        bins = np.arange(len(self.movie) + 1) - 0.5
        locs, _ = np.histogram(self.identifications["frame"], bins=bins)
        fig, ax = plt.subplots()
        ax.plot(frames, locs)
        ax.set_xlabel("frame")
        ax.set_ylabel("number of identifications")
        results["filename"] = filename
        fig.savefig(results["filename"])
        plt.close(fig)
        return results

    @module_decorator
    def localize(self, i, parameters, results):
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
                the analysis results
        """
        em = self.analysis_config["camera_info"]["gain"] > 1
        spots = localize.get_spots(
            self.movie,
            self.identifications,
            parameters["box_size"],
            self.analysis_config["camera_info"],
        )
        if self.analysis_config["gpufit_installed"]:
            theta = gausslq.fit_spots_gpufit(spots)
            self.locs = gausslq.locs_from_fits_gpufit(
                self.identifications, theta, parameters["box_size"], em
            )
        else:
            if parameters["fit_parallel"]:
                # theta = gausslq.fit_spots_parallel(spots, asynch=False)
                fs = gausslq.fit_spots_parallel(spots, asynch=True)
                n_tasks = len(fs)
                with tqdm(total=n_tasks, unit="task") as progress_bar:
                    for f in _futures.as_completed(fs):
                        progress_bar.update()
                theta = gausslq.fits_from_futures(fs)
                em = self.analysis_config["camera_info"]["gain"] > 1
                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters["box_size"], em
                )
            else:
                theta = np.empty((len(spots), 6), dtype=np.float32)
                theta.fill(np.nan)
                for i in tqdm(range(len(spots))):
                    theta[i] = gausslq.fit_spot(spots[i])

                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters["box_size"], em
                )

        if pars := parameters.get("locs_vs_frame"):
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            results["locs_vs_frame"] = self._plot_locs_vs_frame(
                pars["filename"]
            )

        # add info
        localize_info = {
            "Generated by": "Picasso Localize",
            "ROI": None,
            "Box Size": int(parameters["box_size"]),
            # "Min. Net Gradient": min_net_gradient,
            # "Convergence Criterion": convergence,
            # "Max. Iterations": max_iterations,
            "Pixelsize": float(
                self.analysis_config["camera_info"].get("pixelsize")
            ),
            "Fit method": "gausslq",
            "Wrapped by": "picasso-workflow : localize",
            # "parameters": parameters,
        }
        self.info = self.info + [localize_info]

        # save locs
        if pars := parameters.get("save_locs"):
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            self._save_locs(pars["filename"])

        results["locs_columns"] = self.locs.dtype.names
        return parameters, results

    def _plot_locs_vs_frame(self, filename):
        results = {}
        frames = np.arange(len(self.movie))
        # bins = np.arange(len(self.movie) + 1) - .5

        df_locs = pd.DataFrame(self.locs)
        gbframe = df_locs.groupby("frame")
        photons_mean = gbframe["photons"].mean()
        photons_std = gbframe["photons"].std()
        sx_mean = gbframe["sx"].mean()
        sx_std = gbframe["sx"].std()
        sy_mean = gbframe["sy"].mean()
        sy_std = gbframe["sy"].std()

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(frames, photons_mean, color="b", label="mean photons")
        xhull = np.concatenate([frames, frames[::-1]])
        yhull = np.concatenate(
            [
                photons_mean + photons_std,
                photons_mean[::-1] - photons_std[::-1],
            ]
        )
        ax[0].fill_between(
            xhull, yhull, color="b", alpha=0.2, label="std photons"
        )
        ax[0].set_xlabel("frame")
        ax[0].set_ylabel("photons")
        ax[0].legend()
        ax[1].plot(frames, sx_mean, color="c", label="mean sx")
        yhull = np.concatenate(
            [sx_mean + sx_std, sx_mean[::-1] - sx_std[::-1]]
        )
        ax[1].fill_between(xhull, yhull, color="c", alpha=0.2, label="std sx")
        ax[1].plot(frames, sy_mean, color="m", label="mean sy")
        yhull = np.concatenate(
            [sy_mean + sy_std, sy_mean[::-1] - sy_std[::-1]]
        )
        ax[1].fill_between(xhull, yhull, color="m", alpha=0.2, label="std sy")
        ax[1].set_xlabel("frame")
        ax[1].set_ylabel("width")
        ax[1].legend()
        results["filename"] = filename
        fig.savefig(results["filename"])
        plt.close(fig)
        return results

    @module_decorator
    def export_brightfield(self, i, parameters, results):
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
                the analysis results
                    labeled filepath : dict
                        keys : labels
                        values : filepaths
        """
        fps_in = parameters["filepath"]
        if isinstance(fps_in, str):
            fps_in = [fps_in]
        if isinstance(fps_in, list):
            d = {}
            for i, fp in enumerate(fps_in):
                d[str(i)] = fp
            fps_in = d
        fps_out = {}
        for label, fp in fps_in.items():
            mov, _ = io.load_movie(fp)
            frame = mov[0]
            fn = os.path.split(fp)[1]
            fn = os.path.splitext(fn)[0] + ".png"
            fp = os.path.join(results["folder"], fn)
            fps_out[label] = fp
            min_quantile = parameters.get("min_quantile", 0)
            max_quantile = parameters.get("max_quantile", 1)
            process_brightfield.save_frame(
                fp, frame, min_quantile, max_quantile
            )
        results["labeled filepaths"] = fps_out
        results["success"] = True
        return parameters, results

    @module_decorator
    def undrift_rcc(self, i, parameters, results):
        """Undrifts localized data using redundant cross correlation.
        drift is saved in
        self.drift

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    segmentation : the number of frames segmented for RCC
                    dimensions : list
                        the dimensions undrifted. For picasso RCC, this
                        is always ['x', 'y']
                optional items:
                    filename : str
                        the drift txt file name
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results

        """
        seg_init = parameters["segmentation"]
        for i in range(parameters.get("max_iter_segmentations", 3)):
            # if the segmentation is too low, the process raises an error
            # adaptively increase the value.
            try:
                self.drift, self.locs = postprocess.undrift(
                    self.locs,
                    self.info,
                    segmentation=parameters["segmentation"],
                    display=False,
                )
                results["success"] = True
                break
            except ValueError:
                parameters["segmentation"] = 2 * parameters["segmentation"]
                logger.debug(
                    f"""RCC with segmentation {parameters["segmentation"]}
                    raised an error. Doubling."""
                )
                results[
                    "message"
                ] = f"""Initial Segmentation of {seg_init}
                    was too low."""
        else:  # did not work until the end
            logger.error(
                f"""RCC failed up to segmentation {parameters["segmentation"]}.
                Aborting."""
            )
            max_segmentation = parameters["segmentation"]
            # initial segmentation
            parameters["segmentation"] = int(
                parameters["segmentation"]
                / 2 ** parameters["max_iter_segmentations"]
            )
            results[
                "message"
            ] = f"""
                    Undrifting did not work in
                    {parameters['max_iter_segmentations']} iterations
                    up to a segmentation of {max_segmentation}."""
            results["success"] = False

        parameters["dimensions"] = ["x", "y"]

        if parameters.get("filename"):
            results["filepath_driftfile"] = os.path.join(
                results["folder"], parameters["filename"]
            )
            np.savetxt(
                results["filepath_driftfile"], self.drift, delimiter=","
            )
            results["filepath_plot"] = (
                os.path.splitext(results["filepath_driftfile"])[0] + ".png"
            )
            self._plot_drift(
                results["filepath_plot"], parameters["dimensions"]
            )

        # add info
        new_info = {
            "Generated by": "picasso-workflow : undrift_rcc",
            "parameters": parameters,
        }
        self.info = self.info + [new_info]

        # save locs
        if pars := parameters.get("save_locs"):
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            self._save_locs(pars["filename"])

        return parameters, results

    def _plot_drift(self, filename, dimensions):
        fig, ax = plt.subplots()
        frames = np.arange(self.drift.shape[0])
        for i, dim in enumerate(dimensions):
            if isinstance(self.drift, np.recarray):
                ax.plot(frames, self.drift[dim], label=dim)
            else:
                ax.plot(frames, self.drift[:, i], label=dim)
        ax.set_xlabel("frame")
        ax.set_ylabel("drift [px]")
        ax.set_title("drift graph")
        ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    @module_decorator
    def manual(self, i, parameters, results):
        """Handles a manual step: if the files required are not
        present, prompt the user to provide them. if they are, move
        to the next step.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    prompt : str
                        the user prompt
                    filename : str
                        the file the user should provide.
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        filepath = os.path.join(results["folder"], parameters["filename"])
        if os.path.exists(filepath):
            results["filepath"] = filepath
            results["success"] = True
        else:
            msg = "This is a manual step. Please provide input, "
            msg += "and re-execute the workflow. "
            msg += parameters["prompt"]
            msg += f" The resulting file should be {filepath}."
            results["message"] = msg
            logger.debug(msg)
            print(msg)
            results["success"] = False
            # raise ManualInputLackingError(f'{filepath} missing.')
        return parameters, results

    @module_decorator
    def summarize_dataset(self, i, parameters, results):
        for meth, meth_pars in parameters["methods"].items():
            if meth.lower() == "nena":
                try:
                    res, best_val = postprocess.nena(self.locs, self.info)
                    fp_plot = os.path.join(results["folder"], "nena.png")
                    self._plot_nena(res, fp_plot)
                    all_best_vals = {
                        "a": res.best_values["a"],
                        "s": res.best_values["s"],
                        "ac": res.best_values["ac"],
                        "dc": res.best_values["dc"],
                        "sc": res.best_values["sc"],
                    }
                    pixelsize = self.analysis_config["camera_info"][
                        "pixelsize"
                    ]
                    results["nena"] = {
                        "res": str(all_best_vals),
                        "chisqr": res.chisqr,
                        "NeNa": (
                            f"{best_val:.3f} px;"
                            + f" {pixelsize*best_val:.3f} nm "
                        ),
                        "nena-px": best_val,
                        "nena-nm": pixelsize * best_val,
                        "filepath_plot": fp_plot,
                    }
                except ValueError as e:
                    logger.error(e)
                    results["nena"] = {"res": "Fitting Error", "best_vals": ""}
                except Exception as e:
                    logger.error(e)
                    results["nena"] = {
                        "res": str(e),
                        "best_vals": "Error.",
                    }
            else:
                raise NotImplementedError(
                    f"Description method {meth} not implemented."
                )
        return parameters, results

    def _plot_nena(self, nena_result, filepath_plot):
        fig, ax = plt.subplots()
        d = nena_result.userkws["d"]
        ax.set_title("Next frame neighbor distance histogram")
        ax.plot(d, nena_result.data, label="Data")
        ax.plot(d, nena_result.best_fit, label="Fit")
        ax.set_xlabel("Distance [px]")
        ax.set_ylabel("Counts")
        ax.legend(loc="best")
        fig.savefig(filepath_plot)

    # @module_decorator
    # def aggregate_cluster(self, i, parameters, results):
    #     """Aggregate along the 'cluster' column.
    #     Uses picasso.postprocess.cluster_combine
    #     Args:
    #         i : int
    #             the index of the module
    #         parameters: dict
    #             with required keys:
    #             and optional keys:
    #                 save_locs : bool
    #                     whether to save the locs into the results folder
    #         results : dict
    #             the results this function generates. This is created
    #             in the decorator wrapper
    #     """
    #     self.locs = postprocess.cluster_combine(self.locs)
    #     combined_info = {"Generated by": "Picasso Combine"}
    #     self.info.append(combined_info)
    #     results["nlocs"] = np.len(self.locs)
    #     if parameters.get("save"):
    #         self._save_locs(os.path.join(results["folder"], "locs.hdf5"))
    #     return parameters, results

    @module_decorator
    def density(self, i, parameters, results):
        """ACalculate local localization density
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
        self.locs = postprocess.compute_local_density(
            self.locs, self.info, parameters["radius"]
        )
        density_info = {
            "Generated by": "Picasso Density",
            "Wrapped by": "picasso-workflow : density",
            "Radius": float(parameters["radius"]),
        }
        self.info.append(density_info)
        return parameters, results

    @module_decorator
    def dbscan(self, i, parameters, results):
        """Perform dbscan clustering. After this module, the standard
        locs will be the cluster centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    radius : float
                        the dbscan radius
                    min_density : float
                        the dbscan min_density
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        radius = parameters["radius"]
        min_density = parameters["min_density"]
        pixelsize = self.analysis_config["camera_info"]["pixelsize"]
        # label locs according to clusters
        self.locs = clusterer.dbscan(self.locs, radius, min_density, pixelsize)
        dbscan_info = {
            "Generated by": "Picasso DBSCAN",
            "Radius": radius,
            "Minimum local density": min_density,
            "Wrapped by": "picasso-workflow : dbscan",
        }
        self.info.append(dbscan_info)
        filepath = os.path.join(results["folder"], "locs_dbscan.hdf5")
        self._save_locs(filepath)

        self.locs = clusterer.find_cluster_centers(self.locs, pixelsize)
        logger.warning("saving cluster centeras as locs. Is that intended?")
        return parameters, results

    @module_decorator
    def hdbscan(self, i, parameters, results):
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
        min_cluster = parameters["min_cluster"]
        min_samples = parameters["min_samples"]
        pixelsize = self.analysis_config["camera_info"]["pixelsize"]

        # label locs according to clusters
        self.locs = clusterer.hdbscan(
            self.locs, min_cluster, min_samples, pixelsize
        )
        hdbscan_info = {
            "Generated by": "Picasso HDBSCAN",
            "Min. cluster": min_cluster,
            "Min. samples": min_samples,
            "Wrapped by": "picasso-workflow : hdbscan",
        }
        self.info.append(hdbscan_info)
        filepath = os.path.join(results["folder"], "locs_hdbscan.hdf5")
        self._save_locs(filepath)

        self.locs = clusterer.find_cluster_centers(self.locs, pixelsize)
        logger.warning("saving cluster centeras as locs. Is that intended?")

        return parameters, results

    @module_decorator
    def smlm_clusterer(self, i, parameters, results):
        """Perform smlm clustering. After this module, the standard
        locs will be the cluster centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    radius : float
                        the smlm radius, in px
                    min_locs : float
                        the smlm min_locs
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
                    basic_fa : bool
                        the smlm basic fa, default: False
                    radius_z : float
                        the smlm radius_z, default: None
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        radius = parameters["radius"]
        min_locs = parameters["min_locs"]
        basic_fa = parameters.get("basic_fa", False)
        parameters["basic_fa"] = basic_fa
        radius_z = parameters.get("radius_z", None)
        parameters["radius_z"] = radius_z
        pixelsize = self.analysis_config["camera_info"]["pixelsize"]

        if radius_z is not None:  # 3D
            params = [radius, radius_z, min_locs, 0, basic_fa, 0]
        else:  # 2D
            params = [radius, min_locs, 0, basic_fa, 0]

        # label locs according to clusters
        self.locs = clusterer.cluster(self.locs, params, pixelsize)
        smlm_cluster_info = {
            "Generated by": "Picasso SMLM clusterer",
            "Radius_xy": radius,
            "Radius_z": radius_z,
            "Min locs": min_locs,
            "Basic frame analysis": basic_fa,
            "Wrapped by": "picasso-workflow : smlm_clusterer",
        }
        self.info.append(smlm_cluster_info)
        filepath = os.path.join(results["folder"], "locs_smlm.hdf5")
        self._save_locs(filepath)

        self.locs = clusterer.find_cluster_centers(self.locs, pixelsize)
        logger.warning("saving cluster centeras as locs. Is that intended?")

        return parameters, results

    @module_decorator
    def nneighbor(self, i, parameters, results):
        """Perform nearest neighbor calculation
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    dims : list of str
                        the distance dimensions, e.g. ['x', 'y']
                        or ['x', 'y', 'z']
                    nth_NN : int
                        calculate the 1st to nth nearest neighbor distances
                    nth_rdf : int
                        calculate distances up to the 95th percile of the
                        nth_rdf nearest neighbor
                    subsample_1stNN : int
                        by how much fold to subsample distances from the
                        median of the 1st nearest nteighbor. Default is 20
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        if self.locs is not None:
            locs = self.locs
        elif self.channel_locs is not None:
            locs = self.channel_locs[0]
        else:
            raise KeyError("No locs loaded")

        fig, ax = plt.subplots(nrows=2)
        points = np.array(
            locs[parameters["dims"]].tolist()
        )  # c-locs[0] only for now, before sgl/agg workflow refactoring!!
        # convert all dimensions to nanometers
        pixelsize = self.analysis_config["camera_info"]["pixelsize"]
        for i, dim in enumerate(parameters["dims"]):
            if dim in ["x", "y"]:
                points[:, i] = points[:, i] * pixelsize

        # print(points)
        # print(points.shape)
        alldist = distance.cdist(points, points)
        logger.debug("found all distances")
        alldist = np.sort(alldist, axis=1)
        logger.debug("sorted all distances")

        # calculate bins
        NN_median = np.median(alldist[:, 1])
        deltar = NN_median / parameters.get("subsample_1stNN", 20)
        rmax_NN = np.quantile(alldist[:, parameters["nth_NN"]], 0.95)
        rmax_rdf = np.quantile(alldist[:, parameters["nth_rdf"]], 0.95)

        logger.debug("calculated bin parameters")
        _, _, density = self._calc_radial_distribution_function(
            alldist, deltar, rmax_rdf, d=len(parameters["dims"]), ax=ax[0]
        )
        results["density_rdf"] = density
        logger.debug("calculated radial distribution function")
        # print(alldist)
        # print(alldist.shape)
        # out_path = os.path.join(results["folder"], "nneighbors_all.txt")
        # np.savetxt(out_path, np.sort(alldist, axis=1), newline="\r\n")
        # alldist[alldist == 0] = float("inf")
        # nneighbors = np.sort(alldist, axis=1)[:, : parameters["nth"]]
        nneighbors = alldist[:, 1 : parameters["nth_NN"] + 1]
        out_path = os.path.join(results["folder"], "nneighbors.txt")
        np.savetxt(out_path, nneighbors, newline="\r\n")
        results["nneighbors"] = out_path

        # plot results
        colors = cm.get_cmap("viridis", nneighbors.shape[1]).colors
        bins = np.arange(0, rmax_NN, step=deltar)
        nnhist_obs = np.zeros((len(bins), nneighbors.shape[1]))
        for i in range(nnhist_obs.shape[1]):
            k = i + 1
            _ = ax[1].hist(
                nneighbors[:, i],
                bins=bins,
                color=colors[i],
                alpha=0.2,
                label=f"k={k}",
            )
        ax[1].legend()
        ax[1].set_xlabel("Distance [nm]")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("Nearest Neighbor Histogram")
        results["fp_fig"] = os.path.join(results["folder"], "nndist.png")
        plt.tight_layout()
        fig.savefig(results["fp_fig"])

        return parameters, results

    def _calc_radial_distribution_function(
        self, alldist, deltar, rmax, d=2, ax=None
    ):
        rs = np.arange(
            0,
            rmax + 2 * deltar,
            deltar,
        )
        n_means = np.zeros_like(rs)
        d_areas = np.zeros_like(rs)

        logger.debug(f"calculating {len(rs)} rdf points")

        nspots = alldist.shape[0]

        alldist = alldist[:, np.min(alldist, axis=1) <= np.max(rs)]
        logger.debug("cropped 2d alldist")
        # distarray = np.sort(alldist.flatten())
        # logger.debug('flattened distarray')
        # distarray = alldist.flatten()
        # logger.debug('flattened distarray')
        # distarray = distarray[distarray <= np.max(rs)]

        logger.debug("prepared distarray")

        for i, r in enumerate(rs):
            # area = 2 * np.pi * r**2
            # n_mean = np.sum(alldist < r) / len(locs)
            # crdf[i] = n_mean / area
            d_areas[i] = 2 * np.pi * r * deltar
            # n_means[i] = np.sum(distarray <= r) / nspots
            n_means[i] = np.sum(alldist <= r) / nspots
        d_n_means = n_means[1:] - n_means[:-1]
        rdf = d_n_means / d_areas[1:]
        # rdf = crdf[1:] - crdf[:-1]

        density = np.median(rdf[int(len(rs) / 2) :])

        # plot results
        ax.plot(rs[1:], rdf * 1e3**d)
        ax.set_xlabel("Radius [nm]")
        ax.set_ylabel(f"density [µm^{d}]")
        ax.set_title("Radial Distribution Function")
        return rs[1:], rdf, density

    @module_decorator
    def fit_csr(self, i, parameters, results):
        """Fit a Completely Spatially Random Distribution to
        nearest neighbors
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    nneighbors : str or 2D array
                        if str: filepath to a txt file of numpy-saved data
                            (as by module nneighbor above)
                        if 2D array (N, k): kth nearest neighbor distances
                            for N points
                    dimensionality : int
                        the dimensionality: 2 or 3 - 2D or 3D
                    kmin : int
                        the minimum-th NN to fit. default 1
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        if isinstance(parameters["nneighbors"], str):
            nneighbors = np.loadtxt(parameters["nneighbors"])
        else:
            nneighbors = parameters["nneighbors"]
        # print(nneighbors.shape)
        # return
        k_max = nneighbors.shape[1]
        # nspots = nneighbors.shape[0]
        d = parameters["dimensionality"]
        rho_init = 2 / (2 * d * np.pi * np.median(nneighbors[:, 0]) ** d)
        rho_mle, fitresult = (
            picasso_outpost.estimate_density_from_neighbordists(
                nneighbors.T, rho_init, kmin=1, rho_bound_factor=10
            )
        )
        # print(fitresult)
        logger.debug(str(fitresult))
        results["density"] = rho_mle

        # plot results
        fig, ax = plt.subplots()
        colors = cm.get_cmap("viridis", k_max).colors
        bin_max = np.quantile(nneighbors[:, -1], 0.99)
        bins = np.linspace(0, bin_max, num=50)
        nnhist_obs = np.zeros((len(bins), k_max))
        nnhist_an = np.zeros_like(nnhist_obs)
        for i in range(nnhist_an.shape[1]):
            k = i + 1
            # nnhist_obs, edges = np.histogram(nneighbors[:, i], bins=bins)
            nnhist_an = picasso_outpost.nndistribution_from_csr(
                bins, k, rho_mle, d=parameters["dimensionality"]
            )
            if i == 0:
                lbl = f"rho_init {1E6*rho_init:.1f} um^2"
                lblf = f"rho_fit {1E6*rho_mle:.1f} um^2"
            else:
                lbl = f"observed k={k}"
                lblf = f"fitted k={k}"
            _ = ax.hist(
                nneighbors[:, i],
                bins=bins,
                density=True,
                color=colors[i],
                alpha=0.2,
                label=lbl,
            )
            ax.plot(
                bins,  # + (bins[1] - bins[0]) / 2,
                nnhist_an * 1,  # no idea why we need this factor
                color=colors[i],
                linestyle="--",
                label=lblf,
            )
        ax.legend()
        ax.set_xlabel("Distance [nm]")
        ax.set_ylabel("probability density")
        ax.set_title("Nearest Neighbor Distribution")
        results["fp_fig"] = os.path.join(results["folder"], "nndist_fit.png")
        fig.savefig(results["fp_fig"])

        return parameters, results

    # @module_decorator
    # def radial_distribution_function(self, i, parameters, results):
    #     """Generate the Radial Distribution Function,
    #     Whis is the sum of nearest neighbors with geometry factor.
    #     At long radii, its value is the overall density.

    #     Every spot is picked, pick radii are altered and the density
    #     calculated. The RDF is the difference between those densities.
    #     Args:
    #         i : int
    #             the index of the module
    #         parameters: dict
    #             with required keys:
    #                 dims : list of str
    #                     the distance dimensions, e.g. ['x', 'y']
    #                     or ['x', 'y', 'z']
    #                 rmax : float
    #                     the maximum r to evaluate
    #                 deltar : float
    #                     the step size in r
    #             and optional keys:
    #                 save_locs : bool
    #                     whether to save the locs into the results folder
    #         results : dict
    #             the results this function generates. This is created
    #             in the decorator wrapper
    #     """
    #     rs = np.arange(
    #         0,
    #         parameters["rmax"] + 2 * parameters["deltar"],
    #         parameters["deltar"])
    #     n_means = np.zeros_like(rs)
    #     d_areas = np.zeros_like(rs)
    #     locs = self.channel_locs[
    #         0
    #     ]  # c-locs[0] only for now, before sgl/agg workflow refactoring!!

    #     points = np.array(locs[parameters["dims"]].tolist())
    #     # convert all dimensions to nanometers
    #     pixelsize = self.analysis_config["camera_info"]["pixelsize"]
    #     for i, dim in enumerate(parameters["dims"]):
    #         if dim in ["x", "y"]:
    #             points[:, i] = points[:, i] * pixelsize

    #     alldist = distance.cdist(points, points)
    #     # alldist[alldist == 0] = float("inf")
    #     alldist = np.sort(alldist, axis=1)

    #     for i, r in enumerate(rs):
    #         # area = 2 * np.pi * r**2
    #         # n_mean = np.sum(alldist < r) / len(locs)
    #         # crdf[i] = n_mean / area
    #         d_areas[i] = 2 * np.pi * r * parameters["deltar"]
    #         n_means[i] = np.sum(alldist < r) / len(locs)
    #     d_n_means = n_means[1:] - n_means[:-1]
    #     rdf = d_n_means / d_areas[1:]
    #     # rdf = crdf[1:] - crdf[:-1]

    #     results["density"] = np.median(rdf[int(len(rs) / 2):])

    #     # plot results
    #     fig, ax = plt.subplots()
    #     ax.plot(rs[1:], rdf)
    #     ax.set_xlabel("Radius [nm]")
    #     ax.set_ylabel("probability density")
    #     ax.set_title("Radial Distribution Function")
    #     results["fp_fig"] = os.path.join(results["folder"], "rdf.png")
    #     fig.savefig(results["fp_fig"])

    #     return parameters, results

    @module_decorator
    def save_single_dataset(self, i, parameters, results):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                        filename : str
                            the name of the dataset
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        results["filepath"] = os.path.join(
            results["folder"], parameters["filename"]
        )
        res = self._save_locs(results["filepath"])
        for k, v in res.items():
            results[k] = v
        return parameters, results

    def _save_locs(self, filename):
        t00 = time.time()

        io.save_locs(filename, self.locs, self.info)
        # # when the paths get long, the hdf5 library throws an error, so chdir
        # # but apparently, the issue is the length of the filename itself
        # previous_dir = os.getcwd()
        # parent_dir, fn = os.path.split(filename)
        # os.chdir(parent_dir)
        # io.save_locs(fn, self.locs, self.info)
        # os.chdir(previous_dir)

        dt = np.round(time.time() - t00, 2)
        results_save = {"duration": dt}
        return results_save

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    @module_decorator
    def load_datasets_to_aggregate(self, i, parameters, results):
        """Loads the results of single-dataset workflows
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    filepaths : list of str
                        the hdf5 files to load.
                    tags : list of str
                        the tags to name the datasets
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        self.channel_locs = []
        self.channel_info = []
        self.channel_tags = []
        for i, (fp, tag) in enumerate(
            zip(parameters["filepaths"], parameters["tags"])
        ):
            locs, info = io.load_locs(fp)
            # locs = lib.append_to_rec(
            #     locs,
            #     np.full(len(locs), tag, dtype='U10'),
            #     "channel",
            # )
            locs = lib.append_to_rec(
                locs,
                i * np.ones(len(locs), dtype=np.int8),
                "channel",
            )
            self.channel_locs.append(locs)
            self.channel_info.append(info)
            self.channel_tags.append(tag)
        results["filepaths"] = parameters["filepaths"]
        results["tags"] = parameters["tags"]
        return parameters, results

    @module_decorator
    def align_channels(self, i, parameters, results):
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
                    fig_filename : str
                        the location to save the drift figure to
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        if parameters.get("filepaths"):
            self.channel_locs = []
            self.channel_info = []
            self.channel_tags = []
            for fp in parameters["filepaths"]:
                locs, info = io.load_locs(fp)
                self.channel_locs.append(locs)
                self.channel_info.append(info)
                self.channel_tags.append(os.path.split(fp)[1])

        shifts, cum_shifts = picasso_outpost.align_channels(
            self.channel_locs,
            self.channel_info,
            **parameters.get("align_pars", {}),
        )
        results["shifts"] = cum_shifts[:, :, -1]

        if fn := parameters.get("fig_filename"):
            fig_filepath = os.path.join(results["folder"], fn)
            picasso_outpost.plot_shift(shifts, cum_shifts, fig_filepath)
            results["fig_filepath"] = fig_filepath

        # add info
        new_info = {
            "Generated by": "picasso-workflow : align_channels",
            "shifts": str(results["shifts"]),
            # "parameters": parameters,
        }
        for i in range(len(self.channel_info)):
            self.channel_info[i].append(new_info)

        return parameters, results

    @module_decorator
    def combine_channels(self, i, parameters, results):
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
        combine_map = {tag: i for i, tag in enumerate(self.channel_tags)}
        results["combine_map"] = combine_map
        fp_combinemap = os.path.join(results["folder"], "combine_map.yaml")
        with open(fp_combinemap, "w") as f:
            yaml.dump(combine_map, f)
        results["fp_combinemap"] = fp_combinemap
        combine_col = parameters.get("combine_col", "combine_id")
        for i in range(len(self.channel_locs)):
            locs = self.channel_locs[i]
            self.channel_locs[i] = lib.append_to_rec(
                locs, data=i * np.ones(len(locs)), name=combine_col
            )
        combined_locs = np.lib.recfunctions.stack_arrays(
            self.channel_locs,
            asrecarray=True,
            usemask=False,
            autoconvert=True,
        )
        # sort like all Picasso localization lists
        combined_locs.sort(kind="mergesort", order="frame")

        # replace the channel_locs with the one combined dataset
        self.channel_locs = [combined_locs]
        new_info = {
            "Generated by": "picasso-workflow : combine_channels",
            "files combined": self.channel_tags,
        }
        info = self.channel_info[0] + [new_info]
        self.channel_info = [info]
        tag = parameters.get("tag", "combined-channels")
        self.channel_tags = [tag]

        return parameters, results

    @module_decorator
    def save_datasets_aggregated(self, i, parameters, results):
        """save data of multiple single-dataset workflows from one
        aggregation workflow.
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
        allfps = self._save_datasets_agg(results["folder"])
        results["filepaths"] = allfps

        return parameters, results

    def _save_datasets_agg(self, folder):
        """ """
        allfps = []
        for locs, info, tag in zip(
            self.channel_locs, self.channel_info, self.channel_tags
        ):
            filepath = os.path.join(folder, tag + ".hdf5")
            io.save_locs(filepath, locs, info)
            allfps.append(filepath)
        return allfps

    @module_decorator
    def spinna_manual(self, i, parameters, results):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    proposed_labeling_efficiency : float, range 0-100
                        labeling efficiency percentage, default for all targets
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_labeling_uncertainty : float
                        labeling uncertainty [nm]; good value is e.g. 5
                        used proposed value in spinna_config.csv and can be
                         alteredmanually after the first run of this module
                    proposed_n_simulate : int
                        number of target molecules to simulated;
                        good value is e.g. 50000
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_density : int
                        density to simulate;
                        area density if 2D; volume density if 3D
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_nn_plotted : int
                        number of nearest neighbors to plot
                        used proposed value in spinna_config.csv and can be
                         alteredmanually after the first run of this module
                and optional keys:
                    structures : list of dict
                        SPINNA structures. Each structure dict has
                            "Molecular targets": list of str,
                            "Structure title": str,
                            "TARGET_x": list of float,
                            "TARGET_y": list of float,
                            "TARGET_z": list of float,
                        where TARGET is one each of the target names in
                        "Molecular targets"
                    structures_d : float
                        distance between molecules within auto-generated
                        structures, in nm. Only necessary if 'structures'
                        is not given.
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        cfg_fp = os.path.join(results["folder"], "spinna_config.csv")
        if os.path.exists(cfg_fp):
            prepped = True
        else:
            prepped = False

        if not prepped:
            spinna_config = {}
            data_2d = "z" not in self.channel_locs[0].dtype.names
            if data_2d:
                spinna_config["rotation_mode"] = ["2D"]
                area = (
                    parameters["proposed_n_simulate"]
                    / parameters["proposed_density"]
                )
                spinna_config["area"] = [area]
                d = 2
            else:
                spinna_config["rotation_mode"] = ["3D"]
                z_range = int(self.locs["z"].max() - self.locs["z"].min())
                volume = (
                    parameters["proposed_n_simulate"]
                    / parameters["proposed_density"]
                )
                spinna_config["volume"] = [volume]
                spinna_config["z_range"] = [z_range]
                d = 3

            # prepare input files for the user to edit, with default values
            spinna_structs = parameters.get("structures")
            if spinna_structs is None:
                spinna_structs = self._create_spinna_structure(
                    self.channel_tags,
                    [[1, 2]] * len(self.channel_tags),
                    distance=parameters["structures_d"],
                    dimensionality=d,
                )
            structs_fn = "spinna_structs.yaml"
            structs_fp = os.path.join(results["folder"], structs_fn)
            with open(structs_fp, "w") as f:
                yaml.dump_all(spinna_structs, f)

            spinna_config["structures_filename"] = [structs_fp]
            for locs, info, tag in zip(
                self.channel_locs, self.channel_info, self.channel_tags
            ):
                locs_fn = tag + ".hdf5"
                locs_fp = os.path.join(results["folder"], locs_fn)
                io.save_locs(locs_fp, locs, info)

                spinna_config[f"exp_data_{tag}"] = [locs_fp]
                spinna_config[f"le_{tag}"] = [
                    parameters["proposed_labeling_efficiency"]
                ]
                spinna_config[f"label_unc_{tag}"] = [
                    parameters["proposed_labeling_uncertainty"]
                ]
                spinna_config[f"n_simulated_{tag}"] = [
                    parameters["proposed_n_simulate"]
                ]
            spinna_config["res_factor"] = [100]
            spinna_config["save_filename"] = ["spinna_results"]
            spinna_config["nn_plotted"] = [parameters["proposed_nn_plotted"]]

            # bin size: more than Nyquist subsampling
            expected_1stNN_peak = (
                2 / (2 * d * np.pi * parameters["proposed_density"])
            ) ** (1 / d)
            spinna_config["fit_NND_bin"] = [expected_1stNN_peak / 10]
            spinna_config["density"] = parameters["proposed_density"]
            # max dist: a few times the first NN distance peak
            spinna_config["fit_NND_maxdist"] = [20 * expected_1stNN_peak]

            # save config to file
            pd.DataFrame.from_dict(spinna_config).to_csv(cfg_fp)

            msg = "This is a manual step. Please provide input, "
            msg += "and re-execute the workflow. "
            msg += f" The file {cfg_fp} has been prepared for you"
            msg += ", based on the parameters given."
            results["message"] = msg
            logger.debug(msg)
            print(msg)
            results["success"] = False
        else:
            # kick off SPINNA analysis
            print("starting spinna")
            result_dir, fp_summary, fp_fig = picasso_outpost.spinna_temp(
                cfg_fp
            )

            results["message"] = "Successfully performed SPINNA analysis."
            results["result_dir"] = result_dir
            results["fp_summary"] = fp_summary
            results["fp_fig"] = fp_fig
            results["success"] = True

        return parameters, results

    def _create_spinna_structure(
        self, names, multimers, distance, dimensionality=2
    ):
        """
        Args:
            names : list of str
                the names of proteins
            multimers : list of list of int
                for each name, the homo-multimers to implement
            distance : float
                distance between entities, in nm
        """
        spinna_structs = []
        for tag, name_multimers in zip(names, multimers):
            for n in name_multimers:
                # create positions on a cubic lattice
                positions = np.zeros((3, n))
                ux = np.array([1, 0, 0])
                uy = np.array([0, 1, 0])
                uz = np.array([0, 0, 1])
                edgelength = int(np.ceil(n ** (1 / dimensionality)))
                for i in range(n):
                    iz = i // (edgelength**2)
                    iy = i % (edgelength**2)
                    ix = i % edgelength
                    positions[:, i] = ix * ux + iy * uy + iz * uz
                positions[0, :] -= np.mean(positions[0, :])
                positions[1, :] -= np.mean(positions[1, :])
                positions[2, :] -= np.mean(positions[2, :])

                # create structure
                struct = {
                    "Molecular targets": [tag],
                    "Structure title": f"{tag}-{n}-mer",
                    f"{tag}_x": [float(x) for x in positions[0, :]],
                    f"{tag}_y": [float(x) for x in positions[1, :]],
                    f"{tag}_z": [float(x) for x in positions[2, :]],
                }
                spinna_structs.append(struct)
        return spinna_structs

    @module_decorator
    def ripleysk(self, i, parameters, results):
        """Perforn Ripley's K analysis between the channels.
        Args:
            parameters:
                ripleys_n_random_controls : int
                    number of random controls, default: 100
                ripleys_rmax : int
                    the maximum radius, default 200
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                fp_combined_locs : str
                    filepath to the combined locs of all channel_locs
        """
        nRandomControls = parameters.get("ripleys_n_random_controls", 100)
        radii = np.concatenate(
            (
                np.arange(10, 80, 2),
                np.arange(80, parameters.get("ripleys_rmax", 200), 12),
            )
        )

        if isinstance(parameters["fp_combined_locs"], list):
            fp_combined_locs = parameters["fp_combined_locs"][0]
        else:
            fp_combined_locs = parameters["fp_combined_locs"]
        combined_locs, _ = io.load_locs(fp_combined_locs)

        (ripleysResults, ripleysIntegrals) = (
            run_ripleysAnalysis.performRipleysMultiAnalysis(
                path=results["folder"],
                filename="",
                fileIDs=self.channel_tags,
                radii=radii,
                nRandomControls=nRandomControls,
                channel_locs=self.channel_locs,
                combined_locs=combined_locs,
                pixelsize=self.analysis_config["camera_info"].get("pixelsize"),
            )
        )

        results["fp_ripleys_integrals"] = os.path.join(
            results["folder"], "Ripleys_Integrals.txt"
        )
        np.savetxt(results["fp_ripleys_integrals"], ripleysIntegrals)

        results["fp_figintegrals"] = self._plot_ripleys_integrals(
            ripleysIntegrals, results["folder"], self.channel_tags
        )

        results["ripleys_significant"] = self._find_ripleys_significant(
            ripleysIntegrals,
            parameters["ripleys_threshold"],
            self.channel_tags,
        )

        return parameters, results

    def _plot_ripleys_integrals(self, ripleysIntegrals, folder, channel_tags):
        fig, ax = plt.subplots()
        heatmap = ax.imshow(
            ripleysIntegrals, cmap="coolwarm_r", vmin=-1, vmax=1
        )
        ax.grid(False)
        ax.set_xticks(np.arange(ripleysIntegrals.shape[0]))
        ax.set_yticks(np.arange(ripleysIntegrals.shape[1]))
        # Add number annotations to cells
        for i in range(ripleysIntegrals.shape[0]):
            for j in range(ripleysIntegrals.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{ripleysIntegrals[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    size=8,
                )
        ax.set_xticklabels(channel_tags, rotation=45)
        ax.set_yticklabels(channel_tags, rotation=45)
        ax.set_title("Ripleys Integrals")
        plt.colorbar(heatmap, format="%.2f")
        fp_integrals = os.path.join(folder, "RipIntegrals.png")
        fig.set_size_inches((9, 7))
        fig.savefig(fp_integrals)
        return fp_integrals

    def _find_ripleys_significant(
        self, ripleysIntegrals, threshold, channel_tags
    ):
        # elucidate significant pairs
        significant_pairs = []
        for i in range(len(channel_tags)):
            for j in range(i, len(channel_tags)):
                if ripleysIntegrals[i, j] > threshold:
                    significant_pairs.append(
                        (channel_tags[i], channel_tags[j])
                    )
        return significant_pairs

    @module_decorator
    def ripleysk_average(self, i, parameters, results):
        """Average the results of multiple Ripley's K Analyses, analyse
        the significant pairs after averaging, and save them into the
        separate workflow manual folders (for further analysis there)
        Args:
            parameters:
                # fp_ripleys_integrals : list of str
                #     the various single analyses to average, e.g. of
                #     different workflows
                fp_workflows : list of str
                    the paths to the folders of separate workflows
                    where the separate ripleys analyses have been done
                report_names : list of str
                    the report names of those worklfows
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                # output_folders : list of str
                #     folders to write the significant pairs into. This can
                #     e.g. be the 'manual' results folders of the
                #     workflows, so these can proceed.
            optional:
                swkfl_ripleysk_key : str
                    the results key of the ripleysk module.
                    e.g. '05_ripleysk'
                swkfl_manual_key : str
                    the results key of the manual module to save the
                    integrals to
                if those two are not given, saving is not performed
        """
        # from picasso_workflow.workflow import WorkflowRunner

        # all_integrals = np.concat(
        #     [np.loadtxt(fp) for fp in parameters["fp_ripleys_integrals"]])
        # averaged_integrals = np.mean(all_integrals, axis=0)

        # check single intregals based on workflow file
        fp_ripleys_integrals = []  # [""] * len(parameters["fp_workflows"])
        output_folders = []  # [""] * len(parameters["fp_workflows"])

        channel_tags = None
        search_dict = {
            (
                "swkfl_ripleysk_key",
                "fp_ripleys_integrals",
            ): fp_ripleys_integrals,
            ("swkfl_manual_key", "folder"): output_folders,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # for i, (wkflfolder, report_name) in enumerate(
        #     zip(parameters["fp_workflows"], parameters["report_names"])
        # ):
        #     # find analysis folder
        #     postfix = WorkflowRunner._check_previous_runner(
        #         wkflfolder, report_name
        #     )
        #     # find aggregation WorkflowRunner config
        #     fp_wr_cfg = os.path.join(
        #         wkflfolder,
        #         report_name + "_" + postfix,
        #         report_name + "_aggregation_" + postfix,
        #         "WorkflowRunner.yaml",
        #     )
        #     with open(fp_wr_cfg, "r") as f:
        #         data = yaml.load(f, Loader=yaml.FullLoader)
        #     # check for results of 'ripleysk' module
        #     for mod_key, mod_res in data["results"].items():
        #         if mod_key == parameters.get("swkfl_ripleysk_key"):
        #             print(mod_key, mod_res)
        #             print(parameters.get("swkfl_ripleysk_key"))
        #             fp_ripleys_integrals[i] = mod_res["fp_ripleys_integrals"]
        #         elif mod_key == parameters.get("swkfl_manual_key"):
        #             print(mod_key, mod_res)
        #             print(parameters.get("swkfl_manual_key"))
        #             output_folders[i] = mod_res["folder"]
        #     # find AggregationWorkflowRunner config
        #     fp_wr_cfg = os.path.join(
        #         wkflfolder,
        #         report_name + "_" + postfix,
        #         "AggregationWorkflowRunner.yaml",
        #     )
        #     with open(fp_wr_cfg, "r") as f:
        #         data = yaml.load(f, Loader=yaml.FullLoader)
        #     channel_tags = data["aggregation_workflow"][
        #         "single_dataset_tileparameters"
        #     ]["#tags"]
        # fp_ripleys_integrals = [fp for fp in fp_ripleys_integrals if fp != ""]
        # output_folders = [fp for fp in output_folders if fp != ""]
        results["output_folders"] = output_folders

        # load and average the integrals
        all_integrals = np.stack(
            [np.loadtxt(fp) for fp in fp_ripleys_integrals]
        )
        averaged_integrals = np.nanmean(all_integrals, axis=0)

        # save into own results folder
        results["fp_ripleys_integrals"] = os.path.join(
            results["folder"], "Ripleys_Integrals.txt"
        )
        np.savetxt(results["fp_ripleys_integrals"], averaged_integrals)

        results["fp_figintegrals"] = self._plot_ripleys_integrals(
            averaged_integrals, results["folder"], channel_tags
        )

        significant_pairs = self._find_ripleys_significant(
            averaged_integrals, parameters["ripleys_threshold"], channel_tags
        )
        results["ripleys_significant"] = significant_pairs

        # save significant pairs into given folders
        results["fp_ripleys_significant"] = os.path.join(
            results["folder"], "significant_pairs.txt"
        )
        save_fp = [
            os.path.join(fol, "significant_pairs.yaml")
            for fol in output_folders
        ]
        save_fp.append(results["fp_ripleys_significant"])
        for fp in save_fp:
            with open(fp, "w") as f:
                yaml.dump(significant_pairs, f)
            # np.savetxt(fp, significant_pairs)

        return parameters, results

    @module_decorator
    def protein_interactions(self, i, parameters, results):
        """Perform interaction analysis on those dataset pairs that showed
        significance in Ripley's K analysis. The interaction analysis consists
        of
        (1) calculating proportion of singly or doubly co-occurring instances
            of the single receptors (in clusters)
        (2) calculating the co-occurrence of these single or double events of
            one receptor with single or double events of another receptor
            within a cluster (where Ripley's K showed significance)
        This approach stems from a time of early development of SPINNA.
        Nowadays, this could be done directly but potentially with slightly
        different results.
        Fixed to 2D. Fixed to only using 1st nearest neighbor
        Args:
            parameters:
                channel_map : dict
                    maps between channels (protein names, tags before combining)
                    and index in the combine_id column of combined locs
                labeling_efficiency : dict, channel tag to float, range 0-100
                    labeling efficiency percentage, default for all targets
                labeling_uncertainty : dict, channel tag to float
                    labeling uncertainty [nm]; good value is e.g. 5
                n_simulate : int
                    number of target molecules to be simulated;
                    good value is e.g. 50000
                density : dict, channel tag to float
                    density to simulate [nm^2 or nm^3];
                    area density if 2D; volume density if 3D
                nn_nth : int
                    number of nearest neighbors to analyse
                structure_distance : float
                    the protein distance between each other in nm
                res_factor : float
                    the spinna res_factor
                sim_repeats : int
                    number of simulation repeats, for noise reduction
                interaction_pairs: list of list of two strings, or str
                    pairs that are able to interact
                    if str: filepath to a yaml file with list of tuples
        """
        from picasso_workflow.spinna_main import load_structures_from_dict

        logger.debug("Molecular interactions")

        # # homo-analysis (proportions of 1- or 2-mers of the same kind)
        # props = {}
        dimensionality = 2
        pixelsize = self.analysis_config["camera_info"].get("pixelsize")
        if isinstance(parameters["density"], list):
            density = {
                tag: parameters["density"][cid]
                for tag, cid in parameters["channel_map"].items()
            }
        elif isinstance(parameters["density"], dict):
            density = parameters["density"]
        else:
            raise KeyError("density parameter must be list of dict.")
        # compound_density = sum(parameters["density"].values())
        # area = parameters["n_simulate"] / (compound_density / 1e6)
        # n_sim_targets = {
        #     tag:
        #     int(parameters["n_simulate"] * compound_density / den)
        #     for tag, den in parameters["density"].items()
        # }
        # pixelsize = self.analysis_config["camera_info"].get("pixelsize")
        # structures = self._create_spinna_structure(
        #     self.channel_tags, [1, 2], distance=parameters["distance"])
        # N_structures = picasso_outpost.generate_N_structures(
        #     structures, n_sim_targets, parameters["res_factor"]
        # )
        # # bin size: more than Nyquist subsampling
        # expected_1stNN_peak = (
        #     2 / (2 * dimensionality * np.pi * parameters["density"])
        # ) ** (1 / dimensionality)
        # fit_NND_bin = expected_1stNN_peak / 10
        # # max dist: a few times the first NN distance peak
        # fit_NND_maxdist = 20 * expected_1stNN_peak
        # for tag, locs in zip(self.channel_tags, self.channel_locs):
        #     spinna_parameters = {
        #         "structures": self._create_spinna_structure(
        #             [tag], [[1, 2]], parameters["structure_distance"]),
        #         "label_unc": parameters["labeling_uncertainty"],
        #         "le": parameters["labeling_efficiency"],
        #         "mask_dict": None,
        #         "width": np.sqrt(area * 1e6),
        #         "height": np.sqrt(area * 1e6),
        #         "depth": None,
        #         "random_rot_mode": "2D",
        #         "exp_data": {tag: np.stack((locs[['x', 'y']] * pixelsize))},
        #         "sim_repeats": parameters["sim_repeats"],
        #         "fit_NND_bin": [fit_NND_bin],
        #         "fit_NND_maxdist": [fit_NND_maxdist],
        #         "N_structures": N_structures,
        #         "save_filename": os.path.join(results["folder"], "homo-{tag}"),
        #         "asynch": True,
        #         "targets": [tag],
        #         "apply_mask": False,
        #         "nn_plotted": parameters["nn_nth"],
        #         "result_dir": results["folder"],
        #     }
        #     result, fp_fig = picasso_outpost.spinna_sgl_temp(spinna_parameters)
        #     props[tag] = result["Fitted proportions of structures"]
        # logger.debug(f'found proportions of {props}')

        if isinstance(parameters["interaction_pairs"], str):
            with open(parameters["interaction_pairs"], "r") as f:
                interaction_pairs = yaml.safe_load(f)
        else:
            interaction_pairs = parameters["interaction_pairs"]
            # np.savetxt(fp, significant_pairs)

        # hetero-analysis (pairwise up to 2+2-mers)
        # structures: A, B, AA, BB, AB, AABB
        props = {}
        for A, B in interaction_pairs:
            if A == B:  # or should we include homotetramers?
                continue
            logger.debug(
                f"analysing interaction between {A} and {B} with SPINNA."
            )
            # find index of A and B in self.channel_locs
            ia = self.channel_tags.index(A)
            ib = self.channel_tags.index(B)

            # locs, but as np.ndarray
            exp_data = {}
            for i, target in zip([ia, ib], [A, B]):
                locs = self.channel_locs[i]
                if hasattr(locs, "z"):
                    exp_data[target] = np.stack(
                        (locs.x * pixelsize, locs.y * pixelsize, locs.z)
                    ).T
                    # dim = 3
                else:
                    exp_data[target] = np.stack(
                        (locs.x * pixelsize, locs.y * pixelsize)
                    ).T
                    # dim = 2
            structures = self._create_spinna_structure(
                [A], [[1, 2]], parameters["structure_distance"]
            )
            structures += self._create_spinna_structure(
                [B], [[1, 2]], parameters["structure_distance"]
            )
            # heterodimer
            struct = {
                "Molecular targets": [A, B],
                "Structure title": f"{A}-{B}-heterodimer",
                f"{A}_x": [-parameters["structure_distance"] / 2],
                f"{A}_y": [0],
                f"{A}_z": [0],
                f"{B}_x": [parameters["structure_distance"] / 2],
                f"{B}_y": [0],
                f"{B}_z": [0],
            }
            structures.append(struct)
            # heterotetramer, in a square
            struct = {
                "Molecular targets": [A, B],
                "Structure title": f"{A}-{B}-heterotetramer",
                f"{A}_x": [
                    -parameters["structure_distance"] / 2,
                    parameters["structure_distance"] / 2,
                ],
                f"{A}_y": [
                    -parameters["structure_distance"] / 2,
                    -parameters["structure_distance"] / 2,
                ],
                f"{A}_z": [0, 0],
                f"{B}_x": [
                    -parameters["structure_distance"] / 2,
                    parameters["structure_distance"] / 2,
                ],
                f"{B}_y": [
                    parameters["structure_distance"] / 2,
                    parameters["structure_distance"] / 2,
                ],
                f"{B}_z": [0, 0],
            }
            structures.append(struct)

            compound_density = density[A] + density[B]
            area = parameters["n_simulate"] / (compound_density / 1e6)
            n_sim_targets = {
                tag: int(
                    parameters["n_simulate"] * compound_density / density[tag]
                )
                for tag in [A, B]
            }
            structures, targets = load_structures_from_dict(structures)

            N_structures = picasso_outpost.generate_N_structures(
                structures, n_sim_targets, parameters["res_factor"]
            )

            # bin size: more than Nyquist subsampling
            expected_1stNN_peak = (
                2 / (2 * dimensionality * np.pi * (compound_density / 2))
            ) ** (1 / dimensionality)
            fit_NND_bin = expected_1stNN_peak / 10
            # max dist: a few times the first NN distance peak
            fit_NND_maxdist = 20 * expected_1stNN_peak

            spinna_parameters = {
                "structures": structures,
                "label_unc": parameters["labeling_uncertainty"],
                "le": parameters["labeling_efficiency"],
                "mask_dict": None,
                "width": np.sqrt(area * 1e6),
                "height": np.sqrt(area * 1e6),
                "depth": None,
                "random_rot_mode": "2D",
                "exp_data": exp_data,
                "sim_repeats": parameters["sim_repeats"],
                "fit_NND_bin": fit_NND_bin,
                "fit_NND_maxdist": fit_NND_maxdist,
                "N_structures": N_structures,
                "save_filename": os.path.join(
                    results["folder"], f"interaction-{A}-{B}"
                ),
                "asynch": True,
                "targets": [A, B],
                "apply_mask": False,
                "nn_plotted": parameters["nn_nth"],
                "result_dir": results["folder"],
            }

            result, fp_fig = picasso_outpost.spinna_sgl_temp(spinna_parameters)
            props[(A, B)] = result["Fitted proportions of structures"]
        logger.debug(f"proportions: {props}")
        results["Interaction proportions"] = props

        return parameters, results

    @module_decorator
    def create_mask(self, i, parameters, results):
        """
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
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
        from picasso_workflow.dbscan_molint import mask

        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)
        # get Mask
        locs = self.channel_locs[0]
        multi_filename = "multi_ID.hdf5"
        pixelsize = self.analysis_config["camera_info"].get("pixelsize")
        mask_dict = mask.gen_mask(
            locs["x"],
            locs["y"],
            parameters["margin"],
            parameters["binsize"],
            parameters["sigma_mask_blur"],
            parameters["mask_resolution"],
            pixelsize,
            results["folder"],
            filename=multi_filename,
            plot_figures=True,
        )

        # get exp coordinates in mask
        new_info = self.channel_info[0] + [
            {
                "Generated by": "picasso-workflow: create_mask",
            }
        ]
        self.channel_info[0] = new_info
        df_merge_mask, mask_dict = mask.exp_data_in_mask(
            pd.DataFrame(locs),
            mask_dict,
            pixelsize,
            results["folder"],
            multi_filename,
            new_info,
            plot_figures=True,
        )

        results["fp_merge_mask"] = os.path.join(
            results["folder"], "merge_mask.hdf5"
        )
        df_merge_mask.to_hdf(results["fp_merge_mask"], key="locs")

        # Get densities of individual proteins:
        N_proteins = df_merge_mask.groupby(parameters["combine_col"]).size()

        for protein, protein_ID in channel_map.items():
            N = N_proteins.loc[protein_ID]
            area = mask_dict["area"]
            density = N / area

            mask_dict["N_exp_" + protein] = N
            mask_dict["density_exp_" + protein + " (/um^2)"] = density

        mask_dict["info"] = self.channel_info[0]
        mask_dict["filename"] = results["fp_merge_mask"]

        fp_mask_dict = os.path.join(results["folder"], "mask_dict.pkl")
        results["fp_mask_dict"] = fp_mask_dict
        with open(fp_mask_dict, "wb+") as f:
            pickle.dump(mask_dict, f)

        return parameters, results

    @module_decorator
    def dbscan_molint(self, i, parameters, results):
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
        # from picasso_workflow.dbscan_molint import dbscan
        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)

        pixelsize = self.analysis_config["camera_info"].get("pixelsize")
        epsilon_nm = parameters["epsilon_nm"]
        df_mask = pd.read_hdf(parameters["fp_merge_mask"], key="locs")
        fp_out_base = os.path.join(results["folder"], "dbscan.hdf5")
        filepaths = self._do_dbscan_molint(
            results["folder"],
            fp_out_base,
            df_mask,
            self.channel_info[0],
            pixelsize,
            epsilon_nm,
            parameters["minpts"],
            parameters["sigma_linker"],
            parameters["thresh_type"],
            parameters["cell_name"],
            channel_map,
        )
        for k, v in filepaths.items():
            results[k] = v

        return parameters, results

    def _do_dbscan_molint(
        self,
        result_folder,
        fp_out_base,
        df_mask,
        info,
        pixelsize,
        epsilon_nm,
        minpts,
        sigma_linker,
        thresh_type,
        cell_name,
        channel_map,
        it=0,
    ):
        from picasso_workflow.dbscan_molint import dbscan

        filepaths = {}

        if thresh_type == "area":
            # Analysis will also be performed seperatetly for clusters
            # larger or equal than
            # area thresh and clusters smaller than thresh
            thresh = 10000  # nm^2
        elif thresh_type == "density":
            # Analysis will also be performed seperatetly for clusters
            # with densities
            # larger or equal than density thresh
            # area thresh and clusters smaller than thresh
            thresh = 100  # molecules / um^2
            # Change unit to molecules / nm^2
            thresh = thresh / 1000 / 1000

        epsilon_px = epsilon_nm / pixelsize
        sigma_linker_px = sigma_linker / pixelsize

        # DBSCAN on exp data
        new_info = {
            "Generated by": "picasso-workflow: DBSCAN-MOLECULAR INTERACTIONS",
            "epsilon": epsilon_px,
            "minpts": minpts,
            # 'Number of clusters"
        }
        info.append(new_info)
        (
            db_locs_rec,
            db_locs_rec_protein_colorcoding,
            db_cluster_props_rec,
            db_locs_df,
            db_cluster_props_df,
        ) = dbscan.dbscan_f(df_mask, epsilon_px, minpts, sigma_linker_px)

        # save locs in dbscan cluster with colorcoding = dbcluster ID
        dbscan_fp = os.path.join(
            result_folder,
            f"dbscan_{epsilon_nm:.0f}_{minpts}_{it}.hdf5",
        )
        io.save_locs(dbscan_fp, db_locs_rec, info)
        filepaths["fp_dbscan_color-cluster"] = dbscan_fp

        # save locs in dbscan cluster with colorcoding = protein ID
        dbscan_fp = os.path.join(
            result_folder,
            f"dbscan_{epsilon_nm:.0f}_{minpts}_{it}"
            + "_protein_colorcode.hdf5",
        )
        io.save_locs(dbscan_fp, db_locs_rec_protein_colorcoding, info)
        filepaths["fp_dbscan_color-protein"] = dbscan_fp

        # save properties of dbscan clusters
        # (analygously to DBSCAN output in Picasso)
        dbclusters_fp = os.path.join(
            result_folder,
            f"dbclusters_{epsilon_nm:.0f}_{minpts}_{it}.hdf5",
        )
        # print(db_cluster_props_rec)
        # print(db_cluster_props_rec.dtype)
        # THE FOLLOWING LINE DOES NOT WORK BECAUSE db_cluster_props_rec
        # does not have x, y, lpx, or lpy. fails picasso sanity checks.
        # io.save_locs(dbclusters_fp, db_cluster_props_rec, info)
        db_cluster_props_df.to_hdf(dbclusters_fp, key="props")
        filepaths["fp_dbclusters"] = dbclusters_fp

        from picasso_workflow.dbscan_molint import output_metrics

        """
        ===============================================================================
        Output of all clusters in one cell
        ===============================================================================
        """

        # output for each cluster = [N_per_cluster, area, circularity,
        #                            N_CD80, ... % CD80, ...]

        # Calculate and save output metrics for all clusters in the cell
        (
            cluster_filename,
            cluster_large_filename,
            cluster_small_filename,
            db_cluster_output,
        ) = output_metrics.output_cell(
            channel_map,
            db_locs_df,
            db_cluster_props_df,
            fp_out_base,
            pixelsize,
            epsilon_nm,
            minpts,
            thresh,
            thresh_type,
            cell_name,
        )

        filepaths["fpoutput_all_clusters"] = cluster_filename
        filepaths["fpoutput_large_clusters"] = cluster_large_filename
        filepaths["fpoutput_small_clusters"] = cluster_small_filename
        # stimulation_cluster_exp_dict[cell_name] = db_cluster_output
        # stimulation_cluster_exp_large_dict[cell_name] = db_cluster_output_large
        # stimulation_cluster_exp_small_dict[cell_name] = db_cluster_output_small

        # perform Rafal's analysis (binary barcode)
        barcodes, weights = picasso_outpost.DBSCAN_analysis(db_cluster_output)
        filepaths["fp_binary_barcode"] = os.path.join(
            result_folder, "binary_barcode.txt"
        )
        np.savetxt(filepaths["fp_binary_barcode"], barcodes)
        filepaths["fp_binary_barcode_weights"] = os.path.join(
            result_folder, "binary_barcode_weights.txt"
        )
        np.savetxt(filepaths["fp_binary_barcode_weights"], weights)

        """
        ===============================================================================
        mean output of one cell
        ===============================================================================
        """
        # output for each cluster = [N_per_cluster, area, circularity,
        #                            N_CD80, ... % CD80, ...]
        # output for complete cell: [
        #   N_in_cell, N_in_clusters, N_out_clusters, N_per_cluster_mean,
        #   N_per_cluster_CI,, area_mean, area_CI, circularity_mean,
        #   circularity_CI,, N_CD80,, N_CD80_in_clusters, N_CD86_out_clusters,
        #   ... , N_CD80_mean, N_CD80_CI, ...., %_CD80_mean, %_CD80_CI, ....]

        # Calculate and save mean of output metrics of all clusters in the cell
        # + some output metrics specific to the whole cell
        (mean_filename, mean_large_filename, mean_small_filename) = (
            output_metrics.output_cell_mean(
                channel_map,
                df_mask,
                db_locs_df,
                db_cluster_output,
                fp_out_base,
                pixelsize,
                epsilon_nm,
                minpts,
                thresh,
                thresh_type,
                cell_name,
            )
        )

        filepaths["fpoutput_mean_all"] = mean_filename
        filepaths["fpoutput_mean_large"] = mean_large_filename
        filepaths["fpoutput_mean_small"] = mean_small_filename
        # stimulation_exp_dict[cell_name] = db_cell_output
        # stimulation_exp_large_dict[cell_name] = db_cell_output_large
        # stimulation_exp_small_dict[cell_name] = db_cell_output_small

        return filepaths

    def _load_other_workflow_data(self, fp_workflow, report_name, search_keys):
        """Load result data from a different workflow
        Args:
            fp_workflow : str
                the root folder of the other workflow
            report_name : str
                the report name of the other workflow. The
                workflow result data will be in
                fp_workflow/report_name_[postfix]
            search_keys : tuple of
                1st : str
                    the module keys (e.g. '04_manual')
                2nd : str
                    the result entries (e.g. 'filepath')
        Returns:
            loaded_data : dict
                keys : tuple of (str, str)
                    tuple of search key & value
                values : the corresponding loaded data
            channel_tags : list of str
                the channel tags
        """
        from picasso_workflow.workflow import WorkflowRunner

        loaded_data = []

        # find analysis folder
        postfix = WorkflowRunner._check_previous_runner(
            fp_workflow, report_name
        )
        # find aggregation WorkflowRunner config
        fp_wr_cfg = os.path.join(
            fp_workflow,
            report_name + "_" + postfix,
            report_name + "_aggregation_" + postfix,
            "WorkflowRunner.yaml",
        )
        with open(fp_wr_cfg, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # check for results of 'ripleysk' module
        for mod_key, mod_res in data["results"].items():
            for search_mod, search_res in search_keys:
                if mod_key == search_mod:
                    res = mod_res[search_res]
                    loaded_data[(search_mod, search_res)] = res
        # find AggregationWorkflowRunner config
        fp_wr_cfg = os.path.join(
            fp_workflow,
            report_name + "_" + postfix,
            "AggregationWorkflowRunner.yaml",
        )
        with open(fp_wr_cfg, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        channel_tags = data["aggregation_workflow"][
            "single_dataset_tileparameters"
        ]["#tags"]

        return loaded_data, channel_tags

    @module_decorator
    def dbscan_merge_cells(self, i, parameters, results):
        """Merge dataframes containing the cluster properties of
        all cells of the current stimulation
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
                    swkfl_dbscan_key : str
                        the results key of the dbscan module.
                        e.g. '09_dbscan_molint' or '10_CSR_sim_in_mask'
                    stimulation_key : str
                        an identifier of the current stimulation
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        fp_all_clusters = []
        fp_large_clusters = []
        fp_small_clusters = []
        fp_mean_all = []
        fp_mean_large = []
        fp_mean_small = []
        channel_tags = None
        dbscan_key = parameters["swkfl_dbscan_key"]
        search_keys = [
            (dbscan_key, "fp_all_clusters"),
            (dbscan_key, "fp_large_clusters"),
            (dbscan_key, "fp_small_clusters"),
            (dbscan_key, "fp_mean_all"),
            (dbscan_key, "fp_mean_large"),
            (dbscan_key, "fp_mean_small"),
        ]
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_keys
            )
            if fp := loaded_data.get((dbscan_key, "fp_all_clusters")):
                fp_all_clusters.append(fp)
            if fp := loaded_data.get((dbscan_key, "fp_large_clusters")):
                fp_large_clusters.append(fp)
            if fp := loaded_data.get((dbscan_key, "fp_small_clusters")):
                fp_small_clusters.append(fp)
            if fp := loaded_data.get((dbscan_key, "fp_mean_all")):
                fp_mean_all.append(fp)
            if fp := loaded_data.get((dbscan_key, "fp_mean_large")):
                fp_mean_large.append(fp)
            if fp := loaded_data.get((dbscan_key, "fp_mean_small")):
                fp_mean_small.append(fp)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # Create one output csv file containing the clusters'
        # properties for each cell of this stimulation

        # all clusters
        results["fp_merged_all_clusters"] = os.path.join(
            results["folder"], "merged_all_clusters.csv"
        )
        _ = self._merge_df_csvs(
            fp_all_clusters, results["fp_merged_all_clusters"]
        )

        # large clusters
        results["fp_merged_large_clusters"] = os.path.join(
            results["folder"], "merged_large_clusters.csv"
        )
        _ = self._merge_df_csvs(
            fp_large_clusters, results["fp_merged_large_clusters"]
        )

        # small clusters
        results["fp_merged_small_clusters"] = os.path.join(
            results["folder"], "merged_small_clusters.csv"
        )
        _ = self._merge_df_csvs(
            fp_small_clusters, results["fp_merged_small_clusters"]
        )

        # Merge dataframes containing cell means of this stimulation

        # all clusters
        results["fp_merged_mean_all"] = os.path.join(
            results["folder"], "merged_mean_all.csv"
        )
        df_merged_mean_all = self._merge_df_csvs(
            fp_mean_all, results["fp_merged_mean_all"]
        )

        # large clusters
        results["fp_merged_mean_large"] = os.path.join(
            results["folder"], "merged_mean_large.csv"
        )
        df_merged_mean_large = self._merge_df_csvs(
            fp_mean_large, results["fp_merged_mean_large"]
        )

        # small clusters
        results["fp_merged_mean_small"] = os.path.join(
            results["folder"], "merged_mean_small.csv"
        )
        df_merged_mean_small = self._merge_df_csvs(
            fp_mean_small, results["fp_merged_mean_small"]
        )

        # create summaries of the merged_mean data
        from picasso_workflow.dbscan_molint import output_metrics

        summary_all = output_metrics.output_stimulation_mean(
            channel_tags,
            df_merged_mean_all,
            parameters["stimulation_key"],
        )
        fp_summary_all = os.path.join(results["folder"], "summary_all.xlsx")
        summary_all.to_excel(fp_summary_all)
        results["fp_summary_all"] = fp_summary_all

        summary_large = output_metrics.output_stimulation_mean(
            channel_tags,
            df_merged_mean_large,
            parameters["stimulation_key"],
        )
        fp_summary_large = os.path.join(
            results["folder"], "summary_large.xlsx"
        )
        summary_large.to_excel(fp_summary_large)
        results["fp_summary_large"] = fp_summary_large

        summary_small = output_metrics.output_stimulation_mean(
            channel_tags,
            df_merged_mean_small,
            parameters["stimulation_key"],
        )
        fp_summary_small = os.path.join(
            results["folder"], "summary_small.xlsx"
        )
        summary_small.to_excel(fp_summary_small)
        results["fp_summary_small"] = fp_summary_small

        # summary_exp_large_dict[stimulation_key
        # ] = output_metrics.output_stimulation_mean(
        #    channel_ID, df_cell_output_large_merge, stimulation_key)
        # summary_exp_small_dict[stimulation_key
        # ] = output_metrics.output_stimulation_mean(
        #    channel_ID, df_cell_output_small_merge, stimulation_key)

        return parameters, results

    def _merge_df_csvs(self, fps, fp_out):
        """
        Merge multiple pd.DataFrames saved as csv. Subsequently,
        save the result as csv and pkl.
        Args:
            fps : list of str
                filepaths of the csvs to merge
            fp_out : str
                the output csv file path
        Returns:
            df : pd.DataFrame
                the merged DataFrame
        """
        df_merged = pd.concat([pd.read_csv(fp) for fp in fps])
        df_merged.to_csv(fp_out)
        df_merged.to_pickle(fp_out.replace(".csv", ".pkl"))
        return df_merged

    @module_decorator
    def dbscan_merge_stimulations(self, i, parameters, results):
        """Merge dataframes containing the information of
        all stimulations of the current experiment
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_summary_all : list of str
                        filepaths to 'summary_all' csv DataFrame
                        from 'dbscan_merge_cells' module
                        of one stimulation each, for multiple stimulations
                    fp_summary_large : list of str
                        filepaths to 'summary_large' csv DataFrame
                        from 'dbscan_merge_cells' module
                        of one stimulation each, for multiple stimulations
                    fp_summary_small : list of str
                        filepaths to 'summary_small' csv DataFrame
                        from 'dbscan_merge_cells' module
                        of one stimulation each, for multiple stimulations
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        # all clusters
        results["fp_merged_summary_all"] = os.path.join(
            results["folder"], "merged_summary_all.csv"
        )
        _ = self._merge_df_csvs(
            parameters["fp_summary_all"], results["fp_merged_summary_all"]
        )
        # large clusters
        results["fp_merged_summary_large"] = os.path.join(
            results["folder"], "merged_summary_large.csv"
        )
        _ = self._merge_df_csvs(
            parameters["fp_summary_large"], results["fp_merged_summary_large"]
        )
        # small clusters
        results["fp_merged_summary_small"] = os.path.join(
            results["folder"], "merged_summary_small.csv"
        )
        _ = self._merge_df_csvs(
            parameters["fp_summary_small"], results["fp_merged_summary_small"]
        )

        return parameters, results

    @module_decorator
    def CSR_sim_in_mask(self, i, parameters, results):
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
        from picasso_workflow.dbscan_molint import mask

        # from picasso_workflow.dbscan_molint import dbscan

        pixelsize = self.analysis_config["camera_info"].get("pixelsize")
        epsilon_nm = parameters["epsilon_nm"]

        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)
        with open(parameters["fp_mask_dict"], "rb") as f:
            mask_dict = pickle.load(f)
        info = mask_dict["info"]
        # filename_base = mask_dict['filename']

        all_filepaths = []
        for s in range(1, parameters["N_repeats"] + 1):
            # print()
            # print('repeat', s)
            # CSR simulation in mask:
            #     first: for each channel
            #     second: create multi file

            filename = os.path.join(
                results["folder"], f"CSR_in_mask_rep_{s}.hdf5"
            )
            df_CSR_mask, info_CSR = mask.CSR_sim_in_mask_multi_channel(
                channel_map,
                mask_dict,
                pixelsize,
                results["folder"],
                filename,
                info,
                plot_figures=True,
            )
            fp_out_base = os.path.join(results["folder"], f"dbscan_{s}.hdf5")
            filepaths = self._do_dbscan_molint(
                results["folder"],
                fp_out_base,
                df_CSR_mask,
                info,
                pixelsize,
                epsilon_nm,
                parameters["minpts"],
                parameters["sigma_linker"],
                parameters["thresh_type"],
                parameters["cell_name"],
                channel_map,
                it=s,
            )
            all_filepaths.append(filepaths)
        # re-organize: save the list of filepath dicts as
        # different dict values of lists of strings
        for k in all_filepaths[0].keys():
            results[k] = [fp[k] for fp in all_filepaths]

        # results["fp_dbscan_color-cluster"] = dbscan_colcluster_fp
        # results["fp_dbscan_color-protein"] = dbscan_colprot_fp
        # results["fp_dbclusters"] = dbclusters_fp

        return parameters, results

    @module_decorator
    def binary_barcodes(self, i, parameters, results):
        """Analyses the binary barcode results of _do_dbscan_molint.
        Compares experimental to CSR data.
        Merged for multiple cells
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
                    swkfl_dbscan_molint_key : str
                        the results key of the dbscan module.
                        e.g. '09_dbscan_molint'
                    swkfl_CSR_sim_in_mask_key : str
                        the results key of the CSR dbscan module.
                        e.g. '10_CSR_sim_in_mask'
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        from matplotlib.ticker import FormatStrFormatter

        # use fpoutput_all_clusters of _do_dbscan_molint, from output_metrics.

        # of _do_dbscan_molint
        # fp_binary_barcode
        # fp_binary_barcode_weights
        fp_binbc_exp = []  # will be a list of strings (1 for each cell)
        fp_binbc_weights_exp = []  # same as above
        fp_binbc_csr = []  # will be list of list of strings as in each
        fp_binbc_weights_csr = []  # workflow, multiple CSR sims are done
        channel_tags = None
        exp_key = parameters["swkfl_dbscan_molint_key"]
        csr_key = parameters["swkfl_CSR_sim_in_mask_key"]
        search_keys = [
            (exp_key, "fp_binary_barcode"),
            (exp_key, "fp_binary_barcode_weights"),
            (csr_key, "fp_binary_barcode"),
            (csr_key, "fp_binary_barcode_weights"),
        ]
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_keys
            )
            if fp := loaded_data.get((exp_key, "fp_binary_barcode")):
                fp_binbc_exp.append(fp)
            if fp := loaded_data.get((exp_key, "fp_binary_barcode_weights")):
                fp_binbc_weights_exp.append(fp)
            if fp := loaded_data.get((csr_key, "fp_binary_barcode")):
                fp_binbc_csr.append(fp)
            if fp := loaded_data.get((csr_key, "fp_binary_barcode_weights")):
                fp_binbc_weights_csr.append(fp)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        all_bc_list = [
            "{:0{}b}".format(i, len(channel_tags))
            for i in range(len(channel_tags**2 + 1))
        ][1:]

        # take the mean across the cell
        res_cell = np.zeros(  # contains counts for each cell
            (len(parameters["fp_workflows"]), len(all_bc_list))
        )
        res_csr = np.zeros(  # contains counts for each csr simulation
            (len(parameters["fp_workflows"]), len(all_bc_list))
        )
        count = 0
        for fp_bc_exp, fp_bc_csr, fp_weights_exp, fp_weights_csr in zip(
            fp_binbc_exp,
            fp_binbc_csr,
            fp_binbc_weights_exp,
            fp_binbc_weights_csr,
        ):
            barcodes_cell = np.loadtxt(fp_bc_exp)
            barcodes_csr = np.loadtxt(fp_bc_csr)
            weights_cell = np.loadtxt(fp_weights_exp)
            weights_csr = np.loadtxt(fp_weights_csr)
            # get the weighted counts for cell
            b = ["".join(_.astype(str)) for _ in barcodes_cell]
            for weight, barcode in zip(weights_cell, b):
                idx = all_bc_list.index(barcode)
                res_cell[count, idx] += weight
            res_cell[count, :] /= res_cell[count, :].sum()

            # get the weighted counts for csr
            b = ["".join(_.astype(str)) for _ in barcodes_csr]
            for weight, barcode in zip(weights_csr, b):
                idx = all_bc_list.index(barcode)
                res_csr[count, idx] += weight
            res_csr[count, :] /= res_csr[count, :].sum()

            count += 1

        # plot
        x = np.arange(len(all_bc_list))
        width = 0.4
        fig = plt.figure(figsize=(10, 4), constrained_layout=True)

        # take mean and error
        cell_mean = res_cell.mean(axis=0)
        cell_std = res_cell.std(axis=0)
        cell_err = (
            1.96 * cell_std / np.sqrt(len(parameters["fp_workflows"]))
        )  # 95% confidence interval
        csr_mean = res_csr.mean(axis=0)
        csr_std = res_csr.std(axis=0)
        csr_err = (
            1.96 * csr_std / np.sqrt(len(parameters["fp_workflows"]))
        )  # 95% confidence interval

        # save as txt
        results["fp_cell_mean"] = os.path.join(
            results["folder"], "cell_mean.txt"
        )
        np.savetxt(results["fp_cell_mean"], cell_mean)
        results["fp_cell_err"] = os.path.join(
            results["folder"], "cell_err.txt"
        )
        np.savetxt(results["fp_cell_err"], cell_err)
        results["fp_csr_mean"] = os.path.join(
            results["folder"], "csr_mean.txt"
        )
        np.savetxt(results["fp_csr_mean"], csr_mean)
        results["fp_csr_err"] = os.path.join(results["folder"], "csr_err.txt")
        np.savetxt(results["fp_csr_err"], csr_err)

        # frequency bar plot
        plt.bar(
            x - width / 2,
            cell_mean,
            yerr=cell_err,
            width=width,
            edgecolor="black",
            facecolor="lightgray",
            label="Cell",
        )
        plt.bar(
            x + width / 2,
            csr_mean,
            yerr=csr_err,
            width=width,
            edgecolor="black",
            facecolor="dimgrey",
            label="CSR",
        )

        plt.xticks(
            np.arange(len(all_bc_list)),
            labels=all_bc_list,
            rotation=90,
            fontsize=8,
        )
        plt.ylabel("Weighted counts", fontsize=12)
        plt.xlabel("Barcodes", fontsize=12)
        plt.legend()
        fig.axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        # if title is not None:
        #     plt.title(title, fontsize=20)

        results["fp_fig"] = os.path.join(
            results["folder"], "binary_barcodes.png"
        )
        savename = os.path.splitext(results["fp_fig"])[0]
        plt.savefig(savename + ".png", dpi=300, transparent=True)
        plt.savefig(savename + ".svg")

        return parameters, results


class AutoPicassoError(Exception):
    pass


class ManualInputLackingError(AutoPicassoError):
    pass


class PicassoConfigError(AutoPicassoError):
    pass

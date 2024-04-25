#!/usr/bin/env python
"""
Module Name: analyse.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This is the picasso interface of picasso-workflow
"""
from picasso import io, localize, gausslq, postprocess, clusterer
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
import logging
from datetime import datetime

from picasso_workflow.util import AbstractModuleCollection
from picasso_workflow import process_brightfield
from picasso_workflow import picasso_outpost


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
        parameters, results = method(self, i, parameters, results)
        # modules only need to specifically set an error.
        if results.get("success") is None:
            results["success"] = True
            # save locs if desired
            if parameters.get("save_locs") is True:
                self._save_locs(os.path.join(results["folder"], "locs.hdf5"))
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
    info = None
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
                    camera_info : dict
                        as used by picasso. Only necessary if not loaded by
                        module load_dataset
        """
        self.results_folder = results_folder
        self.analysis_config = analysis_config

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
            "Radius": parameters["radius"],
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
                        the smlm radius
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
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        clusters = np.rec.array(self.locs, dtype=self.locs.dtype)
        points = np.array(clusters[["com_x", "com_y"]].tolist())
        alldist = distance.cdist(points, points)
        alldist[alldist == 0] = float("inf")
        minvals = np.amin(alldist, axis=0)
        out_path = os.path.join(results["folder"], "minval.txt")
        np.savetxt(out_path, minvals, newline="\r\n")

        return parameters, results

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
        for fp, tag in zip(parameters["filepaths"], parameters["tags"]):
            locs, info = io.load_locs(fp)
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
        allfps = []
        for locs, info, tag in zip(
            self.channel_locs, self.channel_info, self.channel_tags
        ):
            filepath = os.path.join(results["folder"], tag + ".hdf5")
            io.save_locs(filepath, locs, info)
            allfps.append(filepath)
        results["filepaths"] = allfps

        return parameters, results


class AutoPicassoError(Exception):
    pass


class ManualInputLackingError(AutoPicassoError):
    pass


class PicassoConfigError(AutoPicassoError):
    pass

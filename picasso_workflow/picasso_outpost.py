#!/usr/bin/env python
"""
Module Name: picasso_outpost.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: This is a collection of exploratory DNA-PAINT analysis / picasso
    related functions which if useful should (potentially) be moved into the
    next picasso release. The reasoning to put them here is that it makes
    testing cycles faster.
"""
# import logging
from loguru import logger
import numpy as np

# from numpy.lib.recfunctions import stack_arrays
import pandas as pd
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
import yaml
import os
from datetime import datetime
from aicsimageio import AICSImage

from picasso import (
    io,
    localize,
    render,
    imageprocess,
    postprocess,
    lib,
    spinna,
)
from picasso.__main__ import _spinna_batch_analysis as spinna_batch_analysis
from picasso_workflow import util

from scipy.spatial import KDTree
from scipy.special import gamma as _gamma
from scipy.special import factorial as _factorial
from scipy.optimize import minimize
from sklearn.cluster import OPTICS
from scipy import stats
import itertools
import glob


# logger = logging.getLogger(__name__)


def align_channels(
    channel_locs,
    channel_info,
    channel_tags=None,
    max_iterations=5,
    convergence=0.001,
    fiducial_locs=None,
    force_method=None,
    max_shift=None,
    plot_histogram=False,
    plot_dir=None,
):
    """This is taken from picasso.gui.render.View.align. As the code is not
    modular enough, it is replicated here. Potentially, this could go into
    a non-gui function in picasso.
    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels
        max_iterations : int
            the maximum number of iterations of alignment
        convergence : float
            convergence criterium when a shift is negligible and thus
            alignment convergence achieved. The value is in pixels.
        fiducial_locs : list of recarray
            the localizations to use as a basis for the alignment. If None,
            the channel_locs are used as fiducials.
        force_method : None or str
            "RCC": force usage of RCC algorithm, also with fiducials present.
            "picked": force usage of 'by picked' algorithm
            "filtered_RCC": force usage of filtered RCC algorithm based on
            shift histograms
        max_shift : float
            the maximum shift between picks, if undifting fiducials by picked,
            or the maximum expected shift for filtered_RCC alignment
        plot_histogram : bool, default False
            Whether to save 2D histogram plots for filtered_RCC alignment
        plot_dir : str, optional
            Directory to save histogram plots for filtered_RCC alignment
    Returns:
        shift : list (len 2-3) of lists (len iterations)
            the shifts in x, y, (z) for each iteration, averaged over
            channels (?)
        cumulative_shift : np array (3, channels, iterations)
            the cumulative shift in the three dimensions, in all channels
            the total shift is the last value (in iterations) fo the cum shift
        use_fiducials : bool
            reports on whether fiducials have been used (and aligned by picked)
            or not (and alignment was by rcc)
        algo_used : str
            the algorithm used for alignment
        fp_figs : list
            list of figure file paths if plotting was enabled
        shift_uncertainties : dict
            dictionary containing uncertainty information (only available
            for RSSO method)
    """
    logger.debug("Aligning datasets")
    if force_method is None:
        force_method = ""
    # check whether any of the fiducial locs are empty
    if fiducial_locs is not None:
        use_fiducials = True
        for locs in fiducial_locs:
            if locs.size == 0:
                fiducial_locs = None
                break
    else:
        use_fiducials = False

    if fiducial_locs is not None and force_method not in ["RCC", "RSSO"]:
        # sort and select corresponding fiducials
        nfidu = [len(np.unique(locs["group"])) for locs in fiducial_locs]
        logger.debug(f"# fiducials before match and sort: {nfidu}")
        fiducial_locs = sort_picked_locs(fiducial_locs, max_shift=max_shift)
        nfidu = [len(np.unique(locs["group"])) for locs in fiducial_locs]
        logger.debug(f"# fiducials after match and sort: {nfidu}")

        algo_used = "by picked"
        (shift, cumulative_shift, channel_locs, fiducial_locs) = (
            align_by_picked(
                channel_locs,
                fiducial_locs,
            )
        )
        fp_figs = []
        shift_uncertainties = {}  # No uncertainty analysis for RCC method
    elif force_method == "RSSO":
        algo_used = "RSSO"
        # Use max_shift parameter if provided, otherwise default to 10.0
        max_shift_param = max_shift if max_shift is not None else 10.0
        shift, fp_figs, shift_uncertainties = align_by_rsso(
            channel_locs,
            channel_tags,
            max_shift=max_shift_param,
            plot_histogram=plot_histogram,
            plot_dir=plot_dir,
        )
        # Create cumulative_shift array for compatibility
        cumulative_shift = np.array(shift)[..., np.newaxis]
    else:
        algo_used = "RCC"
        (shift, cumulative_shift, channel_locs, fiducial_locs) = align_by_rcc(
            channel_locs,
            channel_info,
            max_iterations,
            convergence,
            fiducial_locs,
        )
        fp_figs = []
        shift_uncertainties = {}  # No uncertainty analysis for RCC method

    return (
        shift,
        cumulative_shift,
        use_fiducials,
        algo_used,
        fp_figs,
        shift_uncertainties,
    )


def align_by_picked(channel_locs, fiducial_locs):
    # find shift between channels
    shift = shift_from_picked(fiducial_locs)
    # print("Shift {}".format(shift))

    # align each channel
    for i in range(len(channel_locs)):
        channel_locs[i].y -= shift[0][i]
        channel_locs[i].x -= shift[1][i]
        if len(shift) == 3:
            channel_locs[i].z -= shift[2][i]

        fiducial_locs[i].y -= shift[0][i]
        fiducial_locs[i].x -= shift[1][i]
        if len(shift) == 3:
            fiducial_locs[i].z -= shift[2][i]

    cumulative_shift = np.array(shift)[..., np.newaxis]
    return shift, cumulative_shift, channel_locs, fiducial_locs


def align_by_rcc(
    channel_locs,
    channel_info,
    max_iterations=5,
    convergence=0.001,
    fiducial_locs=None,
):
    shift_x = []
    shift_y = []
    shift_z = []
    all_shift = np.zeros((3, len(channel_locs), max_iterations))
    for iteration in range(max_iterations):
        completed = True

        # find shift between channels
        if fiducial_locs is None:
            use_fiducials = False
            # assignment by reference. Any changes to fiducial_locs will act on
            # channel_locs and vice versa.
            rcc_locs = channel_locs
        else:
            use_fiducials = True
            rcc_locs = fiducial_locs
        shift = shift_from_rcc(rcc_locs, channel_info)
        logger.debug("Shifting channels.")
        temp_shift_x = []
        temp_shift_y = []
        temp_shift_z = []
        for i, locs_ in enumerate(rcc_locs):
            if (
                np.absolute(shift[0][i]) + np.absolute(shift[1][i])
                > convergence
            ):
                completed = False

            # shift each channel
            locs_.y -= shift[0][i]
            locs_.x -= shift[1][i]

            temp_shift_x.append(shift[1][i])
            temp_shift_y.append(shift[0][i])
            all_shift[0, i, iteration] = shift[1][i]
            all_shift[1, i, iteration] = shift[0][i]

            if len(shift) == 3:
                locs_.z -= shift[2][i]
                temp_shift_z.append(shift[2][i])
                all_shift[2, i, iteration] = shift[2][i]
        shift_x.append(np.mean(temp_shift_x))
        shift_y.append(np.mean(temp_shift_y))
        if len(shift) == 3:
            shift_z.append(np.mean(temp_shift_z))

        cumulative_shift = np.cumsum(all_shift, axis=2)

        # Skip when converged:
        if completed:
            break
    shift = [shift_x, shift_y]
    if shift_z != []:
        shift.append(shift_z)

    # shift the locs that were not rcc'ed
    if use_fiducials:
        postshift_locs = channel_locs
    else:
        postshift_locs = fiducial_locs
    if use_fiducials:  # channel_locs != fiducial_locs:
        for i in range(len(postshift_locs)):
            postshift_locs[i].x -= cumulative_shift[0, i, -1]
            postshift_locs[i].y -= cumulative_shift[1, i, -1]
            if len(shift) == 3:
                postshift_locs[i].z -= cumulative_shift[2, i, -1]

    return shift, cumulative_shift, channel_locs, fiducial_locs


def plot_shift(shifts, cum_shifts, filepath):
    """Plot the sifts generated by align_channels
    Args:
        shifts : list of 1D array
            the shifts in x, y, and potentially z dimensions
        cum_shifts : 3 D array
            cumulative shifts (dimension, channel, iteration)
        filepath : str
            the filepath to save the plot
    """
    fig, ax = plt.subplots(nrows=1 + len(shifts), sharex=True)
    # ax[0].suptitle("Shift")
    for i, (shift, dim) in enumerate(zip(shifts, ["x", "y", "z"])):
        ax[0].plot(shift, "o-", label=f"{dim} shift")
        ax[1 + i].plot(cum_shifts[i, :, :])
        ax[1 + i].set_ylabel(f"{dim}-shift (Px)")
    ax[0].set_ylabel("Mean Shift (Px)")
    ax[-1].set_xlabel("Iteration")
    fig.set_size_inches((8, 8))
    ax[0].legend(loc="best")
    fig.savefig(filepath)


def shift_from_rcc(channel_locs, channel_info):
    """
    Used by align. Estimates image shifts based on whole images'
    rcc.

    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels

    Returns:
        shifts : tuple
            the channel shifts shape (2,) or (3,) (if z coordinate present)
    """
    n_channels = len(channel_locs)
    images = []
    logger.debug("Rendering localizations.")
    # render each channel and save it in images
    for i, (locs_, info_) in enumerate(zip(channel_locs, channel_info)):
        _, image = render.render(locs_, info_, blur_method="smooth")
        images.append(image)
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    logger.debug(f"Correlating {n_pairs} image pairs.")
    progress = lib.MockProgress()
    return imageprocess.rcc(images, callback=progress.set_value)


def align_by_rsso(
    channel_locs,
    channel_tags=None,
    max_shift=10.0,
    plot_histogram=False,
    plot_dir=None,
):
    """
    Align channels using redundent spot shift overrepresentation (RSSO)
    based on shift histograms of all channel combinations.

    This function calculates shifts between all channel pairs to provide
    redundant measurements, then solves for the optimal alignment using
    least squares. It assumes that localizations in different channels
    correspond to each other but are shifted by delta_x and delta_y with
    some normal distributed error.

    Args:
        channel_locs : list of np.rec.array
            List of localization arrays for different channels. Each array
            should have 'x' and 'y' fields.
        channel_tags : list of str or None
            the tags to the channels
        max_shift : float, default 10.0
            Maximum expected shift in pixels for alignment
        plot_histogram : bool, default False
            Whether to save 2D histogram plots for each channel pair
        plot_dir : str, optional
            Directory to save histogram plots. If None and plot_histogram
            is True, saves to current directory.

    Returns:
        shifts : tuple
            The channel shifts as (shift_y, shift_x) for compatibility with
            existing code
        fp_figs : list
            List of file paths to saved histogram plots
            (empty if plot_histogram=False)
        shift_uncertainties : dict
            Dictionary containing uncertainty information for channel shifts
    """
    n_channels = len(channel_locs)
    if n_channels < 2:
        return (np.zeros(n_channels), np.zeros(n_channels)), [], {}

    logger.debug(
        f"Aligning {n_channels} channels using RSSO method "
        "with all channel combinations"
    )

    # Calculate pairwise shifts between all channel combinations
    pairwise_shifts = {}
    pairwise_uncertainties = {}
    fp_figs = []
    n_pairs = 0
    if channel_tags is None:
        channel_tags = [str(i) for i in range(n_channels)]
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            shift_x, shift_y, plot_filepath, uncertainty_info = (
                _calculate_pairwise_shift(
                    channel_locs[i],
                    channel_locs[j],
                    max_shift,
                    plot_histogram=plot_histogram,
                    plot_dir=plot_dir,
                    channel_pair=(channel_tags[i], channel_tags[j]),
                )
            )

            if shift_x is not None and shift_y is not None:
                # Store shift from channel i to channel j
                pairwise_shifts[(i, j)] = (shift_x, shift_y)
                pairwise_uncertainties[(i, j)] = uncertainty_info
                n_pairs += 1
                x_unc = uncertainty_info.get("shift_x_uncertainty", np.nan)
                y_unc = uncertainty_info.get("shift_y_uncertainty", np.nan)
                logger.debug(
                    f"Channels {i}->{j} shift: "
                    f"dx={shift_x:.3f}, dy={shift_y:.3f}, "
                    f"uncertainty: dx_err={x_unc:.3f}, "
                    f"dy_err={y_unc:.3f}"
                )

                # Collect figure file paths if plotting is enabled
                if plot_filepath is not None:
                    fp_figs.append(plot_filepath)

    if n_pairs == 0:
        logger.warning("No valid pairwise shifts found")
        return (np.zeros(n_channels), np.zeros(n_channels)), [], {}

    # Solve for optimal channel shifts using least squares
    shifts_x, shifts_y, shift_uncertainties = _solve_optimal_shifts(
        pairwise_shifts, n_channels, pairwise_uncertainties
    )

    # Apply shifts to align channels
    for i in range(len(channel_locs)):
        channel_locs[i].x -= shifts_x[i]
        channel_locs[i].y -= shifts_y[i]

    logger.debug(f"Final channel shifts: x={shifts_x}, y={shifts_y}")

    # Return shifts in format compatible with existing code (y, x order)
    # and any figure file paths created during plotting
    return (shifts_x, shifts_y), fp_figs, shift_uncertainties


def _calculate_pairwise_shift(
    locs_i,
    locs_j,
    max_shift,
    plot_histogram=False,
    plot_dir=None,
    channel_pair=None,
    remove_zeroshift=False,
    plot_fn_suffix="",
    ref_frames=None,
    frame_locs_frames=None,
    ton_exclusion=0,
    peak_mode="auto",
    snr_threshold=3.0,
    build_frame_contributions=False,
    enable_fit_quality_check=True,
    fit_quality_thresholds=None,
):
    """
    Calculate shift between two channels using histogram peak finding with temporal filtering.

    Args:
        locs_i : np.rec.array or cKDTree
            Localizations for first channel (or pre-built KDTree)
        locs_j : np.rec.array
            Localizations for second channel
        max_shift : float
            Maximum expected shift in pixels
        plot_histogram : bool, default False
            Whether to save 2D histogram plot
        plot_dir : str, optional
            Directory to save plots
        channel_pair : tuple, optional
            (i, j) channel indices for filename
        remove_zeroshift : bool, default False
            in case locs_j are part of locs_i, this may be useful
        ref_frames : ndarray, optional
            Frame numbers for reference localizations (for temporal filtering)
        frame_locs_frames : ndarray, optional
            Frame numbers for frame localizations (for temporal filtering)
        ton_exclusion : int, default 0
            Exclude pairs from frames within ±2×ton (temporal filtering)
        peak_mode : str, default "auto"
            Peak finding method:
            - "gaussian": Try 2D Gaussian fit, fall back to center of mass if fails
            - "center_of_mass": Use center of mass of top 9 bins directly
            - "auto": Same as "gaussian" (try Gaussian, fall back to CoM)
        snr_threshold : float, default 3.0
            Signal-to-noise ratio threshold (max_bin / median_bin).
            If SNR < threshold, force center_of_mass and mark as failed.
        build_frame_contributions : bool, default False
            Whether to build frame_contributions dictionary for matrix-based
            drift correction. Only set to True if matrix solver will be used,
            as building this data is computationally expensive (O(n*log(bins))).
        enable_fit_quality_check : bool, default True
            Whether to check Gaussian fit quality and fall back to CoM if quality
            is poor. When True, both chi_squared and r_squared thresholds must pass.
        fit_quality_thresholds : dict, optional
            Thresholds for fit quality check. Default: {"chi_squared": 2.0, "r_squared": 0.90}
            Both thresholds must be satisfied for fit to be accepted.

    Returns:
        shift_x, shift_y, plot_filepath : float, float, str or None
            Shift from channel i to channel j, or (None, None, None) if failed.
            plot_filepath is the path to the saved histogram plot if
            plot_histogram=True, otherwise None.
        sigma_x, sigma_y : float, float
            Gaussian widths representing uncertainty in shift measurements
        shift_x_error, shift_y_error : float, float
            Parameter errors from covariance matrix fitting
    """
    # Set default quality thresholds if not provided
    if fit_quality_thresholds is None:
        fit_quality_thresholds = {"chi_squared": 2.0, "r_squared": 0.90}

    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree
    if not isinstance(locs_i, cKDTree):
        tree_i = cKDTree(
            np.column_stack([locs_i.x, locs_i.y]))
    else:
        tree_i = locs_i

    if tree_i.n == 0 or len(locs_j) == 0:
        return None, None, None, None
    # Calculate all pairwise distances and shifts
    coords_j = np.column_stack([locs_j.x, locs_j.y])

    # Determine if temporal filtering is enabled
    use_temporal_filter = (
        ton_exclusion > 0 and ref_frames is not None and frame_locs_frames is not None
    )
    temporal_threshold = 2 * ton_exclusion

    # Find all j points within max_shift of any i point
    valid_shifts_x = []
    valid_shifts_y = []
    ref_indices = []  # Track which reference loc contributed each shift

    for j_idx, coord_j in enumerate(coords_j):
        # Find all i points within max_shift
        indices = tree_i.query_ball_point(coord_j, max_shift)

        for i_idx in indices:
            coord_i = tree_i.data[i_idx]
            dx = coord_j[0] - coord_i[0]  # x shift from i to j
            dy = coord_j[1] - coord_i[1]  # y shift from i to j

            # Skip zero shifts if requested
            if remove_zeroshift and dx ==0 and dy ==0:
                continue

            # Temporal filtering: skip pairs from nearby frames
            if use_temporal_filter:
                ref_frame = ref_frames[i_idx]
                frame_loc_frame = frame_locs_frames[j_idx]
                frame_diff = abs(ref_frame - frame_loc_frame)
                if frame_diff <= temporal_threshold:
                    continue

            valid_shifts_x.append(dx)
            valid_shifts_y.append(dy)
            ref_indices.append(i_idx)  # Track reference localization index

    if len(valid_shifts_x) == 0:
        logger.warning(
            f"No valid shifts. Returning Nones."
        )
        return None, None, None, None
    # Create 2D histogram with adaptive binning
    shift_range = [-max_shift, max_shift]
    bin_size, bins = _calculate_adaptive_bins(
        valid_shifts_x, valid_shifts_y, max_shift
    )

    hist, x_edges, y_edges = np.histogram2d(
        valid_shifts_x,
        valid_shifts_y,
        bins=bins,
        range=[shift_range, shift_range],
    )

    # Build frame contributions dictionary (for matrix-based drift correction)
    # ONLY if explicitly requested, as this is computationally expensive
    frame_contributions = None
    if build_frame_contributions and ref_frames is not None and len(ref_indices) > 0:
        frame_contributions = {}
        n_bins = len(x_edges) - 1

        for shift_idx in range(len(valid_shifts_x)):
            shift_x_val = valid_shifts_x[shift_idx]
            shift_y_val = valid_shifts_y[shift_idx]
            ref_idx = ref_indices[shift_idx]
            ref_frame = ref_frames[ref_idx]

            # Find which bin this shift fell into
            bin_x = np.searchsorted(x_edges, shift_x_val, side='right') - 1
            bin_y = np.searchsorted(y_edges, shift_y_val, side='right') - 1

            # Ensure valid bin indices
            bin_x = np.clip(bin_x, 0, n_bins - 1)
            bin_y = np.clip(bin_y, 0, n_bins - 1)

            # Track this contribution
            if ref_frame not in frame_contributions:
                frame_contributions[ref_frame] = []
            frame_contributions[ref_frame].append((int(bin_x), int(bin_y)))

    # Calculate signal-to-noise ratio
    max_bin_value = np.max(hist)
    non_zero_bins = hist[hist > 0]
    median_bin_value = np.median(non_zero_bins) if len(non_zero_bins) > 0 else 1.0
    snr = max_bin_value / median_bin_value if median_bin_value > 0 else 0.0

    # Check if SNR is too low - if so, force center_of_mass and mark as failed
    # This allows max_shift iterations to increase search radius
    snr_too_low = snr < snr_threshold
    peak_mode_to_use = peak_mode

    # if snr_too_low and peak_mode != "center_of_mass":
    #     peak_mode_to_use = "center_of_mass"

    # Find peak using specified method
    fit_successful = False
    com_threshold = None
    com_use_threshold = None
    goodness_of_fit = None  # Initialize for all paths

    if peak_mode_to_use == "center_of_mass":
        # Use center of mass directly (faster, more robust for broad peaks)
        (shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error,
         com_threshold, com_use_threshold) = (
            _find_peak_center_of_mass(hist, x_edges, y_edges, max_shift, snr_threshold)
        )
        # If user EXPLICITLY requested center_of_mass, ALWAYS mark as successful
        # If FORCED to center_of_mass due to low SNR, mark as failed to trigger max_shift retry
        if snr_too_low and peak_mode != "center_of_mass":
            fit_successful = False  # Was forced by low SNR, trigger max_shift iterations
        else:
            fit_successful = True  # Explicit request OR good SNR - mark as successful

    elif peak_mode_to_use in ["gaussian", "auto"]:
        # Try 2D Gaussian fitting first
        try:
            shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error, goodness_of_fit = (
                _fit_2d_gaussian_peak(hist, x_edges, y_edges, max_shift)
            )

            # Check fit quality if enabled
            fit_quality_passed = True
            if enable_fit_quality_check and goodness_of_fit is not None:
                chi_sq = goodness_of_fit["chi_squared_reduced"]
                r_sq = goodness_of_fit["r_squared"]
                chi_threshold = fit_quality_thresholds["chi_squared"]
                r_threshold = fit_quality_thresholds["r_squared"]

                # Combined metric: BOTH chi_squared and r_squared must pass
                fit_quality_passed = (chi_sq < chi_threshold) and (r_sq > r_threshold)

                if not fit_quality_passed:
                    # logger.warning(
                    #     f"Gaussian fit quality check failed: χ²_red={chi_sq:.3f} "
                    #     f"(threshold={chi_threshold}), R²={r_sq:.3f} "
                    #     f"(threshold={r_threshold}). Falling back to center of mass method."
                    # )
                    raise ValueError("Fit quality check failed")

            fit_successful = True

        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"2D Gaussian fitting failed: {e}. "
                "Falling back to center of mass method."
            )
            # Fallback to center of mass method (better than simple maximum)
            (shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error,
             com_threshold, com_use_threshold) = (
                _find_peak_center_of_mass(hist, x_edges, y_edges, max_shift, snr_threshold)
            )
            fit_successful = False  # Indicates fallback was used
            peak_mode_to_use = "center_of_mass"  # Update to reflect actual method used
            goodness_of_fit = None  # CoM doesn't have goodness of fit

    else:
        raise ValueError(f"Unknown peak_mode: {peak_mode}. Use 'gaussian', 'center_of_mass', or 'auto'.")

    # Return shift values and uncertainties
    uncertainty_info = {
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "shift_x_error": shift_x_error,
        "shift_y_error": shift_y_error,
        "fit_successful": fit_successful,
        "snr": float(snr),
        "snr_threshold": float(snr_threshold),
        "snr_too_low": bool(snr_too_low),
        "peak_mode": peak_mode_to_use,  # Actual peak finding method used
        "com_threshold": com_threshold,  # Threshold value for CoM bin selection
        "com_use_threshold": com_use_threshold,  # Whether threshold mode was used
        "goodness_of_fit": goodness_of_fit,  # Gaussian fit quality metrics (None for CoM)
        "frame_contributions": frame_contributions,  # For matrix-based drift correction
        "hist": hist,  # Histogram for connectivity matrix building
        "hist_edges": (x_edges, y_edges),  # Bin edges for histogram
    }

    # Create and save histogram plot if requested (using unified plotting function)
    plot_filepath = None
    if plot_histogram:
        # Normalize quality metrics for unified plotting function
        quality_metrics = _normalize_quality_metrics(
            uncertainty_info=uncertainty_info,
            hist=hist
        )

        # Call unified plotting function
        plot_filepath = _save_rsso_shift_histogram_plot(
            hist=hist,
            x_edges=x_edges,
            y_edges=y_edges,
            shift_x=shift_x,
            shift_y=shift_y,
            max_shift=max_shift,
            plot_dir=plot_dir,
            quality_metrics=quality_metrics,
            plot_fn_suffix=plot_fn_suffix,
            iteration=None,  # Not used in standard route
            frame_number=None,  # Not used in standard route
            channel_pair=channel_pair,  # Standard route uses channel_pair
            output_subdir=None,  # Standard route saves directly to plot_dir
            shared_plot_dict=None,  # Standard route always saves to disk
        )

    return shift_x, shift_y, plot_filepath, uncertainty_info


def _calculate_adaptive_bins(valid_shifts_x, valid_shifts_y, max_shift):
    """
    Calculate adaptive bin size and number of bins for optimal histogram.

    Args:
        valid_shifts_x : list
            X shift values
        valid_shifts_y : list
            Y shift values
        max_shift : float
            Maximum shift range

    Returns:
        bin_size : float
            Calculated bin size in pixels
        bins : int
            Number of bins for histogram
    """
    base_bin_size = 0.1  # Base bin size of 0.1 pixels
    n_points = len(valid_shifts_x)

    # Adaptive adjustment based on data density
    if n_points < 50:
        # Few points: use larger bins for better statistics
        bin_size = base_bin_size * 2.0
    elif n_points < 200:
        # Moderate points: use base bin size
        bin_size = base_bin_size
    else:
        # Many points: can use smaller bins for higher precision
        bin_size = base_bin_size * 0.5

    # Calculate number of bins
    total_range = 2 * max_shift
    bins = max(10, int(total_range / bin_size))

    # Ensure reasonable limits
    bins = min(bins, 500)  # Upper limit to prevent excessive computation
    bins = max(bins, 20)  # Lower limit for meaningful histogram

    # logger.debug(
    #     f"Adaptive binning: {n_points} points, "
    #     f"bin_size={bin_size:.3f}, bins={bins}"
    # )

    return bin_size, bins


def _find_peak_center_of_mass(hist, x_edges, y_edges, max_shift=None, snr_threshold=3.0):
    """
    Find peak location using center of mass of the highest bins.

    More robust than simple histogram maximum for broad/non-Gaussian peaks.
    Provides sub-bin precision without requiring Gaussian fitting.

    Uses threshold-based bin selection: includes all bins with count >= median × snr_threshold
    in the center of mass calculation (minimum 9 bins). Falls back to 3×3 neighborhood if
    fewer than 9 bins exceed the threshold.

    Args:
        hist : np.array
            2D histogram of shifts
        x_edges : np.array
            Histogram x bin edges
        y_edges : np.array
            Histogram y bin edges
        max_shift : float, optional
            Maximum shift radius (for consistency with Gaussian fit API)
        snr_threshold : float, optional
            Threshold multiplier for median bin count. Bins with count >= median × snr_threshold
            are included in center of mass calculation. Default: 3.0

    Returns:
        shift_x, shift_y : float, float
            Center of mass coordinates
        sigma_x, sigma_y : float, float
            Estimated spread (std dev of 3×3 neighborhood values)
        shift_x_error, shift_y_error : float, float
            Estimated errors (based on bin size and peak sharpness)
        threshold : float
            Threshold value used for bin selection (median × snr_threshold)
        use_threshold : bool
            Whether threshold-based selection was used (True) or 3×3 fallback (False)
    """
    # Create bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    bin_size_x = x_edges[1] - x_edges[0] if len(x_edges) > 1 else 1.0
    bin_size_y = y_edges[1] - y_edges[0] if len(y_edges) > 1 else 1.0

    # Find histogram maximum
    hist_transposed = hist.T  # Match meshgrid convention
    peak_idx = np.unravel_index(np.argmax(hist_transposed), hist_transposed.shape)
    peak_y_bin, peak_x_bin = peak_idx  # Note: unravel_index returns (row, col) = (y, x)

    # Calculate median of non-zero bins for threshold-based selection
    non_zero_bins = hist_transposed[hist_transposed > 0]
    if len(non_zero_bins) == 0:
        # Fallback: return simple maximum position
        shift_x = x_centers[peak_x_bin]
        shift_y = y_centers[peak_y_bin]
        sigma_x = bin_size_x
        sigma_y = bin_size_y
        shift_x_error = bin_size_x / 2
        shift_y_error = bin_size_y / 2
        return shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error, 0.0, False

    median_val = np.median(non_zero_bins)
    threshold = median_val * snr_threshold

    # Create mask for bins above threshold
    threshold_mask = hist_transposed >= threshold
    n_threshold_bins = np.sum(threshold_mask)
    use_threshold = n_threshold_bins >= 9

    if use_threshold:
        # Use threshold-selected bins for center of mass
        y_indices_all, x_indices_all = np.where(threshold_mask)

        # Get unique indices
        x_indices = np.unique(x_indices_all)
        y_indices = np.unique(y_indices_all)

        # Extract selected values and coordinates
        x_idx_grid, y_idx_grid = np.meshgrid(x_indices, y_indices)
        neighborhood_values = hist_transposed[y_idx_grid, x_idx_grid]

        # Apply threshold mask to the meshgrid
        mask_grid = threshold_mask[y_idx_grid, x_idx_grid]
        neighborhood_values = np.where(mask_grid, neighborhood_values, 0)
    else:
        # Fall back to 3×3 neighborhood around maximum
        x_indices = np.clip(
            [peak_x_bin - 1, peak_x_bin, peak_x_bin + 1],
            0,
            hist.shape[0] - 1
        ).astype(int)
        y_indices = np.clip(
            [peak_y_bin - 1, peak_y_bin, peak_y_bin + 1],
            0,
            hist.shape[1] - 1
        ).astype(int)

        # Remove duplicates (in case peak is at edge and clipping created duplicates)
        x_indices = np.unique(x_indices)
        y_indices = np.unique(y_indices)

        # Extract neighborhood values
        x_idx_grid, y_idx_grid = np.meshgrid(x_indices, y_indices)
        neighborhood_values = hist_transposed[y_idx_grid, x_idx_grid]

    # Clip negative values to zero (shouldn't happen but be safe)
    neighborhood_values = np.maximum(neighborhood_values, 0)

    # Check if we have any non-zero values
    total_mass = np.sum(neighborhood_values)
    if total_mass == 0:
        # Fallback: return simple maximum position
        shift_x = x_centers[peak_x_bin]
        shift_y = y_centers[peak_y_bin]
        sigma_x = bin_size_x
        sigma_y = bin_size_y
        shift_x_error = bin_size_x / 2
        shift_y_error = bin_size_y / 2
        return shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error

    # Get coordinate values for the neighborhood
    x_coords = x_centers[x_indices]
    y_coords = y_centers[y_indices]
    x_coord_grid, y_coord_grid = np.meshgrid(x_coords, y_coords)

    # Calculate center of mass
    shift_x = np.sum(neighborhood_values * x_coord_grid) / total_mass
    shift_y = np.sum(neighborhood_values * y_coord_grid) / total_mass

    # Estimate uncertainty based on spread of the neighborhood
    # Calculate weighted standard deviation
    sigma_x = np.sqrt(
        np.sum(neighborhood_values * (x_coord_grid - shift_x)**2) / total_mass
    )
    sigma_y = np.sqrt(
        np.sum(neighborhood_values * (y_coord_grid - shift_y)**2) / total_mass
    )

    # Estimate position error based on bin size and peak sharpness
    # Sharper peaks (high max/mean ratio) have lower uncertainty
    peak_value = np.max(neighborhood_values)
    mean_value = np.mean(neighborhood_values[neighborhood_values > 0])
    sharpness_ratio = peak_value / mean_value if mean_value > 0 else 1.0

    # Error scales inversely with sharpness, bounded by bin size
    shift_x_error = min(bin_size_x / (2 * np.sqrt(sharpness_ratio)), bin_size_x / 2)
    shift_y_error = min(bin_size_y / (2 * np.sqrt(sharpness_ratio)), bin_size_y / 2)

    return shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error, threshold, use_threshold


def _fit_2d_gaussian_peak(hist, x_edges, y_edges, max_shift=None):
    """
    Fit a 2D Gaussian to the histogram peak to find precise shift location.
    Values outside the max_shift circle are set to NaN and excluded from
    fitting.

    Args:
        hist : np.array
            2D histogram of shifts
        x_edges : np.array
            Histogram x bin edges
        y_edges : np.array
            Histogram y bin edges
        max_shift : float, optional
            Maximum shift radius. Values outside this circle are set to NaN.

    Returns:
        shift_x : float
            Fitted peak center x coordinate
        shift_y : float
            Fitted peak center y coordinate
        sigma_x : float
            Gaussian width in x direction
        sigma_y : float
            Gaussian width in y direction
        shift_x_error : float
            Uncertainty in x coordinate from covariance matrix
        shift_y_error : float
            Uncertainty in y coordinate from covariance matrix
        goodness_of_fit : dict
            Dictionary containing fit quality metrics:
            - chi_squared_reduced: Reduced chi-squared statistic
            - r_squared: R-squared (coefficient of determination)
            - rmse: Root mean squared error
            - n_points: Number of data points used in fit
            - degrees_of_freedom: Degrees of freedom (n_points - n_params)
    """
    from scipy.optimize import curve_fit

    # Create coordinate grids
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Apply circular mask if max_shift is specified
    hist_masked = hist.T.copy()  # Transpose to match meshgrid convention
    if max_shift is not None:
        # Calculate distance from origin for each histogram bin
        distances = np.sqrt(X**2 + Y**2)
        # Set values outside max_shift circle to NaN
        outside_circle = distances > max_shift
        hist_masked[outside_circle] = np.nan
        # logger.debug(
        #     f"Applied circular mask: {np.sum(outside_circle)} bins "
        #     f"outside max_shift={max_shift} set to NaN"
        # )

    # Flatten for fitting
    x_data = X.ravel()
    y_data = Y.ravel()
    z_data = hist_masked.ravel()

    # Remove NaN and zero counts for better fitting
    valid_mask = ~np.isnan(z_data) & (z_data > 0)
    if np.sum(valid_mask) < 10:
        raise ValueError("Insufficient non-zero data points for fitting")

    x_fit = x_data[valid_mask]
    y_fit = y_data[valid_mask]
    z_fit = z_data[valid_mask]

    # Initial parameter estimates
    max_idx = np.argmax(z_fit)
    x0_init = x_fit[max_idx]
    y0_init = y_fit[max_idx]

    # Improved background estimation using percentiles of valid data
    background_init = np.percentile(z_fit, 10)  # 10th percentile as background
    amplitude_init = np.max(z_fit) - background_init  # Peak above background
    sigma_init = 0.5  # Initial guess for standard deviation

    # logger.debug(
    #     f"Initial fit parameters: center=({x0_init:.3f}, {y0_init:.3f}), "
    #     f"amplitude={amplitude_init:.1f}, background={background_init:.1f}"
    # )

    # Define 2D Gaussian function
    def gaussian_2d(
        coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    ):
        x, y = coords
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        a = (cos_t**2) / (2 * sigma_x**2) + (sin_t**2) / (2 * sigma_y**2)
        sin_2t = np.sin(2 * theta)
        b = -sin_2t / (4 * sigma_x**2) + sin_2t / (4 * sigma_y**2)
        c = (sin_t**2) / (2 * sigma_x**2) + (cos_t**2) / (2 * sigma_y**2)
        exponent = -(
            a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2
        )
        return offset + amplitude * np.exp(exponent)

    # Initial parameters: [amplitude, x0, y0, sigma_x, sigma_y, theta, offset]
    initial_guess = [
        amplitude_init,
        x0_init,
        y0_init,
        sigma_init,
        sigma_init,
        0.0,
        background_init,
    ]

    # Parameter bounds with improved background constraints
    max_amplitude = np.max(z_fit)
    bounds = (
        [0, x_centers.min(), y_centers.min(), 0.01, 0.01, -np.pi / 4, 0],
        [
            max_amplitude * 2,
            x_centers.max(),
            y_centers.max(),
            2.0,
            2.0,
            np.pi / 4,
            max_amplitude,
        ],
    )

    # Perform the fit
    try:
        # Suppress numerical warnings from scipy optimizer
        # (divide by zero warnings are expected for ill-conditioned histograms)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            popt, pcov = curve_fit(
                gaussian_2d,
                (x_fit, y_fit),
                z_fit,
                p0=initial_guess,
                bounds=bounds,
                maxfev=1000,
            )

        # Extract fitted center coordinates and uncertainties
        shift_x = popt[1]  # x0
        shift_y = popt[2]  # y0
        sigma_x = popt[3]  # Gaussian width in x (uncertainty measure)
        sigma_y = popt[4]  # Gaussian width in y (uncertainty measure)

        # Extract parameter uncertainties from covariance matrix
        param_errors = np.sqrt(np.diag(pcov))
        shift_x_error = param_errors[1]  # Error in x0
        shift_y_error = param_errors[2]  # Error in y0

        # logger.debug(
        #     f"2D Gaussian fit successful: "
        #     f"center=({shift_x:.3f}±{shift_x_error:.3f}, "
        #     f"{shift_y:.3f}±{shift_y_error:.3f}), "
        #     f"sigma=({sigma_x:.3f}, {sigma_y:.3f}), "
        #     f"amplitude={popt[0]:.1f}, background={popt[6]:.1f}"
        # )

        # Calculate goodness of fit metrics
        # Predicted values using fitted parameters
        z_pred = gaussian_2d((x_fit, y_fit), *popt)

        # Residuals
        residuals = z_fit - z_pred

        # Reduced chi-squared (assuming Poisson variance for counts)
        # χ²_red = Σ[(observed - predicted)² / observed] / (n_points - n_params)
        n_points = len(z_fit)
        n_params = len(popt)  # 7 parameters
        degrees_of_freedom = n_points - n_params

        # Use Poisson variance (variance = observed count)
        # For bins with very low counts, use minimum of 1 to avoid division by zero
        variances = np.maximum(z_fit, 1.0)
        chi_squared = np.sum((residuals ** 2) / variances)
        chi_squared_reduced = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.inf

        # R-squared (coefficient of determination)
        # R² = 1 - (SS_residual / SS_total)
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((z_fit - np.mean(z_fit)) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

        # Root mean squared error
        rmse = np.sqrt(np.mean(residuals ** 2))

        goodness_of_fit = {
            "chi_squared_reduced": chi_squared_reduced,
            "r_squared": r_squared,
            "rmse": rmse,
            "n_points": n_points,
            "degrees_of_freedom": degrees_of_freedom,
        }

        return shift_x, shift_y, sigma_x, sigma_y, shift_x_error, shift_y_error, goodness_of_fit

    except Exception as e:
        raise RuntimeError(f"Gaussian fitting failed: {str(e)}")


def _normalize_quality_metrics(uncertainty_info=None, quality_metrics=None, hist=None):
    """
    Normalize quality metrics from either standard or numba route format.

    Converts uncertainty_info (standard route) or quality_metrics (numba route)
    into a unified format suitable for plotting functions.

    Args:
        uncertainty_info : dict or None
            Quality metrics from standard route (_calculate_pairwise_shift)
            Expected keys: sigma_x, sigma_y, fit_successful, peak_mode,
                          com_threshold, com_use_threshold
        quality_metrics : dict or None
            Quality metrics from numba route (_compute_rsso_shift_numba_optimized)
            Expected keys: sigma_x, sigma_y, peak_mode, success,
                          com_threshold, com_use_threshold, total_pairs
        hist : np.array or None
            2D histogram for calculating total_pairs if not in quality_metrics

    Returns:
        normalized : dict
            Unified quality metrics dict with keys:
                - peak_mode : str
                - sigma_x : float
                - sigma_y : float
                - total_pairs : float
                - com_threshold : float or None
                - com_use_threshold : bool or None
    """
    normalized = {}

    # Determine source and extract values
    if quality_metrics is not None:
        # Numba route - already in good format
        normalized["peak_mode"] = quality_metrics.get("peak_mode", "unknown")
        normalized["sigma_x"] = quality_metrics.get("sigma_x", 0.0)
        normalized["sigma_y"] = quality_metrics.get("sigma_y", 0.0)
        normalized["total_pairs"] = quality_metrics.get(
            "total_pairs",
            np.sum(hist) if hist is not None else 0
        )
        normalized["com_threshold"] = quality_metrics.get("com_threshold")
        normalized["com_use_threshold"] = quality_metrics.get("com_use_threshold")

    elif uncertainty_info is not None:
        # Standard route - convert from uncertainty_info format
        fit_successful = uncertainty_info.get("fit_successful", False)
        peak_mode = uncertainty_info.get("peak_mode", "unknown")

        # Convert fit_successful to peak_mode if not explicitly set
        if peak_mode == "unknown":
            peak_mode = "gaussian" if fit_successful else "histogram_maximum"

        normalized["peak_mode"] = peak_mode
        normalized["sigma_x"] = uncertainty_info.get("sigma_x", 0.0)
        normalized["sigma_y"] = uncertainty_info.get("sigma_y", 0.0)
        normalized["total_pairs"] = np.sum(hist) if hist is not None else 0
        normalized["com_threshold"] = uncertainty_info.get("com_threshold")
        normalized["com_use_threshold"] = uncertainty_info.get("com_use_threshold")

    else:
        # No metrics provided - use defaults
        normalized["peak_mode"] = "unknown"
        normalized["sigma_x"] = 0.0
        normalized["sigma_y"] = 0.0
        normalized["total_pairs"] = np.sum(hist) if hist is not None else 0
        normalized["com_threshold"] = None
        normalized["com_use_threshold"] = None

    return normalized


def _save_rsso_shift_histogram_plot(
    hist,
    x_edges,
    y_edges,
    shift_x,
    shift_y,
    max_shift,
    plot_dir,
    quality_metrics,
    plot_fn_suffix=None,
    iteration=None,
    frame_number=None,
    channel_pair=None,
    output_subdir=None,
    shared_plot_dict=None,
):
    """
    Unified function to save 2D histogram plots showing RSSO shift distribution and peak.

    Works for both standard (channel alignment) and numba (iterative drift) routes.
    Can either save to disk (traditional mode) or store in shared memory
    for incremental video writing (memory-efficient mode).

    Args:
        hist : np.array
            2D histogram of shifts
        x_edges : np.array
            Histogram x bin edges
        y_edges : np.array
            Histogram y bin edges
        shift_x : float
            Estimated x shift
        shift_y : float
            Estimated y shift
        max_shift : float
            Maximum shift range
        plot_dir : str
            Directory to save plot (only used if shared_plot_dict is None)
        quality_metrics : dict
            Quality metrics to display on plot. Expected keys:
                - peak_mode : str (method used)
                - sigma_x, sigma_y : float (uncertainties)
                - total_pairs : float (number of pairs)
                - com_threshold : float or None (CoM threshold)
                - com_use_threshold : bool or None (whether threshold was used)
        iteration : int or None, optional
            Iteration number for filename (numba route)
        frame_number : int or None, optional
            Frame number for filename (numba route)
        channel_pair : tuple or None, optional
            (i, j) channel indices for filename (standard route)
        output_subdir : str or None, optional
            Subdirectory structure to create (e.g., "rsso_plots/iteration_00/sglframe/")
            If None, saves directly to plot_dir
        shared_plot_dict : multiprocessing.Manager.dict() or None, optional
            If provided, stores plot as numpy array in this dict instead of saving to disk.
            Key is frame_number, value is numpy array of shape (height, width, 3).

    Returns:
        filepath : str or frame_number
            If shared_plot_dict is None: Path to saved plot file
            If shared_plot_dict is provided: Frame number (key in dict)
    """
    import matplotlib.pyplot as plt
    import random
    import string

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create coordinate grids - use bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)

    # Apply circular mask for visualization
    hist_plot = hist.T.copy()
    if max_shift is not None:
        distances = np.sqrt(X_centers**2 + Y_centers**2)
        outside_circle = distances > max_shift
        hist_plot[outside_circle] = np.nan

    # Plot the 2D histogram
    im = ax.pcolormesh(
        X_centers, Y_centers, hist_plot, cmap="viridis", shading="nearest"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20)

    # Add circular boundary
    if max_shift is not None:
        circle = plt.Circle(
            (0, 0),
            max_shift,
            fill=False,
            color="white",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(circle)

    # Mark the detected shift with a red cross
    # Include peak finding method in legend
    peak_mode_display = quality_metrics.get("peak_mode", "unknown")
    ax.plot(
        shift_x,
        shift_y,
        "r+",
        markersize=15,
        markeredgewidth=2,
        label=f"Shift: ({shift_x:.3f}, {shift_y:.3f}) px\nMethod: {peak_mode_display}",
    )

    # Visualize bins used for center of mass calculation
    com_threshold = quality_metrics.get("com_threshold")
    com_use_threshold = quality_metrics.get("com_use_threshold")

    if com_threshold is not None and peak_mode_display == "center_of_mass":
        # Recreate bin selection mask
        hist_t = hist.T
        bin_size_x = x_edges[1] - x_edges[0] if len(x_edges) > 1 else 1.0
        bin_size_y = y_edges[1] - y_edges[0] if len(y_edges) > 1 else 1.0

        if com_use_threshold:
            # Threshold-based selection: outline all bins >= threshold
            selected_bins = hist_t >= com_threshold
        else:
            # 3×3 fallback: outline 3×3 neighborhood around peak
            peak_idx = np.unravel_index(np.argmax(hist_t), hist_t.shape)
            peak_y, peak_x = peak_idx

            # Create 3×3 mask
            selected_bins = np.zeros_like(hist_t, dtype=bool)
            y_start = max(0, peak_y - 1)
            y_end = min(hist_t.shape[0], peak_y + 2)
            x_start = max(0, peak_x - 1)
            x_end = min(hist_t.shape[1], peak_x + 2)
            selected_bins[y_start:y_end, x_start:x_end] = True

        # Draw outlines around selected bins
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                if selected_bins[j, i]:  # Note: hist_t is transposed
                    # Calculate bin edges
                    x_left = x_edges[i]
                    x_right = x_edges[i+1] if i+1 < len(x_edges) else x_edges[i] + bin_size_x
                    y_bottom = y_edges[j]
                    y_top = y_edges[j+1] if j+1 < len(y_edges) else y_edges[j] + bin_size_y

                    # Draw rectangle outline
                    rect = plt.Rectangle(
                        (x_left, y_bottom),
                        x_right - x_left,
                        y_top - y_bottom,
                        fill=False,
                        edgecolor='cyan',
                        linewidth=1.5,
                        alpha=0.8
                    )
                    ax.add_patch(rect)

    # Set labels and title
    ax.set_xlabel("X Shift (pixels)")
    ax.set_ylabel("Y Shift (pixels)")

    # Build title based on available information
    title_parts = []
    if channel_pair is not None:
        # Standard route - channel alignment
        title_parts.append(f"RSSO: Ch {channel_pair[0]} → Ch {channel_pair[1]}")
    else:
        # Numba route - iterative drift
        title_parts.append("RSSO Shift Histogram")

    if iteration is not None:
        title_parts.append(f"Iter {iteration}")
    if frame_number is not None:
        title_parts.append(f"Frame {frame_number}")

    ax.set_title(" - ".join(title_parts))

    # Set axis limits
    ax.set_xlim(-max_shift, max_shift)
    ax.set_ylim(-max_shift, max_shift)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Add text box with statistics
    total_pairs = quality_metrics.get("total_pairs", np.sum(hist))
    sigma_x = quality_metrics.get("sigma_x", 0)
    sigma_y = quality_metrics.get("sigma_y", 0)

    # Calculate bin statistics (min, max, median) from bins within circular mask
    valid_bins = hist_plot[~np.isnan(hist_plot)]
    max_bin_value = np.max(valid_bins) if len(valid_bins) > 0 else 0
    min_bin_value = np.min(valid_bins) if len(valid_bins) > 0 else 0
    median_bin_value = np.median(valid_bins) if len(valid_bins) > 0 else 0

    textstr = (
        f"Total pairs: {total_pairs:.0f}\n"
        f"Max bin: {max_bin_value:.0f}\n"
        f"Min bin: {min_bin_value:.0f}\n"
        f"Median bin: {median_bin_value:.1f}\n"
        f"σx: {sigma_x:.3f} px\n"
        f"σy: {sigma_y:.3f} px"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Either save to disk or store in shared memory
    if shared_plot_dict is not None:
        # Memory-efficient mode: Render to numpy array and store in shared dict
        fig.canvas.draw()

        # Convert figure to numpy array (RGB format)
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Store in shared dict with frame number as key
        if frame_number is not None:
            shared_plot_dict[frame_number] = img_array

        plt.close(fig)
        return frame_number

    else:
        # Traditional mode: Save to disk
        # Determine save directory
        if output_subdir is not None:
            # Use provided subdirectory structure
            save_dir = os.path.join(plot_dir, output_subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            # Save directly to plot_dir
            os.makedirs(plot_dir, exist_ok=True)
            save_dir = plot_dir

        # Generate filename based on available information
        if channel_pair is not None:
            # Standard route naming
            filename = f"shift_histogram_ch{channel_pair[0]}_to_ch{channel_pair[1]}.png"
        else:
            # Numba route naming
            filename_parts = ["rsso"]
            if iteration is not None:
                filename_parts.append(f"iter{iteration:02d}")
            if frame_number is not None:
                filename_parts.append(f"frame{frame_number:04d}")
            if plot_fn_suffix is not None:
                filename_parts.append(plot_fn_suffix)

            # Add random code for uniqueness
            rcode = "".join(random.choices(string.ascii_letters, k=6))
            filename_parts.append(rcode)
            filename = "_".join(filename_parts) + ".png"

        filepath = os.path.join(save_dir, filename)

        # Handle filename collision for standard route
        if channel_pair is not None and os.path.exists(filepath):
            r, e = os.path.splitext(filename)
            nfiles = len([f for f in os.listdir(save_dir) if r in f])
            r, e = os.path.splitext(filepath)
            filepath = f"{r}-{nfiles}{e}"

        # Save the plot
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return filepath


def _save_shift_histogram_plot(
    hist,
    x_edges,
    y_edges,
    shift_x,
    shift_y,
    max_shift,
    plot_dir,
    channel_pair,
    fit_successful=False,
    plot_fn_suffix=""
):
    """
    DEPRECATED: Use _save_rsso_shift_histogram_plot instead.

    This function is kept for backward compatibility and redirects to the
    unified plotting function.

    Args:
        hist : np.array - 2D histogram of shifts
        x_edges : np.array - Histogram x bin edges
        y_edges : np.array - Histogram y bin edges
        shift_x : float - Estimated x shift
        shift_y : float - Estimated y shift
        max_shift : float - Maximum shift range
        plot_dir : str or None - Directory to save plot
        channel_pair : tuple - (i, j) channel indices
        fit_successful : bool - Whether Gaussian fit succeeded
        plot_fn_suffix : str - Suffix for filename (ignored in new function)

    Returns:
        filepath : str - Path to saved plot file
    """
    import warnings
    warnings.warn(
        "_save_shift_histogram_plot is deprecated. "
        "Use _save_rsso_shift_histogram_plot instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Build quality_metrics from old parameters
    peak_mode = "gaussian" if fit_successful else "histogram_maximum"
    quality_metrics = {
        "peak_mode": peak_mode,
        "sigma_x": 0.0,  # Not available in old function
        "sigma_y": 0.0,  # Not available in old function
        "total_pairs": np.sum(hist),
        "com_threshold": None,
        "com_use_threshold": None,
    }

    if plot_dir is None:
        plot_dir = "."

    # Call unified function
    return _save_rsso_shift_histogram_plot(
        hist=hist,
        x_edges=x_edges,
        y_edges=y_edges,
        shift_x=shift_x,
        shift_y=shift_y,
        max_shift=max_shift,
        plot_dir=plot_dir,
        quality_metrics=quality_metrics,
        iteration=None,
        frame_number=None,
        channel_pair=channel_pair,
        output_subdir=None,  # Save directly to plot_dir
        shared_plot_dict=None,
    )


def _solve_optimal_shifts(
    pairwise_shifts, n_channels, pairwise_uncertainties=None
):
    """
    Solve for optimal channel shifts using least squares given pairwise
    measurements.

    The constraint is that shift_j - shift_i = measured_shift_ij for all
    pairs (i,j). We set the first channel as reference (shift = 0) and
    solve for the others.

    Args:
        pairwise_shifts : dict
            Dictionary with keys (i,j) and values (shift_x, shift_y)
        n_channels : int
            Number of channels
        pairwise_uncertainties : dict, optional
            Dictionary with keys (i,j) and values containing uncertainty
            information

    Returns:
        shifts_x, shifts_y : np.array, np.array
            Optimal shifts for each channel
        shift_uncertainties : dict
            Dictionary containing uncertainty information for channel shifts
    """
    if len(pairwise_shifts) == 0:
        return np.zeros(n_channels), np.zeros(n_channels), {}

    # Build linear system: A * shifts = b
    # Each pairwise measurement gives us: shift_j - shift_i = measured_shift
    n_equations = len(pairwise_shifts)
    n_unknowns = n_channels - 1  # First channel is reference (shift = 0)

    A_x = np.zeros((n_equations, n_unknowns))
    A_y = np.zeros((n_equations, n_unknowns))
    b_x = np.zeros(n_equations)
    b_y = np.zeros(n_equations)

    # Build weight matrices for uncertainty propagation if available
    W_x = np.eye(n_equations)  # Default to identity (equal weights)
    W_y = np.eye(n_equations)
    if pairwise_uncertainties is not None:
        for eq_idx, (i, j) in enumerate(pairwise_shifts.keys()):
            uncertainty_info = pairwise_uncertainties.get((i, j), {})
            shift_x_uncertainty = uncertainty_info.get(
                "shift_x_uncertainty", 1.0
            )
            shift_y_uncertainty = uncertainty_info.get(
                "shift_y_uncertainty", 1.0
            )

            # Use inverse variance weighting (higher precision = higher weight)
            if shift_x_uncertainty > 0:
                W_x[eq_idx, eq_idx] = 1.0 / (shift_x_uncertainty**2)
            if shift_y_uncertainty > 0:
                W_y[eq_idx, eq_idx] = 1.0 / (shift_y_uncertainty**2)

    eq_idx = 0
    for (i, j), (shift_x, shift_y) in pairwise_shifts.items():
        # Equation: shift_j - shift_i = measured_shift

        # Handle reference channel (channel 0 has shift = 0)
        if i > 0:  # shift_i is unknown
            A_x[eq_idx, i - 1] = -1
            A_y[eq_idx, i - 1] = -1

        if j > 0:  # shift_j is unknown
            A_x[eq_idx, j - 1] = 1
            A_y[eq_idx, j - 1] = 1

        b_x[eq_idx] = shift_x
        b_y[eq_idx] = shift_y
        eq_idx += 1

    # Solve weighted least squares problem
    shift_x_uncertainties = np.zeros(n_channels)
    shift_y_uncertainties = np.zeros(n_channels)

    try:
        if n_unknowns > 0:
            # Weighted least squares: (A^T W A)^-1 A^T W b
            AtWA_x = A_x.T @ W_x @ A_x
            AtWb_x = A_x.T @ W_x @ b_x
            AtWA_y = A_y.T @ W_y @ A_y
            AtWb_y = A_y.T @ W_y @ b_y

            # Solve for shifts
            shifts_x_unknowns = np.linalg.solve(AtWA_x, AtWb_x)
            shifts_y_unknowns = np.linalg.solve(AtWA_y, AtWb_y)

            # Calculate uncertainties from covariance matrix
            # Covariance = (A^T W A)^-1
            cov_x = np.linalg.inv(AtWA_x)
            cov_y = np.linalg.inv(AtWA_y)

            # Diagonal elements give variances, sqrt gives standard errors
            shift_x_uncertainties[1:] = np.sqrt(np.diag(cov_x))
            shift_y_uncertainties[1:] = np.sqrt(np.diag(cov_y))

        else:
            shifts_x_unknowns = np.array([])
            shifts_y_unknowns = np.array([])
    except (np.linalg.LinAlgError, np.linalg.LinAlgError):
        logger.warning(
            "Failed to solve weighted least squares system, "
            "using first valid pairwise shift"
        )
        # Fallback: use first available pairwise shift
        (i, j), (shift_x, shift_y) = next(iter(pairwise_shifts.items()))
        shifts_x_unknowns = np.zeros(n_unknowns)
        shifts_y_unknowns = np.zeros(n_unknowns)
        if j > 0:
            shifts_x_unknowns[j - 1] = shift_x
            shifts_y_unknowns[j - 1] = shift_y

        # Use fallback uncertainty if available
        if pairwise_uncertainties is not None:
            uncertainty_info = pairwise_uncertainties.get((i, j), {})
            if j > 0:
                shift_x_uncertainties[j] = uncertainty_info.get(
                    "shift_x_uncertainty", 1.0
                )
                shift_y_uncertainties[j] = uncertainty_info.get(
                    "shift_y_uncertainty", 1.0
                )

    # Reconstruct full shift arrays (with reference channel = 0)
    shifts_x = np.zeros(n_channels)
    shifts_y = np.zeros(n_channels)

    if n_unknowns > 0:
        shifts_x[1:] = shifts_x_unknowns
        shifts_y[1:] = shifts_y_unknowns

    # Create uncertainty summary
    shift_uncertainties = {
        "shift_x_uncertainties": shift_x_uncertainties,
        "shift_y_uncertainties": shift_y_uncertainties,
        "mean_x_uncertainty": (
            np.mean(shift_x_uncertainties[shift_x_uncertainties > 0])
            if np.any(shift_x_uncertainties > 0)
            else np.nan
        ),
        "mean_y_uncertainty": (
            np.mean(shift_y_uncertainties[shift_y_uncertainties > 0])
            if np.any(shift_y_uncertainties > 0)
            else np.nan
        ),
        "max_x_uncertainty": (
            np.max(shift_x_uncertainties)
            if len(shift_x_uncertainties) > 0
            else np.nan
        ),
        "max_y_uncertainty": (
            np.max(shift_y_uncertainties)
            if len(shift_y_uncertainties) > 0
            else np.nan
        ),
        "pairwise_uncertainties": pairwise_uncertainties or {},
    }

    return shifts_x, shifts_y, shift_uncertainties


def convert_zeiss_file(filepath_czi, filepath_raw, info=None):
    """Convert Zeiss .czi file into a picasso-readable .raw file.
    Args:
        filepath_csi : str
            the filepath to the .czi file to load
        filepath_raw : str
            the filepath to the .raw file to write
        info : dict, default None
            the metadata to make the raw file picasso-readable.
            If None is given, dummy values are entered.
            Necesary keys:
                'Byte Order', 'Camera', 'Micro-Manager Metadata'
    """
    img = AICSImage(filepath_czi)

    with open(filepath_raw, "wb") as f:
        img.get_image_data().squeeze().tofile(f)

    if info is None:
        info = {"Byte Order": "<", "Camera": "FusionBT"}
        info["File"] = filepath_raw
        info["Height"] = img.get_image_data().shape[-2]
        info["Width"] = img.get_image_data().shape[-1]
        info["Frames"] = img.get_image_data().shape[0]
        info["Data Type"] = img.get_image_data().dtype.name
        info["Micro-Manager Metadata"] = {
            "FusionBT-ReadoutMode": 1,
            "Filter": 561,
        }

    filepath_info = os.path.splitext(filepath_raw)[0] + ".yaml"

    with open(filepath_info, "w") as f:
        yaml.dump(info, f)


#############################################################################
# for plotting single spots in analyse.AutoPicasso.
#############################################################################


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return localize._to_photons(spots, camera_info)


def _cut_spots(movie, ids, box):
    N = len(ids.frame)
    spots = np.zeros((N, box, box), dtype=movie.dtype)
    spots = _cut_spots_byrandomframe(
        movie, ids.frame, ids.x, ids.y, box, spots
    )
    return spots


def _cut_spots_byrandomframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie by non-sorted frames.

    Args:
        movie : AbstractPicassoMovie (t, x, y)
            the image data
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    for j, (fr, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        frame = movie[fr]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


def normalize_spot(spot, maxval=255, dtype=np.uint8):
    # logger.debug('spot input: ' + str(spot))
    sp = spot - np.min(spot)
    imgmax = np.max(sp)
    imgmax = 1 if imgmax == 0 else imgmax
    sp = sp.astype(np.float32) / imgmax * maxval
    # logger.debug('spot output: ' + str(sp.astype(dtype)))
    return sp.astype(dtype)


def spinna_batch(parameters_filename):
    """This function runs a spinna batch analysis from file,
    as run via command line in picasso.__main__.

    picasso's ``_spinna_batch_analysis`` derives the result directory
    from the parameters filename (appending an index if a directory of
    that name already exists) and saves the summary csv and the NND
    figures there, but it does not return any paths. They are therefore
    reconstructed here with the same logic picasso uses internally.

    Args:
        parameters_filename : str
            the filepath of the spinna batch analysis config (.csv) file

    Returns:
        result_dir : str
            folder containing the results
        fp_summary : str
            the filepath of the summary csv file
        fp_fig : list of str
            filepaths of the NND figures
    """
    result_dir = parameters_filename.replace(".csv", "_fitting_results")
    if os.path.isdir(result_dir):
        i = 1
        while os.path.isdir(f"{result_dir}_{i}"):
            i += 1
        result_dir = f"{result_dir}_{i}"

    spinna_batch_analysis(parameters_filename)

    fp_summary = os.path.join(result_dir, "summary_results.csv")
    fp_fig = sorted(glob.glob(os.path.join(result_dir, "*_NND_*.png")))
    return result_dir, fp_summary, fp_fig


def single_spinna_run(
    structures,
    label_unc,
    le,
    mask_dict,
    width,
    height,
    depth,
    random_rot_mode,
    exp_data,
    sim_repeats,
    NND_bin,
    NND_maxdist,
    N_structures,
    save_filename,
    asynch,
    targets,
    apply_mask,
    nn_plotted,
    result_dir,
    n_simulated,
    bootstrap=False,
):
    """This function directly runs one spinna simulation.
    The implementation is taken from spinna batch analysis
    (picasso.__main__._spinna_batch_analysis), and adapted for one run,
    with parameters directly given.

    Args:
        parameters : dict with keys:
            structures, label_unc, le, mask_dict, width, height, depth,
            random_rot_mode, exp_data, sim_repeats, fit_NND_bin,
            fit_NND_maxdist, N_structures, save_filename, asynch, targets,
            apply_mask, nn_plotted, result_dir
    Returns:
        spinna_result : dict
            dictionary containing the results of the spinna run
        fp_fig : list of str
            filepaths of the NND figures
    """
    mixer = spinna.StructureMixer(
        structures=structures,
        label_unc=label_unc,
        le=le,
        mask_dict=mask_dict,
        width=width,
        height=height,
        depth=depth,
        random_rot_mode=random_rot_mode,
    )

    # set up and run fitting
    opt_props, score = spinna.SPINNA(
        mixer=mixer,
        gt_coords=exp_data,
        N_sim=sim_repeats,
    ).fit_stoichiometry(
        N_structures,
        save=f"{save_filename}_fit_scores.csv",
        asynch=asynch,
        bootstrap=bootstrap,
    )

    # save the results
    results = {}
    results["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["File location of structures"] = save_filename
    results["Molecular targets"] = targets
    results["Labelling efficiency (%)"] = [
        le[target] * 100 for target in targets
    ]
    results["Label uncertainty (nm)"] = list(label_unc.values())
    results["Rotation mode"] = random_rot_mode
    results["AICc fitting score"] = score
    results["Fitted structures names"] = list(N_structures.keys())
    if isinstance(opt_props, tuple):
        props_mean, props_std = opt_props
        results["Modified Kolmogorov-Smirnov score +/- s.d."] = score
        results["Fitted proportions of structures"] = ", ".join(
            [
                f"{props_mean[i]:.2f} +/- {props_std[i]:.2f}%"
                for i in range(len(props_mean))
            ]
        )
        results["props"] = props_mean
        results["props_std"] = props_std
    else:
        results["Modified Kolmogorov-Smirnov score"] = score
        results["Fitted proportions of structures"] = opt_props
        results["props"] = opt_props
        results["props_std"] = [0] * len(opt_props)
    results["NND bin size (nm)"] = NND_bin
    results["NND max distance (nm)"] = NND_maxdist

    # relative proportions of structures for each target
    if len(targets) > 1:
        for target in targets:
            if isinstance(opt_props, tuple):
                rel_props = mixer.convert_props_for_target(
                    opt_props[0],
                    target,
                    n_simulated,
                )
                # rel_props_sd = mixer.convert_props_for_target(
                #     opt_props[1], target, n_simulated,
                # )
            else:
                rel_props = mixer.convert_props_for_target(
                    opt_props,
                    target,
                    n_simulated,
                )
            idx_valid = np.where(rel_props != np.inf)[0]
            value = ", ".join(
                [
                    f"{structures[i].title}: {rel_props[i]:.2f}%"
                    for i in idx_valid
                ]
            )
            results[f"Relative proportions of {target} in"] = value

    # save .txt with summary of the results
    with open(f"{save_filename}_fit_summary.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    # print(f"Results saved to {save_filename}_fit_summary.txt")

    # plot and save the NND plots
    nn_counts = {}
    for i, t1 in enumerate(targets):
        for t2 in targets[i:]:
            nn_counts[f"{t1}-{t2}"] = nn_plotted
    mixer.nn_counts = nn_counts
    # dist_sim = spinna.get_NN_dist_repeated(
    # # dist_sim = get_NN_dist_repeated(
    #     opt_N_str, sim_repeats, mixer,
    #     duplicate=True
    # )
    n_total = sum(n_simulated.values())
    if isinstance(opt_props, tuple):
        opt_prop_vals = opt_props[0]
    else:
        opt_prop_vals = opt_props
    dist_sim = spinna.get_NN_dist_simulated(
        mixer.convert_props_to_counts(opt_prop_vals, n_total),
        sim_repeats,
        mixer,
        duplicate=True,
    )
    fp_fig = []
    for i, (t1, t2, _) in enumerate(mixer.get_neighbor_idx(duplicate=True)):
        # fig, ax = plot_NN(
        fig, ax = spinna.plot_NN(
            dist=dist_sim[i],
            mode="plot",
            show_legend=False,
            return_fig=True,
            figsize=(5.5, 4),
            alpha=1.0,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
        )
        exp1 = exp_data[t1]
        exp2 = exp_data[t2]
        # fig, ax = plot_NN(
        fig, ax = spinna.plot_NN(
            data1=exp1,
            data2=exp2,
            n_neighbors=nn_plotted,
            show_legend=False,
            fig=fig,
            ax=ax,
            mode="hist",
            return_fig=True,
            binsize=NND_bin,
            xlim=[0, NND_maxdist],
            title=f"Nearest Neighbors Distances: {t1} -> {t2}",
            savefig=[
                f"{save_filename}_NND_{t1}_{t2}.{_}" for _ in ["png", "svg"]
            ],
        )
        fp_fig.append(
            os.path.join(result_dir, f"{save_filename}_NND_{t1}_{t2}.png")
        )

    return results, fp_fig


def load_structures_from_dict(structure_dict):
    """Loads structures (SingleStructure's) from dict with format as
    those saved in .yaml files.

    Parameters
    ----------
    structure_dict : list of dict
        structure description as dict.

    Returns
    -------
    structures : list of SingleStructure's
        List of structures loaded from the file.
    targets : list of strs
        List of all unique molecular targets in the structures.
    """
    if "Structure title" not in structure_dict[0].keys():
        raise TypeError(
            "Incorrect file. Please choose a file that was created"
            " that was created with Picasso SPINNA."
        )
    # continue if the correct file is loaded
    structures = []
    targets = []
    for m_info in structure_dict:
        structure = spinna.Structure(m_info["Structure title"])
        for target in m_info["Molecular targets"]:
            x = m_info[f"{target}_x"]
            y = m_info[f"{target}_y"]
            z = m_info[f"{target}_z"]
            structure.define_coordinates(target, x, y, z)
            if target not in targets:
                targets.append(target)
        structures.append(structure)
    return structures, targets


def generate_N_structures(structures, N_total, res_factor, save=""):
    return spinna.generate_N_structures(
        structures,
        N_total,
        res_factor,
        save="",
    )


########################################################################
# Begin Log likelihood CSR estimation
########################################################################


def estimate_density_from_neighbordists(
    nn_dists,
    rho_init,
    kmin=1,
    rho_bound_factor=10,
    d=2,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
    fit_bkg=False,
):
    """For one point with k nearest neighbor distances (all assumed from
    a CSR distribution), do a maximum likelihood estimation for the
    density.
    Args:
        nn_dists : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
        rho_init : float
            the initial estimation of density
        min_dist, max_dist : float
            ignore nn distances outside this range
        bkg_fraction : float
            the fraction of nn distances that are background, i.e. randomly
            distributed over all distances, independent on spot density
        fit_bkg : bool
            whether to take bkg_fraction as granted, or as an input fit value
    Returns:
        mle_rho : float
            the maximum likelihood estimate for the local density
            based on the nearest neighbor distances
    """
    bounds = [
        (rho_init / rho_bound_factor, rho_init * rho_bound_factor)
    ]  # rho must be positive
    if fit_bkg:
        x0 = [rho_init, bkg_fraction]
    else:
        x0 = [rho_init]
    mle_rho = minimize(
        minimization_loglike,
        x0=x0,
        args=(nn_dists, d, kmin, min_dist, max_dist, bkg_fraction),
        bounds=bounds,
        # tol=1e-8, options={'maxiter': 1e5}, method='Powell'
        # options={'maxiter': 1e5}, method='L-BFGS-B'
        # method='BFGS'#,
        # options={'maxiter': 1e5, 'gtol': 1e-6, 'eps': 1e-9},
        method="L-BFGS-B",
        options={
            "disp": None,
            "maxcor": 10,
            "ftol": 2e-15,
            "gtol": 1e-15,
            "eps": 1e-15,
            "maxfun": 150,
            "maxiter": 150,
            "iprint": -1,
            "maxls": 100,
            "finite_diff_rel_step": None,
        },
    )
    # print(mle_rho)
    return mle_rho.x[0], mle_rho


def minimization_loglike(
    parameters,
    nndist_observed,
    d=2,
    kmin=1,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
):
    """The minimization function for nndist loglikelihood fun
    based on k-th nearest neighbor CSR distributions
    Args:
        parameters : list, len 1 or 2
            the estimated density, and potentially bkg_fraciton
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
        bkg_fraction : the background fraction. ignored if there
            is a second entry of parameters
    Returns:
        loglike : float
            the log likelihood of finding the observed neighbor distances
            in the model of CSR and given rho
    """
    rho = parameters[0]
    if len(parameters) == 2:
        bkg_fraction = parameters[1]
    return -nndist_loglikelihood_csr(
        nndist_observed, rho, d, kmin, min_dist, max_dist, bkg_fraction
    )


def nndist_loglikelihood_csr(
    nndist_observed,
    rho,
    d=2,
    kmin=1,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
):
    """get the Log-Likelihood of observed nearest neighbors assuming
    a CSR distribution with density rho.
    Args:
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of one or N spots)
        rho : float
            the density
    Returns:
        log_like : float
            the log likelihood of all distances observed being drawn
            from CSR
    """
    log_like = 0
    # print("nndist_obs shape", nndist_observed.shape)
    for i, dist in enumerate(nndist_observed):
        k = i + kmin
        # assert False
        dist_consider = dist[(dist >= min_dist) & (dist <= max_dist)]
        prob = nndistribution_from_csr(
            dist_consider,
            k,
            rho,
            d=d,
            bkg_fraction=bkg_fraction,
            min_dist=min_dist,
            max_dist=max_dist,
        )
        # print(i, dist, prob, np.log(prob))
        log_like += np.sum(np.log(prob))
    return log_like


def nndistribution_from_csr(
    r,
    k,
    rho,
    d=2,
    bkg_fraction=0,
    min_dist=0,
    max_dist=np.inf,
    renormalize=True,
):
    """The CSR Nearest Neighbor distribution of finding the k-th nearest
    neighbor at r. with the spatial randomness covering d dimensions
    Args:
        r : float or array of floats
            the distance(s) to evaluate the probability density at
        k : int
            evaluation of the k-th nearest neighbor
        rho : float
            the density
        d : int
            the dimensionality of the problem
        min_dist : float
            the minimum distance observable (e.g. due to technical reasons),
            the model is cut off below that and renormalized
    Returns:
        p : same as r
            the probability density of k-th nearest neighbor at r
    """
    # if k != 1:
    #     print(f'evaluating CSR not at k=1 but k={k}')

    # def gaussian_pdf(x, mean, std):
    #     factor = (1 / (np.sqrt(2 * np.pi) * std))
    #     return factor * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # pdf = gaussian_pdf(r, 4, k*rho*4)
    # # pdf = gaussian_pdf(r, 4+k*rho, .8)
    # return pdf #/ np.sum(pdf)
    lam = rho * np.pi ** (d / 2) / _gamma(d / 2 + 1)
    factor = d / _factorial(k - 1) * lam**k * r ** (d * k - 1)
    dist = factor * np.exp(-lam * r**d)
    # add am even background of observed nn distances, at all distances
    if len(r) > 1:
        dist += (bkg_fraction / (np.max(r) - np.min(r))) / (1 + bkg_fraction)
    dist[dist <= 0] = 1e-200  # np.finfo().eps
    # re-normalize
    if min_dist > 0 or max_dist < np.inf:
        r_temp = np.linspace(0, np.max(r), 300)
        prob = nndistribution_from_csr(
            r_temp,
            k,
            rho,
            d=d,
            bkg_fraction=bkg_fraction,
            min_dist=0,
            max_dist=np.inf,
        )
        renorm_fact = np.sum(
            prob[(r_temp >= min_dist) & (r_temp <= max_dist)]
        ) / np.sum(prob)
        # print(renorm_fact)
        dist[(r < min_dist) | (r > max_dist)] = 0
        if renormalize:
            dist = dist / renorm_fact
    # try:
    #     renorm_factor = np.sum(dist) / np.sum(dist[r >= min_dist])
    # except ZeroDivisionError:
    #     renorm_factor = 1
    # dist[r < min_dist] = 0
    # dist *= renorm_factor
    return dist  # / np.sum(dist)


def csr_cdf_for_ks_test(
    x, k, rho, d=2, min_dist=0, max_dist=np.inf, bkg_fraction=0
):
    """CDF of the theoretical CSR distribution for k-th nearest neighbor.

    Used for Kolmogorov-Smirnov goodness-of-fit testing.

    Args:
        x : float or array-like
            Distance values to evaluate CDF at
        k : int
            k-th nearest neighbor order
        rho : float
            Density parameter from CSR fit
        d : int, default 2
            Dimensionality (2D or 3D)
        min_dist : float, default 0
            Minimum observable distance
        max_dist : float, default np.inf
            Maximum observable distance
        bkg_fraction : float, default 0
            Background fraction parameter

    Returns:
        float or array
            CDF values at input distances
    """
    x = np.atleast_1d(x)
    cdf_values = np.zeros_like(x, dtype=float)

    for i, xi in enumerate(x):
        if xi <= min_dist:
            cdf_values[i] = 0.0
        elif xi >= max_dist:
            cdf_values[i] = 1.0
        else:
            # Calculate CDF by integrating PDF up to xi
            r_integrate = np.linspace(
                min_dist, xi, num=min(1000, int(xi * 10))
            )
            if len(r_integrate) > 1:
                pdf_vals = nndistribution_from_csr(
                    r_integrate,
                    k,
                    rho,
                    d=d,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    bkg_fraction=bkg_fraction,
                    renormalize=True,
                )
                # Numerical integration using trapezoidal rule
                cdf_values[i] = np.trapz(pdf_vals, r_integrate)
            else:
                cdf_values[i] = 0.0

    return cdf_values if len(cdf_values) > 1 else cdf_values[0]


########################################################################
# End Log likelihood CSR estimation
########################################################################


########################################################################
# Start Molecular Interaction Patterns (Joschka)
########################################################################


def DBSCAN_analysis(clusters_csv):
    """Calculates barcodes and weights (by cluster area) for further
    DBSCAN data analysis.

    Parameters
    ----------
    clusters_csv : str or pd.DataFrame
        Path to csv file with DBSCAN results

    Returns
    -------
    barcodes : np.array
        Array of shape (N, 6) with binary barcodes for each of
        N DBSCAN clusters
    weights : np.array
        Array of shape (N,) with weights for each of N DBSCAN
        clusters
    """

    if isinstance(clusters_csv, str):
        clusters = pd.read_csv(clusters_csv)  # DBSCAN data
    elif isinstance(clusters_csv, pd.DataFrame):
        clusters = clusters_csv
    else:
        raise NotImplementedError("Type of clusters_csv not implemented.")

    columns = [
        "N_MHC-I_per_cluster",
        "N_MHC-II_per_cluster",
        "N_CD86_per_cluster",
        "N_CD80_per_cluster",
        "N_PDL1_per_cluster",
        "N_PDL2_per_cluster",
        "area (nm^2)",
    ]
    clusters = clusters[columns]

    areas = clusters.values[:, -1]  # this is for weights later

    # find the all or none (binary) barcodes
    clusters = clusters[columns[:-1]]
    idx = np.where(clusters.values > 0)
    barcodes = clusters.values.copy()
    barcodes[idx] = 1

    weights = areas

    return barcodes, weights


def DBSCAN_analysis_pd(clusters_csv, channel_tags):
    """Calculates barcodes and weights (by cluster area) for further
    DBSCAN data analysis.

    Parameters
    ----------
    clusters_csv : str or pd.DataFrame
        Path to csv file with DBSCAN results
    channel_tags : list of str
        the names of the channels, in the correct order

    Returns
    -------
    barcodes_df : pd.DataFrame, columns 'barcode', 'weight'
        Array of shape (N, 6) with binary barcodes for each of
        N DBSCAN clusters
    barcodes_agg : pd.DataFrame, columns are barcodes
        Descriptive aggregation of barcodes_df. with indexes
        count, mean, std, 25%, 50%, 75%
    barcode_map : pd.DataFrame
        index: arange
        cols: 'barcode': string with the binary barcode
            {target}_per_cluster:
    """

    if isinstance(clusters_csv, str):
        clusters = pd.read_csv(clusters_csv)  # DBSCAN data
    elif isinstance(clusters_csv, pd.DataFrame):
        clusters = clusters_csv
    else:
        raise NotImplementedError("Type of clusters_csv not implemented.")

    # per_cluster_cols = [
    #     col for col in clusters.columns
    #     if col.endswith("_per_cluster") and col.startswith('N_')
    # ]
    targets = channel_tags
    # [col[: -len("_per_cluster")] for col in per_cluster_cols]
    # sort per cluster cols
    per_cluster_cols = [f"N_{target}_per_cluster" for target in targets]

    barcode_df = pd.DataFrame(
        index=clusters.index,
        columns=["barcode", "area (nm^2)"] + per_cluster_cols,
    )

    def decimal_to_binary(decimal, digits):
        binstr = bin(decimal)
        if len(binstr) - 2 < digits:
            binstr = (
                binstr[:2] + "0" * (digits - (len(binstr) - 2)) + binstr[2:]
            )
        return binstr

    def assemble_barcode(df, cols):
        barcode = np.zeros(len(df.index), dtype=np.int32)
        for i, col in enumerate(cols[::-1]):
            barcode += 2**i * (df[col] > 0)
        return barcode

    barcode_df["barcode"] = np.vectorize(decimal_to_binary)(
        assemble_barcode(clusters, per_cluster_cols), len(targets)
    )
    barcode_df["area (nm^2)"] = clusters["area (nm^2)"]
    for col in per_cluster_cols:
        barcode_df.loc[:, col] = clusters[col]

    barcodes_agg = barcode_df.groupby("barcode").describe()
    # print(barcodes_agg.columns)

    barcode_map = pd.DataFrame(
        index=np.arange(2 ** len(targets)),
        columns=["barcode"] + targets,
    )

    barcode_map["barcode"] = np.vectorize(decimal_to_binary)(
        barcode_map.index, len(targets)
    )
    for i, col in enumerate(targets):
        barcode_map[col] = (barcode_map["barcode"].str[2 + i] == "1").astype(
            np.int32
        )

    return barcode_df, barcodes_agg, barcode_map


def _do_dbscan_molint(
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
        f"dbscan_{epsilon_nm:.0f}_{minpts}_{it}" + "_protein_colorcode.hdf5",
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
    barcodes, weights = DBSCAN_analysis(db_cluster_output)
    filepaths["fp_binary_barcode"] = os.path.join(
        result_folder, "binary_barcode.txt"
    )
    np.savetxt(filepaths["fp_binary_barcode"], barcodes)
    filepaths["fp_binary_barcode_weights"] = os.path.join(
        result_folder, "binary_barcode_weights.txt"
    )
    np.savetxt(filepaths["fp_binary_barcode_weights"], weights)

    # perform adapteation of Rafal's analysis
    channel_map_r = {v: k for k, v in channel_map.items()}
    targets = [channel_map_r[i] for i in sorted(channel_map_r.keys())]
    (barcode_df, barcode_agg, barcode_map) = DBSCAN_analysis_pd(
        db_cluster_output, targets
    )
    filepaths["fp_barcode"] = os.path.join(result_folder, f"barcode_{it}.xlsx")
    barcode_df.to_excel(filepaths["fp_barcode"])
    filepaths["fp_barcode_agg"] = os.path.join(
        result_folder, f"barcode_described_{it}.xlsx"
    )
    barcode_agg.to_excel(filepaths["fp_barcode_agg"])
    filepaths["fp_barcode_map"] = os.path.join(
        result_folder, f"barcode_map_{it}.xlsx"
    )
    barcode_map.to_excel(filepaths["fp_barcode_map"])

    # number of nonclustered
    cluster_info = {
        "n_input_locs": len(df_mask.index),
        "n_clustered_locs": len(db_locs_df.index),
        "n_nonclustered_locs": len(df_mask.index) - len(db_locs_df.index),
        "n_clusters": len(db_cluster_props_df.index),
    }
    fp_cluster_info = os.path.join(result_folder, f"cluster_info_{it}.yaml")
    filepaths["fp_cluster_info"] = fp_cluster_info
    with open(fp_cluster_info, "w") as f:
        yaml.dump(cluster_info, f)

    # plot results
    fig, ax = plt.subplots(nrows=3, sharex=True)
    # barplot: number of clusters
    ax[0].bar(
        np.arange(len(barcode_agg.index)),
        barcode_agg[("area (nm^2)", "count")],
    )
    ax[0].set_ylabel("# clusters found")

    fig.set_size_inches((13, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, "barcodes.png")
    fig.savefig(filepaths["fp_fig"])

    # boxplot: area distribution of clusters
    # sns.boxplot(data=barcode_df, x='barcode', y='area (nm^2)', ax=ax[1])
    dflist = [
        subdf.values
        for idx, subdf in barcode_df.groupby("barcode")["area (nm^2)"]
    ]
    ax[1].boxplot(
        dflist,
        positions=np.arange(len(barcode_agg.index)),
        showfliers=False,
    )
    ax[1].set_ylabel("area per cluster (nm^2)")
    # boxplots: number of targets per cluster, for each target

    fig.set_size_inches((13, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, "barcodes.png")
    fig.savefig(filepaths["fp_fig"])

    bxwidth = 1 / (len(targets) + 2)
    bxpos_init = np.arange(len(barcode_agg.index)) - bxwidth * len(targets) / 2

    target_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    legend_handles = []
    import matplotlib.lines as mlines

    for i, tgt in enumerate(targets):
        col = f"N_{tgt}_per_cluster"
        dflist = [
            subdf.values if idx[2 + i] == "1" else np.array([])
            for idx, subdf in barcode_df.groupby("barcode")[col]
        ]
        # remove values (zeros) if target is not in barcode
        lineprops = {"color": target_colors[i]}
        ax[2].boxplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showfliers=False,
            boxprops=lineprops,
            whiskerprops=lineprops,
            medianprops=lineprops,
            capprops=lineprops,
        )
        line = mlines.Line2D([], [], color=target_colors[i], label=tgt)
        legend_handles.append(line)
    ax[2].set_ylabel("# targets per cluster")
    ax[2].set_xticks(np.arange(len(barcode_agg.index)))
    xtilabels = [ti[2:] for ti in barcode_agg.index]
    ax[2].set_xticklabels(xtilabels, rotation=90)
    # plot separator lines
    xpos = np.arange(len(barcode_agg.index) - 1) + 0.5
    ylims = ax[2].get_ylim()
    for x in xpos:
        ax[2].plot([x, x], ylims, color="gray")
    ax[2].legend(handles=legend_handles)
    fig.set_size_inches((15, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, f"barcodes_{it}.png")
    fig.savefig(filepaths["fp_fig"])
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


def degree_of_clustering(
    cluster_info_exp, cluster_info_csr, origin_colors, folder
):
    # plot number of clustered vs non-clustered locs
    data = {
        "exp": [
            cluster_info_exp["n_clustered_locs"],
            cluster_info_exp["n_nonclustered_locs"],
        ],
        "csr": [
            cluster_info_csr["n_clustered_locs"],
            cluster_info_csr["n_nonclustered_locs"],
        ],
    }
    fp_fig_dog = os.path.join(folder, "degree_of_clustering.png")
    _ = _plot_degreeofclustering(
        data, origin_colors, fp_fig_dog, ylabel="# locs per cell"
    )

    # plot fraction of clustered vs non-clustered locs
    data_fract = {
        "exp": [
            np.array(cluster_info_exp["n_clustered_locs"])
            / (
                np.array(cluster_info_exp["n_clustered_locs"])
                + np.array(cluster_info_exp["n_nonclustered_locs"])
            ),
            np.array(cluster_info_exp["n_nonclustered_locs"])
            / (
                np.array(cluster_info_exp["n_clustered_locs"])
                + np.array(cluster_info_exp["n_nonclustered_locs"])
            ),
        ],
        "csr": [
            np.array(cluster_info_csr["n_clustered_locs"])
            / (
                np.array(cluster_info_csr["n_clustered_locs"])
                + np.array(cluster_info_csr["n_nonclustered_locs"])
            ),
            np.array(cluster_info_csr["n_nonclustered_locs"])
            / (
                np.array(cluster_info_csr["n_clustered_locs"])
                + np.array(cluster_info_csr["n_nonclustered_locs"])
            ),
        ],
    }
    fp_fig_tractdog = os.path.join(folder, "fracdegree_of_clustering.png")
    _ = _plot_degreeofclustering(
        data_fract,
        origin_colors,
        fp_fig_tractdog,
        ylabel="fraction of locs per cell",
    )

    return [fp_fig_dog, fp_fig_tractdog]


def _plot_degreeofclustering(
    data, origin_colors, fp_fig, ylabel="fraction of locs per cell"
):
    """
    Plot the degree of clustering of experimental versus simulated
    data in violin plots, including stripplots of the data.
    Args:
        data: dict of array
            the underlying data to plot (numer/fraction of clustered
            or unclustered locs for each cell)
            keys: 'exp' and 'csr'
        origin_colors : list of str
            the colors of 'exp', and 'csr' data, respectively
        fp_fig : str
            the path to save the figure at
    Returns:
        t_stats : xyz
            the results of the t_test between experimental and csr
            data for clustered and non-clustered data comparison
        p_values : array len 2
            the p_values of exp and csr being drawn from the same
            distribution for clustered and non-clustered data
    """
    categories = ["clustered", "non-clustered"]

    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(categories))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    legend_handles = []

    fig, ax = plt.subplots()
    for i, org in enumerate(["exp", "csr"]):
        parts = ax.violinplot(
            data[org],
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(origin_colors[i])
            pc.set_edgecolor(origin_colors[i])
        util.stripplot(
            data[org],
            bxpos_init + i * bxwidth,
            bxwidth,
            ax,
            origin_colors[i],
            alpha=0.5,
        )
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    # test for significance
    ylims = ax.get_ylim()
    p_values = np.ones(2)
    t_stats = np.ones(2)
    for i, (n_exp, n_csr) in enumerate(zip(data["exp"], data["csr"])):
        t_stats[i], p_values[i] = stats.ttest_ind(n_exp, n_csr)
        if p_values[i] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[i] < 1e-2:
            siglabel = "p < 0.01"
        else:
            siglabel = "n.s."
        ax.text(
            i,
            0.8 * ylims[1],
            siglabel,
            fontsize=14,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)  # , rotation=90)
    ax.set_title("degree of clustering")
    ax.legend(handles=legend_handles)
    fig.set_size_inches((8, 5))
    fig.savefig(fp_fig)
    plt.close(fig)
    return t_stats, p_values


def _plot_and_compare_barcodes(
    pivot_table,
    origin_colors,
    targets,
    ttest_pvalue_max,
    population_threshold,
    cellfraction_threshold,
    fp_fig,
    title="",
    ylabel="",
):
    """Plot the comparison of barcodes between experiment and simulation,
    and perform a t-test to evaluate whether the distributions are
    different.
    Args:
        pivot_table : pd.DataFrame
            index: barcodes (str, 0b...)
            columns: multiindex, first index: origin (['exp', 'csr'])
        origin_colors : list of str
            the colors to use for the two conditions
        targets : list of str
            the protein targets
        ttest_pvalue_max : float
            the pvalue above which no significance is attributed to
            the difference of exp and csr
        population_threshold : float
            the relative population a barcode needs to be significant
            (e.g. 1% of all clusters need to have a barcode for it
            to pop up)
        population_threshold : float, between 0 and 1
            the fraction of cells that need to have this barcode at least once.
        fp_fig : str
            the filepath to save the figure at
        title : str
            the title addition for the plot
    Returns:
        significant_barcodes : list of str
            the barcodes that evaluated as significantly changed between
            exp and csr
        p_values : list of float
            the p_values of the t-test for all barcodes
    """
    # plot distribution of number of barcodes
    fig, ax = plt.subplots(nrows=1, sharex=True)

    legend_handles = []
    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(pivot_table.index))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    all_occurrence_lists = {}
    for i, org in enumerate(["exp", "csr"]):
        dflist = [row[org].values for bc, row in pivot_table.iterrows()]
        all_occurrence_lists[org] = dflist
        # remove values (zeros) if target is not in barcode
        lineprops = {"color": origin_colors[i]}
        ax.boxplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showfliers=False,
            boxprops=lineprops,
            whiskerprops=lineprops,
            medianprops=lineprops,
            capprops=lineprops,
        )
        # print(bplot.keys())
        # for patch in bplot['boxes']:
        #     patch.set_edgecolor(target_colors[i])
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}; barcoding: " + "-".join(targets))
    ax.set_xticks(np.arange(len(pivot_table.index)))
    xtilabels = [ti[2:] for ti in pivot_table.index]
    ax.set_xticklabels(xtilabels, rotation=90)
    # # plot separator lines
    # xpos = np.arange(len(barcode_numbers.index) - 1) + .5
    ylims = ax.get_ylim()
    ax.set_ylim([0, ylims[1]])
    # for x in xpos:
    #     ax.plot([x, x], ylims, color='gray')
    ax.legend(handles=legend_handles)

    # test for significant difference in the number of barcodes found
    # between exp and csr
    p_values = np.ones(len(pivot_table.index))
    t_stats = np.ones(len(pivot_table.index))
    for i, (n_exp, n_csr) in enumerate(
        zip(all_occurrence_lists["exp"], all_occurrence_lists["csr"])
    ):
        t_stats[i], p_values[i] = stats.ttest_ind(n_exp, n_csr)

    significant_barcodes_idx = np.argwhere(
        p_values < ttest_pvalue_max
    ).flatten()
    # print(significant_barcodes_idx)

    # select for barcodes that have a relevant population
    fraction_barcodes_exp = np.array(
        [sum(occ) for occ in all_occurrence_lists["exp"]], dtype=np.float64
    )
    fraction_barcodes_exp /= np.sum(fraction_barcodes_exp)
    # print(fraction_barcodes_exp)
    relevant_barcodes_idx = np.argwhere(
        fraction_barcodes_exp > population_threshold
    ).flatten()

    # select for barcodes that occur in a given fraciton of cells at
    # least once
    cell_fraction_barcodes = np.array(
        [sum(occ > 0) / len(occ) for occ in all_occurrence_lists["exp"]],
        dtype=np.float64,
    )
    enough_cells_have_barcodes_idx = np.argwhere(
        cell_fraction_barcodes > cellfraction_threshold
    ).flatten()
    # print(relevant_barcodes_idx)
    significant_barcodes_idx = [
        idx
        for idx in significant_barcodes_idx
        if (
            idx in relevant_barcodes_idx
            and idx in enough_cells_have_barcodes_idx
        )
    ]

    for pos in significant_barcodes_idx:
        if p_values[pos] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[pos] < 1e-2:
            siglabel = "p < 0.01"
        elif p_values[pos] < ttest_pvalue_max:
            siglabel = f"p < {ttest_pvalue_max:.2f}"
        else:
            siglabel = "n.s."
        ax.text(
            pos,
            0.8 * ylims[1],
            siglabel,
            fontsize=10,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
        )

    significant_barcodes = [
        pivot_table.index[i] for i in significant_barcodes_idx
    ]

    fig.set_size_inches((15, 6))
    fig.savefig(fp_fig)
    plt.close(fig)

    return significant_barcodes, p_values


def _plot_and_compare_ntargets_in_barcodes(
    df, bc, origin_colors, targets, fp_fig
):
    """For a significant cluster, plot the distribution of
    number of targets for exp and csr cases, and determine
    whether they are stastistically differnt
    Args:
        df : DataFrame
            the list of all clusters with this barcode
        bc : str
            the barcode ('0b...')
        origin_colors : list of str
            the colors for exp and csr
        targets : list of str
            the protein target names
        fp_fig : str
            the filepath to save the figure as
    """
    fig, ax = plt.subplots()
    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(targets))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    legend_handles = []

    pts = {}
    ntgt_data = {}
    for i, tgt in enumerate(targets):
        pivot_table = pd.pivot_table(
            df[["origin", "name", "iter", f"N_{tgt}_per_cluster"]],
            index="origin",
            columns=["name", "iter"],
            values=f"N_{tgt}_per_cluster",
            aggfunc="mean",
            fill_value=np.nan,
        )
        # average over 'iter'
        pivot_table = pivot_table.T.groupby(level=["name"]).mean().T
        # fp = os.path.join(os.path.split(fp_fig)[0], f"bc{bc}-{tgt}.xlsx")
        # print(fp)
        # pivot_table.to_excel(fp)
        pts[tgt] = pivot_table
        pts_exp = pts[tgt].loc["exp", :].values.flatten()
        pts_csr = pts[tgt].loc["csr", :].values.flatten()
        ntgt_data[tgt] = {
            "exp": pts_exp[~np.isnan(pts_exp)],
            "csr": pts_csr[~np.isnan(pts_csr)],
        }
    for i, org in enumerate(["exp", "csr"]):
        # subdf = df.loc[df["origin"] == org]
        dflist = [
            (
                # subdf.groupby()[f"N_{tgt}_per_cluster"]
                # pts[tgt].loc[org, :].values
                ntgt_data[tgt][org]
                if bc[2 + k] == "1"
                else np.array([np.nan] * 3)
            )
            for k, tgt in enumerate(targets)
        ]
        parts = ax.violinplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showmedians=True,
            showextrema=False,
            # quantiles=[.25, .75]
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(origin_colors[i])
            pc.set_edgecolor(origin_colors[i])
        util.stripplot(
            dflist,
            bxpos_init + i * bxwidth,
            bxwidth,
            ax,
            origin_colors[i],
            alpha=0.2,
        )
        # lineprops = {"color": origin_colors[i]}
        # ax.boxplot(
        #     dflist,
        #     positions=bxpos_init + i * bxwidth,
        #     widths=bxwidth,
        #     showfliers=False,
        #     boxprops=lineprops,
        #     whiskerprops=lineprops,
        #     medianprops=lineprops,
        #     capprops=lineprops,
        # )
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    ax.set_ylabel("# targets per cluster")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=90)
    ax.legend(handles=legend_handles)

    # evaluate statistical difference
    p_values = np.ones(len(targets))
    t_stats = np.ones(len(targets))
    ylims = ax.get_ylim()
    ax.set_ylim([0, ylims[1]])
    for i, tgt in enumerate(targets):
        if bc[2 + i] != "1":
            continue
        # exp_data = df.loc[df["origin"] == "exp", f"N_{tgt}_per_cluster"]
        # csr_data = df.loc[df["origin"] == "csr", f"N_{tgt}_per_cluster"]
        t_stats[i], p_values[i] = stats.ttest_ind(
            ntgt_data[tgt]["exp"], ntgt_data[tgt]["csr"]
        )
        if p_values[i] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[i] < 1e-2:
            siglabel = "p < 0.01"
        else:
            siglabel = "n.s."
        # print(f'{bc} target {targets[i]} p-values: ', p_values[i])
        ax.text(
            i,
            0.8 * ylims[1],
            siglabel,
            fontsize=12,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # n_exp = np.sum(df["origin"] == "exp")
    # n_csr = np.sum(df["origin"] == "csr")
    # ax.set_title(
    #     f"Significantly altered barcode {bc[2:]}; "
    #     + f"data points exp {n_exp}, csr {n_csr}")
    ax.set_title(f"Significantly altered barcode {bc[2:]}")
    fig.set_size_inches((8, 5))
    fig.savefig(fp_fig)
    plt.close(fig)


def _plot_interaction_graph(
    node_sizes: np.ndarray,
    edge_sizes: np.ndarray,
    target_colors: list,
    targets: list,
):
    """Create an interaction graph plot to show both density of proteins
    (node sizes), and interaction strength (edge sizes)
    Args:
        node_sizes : np array (N,)
            the size of the nodes
        edge_sizes : np array (N, N)
            the interaction strength between edges, including self
    """
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    N = len(node_sizes)
    # Create a figure and a subplot
    fig, ax = plt.subplots()
    ax.set_xlim([-1.75, 1.75])
    ax.set_ylim([-1.75, 1.75])
    ax.set_aspect("equal")
    # Calculate the positions of the nodes
    theta = -np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta += np.pi * 2 / 3  # shift by 60 deg to match Joschka's positions
    x = np.cos(theta)
    y = np.sin(theta)
    xnode = np.cos(theta + np.pi * 1 / 18)
    ynode = np.sin(theta + np.pi * 1 / 18)
    # Draw the edges
    for i in range(N):
        for j in range(N):
            if i != j:
                # ax.plot(
                #     [x[i], x[j]], [y[i], y[j]],
                #     color='black', linewidth=edge_sizes[i][j])
                # pass
                offset_dist = np.abs(edge_sizes[i][j]) + np.abs(
                    edge_sizes[j][i]
                )
                offset_dir = np.array([(y[j] - y[i]), (x[j] - x[i])])
                offset_dir = offset_dir / np.sqrt(np.sum(offset_dir**2))
                offset = offset_dist * offset_dir
                trans = ax.transData
                coords_start = np.array([x[i], y[i]])
                coords_end = np.array([x[j], y[j]])
                coords_start_pt = trans.transform(coords_start)
                coords_end_pt = trans.transform(coords_end)
                coords_start = trans.inverted().transform(
                    coords_start_pt + offset
                )
                coords_end = trans.inverted().transform(coords_end_pt + offset)
                x_coords = [coords_start[0], coords_end[0]]
                y_coords = [coords_start[1], coords_end[1]]
                lineprops = dict(
                    color=target_colors[i],
                    linewidth=np.abs(edge_sizes[i][j]),
                    solid_capstyle="round",
                )
                if edge_sizes[j][i] < 0:
                    lineprops["linestyle"] = ":"
                line = Line2D(x_coords, y_coords, **lineprops)
                ax.add_line(line)
            else:
                # Draw a circular arrow for self-interaction
                start_angle = int(180 / np.pi * theta[i])
                aA = start_angle + 0
                aB = start_angle + 225
                ax.annotate(
                    "",
                    xy=(1.1 * x[i], 1.1 * y[i]),
                    xytext=(1.65 * x[i], 1.65 * y[i]),
                    arrowprops=dict(
                        arrowstyle="<-",
                        # connectionstyle="arc,rad=.7,angleA=0,angleB=225",
                        connectionstyle=f"angle3,angleA={aA},angleB={aB}",
                        linewidth=np.abs(edge_sizes[i][i]),
                        color=target_colors[i],
                    ),
                )
    # Draw the nodes
    for i in range(N):
        start_angle = 0
        end_angle = 360
        radius = np.sqrt(node_sizes[i])
        wedge = mpatches.Wedge(
            (x[i], y[i]),
            radius,
            start_angle,
            end_angle,
            facecolor=target_colors[i],
        )
        ax.add_patch(wedge)
        ax.text(
            1.5 * xnode[i],
            1.5 * ynode[i],
            targets[i],
            fontsize=14,
            color=target_colors[i],
            horizontalalignment="center",
            verticalalignment="center",
        )
    # ax.scatter(x, y, s=node_sizes, color='blue')
    ax.axis("off")

    return fig, ax


########################################################################
# End Molecular Interaction Patterns (Joschka)
########################################################################


########################################################################
# Start Labeling Efficiency Workflow Modules
########################################################################


def prep_pick_similar_kwargs(locs, info, diameter):
    d = diameter

    maxheight = info[0]["Height"]
    maxwidth = info[0]["Width"]
    r = d / 2
    d2 = d**2

    # extract n_locs and rmsd from current picks
    (locs_temp, r, _, _, block_starts, block_ends, K, L) = (
        postprocess.get_index_blocks(locs, info, r)
    )

    # x, y coordinates of found regions:
    x_similar = np.array([])
    y_similar = np.array([])

    # preparations for grid search
    x_range = np.arange(d / 2, maxwidth, np.sqrt(3) * d / 2)
    y_range_base = np.arange(d / 2, maxheight - d / 2, d)
    y_range_shift = y_range_base + d / 2

    locs_x = locs_temp.x
    locs_y = locs_temp.y
    locs_xy = np.stack((locs_x, locs_y))
    x_r = np.uint64(x_range / r)
    y_r1 = np.uint64(y_range_shift / r)
    y_r2 = np.uint64(y_range_base / r)
    kwargs = {
        "x": x_range,
        "y_shift": y_range_shift,
        "y_base": y_range_base,
        "x_r": x_r,
        "y_r1": y_r1,
        "y_r2": y_r2,
        "locs_xy": locs_xy,
        "block_starts": block_starts,
        "block_ends": block_ends,
        "K": K,
        "L": L,
        "x_similar": x_similar,
        "y_similar": y_similar,
        "r": r,
        "d2": d2,
    }
    return kwargs


def pick_gold(locs, info, diameter=2, std_range=1.4, mean_rmsd=0.4):
    """
    Searches picks similar to Gold clusters.

    Focuses on the number of locs and their root mean square
    displacement from center of mass. Std is defined in Tools
    Settings Dialog.

    Args:
        diameter : float
            the pick similar diameter
        std_range, mean_rmsd : float
            the pick similar parameters identifying gold
    Returns:
        similar : list of [x, y] position pairs
            the positions (picks) of gold beads

    Raises
    ------
    NotImplementedError
        If pick shape is rectangle
    """
    maxframe = info[0]["Frames"]

    # calculate min and max n_locs and rmsd for picking similar
    mean_n_locs = maxframe
    std_n_locs = 0.25 * mean_n_locs
    std_rmsd = 0.25 * mean_n_locs
    min_n_locs = mean_n_locs - std_range * std_n_locs
    max_n_locs = mean_n_locs + std_range * std_n_locs
    min_rmsd = mean_rmsd - std_range * std_rmsd
    max_rmsd = mean_rmsd + std_range * std_rmsd

    kwargs = prep_pick_similar_kwargs(locs, info, diameter)
    kwargs["min_n_locs"] = min_n_locs
    kwargs["max_n_locs"] = max_n_locs
    kwargs["min_rmsd"] = min_rmsd
    kwargs["max_rmsd"] = max_rmsd

    # pick similar
    x_similar, y_similar = postprocess._pick_similar(**kwargs)
    # add picks
    similar = list(zip(x_similar, y_similar))
    return similar


def index_locs(locs, info, pick_diameter):
    """
    Indexes localizations from a given channel in a grid with grid
    size equal to the pick radius.
    """
    d = pick_diameter
    size = d / 2
    index_blocks = postprocess.get_index_blocks(locs, info, size)
    return index_blocks


def get_block_locs_at(x, y, index_blocks, return_indices=False):
    """Copied from picasso.postprocess.get_block_locs_at.
    But the block indices are needed as well.
    """
    locs, size, _, _, block_starts, block_ends, K, L = index_blocks
    x_index = np.uint32(x / size)
    y_index = np.uint32(y / size)
    indices = []
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for li in range(x_index - 1, x_index + 2):
                if 0 <= li < L:
                    indices.append(
                        list(range(block_starts[k, li], block_ends[k, li]))
                    )
    indices = list(itertools.chain(*indices))
    if return_indices:
        return locs.iloc[indices], np.array(indices)
    else:
        return locs.iloc[indices]


def locs_at(x, y, locs, r, return_indices=False):
    """Returns localizations at position (x, y) within radius r.

    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    picked_locs : np.rec.array
        Localizations at position.
    """

    is_picked = lib.is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    if return_indices:
        return picked_locs, is_picked
    else:
        return picked_locs


def picked_locs(
    locs, info, _centers, pick_diameter, add_group=True, return_nonpicked=False
):
    """
    Returns picked localizations in the specified channel.

    Parameters
    ----------
    channel : int
        Channel of locs to be processed
    add_group : boolean (default=True)
        True if group id should be added to locs. Each pick will be
        assigned a different id
    return_nonpicked : bool
        whether to return the non-picked locs

    Returns:
        all_picked_locs : np.recarray
            locs within pick_diameter around _centers, linked to
            common centers by field 'group'
        # all_picked_locs : list of np.recarray
        #     locs within pick_diameter around _centers, linked to
        #     common centers by field 'group'
        non_picked_locs : np.recarray
            locs that have not been picked.
    """

    picked_locs = []
    is_not_picked = []
    d = pick_diameter
    r = d / 2
    index_blocks = index_locs(locs, info, d)
    # print('index blocks: ', index_blocks)
    for i, pick in enumerate(_centers):
        x, y = pick
        block_locs, block_indices = get_block_locs_at(
            x, y, index_blocks, return_indices=True
        )
        # print(f'block locs: {block_locs}')

        group_locs, is_picked = locs_at(
            x, y, block_locs, r, return_indices=True
        )
        # logger.debug(block_indices)
        # logger.debug(is_picked)
        # logger.debug(is_picked.shape)
        is_not_picked.append(block_indices[~is_picked])
        # print(f'grouplocs: {group_locs}')
        if add_group:
            group = i * np.ones(len(group_locs), dtype=np.int32)
            group_locs["group"] = group
        group_locs.sort_values("frame", kind="mergesort", inplace=True)
        picked_locs.append(group_locs)

    all_picked_locs = pd.concat(picked_locs, ignore_index=True)
    # all_picked_locs = picked_locs

    if return_nonpicked:
        # Create a merge to identify picked locs
        locs_with_key = locs.copy()
        locs_with_key['_key'] = 1
        picked_with_key = all_picked_locs[["frame", "x", "y", "photons"]].copy()
        picked_with_key['_picked'] = 1

        merged = locs_with_key.merge(
            picked_with_key,
            on=["frame", "x", "y", "photons"],
            how='left'
        )
        mask = merged['_picked'].notna().values
        non_picked_locs = locs[~mask]
        return all_picked_locs, non_picked_locs
    else:
        return all_picked_locs


def _undrift_from_picked_coordinate(info, picked_locs, coordinate):
    """
    Calculates drift in a given coordinate.

    Parameters
    ----------
    channel : int
        Channel where locs are being undrifted
    picked_locs : list
        List of pd.DataFrames with locs for each pick
    coordinate : str
        Spatial coordinate where drift is to be found

    Returns
    -------
    np.array
        Contains average drift across picks for all frames
    """

    n_picks = len(picked_locs)
    n_frames = info[0]["Frames"]

    # Drift per pick per frame
    drift = np.empty((n_picks, n_frames))
    drift.fill(np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = locs[coordinate]
        drift[i, locs['frame']] = coordinates - np.mean(coordinates)

    # Mean drift over picks
    drift_mean = np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = np.isnan(drift)
    drift = np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = np.ma.average(drift, axis=0, weights=1 / msd)
    drift_mean = drift_mean.filled(np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = np.interp(
        nonzero(nans), nonzero(~nans), drift_mean[~nans]
    )

    return drift_mean


def _undrift_from_picked(locs, info, picked_locs):
    """
    Undrifts in x and y based on picked locs in a given channel.
    Parameters
    ----------
    channel : int
        Channel to be undrifted
    """
    drift_x = _undrift_from_picked_coordinate(info, picked_locs, "x")
    drift_y = _undrift_from_picked_coordinate(info, picked_locs, "y")
    locs['x'] -= drift_x[locs['frame']]
    locs['y'] -= drift_y[locs['frame']]

    drift = {'x': drift_x, 'y': drift_y}
    if "z" in locs.columns:
        drift_z = _undrift_from_picked_coordinate(info, picked_locs, "z")
        locs['z'] -= drift_z[locs['frame']]
        drift['z'] = drift_z
    # drift = np.array(drift).T
    return locs, info, drift


def shift_from_picked(channel_fiducials):
    """
    Calculate shift based on picked fiducials

    Args:
        channel_fiducials : list of np.recarray
            the picked localizations to evaluate shifts from.
            Must contain a 'x', 'y', 'group' columns

    Returns
    -------
    tuple
        With shifts; shape (2,) or (3,) (if z coordinate present)
    """
    dy = shifts_from_picked_coordinate(channel_fiducials, "y")
    dx = shifts_from_picked_coordinate(channel_fiducials, "x")
    try:
        dz = shifts_from_picked_coordinate(channel_fiducials, "z")
    except (IndexError, KeyError, AttributeError):
        dz = None
    # if all([hasattr(_[0], "z") for _ in channel_fiducials]):
    #     dz = shifts_from_picked_coordinate(channel_fiducials, "z")
    # else:
    #     dz = None
    return lib.minimize_shifts(dx, dy, shifts_z=dz)


def sort_picked_locs(channel_picks, max_shift=None):
    """Sorts picked localizations to match between channels.
    Args:
        max_shift : None or float
            the maximum shift between channel picks. If given, picks are only
            considered if they have corresponding picks in all other channels,
            and resorted accordingly.
    Returns:
        channel_locs : list of np.rec.array
            the accepted picks in corresponding order
    """
    n_channels = len(channel_picks)
    # ngroups = [len(np.unique(picks['group'])) for picks in channel_picks]
    # logger.debug(f"#groups in: {str(ngroups)}")
    max_picks = max(
        [len(np.unique(picks["group"])) for picks in channel_picks]
    )
    pick_means = np.nan * np.ones((n_channels, max_picks, 2), dtype=np.float64)
    for chan in range(n_channels):
        picks = channel_picks[chan]
        for i, pick_group in enumerate(np.unique(picks["group"])):
            pick_locs = picks[picks["group"] == pick_group]
            pick_means[chan, i, 0] = np.mean(pick_locs.x)
            pick_means[chan, i, 1] = np.mean(pick_locs.y)

    # logger.debug(f"pick means: {str(pick_means)}")

    def mean_distances(means, ch, picki, ref):
        """returns the distances of all picks in channel ch
        from pick picki in channel ref
        """
        dist = np.sqrt(
            (means[ch, picki, 0] - means[ref, :, 0]) ** 2
            + (means[ch, picki, 1] - means[ref, :, 1]) ** 2
        )
        return dist

    # offset the channels to avoid collisions
    for i in range(n_channels):
        channel_picks[i]["group"] += max_picks

    # find pick groups corresponding to first channel picks
    pick_group = -1 * np.ones((n_channels, max_picks), dtype=np.int16)
    pick_drop = {ch: [] for ch in range(n_channels)}
    for chan in range(n_channels):
        chan_groups = np.unique(channel_picks[chan]["group"])
        logger.debug(f"channel {chan} groups: {str(chan_groups)}")
        if chan == 0:
            pick_group[chan, : len(chan_groups)] = chan_groups
            continue

        for i, group in enumerate(chan_groups):
            dists = mean_distances(pick_means, chan, i, 0)
            dists = dists[: len(chan_groups)]
            try:
                mindist_i = np.nanargmin(dists).flatten()
                mindist_i = mindist_i[0]
            except (IndexError, ValueError):
                # discard the pick
                # pick_drop[chan].append(group)
                # logger.debug(
                #     f"dropping: chan {chan}, group {group}, dists: {dists}")
                continue

            mindist = dists[mindist_i]
            if (max_shift is not None) and (mindist > max_shift):
                # discard the pick
                # logger.debug(
                #     f"""dropping: chan {chan}, group {group},
                #     mindist: {mindist}, i: {mindist_i}""")
                pick_drop[chan].append(group)
                continue

            # set the index as the corresponding pick index
            # logger.debug(
            #     f"""keeping: chan {chan}, group {group},
            #     mindist: {mindist}, i: {mindist_i}""")
            pick_group[chan, mindist_i] = group

    # logger.debug(f"pick groups: {str(pick_group)}")
    # logger.debug(f"dropping groups: {pick_drop}")

    # now check back: all cols where at least one entry is -1 are incomplete,
    # corresponding picks need to be dropped.
    for i in range(max_picks):
        if np.min(pick_group[:, i]) < 0:
            for chan in range(n_channels):
                if pick_group[chan, i] > 0:
                    pick_drop[chan].append(pick_group[chan, i])
                    pick_group[chan, i] = -1

    # logger.debug(f"pick groups doublechecked: {str(pick_group)}")
    # logger.debug(f"dropping groups: {pick_drop}")

    # re-sort the groups, and drop
    for chan in range(n_channels):
        for dropgroup in pick_drop[chan]:
            channel_picks[chan] = channel_picks[chan][
                channel_picks[chan]["group"] != dropgroup
            ]
        for i in range(max_picks):
            corresponding_group = pick_group[chan, i]
            dest_group = i
            if corresponding_group >= 0:
                # set the group to the index
                selected_locs = (
                    channel_picks[chan]["group"] == corresponding_group
                )
                channel_picks[chan]["group"][selected_locs] = dest_group

    # ngroups = [len(np.unique(picks['group'])) for picks in channel_picks]
    # logger.debug(f"#groups out: {str(ngroups)}")

    return channel_picks


def shifts_from_picked_coordinate(locs, coordinate):
    """
    Calculates shifts between channels along a given coordinate.

    Parameters
    ----------
    locs : list of np.recarray
        Picked locs from all channels
    coordinate : str
        Specifies which coordinate should be used (x, y, z)

    Returns
    -------
    np.array
        Array of shape (n_channels, n_channels) with shifts between
        all channels
    """

    n_channels = len(locs)
    # Calculating center of mass for each channel and pick
    coms = []
    for channel_locs in locs:
        coms.append([])
        n_pick_groups = np.unique(channel_locs["group"])
        for pick_group_idx in n_pick_groups:
            group_locs = channel_locs[channel_locs["group"] == pick_group_idx]
            group_com = np.mean(getattr(group_locs, coordinate))
            coms[-1].append(group_com)
    # for i, c in enumerate(coms):
    #     logger.debug(f"coms {i}: {str(c)}")

    # Calculating image shifts
    d = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        for j in range(i + 1, n_channels):
            d[i, j] = np.nanmean([cj - ci for ci, cj in zip(coms[i], coms[j])])
    return d


########################################################################
# End Labeling Efficiency Workflow Modules
########################################################################


def pick_similar(
    locs,
    info,
    diameter=2,
    min_n_locs_per_frame=0.01,
    max_n_locs_per_frame=0.1,
    min_rmsd=0.1,
    max_rmsd=0.3,
):
    """
    Searches picks similar to given nlocs/rmsd parameters.

    Focuses on the number of locs and their root mean square
    displacement from center of mass.
    Instead of picking "similar" to a few manual picks, the rectangle
    in nlocs/rmsd space is given directly here

    Args:
        diameter : float
            the pick similar diameter
        min_n_locs_per_frame, max_n_locs_per_frame : float or str
            the boundaries for min/max nlocs per frame per pick
            if str: "q0.25" - 0.25-quantile
        min_rmsd, max_rmsd : float
            the boundaries for min/max rmsd per pick
    Returns:
        similar : list of [x, y] position pairs
            the positions (picks) of gold beads

    Raises
    ------
    NotImplementedError
        If pick shape is rectangle
    """
    maxframe = info[0]["Frames"]
    # min_n_locs = int(maxframe * min_n_locs_per_frame)
    # get rmsd and nlocs
    x_similar, y_similar, rmsds, nlocs = get_pick_similar_vals(
        locs, info, diameter
    )

    if isinstance(min_n_locs_per_frame, str):
        if min_n_locs_per_frame[0] == "q":
            min_n_locs = np.quantile(nlocs, float(min_n_locs_per_frame[1:]))
        else:
            raise AttributeError(
                "min_n_locs_per_frame must start with q if string"
            )
    else:
        min_n_locs = maxframe * min_n_locs_per_frame
    if isinstance(max_n_locs_per_frame, str):
        if max_n_locs_per_frame[0] == "q":
            max_n_locs = np.quantile(nlocs, float(max_n_locs_per_frame[1:]))
        else:
            raise AttributeError(
                "max_n_locs_per_frame must start with q if string"
            )
    else:
        max_n_locs = maxframe * max_n_locs_per_frame

    labels = -1 * np.ones_like(x_similar)
    pick_idcs = (
        (nlocs >= min_n_locs)
        & (nlocs < max_n_locs)
        & (rmsds >= min_rmsd)
        & (rmsds < max_rmsd)
    )
    labels[pick_idcs] = 0
    x_picked = x_similar[pick_idcs]
    y_picked = y_similar[pick_idcs]
    picks = list(zip(x_picked, y_picked))
    return picks, nlocs, rmsds, labels


########################################################################
# Start pick similar analysis
# this is basically picasso.postprocess.pick_similar, but instead
# of filtering for rmsd, it returns rmsd and n_locs
########################################################################


def cluster_picksim(rmsds, nlocs, nframes, xi=0.05, min_cluster_size=0.05):
    # find different clusters
    X = np.stack([rmsds, nlocs / nframes]).T
    clustering = OPTICS(
        min_samples=int(nframes * min_cluster_size / 5),
        xi=xi,
        min_cluster_size=min_cluster_size,
    )
    clustering.fit(X)

    return clustering.labels_


def picksim_kwargs_for_clusters(rmsds, nlocs, labels, std_range=1):
    # find median nlocs and rmsd of the cluster groups, and
    # set min and max parameters for picking similar
    cluster_grouplabels = np.unique(labels)
    cluster_picksim_kwargs = []
    for grouplabel in cluster_grouplabels:
        if grouplabel == -1:
            continue
        group_idcs = labels == grouplabel
        median_nlocs = np.median(nlocs[group_idcs])
        median_rmsd = np.median(rmsds[group_idcs])
        std_nlocs = np.std(nlocs[group_idcs])
        std_rmsd = np.std(rmsds[group_idcs])
        cluster_picksim_kwargs.append(
            {
                "min_n_locs": median_nlocs - std_range * std_nlocs,
                "max_n_locs": median_nlocs + std_range * std_nlocs,
                "min_rmsd": median_rmsd - std_range * std_rmsd,
                "max_rmsd": median_rmsd + std_range * std_rmsd,
            }
        )
    return cluster_picksim_kwargs


def find_structures(
    locs,
    info,
    diameter,
    min_n_locs_per_frame=0.01,
    xi=0.05,
    min_cluster_size=0.05,
):
    """ """
    nframes = info[0]["Frames"]
    min_n_locs = int(nframes * min_n_locs_per_frame)
    # get rmsd and nlocs
    x_similar, y_similar, rmsds, nlocs = get_pick_similar_vals(
        locs, info, diameter, min_n_locs
    )
    # cluster based on rmsd/nlocs
    labels = cluster_picksim(rmsds, nlocs, nframes, xi, min_cluster_size)
    # the clusters found might be too widespread. therefore,
    # pick by their median +/- std
    cluster_picksim_kwargs = picksim_kwargs_for_clusters(rmsds, nlocs, labels)

    newlabels = -1 * np.ones_like(labels)
    cluster_picks = []
    for clustergroup, pick_kwargs in enumerate(cluster_picksim_kwargs):
        pick_idcs = (
            (nlocs >= pick_kwargs["min_n_locs"])
            & (nlocs < pick_kwargs["max_n_locs"])
            & (rmsds >= pick_kwargs["min_rmsd"])
            & (rmsds < pick_kwargs["max_rmsd"])
        )
        newlabels[pick_idcs] = clustergroup
        x_picked = x_similar[pick_idcs]
        y_picked = y_similar[pick_idcs]
        cluster_picks.append(list(zip(x_picked, y_picked)))
    return cluster_picks, nlocs, rmsds, labels, newlabels


def get_pick_similar_vals(locs, info, diameter, min_n_locs=1):
    """
    Usage:
    x_similar, y_similar, rmsds, nlocs = get_pick_similar_vals(
        locs, info, diameter=1.5, min_n_locs=100)
    plt.hexbin(rmsds, nlocs)
    plt.show()
    """
    kwargs = prep_pick_similar_kwargs(locs, info, diameter)
    kwargs["rmsds"] = np.array([])
    kwargs["nlocs"] = np.array([])
    kwargs["min_n_locs"] = min_n_locs
    x_similar, y_similar, rmsds, nlocs = pick_similar_analysis(**kwargs)
    return x_similar, y_similar, rmsds, nlocs


@nb.jit(nopython=True, nogil=True, cache=True)
def pick_similar_analysis(
    x,
    y_shift,
    y_base,
    x_r,
    y_r1,
    y_r2,
    locs_xy,
    block_starts,
    block_ends,
    K,
    L,
    x_similar,
    y_similar,
    rmsds,
    nlocs,
    r,
    d2,
    min_n_locs=1,
):
    for i, x_grid in enumerate(x):
        x_range = x_r[i]
        # y_grid is shifted for odd columns
        if i % 2:
            y = y_shift
            y_r = y_r1
        else:
            y = y_base
            y_r = y_r2
        for j, y_grid in enumerate(y):
            y_range = y_r[j]
            n_block_locs = postprocess.n_block_locs_at(
                x_range, y_range, K, L, block_starts, block_ends
            )
            if n_block_locs >= min_n_locs:
                block_locs_xy = _get_block_locs_at(
                    x_range,
                    y_range,
                    locs_xy,
                    block_starts,
                    block_ends,
                    K,
                    L,
                )
                # picked_locs_xy = postprocess.locs_at(
                #     x_grid, y_grid, block_locs_xy, r
                # )
                picked_locs_xy = _locs_at(
                    x_grid, y_grid, block_locs_xy, r
                )
                if picked_locs_xy.shape[1] > 1:
                    # Move to COM peak
                    x_test_old = x_grid
                    y_test_old = y_grid
                    x_test = np.mean(picked_locs_xy[0])
                    y_test = np.mean(picked_locs_xy[1])
                    count = 0
                    while (
                        np.abs(x_test - x_test_old) > 1e-3
                        or np.abs(y_test - y_test_old) > 1e-3
                    ):
                        count += 1
                        # skip the locs if the loop is too long
                        if count > 500:
                            break
                        x_test_old = x_test
                        y_test_old = y_test
                        # picked_locs_xy = postprocess.locs_at(
                        #     x_test, y_test, block_locs_xy, r
                        # )
                        picked_locs_xy = _locs_at(
                            x_test, y_test, block_locs_xy, r
                        )
                        if picked_locs_xy.shape[1] > 1:
                            x_test = np.mean(picked_locs_xy[0])
                            y_test = np.mean(picked_locs_xy[1])
                        else:
                            break
                    if np.all(
                        (x_similar - x_test) ** 2 + (y_similar - y_test) ** 2
                        > d2
                    ):
                        # now, instead of filtering, record and return rmsd
                        # and n_locs values
                        if min_n_locs <= picked_locs_xy.shape[1]:
                            x_similar = np.append(x_similar, x_test)
                            y_similar = np.append(y_similar, y_test)
                            rmsds = np.append(
                                rmsds, _rmsd_at_com(picked_locs_xy)
                            )
                            nlocs = np.append(nlocs, picked_locs_xy.shape[1])
    return x_similar, y_similar, rmsds, nlocs


@nb.jit(nopython=True, nogil=True, cache=True)
def _get_block_locs_at(
        x_range, y_range, locs_xy, 
        block_starts, block_ends, K, L,
    ):
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if block_ends[k, l] - block_starts[k, l] > 0:
                        # numba does not work if you attach arange to an empty list so the first step is different
                        # this is because of dtype issues
                        if step == 0:
                            indices = np.arange(float(block_starts[k, l]), float(block_ends[k, l]), 
                                dtype=np.uint32)
                            step = 1
                        else:
                            indices = np.concatenate((indices, 
                                np.arange(float(block_starts[k, l]), float(block_ends[k, l]), dtype=np.uint32)
                            ))
    return locs_xy[:, indices]


@nb.jit(nopython=True, nogil=True, cache=True)
def _locs_at(x, y, locs_xy, r):
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r ** 2
    is_picked = dx ** 2 + dy ** 2 < r2
    return locs_xy[:, is_picked]


@nb.jit(nopython=True, nogil=True)
def _rmsd_at_com(locs_xy):
    com_x = np.mean(locs_xy[0])
    com_y = np.mean(locs_xy[1])
    return np.sqrt(np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2))



#####


def plot_1dhist(locs, field, fig, ax):
    data = locs[field]
    data = data[np.isfinite(data)]
    bins = lib.calculate_optimal_bins(data, 1000)
    # Prepare the figure
    fig.suptitle(field)
    ax.hist(data, bins, rwidth=1, linewidth=0)
    data_range = np.ptp(data)
    ax.set_xlim([bins[0] - 0.05 * data_range, data.max() + 0.05 * data_range])


def plot_2dhist(locs, field_x, field_y, fig, ax):
    x = locs[field_x]
    y = locs[field_y]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    # Start hist2 version
    bins_x = lib.calculate_optimal_bins(x, 1000)
    bins_y = lib.calculate_optimal_bins(y, 1000)
    counts, x_edges, y_edges, image = ax.hist2d(
        x, y, bins=[bins_x, bins_y], norm=LogNorm()
    )
    x_range = np.ptp(x)
    ax.set_xlim([bins_x[0] - 0.05 * x_range, x.max() + 0.05 * x_range])
    y_range = np.ptp(y)
    ax.set_ylim([bins_y[0] - 0.05 * y_range, y.max() + 0.05 * y_range])
    fig.colorbar(image, ax=ax)
    ax.grid(False)
    ax.get_xaxis().set_label_text(field_x)
    ax.get_yaxis().set_label_text(field_y)


########################################################################
# resolution by point-pattern auto correlation
########################################################################


def resolution_ppac(
    locs,
    pixelsize,
    delta_r,
    r_max,
    batch_size=None,
    n_processes=None,
    use_chunking=False,
    use_sparse=False,
):
    """Calculate the resolution by 2D point pattern autocorrelation

    Args:
        locs: DataFrame with 'x' and 'y' columns of localizations
        pixelsize: Pixel size in physical units (e.g., nm)
        delta_r: Grid spacing for autocorrelation calculation
        r_max: Maximum radius for autocorrelation
        batch_size: Deprecated parameter (no longer used)
        n_processes: Number of parallel threads (auto-detected if None)
        use_chunking: Deprecated - chunking has been removed as it was
            mathematically incorrect. This parameter is ignored.
        use_sparse: Whether to use sparse matrices for very large grids
            (default: False)

    Returns:
        2D autocorrelation intensity map normalized by central value

    Notes:
        - Now uses ThreadPoolExecutor for true parallelization
        - Exploits autocorrelation symmetry for 2× speedup
        - Removed incorrect chunking implementation
    """
    from multiprocessing import cpu_count
    from concurrent.futures import ThreadPoolExecutor
    import gc

    # Warn if deprecated parameters are used
    if use_chunking:
        import warnings
        warnings.warn(
            "use_chunking parameter is deprecated and ignored. "
            "The chunking implementation was mathematically incorrect and has been removed.",
            DeprecationWarning,
            stacklevel=2
        )

    r_max = (r_max // delta_r) * delta_r
    r_search = delta_r / 2
    rs = np.arange(
        -r_max, r_max + delta_r / 2, step=delta_r
    )  # Include endpoint
    idx_ctr = len(rs) // 2

    # Choose data structure based on grid size and use_sparse flag
    grid_size = len(rs)
    if use_sparse and grid_size > 50:
        from scipy import sparse

        intensities = sparse.lil_matrix((grid_size, grid_size))
        using_sparse = True
    else:
        intensities = np.zeros((grid_size, grid_size))
        using_sparse = False

    # Convert to physical coordinates once
    xy = np.column_stack([locs["x"] * pixelsize, locs["y"] * pixelsize])
    n_points = len(xy)

    # Auto-detect number of threads
    if n_processes is None:
        n_processes = min(4, cpu_count())  # Cap at 4 to avoid memory pressure

    if n_processes > 1 and len(rs) > 4:
        # Use thread-based parallelization with symmetry optimization
        result_intensities = _resolution_ppac_parallel_optimized(
            xy, rs, r_search, n_processes
        )
        if using_sparse:
            intensities = sparse.lil_matrix(result_intensities)
        else:
            intensities = result_intensities
    else:
        # Use optimized sequential algorithm with symmetry
        intensities = _resolution_ppac_sequential_optimized(
            xy, rs, r_search
        )

    # Convert back to dense array if using sparse
    if using_sparse:
        intensities = intensities.toarray()

    # Normalize by maximum (no shift) and handle division by zero
    max_intensity = intensities[idx_ctr, idx_ctr]
    if max_intensity > 0:
        intensities = intensities / max_intensity
    # set the center to the mean of its 4-connected neighbors
    intensities[idx_ctr, idx_ctr] = np.mean([
        intensities[idx_ctr - 1, idx_ctr],
        intensities[idx_ctr, idx_ctr - 1],
        intensities[idx_ctr + 1, idx_ctr],
        intensities[idx_ctr, idx_ctr + 1]
        ])

    return intensities


def _resolution_ppac_sequential_optimized(xy, rs, r_search):
    """Optimized sequential PPAC with symmetry exploitation

    Args:
        xy: Nx2 array of coordinates
        rs: Array of shift values
        r_search: Search radius for neighbor counting

    Returns:
        2D autocorrelation intensity map
    """
    import gc

    grid_size = len(rs)
    intensities = np.zeros((grid_size, grid_size))

    # Build base tree once
    tree_base = KDTree(xy)

    # Compute only half the grid using symmetry
    for i, delta_x in enumerate(rs):
        for j, delta_y in enumerate(rs):
            # Use symmetry: I(δx, δy) = I(-δx, -δy)
            # Only compute upper half + diagonal
            flat_idx = i * grid_size + j
            if flat_idx > grid_size * grid_size // 2:
                # Use symmetry from already computed point
                sym_i = grid_size - 1 - i
                sym_j = grid_size - 1 - j
                intensities[i, j] = intensities[sym_i, sym_j]
            else:
                # Compute this point
                xy_shift = xy + np.array([[delta_x, delta_y]])
                tree_probe = KDTree(xy_shift)
                intensities[i, j] = tree_base.count_neighbors(
                    tree_probe, r_search
                )
                del tree_probe

    del tree_base
    gc.collect()

    return intensities


def _resolution_ppac_parallel_optimized(xy, rs, r_search, n_threads):
    """Optimized parallel PPAC using ThreadPoolExecutor with symmetry

    Args:
        xy: Nx2 array of coordinates
        rs: Array of shift values
        r_search: Search radius for neighbor counting
        n_threads: Number of threads to use

    Returns:
        2D autocorrelation intensity map
    """
    from concurrent.futures import ThreadPoolExecutor
    import functools

    grid_size = len(rs)
    intensities = np.zeros((grid_size, grid_size))

    # Build base tree once (shared across threads)
    tree_base = KDTree(xy)

    # Generate only half the grid points (exploit symmetry)
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            flat_idx = i * grid_size + j
            if flat_idx <= grid_size * grid_size // 2:
                grid_points.append((i, j))

    # Worker function
    def compute_point(ij_tuple):
        """Compute autocorrelation at one grid point"""
        i, j = ij_tuple
        delta_x, delta_y = rs[i], rs[j]
        xy_shift = xy + np.array([[delta_x, delta_y]])
        tree_probe = KDTree(xy_shift)
        intensity = tree_base.count_neighbors(tree_probe, r_search)
        return (i, j, intensity)

    # Execute in parallel using threads (shares memory, avoids serialization)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(compute_point, grid_points)

        # Fill in computed values and symmetric counterparts
        for i, j, intensity in results:
            intensities[i, j] = intensity

            # Fill symmetric point
            sym_i = grid_size - 1 - i
            sym_j = grid_size - 1 - j
            if sym_i != i or sym_j != j:  # Avoid overwriting center point
                intensities[sym_i, sym_j] = intensity

    return intensities


def _resolution_ppac_chunked(xy, rs, r_search, batch_size, n_processes):
    """DEPRECATED: Mathematically incorrect chunked autocorrelation

    This function is kept for backward compatibility but should not be used.
    The chunking approach used here produces incorrect results because it
    computes autocorrelation within data chunks rather than across the full
    dataset, then incorrectly averages them.

    Use _resolution_ppac_parallel_optimized() instead.
    """
    import gc

    n_points = len(xy)
    intensities = np.zeros((len(rs), len(rs)))

    # Process data in chunks to manage memory usage
    n_chunks = (n_points + batch_size - 1) // batch_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * batch_size
        end_idx = min((chunk_idx + 1) * batch_size, n_points)
        xy_chunk = xy[start_idx:end_idx]

        # Build KDTree for this chunk
        tree_chunk = KDTree(xy_chunk)

        # Process grid points in parallel for this chunk
        if n_processes > 1:
            chunk_intensities = _process_grid_parallel(
                xy_chunk, tree_chunk, rs, r_search, n_processes
            )
        else:
            chunk_intensities = _process_grid_sequential(
                xy_chunk, tree_chunk, rs, r_search
            )

        # Accumulate results (weighted by chunk size)
        weight = len(xy_chunk) / n_points
        intensities += chunk_intensities * weight

        # Cleanup
        del tree_chunk, xy_chunk, chunk_intensities
        gc.collect()

    return intensities


def _resolution_ppac_parallel(xy, rs, r_search, n_processes):
    """DEPRECATED: Inefficient multiprocessing-based parallel autocorrelation

    This function is kept for backward compatibility but should not be used.
    It uses multiprocessing.Pool which causes massive data serialization overhead
    and rebuilds KDTrees redundantly in each worker process.

    Use _resolution_ppac_parallel_optimized() instead, which uses ThreadPoolExecutor
    for true memory sharing and 3-7× better performance.
    """
    from multiprocessing import Pool

    tree_i = KDTree(xy)

    # Create grid point tasks
    tasks = [
        (i, j, xy, rs[i], rs[j], r_search)
        for i in range(len(rs))
        for j in range(len(rs))
    ]

    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(_compute_autocorr_point, tasks)

    # Reconstruct intensity matrix
    intensities = np.zeros((len(rs), len(rs)))
    for i, j, intensity in results:
        intensities[i, j] = intensity

    del tree_i
    return intensities


def _process_grid_parallel(xy_chunk, tree_chunk, rs, r_search, n_processes):
    """Process grid points in parallel for a data chunk"""
    from multiprocessing import Pool

    tasks = [
        (i, j, xy_chunk, rs[i], rs[j], r_search)
        for i in range(len(rs))
        for j in range(len(rs))
    ]

    with Pool(n_processes) as pool:
        results = pool.map(_compute_autocorr_point_chunk, tasks)

    intensities = np.zeros((len(rs), len(rs)))
    for i, j, intensity in results:
        intensities[i, j] = intensity

    return intensities


def _process_grid_sequential(xy_chunk, tree_chunk, rs, r_search):
    """Process grid points sequentially for a data chunk"""
    intensities = np.zeros((len(rs), len(rs)))

    for i, delta_x in enumerate(rs):
        for j, delta_y in enumerate(rs):
            xy_shift = xy_chunk + np.array([[delta_x, delta_y]])
            tree_probe = KDTree(xy_shift)
            intensities[i, j] = tree_chunk.count_neighbors(
                tree_probe, r_search
            )
            del tree_probe

    return intensities


def _compute_autocorr_point(task):
    """Compute autocorrelation for a single grid point (full dataset)"""
    i, j, xy, delta_x, delta_y, r_search = task
    tree_i = KDTree(xy)
    xy_shift = xy + np.array([[delta_x, delta_y]])
    tree_probe = KDTree(xy_shift)
    intensity = tree_i.count_neighbors(tree_probe, r_search)
    return (i, j, intensity)


def _compute_autocorr_point_chunk(task):
    """Compute autocorrelation for a single grid point (chunk-based)"""
    i, j, xy_chunk, delta_x, delta_y, r_search = task
    tree_chunk = KDTree(xy_chunk)
    xy_shift = xy_chunk + np.array([[delta_x, delta_y]])
    tree_probe = KDTree(xy_shift)
    intensity = tree_chunk.count_neighbors(tree_probe, r_search)
    return (i, j, intensity)


def analyse_resolution_ppac(intensities, delta_r):
    """Fit 2D Gaussian to autocorrelation map to extract resolution

    Args:
        intensities: 2D autocorrelation intensity map from resolution_ppac
        delta_r: Grid spacing used for autocorrelation calculation

    Returns:
        dict: Resolution analysis results containing:
            - sigma_x, sigma_y: Gaussian standard deviations in x,y
                (physical units)
            - resolution: Average resolution (2.35 * mean(sigma_x, sigma_y))
            - fwhm_x, fwhm_y: Full-width half-maximum values
            - amplitude: Fitted Gaussian amplitude
            - background: Fitted background level
            - fit_quality: R-squared goodness of fit
    """
    from scipy.optimize import curve_fit

    def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, background):
        """2D Gaussian function for fitting"""
        x, y = coords
        return (
            amplitude
            * np.exp(
                -(
                    (x - x0) ** 2 / (2 * sigma_x**2)
                    + (y - y0) ** 2 / (2 * sigma_y**2)
                )
            )
            + background
        )

    # Create coordinate grids
    size = intensities.shape[0]
    center = size // 2
    x_grid = np.arange(size) * delta_r - center * delta_r
    y_grid = np.arange(size) * delta_r - center * delta_r
    X, Y = np.meshgrid(x_grid, y_grid)
    coords = (X.flatten(), Y.flatten())
    z_data = intensities.flatten()

    # Initial parameter guess
    amplitude_guess = np.max(intensities) - np.min(intensities)
    background_guess = np.min(intensities)
    sigma_guess = delta_r * 3  # Initial guess for sigma

    initial_guess = [
        amplitude_guess,
        0,
        0,
        sigma_guess,
        sigma_guess,
        background_guess,
    ]

    # Set parameter bounds
    bounds = (
        [
            0,
            -center * delta_r,
            -center * delta_r,
            delta_r / 2,
            delta_r / 2,
            0,
        ],  # Lower bounds
        [
            np.inf,
            center * delta_r,
            center * delta_r,
            center * delta_r,
            center * delta_r,
            np.inf,
        ],  # Upper bounds
    )

    try:
        # Perform the fit
        # Suppress numerical warnings from scipy optimizer
        # (divide by zero warnings are expected for ill-conditioned histograms)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            popt, pcov = curve_fit(
                gaussian_2d,
                coords,
                z_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )

        amplitude, x0, y0, sigma_x, sigma_y, background = popt

        # Calculate fit quality (R-squared)
        fitted_data = gaussian_2d(coords, *popt)
        ss_res = np.sum((z_data - fitted_data) ** 2)
        ss_tot = np.sum((z_data - np.mean(z_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate resolution metrics
        fwhm_x = 2.35 * sigma_x  # FWHM = 2.35 * sigma for Gaussian
        fwhm_y = 2.35 * sigma_y
        resolution = np.mean([fwhm_x, fwhm_y])  # Average resolution

        return {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "resolution": resolution,
            "fwhm_x": fwhm_x,
            "fwhm_y": fwhm_y,
            "amplitude": amplitude,
            "background": background,
            "center_x": x0,
            "center_y": y0,
            "fit_quality": r_squared,
            "fit_success": True,
            "fit_params": popt,
            "fit_covariance": pcov,
        }

    except Exception as e:
        # Return fallback values if fit fails
        return {
            "sigma_x": np.nan,
            "sigma_y": np.nan,
            "resolution": np.nan,
            "fwhm_x": np.nan,
            "fwhm_y": np.nan,
            "amplitude": np.nan,
            "background": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "fit_quality": np.nan,
            "fit_success": False,
            "error": str(e),
            "fit_params": None,
            "fit_covariance": None,
        }

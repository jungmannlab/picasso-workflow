"""Point-set pairwise-shift estimation (extracted from picasso_outpost).

See the package docstring. The numeric core (``_calculate_pairwise_shift`` and
its peak/fit/bin/quality helpers) is `promote` candidate #2; the plot savers are
workflow-side and the core still calls them (strip before upstreaming).
"""

import os

import numpy as np
from loguru import logger


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
    """Calculate the shift between two channels via histogram peak finding.

    Uses temporal filtering of localization pairs where frame information is
    available.

    Parameters
    ----------
    locs_i : np.rec.array or scipy.spatial.cKDTree
        Localizations for the first channel (or a pre-built KDTree).
    locs_j : np.rec.array
        Localizations for the second channel.
    max_shift : float
        Maximum expected shift in pixels.
    plot_histogram : bool, optional
        Whether to save a 2D histogram plot. Default is False.
    plot_dir : str, optional
        Directory to save plots.
    channel_pair : tuple, optional
        ``(i, j)`` channel indices, used in the filename.
    remove_zeroshift : bool, optional
        Skip zero shifts; useful when ``locs_j`` are part of ``locs_i``.
        Default is False.
    plot_fn_suffix : str, optional
        Suffix appended to the plot filename.
    ref_frames, frame_locs_frames : ndarray, optional
        Frame numbers for the reference and frame localizations (for temporal
        filtering).
    ton_exclusion : int, optional
        Exclude pairs from frames within ±2×ton (temporal filtering).
        Default is 0.
    peak_mode : str, optional
        Peak-finding method: ``"gaussian"`` (2D Gaussian fit, falling back to
        center of mass), ``"center_of_mass"`` (center of mass of the top 9
        bins) or ``"auto"`` (same as ``"gaussian"``). Default is ``"auto"``.
    snr_threshold : float, optional
        Signal-to-noise threshold (``max_bin / median_bin``); below it,
        center-of-mass is forced and the result marked as failed. Default
        is 3.0.
    build_frame_contributions : bool, optional
        Whether to build the ``frame_contributions`` dict for matrix-based
        drift correction. Expensive (``O(n*log(bins))``); enable only if the
        matrix solver is used. Default is False.
    enable_fit_quality_check : bool, optional
        Whether to check Gaussian fit quality and fall back to center of mass
        when poor. Default is True.
    fit_quality_thresholds : dict, optional
        Thresholds for the fit-quality check (default
        ``{"chi_squared": 2.0, "r_squared": 0.90}``); both must pass for the
        fit to be accepted.

    Returns
    -------
    shift_x, shift_y : float or None
        Shift from channel i to channel j, or None if the calculation failed.
    plot_filepath : str or None
        Path to the saved histogram plot if ``plot_histogram`` is True, else
        None.
    uncertainty_info : dict or None
        Uncertainty information (Gaussian widths and parameter errors), or
        None on failure.
    """
    # Set default quality thresholds if not provided
    if fit_quality_thresholds is None:
        fit_quality_thresholds = {"chi_squared": 2.0, "r_squared": 0.90}

    # Use KDTree for efficient nearest neighbor search
    from scipy.spatial import cKDTree

    if not isinstance(locs_i, cKDTree):
        tree_i = cKDTree(np.column_stack([locs_i.x, locs_i.y]))
    else:
        tree_i = locs_i

    if tree_i.n == 0 or len(locs_j) == 0:
        return None, None, None, None
    # Calculate all pairwise distances and shifts
    coords_j = np.column_stack([locs_j.x, locs_j.y])

    # Determine if temporal filtering is enabled
    use_temporal_filter = (
        ton_exclusion > 0
        and ref_frames is not None
        and frame_locs_frames is not None
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
            if remove_zeroshift and dx == 0 and dy == 0:
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
        logger.warning("No valid shifts. Returning Nones.")
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
    if (
        build_frame_contributions
        and ref_frames is not None
        and len(ref_indices) > 0
    ):
        frame_contributions = {}
        n_bins = len(x_edges) - 1

        for shift_idx in range(len(valid_shifts_x)):
            shift_x_val = valid_shifts_x[shift_idx]
            shift_y_val = valid_shifts_y[shift_idx]
            ref_idx = ref_indices[shift_idx]
            ref_frame = ref_frames[ref_idx]

            # Find which bin this shift fell into
            bin_x = np.searchsorted(x_edges, shift_x_val, side="right") - 1
            bin_y = np.searchsorted(y_edges, shift_y_val, side="right") - 1

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
    median_bin_value = (
        np.median(non_zero_bins) if len(non_zero_bins) > 0 else 1.0
    )
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
        (
            shift_x,
            shift_y,
            sigma_x,
            sigma_y,
            shift_x_error,
            shift_y_error,
            com_threshold,
            com_use_threshold,
        ) = _find_peak_center_of_mass(
            hist, x_edges, y_edges, max_shift, snr_threshold
        )
        # If user EXPLICITLY requested center_of_mass, ALWAYS mark as successful
        # If FORCED to center_of_mass due to low SNR, mark as failed to trigger max_shift retry
        if snr_too_low and peak_mode != "center_of_mass":
            fit_successful = (
                False  # Was forced by low SNR, trigger max_shift iterations
            )
        else:
            fit_successful = (
                True  # Explicit request OR good SNR - mark as successful
            )

    elif peak_mode_to_use in ["gaussian", "auto"]:
        # Try 2D Gaussian fitting first
        try:
            (
                shift_x,
                shift_y,
                sigma_x,
                sigma_y,
                shift_x_error,
                shift_y_error,
                goodness_of_fit,
            ) = _fit_2d_gaussian_peak(hist, x_edges, y_edges, max_shift)

            # Check fit quality if enabled
            fit_quality_passed = True
            if enable_fit_quality_check and goodness_of_fit is not None:
                chi_sq = goodness_of_fit["chi_squared_reduced"]
                r_sq = goodness_of_fit["r_squared"]
                chi_threshold = fit_quality_thresholds["chi_squared"]
                r_threshold = fit_quality_thresholds["r_squared"]

                # Combined metric: BOTH chi_squared and r_squared must pass
                fit_quality_passed = (chi_sq < chi_threshold) and (
                    r_sq > r_threshold
                )

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
            (
                shift_x,
                shift_y,
                sigma_x,
                sigma_y,
                shift_x_error,
                shift_y_error,
                com_threshold,
                com_use_threshold,
            ) = _find_peak_center_of_mass(
                hist, x_edges, y_edges, max_shift, snr_threshold
            )
            fit_successful = False  # Indicates fallback was used
            peak_mode_to_use = (
                "center_of_mass"  # Update to reflect actual method used
            )
            goodness_of_fit = None  # CoM doesn't have goodness of fit

    else:
        raise ValueError(
            f"Unknown peak_mode: {peak_mode}. Use 'gaussian', 'center_of_mass', or 'auto'."
        )

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
            uncertainty_info=uncertainty_info, hist=hist
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
    """Calculate an adaptive bin size and bin count for the histogram.

    Parameters
    ----------
    valid_shifts_x : list
        X shift values.
    valid_shifts_y : list
        Y shift values.
    max_shift : float
        Maximum shift range.

    Returns
    -------
    bin_size : float
        Calculated bin size in pixels.
    bins : int
        Number of histogram bins.
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


def _find_peak_center_of_mass(
    hist, x_edges, y_edges, max_shift=None, snr_threshold=3.0
):
    """Find a peak via the center of mass of the highest histogram bins.

    More robust than a simple histogram maximum for broad/non-Gaussian peaks,
    and gives sub-bin precision without Gaussian fitting. Uses threshold-based
    bin selection (all bins with ``count >= median × snr_threshold``, minimum
    9 bins), falling back to a 3×3 neighborhood when fewer than 9 bins exceed
    the threshold.

    Parameters
    ----------
    hist : np.ndarray
        2D histogram of shifts.
    x_edges, y_edges : np.ndarray
        Histogram bin edges in x and y.
    max_shift : float, optional
        Maximum shift radius (for consistency with the Gaussian-fit API).
    snr_threshold : float, optional
        Threshold multiplier for the median bin count; bins with
        ``count >= median × snr_threshold`` are included. Default is 3.0.

    Returns
    -------
    shift_x, shift_y : float
        Center-of-mass coordinates.
    sigma_x, sigma_y : float
        Estimated spread (std dev of the neighborhood values).
    shift_x_error, shift_y_error : float
        Estimated errors (based on bin size and peak sharpness).
    threshold : float
        Threshold value used for bin selection (``median × snr_threshold``).
    use_threshold : bool
        Whether threshold-based selection (True) or the 3×3 fallback (False)
        was used.
    """
    # Create bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    bin_size_x = x_edges[1] - x_edges[0] if len(x_edges) > 1 else 1.0
    bin_size_y = y_edges[1] - y_edges[0] if len(y_edges) > 1 else 1.0

    # Find histogram maximum
    hist_transposed = hist.T  # Match meshgrid convention
    peak_idx = np.unravel_index(
        np.argmax(hist_transposed), hist_transposed.shape
    )
    peak_y_bin, peak_x_bin = (
        peak_idx  # Note: unravel_index returns (row, col) = (y, x)
    )

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
        return (
            shift_x,
            shift_y,
            sigma_x,
            sigma_y,
            shift_x_error,
            shift_y_error,
            0.0,
            False,
        )

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
            [peak_x_bin - 1, peak_x_bin, peak_x_bin + 1], 0, hist.shape[0] - 1
        ).astype(int)
        y_indices = np.clip(
            [peak_y_bin - 1, peak_y_bin, peak_y_bin + 1], 0, hist.shape[1] - 1
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
        np.sum(neighborhood_values * (x_coord_grid - shift_x) ** 2)
        / total_mass
    )
    sigma_y = np.sqrt(
        np.sum(neighborhood_values * (y_coord_grid - shift_y) ** 2)
        / total_mass
    )

    # Estimate position error based on bin size and peak sharpness
    # Sharper peaks (high max/mean ratio) have lower uncertainty
    peak_value = np.max(neighborhood_values)
    mean_value = np.mean(neighborhood_values[neighborhood_values > 0])
    sharpness_ratio = peak_value / mean_value if mean_value > 0 else 1.0

    # Error scales inversely with sharpness, bounded by bin size
    shift_x_error = min(
        bin_size_x / (2 * np.sqrt(sharpness_ratio)), bin_size_x / 2
    )
    shift_y_error = min(
        bin_size_y / (2 * np.sqrt(sharpness_ratio)), bin_size_y / 2
    )

    return (
        shift_x,
        shift_y,
        sigma_x,
        sigma_y,
        shift_x_error,
        shift_y_error,
        threshold,
        use_threshold,
    )


def _fit_2d_gaussian_peak(hist, x_edges, y_edges, max_shift=None):
    """Fit a 2D Gaussian to the histogram peak for a precise shift location.

    Values outside the ``max_shift`` circle are set to NaN and excluded from
    the fit.

    Parameters
    ----------
    hist : np.ndarray
        2D histogram of shifts.
    x_edges, y_edges : np.ndarray
        Histogram bin edges in x and y.
    max_shift : float, optional
        Maximum shift radius; values outside this circle are set to NaN.

    Returns
    -------
    shift_x, shift_y : float
        Fitted peak center coordinates.
    sigma_x, sigma_y : float
        Gaussian widths in x and y.
    shift_x_error, shift_y_error : float
        Coordinate uncertainties from the covariance matrix.
    goodness_of_fit : dict
        Fit-quality metrics: ``chi_squared_reduced``, ``r_squared``,
        ``rmse``, ``n_points`` and ``degrees_of_freedom``.
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
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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
        chi_squared = np.sum((residuals**2) / variances)
        chi_squared_reduced = (
            chi_squared / degrees_of_freedom
            if degrees_of_freedom > 0
            else np.inf
        )

        # R-squared (coefficient of determination)
        # R² = 1 - (SS_residual / SS_total)
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((z_fit - np.mean(z_fit)) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0.0

        # Root mean squared error
        rmse = np.sqrt(np.mean(residuals**2))

        goodness_of_fit = {
            "chi_squared_reduced": chi_squared_reduced,
            "r_squared": r_squared,
            "rmse": rmse,
            "n_points": n_points,
            "degrees_of_freedom": degrees_of_freedom,
        }

        return (
            shift_x,
            shift_y,
            sigma_x,
            sigma_y,
            shift_x_error,
            shift_y_error,
            goodness_of_fit,
        )

    except Exception as e:
        raise RuntimeError(f"Gaussian fitting failed: {str(e)}")


def _normalize_quality_metrics(
    uncertainty_info=None, quality_metrics=None, hist=None
):
    """Normalize quality metrics from the standard or numba route.

    Converts ``uncertainty_info`` (standard route) or ``quality_metrics``
    (numba route) into a unified format suitable for the plotting functions.

    Parameters
    ----------
    uncertainty_info : dict or None
        Quality metrics from the standard route
        (:func:`_calculate_pairwise_shift`); expected keys ``sigma_x``,
        ``sigma_y``, ``fit_successful``, ``peak_mode``, ``com_threshold``,
        ``com_use_threshold``.
    quality_metrics : dict or None
        Quality metrics from the numba route; expected keys ``sigma_x``,
        ``sigma_y``, ``peak_mode``, ``success``, ``com_threshold``,
        ``com_use_threshold``, ``total_pairs``.
    hist : np.ndarray or None
        2D histogram, used to compute ``total_pairs`` if not already present.

    Returns
    -------
    dict
        Unified metrics with keys ``peak_mode``, ``sigma_x``, ``sigma_y``,
        ``total_pairs``, ``com_threshold`` and ``com_use_threshold``.
    """
    normalized = {}

    # Determine source and extract values
    if quality_metrics is not None:
        # Numba route - already in good format
        normalized["peak_mode"] = quality_metrics.get("peak_mode", "unknown")
        normalized["sigma_x"] = quality_metrics.get("sigma_x", 0.0)
        normalized["sigma_y"] = quality_metrics.get("sigma_y", 0.0)
        normalized["total_pairs"] = quality_metrics.get(
            "total_pairs", np.sum(hist) if hist is not None else 0
        )
        normalized["com_threshold"] = quality_metrics.get("com_threshold")
        normalized["com_use_threshold"] = quality_metrics.get(
            "com_use_threshold"
        )

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
        normalized["com_use_threshold"] = uncertainty_info.get(
            "com_use_threshold"
        )

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
    """Save a 2D histogram plot of the RSSO shift distribution and peak.

    Works for both the standard (channel alignment) and numba (iterative
    drift) routes. Either saves to disk (traditional mode) or stores the plot
    in shared memory for incremental video writing (memory-efficient mode).

    Parameters
    ----------
    hist : np.ndarray
        2D histogram of shifts.
    x_edges, y_edges : np.ndarray
        Histogram bin edges in x and y.
    shift_x, shift_y : float
        Estimated x and y shift.
    max_shift : float
        Maximum shift range.
    plot_dir : str
        Directory to save the plot (only used if ``shared_plot_dict`` is None).
    quality_metrics : dict
        Metrics to display, with keys ``peak_mode``, ``sigma_x``/``sigma_y``
        (uncertainties), ``total_pairs``, ``com_threshold`` and
        ``com_use_threshold``.
    plot_fn_suffix : str or None, optional
        Suffix appended to the plot filename.
    iteration, frame_number : int or None, optional
        Iteration / frame number for the filename (numba route).
    channel_pair : tuple or None, optional
        ``(i, j)`` channel indices for the filename (standard route).
    output_subdir : str or None, optional
        Subdirectory structure to create (e.g.
        ``"rsso_plots/iteration_00/sglframe/"``). If None, saves directly to
        ``plot_dir``.
    shared_plot_dict : multiprocessing.Manager.dict or None, optional
        If provided, stores the plot as a ``(height, width, 3)`` numpy array
        in this dict (keyed by ``frame_number``) instead of saving to disk.

    Returns
    -------
    str or int
        The saved plot's filepath, or the ``frame_number`` key if
        ``shared_plot_dict`` was provided.
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
                    x_right = (
                        x_edges[i + 1]
                        if i + 1 < len(x_edges)
                        else x_edges[i] + bin_size_x
                    )
                    y_bottom = y_edges[j]
                    y_top = (
                        y_edges[j + 1]
                        if j + 1 < len(y_edges)
                        else y_edges[j] + bin_size_y
                    )

                    # Draw rectangle outline
                    rect = plt.Rectangle(
                        (x_left, y_bottom),
                        x_right - x_left,
                        y_top - y_bottom,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.5,
                        alpha=0.8,
                    )
                    ax.add_patch(rect)

    # Set labels and title
    ax.set_xlabel("X Shift (pixels)")
    ax.set_ylabel("Y Shift (pixels)")

    # Build title based on available information
    title_parts = []
    if channel_pair is not None:
        # Standard route - channel alignment
        title_parts.append(
            f"RSSO: Ch {channel_pair[0]} → Ch {channel_pair[1]}"
        )
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
        img_array = img_array.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

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
    plot_fn_suffix="",
):
    """Save a shift histogram plot (deprecated).

    .. deprecated::
        Use :func:`_save_rsso_shift_histogram_plot` instead. Kept for
        backward compatibility; redirects to the unified plotting function.

    Parameters
    ----------
    hist : np.ndarray
        2D histogram of shifts.
    x_edges, y_edges : np.ndarray
        Histogram bin edges in x and y.
    shift_x, shift_y : float
        Estimated x and y shift.
    max_shift : float
        Maximum shift range.
    plot_dir : str or None
        Directory to save the plot.
    channel_pair : tuple
        ``(i, j)`` channel indices.
    fit_successful : bool, optional
        Whether the Gaussian fit succeeded. Default is False.
    plot_fn_suffix : str, optional
        Suffix for the filename (ignored by the new function).

    Returns
    -------
    str
        Path to the saved plot file.
    """
    import warnings

    warnings.warn(
        "_save_shift_histogram_plot is deprecated. "
        "Use _save_rsso_shift_histogram_plot instead.",
        DeprecationWarning,
        stacklevel=2,
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

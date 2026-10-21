"""Multi-channel alignment orchestration (extracted from picasso_outpost).

See the package docstring. `keep` -- glue over ``shift_kernel`` + upstream
``imageprocess.rcc`` / ``lib.minimize_shifts``.
"""

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from picasso_workflow._picasso_api import imageprocess, lib, render
from picasso_workflow.outpost.alignment.shift_kernel import (
    _calculate_pairwise_shift,
)

# ``shift_from_picked`` / ``sort_picked_locs`` are pick-fiducial helpers that
# still live in ``picasso_outpost`` (they move to ``outpost/picks/`` in split
# step S6; update this import then). Safe module-level import: the monolith does
# not import this package back, so there is no cycle.
from picasso_workflow.picasso_outpost import (
    shift_from_picked,
    sort_picked_locs,
)


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
    """Align multiple channels to each other.

    Adapted from ``picasso.gui.render.View.align``; replicated here because
    the upstream code is not modular enough. This could eventually become a
    non-GUI function in picasso.

    Parameters
    ----------
    channel_locs : list of recarray
        The localizations of the different channels.
    channel_info : list of dict
        The infos of the different channels.
    channel_tags : list of str, optional
        Names of the channels (used by the RSSO method).
    max_iterations : int, optional
        Maximum number of alignment iterations. Default is 5.
    convergence : float, optional
        Convergence criterion (in pixels) below which a shift is negligible
        and alignment is considered converged. Default is 0.001.
    fiducial_locs : list of recarray, optional
        Localizations to use as the alignment basis. If None, the
        ``channel_locs`` are used as fiducials.
    force_method : str, optional
        Force a specific algorithm: ``"RCC"`` (even with fiducials present),
        ``"picked"`` (the by-picked algorithm), ``"RSSO"``, or
        ``"filtered_RCC"`` (filtered RCC based on shift histograms).
    max_shift : float, optional
        Maximum shift between picks (when undrifting fiducials by picked) or
        the maximum expected shift for filtered_RCC / RSSO alignment.
    plot_histogram : bool, optional
        Whether to save 2D histogram plots for filtered_RCC alignment.
        Default is False.
    plot_dir : str, optional
        Directory to save the histogram plots.

    Returns
    -------
    shift : list of list
        Length 2-3 (x, y, [z]); for each dimension, the per-iteration shifts
        averaged over channels.
    cumulative_shift : numpy.ndarray
        Shape ``(3, channels, iterations)``; the cumulative shift per
        dimension and channel (the total is the last iteration value).
    use_fiducials : bool
        Whether fiducials were used (aligned by picked) or not (by RCC).
    algo_used : str
        The algorithm used for alignment.
    fp_figs : list
        Figure file paths, if plotting was enabled.
    shift_uncertainties : dict
        Uncertainty information (only populated for the RSSO method).
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
        shift, cumulative_shift, channel_locs, fiducial_locs = align_by_picked(
            channel_locs,
            fiducial_locs,
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
        shift, cumulative_shift, channel_locs, fiducial_locs = align_by_rcc(
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


def _has_z(locs):
    """Check whether a locs object (np.recarray or pd.DataFrame) has a
    'z' coordinate. Returns False for 2D data missing the z column/field.
    """
    if hasattr(locs, "columns"):  # pandas DataFrame
        return "z" in locs.columns
    if getattr(locs, "dtype", None) is not None and locs.dtype.names:
        return "z" in locs.dtype.names  # np.recarray / structured array
    return hasattr(locs, "z")


def align_by_picked(channel_locs, fiducial_locs):
    """Align channels using picked fiducials.

    Computes the inter-channel shift from the fiducials and applies it to both
    the channel and fiducial localizations.

    Parameters
    ----------
    channel_locs : list of recarray
        The localizations of the different channels.
    fiducial_locs : list of recarray
        The fiducial localizations of the different channels.

    Returns
    -------
    shift : list of list
        The shift in y, x, (z) per channel.
    cumulative_shift : numpy.ndarray
        The shift broadcast to shape ``(dims, channels, 1)``.
    channel_locs : list of recarray
        The shifted channel localizations.
    fiducial_locs : list of recarray
        The shifted fiducial localizations.
    """
    # find shift between channels
    shift = shift_from_picked(fiducial_locs)
    # print("Shift {}".format(shift))

    # align each channel
    has_z = len(shift) == 3
    for i in range(len(channel_locs)):
        channel_locs[i].y -= shift[0][i]
        channel_locs[i].x -= shift[1][i]
        if has_z and _has_z(channel_locs[i]):
            channel_locs[i].z -= shift[2][i]

        fiducial_locs[i].y -= shift[0][i]
        fiducial_locs[i].x -= shift[1][i]
        if has_z and _has_z(fiducial_locs[i]):
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
    """Align channels iteratively using redundant cross-correlation (RCC).

    Parameters
    ----------
    channel_locs : list of recarray
        The localizations of the different channels.
    channel_info : list of dict
        The infos of the different channels.
    max_iterations : int, optional
        Maximum number of alignment iterations. Default is 5.
    convergence : float, optional
        Per-iteration shift (pixels) below which alignment is converged.
        Default is 0.001.
    fiducial_locs : list of recarray, optional
        Fiducials to align on. If None, the ``channel_locs`` are used.

    Returns
    -------
    shift : list of list
        Mean per-iteration shift in x, y, (z).
    cumulative_shift : numpy.ndarray
        Cumulative shift, shape ``(3, channels, iterations)``.
    channel_locs : list of recarray
        The shifted channel localizations.
    fiducial_locs : list of recarray
        The shifted fiducial localizations.
    """
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
    """Plot the shifts generated by :func:`align_channels`.

    Parameters
    ----------
    shifts : list of 1D array
        The shifts in the x, y and potentially z dimensions.
    cum_shifts : numpy.ndarray
        Cumulative shifts, shape ``(dimension, channel, iteration)``.
    filepath : str
        The filepath to save the plot to.
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
    """Estimate inter-channel image shifts via whole-image RCC.

    Used by :func:`align_by_rcc`.

    Parameters
    ----------
    channel_locs : list of recarray
        The localizations of the different channels.
    channel_info : list of dict
        The infos of the different channels.

    Returns
    -------
    tuple
        The channel shifts, of shape ``(2,)`` or ``(3,)`` (if a z coordinate
        is present).
    """
    n_channels = len(channel_locs)
    images = []
    logger.debug("Rendering localizations.")
    # render each channel and save it in images
    for i, (locs_, info_) in enumerate(zip(channel_locs, channel_info)):
        # Pass disp_px_size explicitly (oversampling=1 equivalent) to avoid
        # the deprecated-`oversampling` warning in picasso >= 0.10.
        pixelsize = lib.get_from_metadata(info_, "Pixelsize", raise_error=True)
        _, image = render.render(
            locs_, info_, blur_method="smooth", disp_px_size=pixelsize
        )
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
    """Align channels using redundant spot-shift overrepresentation (RSSO).

    Computes shifts between all channel pairs (redundant measurements) from
    their shift histograms, then solves for the optimal alignment by least
    squares. Assumes localizations in different channels correspond to each
    other but are shifted by ``delta_x``/``delta_y`` with normally distributed
    error.

    Parameters
    ----------
    channel_locs : list of np.rec.array
        Localization arrays for the different channels; each must have ``x``
        and ``y`` fields.
    channel_tags : list of str or None, optional
        The names of the channels.
    max_shift : float, optional
        Maximum expected shift in pixels for alignment. Default is 10.0.
    plot_histogram : bool, optional
        Whether to save 2D histogram plots for each channel pair. Default is
        False.
    plot_dir : str, optional
        Directory to save histogram plots. If None and ``plot_histogram`` is
        True, saves to the current directory.

    Returns
    -------
    shifts : tuple
        The channel shifts as ``(shift_y, shift_x)`` for compatibility with
        existing code.
    fp_figs : list
        File paths to saved histogram plots (empty if ``plot_histogram`` is
        False).
    shift_uncertainties : dict
        Uncertainty information for the channel shifts.
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


def _solve_optimal_shifts(
    pairwise_shifts, n_channels, pairwise_uncertainties=None
):
    """Solve for optimal channel shifts from pairwise measurements.

    Uses least squares under the constraint ``shift_j - shift_i =
    measured_shift_ij`` for all pairs ``(i, j)``. The first channel is set as
    the reference (shift = 0) and the others are solved for.

    Parameters
    ----------
    pairwise_shifts : dict
        Keys ``(i, j)``, values ``(shift_x, shift_y)``.
    n_channels : int
        Number of channels.
    pairwise_uncertainties : dict, optional
        Keys ``(i, j)``, values containing uncertainty information.

    Returns
    -------
    shifts_x, shifts_y : np.ndarray
        Optimal shifts for each channel.
    shift_uncertainties : dict
        Uncertainty information for the channel shifts.
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

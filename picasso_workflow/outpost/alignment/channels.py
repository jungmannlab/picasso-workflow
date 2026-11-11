"""Multi-channel alignment orchestration (extracted from picasso_outpost).

See the package docstring. `keep` -- glue over upstream
``postprocess.align`` / ``imageprocess.rcc``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from picasso_workflow._picasso_api import (
    imageprocess,
    lib,
    postprocess,
    render,
)

from picasso_workflow.outpost.picks.picks import (
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
        the maximum expected shift for filtered_RCC / RSSO alignment. For the
        ``"RSSO"`` method this is the ``max_shift`` of
        ``picasso.postprocess.align(method="points")`` and is in **nm**
        (picasso 1.0); when None, picasso's own default (500 nm) is used. This
        is a unit change -- the retired forked RSSO treated ``max_shift`` as
        ambiguous "px".
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
        shift, fp_figs, shift_uncertainties = _align_by_points(
            channel_locs,
            channel_info,
            max_shift=max_shift,
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
    # picasso 1.0 reports through a single ``progress`` tracker (0.11 took a
    # ``callback``); MockProgress satisfies the lib.Progress interface.
    progress = lib.MockProgress()
    return imageprocess.rcc(images, progress=progress)


def _align_by_points(channel_locs, channel_info, max_shift=None):
    """Align channels by displacements via ``postprocess.align``.

    Adapter around ``picasso.postprocess.align(method="points")`` (picasso
    1.0): it cross-correlates the localizations of every channel pair, fits
    the shift of each pair and solves the per-channel shifts by weighted least
    squares (``lib.minimize_shifts``), with the first channel as the
    reference. The shifts carry a jackknife uncertainty (nm).

    This replaces the retired forked RSSO (``shift_kernel`` cross-correlation
    + ``_solve_optimal_shifts``) while reproducing the output contract that
    :func:`align_channels` and the reporter expect. The channel
    localizations are mutated in place -- the subtracted shift is applied to
    ``channel_locs`` -- matching the retired RSSO branch.

    Parameters
    ----------
    channel_locs : list of recarray
        The localizations of the different channels; each has ``x`` and ``y``
        fields (nm). Shifted in place.
    channel_info : list of list of dict
        The infos of the different channels (picasso's ``list[list[dict]]``
        shape per channel).
    max_shift : float or None, optional
        Largest shift searched for, in **nm** (picasso 1.0). When None,
        picasso's own default (500 nm) is used. Note the unit change: the
        retired forked RSSO treated ``max_shift`` as ambiguous "px" and
        defaulted to 10.

    Returns
    -------
    shift : list of list
        ``[list(shift_x), list(shift_y)]`` -- the shift subtracted from each
        channel (nm); channel 0 is the reference (0).
    fp_figs : list
        Always empty -- the forked per-pair histogram figures are not
        produced.
    shift_uncertainties : dict
        ``mean_x_uncertainty`` / ``mean_y_uncertainty`` /
        ``max_x_uncertainty`` / ``max_y_uncertainty`` (nm), aggregated over
        the non-reference channels from the jackknife standard errors picasso
        records in the appended info.
    """
    # picasso's points-align takes DataFrames and does not mutate them.
    locs_df = [pd.DataFrame(locs) for locs in channel_locs]
    # channel_info[i] is already picasso's list[dict] (io.load_locs returns a
    # list and analyse.py .append()s to it), so pass it as-is; wrap a bare
    # dict defensively.
    infos = [
        info if isinstance(info, list) else [info] for info in channel_info
    ]

    align_kwargs = {"method": "points", "return_shifts": True}
    if max_shift is not None:
        align_kwargs["max_shift"] = max_shift
    try:
        _, infos_out, (shift_x, shift_y) = postprocess.align(
            locs_df, infos, **align_kwargs
        )
    except ValueError as exc:
        # picasso raises when it can't find the displacement peak between
        # enough channel pairs (sparse / non-overlapping channels, or a shift
        # beyond max_shift). The retired RSSO returned a zero-shift no-op
        # instead of aborting the workflow; preserve that graceful fallback.
        logger.warning(
            f"Points alignment could not be computed ({exc}); leaving "
            "channels unshifted."
        )
        n = len(channel_locs)
        return [[0.0] * n, [0.0] * n], [], {}

    # Apply the shifts in place, matching the retired RSSO branch
    # (channel_locs[i].x -= shift). picasso subtracts the same shift.
    for i in range(len(channel_locs)):
        channel_locs[i].x -= shift_x[i]
        channel_locs[i].y -= shift_y[i]

    shift = [list(shift_x), list(shift_y)]

    # Aggregate the jackknife uncertainties (nm) over the non-reference
    # channels (channel 0 is the reference and has uncertainty 0).
    x_unc = np.array(
        [
            info[-1].get("Shift x uncertainty (nm)", np.nan)
            for info in infos_out[1:]
        ]
    )
    y_unc = np.array(
        [
            info[-1].get("Shift y uncertainty (nm)", np.nan)
            for info in infos_out[1:]
        ]
    )
    if x_unc.size == 0:
        mean_x = mean_y = max_x = max_y = np.nan
    else:
        mean_x = np.nanmean(x_unc)
        mean_y = np.nanmean(y_unc)
        max_x = np.nanmax(x_unc)
        max_y = np.nanmax(y_unc)
    shift_uncertainties = {
        "mean_x_uncertainty": mean_x,
        "mean_y_uncertainty": mean_y,
        "max_x_uncertainty": max_x,
        "max_y_uncertainty": max_y,
    }

    return shift, [], shift_uncertainties

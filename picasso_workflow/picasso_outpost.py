#!/usr/bin/env python
"""Exploratory DNA-PAINT analysis functions related to picasso.

A collection of picasso-related functions that, if they prove useful, may be
moved into a future picasso release. Keeping them here makes testing cycles
faster.

Author: Heinrich Grabmayr
Initial date: March 8, 2024
"""

from __future__ import annotations

# import logging
from loguru import logger
import numpy as np

# from numpy.lib.recfunctions import stack_arrays
import pandas as pd
import numba as nb
from matplotlib.colors import LogNorm
import yaml
from dataclasses import dataclass

from picasso_workflow._picasso_api import (
    postprocess,
    lib,
)
from picasso_workflow._picasso_api import get_internal

from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from collections import Counter

# Resolved once at import time (not per-call) because the numba-jitted
# ``pick_similar_analysis`` below calls it: nopython mode cannot call the plain
# ``get_internal`` function, but it can call this module-level dispatcher
# global. Still guarded + contract-tested via INTERNAL_SYMBOLS.
_n_block_locs_at = get_internal("_n_block_locs_at")

# logger = logging.getLogger(__name__)


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
    locs_temp, r, _, _, block_starts, block_ends, K, L = (
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
    """Search for picks similar to gold clusters.

    Focuses on the number of locs and their root-mean-square displacement
    from the center of mass.

    Parameters
    ----------
    locs : np.rec.array
        The localizations.
    info : list of dict
        The localization metadata.
    diameter : float, optional
        The pick-similar diameter. Default is 2.
    std_range : float, optional
        The pick-similar std range identifying gold. Default is 1.4.
    mean_rmsd : float, optional
        The pick-similar mean RMSD identifying gold. Default is 0.4.

    Returns
    -------
    similar : list of [x, y]
        The positions (picks) of gold beads.

    Raises
    ------
    NotImplementedError
        If the pick shape is a rectangle.
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
    x_similar, y_similar = get_internal("_pick_similar")(**kwargs)
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


def picked_locs(
    locs, info, _centers, pick_diameter, add_group=True, return_nonpicked=False
):
    """Return localizations picked around the given centers.

    Parameters
    ----------
    locs : np.rec.array
        The localizations to pick from.
    info : list of dict
        The localization metadata.
    _centers : array-like
        The pick centers.
    pick_diameter : float
        The pick diameter.
    add_group : bool, optional
        Whether to add a ``group`` id to the locs; each pick gets a different
        id. Default is True.
    return_nonpicked : bool, optional
        Whether to also return the non-picked locs. Default is False.

    Returns
    -------
    all_picked_locs : np.recarray
        Locs within ``pick_diameter`` of ``_centers``, linked to common
        centers by the ``group`` field.
    non_picked_locs : np.recarray
        Locs that were not picked (only if ``return_nonpicked`` is True).
    """

    # Pick-finding is delegated to picasso (postprocess.picked_locs, circular
    # picks of radius pick_diameter / 2); this wrapper keeps the workflow's
    # concatenated-DataFrame + group + optional non-picked-locs contract. It is
    # equivalent to the former in-house loop in picked membership and group ids
    # (up to intra-frame row order) -- see tests/test_outpost_dedup_equivalence.
    per_pick = postprocess.picked_locs(
        locs,
        info,
        _centers,
        "Circle",
        pick_size=pick_diameter / 2,
        add_group=add_group,
    )
    per_pick = [
        group_locs.sort_values("frame", kind="mergesort")
        for group_locs in per_pick
    ]
    all_picked_locs = pd.concat(per_pick, ignore_index=True)

    if return_nonpicked:
        # Create a merge to identify picked locs
        locs_with_key = locs.copy()
        locs_with_key["_key"] = 1
        picked_with_key = all_picked_locs[
            ["frame", "x", "y", "photons"]
        ].copy()
        picked_with_key["_picked"] = 1

        merged = locs_with_key.merge(
            picked_with_key, on=["frame", "x", "y", "photons"], how="left"
        )
        mask = merged["_picked"].notna().values
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
        drift[i, locs["frame"]] = coordinates - np.mean(coordinates)

    # Mean drift over picks
    drift_mean = np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = np.nanmean(sd, 1)
    # New mean drift over picks, each weighted by 1/msd. A pick that spans
    # (essentially) a single frame has msd 0 -> an infinite weight that
    # poisons np.ma.average into NaN everywhere; exclude such degenerate
    # picks (weight 0) instead. This is exactly the case that arises when
    # many spurious single-loc picks are fed in.
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / msd
    weights[~np.isfinite(weights)] = 0.0
    nan_mask = np.isnan(drift)
    drift = np.ma.MaskedArray(drift, mask=nan_mask)
    if not np.any(weights > 0):
        raise ValueError(
            "undrift_from_picked: cannot estimate drift - no pick spans "
            "more than one frame (all picks are degenerate). The picks are "
            "too sparse; use more/denser fiducial picks."
        )
    drift_mean = np.ma.average(drift, axis=0, weights=weights)
    drift_mean = drift_mean.filled(np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    if not np.any(~nans):
        raise ValueError(
            "undrift_from_picked: cannot estimate drift - the picks cover "
            "no frames with a usable localization. The picks are empty or "
            "too sparse to undrift from."
        )
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
    locs["x"] -= drift_x[locs["frame"]]
    locs["y"] -= drift_y[locs["frame"]]

    drift = {"x": drift_x, "y": drift_y}
    if "z" in locs.columns:
        drift_z = _undrift_from_picked_coordinate(info, picked_locs, "z")
        locs["z"] -= drift_z[locs["frame"]]
        drift["z"] = drift_z
    # drift = np.array(drift).T
    return locs, info, drift


def shift_from_picked(channel_fiducials):
    """Calculate the inter-channel shift from picked fiducials.

    Parameters
    ----------
    channel_fiducials : list of np.recarray
        The picked localizations to evaluate shifts from; must contain ``x``,
        ``y`` and ``group`` columns.

    Returns
    -------
    tuple
        The shifts, of shape ``(2,)`` or ``(3,)`` (if a z coordinate is
        present).
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
    """Sort picked localizations to match between channels.

    Parameters
    ----------
    channel_picks : list of np.rec.array
        The picked localizations per channel (each with a ``group`` field).
    max_shift : float or None, optional
        Maximum shift between channel picks. If given, picks are kept only if
        they have corresponding picks in all other channels, and are resorted
        accordingly.

    Returns
    -------
    channel_locs : list of np.rec.array
        The accepted picks in corresponding order.
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
            n_ref_groups = len(chan_groups)
            continue

        for i, group in enumerate(chan_groups):
            dists = mean_distances(pick_means, chan, i, 0)
            # Match only against the reference channel's (channel 0) actual
            # picks; the remaining pick_means rows are NaN placeholders.
            # Truncating to the *current* channel's pick count (the previous
            # behaviour) barred matches to reference picks beyond that index,
            # so a channel with fewer picks than channel 0 could never match
            # the later reference picks. With several such channels every
            # column ended up incomplete and all fiducials were dropped.
            dists = dists[:n_ref_groups]
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
    """Search for picks matching given nlocs/rmsd parameters.

    Focuses on the number of locs and their root-mean-square displacement
    from the center of mass. Instead of picking "similar" to a few manual
    picks, the rectangle in nlocs/rmsd space is given directly.

    Parameters
    ----------
    locs : np.rec.array
        The localizations.
    info : list of dict
        The localization metadata.
    diameter : float, optional
        The pick-similar diameter. Default is 2.
    min_n_locs_per_frame, max_n_locs_per_frame : float or str, optional
        Boundaries for min/max nlocs per frame per pick. A string like
        ``"q0.25"`` denotes the 0.25-quantile.
    min_rmsd, max_rmsd : float, optional
        Boundaries for min/max RMSD per pick.

    Returns
    -------
    similar : list of [x, y]
        The positions (picks) matching the parameters.

    Raises
    ------
    NotImplementedError
        If the pick shape is a rectangle.
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
            n_block_locs = _n_block_locs_at(
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
                picked_locs_xy = _locs_at(x_grid, y_grid, block_locs_xy, r)
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
    x_range,
    y_range,
    locs_xy,
    block_starts,
    block_ends,
    K,
    L,
):
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for lx in range(x_range - 1, x_range + 2):
                if 0 < lx < L:
                    if block_ends[k, lx] - block_starts[k, lx] > 0:
                        # numba does not work if you attach arange to an
                        # empty list, so the first step is different; this
                        # is because of dtype issues
                        if step == 0:
                            indices = np.arange(
                                float(block_starts[k, lx]),
                                float(block_ends[k, lx]),
                                dtype=np.uint32,
                            )
                            step = 1
                        else:
                            indices = np.concatenate(
                                (
                                    indices,
                                    np.arange(
                                        float(block_starts[k, lx]),
                                        float(block_ends[k, lx]),
                                        dtype=np.uint32,
                                    ),
                                )
                            )
    return locs_xy[:, indices]


@nb.jit(nopython=True, nogil=True, cache=True)
def _locs_at(x, y, locs_xy, r):
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return locs_xy[:, is_picked]


@nb.jit(nopython=True, nogil=True)
def _rmsd_at_com(locs_xy):
    com_x = np.mean(locs_xy[0])
    com_y = np.mean(locs_xy[1])
    return np.sqrt(
        np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2)
    )


########################################################################
# Start Design-aware origami picking
#
# A design-aware picker: given an origami's designed geometry (a picasso
# design file, a regular grid, or an explicit site list) it finds and
# picks the origami structures, tolerating a configurable number of
# missing docking sites. The geometry/kinetics/registration helpers below
# are pure numpy/scipy/sklearn (no picasso calls) and work internally in
# nanometres; only the orchestrator ``pick_origami`` couples to the
# picasso-backed pick_similar family and works in camera pixels.
########################################################################


@dataclass
class OrigamiTemplate:
    """Expected docking-site geometry of an origami design.

    Attributes
    ----------
    sites_nm : np.ndarray
        ``(N, 2)`` mean-centered expected site coordinates in nanometres.
    n_sites_expected : int
        Number of expected docking sites (``len(sites_nm)``).
    grid_spacing_nm : float
        Median nearest-neighbour spacing of the expected sites, in nm.
    extent_nm : float
        Maximum pairwise site distance (bounding size), in nm.
    source : str
        Human-readable provenance (e.g. the design file path or
        ``"grid"`` / ``"sites"``).
    """

    sites_nm: np.ndarray
    n_sites_expected: int
    grid_spacing_nm: float
    extent_nm: float
    source: str = ""


def _median_nn_spacing(sites):
    """Median nearest-neighbour distance of a point set."""
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    if len(sites) < 2:
        return 0.0
    tree = KDTree(sites)
    dist, _ = tree.query(sites, k=2)
    return float(np.median(dist[:, 1]))


def _point_extent(sites):
    """Maximum pairwise distance of a (small) point set."""
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    if len(sites) < 2:
        return 0.0
    diffs = sites[:, None, :] - sites[None, :, :]
    return float(np.sqrt((diffs**2).sum(axis=-1)).max())


def origami_template_from_sites(sites_nm, source="sites"):
    """Build an :class:`OrigamiTemplate` from explicit site coordinates.

    Parameters
    ----------
    sites_nm : array-like
        ``(N, 2)`` expected docking-site coordinates in nanometres.
    source : str, optional
        Provenance tag. Default ``"sites"``.

    Returns
    -------
    OrigamiTemplate
        Mean-centered template with derived spacing/extent.
    """
    sites = np.asarray(sites_nm, dtype=float).reshape(-1, 2)
    if len(sites) == 0:
        raise ValueError("origami template must have at least one site")
    sites = sites - sites.mean(axis=0)
    return OrigamiTemplate(
        sites_nm=sites,
        n_sites_expected=len(sites),
        grid_spacing_nm=_median_nn_spacing(sites),
        extent_nm=_point_extent(sites),
        source=source,
    )


def origami_template_from_grid(n_rows, n_cols, spacing_nm, angle=0.0):
    """Build an :class:`OrigamiTemplate` from a regular grid.

    Parameters
    ----------
    n_rows, n_cols : int
        Number of rows and columns of docking sites.
    spacing_nm : float
        Inter-site spacing in nanometres.
    angle : float, optional
        Rotation of the grid in degrees. Default 0.

    Returns
    -------
    OrigamiTemplate
        Mean-centered grid template.
    """
    xs, ys = np.meshgrid(
        np.arange(n_cols) * float(spacing_nm),
        np.arange(n_rows) * float(spacing_nm),
    )
    sites = np.column_stack([xs.ravel(), ys.ravel()])
    if angle:
        rot = _rotation_matrix(np.deg2rad(angle))
        sites = sites @ rot.T
    return origami_template_from_sites(sites, source="grid")


def parse_design_sites(path):
    """Parse expected site coordinates from a picasso design ``.yaml``.

    The design file (written by ``picasso.gui.design``) stores the
    docking-site coordinates as comma-separated strings under the
    ``Structure.StructureX`` / ``Structure.StructureY`` keys. The file is
    read as YAML directly (no picasso import) so this stays picasso-free.

    Parameters
    ----------
    path : str
        Path to the picasso design ``.yaml`` file.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` site coordinates in the design's native units.

    Raises
    ------
    ValueError
        If the file has no ``Structure.StructureX``/``Y`` entries.
    """
    with open(path, "r") as f:
        docs = [d for d in yaml.safe_load_all(f) if isinstance(d, dict)]
    doc = None
    for candidate in docs:
        if "Structure.StructureX" in candidate:
            doc = candidate
            break
    if doc is None:
        raise ValueError(
            f"{path}: no 'Structure.StructureX' entry (not a picasso "
            "design file?)"
        )

    def _to_floats(value):
        if isinstance(value, str):
            parts = [p for p in value.replace(";", ",").split(",")]
            return np.array([float(p) for p in parts if p.strip() != ""])
        return np.asarray(value, dtype=float).ravel()

    xs = _to_floats(doc["Structure.StructureX"])
    ys = _to_floats(doc["Structure.StructureY"])
    if len(xs) != len(ys):
        raise ValueError(
            f"{path}: StructureX/StructureY length mismatch "
            f"({len(xs)} vs {len(ys)})"
        )
    return np.column_stack([xs, ys])


def origami_template_from_design_file(path, grid_spacing_nm=None):
    """Build an :class:`OrigamiTemplate` from a picasso design file.

    Parameters
    ----------
    path : str
        Path to the picasso design ``.yaml`` file.
    grid_spacing_nm : float, optional
        If given, the parsed pattern is uniformly rescaled so its median
        nearest-neighbour spacing equals this value. This anchors the
        physical scale independent of the design file's native units (see
        C20 ``grid_spacing_design``). If omitted, the native coordinates
        are used unchanged.

    Returns
    -------
    OrigamiTemplate
        Mean-centered template.
    """
    sites = parse_design_sites(path)
    if grid_spacing_nm is not None:
        native = _median_nn_spacing(sites)
        if native > 0:
            sites = sites * (float(grid_spacing_nm) / native)
    return origami_template_from_sites(sites, source=str(path))


def load_origami_template(spec, grid_spacing_nm=None):
    """Load an :class:`OrigamiTemplate` from a flexible specification.

    Parameters
    ----------
    spec : str or dict or array-like
        One of:

        - ``str`` : path to a picasso design ``.yaml`` file.
        - ``dict`` with ``n_rows``/``n_cols``/``spacing_nm`` (+ optional
          ``angle``) : a regular grid.
        - ``dict`` with ``design_file`` (+ optional ``grid_spacing_nm``) :
          a design file.
        - ``dict`` with ``sites``/``sites_nm`` or a plain ``(N, 2)``
          array-like : an explicit site list.
    grid_spacing_nm : float, optional
        Physical spacing used to anchor a design-file scale (only used for
        the design-file paths). Overridden by ``spec['grid_spacing_nm']``
        if present.

    Returns
    -------
    OrigamiTemplate
    """
    if isinstance(spec, str):
        return origami_template_from_design_file(spec, grid_spacing_nm)
    if isinstance(spec, dict):
        if "design_file" in spec:
            return origami_template_from_design_file(
                spec["design_file"],
                spec.get("grid_spacing_nm", grid_spacing_nm),
            )
        if "n_rows" in spec and "n_cols" in spec:
            return origami_template_from_grid(
                spec["n_rows"],
                spec["n_cols"],
                spec["spacing_nm"],
                spec.get("angle", 0.0),
            )
        if "sites_nm" in spec or "sites" in spec:
            return origami_template_from_sites(
                spec.get("sites_nm", spec.get("sites"))
            )
        raise ValueError(
            "unrecognised origami geometry dict "
            f"{sorted(spec)}: expected a grid "
            "{'n_rows', 'n_cols', 'spacing_nm'[, 'angle']}, a design file "
            "{'design_file'[, 'grid_spacing_nm']}, or explicit sites "
            "{'sites_nm': [[x, y], ...]}"
        )
    if isinstance(spec, (list, tuple, np.ndarray)):
        # assume an ordered array-like of (x, y) coordinate pairs
        return origami_template_from_sites(spec)
    raise TypeError(
        f"origami geometry must be a design-file path (str), a spec dict, "
        f"or an ordered list of [x, y] site coordinates, not "
        f"{type(spec).__name__} ({spec!r}). Note: a Python set literal like "
        "{0, 3, 4, 20} is NOT a valid geometry - use a dict, e.g. "
        "{'n_rows': 3, 'n_cols': 4, 'spacing_nm': 20.0}."
    )


def predict_locs_per_site(k_on, tau_b, concentration, n_frames, exposure):
    """Predict the mean localizations per docking site over an acquisition.

    Uses the standard DNA-PAINT kinetics: the mean dark time is
    ``tau_d = 1 / (k_on * concentration)``, the number of binding events
    over the acquisition is ``T / (tau_b + tau_d)`` with
    ``T = n_frames * exposure``, and each event yields
    ``max(1, tau_b / exposure)`` localizations.

    Parameters
    ----------
    k_on : float
        Association rate constant in ``1 / (M s)``.
    tau_b : float
        Mean bright (bound) time in seconds.
    concentration : float
        Imager concentration in molar (M).
    n_frames : int
        Number of frames in the acquisition.
    exposure : float
        Camera exposure per frame in seconds.

    Returns
    -------
    float
        Predicted mean number of localizations per docking site.
    """
    tau_d = 1.0 / (k_on * concentration)
    total_time = n_frames * exposure
    n_events = total_time / (tau_b + tau_d)
    locs_per_event = max(1.0, tau_b / exposure)
    return float(n_events * locs_per_event)


def kinetics_nlocs_window(
    k_on,
    tau_b,
    concentration,
    n_frames,
    exposure,
    n_sites_expected,
    rel_tol=0.5,
    missing_sites_allowed=0,
):
    """Predict a per-origami localization-count window from kinetics.

    Converts a kinetics prediction into an absolute ``(min, max)`` window
    on the total number of localizations expected in a full origami
    footprint, suitable for seeding :func:`pick_similar` (instead of the
    quantile defaults).

    Parameters
    ----------
    k_on, tau_b, concentration, n_frames, exposure : float
        Imager kinetics / acquisition parameters (see
        :func:`predict_locs_per_site`).
    n_sites_expected : int
        Number of docking sites in a fully-resolved origami.
    rel_tol : float, optional
        Relative half-width of the accepted band. Default 0.5.
    missing_sites_allowed : int, optional
        Sites that may be missing; lowers the minimum-count edge
        accordingly. Default 0.

    Returns
    -------
    (float, float)
        ``(min_n_locs, max_n_locs)`` absolute localization counts.
    """
    per_site = predict_locs_per_site(
        k_on, tau_b, concentration, n_frames, exposure
    )
    min_sites = max(1, n_sites_expected - missing_sites_allowed)
    min_n = per_site * min_sites * (1.0 - rel_tol)
    max_n = per_site * n_sites_expected * (1.0 + rel_tol)
    return float(max(0.0, min_n)), float(max_n)


def simulate_origami_nlocs_rmsd(
    template,
    site_uncertainty_nm,
    missing_sites_allowed,
    mean_locs_per_site,
    n_sim=1500,
    random_seed=0,
):
    """Sample the (nlocs, rmsd) phase space of expected origami structures.

    Draws ``n_sim`` origami realisations from the design geometry, each with
    a random number of missing sites (0..``missing_sites_allowed``, random
    combination), Gaussian site jitter (``site_uncertainty_nm``) and Poisson
    localization counts per site (``mean_locs_per_site``). Returns the total
    localization count and the RMS distance from the centre of mass of each
    realisation - i.e. where real (partial, noisy) origami land in the same
    nlocs/rmsd space that :func:`pick_similar` scores candidates in.

    Parameters
    ----------
    template : OrigamiTemplate
        Expected design geometry (mean-centred sites in nm).
    site_uncertainty_nm : float
        Per-localization Gaussian spread around each docking site (nm).
    missing_sites_allowed : int
        Maximum number of sites that may be missing in a realisation.
    mean_locs_per_site : float
        Poisson mean of localizations per present docking site.
    n_sim : int, optional
        Number of realisations to sample. Default 1500.
    random_seed : int, optional
        Seed for reproducibility. Default 0.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(nlocs, rmsd_nm)`` per realisation, each length ``n_sim``.
    """
    rng = np.random.default_rng(random_seed)
    sites = np.asarray(template.sites_nm, dtype=float).reshape(-1, 2)
    n_sites = len(sites)
    kmax = int(min(max(0, missing_sites_allowed), n_sites - 1))
    nlocs_out = np.zeros(n_sim)
    rmsd_out = np.zeros(n_sim)
    for s in range(n_sim):
        k = int(rng.integers(0, kmax + 1))
        present = (
            sites
            if k == 0
            else sites[rng.choice(n_sites, size=n_sites - k, replace=False)]
        )
        n_per = rng.poisson(mean_locs_per_site, size=len(present))
        total = int(n_per.sum())
        nlocs_out[s] = total
        if total < 2:
            continue
        locs = np.repeat(present, n_per, axis=0) + rng.normal(
            0.0, site_uncertainty_nm, size=(total, 2)
        )
        com = locs.mean(axis=0)
        rmsd_out[s] = float(np.sqrt(np.mean(((locs - com) ** 2).sum(axis=1))))
    return nlocs_out, rmsd_out


def phase_space_window_from_sim(
    nlocs_sim, rmsd_sim_nm, pixelsize, quantile=0.01
):
    """Derive a pick_similar rectangle from a simulated phase-space cloud.

    Takes per-axis quantiles of the simulated ``(nlocs, rmsd)`` cloud as the
    accepted window, dropping the ``quantile`` tail on each side.

    Parameters
    ----------
    nlocs_sim : array-like
        Simulated total localization counts (see
        :func:`simulate_origami_nlocs_rmsd`).
    rmsd_sim_nm : array-like
        Simulated RMSDs in nanometres.
    pixelsize : float
        Camera pixel size in nm (RMSD is returned in camera px to match
        :func:`pick_similar`).
    quantile : float, optional
        Tail fraction dropped on each side per axis. Default 0.01.

    Returns
    -------
    dict
        ``min_nlocs`` / ``max_nlocs`` (total counts) and ``min_rmsd`` /
        ``max_rmsd`` (camera px).
    """
    nlocs_sim = np.asarray(nlocs_sim, dtype=float)
    rmsd_sim_nm = np.asarray(rmsd_sim_nm, dtype=float)
    lo, hi = quantile, 1.0 - quantile
    return {
        "min_nlocs": float(np.quantile(nlocs_sim, lo)),
        "max_nlocs": float(np.quantile(nlocs_sim, hi)),
        "min_rmsd": float(np.quantile(rmsd_sim_nm, lo) / pixelsize),
        "max_rmsd": float(np.quantile(rmsd_sim_nm, hi) / pixelsize),
    }


def origami_phase_space_preview(
    geometry,
    n_frames,
    kinetics=None,
    mean_locs_per_site=None,
    missing_sites_allowed=2,
    site_uncertainty_nm=3.0,
    pixelsize=130.0,
    n_sim=1500,
    sim_quantile=0.01,
    random_seed=0,
    grid_spacing_nm=None,
):
    """Preview the expected origami nlocs/rmsd phase space for a geometry.

    One-call helper intended for interactive/GUI use (cheap - order 0.1 s):
    loads the geometry, turns kinetics into a mean-locs-per-site, simulates
    the expected ``(nlocs, rmsd)`` cloud and derives the suggested pick
    window. The GUI can show the cloud and offer
    ``min_n_locs_per_frame`` / ``max_n_locs_per_frame`` for the user to
    accept or edit.

    Parameters
    ----------
    geometry : str or dict or array-like
        Origami geometry spec (see :func:`load_origami_template`).
    n_frames : int
        Number of frames in the acquisition (for per-frame nlocs).
    kinetics : dict, optional
        ``{k_on, tau_b, concentration, exposure}`` -> mean locs per site.
    mean_locs_per_site : float, optional
        Explicit mean locs per site (overrides ``kinetics``).
    missing_sites_allowed, site_uncertainty_nm, n_sim, sim_quantile,
    random_seed : see :func:`simulate_origami_nlocs_rmsd` /
        :func:`phase_space_window_from_sim` / :func:`pick_origami`.
    pixelsize : float, optional
        Camera pixel size in nm. Default 130.
    grid_spacing_nm : float, optional
        Physical spacing to anchor a design-file scale.

    Returns
    -------
    dict
        ``n_sites_expected``, ``grid_spacing_nm``, ``mean_locs_per_site``,
        the simulated cloud (``sim_nlocs`` counts / ``sim_nlocs_per_frame`` /
        ``sim_rmsd_px``), and the suggested window
        (``min_n_locs_per_frame`` / ``max_n_locs_per_frame`` /
        ``min_rmsd`` / ``max_rmsd`` in px).
    """
    template = load_origami_template(geometry, grid_spacing_nm=grid_spacing_nm)
    if mean_locs_per_site is None:
        if not kinetics:
            raise ValueError(
                "origami_phase_space_preview needs mean_locs_per_site or "
                "kinetics {k_on, tau_b, concentration, exposure}"
            )
        mean_locs_per_site = predict_locs_per_site(
            kinetics["k_on"],
            kinetics["tau_b"],
            kinetics["concentration"],
            n_frames,
            kinetics["exposure"],
        )
    sim_nlocs, sim_rmsds_nm = simulate_origami_nlocs_rmsd(
        template,
        site_uncertainty_nm,
        missing_sites_allowed,
        mean_locs_per_site,
        n_sim=n_sim,
        random_seed=random_seed,
    )
    window = phase_space_window_from_sim(
        sim_nlocs, sim_rmsds_nm, pixelsize, sim_quantile
    )
    return {
        "n_sites_expected": template.n_sites_expected,
        "grid_spacing_nm": template.grid_spacing_nm,
        "mean_locs_per_site": float(mean_locs_per_site),
        "sim_nlocs": sim_nlocs,
        "sim_nlocs_per_frame": sim_nlocs / n_frames,
        "sim_rmsd_px": sim_rmsds_nm / pixelsize,
        "min_n_locs_per_frame": window["min_nlocs"] / n_frames,
        "max_n_locs_per_frame": window["max_nlocs"] / n_frames,
        "min_rmsd": window["min_rmsd"],
        "max_rmsd": window["max_rmsd"],
    }


# Default DBSCAN neighbourhood for resolving docking sites, as a fraction of
# the design site spacing. Must be well below the inter-site gap or DBSCAN
# *chains* adjacent sites through their localization tails and under-counts
# sites (worse the brighter the structure). A fifth of the spacing groups a
# site's own localizations while keeping neighbours apart. Tunable per run via
# pick_origami's ``pattern_eps_frac``.
_SITE_EPS_FRAC = 0.2


def subcluster_docking_sites(
    xy_nm,
    expected_spacing_nm,
    min_samples=3,
    eps_frac=_SITE_EPS_FRAC,
    return_stats=False,
):
    """Cluster localizations within a footprint into docking-site centers.

    Parameters
    ----------
    xy_nm : array-like
        ``(M, 2)`` localization coordinates (nanometres) within one origami
        footprint.
    expected_spacing_nm : float
        Expected inter-site spacing; sets the DBSCAN neighbourhood radius.
    min_samples : int, optional
        DBSCAN ``min_samples``. Default 3.
    eps_frac : float, optional
        DBSCAN ``eps`` as a fraction of ``expected_spacing_nm``. Default
        ``_SITE_EPS_FRAC`` (a fifth of the site spacing). Values much above
        this let DBSCAN *chain* adjacent sites together through their tails -
        the more localizations a site has, the more likely the bridge - which
        systematically **under**-counts the sites of bright, well-resolved
        structures. Lower it further if sites are still merged; raise it if a
        single site splits into several.
    return_stats : bool, optional
        Also return, per resolved site (in ``centers`` order), the number of
        localizations and their RMS spread (nm) about the site centre.

    Returns
    -------
    np.ndarray
        ``(K, 2)`` resolved docking-site centroids in nanometres (empty if
        none). If ``return_stats``, a ``(centers, site_nlocs, site_spread_nm)``
        tuple instead.
    """
    # DBSCAN min_samples=1 is degenerate - every isolated localization becomes
    # its own "site", grossly over-counting sites (and wrecking the lattice
    # gate's frac-on-lattice / count-uniformity). A docking site needs >= 2.
    min_samples = max(2, int(min_samples))
    xy = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    empty = np.empty((0, 2))
    if len(xy) == 0 or expected_spacing_nm <= 0:
        if return_stats:
            return empty, np.empty(0), np.empty(0)
        return empty
    eps = eps_frac * expected_spacing_nm
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    centers, nlocs, spread = [], [], []
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        pts = xy[labels == lbl]
        c = pts.mean(axis=0)
        centers.append(c)
        nlocs.append(len(pts))
        spread.append(
            float(np.sqrt(np.mean(((pts - c) ** 2).sum(axis=1))))
            if len(pts) > 1
            else 0.0
        )
    centers = np.array(centers) if centers else empty
    if return_stats:
        return (
            centers,
            np.asarray(nlocs, dtype=float),
            np.asarray(spread, dtype=float),
        )
    return centers


def _rotation_matrix(theta):
    """2x2 rotation matrix for angle ``theta`` (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _rigid_fit(src, dst):
    """Least-squares proper-rotation + translation mapping src onto dst.

    Kabsch/Umeyama without scaling or reflection: returns ``R`` (det=+1)
    and ``t`` minimizing ``||R @ src.T + t - dst.T||``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    h = src_c.T @ dst_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, d])
    rot = vt.T @ correction @ u.T
    trans = dst.mean(axis=0) - rot @ src.mean(axis=0)
    return rot, trans


def _match_within_gate(transformed, template, gate):
    """Optimal one-to-one assignment of points within a distance gate.

    Returns matched ``(src_indices, template_indices, distances)``.
    """
    if len(transformed) == 0 or len(template) == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0)
    cost = np.linalg.norm(
        transformed[:, None, :] - template[None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    dists = cost[rows, cols]
    keep = dists <= gate
    return rows[keep], cols[keep], dists[keep]


def register_to_template(
    observed_nm,
    template_nm,
    allow_mirror=True,
    match_gate_nm=None,
    angle_step_deg=2.0,
    n_refine=4,
):
    """Register observed docking sites onto a design template.

    Finds the best rigid alignment (rotation + translation, optional
    mirror) of ``observed_nm`` onto ``template_nm`` under an unknown
    correspondence, tolerating missing/extra observed sites. A coarse
    rotation sweep with gated optimal assignment seeds a Kabsch + ICP
    refinement.

    Parameters
    ----------
    observed_nm : array-like
        ``(M, 2)`` resolved docking-site coordinates in nanometres.
    template_nm : array-like
        ``(N, 2)`` expected site coordinates in nanometres.
    allow_mirror : bool, optional
        Whether to also try a mirrored observed pattern. Default True.
    match_gate_nm : float, optional
        Max distance for a site to count as matched. Defaults to half the
        template's median nearest-neighbour spacing.
    angle_step_deg : float, optional
        Coarse rotation-sweep step in degrees. Default 2.
    n_refine : int, optional
        Number of best coarse seeds to ICP-refine. The rotation sweep is
        scored cheaply (one gated assignment per seed) and only the best
        ``n_refine`` seeds get the full ICP, instead of ICP-ing every seed.
        Default 4.

    Returns
    -------
    dict
        Registration result with keys ``n_resolved``,
        ``matched_template_indices``, ``matched_observed_indices``,
        ``rmse_nm``, ``mean_spacing_nm``, ``orientation_deg``, ``mirror``,
        ``R`` and ``t``.
    """
    observed = np.asarray(observed_nm, dtype=float).reshape(-1, 2)
    template = np.asarray(template_nm, dtype=float).reshape(-1, 2)
    template = template - template.mean(axis=0)
    if match_gate_nm is None:
        match_gate_nm = 0.5 * _median_nn_spacing(template)
    if match_gate_nm <= 0:
        match_gate_nm = np.inf

    def _empty_result():
        return {
            "n_resolved": 0,
            "matched_template_indices": np.empty(0, dtype=int),
            "matched_observed_indices": np.empty(0, dtype=int),
            "rmse_nm": np.inf,
            "mean_spacing_nm": 0.0,
            "orientation_deg": 0.0,
            "mirror": False,
            "R": np.eye(2),
            "t": np.zeros(2),
        }

    if len(observed) == 0:
        return _empty_result()

    mirror_options = [False, True] if allow_mirror else [False]

    # Stage 1 - cheaply score every rotation seed with a single gated
    # assignment (no ICP). Stage 2 - ICP-refine only the best `n_refine`
    # seeds. This is ~(iters x) cheaper than ICP-ing every seed while still
    # converging on the true alignment (the coarse best is within
    # angle_step/2 of it).
    seeds = []  # (n_matched, -rmse0, mirror, rot0, trans0)
    for mirror in mirror_options:
        src = observed * np.array([-1.0, 1.0]) if mirror else observed
        src_mean = src.mean(axis=0)
        for angle in np.arange(0.0, 360.0, angle_step_deg):
            rot0 = _rotation_matrix(np.deg2rad(angle))
            trans0 = -(rot0 @ src_mean)
            transformed = src @ rot0.T + trans0
            rows, cols, dists = _match_within_gate(
                transformed, template, match_gate_nm
            )
            rmse0 = float(np.sqrt(np.mean(dists**2))) if len(dists) else np.inf
            seeds.append((len(rows), -rmse0, mirror, rot0, trans0))
    seeds.sort(key=lambda s: (s[0], s[1]), reverse=True)

    best = None  # (n_matched, rmse, rot, trans, mirror, rows, cols)
    for n0, _negrmse0, mirror, rot0, trans0 in seeds[: max(1, n_refine)]:
        src = observed * np.array([-1.0, 1.0]) if mirror else observed
        rot, trans, rows, cols, rmse = _icp_refine(
            src, template, rot0, trans0, match_gate_nm
        )
        n_matched = len(rows)
        key = (n_matched, -rmse)
        if best is None or key > (best[0], -best[1]):
            best = (n_matched, rmse, rot, trans, mirror, rows, cols)

    if best is None or best[0] == 0:
        return _empty_result()

    n_matched, rmse, rot, trans, mirror, rows, cols = best
    src = observed * np.array([-1.0, 1.0]) if mirror else observed
    matched_obs = src[rows]
    orientation = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    return {
        "n_resolved": int(n_matched),
        "matched_template_indices": cols,
        "matched_observed_indices": rows,
        "rmse_nm": float(rmse),
        "mean_spacing_nm": _median_nn_spacing(matched_obs),
        "orientation_deg": orientation,
        "mirror": bool(mirror),
        "R": rot,
        "t": trans,
    }


def _icp_refine(src, template, rot, trans, gate, iters=5):
    """Iterative closest-point refinement of a rigid alignment.

    Returns ``(rot, trans, matched_src_idx, matched_template_idx, rmse)``.
    """
    rows = cols = np.empty(0, dtype=int)
    rmse = np.inf
    for _ in range(iters):
        transformed = src @ rot.T + trans
        rows, cols, dists = _match_within_gate(transformed, template, gate)
        if len(rows) < 2:
            rmse = float(np.sqrt(np.mean(dists**2))) if len(dists) else np.inf
            break
        rmse = float(np.sqrt(np.mean(dists**2)))
        new_rot, new_trans = _rigid_fit(src[rows], template[cols])
        if np.allclose(new_rot, rot) and np.allclose(new_trans, trans):
            rot, trans = new_rot, new_trans
            transformed = src @ rot.T + trans
            rows, cols, dists = _match_within_gate(transformed, template, gate)
            rmse = float(np.sqrt(np.mean(dists**2))) if len(dists) else np.inf
            break
        rot, trans = new_rot, new_trans
    return rot, trans, rows, cols, rmse


def classify_candidate(
    reg,
    n_sites_expected,
    missing_sites_allowed=2,
    spacing_tol=0.3,
    grid_spacing_nm=None,
    max_rmse_nm=None,
):
    """Classify a registered candidate as accepted or by rejection reason.

    Applies the acceptance criteria in order and returns the first one that
    fails, so the caller can build a rejection funnel.

    Parameters
    ----------
    reg : dict
        A :func:`register_to_template` result.
    n_sites_expected : int
        Number of docking sites in a fully-resolved origami.
    missing_sites_allowed : int, optional
        Maximum number of missing sites tolerated. Default 2.
    spacing_tol : float, optional
        Relative tolerance on the resolved-site spacing vs the design
        spacing. Default 0.3.
    grid_spacing_nm : float, optional
        Design inter-site spacing; enables the spacing check when given.
    max_rmse_nm : float, optional
        Maximum RMSE-vs-design; enables the RMSE check when given.

    Returns
    -------
    str
        ``"accepted"``, or one of ``"missing_sites"`` / ``"rmse"`` /
        ``"spacing"`` naming the first criterion that rejected it.
    """
    if reg["n_resolved"] < n_sites_expected - missing_sites_allowed:
        return "missing_sites"
    if max_rmse_nm is not None and reg["rmse_nm"] > max_rmse_nm:
        return "rmse"
    if grid_spacing_nm is not None and grid_spacing_nm > 0:
        spacing = reg["mean_spacing_nm"]
        if spacing > 0:
            rel = abs(spacing - grid_spacing_nm) / grid_spacing_nm
            if rel > spacing_tol:
                return "spacing"
    return "accepted"


def accept_candidate(
    reg,
    n_sites_expected,
    missing_sites_allowed=2,
    spacing_tol=0.3,
    grid_spacing_nm=None,
    max_rmse_nm=None,
):
    """Decide whether a registered candidate is an accepted origami.

    Thin wrapper over :func:`classify_candidate` (see it for the criteria).

    Returns
    -------
    bool
        Whether the candidate is accepted.
    """
    return (
        classify_candidate(
            reg,
            n_sites_expected,
            missing_sites_allowed=missing_sites_allowed,
            spacing_tol=spacing_tol,
            grid_spacing_nm=grid_spacing_nm,
            max_rmse_nm=max_rmse_nm,
        )
        == "accepted"
    )


def _candidate_centers_cluster_of_clusters(
    locs, info, site_diameter, footprint_extent_px, min_n_locs, min_samples
):
    """Coarse origami-candidate centers by clustering docking-site picks.

    Finds docking-site-like pick centers with a small (single-site)
    diameter, then spatially clusters those centers into origami
    footprints; each footprint's centroid is a candidate.
    """
    x_sim, y_sim, rmsds, nlocs = get_pick_similar_vals(
        locs, info, site_diameter, min_n_locs
    )
    site_centers = np.column_stack([x_sim, y_sim])
    if len(site_centers) == 0:
        return [], nlocs, rmsds, np.empty(0)
    labels = DBSCAN(
        eps=footprint_extent_px, min_samples=min_samples
    ).fit_predict(site_centers)
    centers = [
        tuple(site_centers[labels == lbl].mean(axis=0))
        for lbl in np.unique(labels)
        if lbl != -1
    ]
    return centers, nlocs, rmsds, labels


########################################################################
# Pattern clustering: group picked structures by their resolved geometry
#
# Describes each picked structure with a rotation- and translation-
# invariant, scale-aware "site-graph" descriptor (resolved docking-site
# count + pairwise-distance histogram + spread/elongation), then clusters
# those descriptors so distinct designs/incomplete-labelling states/
# aggregates fall into separate groups. Pure numpy/scipy/sklearn.
########################################################################


def _elongation(pts):
    """Isotropy of a point set: sqrt(min/max covariance eigenvalue).

    Rotation-invariant. 0 for a perfectly collinear set (a line), 1 for an
    isotropic set (e.g. a square grid). 0 for < 2 points.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    cov = np.cov(pts.T)
    w = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    if w[-1] <= 0:
        return 0.0
    return float(np.sqrt(w[0] / w[-1]))


def _radius_of_gyration(xy):
    """RMS distance of points from their centre of mass (0 if < 2 points)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        return 0.0
    com = xy.mean(axis=0)
    return float(np.sqrt(np.mean(((xy - com) ** 2).sum(axis=1))))


def structure_pattern_features(
    xy_nm,
    expected_spacing_nm,
    max_pair_nm,
    n_dist_bins=12,
    min_samples=3,
    eps_frac=_SITE_EPS_FRAC,
):
    """Rotation/translation-invariant pattern descriptor of one structure.

    Resolves the localizations into docking sites (:func:`
    subcluster_docking_sites`) and describes the site arrangement with
    invariant features: the resolved site count, the localization and
    site-centre radii of gyration (nm), the median nearest-neighbour site
    spacing (nm), the site-set elongation, and a normalized histogram of
    the pairwise site-centre distances (the "site graph"). Translation and
    rotation invariant by construction; scale-aware because distances stay
    in nanometres.

    Parameters
    ----------
    xy_nm : array-like
        ``(M, 2)`` localization coordinates (nm) of one picked structure.
    expected_spacing_nm : float
        Design inter-site spacing; sets the subclustering neighbourhood.
    max_pair_nm : float
        Upper edge of the pairwise-distance histogram (nm). Distances above
        this land in the last bin.
    n_dist_bins : int, optional
        Number of pairwise-distance histogram bins. Default 12.
    min_samples, eps_frac : see :func:`subcluster_docking_sites`.

    Returns
    -------
    (np.ndarray, list of str, dict)
        The feature vector, the matching feature names, and an ``aux`` dict
        of interpretable scalars (``n_locs``, ``n_sites``, ``loc_rg_nm``,
        ``site_rg_nm``, ``nn_nm``, ``elongation``, ``site_centers_nm``).
    """
    xy = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    n_locs = len(xy)
    loc_rg = _radius_of_gyration(xy)
    sites = subcluster_docking_sites(
        xy, expected_spacing_nm, min_samples=min_samples, eps_frac=eps_frac
    )
    n_sites = len(sites)

    edges = np.linspace(0.0, float(max_pair_nm), n_dist_bins + 1)
    hist = np.zeros(n_dist_bins, dtype=float)
    if n_sites >= 2:
        d = pdist(sites)
        # clip to the last bin so far-apart sites still register
        d = np.clip(d, 0.0, edges[-1] - 1e-9)
        hist, _ = np.histogram(d, bins=edges)
        hist = hist.astype(float)
        total = hist.sum()
        if total > 0:
            hist /= total
        site_rg = _radius_of_gyration(sites)
        nn = _median_nn_spacing(sites)
        elong = _elongation(sites)
    else:
        site_rg = 0.0
        nn = 0.0
        elong = 0.0

    scalars = np.array([n_sites, loc_rg, site_rg, nn, elong], dtype=float)
    feats = np.concatenate([scalars, hist])
    names = [
        "n_sites",
        "loc_rg_nm",
        "site_rg_nm",
        "nn_nm",
        "elongation",
    ] + [f"pdist_{i}" for i in range(n_dist_bins)]
    aux = {
        "n_locs": n_locs,
        "n_sites": n_sites,
        "loc_rg_nm": loc_rg,
        "site_rg_nm": site_rg,
        "nn_nm": nn,
        "elongation": elong,
        "site_centers_nm": sites,
    }
    return feats, names, aux


def cluster_structure_patterns(
    structures_xy_nm,
    expected_spacing_nm,
    max_pair_nm=None,
    n_dist_bins=12,
    k=None,
    min_cluster_size=25,
    min_samples=3,
    eps_frac=_SITE_EPS_FRAC,
    random_seed=0,
):
    """Cluster picked structures into geometric-pattern groups.

    Builds the invariant site-graph descriptor
    (:func:`structure_pattern_features`) for every structure, standardizes
    the feature matrix, and clusters it. With ``k`` unset the cluster count
    is auto-discovered with HDBSCAN (structures too dissimilar to join a
    cluster get label ``-1``); with ``k`` given a ``k``-component Gaussian
    mixture assigns every structure to one of ``k`` clusters.

    Parameters
    ----------
    structures_xy_nm : sequence of array-like
        One ``(M_i, 2)`` array of localization coordinates (nm) per picked
        structure.
    expected_spacing_nm : float
        Design inter-site spacing (nm).
    max_pair_nm : float, optional
        Upper edge of the pairwise-distance histogram (nm). Defaults to the
        99th percentile of observed site-centre spans (fallback: 3x the
        expected spacing).
    n_dist_bins : int, optional
        Pairwise-distance histogram bins. Default 12.
    k : int, optional
        Fixed number of clusters (Gaussian mixture). If None (default), the
        count is auto-discovered with HDBSCAN.
    min_cluster_size : int, optional
        HDBSCAN minimum cluster size (auto mode only). Default 25.
    min_samples, eps_frac : see :func:`subcluster_docking_sites`.
    random_seed : int, optional
        Seed for the Gaussian mixture (fixed-k mode). Default 0.

    Returns
    -------
    dict
        ``labels`` (per structure; ``-1`` = unclustered/noise),
        ``features`` (n x d matrix), ``feature_names``, ``aux`` (list of
        per-structure aux dicts), ``max_pair_nm``, ``method``, and
        ``cluster_summary`` (per-label counts and median n_sites / n_locs /
        spacing / rg, sorted by median site count descending).
    """
    structures = [
        np.asarray(s, dtype=float).reshape(-1, 2) for s in structures_xy_nm
    ]
    n = len(structures)
    if n == 0:
        return {
            "labels": np.empty(0, dtype=int),
            "features": np.empty((0, 0)),
            "feature_names": [],
            "aux": [],
            "max_pair_nm": 0.0,
            "method": "none",
            "cluster_summary": [],
        }

    # choose a histogram span from the data if not given
    if max_pair_nm is None or max_pair_nm <= 0:
        spans = []
        for s in structures:
            if len(s) >= 2:
                spans.append(float(np.max(pdist(s))))
        max_pair_nm = (
            float(np.percentile(spans, 99))
            if spans
            else 3.0 * expected_spacing_nm
        )
        max_pair_nm = max(max_pair_nm, 1e-6)

    # Per-structure descriptor (subcluster + pairwise histogram). This is a
    # single-threaded Python loop over every accepted structure and is the
    # dominant cost; a heartbeat every ~2000 structures makes a long run
    # visibly progressing rather than looking hung.
    logger.debug(
        "cluster_structure_patterns: describing %d structures "
        "(max_pair_nm=%.1f)",
        n,
        max_pair_nm,
    )
    feats, names, aux = [], None, []
    for idx, s in enumerate(structures):
        f, names, a = structure_pattern_features(
            s,
            expected_spacing_nm,
            max_pair_nm,
            n_dist_bins=n_dist_bins,
            min_samples=min_samples,
            eps_frac=eps_frac,
        )
        feats.append(f)
        aux.append(a)
        if n > 4000 and (idx + 1) % 2000 == 0:
            logger.debug(
                "cluster_structure_patterns: described %d/%d structures",
                idx + 1,
                n,
            )
    features = np.vstack(feats)

    # standardize so the histogram bins and scalar features are comparable
    scaled = StandardScaler().fit_transform(features)

    if k is not None and k > 0:
        gm = GaussianMixture(n_components=int(k), random_state=random_seed)
        labels = gm.fit_predict(scaled).astype(int)
        method = f"gmm(k={int(k)})"
    else:
        hdb = HDBSCAN(min_cluster_size=int(min_cluster_size))
        labels = hdb.fit_predict(scaled).astype(int)
        method = "hdbscan"

    cluster_summary = _summarize_pattern_clusters(labels, aux)
    logger.debug(
        "cluster_structure_patterns: %s -> %d clusters (+%d noise) "
        "from %d structures",
        method,
        int(len(set(labels.tolist()) - {-1})),
        int(np.count_nonzero(labels == -1)),
        n,
    )
    return {
        "labels": labels,
        "features": features,
        "feature_names": names,
        "aux": aux,
        "max_pair_nm": float(max_pair_nm),
        "method": method,
        "cluster_summary": cluster_summary,
    }


def _summarize_pattern_clusters(labels, aux):
    """Per-cluster medians (n_sites/n_locs/spacing/rg), site-count sorted."""
    labels = np.asarray(labels, dtype=int)
    summary = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]

        def _med(key):
            vals = [aux[i][key] for i in idx]
            return float(np.median(vals)) if vals else 0.0

        summary.append(
            {
                "label": int(lbl),
                "n_structures": int(len(idx)),
                "median_n_sites": _med("n_sites"),
                "median_n_locs": _med("n_locs"),
                "median_nn_nm": _med("nn_nm"),
                "median_site_rg_nm": _med("site_rg_nm"),
                "is_noise": bool(lbl == -1),
            }
        )
    summary.sort(
        key=lambda c: (c["is_noise"], -c["median_n_sites"], -c["n_structures"])
    )
    return summary


########################################################################
# Design-aware (lattice) pattern clustering
#
# When the design is a known lattice, register each pick onto it and
# describe it in the lattice's own frame: (a) how well it fits (best-fit
# similarity residual, recovered spacing, on-lattice fraction) and (b) its
# defect pattern (which design nodes are occupied). This sees lattice
# quality and defect geometry that the template-agnostic pairwise-distance
# descriptor cannot. Reuses register_to_template.
########################################################################


def _similarity_fit(src, dst):
    """Best-fit similarity (rotation + isotropic scale + translation) mapping
    ``src`` onto ``dst`` (Umeyama). Returns ``(scale, R, t, rmse)``."""
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    n = len(src)
    if n == 0:
        return 1.0, np.eye(2), np.zeros(2), np.inf
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    sigma = (xd.T @ xs) / n
    u, d, vt = np.linalg.svd(sigma)
    s_corr = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_corr[-1, -1] = -1.0
    rot = u @ s_corr @ vt
    var_s = (xs**2).sum() / n
    scale = float((d * np.diag(s_corr)).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (rot @ mu_s)
    transformed = scale * (src @ rot.T) + t
    rmse = float(np.sqrt(np.mean(((transformed - dst) ** 2).sum(axis=1))))
    return scale, rot, t, rmse


def template_symmetry_permutations(template_nm, tol_frac=0.15):
    """Node-index permutations of the design's symmetry group.

    The rigid + mirror transforms that map the node set onto itself, as
    permutations of the node indices. Canonicalizing the defect-occupancy
    vector over these makes symmetry-equivalent defect patterns identical
    (e.g. a missing corner is the same defect whichever corner registration
    happened to land it on).
    """
    template = np.asarray(template_nm, dtype=float).reshape(-1, 2)
    template = template - template.mean(axis=0)
    n = len(template)
    if n == 0:
        return [np.arange(0, dtype=int)]
    nn = _median_nn_spacing(template)
    tol = tol_frac * nn if nn > 0 else 1e-6
    tree = KDTree(template)
    perms, seen = [], set()
    for k in range(4):  # rotations 0/90/180/270 x {identity, x-reflection}
        a = 0.5 * np.pi * k
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for mirror in (np.eye(2), np.array([[-1.0, 0.0], [0.0, 1.0]])):
            transformed = template @ (rot @ mirror).T
            dist, idx = tree.query(transformed)
            if np.all(dist <= tol) and len(set(idx.tolist())) == n:
                key = tuple(int(v) for v in idx)
                if key not in seen:
                    seen.add(key)
                    perms.append(np.asarray(idx, dtype=int))
    return perms or [np.arange(n, dtype=int)]


def _canonical_occupancy(occ, perms):
    """Lexicographically maximal occupancy over the symmetry group - a
    canonical, symmetry-invariant representative of the defect pattern."""
    occ = np.asarray(occ, dtype=int)
    best = tuple(int(v) for v in occ)
    for p in perms:
        cand = tuple(int(v) for v in occ[p])
        if cand > best:
            best = cand
    return best


def lattice_pair_score(
    xy_nm,
    expected_spacing_nm,
    band_nm=4.0,
    ref_lo_nm=6.0,
    ref_hi_nm=14.0,
    max_locs=1500,
    random_seed=0,
):
    """Cheap, registration-free 'is this a lattice?' score for one pick.

    A windowed pair-correlation: the fraction of localization pairs
    separated by the design spacing (``expected_spacing_nm`` +/- ``band_nm``)
    over the fraction in a short-range reference band (``ref_lo_nm`` ..
    ``ref_hi_nm``), each divided by its band centre so a structureless disk
    scores ~1. A genuine lattice has a sharp excess at the design spacing and
    scores well above 1; an amorphous blob or a 1-2 spot cluster scores near
    0. Costs only the intra-footprint pairwise distances (no registration),
    so it is the first-layer discriminator computed on every pick.

    For very dense footprints the localizations are randomly subsampled to
    ``max_locs`` to bound the pairwise cost (the ratio is unbiased by this).
    """
    p = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    n = len(p)
    if n < 4 or expected_spacing_nm <= 0:
        return 0.0
    if n > max_locs:
        rng = np.random.default_rng(random_seed)
        p = p[rng.choice(n, size=max_locs, replace=False)]
    d = pdist(p)
    sp = float(expected_spacing_nm)
    peak = np.mean((d > sp - band_nm) & (d < sp + band_nm)) / sp
    ref_c = 0.5 * (ref_lo_nm + ref_hi_nm)
    ref = np.mean((d > ref_lo_nm) & (d < ref_hi_nm)) / ref_c
    return float(peak / ref) if ref > 0 else 0.0


def _empty_lattice_aux(n_nodes):
    """Default off-lattice ``aux`` (no registration) - used for picks the
    cheap pair-score pre-screen rejects before the expensive fit."""
    occ = np.zeros(n_nodes, dtype=int)
    return {
        "n_sites": 0,
        "n_matched": 0,
        "frac_on_lattice": 0.0,
        "rmse_nm": float("inf"),
        "fitted_spacing_nm": 0.0,
        "nlocs_cv": float("inf"),
        "spread_cv": float("inf"),
        "mean_site_spread_nm": float("inf"),
        "occupancy": tuple(int(v) for v in occ),
        "site_centers_nm": np.empty((0, 2)),
        "site_nodes": np.empty(0, dtype=int),
        "site_nlocs_matched": np.empty(0),
    }


def lattice_defect_features(
    xy_nm,
    template_nm,
    expected_spacing_nm,
    allow_mirror=True,
    min_samples=3,
    eps_frac=_SITE_EPS_FRAC,
    sym_perms=None,
):
    """Design-aware descriptor of one pick.

    Registers the pick's resolved sites onto the design lattice and reports
    (a) lattice-fit quality - best-fit similarity residual (nm), recovered
    spacing, matched fraction - and (b) the defect pattern - a canonical
    per-node occupancy vector. Returns a dict of features (``occupancy`` is
    the symmetry-canonical tuple; ``rmse_nm`` is the residual after removing
    the global best-fit scale, i.e. local site disorder).
    """
    template = np.asarray(template_nm, dtype=float).reshape(-1, 2)
    template = template - template.mean(axis=0)
    n_nodes = len(template)
    if sym_perms is None:
        sym_perms = template_symmetry_permutations(template)
    sites, site_nlocs, _ = subcluster_docking_sites(
        xy_nm,
        expected_spacing_nm,
        min_samples=min_samples,
        eps_frac=eps_frac,
        return_stats=True,
    )
    occ = np.zeros(n_nodes, dtype=int)
    # per resolved site: the design-node index it registers to (-1 = no match)
    site_nodes = -np.ones(len(sites), dtype=int)
    aux = {
        "n_sites": int(len(sites)),
        "n_matched": 0,
        "frac_on_lattice": 0.0,
        "rmse_nm": float("inf"),
        "fitted_spacing_nm": 0.0,
        "nlocs_cv": float("inf"),
        "spread_cv": float("inf"),
        "mean_site_spread_nm": float("inf"),
        "occupancy": tuple(int(v) for v in occ),
        "site_centers_nm": sites,
        "site_nodes": site_nodes,
        "site_nlocs_matched": np.empty(0),
    }
    if len(sites) < 3:
        return aux
    reg = register_to_template(sites, template, allow_mirror=allow_mirror)
    cols = np.asarray(reg["matched_template_indices"], dtype=int)
    rows = np.asarray(reg["matched_observed_indices"], dtype=int)
    if len(cols) < 3:
        aux["rmse_nm"] = float(reg["rmse_nm"])
        return aux
    occ[cols] = 1
    site_nodes[rows] = cols
    src = sites * np.array([-1.0, 1.0]) if reg["mirror"] else sites
    # similarity maps observed sites onto the design template, so its scale is
    # design/observed; the pick's actual spacing is design_nn / scale.
    scale, _, _, rmse_scaled = _similarity_fit(src[rows], template[cols])
    design_nn = _median_nn_spacing(template)
    # uniformity of the ON-LATTICE sites: a genuine origami blinks similarly at
    # every site and every site's localization cloud has the same spread;
    # aggregates / merged or smeared sites do not. CV = std / mean.
    #  - counts per site from the (tight) DBSCAN cores (background-free)
    #  - spread per site from ALL locs assigned to their nearest matched site
    #    (within half a spacing) - the eps-bounded core spread cannot see a
    #    smeared/merged site, this can.
    m_nlocs = site_nlocs[rows]
    nlocs_cv = (
        float(np.std(m_nlocs) / np.mean(m_nlocs))
        if m_nlocs.mean()
        else float("inf")
    )
    matched_centers = sites[rows]
    xy = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    dist, assign = KDTree(matched_centers).query(xy)
    keep = dist < 0.5 * expected_spacing_nm
    per_site_spread = []
    for k in range(len(matched_centers)):
        sel = keep & (assign == k)
        if np.count_nonzero(sel) > 1:
            per_site_spread.append(float(np.sqrt(np.mean(dist[sel] ** 2))))
    per_site_spread = np.asarray(per_site_spread)
    spread_cv = (
        float(np.std(per_site_spread) / np.mean(per_site_spread))
        if len(per_site_spread) >= 2 and per_site_spread.mean() > 0
        else float("inf")
    )
    mean_spread = (
        float(np.mean(per_site_spread))
        if len(per_site_spread)
        else float("inf")
    )
    aux.update(
        {
            "n_matched": int(len(cols)),
            "frac_on_lattice": float(len(cols) / len(sites)),
            "rmse_nm": float(rmse_scaled),
            "fitted_spacing_nm": (
                float(design_nn / scale) if scale > 0 else 0.0
            ),
            "nlocs_cv": nlocs_cv,
            "spread_cv": spread_cv,
            "mean_site_spread_nm": mean_spread,
            "median_site_nlocs": float(np.median(m_nlocs)),
            "site_nlocs_matched": m_nlocs,
            "occupancy": _canonical_occupancy(occ, sym_perms),
        }
    )
    return aux


def _summarize_lattice_clusters(labels, aux, n_nodes):
    """Per-class summary for the lattice-defect clustering.

    ``occupancy`` is the per-node occupancy *probability* over the class
    members (0/1 for an exact-pattern class; a fraction for a completeness
    class, revealing which nodes tend to be missing).
    """
    labels = np.asarray(labels, dtype=int)
    summary = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]

        def _med(key):
            vals = [aux[i][key] for i in idx if np.isfinite(aux[i][key])]
            return float(np.median(vals)) if vals else float("nan")

        off = lbl == -1
        if len(idx) and not off:
            occ_mat = np.array([aux[i]["occupancy"] for i in idx], dtype=float)
            occ = [float(v) for v in occ_mat.mean(axis=0)]
            n_occ = int(round(float(occ_mat.sum(axis=1).mean())))
        else:
            occ = [0.0] * n_nodes
            n_occ = 0
        summary.append(
            {
                "label": int(lbl),
                "n_structures": int(len(idx)),
                "is_offlattice": bool(off),
                "occupancy": occ,
                "n_sites_occupied": n_occ,
                "n_defects": (None if off else int(n_nodes - n_occ)),
                "median_rmse_nm": _med("rmse_nm"),
                "median_fitted_spacing_nm": _med("fitted_spacing_nm"),
                "median_frac_on_lattice": _med("frac_on_lattice"),
                "median_n_sites": _med("n_sites"),
                "median_nlocs_cv": _med("nlocs_cv"),
                "median_spread_cv": _med("spread_cv"),
                "median_site_spread_nm": _med("mean_site_spread_nm"),
                "median_site_nlocs": (
                    float(
                        np.median(
                            np.concatenate(
                                [
                                    np.asarray(
                                        aux[i].get("site_nlocs_matched", [])
                                    )
                                    for i in idx
                                ]
                            )
                        )
                    )
                    if not off
                    and any(
                        len(aux[i].get("site_nlocs_matched", [])) for i in idx
                    )
                    else float("nan")
                ),
            }
        )
    # on-lattice first, then most complete (fewest defects), then most populous
    summary.sort(
        key=lambda c: (
            c["is_offlattice"],
            -c["n_sites_occupied"],
            -c["n_structures"],
        )
    )
    return summary


def cluster_lattice_defects(
    structures_xy_nm,
    template_nm,
    expected_spacing_nm,
    allow_mirror=True,
    min_on_lattice_sites=None,
    min_on_lattice_frac=0.66,
    rmse_gate_frac=0.2,
    frac_on_lattice_gate=0.8,
    spacing_tol=0.3,
    max_nlocs_cv=0.5,
    max_spread_cv=0.3,
    defect_grouping="completeness",
    min_samples=3,
    eps_frac=_SITE_EPS_FRAC,
    min_pair_score=0.5,
    pair_score_band_nm=4.0,
):
    """Two-stage design-aware clustering of picks on a known lattice.

    Stage A separates *on-lattice* picks from off-lattice / sparse ones
    (label ``-1``). A pick is on-lattice when it registers to enough of the
    design nodes (``min_on_lattice_sites`` / ``min_on_lattice_frac``), with a
    low best-fit residual, recovered spacing near design, few off-lattice
    extras (``frac_on_lattice_gate``), **and** localizations similarly
    distributed across its sites - similar counts (``max_nlocs_cv``) and
    similar spread (``max_spread_cv``) - as a genuine origami has but an
    aggregate or misregistered blob does not.

    Stage B groups the on-lattice picks. ``defect_grouping="completeness"``
    (default) groups by the number of occupied sites (full / 1-defect /
    2-defect / ... - a handful of interpretable classes robust to the
    per-structure blinking noise that fragments exact patterns);
    ``"exact"`` groups by the canonical defect pattern (which specific nodes
    are missing).

    A cheap **pair-score pre-screen** runs first (:func:`lattice_pair_score`,
    computed on every pick like nlocs/rmsd): picks whose windowed
    pair-correlation at the design spacing is below ``min_pair_score`` are
    marked off-lattice without the expensive registration, so the fit only
    runs on genuine lattice candidates (set ``min_pair_score=None`` to
    disable). Every structure's score is kept in ``aux[i]["pair_score"]``.

    Returns ``labels``, per-structure ``aux``, ``cluster_summary``,
    ``n_nodes`` and the ``symmetry_perms``.
    """
    template = np.asarray(template_nm, dtype=float).reshape(-1, 2)
    template = template - template.mean(axis=0)
    n_nodes = len(template)
    perms = template_symmetry_permutations(template)
    design_nn = _median_nn_spacing(template)
    rmse_gate = rmse_gate_frac * expected_spacing_nm
    # min matched sites: the larger of an absolute floor and a fraction of the
    # design nodes, so junk that matches only a few nodes is rejected.
    floor = 4 if min_on_lattice_sites is None else int(min_on_lattice_sites)
    min_sites = max(floor, int(np.ceil(min_on_lattice_frac * n_nodes)))
    # first layer: cheap pair-correlation score on every pick; only picks that
    # pass pay for the (expensive) registration below.
    aux = []
    for s in structures_xy_nm:
        score = lattice_pair_score(
            s, expected_spacing_nm, band_nm=pair_score_band_nm
        )
        if min_pair_score is not None and score < min_pair_score:
            a = _empty_lattice_aux(n_nodes)
        else:
            a = lattice_defect_features(
                s,
                template,
                expected_spacing_nm,
                allow_mirror=allow_mirror,
                min_samples=min_samples,
                eps_frac=eps_frac,
                sym_perms=perms,
            )
        a["pair_score"] = float(score)
        aux.append(a)
    n = len(aux)
    labels = np.full(n, -1, dtype=int)
    on = np.zeros(n, dtype=bool)
    for i, a in enumerate(aux):
        spacing_ok = a["fitted_spacing_nm"] > 0 and (
            abs(a["fitted_spacing_nm"] - design_nn) <= spacing_tol * design_nn
        )
        on[i] = (
            a["n_matched"] >= min_sites
            and a["rmse_nm"] <= rmse_gate
            and a["frac_on_lattice"] >= frac_on_lattice_gate
            and spacing_ok
            and a["nlocs_cv"] <= max_nlocs_cv
            and a["spread_cv"] <= max_spread_cv
        )
    on_idx = [i for i in range(n) if on[i]]
    if defect_grouping == "exact":
        counts = Counter(aux[i]["occupancy"] for i in on_idx)
        key_to_label = {
            occ: lbl for lbl, (occ, _) in enumerate(counts.most_common())
        }
        for i in on_idx:
            labels[i] = key_to_label[aux[i]["occupancy"]]
    else:  # completeness: group by number of occupied sites
        for i in on_idx:
            labels[i] = int(sum(aux[i]["occupancy"]))
    summary = _summarize_lattice_clusters(labels, aux, n_nodes)
    return {
        "labels": labels,
        "aux": aux,
        "cluster_summary": summary,
        "n_nodes": n_nodes,
        "symmetry_perms": perms,
        "defect_grouping": defect_grouping,
        "min_pair_score": min_pair_score,
        # the exact thresholds every on-lattice gate decided on, so the
        # report can show them against the per-structure metric distributions
        # (see analyse._plot_lattice_gate_panels).
        "gates": {
            "min_pair_score": min_pair_score,
            "min_sites": int(min_sites),
            "rmse_gate_nm": float(rmse_gate),
            "frac_on_lattice_gate": float(frac_on_lattice_gate),
            "design_nn_nm": float(design_nn),
            "spacing_tol": float(spacing_tol),
            "max_nlocs_cv": float(max_nlocs_cv),
            "max_spread_cv": float(max_spread_cv),
        },
    }


def pick_origami(
    locs,
    info,
    template,
    pixelsize,
    candidate_method="footprint",
    footprint_diameter=None,
    pick_diameter_factor=1.5,
    missing_sites_allowed=2,
    site_uncertainty_nm=3.0,
    mean_locs_per_site=None,
    n_sim=1500,
    sim_quantile=0.01,
    random_seed=0,
    min_n_locs_per_frame=None,
    max_n_locs_per_frame=None,
    min_rmsd=None,
    max_rmsd=None,
    filter_by_geometry=False,
    spacing_tol=0.5,
    max_rmse_nm=None,
    subcluster_min_samples=3,
    allow_mirror=True,
):
    """Design-aware picking of origami structures.

    Predicts where expected origami land in ``pick_similar``'s nlocs/rmsd
    phase space by simulating realisations of the design geometry (random
    missing sites, Gaussian site jitter, Poisson kinetics), derives the
    pick window from that simulated cloud, and picks with
    :func:`pick_similar`. The window itself is the primary filter; matching
    each pick's resolved sites against the design geometry is an optional
    post-filter (``filter_by_geometry``).

    Parameters
    ----------
    locs, info : object
        Localizations and metadata (as elsewhere in this module).
    template : OrigamiTemplate
        Expected design geometry (see :func:`load_origami_template`).
    pixelsize : float
        Camera pixel size in nanometres (px <-> nm conversion).
    candidate_method : {"footprint", "cluster_of_clusters"}, optional
        Coarse candidate-detection strategy. Default ``"footprint"``.
    footprint_diameter : float, optional
        Pick diameter (camera px) spanning one origami. If not given, it is
        derived from the template extent (see ``pick_diameter_factor``).
    pick_diameter_factor : float, optional
        When ``footprint_diameter`` is not given, the pick diameter is
        ``pick_diameter_factor * template.extent_nm / pixelsize``. Default
        1.5 (150 % of the origami size).
    missing_sites_allowed : int, optional
        Maximum missing sites in a simulated realisation (and, when
        ``filter_by_geometry``, tolerated per accepted structure). Default 2.
    site_uncertainty_nm : float, optional
        Per-localization Gaussian spread around each docking site, for the
        simulation. Default 3.
    mean_locs_per_site : float, optional
        Poisson mean localizations per docking site (from kinetics). When
        given, the pick window is derived by simulation; otherwise the
        overrides / quantile defaults are used.
    n_sim : int, optional
        Number of simulated realisations. Default 1500.
    sim_quantile : float, optional
        Per-axis tail fraction dropped when turning the simulated cloud into
        the pick rectangle. Default 0.01.
    random_seed : int, optional
        Simulation seed (reproducibility). Default 0.
    min_n_locs_per_frame, max_n_locs_per_frame : float or str, optional
        Override the simulated nlocs window (per-frame value, or a quantile
        string like ``"q0.25"``). When unset, the simulated window (or, with
        no simulation, ``"q0.25"``/``"q0.98"``) is used.
    min_rmsd, max_rmsd : float, optional
        Override the simulated RMSD window (camera px). When unset, the
        simulated window (or, with no simulation, ``0``/``inf``) is used.
    filter_by_geometry : bool, optional
        If True, sub-cluster each pick into docking sites, register against
        the design and reject mismatches (and emit docking-site picks).
        Default False - the phase-space window is the only filter.
    spacing_tol : float, optional
        Relative spacing tolerance for the geometry filter. Default 0.5.
    max_rmse_nm : float, optional
        Maximum RMSE-vs-design for the geometry filter (disabled if None).
    subcluster_min_samples : int, optional
        DBSCAN ``min_samples`` for docking-site sub-clustering. Default 3.
    allow_mirror : bool, optional
        Allow a mirrored match during the geometry filter. Default True.

    Returns
    -------
    dict
        Keys include ``accepted_centers_px``, ``docking_site_centers_px``,
        ``geometry_table``, ``n_candidates`` / ``n_accepted`` /
        ``n_registered``, ``funnel``, ``footprint_diameter``, the candidate
        phase-space arrays ``candidate_nlocs`` / ``candidate_rmsds`` /
        ``candidate_labels``, ``accepted_nlocs`` / ``accepted_rmsds``, the
        simulated cloud ``sim_nlocs`` (counts) / ``sim_rmsds`` (px), and the
        resolved ``pick_window`` (bounds actually used).
    """
    n_frames = info[0]["Frames"]
    spacing_nm = template.grid_spacing_nm
    # Fall back on a template-derived footprint whenever the caller did not
    # supply a usable one (None or a 0/negative placeholder). A zero diameter
    # would divide by zero deep inside picasso's get_index_blocks. The
    # pick_diameter_factor gives a margin around the origami (default 150 %).
    if not footprint_diameter or footprint_diameter <= 0:
        footprint_diameter = (
            pick_diameter_factor * template.extent_nm / pixelsize
        )
    if not footprint_diameter or footprint_diameter <= 0:
        raise ValueError(
            "pick_origami could not derive a positive footprint_diameter: "
            f"template extent={template.extent_nm} nm, spacing={spacing_nm} "
            f"nm, pixelsize={pixelsize} nm. A single-site template has no "
            "extent - supply footprint_diameter (camera px) explicitly, or "
            "provide a multi-site geometry."
        )
    match_gate_nm = 0.5 * spacing_nm if spacing_nm > 0 else None

    # --- 1. simulate the expected nlocs/rmsd phase space -----------------
    sim_nlocs = np.empty(0)
    sim_rmsds = np.empty(0)  # camera px
    window = {}
    if mean_locs_per_site is not None and mean_locs_per_site > 0:
        sim_nlocs, sim_rmsds_nm = simulate_origami_nlocs_rmsd(
            template,
            site_uncertainty_nm,
            missing_sites_allowed,
            mean_locs_per_site,
            n_sim=n_sim,
            random_seed=random_seed,
        )
        sim_rmsds = sim_rmsds_nm / pixelsize
        window = phase_space_window_from_sim(
            sim_nlocs, sim_rmsds_nm, pixelsize, sim_quantile
        )

    # --- 2. resolve the pick_similar window ------------------------------
    # explicit overrides win; else the simulated window; else quantile/wide
    # defaults. nlocs bounds are per-frame for pick_similar.
    def _unset(v):
        return v is None or v == ""

    if not _unset(min_n_locs_per_frame):
        min_nlpf = min_n_locs_per_frame
    elif "min_nlocs" in window:
        min_nlpf = window["min_nlocs"] / n_frames
    else:
        min_nlpf = "q0.25"
    if not _unset(max_n_locs_per_frame):
        max_nlpf = max_n_locs_per_frame
    elif "max_nlocs" in window:
        max_nlpf = window["max_nlocs"] / n_frames
    else:
        max_nlpf = "q0.98"
    pick_min_rmsd = (
        min_rmsd if not _unset(min_rmsd) else window.get("min_rmsd", 0.0)
    )
    pick_max_rmsd = (
        max_rmsd if not _unset(max_rmsd) else window.get("max_rmsd", np.inf)
    )

    # --- 3. pick candidates within the window ---------------------------
    if candidate_method == "footprint":
        picks, nlocs, rmsds, labels = pick_similar(
            locs,
            info,
            diameter=footprint_diameter,
            min_n_locs_per_frame=min_nlpf,
            max_n_locs_per_frame=max_nlpf,
            min_rmsd=pick_min_rmsd,
            max_rmsd=pick_max_rmsd,
        )
        candidates = list(picks)
        sel = np.asarray(labels) == 0
        if int(np.count_nonzero(sel)) == len(candidates):
            cand_point_nlocs = np.asarray(nlocs)[sel]
            cand_point_rmsds = np.asarray(rmsds)[sel]
        else:
            cand_point_nlocs = np.full(len(candidates), np.nan)
            cand_point_rmsds = np.full(len(candidates), np.nan)
    elif candidate_method == "cluster_of_clusters":
        site_diameter = max(spacing_nm / pixelsize, 1e-6)
        min_n_locs = (
            1 if isinstance(min_nlpf, str) else int(n_frames * min_nlpf)
        )
        candidates, nlocs, rmsds, labels = (
            _candidate_centers_cluster_of_clusters(
                locs,
                info,
                site_diameter,
                footprint_diameter,
                max(1, min_n_locs),
                subcluster_min_samples,
            )
        )
        cand_point_nlocs = np.full(len(candidates), np.nan)
        cand_point_rmsds = np.full(len(candidates), np.nan)
    else:
        raise ValueError(
            f"unknown candidate_method: {candidate_method!r} "
            "(expected 'footprint' or 'cluster_of_clusters')"
        )

    # The pick_similar active range (nlocs/rmsd rectangle), resolved exactly
    # as pick_similar applies it - quantile strings ("q0.25") resolve against
    # the candidate nlocs, plain values scale by the frame count - so it can
    # be drawn on the phase diagrams. Total nlocs (analyse divides by frames).
    def _resolve_nlocs(v):
        if isinstance(v, str) and v[:1] == "q":
            return (
                float(np.quantile(nlocs, float(v[1:])))
                if len(nlocs)
                else (np.nan)
            )
        return float(n_frames) * float(v)

    if candidate_method == "footprint":
        pick_window = {
            "min_nlocs": _resolve_nlocs(min_nlpf),
            "max_nlocs": _resolve_nlocs(max_nlpf),
            "min_rmsd": (
                float(pick_min_rmsd)
                if not isinstance(pick_min_rmsd, str)
                else np.nan
            ),
            "max_rmsd": (
                float(pick_max_rmsd)
                if not isinstance(pick_max_rmsd, str)
                and np.isfinite(pick_max_rmsd)
                else np.nan
            ),
        }
    else:
        pick_window = {
            "min_nlocs": np.nan,
            "max_nlocs": np.nan,
            "min_rmsd": np.nan,
            "max_rmsd": np.nan,
        }

    funnel = {
        "n_candidates": len(candidates),
        "no_locs": 0,
        "too_few_sites": 0,
        "rejected_missing_sites": 0,
        "rejected_rmse": 0,
        "rejected_spacing": 0,
        "accepted": 0,
    }
    base_return = {
        "n_candidates": len(candidates),
        "candidate_nlocs": nlocs,
        "candidate_rmsds": rmsds,
        "candidate_labels": labels,
        "sim_nlocs": sim_nlocs,
        "sim_rmsds": sim_rmsds,
        "pick_window": pick_window,
        "footprint_diameter": footprint_diameter,
    }

    # Diagnostic: does the pick window bracket the candidate cloud? If not
    # (e.g. the simulated RMSD band sits below the real localization spread),
    # nothing is picked - this line makes the mismatch visible in the log.
    all_nlocs = np.asarray(nlocs)
    all_rmsds = np.asarray(rmsds)
    if len(all_nlocs):
        nl5, nl50, nl95 = np.percentile(all_nlocs, [5, 50, 95])
        rm5, rm50, rm95 = np.percentile(all_rmsds, [5, 50, 95])
        logger.debug(
            "pick_origami window: nlocs [%.0f, %.0f] total, rmsd [%.3f, %.3f] "
            "px (footprint_diameter %.3f px). Candidate nlocs p5/50/95 = "
            "%.0f/%.0f/%.0f, rmsd p5/50/95 = %.3f/%.3f/%.3f. %d candidates in "
            "window."
            % (
                pick_window["min_nlocs"],
                pick_window["max_nlocs"],
                pick_window["min_rmsd"],
                pick_window["max_rmsd"],
                footprint_diameter,
                nl5,
                nl50,
                nl95,
                rm5,
                rm50,
                rm95,
                len(candidates),
            )
        )
    else:
        logger.debug(
            "pick_origami: pick_similar found no footprint regions at all "
            "(footprint_diameter %.3f px) - check the geometry/pixelsize."
            % footprint_diameter
        )

    accepted_centers_px = []
    accepted_nlocs = []
    accepted_rmsds = []
    docking_site_centers_px = []
    geometry_table = []
    n_registered = 0

    if not candidates:
        logger.debug("pick_origami: no candidates within the pick window.")
        return {
            **base_return,
            "accepted_centers_px": [],
            "docking_site_centers_px": [],
            "geometry_table": [],
            "n_registered": 0,
            "n_accepted": 0,
            "funnel": funnel,
            "accepted_nlocs": np.empty(0),
            "accepted_rmsds": np.empty(0),
        }

    # --- 4a. no geometry filter: the window is the filter ---------------
    if not filter_by_geometry:
        for cand_idx, (cx, cy) in enumerate(candidates):
            accepted_centers_px.append((float(cx), float(cy)))
            accepted_nlocs.append(cand_point_nlocs[cand_idx])
            accepted_rmsds.append(cand_point_rmsds[cand_idx])
            geometry_table.append(
                {
                    "center_x_px": float(cx),
                    "center_y_px": float(cy),
                    "nlocs": float(cand_point_nlocs[cand_idx]),
                    "rejection_reason": "",
                    "accepted": True,
                }
            )
        funnel["accepted"] = len(accepted_centers_px)
        logger.debug(
            f"pick_origami: {funnel['n_candidates']} candidates accepted "
            "within the pick window (no geometry filter)."
        )
        return {
            **base_return,
            "accepted_centers_px": accepted_centers_px,
            "docking_site_centers_px": [],
            "geometry_table": geometry_table,
            "n_registered": 0,
            "n_accepted": len(accepted_centers_px),
            "funnel": funnel,
            "accepted_nlocs": np.asarray(accepted_nlocs),
            "accepted_rmsds": np.asarray(accepted_rmsds),
        }

    # --- 4b. geometry filter: register each pick against the design -----
    min_required_sites = max(
        1, template.n_sites_expected - missing_sites_allowed
    )
    all_foot_locs = picked_locs(
        locs,
        info,
        candidates,
        pick_diameter=footprint_diameter,
        add_group=True,
    )
    foot_groups = {int(g): sub for g, sub in all_foot_locs.groupby("group")}

    for cand_idx, (cx, cy) in enumerate(candidates):
        foot_locs = foot_groups.get(cand_idx)
        if foot_locs is None or len(foot_locs) == 0:
            funnel["no_locs"] += 1
            continue
        xy_nm = (
            np.column_stack(
                [np.asarray(foot_locs["x"]), np.asarray(foot_locs["y"])]
            )
            * pixelsize
        )
        site_centers_nm = subcluster_docking_sites(
            xy_nm, spacing_nm, min_samples=subcluster_min_samples
        )
        # cheap pre-filter: too few resolved sites -> cannot be accepted,
        # so don't pay for the full registration sweep.
        if len(site_centers_nm) < min_required_sites:
            funnel["too_few_sites"] += 1
            continue
        n_registered += 1
        reg = register_to_template(
            site_centers_nm,
            template.sites_nm,
            allow_mirror=allow_mirror,
            match_gate_nm=match_gate_nm,
        )
        verdict = classify_candidate(
            reg,
            template.n_sites_expected,
            missing_sites_allowed=missing_sites_allowed,
            spacing_tol=spacing_tol,
            grid_spacing_nm=spacing_nm,
            max_rmse_nm=max_rmse_nm,
        )
        accepted = verdict == "accepted"
        funnel[
            {
                "accepted": "accepted",
                "missing_sites": "rejected_missing_sites",
                "rmse": "rejected_rmse",
                "spacing": "rejected_spacing",
            }[verdict]
        ] += 1
        # Centre the structure on the centre of mass of its resolved
        # docking sites, not the coarse pick_similar seed (which can sit
        # off-centre), so the origami is centred in its pick.
        matched_obs_nm = site_centers_nm[reg["matched_observed_indices"]]
        if len(matched_obs_nm):
            com_px = matched_obs_nm.mean(axis=0) / pixelsize
            center_x, center_y = float(com_px[0]), float(com_px[1])
        else:
            center_x, center_y = float(cx), float(cy)
        geometry_table.append(
            {
                "center_x_px": center_x,
                "center_y_px": center_y,
                "n_resolved_sites": int(reg["n_resolved"]),
                "n_missing_sites": int(
                    template.n_sites_expected - reg["n_resolved"]
                ),
                "mean_spacing_nm": float(reg["mean_spacing_nm"]),
                "rmse_nm": float(reg["rmse_nm"]),
                "orientation_deg": float(reg["orientation_deg"]),
                "mirror": bool(reg["mirror"]),
                "nlocs": float(cand_point_nlocs[cand_idx]),
                "rejection_reason": "" if accepted else verdict,
                "accepted": bool(accepted),
            }
        )
        if accepted:
            accepted_centers_px.append((center_x, center_y))
            accepted_nlocs.append(cand_point_nlocs[cand_idx])
            accepted_rmsds.append(cand_point_rmsds[cand_idx])
            for px in matched_obs_nm / pixelsize:
                docking_site_centers_px.append((float(px[0]), float(px[1])))

    logger.debug(
        f"pick_origami rejection funnel: {funnel['n_candidates']} candidates "
        f"-> {funnel['no_locs']} empty, {funnel['too_few_sites']} too few "
        f"sites, {funnel['rejected_missing_sites']} missing-sites, "
        f"{funnel['rejected_rmse']} rmse, {funnel['rejected_spacing']} "
        f"spacing -> {funnel['accepted']} accepted."
    )
    return {
        **base_return,
        "accepted_centers_px": accepted_centers_px,
        "docking_site_centers_px": docking_site_centers_px,
        "geometry_table": geometry_table,
        "n_registered": n_registered,
        "n_accepted": len(accepted_centers_px),
        "funnel": funnel,
        "accepted_nlocs": np.asarray(accepted_nlocs),
        "accepted_rmsds": np.asarray(accepted_rmsds),
    }


########################################################################
# End Design-aware origami picking
########################################################################


#####


def plot_1dhist(locs, field, fig, ax):
    # np.asarray: lib.calculate_optimal_bins (picasso >= 0.10) samples via
    # positional integer indexing (data[rng.choice(...)]), which fails on a
    # pandas Series with a non-RangeIndex. Force a numpy array.
    data = np.asarray(locs[field])
    data = data[np.isfinite(data)]
    bins = lib.calculate_optimal_bins(data, 1000)
    # Prepare the figure
    fig.suptitle(field)
    ax.hist(data, bins, rwidth=1, linewidth=0)
    data_range = np.ptp(data)
    ax.set_xlim([bins[0] - 0.05 * data_range, data.max() + 0.05 * data_range])


def plot_2dhist(locs, field_x, field_y, fig, ax):
    # np.asarray: see plot_1dhist - lib.calculate_optimal_bins samples by
    # positional index, which breaks on a pandas Series with a non-default
    # index. Force numpy arrays.
    x = np.asarray(locs[field_x])
    y = np.asarray(locs[field_y])
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

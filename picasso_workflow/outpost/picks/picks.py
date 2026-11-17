"""Pick / fiducial processing (extracted from picasso_outpost).

See the package docstring.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import OPTICS

from picasso_workflow._picasso_api import postprocess, lib


def pick_gold(locs, info, diameter=2, std_range=1.4, mean_rmsd=50.0):
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
        The pick-similar mean RMSD (in nm) identifying gold. picasso 1.0
        localizations are in nm, so ``rmsd_at_com`` returns nm; the default
        50.0 nm replaces the pre-1.0 pixel-scale value (0.4 px). Default is
        50.0.

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
    # RMSD spread must scale with the RMSD, not the loc count: the previous
    # ``0.25 * mean_n_locs`` made the [min_rmsd, max_rmsd] window enormous, so
    # the RMSD criterion was a no-op.
    std_rmsd = 0.25 * mean_rmsd
    min_n_locs = mean_n_locs - std_range * std_n_locs
    max_n_locs = mean_n_locs + std_range * std_n_locs
    min_rmsd = mean_rmsd - std_range * std_rmsd
    max_rmsd = mean_rmsd + std_range * std_rmsd

    # picasso 1.0: localizations are in nm while ``diameter`` is a pick size
    # in camera pixels, so convert to the nm diameter ``pick_size`` expects.
    # ``pick_similar`` with ``n_locs_range`` / ``rmsd_range`` (and no seed
    # picks) scans the FOV for circular picks whose loc count and RMSD-at-COM
    # fall inside the windows (inclusive bounds), replacing the former
    # private ``_pick_similar`` grid scan. It returns the pick centres as a
    # list of ``(x, y)`` tuples.
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    similar = postprocess.pick_similar(
        locs,
        info,
        pick_size=diameter * pixelsize,
        n_locs_range=(min_n_locs, max_n_locs),
        rmsd_range=(min_rmsd, max_rmsd),
    )
    # ``pick_similar`` already returns a list of ``(x, y)`` tuples.
    return [(x, y) for x, y in similar]


def index_locs(locs, info, pick_diameter):
    """
    Indexes localizations from a given channel in a grid with grid
    size equal to the pick radius.
    """
    # picasso 1.0 indexes in nm (0.11 used camera pixels); the grid size was
    # the pick radius in camera pixels.
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    size = (pick_diameter / 2) * pixelsize
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
    # picasso 1.0: picks / pick_shape / pick_size are keyword-only, pick_size
    # is the circle DIAMETER in nm (0.11 took the radius in camera pixels).
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    per_pick = postprocess.picked_locs(
        locs,
        info,
        picks=_centers,
        pick_shape="Circle",
        pick_size=pick_diameter * pixelsize,
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
    """Per-candidate centre, RMSD and loc count across the FOV.

    Scans the field of view for circular candidate picks (as
    ``pick_similar`` does, but without filtering by loc count / RMSD) and
    returns their centres together with the RMSD-at-COM and localization
    count of each, for selecting structures in the (RMSD, n_locs) plane.

    Delegates to :func:`picasso.postprocess.pick_similar_candidates`
    (picasso 1.0's public replacement for the former private grid scan);
    ``diameter`` is a pick size in camera pixels, converted to the nm
    ``pick_size`` that function expects.

    Usage:
    x_similar, y_similar, rmsds, nlocs = get_pick_similar_vals(
        locs, info, diameter=1.5, min_n_locs=100)
    plt.hexbin(rmsds, nlocs)
    plt.show()
    """
    pixelsize = lib.get_from_metadata(info, "Pixelsize", raise_error=True)
    candidates = postprocess.pick_similar_candidates(
        locs,
        info,
        pick_size=diameter * pixelsize,
        min_n_locs=int(min_n_locs),
    )
    x_similar = candidates["x"].to_numpy()
    y_similar = candidates["y"].to_numpy()
    rmsds = candidates["rmsd"].to_numpy()
    nlocs = candidates["n_locs"].to_numpy()
    return x_similar, y_similar, rmsds, nlocs

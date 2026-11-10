"""Correspondence-free rigid registration (extracted from picasso_outpost).

See the package docstring. `promote` candidate #3 -- pure numpy/scipy/sklearn.
"""

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from picasso_workflow import _picasso_api

# ``get_pick_similar_vals`` is a picks helper (``outpost/picks/``);
# ``_candidate_centers_cluster_of_clusters`` scores candidate clusters with it.
# One-way import -- picks does not import this package back.
from picasso_workflow.outpost.picks.picks import get_pick_similar_vals

# Default DBSCAN neighbourhood for resolving docking sites, as a fraction of
# the design site spacing. Must be well below the inter-site gap or DBSCAN
# *chains* adjacent sites through their localization tails and under-counts
# sites (worse the brighter the structure). A fifth of the spacing groups a
# site's own localizations while keeping neighbours apart. Tunable per run via
# pick_origami's ``pattern_eps_frac``.
_SITE_EPS_FRAC = 0.2


def _median_nn_spacing(sites):
    """Median nearest-neighbour distance of a point set."""
    sites = np.asarray(sites, dtype=float).reshape(-1, 2)
    if len(sites) < 2:
        return 0.0
    tree = KDTree(sites)
    dist, _ = tree.query(sites, k=2)
    return float(np.median(dist[:, 1]))


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


def register_to_template(
    observed_nm,
    template_nm,
    allow_mirror=True,
    match_gate_nm=None,
    angle_step_deg=2.0,
    n_refine=4,
):
    """Register observed docking sites onto a design template.

    Thin adapter over :func:`picasso.registration.register_to_template`
    (picassosr >= 1.0), reached through the ``_picasso_api`` boundary. The
    picasso function finds the best rigid alignment (rotation + translation,
    optional mirror) of ``observed_nm`` onto ``template_nm`` under an unknown
    correspondence, tolerating missing/extra observed sites, and returns a
    :class:`picasso.transforms.RigidTransform`. This wrapper preserves the
    dict contract picasso-workflow's origami callers expect (nanometre
    coordinates throughout; picasso 1.0 is nm-based).

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
        Number of best coarse seeds to ICP-refine. Default 4.

    Returns
    -------
    dict
        Registration result with keys ``n_resolved``,
        ``matched_template_indices``, ``matched_observed_indices``,
        ``rmse_nm``, ``mean_spacing_nm``, ``orientation_deg``, ``mirror``,
        ``R`` and ``t``.
    """
    registration = _picasso_api.registration
    observed = np.asarray(observed_nm, dtype=float).reshape(-1, 2)
    template = np.asarray(template_nm, dtype=float).reshape(-1, 2)

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

    # picasso needs >= 2 observed and >= 2 template points; below that the
    # fork returned an all-empty result rather than raising.
    if len(observed) < 2 or len(template) < 2:
        return _empty_result()

    result = registration.register_to_template(
        observed,
        template,
        mirror=allow_mirror,
        gate=match_gate_nm,
        angle_step=angle_step_deg,
        n_seeds=n_refine,
    )
    transform = result["transform"]
    if transform is None or result["n_matched"] == 0:
        return _empty_result()

    rows = np.asarray(result["point_indices"], dtype=int)
    cols = np.asarray(result["template_indices"], dtype=int)
    matched_obs = observed[rows]
    return {
        "n_resolved": int(result["n_matched"]),
        "matched_template_indices": cols,
        "matched_observed_indices": rows,
        "rmse_nm": float(result["rmse"]),
        "mean_spacing_nm": _median_nn_spacing(matched_obs),
        "orientation_deg": float(transform.rotation_deg),
        "mirror": bool(transform.mirror),
        "R": transform.linear,
        "t": transform.shift,
    }


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

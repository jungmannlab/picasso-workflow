"""Compute the registry metric vector from a FLAT picasso ``_locs.hdf5``.

WP-7 fact: **most quality metrics come straight from the hdf5** — the A1
primaries ``nena`` and localization precision, plus photons / background /
PSF σ / ``n_locs`` / density. Only kinetics / drift / clustering need a
processing pass, which is out of scope for this MVP; those keys are simply left
absent so the registry stores them as NULL (the service only writes columns it
is given — missing features stay NULL).

Every picasso call goes through the declared boundary
``picasso_workflow._picasso_api`` (NeNA via ``postprocess.nena``, loading via
``io.load_locs``) — never ``from picasso import ...`` (an AST test enforces
this). NeNA math is NOT reimplemented here; it is picasso's.

The returned dict's keys are already the registry ``Metrics`` column names
(see :mod:`picasso_registry.schemas`), so the ingest mapper can pass them
straight through. Unknown/extra keys would ride into ``Metrics.extra``; here we
deliberately emit only typed columns.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from picasso_workflow._picasso_api import postprocess

logger = logging.getLogger(__name__)

#: Metrics that require a processing pass (kinetics / drift / clustering) or the
#: raw movie — OUT of scope for the MVP backfill. Listed so the scope is
#: explicit and so a future pass can fill them without reshaping the ingest.
DEFERRED_METRICS = (
    "drift_nm",
    "drift_residual_nm",
    "drift_rate_nm_min",
    "dark_time_s",
    "bright_time_s",
    "k_on",
    "k_off",
    "binding_freq_hz",
    "events_per_site",
    "duty_cycle",
    "qpaint_count",
    "n_clusters",
    "mean_cluster_size",
    "nnd_median_nm",
)


def _pixelsize_nm(info: Sequence[Mapping[str, Any]]) -> float | None:
    """Pixel size (nm) from any info block that carries ``Pixelsize``.

    picasso info is a list of dicts (one per processing step); the camera /
    localize step carries ``Pixelsize``. Returns None when absent — then the
    nm-scaled metrics (NeNA-nm, density) can't be computed and stay NULL.
    """
    for part in info or ():
        px = part.get("Pixelsize")
        if px is not None:
            try:
                return float(px)
            except (TypeError, ValueError):
                return None
    return None


def _fov_area_um2(
    info: Sequence[Mapping[str, Any]], pixelsize_nm: float | None
) -> float | None:
    """FOV area in µm² from Width×Height (px) × pixelsize, for density.

    None when either the frame extent or the pixel size is missing — density
    then stays NULL rather than being guessed.
    """
    if not pixelsize_nm:
        return None
    width = height = None
    for part in info or ():
        if width is None:
            width = part.get("Width")
        if height is None:
            height = part.get("Height")
    if width is None or height is None:
        return None
    try:
        area_nm2 = float(width) * float(height) * pixelsize_nm * pixelsize_nm
    except (TypeError, ValueError):
        return None
    return area_nm2 / 1e6  # nm² → µm²


def _col(locs: Any, name: str) -> np.ndarray | None:
    """Column ``name`` from a DataFrame or structured array, else None."""
    try:
        if hasattr(locs, "columns"):  # pandas DataFrame
            if name in locs.columns:
                return np.asarray(locs[name].to_numpy())
            return None
        names = getattr(getattr(locs, "dtype", None), "names", None)
        if names and name in names:
            return np.asarray(locs[name])
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _median(locs: Any, name: str) -> float | None:
    arr = _col(locs, name)
    if arr is None or len(arr) == 0:
        return None
    val = float(np.nanmedian(arr))
    return val if np.isfinite(val) else None


def compute_nena_nm(
    locs: Any, info: Sequence[Mapping[str, Any]]
) -> float | None:
    """Whole-FOV NeNA in nm via picasso ``postprocess.nena`` (the boundary).

    ``postprocess.nena`` returns ``(result, best_val)`` with ``best_val`` the
    NeNA in **pixels**; we scale by the ``.yaml`` pixel size to nm (the same
    convention as ``analyse.AnalysisDescribe``). Any fit/compute failure — too
    few locs, a fitting error — is logged and yields None, so the metric stays
    NULL instead of aborting the ingest. Not reimplemented here: it is picasso.
    """
    pixelsize = _pixelsize_nm(info)
    if pixelsize is None:
        logger.warning("no Pixelsize in info; NeNA stays NULL")
        return None
    try:
        _result, best_val_px = postprocess.nena(locs, info)
    except Exception as exc:  # fitting error, too few locs, etc.
        logger.warning("NeNA computation failed, leaving NULL: %s", exc)
        return None
    try:
        nm = float(best_val_px) * pixelsize
    except (TypeError, ValueError):
        return None
    # A fit can converge to a non-finite best_val WITHOUT raising; NaN/inf must
    # stay NULL (unlike _median, nena has no built-in finite guard), else it
    # rides through compute_metrics' filter and POSTs an invalid-JSON NaN.
    return nm if np.isfinite(nm) else None


def _n_locs(locs: Any) -> int | None:
    try:
        return int(len(locs))
    except TypeError:  # pragma: no cover - defensive
        return None


def compute_metrics(
    locs: Any,
    info: Sequence[Mapping[str, Any]],
    *,
    with_nena: bool = True,
) -> dict[str, Any]:
    """Metric vector (registry ``Metrics`` columns) from loaded locs + info.

    Primaries straight from the hdf5 columns:

    * ``n_locs``           — row count
    * ``lp_x_nm`` / ``lp_y_nm`` — median ``lpx`` / ``lpy`` × pixel size
    * ``photons_median``   — median ``photons``
    * ``background``       — median ``bg``
    * ``psf_sigma_x`` / ``psf_sigma_y`` — median ``sx`` / ``sy``
    * ``net_gradient_median`` — median ``net_gradient`` (when present)
    * ``density_locs_um2``  — ``n_locs`` / FOV area (µm²) from the ``.yaml``

    Computed (through the boundary): ``nena_nm`` via ``postprocess.nena``
    (skippable with ``with_nena=False`` for a movie-free fast path / tests).

    Kinetics / drift / clustering metrics are OUT of scope (see
    :data:`DEFERRED_METRICS`) and are simply NOT emitted, so the registry
    stores them as NULL. Any per-metric compute that can't be done (missing
    column, missing pixel size) is likewise omitted rather than guessed —
    missing features stay absent/NULL. Only finite values are returned.
    """
    pixelsize = _pixelsize_nm(info)
    out: dict[str, Any] = {}

    n_locs = _n_locs(locs)
    if n_locs is not None:
        out["n_locs"] = n_locs

    out["photons_median"] = _median(locs, "photons")
    out["background"] = _median(locs, "bg")
    out["psf_sigma_x"] = _median(locs, "sx")
    out["psf_sigma_y"] = _median(locs, "sy")
    out["net_gradient_median"] = _median(locs, "net_gradient")

    # Localization precision: median lpx/lpy (px) scaled to nm by pixel size.
    if pixelsize is not None:
        lp_x_px = _median(locs, "lpx")
        lp_y_px = _median(locs, "lpy")
        out["lp_x_nm"] = lp_x_px * pixelsize if lp_x_px is not None else None
        out["lp_y_nm"] = lp_y_px * pixelsize if lp_y_px is not None else None

    # Density: locs per µm² over the FOV extent from the .yaml. Requires
    # n_locs > 0 — an empty (zero-loc) dataset is not a real density sample and
    # a 0.0 would otherwise drag a cohort's learned band (ranges.py).
    area_um2 = _fov_area_um2(info, pixelsize)
    if area_um2 and area_um2 > 0 and n_locs:
        out["density_locs_um2"] = n_locs / area_um2

    if with_nena:
        out["nena_nm"] = compute_nena_nm(locs, info)

    # Drop keys we couldn't compute: a missing feature must stay NULL (absent),
    # not be POSTed as an explicit null that pads every row. Also drop any
    # non-finite float (NaN/inf) so the invariant "only finite values are
    # returned" holds at THIS boundary, not just per-source in _median.
    def _emit(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, float) and not np.isfinite(v):
            return False
        return True

    return {k: v for k, v in out.items() if _emit(v)}


def locs_signature(locs: Any) -> str:
    """A cheap, stable content signature of a locs table, for dataset identity.

    ``n_locs`` + the nan-summed ``x`` / ``y`` columns, hashed. Folded into the
    backfill ``run_id`` so two DISTINCT datasets that happen to share a filename
    and frame count do NOT collide to one id (and get silently deduped) — while
    an exact copy of the same dataset still hashes identically and dedupes. One
    pass over two columns; no full-file hash.
    """
    import hashlib

    n = _n_locs(locs) or 0

    def _sum(col: str) -> float:
        arr = _col(locs, col)
        if arr is None or len(arr) == 0:
            return 0.0
        val = float(np.nansum(arr))
        return val if np.isfinite(val) else 0.0

    raw = f"{n}|{_sum('x'):.6f}|{_sum('y'):.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

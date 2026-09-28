#!/usr/bin/env python3
"""
filter_advisor.py -- data-driven Picasso filter-settings advisor, a min-net-
gradient estimator, and a clustering guide.

These turn the raw localization/candidate distributions into concrete Picasso
parameters (filter thresholds, detection cut, DBSCAN eps/min_samples) that get
written into ``qc.json`` as a parameter-recommendation input, plus a
kept-vs-removed preview payload for the GUI.

Ported from LiveLocalization V0.8 (``filter_suggestions`` / ``filter_keep_mask``
/ ``estimate_min_net_gradient`` from the GUI module, ``suggest_clustering`` from
``liveloc_tools``). The pure-numpy paths import headlessly; ``estimate_min_net_
gradient`` imports ``picasso.localize`` lazily so this module stays importable
without picasso.

The advisor writes into a ``qc.json`` sidecar via :func:`write_filter_suggestions
_to_qc` / :func:`write_clustering_to_qc`, matching the V0.8 qc.json layout
(a ``recommendations`` block keyed by ``filter`` / ``clustering``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

# ── small numeric helpers (ported from V0.8) ──────────────────────────────────


def _pct_safe(a: Any, q: float) -> Optional[float]:
    try:
        return float(np.percentile(np.asarray(a, float), q))
    except (TypeError, ValueError):
        return None


def _otsu_threshold(vals: Any, bins: int = 128) -> Optional[float]:
    """Otsu's threshold on 1-D values (used on log10(net_gradient) to split the
    noise population from the signal). Returns the threshold value or None."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return None
    hist, edges = np.histogram(vals, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    m_total = csum[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = csum / w0
        mu1 = (m_total - csum) / w1
        var_between = w0 * w1 * (mu0 - mu1) ** 2
    var_between = np.nan_to_num(var_between, nan=0.0)
    idx = int(np.argmax(var_between[:-1])) if var_between.size > 1 else 0
    return float(centers[idx])


# ── filter settings advisor ───────────────────────────────────────────────────


def filter_keep_mask(
    s: Any, names: Sequence[str], fs: Dict[str, Any], px: float
) -> np.ndarray:
    """Boolean mask of localizations ``s`` that PASS the suggested filter ``fs``.

    ``s`` is a structured array (picasso locs recarray) with columns ``names``;
    ``fs`` is a filter dict as returned by :func:`filter_suggestions`.
    """
    keep = np.ones(len(s), dtype=bool)
    if "photons" in names:
        ph = s["photons"].astype(float)
        if fs.get("photons_min") is not None:
            keep &= ph >= float(fs["photons_min"])
        if fs.get("photons_max") is not None:
            keep &= ph <= float(fs["photons_max"])
    for col in ("sx", "sy"):
        rng = fs.get(f"{col}_range_px")
        if rng and col in names:
            v = s[col].astype(float)
            keep &= (v >= rng[0]) & (v <= rng[1])
    if fs.get("ellipticity_max") is not None and {"sx", "sy"} <= set(names):
        _sx = np.maximum(s["sx"].astype(float), 1e-6)
        _sy = np.maximum(s["sy"].astype(float), 1e-6)
        ell = 1.0 - np.minimum(_sx, _sy) / np.maximum(_sx, _sy)
        keep &= ell <= float(fs["ellipticity_max"])
    if fs.get("precision_max_nm") is not None and {"lpx", "lpy"} <= set(names):
        lp = 0.5 * (s["lpx"].astype(float) + s["lpy"].astype(float)) * px
        keep &= lp <= float(fs["precision_max_nm"])
    return keep


def filter_suggestions(
    locs: Any,
    names: Sequence[str],
    picks_xy: Optional[Sequence] = None,
    pick_r: float = 5.0,
    vp_zoom: Optional[Sequence] = None,
    vp_bg: Optional[Sequence] = None,
    px: float = 130.0,
) -> Optional[Dict[str, Any]]:
    """Data-driven Picasso filter thresholds separating real signal from noise /
    nonspecific events. Signal = localizations inside the picks (marked real
    events) when present, else the zoom ROI; the photon floor uses the background
    ROI. Returns a dict (see keys below), ``{'error': ...}``, or ``None``.

    Rationale: each PSF-fit column has a tight distribution for true single
    molecules; mis-fits, aggregates and background sit in the tails. Keeping the
    central 1-99% of the SIGNAL distribution removes those tails without biasing
    the structure. precision is tied to the achieved localization precision,
    ellipticity to PSF symmetry (2D). Ported from V0.8.
    """
    if locs is None or len(locs) == 0 or not ({"x", "y"} <= set(names)):
        return None
    # ── signal mask ────────────────────────────────────────────────────────
    src = ""
    sig_mask = None
    if picks_xy is not None and len(picks_xy):
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.asarray(picks_xy, float))
            xy = np.column_stack(
                [locs["x"].astype(float), locs["y"].astype(float)]
            )
            dist, _ = tree.query(xy, k=1)
            sig_mask = dist <= pick_r
            src = f"{len(picks_xy)} picks"
        except Exception:
            sig_mask = None
    if sig_mask is None:
        if vp_zoom is None:
            return None
        (y0, x0), (y1, x1) = vp_zoom
        sig_mask = (
            (locs["x"] >= x0)
            & (locs["x"] < x1)
            & (locs["y"] >= y0)
            & (locs["y"] < y1)
        )
        src = "zoom ROI"
    sig = locs[sig_mask]
    if len(sig) < 50:
        return {
            "error": f"too few signal localizations ({len(sig)}) — place the "
            "zoom/picks on a real structure."
        }
    # ── background locs (photon floor) ───────────────────────────────────────
    bg_locs = None
    if vp_bg is not None:
        (gy0, gx0), (gy1, gx1) = vp_bg
        bgm = (
            (locs["x"] >= gx0)
            & (locs["x"] < gx1)
            & (locs["y"] >= gy0)
            & (locs["y"] < gy1)
        )
        bg_locs = locs[bgm]
    out: Dict[str, Any] = {
        "source": src,
        "n_signal": int(len(sig)),
        "n_background": int(len(bg_locs)) if bg_locs is not None else 0,
        "pixelsize_nm": px,
        "rule": "keep central 1-99% of signal; ellipticity <= min(0.6, p95); "
        "precision <= p95; photons floor from the background only if it is "
        "clearly dimmer than the signal (else signal p1), capped at signal p10",
    }
    if "photons" in names:
        p1 = _pct_safe(sig["photons"], 1)
        p10 = _pct_safe(sig["photons"], 10)
        p50 = _pct_safe(sig["photons"], 50)
        photons_min = p1
        out["photons_floor_source"] = "signal"
        if bg_locs is not None and len(bg_locs) >= 20:
            bg95 = _pct_safe(bg_locs["photons"], 95)
            # Only trust the background photon floor if the background is clearly
            # DIMMER than the signal (its bright end below the signal median).
            if bg95 is not None and p50 is not None and bg95 < p50:
                cap = p10 if p10 is not None else (p1 or bg95)
                photons_min = max(p1 or 0, min(bg95, cap))
                out["photons_floor_source"] = "background"
            else:
                out["photons_floor_source"] = (
                    "signal (background not separable)"
                )
        if photons_min is not None:
            out["photons_min"] = round(photons_min)
        # Upper photon bound: real single-molecule events sit below ~p99.5 of the
        # signal; anything far brighter is a gold fiducial or aggregate -> cut.
        p995 = _pct_safe(sig["photons"], 99.5)
        if p995 is not None:
            out["photons_max"] = round(p995)
        out["photons_signal_median"] = round(
            _pct_safe(sig["photons"], 50) or 0
        )
    for col in ("sx", "sy"):
        if col in names:
            lo, hi = _pct_safe(sig[col], 1), _pct_safe(sig[col], 99)
            if lo is not None and hi is not None:
                out[f"{col}_range_px"] = [round(lo, 3), round(hi, 3)]
    if {"sx", "sy"} <= set(names):
        _sx = np.maximum(sig["sx"].astype(float), 1e-6)
        _sy = np.maximum(sig["sy"].astype(float), 1e-6)
        ell = 1.0 - np.minimum(_sx, _sy) / np.maximum(_sx, _sy)
        p95 = _pct_safe(ell, 95)
        if p95 is not None:
            out["ellipticity_max"] = round(min(0.6, p95), 2)
    if {"lpx", "lpy"} <= set(names):
        lp = 0.5 * (sig["lpx"].astype(float) + sig["lpy"].astype(float)) * px
        p95 = _pct_safe(lp, 95)
        if p95 is not None:
            out["precision_max_nm"] = round(p95, 1)
    return out


def filter_preview_counts(
    locs: Any, names: Sequence[str], fs: Dict[str, Any], px: float
) -> Dict[str, int]:
    """Kept-vs-removed counts for the GUI's filter preview -- the headless,
    render-free core of V0.8's ``filter_preview_rgba``. Returns a payload the
    Advisor tab consumes: ``{n_total, n_kept, n_removed}``."""
    if locs is None or len(locs) == 0:
        return {"n_total": 0, "n_kept": 0, "n_removed": 0}
    keep = filter_keep_mask(locs, names, fs, px)
    n_total = int(len(locs))
    n_kept = int(np.count_nonzero(keep))
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_removed": n_total - n_kept,
    }


# ── min-net-gradient estimator ────────────────────────────────────────────────


def estimate_min_net_gradient(
    frames: Any,
    box_size: int,
    camera_info: Optional[Dict[str, Any]] = None,
    low_frac: float = 0.1,
    low_floor: int = 100,
    min_candidates: int = 150,
) -> Dict[str, Any]:
    """Data-driven min-net-gradient suggestion. Runs Picasso ``identify`` at a
    LOW threshold on a small frame sample so the NOISE population (normally
    hidden below the current threshold) is visible, then splits noise vs signal
    with Otsu on log10(net_gradient) and picks either the bimodal valley or the
    noise-peak shoulder. Returns a dict with the recommendation and the (log)
    histogram for display, or ``{'error': ...}``.

    ``picasso.localize`` is imported lazily so the advisor package imports
    without picasso. Ported from V0.8.
    """
    try:
        from picasso import localize
    except Exception as e:  # pragma: no cover - env-dependent
        return {"error": f"picasso.localize unavailable: {e}"}
    try:
        if frames is None or len(frames) == 0:
            return {
                "error": "no frames available (start/continue a run first)."
            }
        low = max(int(low_floor), 1)
        ids = localize.identify(
            np.ascontiguousarray(frames), low, int(box_size), threaded=True
        )
        if isinstance(ids, tuple):
            ids = ids[0]
        if ids is None or len(ids) == 0:
            return {"error": "no spots detected even at the low threshold."}
        try:
            ng = np.asarray(ids["net_gradient"], float)
        except Exception:
            return {"error": "identify returned no net_gradient column."}
        return _min_net_gradient_from_ng(ng, low, min_candidates)
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"estimation failed: {e}"}


def _min_net_gradient_from_ng(
    ng: Any, low: int, min_candidates: int
) -> Dict[str, Any]:
    """The pure-numpy core of :func:`estimate_min_net_gradient`: given the
    candidate net-gradient values, return the recommendation + histogram. Kept
    separate so it is unit-testable without picasso (feed a synthetic bimodal /
    noise+tail distribution)."""
    ng = np.asarray(ng, float)
    ng = ng[np.isfinite(ng) & (ng > 0)]
    if ng.size < min_candidates:
        return {
            "error": f"too few candidate spots ({ng.size}) — need a busier "
            "frame or a lower floor."
        }
    logng = np.log10(ng)
    otsu_log = _otsu_threshold(logng, bins=128)  # noise/signal split (display)
    hist, edges = np.histogram(logng, bins=80)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hs = np.convolve(hist.astype(float), np.ones(5) / 5.0, mode="same")
    gmax = float(hs.max()) if hs.size else 1.0

    # local maxima above 12% of the global max = candidate peaks
    peaks = [
        i
        for i in range(1, len(hs) - 1)
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and hs[i] >= 0.12 * gmax
    ]
    regime = "noise_tail"
    valley_log = noise_pk_log = signal_pk_log = None
    if len(peaks) >= 2:
        p_lo, p_hi = sorted(
            sorted(peaks, key=lambda i: hs[i], reverse=True)[:2]
        )
        sep_ok = (centers[p_hi] - centers[p_lo]) >= 0.5  # >= ~0.5 decade apart
        v = p_lo + int(np.argmin(hs[p_lo : p_hi + 1]))
        dip_ok = hs[v] <= 0.75 * min(hs[p_lo], hs[p_hi])  # a genuine valley
        if sep_ok and dip_ok:
            regime = "bimodal"
            valley_log = float(centers[v])
            noise_pk_log = float(centers[p_lo])
            signal_pk_log = float(centers[p_hi])

    if regime == "bimodal":
        thr_log = valley_log
    else:
        pk = int(np.argmax(hs))
        peak_h = hs[pk] if hs[pk] > 0 else 1.0
        shoulder_frac = 0.05
        thr_log = None
        for i in range(pk + 1, len(hs)):
            if hs[i] < shoulder_frac * peak_h:
                thr_log = float(centers[i])
                break
        if thr_log is None:
            thr_log = (
                float(otsu_log) if otsu_log is not None else float(centers[pk])
            )
        # Floor at the noise/signal Otsu split (noise-dominated case only).
        if otsu_log is not None:
            thr_log = max(thr_log, float(otsu_log))
    # Never exceed the 99.5th percentile (guards a runaway on a heavy tail).
    thr_log = min(thr_log, float(np.percentile(logng, 99.5)))
    rec = float(10**thr_log)
    if rec >= 1000:
        rec_round = int(round(rec / 100.0) * 100)
    elif rec >= 100:
        rec_round = int(round(rec / 10.0) * 10)
    else:
        rec_round = int(round(rec))
    return {
        "recommended": rec_round,
        "recommended_raw": rec,
        "n_candidates": int(ng.size),
        "low_used": low,
        "median_ng": float(np.median(ng)),
        "hist_counts": hist.astype(int).tolist(),
        "hist_edges_log": edges.tolist(),
        "threshold_log": float(thr_log),
        "regime": regime,
        "signal_peak_ng": (
            float(10**signal_pk_log) if signal_pk_log is not None else None
        ),
        "noise_peak_ng": (
            float(10**noise_pk_log) if noise_pk_log is not None else None
        ),
        "otsu_log": (float(otsu_log) if otsu_log is not None else None),
    }


# ── clustering guide ──────────────────────────────────────────────────────────


def suggest_clustering(
    goal: str = "find protein clusters",
    nena_nm: Optional[float] = None,
    dimensionality: str = "2D",
    density: str = "unknown",
) -> Dict[str, Any]:
    """Rule-based recommendation for SMLM/DNA-PAINT cluster analysis (algorithm +
    eps ~ 2.5xNeNA + min_samples). Ported from V0.8 ``liveloc_tools``."""
    eps = None
    if nena_nm:
        try:
            eps = round(
                2.5 * float(nena_nm), 1
            )  # ~2-3x localization precision
        except (TypeError, ValueError):
            eps = None
    return {
        "goal": goal,
        "dimensionality": dimensionality,
        "density": density,
        "primary": {
            "algorithm": "DBSCAN",
            "why": "robust, widely used for SMLM point clusters; density-based, "
            "no preset cluster count.",
            "parameters": {
                "eps_nm": eps
                or "≈2–3× localization precision (NeNA); e.g. 8–15 nm for "
                "DNA-PAINT",
                "min_samples": "3–10 (raise to suppress noise/overcounting)",
            },
            "caveat": "DNA-PAINT overcounts (one docking strand blinks many "
            "times) — either group localizations per binding site first (e.g. "
            "Picasso 'link'/cluster by frame gaps) or use min_samples to "
            "compensate.",
        },
        "alternatives": [
            {
                "algorithm": "HDBSCAN",
                "when": "clusters of varying density; fewer manual parameters "
                "(set min_cluster_size).",
            },
            {
                "algorithm": "Voronoi tessellation (SR-Tesseler)",
                "when": "density-based segmentation without an eps; good for "
                "heterogeneous densities.",
            },
            {
                "algorithm": "Ripley's K / pair-correlation g(r)",
                "when": "you want to quantify clustering statistically rather "
                "than assign cluster labels.",
            },
            {
                "algorithm": "Bayesian cluster analysis (Rubin-Delanchy)",
                "when": "rigorous cluster inference accounting for localization "
                "uncertainty.",
            },
        ],
        "note": "For DNA-PAINT specifically, correct for blinking/overcounting "
        "before cluster metrics, and validate eps/min_samples on a known "
        "structure (e.g. your origami grid).",
    }


# ── qc.json sidecar writers ───────────────────────────────────────────────────


def _load_qc(path: str) -> Dict[str, Any]:
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _write_recommendation(
    qc_path: str, key: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge ``payload`` under ``recommendations.<key>`` in the qc.json sidecar,
    creating the file/section as needed. Returns the updated qc dict."""
    qc = _load_qc(qc_path)
    recs = qc.setdefault("recommendations", {})
    if not isinstance(recs, dict):
        recs = {}
        qc["recommendations"] = recs
    recs[key] = payload
    with open(qc_path, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2)
    return qc


def write_filter_suggestions_to_qc(
    qc_path: str, suggestions: Dict[str, Any]
) -> Dict[str, Any]:
    """Write a :func:`filter_suggestions` result into the qc.json sidecar under
    ``recommendations.filter`` (a #4 parameter-recommendation input)."""
    return _write_recommendation(qc_path, "filter", suggestions)


def write_clustering_to_qc(
    qc_path: str, clustering: Dict[str, Any]
) -> Dict[str, Any]:
    """Write a :func:`suggest_clustering` result into the qc.json sidecar under
    ``recommendations.clustering``."""
    return _write_recommendation(qc_path, "clustering", clustering)


def write_min_net_gradient_to_qc(
    qc_path: str, estimate: Dict[str, Any]
) -> Dict[str, Any]:
    """Write an :func:`estimate_min_net_gradient` result into the qc.json
    sidecar under ``recommendations.min_net_gradient``."""
    return _write_recommendation(qc_path, "min_net_gradient", estimate)

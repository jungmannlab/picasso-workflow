#!/usr/bin/env python3
"""
diagnose.py -- deterministic, LLM-free QC diagnostics for DNA-PAINT / SMLM.

Given the metrics the pipeline already collects (NeNA, FRC, SBR, photons,
background, localization density, on/off duty cycle, drift, overlap), this
returns a prioritised list of :class:`Finding` -- each with a likely cause and a
concrete action -- so a user asking "why is my NeNA so bad?" gets an
explainable, offline answer. This is the reliable rule-engine backbone that the
later project-aware chatbot (WP-14) phrases in natural language.

Public functions
----------------
diagnose(metrics, targets=None)            -> list[Finding]   (full sweep)
focus(topic, metrics, targets=None)        -> list[Finding]   (topic subset)
summary(metrics, targets=None, top=None)   -> str             (plain text)
frc_trend(frames, frc_nm, ...)             -> dict | None     (stop signal)
center_edge_trend(frames, center, edge, .) -> dict | None     (photodamage)
cohort_stats(runs, metrics=None, min_n=1)  -> dict            (registry -> stats)
db_anomalies(metrics, db_stats, ...)       -> list[Finding]   (cohort outliers)

``metrics`` is a flat dict using the qc.json metric keys (see :data:`METRIC_KEYS`);
missing keys are simply skipped.

Ported from LiveLocalization V0.8 ``qc_advisor.py`` (the numeric behaviour is
preserved on shared fixtures); ``cohort_stats`` is the new bridge from the
picasso-registry client's cohort run list to the p20/p80 stats ``db_anomalies``
consumes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from picasso_workflow.qc_advisor.findings import (
    Finding,
    sort_findings,
)

# ── tunable thresholds (DNA-PAINT defaults; override via diagnose(..., targets=))
THRESHOLDS: Dict[str, float] = {
    "nena_good_nm": 4.0,  # NeNA precision: good <= 4 nm
    "nena_poor_nm": 8.0,  # poor >= 8 nm
    "frc_good_nm": 10.0,
    "frc_poor_nm": 20.0,
    "photons_low": 2500,  # photons/loc: low if below (Cy3B DNA-PAINT ~5-20k)
    "photons_good": 5000,
    # SBR on the Steen-2024 scale (central-pixel signal / per-pixel background),
    # which reads lower than the old median-photon/median-bg ratio. Calibrated
    # to typical DNA-PAINT distributions (good >~ 6, poor <~ 2); tune per setup.
    "sbr_low": 2.0,
    "sbr_good": 6.0,
    "background_high": 400,  # per-frame background (photons); scope-dependent
    "laser_high_mw_origami": 80,  # laser power (mW) considered high for origami
    "laser_high_mw_cells": 40,  # ...and for cells (lower -- bleaching/phototox)
    "overlap_high_pct": 10.0,  # % of locs with a near-neighbour within a PSF box
    "duty_cycle_high": 0.10,  # on/(on+off) -- DNA-PAINT should be a few %
    "duty_cycle_ok": 0.05,
    "drift_high_nm": 30.0,  # peak-to-peak uncorrected drift over the run
    "locs_min": 50000,  # very rough "enough sampling" floor for a FOV
    "frc_vs_nena_factor": 3.0,  # FRC >> factor x NeNA -> undersampling/drift
}

# qc.json metric keys this module understands (all optional)
METRIC_KEYS = (
    "nena_zoom_nm",
    "nena_global_nm",
    "frc_resolution_nm",
    "sbr",
    "photons_per_loc",
    "background",
    "localizations",
    "loc_density_fov_um2",
    "loc_density_zoom_um2",
    "on_time_ms",
    "off_time_ms",
    "on_time_frames",
    "off_time_frames",
    "overlap_pct",
    "drift_ptp_x_nm",
    "drift_ptp_y_nm",
    "conc_pm",
    "power_mw",
    "exposure_ms",
)


def _g(m: Dict[str, Any], k: str) -> Optional[float]:
    v = m.get(k)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _duty_cycle(m: Dict[str, Any]) -> Optional[float]:
    on = _g(m, "on_time_frames") or _g(m, "on_time_ms")
    off = _g(m, "off_time_frames") or _g(m, "off_time_ms")
    if on is not None and off is not None and (on + off) > 0:
        return on / (on + off)
    return None


def diagnose(
    metrics: Optional[Dict[str, Any]],
    targets: Optional[Dict[str, float]] = None,
) -> List[Finding]:
    """Return a prioritised list[Finding] over the metrics present in
    ``metrics``. ``targets`` overrides the default :data:`THRESHOLDS`."""
    T = dict(THRESHOLDS)
    T.update(targets or {})
    m = metrics or {}
    f: List[Finding] = []

    nena = _g(m, "nena_zoom_nm") or _g(m, "nena_global_nm")
    frc = _g(m, "frc_resolution_nm")
    sbr = _g(m, "sbr")
    ph = _g(m, "photons_per_loc")
    bg = _g(m, "background")
    locs = _g(m, "localizations")
    overlap = _g(m, "overlap_pct")
    duty = _duty_cycle(m)
    dx = _g(m, "drift_ptp_x_nm")
    dy = _g(m, "drift_ptp_y_nm")
    drift = max([v for v in (dx, dy) if v is not None], default=None)

    # ── NeNA (localization precision) ──────────────────────────────────────
    if nena is not None:
        if nena <= T["nena_good_nm"]:
            f.append(
                Finding(
                    "nena",
                    "ok",
                    f"NeNA {nena:.1f} nm — good precision.",
                    value=nena,
                )
            )
        elif nena >= T["nena_poor_nm"]:
            causes: List[str] = []
            actions: List[str] = []
            if ph is not None and ph < T["photons_low"]:
                causes.append(f"low photon count ({ph:.0f}/loc)")
                actions.append(
                    "raise laser power or exposure; check the dye isn't "
                    "bleaching"
                )
            if sbr is not None and sbr < T["sbr_low"]:
                causes.append(f"low SBR ({sbr:.1f})")
                actions.append(
                    "reduce imager concentration and/or improve TIRF/HILO angle"
                )
            if overlap is not None and overlap > T["overlap_high_pct"]:
                causes.append(f"high spot overlap ({overlap:.0f}%)")
                actions.append(
                    "lower the imager concentration to reduce overlapping PSFs"
                )
            if duty is not None and duty > T["duty_cycle_high"]:
                causes.append(f"high duty cycle ({duty * 100:.1f}%)")
                actions.append(
                    "lower imager concentration / shorten bright time"
                )
            f.append(
                Finding(
                    "nena",
                    "bad",
                    f"NeNA {nena:.1f} nm — poor precision.",
                    cause="; ".join(causes)
                    or "few photons or high background",
                    action=" · ".join(actions)
                    or "increase photons (power/exposure) and reduce "
                    "background",
                    value=nena,
                )
            )
        else:
            f.append(
                Finding(
                    "nena",
                    "warn",
                    f"NeNA {nena:.1f} nm — moderate.",
                    value=nena,
                )
            )

    # ── photons ────────────────────────────────────────────────────────────
    if ph is not None:
        if ph < T["photons_low"]:
            f.append(
                Finding(
                    "photons_per_loc",
                    "bad",
                    f"{ph:.0f} photons/loc — low.",
                    cause="insufficient excitation, short exposure, or a "
                    "dim/bleaching dye",
                    action="increase laser power or exposure time; verify the "
                    "imager dye",
                    value=ph,
                )
            )
        elif ph >= T["photons_good"]:
            f.append(
                Finding(
                    "photons_per_loc",
                    "ok",
                    f"{ph:.0f} photons/loc — healthy.",
                    value=ph,
                )
            )
        else:
            f.append(
                Finding(
                    "photons_per_loc",
                    "warn",
                    f"{ph:.0f} photons/loc — modest.",
                    value=ph,
                )
            )

    # ── SBR / background ─────────────────────────────────────────────────────
    if sbr is not None:
        if sbr < T["sbr_low"]:
            over_conc = (
                overlap is not None and overlap > T["overlap_high_pct"]
            ) or (duty is not None and duty > T["duty_cycle_high"])
            bg_high = bg is not None and bg > T["background_high"]
            if over_conc:
                _cause = (
                    "too much imager in solution → overlapping / out-of-focus "
                    "signal"
                )
                _action = "reduce imager concentration; check buffer refractive index"
            elif bg_high:
                _cause = (
                    "elevated background (stray/out-of-focus light or "
                    "autofluorescence)"
                )
                _action = (
                    "reduce background: tighten TIRF/HILO angle, shield ambient "
                    "light, check the buffer"
                )
            else:
                _cause = (
                    "photon-limited — emitter density, duty cycle and "
                    "background are all fine, so concentration/background are "
                    "not the issue"
                )
                _action = (
                    "increase photons: more laser power or longer exposure "
                    "(a brighter/less-bleaching dye also helps)"
                )
            f.append(
                Finding(
                    "sbr",
                    "bad",
                    f"SBR {sbr:.1f} — low signal-to-background.",
                    cause=_cause,
                    action=_action,
                    value=sbr,
                )
            )
        elif sbr >= T["sbr_good"]:
            f.append(
                Finding(
                    "sbr",
                    "ok",
                    f"SBR {sbr:.1f} — good contrast.",
                    value=sbr,
                )
            )
        else:
            f.append(
                Finding(
                    "sbr",
                    "warn",
                    f"SBR {sbr:.1f} — borderline.",
                    value=sbr,
                )
            )

    # Laser power context: high power is a common, expected cause of elevated
    # background (and faster docking-site damage). Thresholds differ by sample.
    power = _g(m, "power_mw")
    _stype = str(m.get("sample_type") or "").lower()
    _is_cells = _stype.startswith("cell")
    _laser_hi = (
        T["laser_high_mw_cells"] if _is_cells else T["laser_high_mw_origami"]
    )
    _power_high = power is not None and power > _laser_hi
    if bg is not None and bg > T["background_high"]:
        if _power_high:
            _cause = (
                f"high laser power ({power:.0f} mW > {_laser_hi:.0f} mW for "
                f"{'cells' if _is_cells else 'DNA-Origami'}) — expected at this "
                "power; also check imager conc. / stray light / autofluorescence"
            )
            _action = (
                "if the background is acceptable for your goal, this is a "
                "power trade-off; otherwise lower the power, improve TIRF/HILO, "
                "shield ambient light or lower imager conc."
            )
        else:
            _cause = (
                "high imager concentration, stray/ambient light, or "
                "autofluorescence"
            )
            _action = (
                "lower imager conc.; improve TIRF/HILO; shield ambient light"
            )
        f.append(
            Finding(
                "background",
                "warn",
                f"Background {bg:.0f} — elevated.",
                cause=_cause,
                action=_action,
                value=bg,
            )
        )
    # Standalone laser-power note (even if the background happens to be fine).
    if _power_high:
        f.append(
            Finding(
                "laser_power",
                "info",
                f"Laser power {power:.0f} mW — high for "
                f"{'cells' if _is_cells else 'DNA-Origami'} "
                f"(> {_laser_hi:.0f} mW).",
                cause="expect elevated background and faster docking-site / "
                "dye damage (photobleaching, phototoxicity in cells)",
                action="intended for more photons/faster sampling — just keep "
                "it consistent across runs you compare, and watch the "
                "docking-site damage (dye analysis) and background",
                value=power,
            )
        )

    # ── concentration / overlap / duty cycle ─────────────────────────────────
    if overlap is not None and overlap > T["overlap_high_pct"]:
        f.append(
            Finding(
                "overlap",
                "bad",
                f"{overlap:.0f}% of locs overlap within a PSF box.",
                cause="imager concentration too high → overlapping "
                "single-emitter fits bias x/y",
                action="reduce imager concentration (or shorten exposure) "
                "until overlap < 5–10%",
                value=overlap,
            )
        )
    if duty is not None:
        if duty > T["duty_cycle_high"]:
            f.append(
                Finding(
                    "duty_cycle",
                    "warn",
                    f"Duty cycle {duty * 100:.1f}% — high for DNA-PAINT.",
                    cause="too many simultaneously-bound imagers (conc too "
                    "high / bright time long)",
                    action="lower imager concentration; DNA-PAINT works best "
                    "at a few % duty cycle",
                    value=duty,
                )
            )
        elif duty <= T["duty_cycle_ok"]:
            f.append(
                Finding(
                    "duty_cycle",
                    "ok",
                    f"Duty cycle {duty * 100:.1f}% — good sparsity.",
                    value=duty,
                )
            )

    # ── drift ─────────────────────────────────────────────────────────────
    if drift is not None and drift > T["drift_high_nm"]:
        f.append(
            Finding(
                "drift",
                "warn",
                f"Residual drift {drift:.0f} nm p-p.",
                cause="incomplete drift correction or no fiducials",
                action="add fiducials / enable undrift; check stage/thermal "
                "stability",
                value=drift,
            )
        )

    # ── FRC vs NeNA cross-check (resolution limited by sampling, not precision)
    if (
        frc is not None
        and nena is not None
        and frc > T["frc_vs_nena_factor"] * nena
    ):
        f.append(
            Finding(
                "frc",
                "warn",
                f"FRC {frc:.1f} nm ≫ {T['frc_vs_nena_factor']:.0f}×NeNA "
                f"({nena:.1f} nm).",
                cause="resolution limited by undersampling or residual drift, "
                "not by precision",
                action="acquire more frames (more localizations) and verify "
                "drift correction",
                value=frc,
            )
        )
    elif frc is not None:
        if frc <= T["frc_good_nm"]:
            f.append(
                Finding(
                    "frc",
                    "ok",
                    f"FRC {frc:.1f} nm — good resolution.",
                    value=frc,
                )
            )
        elif frc >= T["frc_poor_nm"]:
            f.append(
                Finding(
                    "frc",
                    "bad",
                    f"FRC {frc:.1f} nm — coarse resolution.",
                    cause="too few localizations, drift, or poor precision",
                    action="acquire longer; check NeNA and drift above",
                    value=frc,
                )
            )

    if locs is not None and locs < T["locs_min"]:
        f.append(
            Finding(
                "localizations",
                "info",
                f"Only {locs:.0f} localizations — sampling may be sparse.",
                action="acquire more frames if the structure looks "
                "under-sampled",
                value=locs,
            )
        )

    return sort_findings(f)


# ── FRC-over-frames trend: keep acquiring vs you can stop ────────────────────
def frc_trend(
    frames: Optional[Sequence[float]],
    frc_nm: Optional[Sequence[float]],
    target_nm: Optional[float] = None,
    plateau_pct: float = 3.0,
    min_points: int = 4,
    undrift_done: Optional[bool] = None,
    progress_frac: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Decide whether FRC resolution is still improving with more frames or has
    plateaued -- a 'keep acquiring vs you can stop' recommendation.

    Fits a power law ``FRC ≈ a·N^(-b)`` on the RECENT part of the FRC-over-
    frames series. The exponent ``b`` says how fast resolution still improves::

        gain_if_doubled = (1 - 2**(-b)) * 100      # % FRC gain if N doubles

    ``b ≈ 0`` -> plateau; ``b < 0`` -> FRC getting worse (drift building up).

    Returns a dict (status, message, action, ...) or ``None`` if the inputs are
    unusable. status in {'insufficient','improving','plateau','plateau_check',
    'plateau_early','degrading'}.

    NET-NEW to the plan: the positive complement to early-abort. Reaching a
    resolution NUMBER is deliberately NOT a stop signal (the structure can still
    be undersampled); only a genuine plateau of the FRC-vs-frames curve is.
    """
    try:
        raw = [
            (float(n), float(r))
            for n, r in zip(frames or [], frc_nm or [])
            if n is not None
            and r is not None
            and math.isfinite(float(n))
            and math.isfinite(float(r))
            and float(n) > 0
            and float(r) > 0
        ]
    except (TypeError, ValueError):
        return None
    # de-duplicate by frame (keep the latest value), then sort by frame
    seen: Dict[float, float] = {}
    for n, r in raw:
        seen[n] = r
    pts = sorted(seen.items())
    if len(pts) < min_points:
        return {
            "status": "insufficient",
            "n_points": len(pts),
            "message": "Not enough FRC-over-time points yet to judge the "
            "trend.",
            "action": "keep acquiring; the trend appears once a few FRC "
            "updates are in.",
        }

    # recent segment = last half of the series (at least min_points)
    k = max(min_points, (len(pts) + 1) // 2)
    seg = pts[-k:]
    xs = [math.log(n) for n, _ in seg]
    ys = [math.log(r) for _, r in seg]
    ns = len(seg)
    mx = sum(xs) / ns
    my = sum(ys) / ns
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {
            "status": "insufficient",
            "n_points": len(pts),
            "message": "FRC series has no frame spread yet.",
            "action": "keep acquiring.",
        }
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx  # d(log FRC)/d(log N)
    b = -slope  # FRC ~ a*N^(-b); b>0 => resolution improving
    loga = my - slope * mx
    cur_frames, cur_frc = seg[-1]
    gain = (1.0 - 2.0 ** (-b)) * 100.0  # % FRC improvement if frames double

    out: Dict[str, Any] = {
        "n_points": len(pts),
        "frames": round(cur_frames),
        "frc_nm": round(cur_frc, 2),
        "exponent_b": round(b, 3),
        "gain_if_doubled_pct": round(gain, 1),
    }

    if b <= -0.02 and gain < -plateau_pct:
        out.update(
            status="degrading",
            message=f"FRC is getting worse over recent frames "
            f"({cur_frc:.1f} nm and rising).",
            cause="drift building up during the run (or the FRC region "
            "emptying out)",
            action="check/enable undrift and verify stage & focus stability, "
            "then continue.",
        )
        return out
    if gain < plateau_pct:
        pct = (
            f" (~{progress_frac * 100:.0f}% of frames)"
            if progress_frac is not None
            else ""
        )
        if undrift_done is False:
            out.update(
                status="plateau_check",
                message=f"FRC is flat at ≈{cur_frc:.1f} nm, but this run isn't "
                f"drift-corrected yet.",
                cause="before undrift a flat FRC is usually drift-limited, not "
                "the true resolution — the structure may also still be "
                "undersampled",
                action="run undrift; if the FRC stays flat afterwards and the "
                "structure looks complete, you can stop.",
            )
            return out
        if progress_frac is not None and progress_frac < 0.2:
            out.update(
                status="plateau_early",
                message=f"FRC isn't improving yet{pct} at ≈{cur_frc:.1f} nm.",
                cause="this early a flat FRC is often a sparse/empty FRC region "
                "or residual drift rather than a finished reconstruction",
                action="keep acquiring; make sure the FRC is measured on a "
                "structure-rich zoom, then re-check later.",
            )
            return out
        out.update(
            status="plateau",
            message=f"FRC has plateaued at ≈{cur_frc:.1f} nm — doubling the "
            f"frames would improve it by only ~{max(gain, 0.0):.0f}%.",
            cause="resolution now limited by precision/drift, not by sampling",
            action="you can stop acquiring — more frames won't sharpen the "
            "image much.",
        )
        return out

    msg = (
        f"FRC still improving (≈{cur_frc:.1f} nm now; ~{gain:.0f}% better if "
        f"you double the frames)."
    )
    action = (
        "keep acquiring — resolution is still gaining from more localizations."
    )
    if target_nm and cur_frc <= float(target_nm):
        action = (
            "keep acquiring — the FRC already reads below your target, but it "
            "is still improving, so the structure is not fully sampled yet."
        )
    elif target_nm and b > 1e-6:
        try:
            n_target = math.exp((loga - math.log(float(target_nm))) / b)
            if math.isfinite(n_target) and n_target > cur_frames:
                extra = n_target - cur_frames
                out["eta_frames"] = round(n_target)
                out["extra_frames"] = round(extra)
                action = (
                    f"keep acquiring — about {round(extra):,} more frames "
                    f"(~{round(n_target):,} total) to reach "
                    f"{float(target_nm):.1f} nm at the current rate."
                )
        except (ValueError, OverflowError):
            pass
    out.update(status="improving", message=msg, action=action)
    return out


# ── Center-vs-edge density trend: photodamage vs illumination profile ────────
def center_edge_trend(
    frames: Optional[Sequence[float]],
    center: Optional[Sequence[float]],
    edge: Optional[Sequence[float]],
    min_points: int = 6,
    drop_flag_pct: float = 15.0,
    dip_ratio: float = 0.85,
    rise_ratio: float = 1.15,
) -> Optional[Dict[str, Any]]:
    """Diagnose a spatially non-uniform reconstruction: is the CENTRE of the FOV
    losing localization density relative to the EDGES over the run?

    ``center`` / ``edge`` are per-batch density (or luminance) proxies. The ratio
    ``r = centre/edge`` is tracked over frames:

      * r FALLING over time -> 'declining' (centre degrading; at high laser
        power this is the docking-strand photodamage signature: lower power).
      * r low but ~FLAT     -> 'static_dip' (an illumination profile / off-centre
        TIRF, present from the start -- not damage).
      * r ~ 1 (or rising)   -> 'uniform'.

    Returns a dict (status, message, ...) or ``None`` if unusable. NET-NEW.
    """
    try:
        pts = [
            (float(f), float(c) / float(e))
            for f, c, e in zip(frames or [], center or [], edge or [])
            if f is not None
            and c is not None
            and e is not None
            and math.isfinite(float(f))
            and math.isfinite(float(c))
            and math.isfinite(float(e))
            and float(e) > 0
            and float(c) >= 0
        ]
    except (TypeError, ValueError):
        return None
    seen: Dict[float, float] = {}
    for f, r in pts:
        seen[f] = r
    pts = sorted(seen.items())
    if len(pts) < min_points:
        return {"status": "insufficient", "n_points": len(pts)}

    rs = [r for _, r in pts]
    k = max(2, len(pts) // 3)
    early = sorted(rs[:k])[len(rs[:k]) // 2]  # median of first third
    now = sorted(rs[-k:])[len(rs[-k:]) // 2]  # median of last third
    drop_pct = (early - now) / early * 100.0 if early > 0 else 0.0

    # slope of r over the recent half (sign only)
    seg = pts[len(pts) // 2 :]
    n = len(seg)
    mx = sum(f for f, _ in seg) / n
    my = sum(r for _, r in seg) / n
    sxx = sum((f - mx) ** 2 for f, _ in seg)
    slope = (
        (sum((f - mx) * (r - my) for f, r in seg) / sxx) if sxx > 0 else 0.0
    )

    out: Dict[str, Any] = {
        "n_points": len(pts),
        "ratio_now": round(now, 3),
        "ratio_early": round(early, 3),
        "drop_pct": round(drop_pct, 1),
    }

    if drop_pct >= drop_flag_pct and slope < 0:
        out.update(
            status="declining",
            message=f"Central density is falling relative to the edges "
            f"(centre/edge {early:.2f} → {now:.2f}, −{drop_pct:.0f}% over the "
            f"run).",
            cause="at high laser power this is the signature of photodamage to "
            "the docking strands in the brightest (central) illumination",
            action="lower the laser power (or defocus/expand the beam) and "
            "check whether the centre recovers; compare a fresh area at lower "
            "power.",
        )
    elif now < dip_ratio:
        out.update(
            status="static_dip",
            message=f"The centre is dimmer than the edges but roughly stable "
            f"(centre/edge ≈ {now:.2f}).",
            cause="looks like an illumination profile (off-centre / uneven "
            "TIRF), not photodamage",
            action="if it bothers you, flatten/centre the illumination; it is "
            "not a power problem.",
        )
    elif now > rise_ratio:
        out.update(
            status="uniform",
            message=f"Centre is denser than the edges (centre/edge ≈ {now:.2f})"
            f" — normal Gaussian illumination.",
        )
    else:
        out.update(
            status="uniform",
            message=f"Central vs edge density is uniform "
            f"(centre/edge ≈ {now:.2f}).",
        )
    return out


# ── DB anomaly check: live metrics vs the user's own cohort ───────────────────
# NB: the absolute 'localizations' count is deliberately NOT compared -- it
# scales with the imaged FOV, so only the area-normalized density is comparable.
_ANOM_LOWER_BETTER = {
    "frc_resolution_nm",
    "nena_zoom_nm",
    "nena_global_nm",
    "background",
}
_ANOM_HIGHER_BETTER = {
    "sbr",
    "photons_per_loc",
    "loc_density_fov_um2",
    "loc_density_zoom_um2",
}
# metrics summarised into cohort stats (both directions).
STAT_METRICS = tuple(sorted(_ANOM_LOWER_BETTER | _ANOM_HIGHER_BETTER))

_ANOM_LABEL = {
    "frc_resolution_nm": "FRC",
    "nena_zoom_nm": "NeNA",
    "nena_global_nm": "NeNA",
    "background": "Background",
    "sbr": "SBR",
    "photons_per_loc": "Photons/loc",
    "localizations": "Localizations",
    "loc_density_fov_um2": "Loc density",
    "loc_density_zoom_um2": "Loc density",
}
_ANOM_CAUSE = {
    "photons_per_loc": "laser power dropped, TIRF/HILO misalignment, or "
    "dye/imager degradation",
    "background": "stray/ambient light, buffer autofluorescence, or laser "
    "instability vs your usual",
    "sbr": "over-concentrated imager or elevated background vs your usual runs",
    "nena_zoom_nm": "fewer photons / higher background than your usual runs",
    "nena_global_nm": "fewer photons / higher background than your usual runs",
    "frc_resolution_nm": "undersampling or residual drift vs your usual runs",
    "localizations": "weak binding — imager concentration, target density, or "
    "too-short acquisition",
    "loc_density_fov_um2": "weak binding — imager concentration, target "
    "density, or too-short acquisition",
    "loc_density_zoom_um2": "weak binding — imager concentration, target "
    "density, or too-short acquisition",
}
_ANOM_ACTION = {
    "photons_per_loc": "check laser power/alignment and the TIRF angle; verify "
    "the imager is fresh",
    "background": "tighten TIRF/HILO, shield ambient light, check the buffer",
    "sbr": "reduce imager concentration and/or lower background; compare to a "
    "good past run",
    "nena_zoom_nm": "raise photons (power/exposure) and reduce background",
    "nena_global_nm": "raise photons (power/exposure) and reduce background",
    "frc_resolution_nm": "acquire longer and verify drift correction",
    "localizations": "check imager concentration and that the sample binds; "
    "acquire longer",
    "loc_density_fov_um2": "check imager concentration and that the sample "
    "binds; acquire longer",
    "loc_density_zoom_um2": "check imager concentration and that the sample "
    "binds; acquire longer",
}


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def cohort_stats(
    runs: Iterable[Dict[str, Any]],
    metrics: Optional[Sequence[str]] = None,
    min_n: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Compute the ``db_stats``-shaped per-metric summary that
    :func:`db_anomalies` consumes, from a list of cohort run dicts.

    This is the bridge from the picasso-registry client's ``cohort(taxon_id)``
    output (a list of run records, each a flat dict that may carry the metric
    keys) to the ``{metric: {n, median, p20, p80, min, max}}`` structure. Kept
    separate from :func:`db_anomalies` so the rule engine stays hermetic and the
    registry stays an injected dependency, not an import.

    Only :data:`STAT_METRICS` are summarised. A metric with fewer than ``min_n``
    finite values is omitted.
    """
    wanted = [k for k in (metrics or STAT_METRICS) if k in STAT_METRICS]
    pooled: Dict[str, List[float]] = {k: [] for k in wanted}
    for run in runs or []:
        if not isinstance(run, dict):
            continue
        for k in wanted:
            v = _g(run, k)
            if v is not None and math.isfinite(v):
                pooled[k].append(v)
    out: Dict[str, Dict[str, float]] = {}
    for met, vals in pooled.items():
        if len(vals) < min_n:
            continue
        vals = sorted(vals)
        out[met] = {
            "n": len(vals),
            "median": round(_percentile(vals, 50), 2),
            "p20": round(_percentile(vals, 20), 2),
            "p80": round(_percentile(vals, 80), 2),
            "min": round(vals[0], 2),
            "max": round(vals[-1], 2),
        }
    return out


def db_anomalies(
    metrics: Optional[Dict[str, Any]],
    db_stats: Optional[Dict[str, Dict[str, float]]],
    z_flag: float = 2.5,
    z_bad: float = 4.0,
    min_n: int = 5,
) -> List[Finding]:
    """Flag metrics that are strong OUTLIERS versus the user's own cohort -- an
    early warning like 'photons unusually low -> possible laser/alignment
    problem'.

    ``db_stats`` is the per-metric dict ``{metric: {'n','median','p20','p80',
    'min','max'}}`` (from :func:`cohort_stats`, or the registry / mock). The
    deviation is measured in robust 'spread' units (distance from the median in
    p20<->p80 half-widths), so no normality assumption is needed. Only the WORSE
    direction is flagged as a problem; an unusually GOOD value is reported as
    info. Returns list[Finding] sorted by severity.
    """
    m = metrics or {}
    stats = db_stats or {}
    out: List[Finding] = []
    for met, st in stats.items():
        if met not in _ANOM_LOWER_BETTER and met not in _ANOM_HIGHER_BETTER:
            continue
        try:
            n = int(st.get("n", 0))
            median = float(st["median"])
            p20 = float(st["p20"])
            p80 = float(st["p80"])
        except (KeyError, TypeError, ValueError):
            continue
        if n < min_n:
            continue
        val = _g(m, met)
        if val is None:
            continue
        eps = max(abs(median) * 0.02, 1e-9)
        if val >= median:
            spread = max(p80 - median, eps)
            dev = (val - median) / spread  # >= 0
        else:
            spread = max(median - p20, eps)
            dev = (val - median) / spread  # <= 0
        mag = abs(dev)
        if mag < z_flag:
            continue
        lower_better = met in _ANOM_LOWER_BETTER
        worse = (val > median) if lower_better else (val < median)
        lbl = _ANOM_LABEL.get(met, met)
        above = val > median
        try:
            fold = (val / median) if above else (median / val)
        except ZeroDivisionError:
            fold = float("inf")
        fold_txt = f"{fold:.1f}× " if math.isfinite(fold) else ""
        dir_txt = "above" if above else "below"
        band = f"median {median:g}, p20–p80 {p20:g}–{p80:g}, n={n}"
        if worse:
            sev = "bad" if mag >= z_bad else "warn"
            out.append(
                Finding(
                    met,
                    sev,
                    f"{lbl} {val:g} — {fold_txt}{dir_txt} your usual "
                    f"({band}).",
                    cause=_ANOM_CAUSE.get(
                        met,
                        "an unusual acquisition condition vs your database",
                    ),
                    action=_ANOM_ACTION.get(
                        met, "compare against a known-good past run"
                    ),
                    value=val,
                    source="db_anomalies",
                    detail={"deviation": round(dev, 3), "n": n},
                )
            )
        else:
            out.append(
                Finding(
                    met,
                    "info",
                    f"{lbl} {val:g} — unusually good ({fold_txt}{dir_txt} your "
                    f"usual; {band}).",
                    cause="better than your typical runs at this setup",
                    value=val,
                    source="db_anomalies",
                    detail={"deviation": round(dev, 3), "n": n},
                )
            )
    return sort_findings(out)


# which metrics each "focus" topic depends on
FOCUS_MAP = {
    "nena": (
        "nena",
        "photons_per_loc",
        "sbr",
        "overlap",
        "duty_cycle",
        "background",
    ),
    "frc": ("frc", "nena", "localizations", "drift"),
    "sbr": ("sbr", "background", "overlap", "duty_cycle"),
    "overlap": ("overlap", "duty_cycle"),
    "drift": ("drift", "frc"),
    "photons": ("photons_per_loc", "sbr"),
}


def focus(
    topic: str,
    metrics: Optional[Dict[str, Any]],
    targets: Optional[Dict[str, float]] = None,
) -> List[Finding]:
    """Return only the findings relevant to a topic, e.g. focus('nena', m)."""
    keep = set(FOCUS_MAP.get(topic, (topic,)))
    return [fd for fd in diagnose(metrics, targets) if fd.metric in keep]


def summary(
    metrics: Optional[Dict[str, Any]],
    targets: Optional[Dict[str, float]] = None,
    top: Optional[int] = None,
) -> str:
    """Plain-text report; ``top`` limits to the N most severe findings."""
    fs = diagnose(metrics, targets)
    if top:
        fs = fs[:top]
    if not fs:
        return "No metrics available to assess."
    return "\n".join(str(fd) for fd in fs)

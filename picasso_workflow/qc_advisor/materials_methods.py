#!/usr/bin/env python3
"""
materials_methods.py -- deterministic Materials & Methods paragraph generation.

Builds a Materials & Methods paragraph for a report from a measurement's qc.json
(the descriptor + metrics) plus a hardware-profile YAML (microscope_config.yaml).
Deterministic templating -- no LLM. Missing fields degrade gracefully to
"not specified" so the text always renders. The (later) agent layer (9.5) layers
natural-language phrasing on top of this reliable backbone.

    from picasso_workflow.qc_advisor import materials_and_methods
    text = materials_and_methods(qc_record)                # setup from label
    text = materials_and_methods(qc_record, setup_name="Voyager")
    text = materials_and_methods(qc_record, config_path="microscope_config.yaml")

Ported from LiveLocalization V0.8 ``materials_methods.py``. The V0.8 module
defaulted the config path to a file sitting next to itself; here there is no
bundled hardware profile, so ``config_path`` defaults to ``None`` (renders with
"not specified" for hardware fields) and callers pass their own profile.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a transitive dep
    yaml = None

NS = "not specified"


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path or yaml is None or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _clean(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or (s.startswith("<") and s.endswith(">")):
        return None
    return s


def _pick_setup(
    cfg: Dict[str, Any],
    setup_name: Optional[str],
    scope_label: Optional[str],
) -> Dict[str, Any]:
    """Return common (+) setup (setup keys override common)."""
    common = dict(cfg.get("common", {}) or {})
    setups = cfg.get("setups", {}) or {}
    chosen: Dict[str, Any] = {}
    if setup_name and setup_name in setups:
        chosen = setups[setup_name]
    elif scope_label:
        for name, blk in setups.items():
            if str(scope_label).lower() == name.lower():
                chosen = blk
                break
        else:
            for name, blk in setups.items():
                if (
                    str(scope_label).lower() in name.lower()
                    or name.lower() in str(scope_label).lower()
                ):
                    chosen = blk
                    break
    if not chosen:
        dft = cfg.get("default_setup")
        chosen = (
            setups.get(dft, {})
            if dft
            else (next(iter(setups.values()), {}) if setups else {})
        )
    merged = dict(common)
    merged.update(chosen or {})
    return merged


def _gk(d: Any, k: str) -> Optional[str]:
    return _clean(d.get(k)) if isinstance(d, dict) else None


def materials_and_methods(
    qc: Dict[str, Any],
    setup_name: Optional[str] = None,
    config_path: Optional[str] = None,
) -> str:
    """Render a deterministic Materials & Methods paragraph from a qc record and
    an optional hardware-profile YAML. Ported from V0.8."""
    cfg = _load_config(config_path)
    ms = qc.get("measurement", {}) or {}
    sm = qc.get("sample", {}) or {}
    ac = qc.get("acquisition", {}) or {}
    me = qc.get("metrics", {}) or {}
    S = _pick_setup(cfg, setup_name, ms.get("microscope"))

    obj = S.get("objective", {}) or {}
    cam = S.get("camera", {}) or {}
    comb = S.get("laser_combiners", {}) or {}
    coup = S.get("laser_coupling", {}) or {}
    a3d = S.get("astigmatism_3d", {}) or {}
    fsets = S.get("filter_sets", {}) or {}
    sw = S.get("software", {}) or {}
    lasers = S.get("lasers", []) or []

    p = []

    # 1) microscope + objective + illumination
    stand = _gk(S, "stand") or NS
    om, omod = _gk(obj, "manufacturer"), _gk(obj, "model")
    obj_name = " ".join([x for x in (om, omod) if x]) or NS
    specs = []
    if _gk(obj, "na"):
        specs.append(f"NA {_gk(obj, 'na')}")
    if _gk(obj, "working_distance_mm"):
        specs.append(f"WD {_gk(obj, 'working_distance_mm')} mm")
    obj_spec = f" ({', '.join(specs)})" if specs else ""
    imm = _gk(obj, "immersion")
    imm_txt = (
        f" {imm}-immersion"
        if (imm and imm.lower() not in (omod or "").lower())
        else ""
    )
    obj_txt = f"a {obj_name}{imm_txt} objective{obj_spec}"
    illum = _gk(S, "illumination") or NS
    p.append(
        f"Super-resolution imaging was performed on a {stand} equipped with "
        f"{obj_txt} under {illum} illumination."
    )

    # 2) sample / DNA-PAINT
    sample_type = _gk(sm, "sample_type") or "sample"
    target = _gk(sm, "target")
    binder = _gk(sm, "binder")
    dye = _gk(sm, "dye")
    imager = _gk(sm, "imager") or _gk(ac, "imager_sequence") or NS
    conc = sm.get("conc_pm") or sm.get("conc_pM")
    buffer_ = _gk(sm, "buffer")
    tgt = f" targeting {target}" if target else ""
    bnd = f", labelled with {binder}," if binder else ""
    p.append(
        f"DNA-PAINT imaging of the {sample_type}{tgt}{bnd} used the imager "
        f"{imager}"
        + (f" ({dye})" if dye else "")
        + (f" at {int(conc)} pM" if conc else "")
        + (f" in {buffer_}" if buffer_ else "")
        + "."
    )

    # 3) excitation: lasers + combiners + fiber
    laser_bits = []
    for laser in lasers:
        wl = _gk(laser, "wavelength_nm")
        if wl:
            laser_bits.append(
                f"{wl} nm"
                + (
                    f" ({_gk(laser, 'max_power_mw')} mW)"
                    if _gk(laser, "max_power_mw")
                    else ""
                )
            )
    lc = _gk(S, "laser_company")
    lcls = _gk(S, "laser_class")
    lasers_txt = ", ".join(laser_bits) if laser_bits else NS
    exc = "Laser excitation was provided by "
    exc += (f"{lc} " if lc else "") + f"fiber lasers ({lasers_txt}"
    exc += (f"; class {lcls}" if lcls else "") + ")"
    power = sm.get("power_mw") or (
        ac.get("power_mw") if isinstance(ac, dict) else None
    )
    if power:
        try:
            factor = float(_clean(S.get("power_density_wcm2_per_mw")) or 0)
        except (TypeError, ValueError):
            factor = 0
        if factor:
            wcm2 = float(power) * factor
            exc += f", set to {wcm2:g} W/cm² ({power} mW) for this measurement"
        else:
            exc += f", set to {power} mW for this measurement"
    pc = S.get("power_control", {}) or {}
    if _gk(pc, "half_wave_plate") or _gk(pc, "polarizing_beam_splitter"):
        hw = _gk(pc, "half_wave_plate")
        rm = _gk(pc, "rotation_mount")
        pbs = _gk(pc, "polarizing_beam_splitter")
        pcm = _gk(pc, "manufacturer") or ""
        exc += (
            f". Laser power was adjusted with a motorized half-wave plate "
            f"({pcm} {hw}"
            + (f", {rm} rotation mount" if rm else "")
            + ")"
            + (f" and a polarizing beam splitter ({pbs})" if pbs else "")
        )
    combs = comb.get("dichroics", []) or []
    if combs:
        parts = [
            f"{_gk(d, 'part_number')}" for d in combs if _gk(d, "part_number")
        ]
        exc += (
            (
                f", overlaid with {_gk(comb, 'manufacturer') or ''} dichroic "
                f"beam-combiners ({', '.join(parts)})"
            )
            if parts
            else ""
        )
    if _gk(coup, "manufacturer") or _gk(coup, "fiber"):
        fib = _gk(coup, "fiber")
        exc += (
            f" and coupled into the microscope through a "
            f"{_gk(coup, 'description') or 'single-mode fiber'} "
            f"({_gk(coup, 'manufacturer') or ''}"
            + (f", {fib}" if fib else "")
            + ")"
        )
    p.append(exc + ".")

    # 4) filters
    if fsets:
        cube_bits = []
        for k, v in fsets.items():
            name = _gk(v, "cube") or _gk(v, "set")
            if name:
                cube_bits.append(
                    f"{k} nm: {name}" if str(k).isdigit() else name
                )
        if cube_bits:
            seen = []
            for s in cube_bits:
                if s not in seen:
                    seen.append(s)
            p.append(
                "Excitation and emission were separated with AHF Nikon TIRF "
                "filter cubes (" + "; ".join(seen) + "), each comprising a "
                "laser clean-up filter, a dichroic beam splitter and longpass + "
                "bandpass emission filters."
            )

    # 5) camera / acquisition
    cm, cmod = _gk(cam, "manufacturer"), _gk(cam, "model")
    cam_name = " ".join([x for x in (cm, cmod) if x]) or NS
    ctype = _gk(cam, "type") or "sCMOS"
    px_nm = (
        _gk(obj, "effective_pixel_nm")
        or (str(ac.get("pixelsize_nm")) if ac.get("pixelsize_nm") else None)
        or NS
    )
    expo = ac.get("exposure_ms")
    frames = ac.get("total_frames")
    soft_acq = _gk(sw, "acquisition") or NS
    binning = _gk(cam, "binning")
    acq = f"Images were recorded on a {cam_name} {ctype} camera"
    if px_nm != NS:
        acq += (
            f" ({px_nm} nm effective pixel size"
            + (f" after {binning} binning" if binning else "")
            + ")"
        )
    if expo:
        e = int(expo) if float(expo).is_integer() else expo
        acq += f" at {e} ms per frame"
    if frames:
        acq += f" for {int(frames)} frames"
    acq += f", controlled by {soft_acq}."
    p.append(acq)

    # optional 3D note
    is3d = bool(
        sm.get("3d") or sm.get("mode") == "3D" or ac.get("z_calibration")
    )
    if is3d and (_gk(a3d, "model") or _gk(a3d, "part_number")):
        p.append(
            f"For 3D imaging, astigmatism was introduced with a "
            f"{_gk(a3d, 'manufacturer') or ''} {_gk(a3d, 'model') or ''} module"
            + (
                f" ({_gk(a3d, 'part_number')})"
                if _gk(a3d, "part_number")
                else ""
            )
            + "."
        )

    # 6) localization + QC
    soft_loc = _gk(sw, "localization") or "Picasso"
    picasso_v = _clean(qc.get("software_version"))
    nlocs = me.get("localizations")
    loc = f"Single-molecule localization was performed with {soft_loc}"
    if picasso_v:
        loc += f" ({picasso_v})"
    loc += " using a least-squares 2D Gaussian fit."
    if nlocs:
        loc += (
            f" A total of {int(nlocs):,} localizations were obtained.".replace(
                ",", " "
            )
        )
    p.append(loc)

    frc = me.get("frc_resolution_nm")
    nena = me.get("nena_zoom_nm") or me.get("nena_global_nm")
    qb = []
    if frc is not None:
        qb.append(
            f"the resolution was estimated by Fourier ring correlation "
            f"({frc:.1f} nm)"
        )
    if nena is not None:
        qb.append(
            f"the localization precision by nearest-neighbour analysis "
            f"(NeNA, {nena:.1f} nm)"
        )
    if qb:
        p.append("For quality control, " + " and ".join(qb) + ".")

    # 7) protocol (built-in / Confluence) + deviations, if one was attached.
    proto = qc.get("protocol") or {}
    if isinstance(proto, dict) and (proto.get("name") or proto.get("steps")):
        pname = _clean(proto.get("name"))
        src = _clean(proto.get("source"))
        av = proto.get("actual_values", {}) or {}
        line = "Sample preparation and acquisition followed the "
        line += f"“{pname}” protocol" if pname else "documented protocol"
        if src == "confluence":
            line += " (from Confluence)"
        line += "."
        dev = _clean(av.get("deviations"))
        if dev:
            line += f" Deviations for this measurement: {dev}."
        p.append(line)

    return "\n\n".join(p)

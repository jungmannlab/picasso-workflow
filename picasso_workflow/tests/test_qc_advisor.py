#!/usr/bin/env python
"""
test_qc_advisor.py -- Tier-2 known-answer + property tests for the shared,
LLM-free QC advisor (WP-ADVISOR). Hermetic: no instrument, no registry service
(the cohort stats are synthetic / mocked), no Qt.

Numbers are checked against the LiveLocalization V0.8 reference behaviour on
shared fixtures.
"""

import json
import math

import pytest

from picasso_workflow.qc_advisor import (
    Finding,
    SEV_ORDER,
    center_edge_trend,
    classify_query,
    cohort_stats,
    db_anomalies,
    diagnose,
    filter_keep_mask,
    filter_preview_counts,
    filter_suggestions,
    focus,
    frc_trend,
    materials_and_methods,
    sort_findings,
    suggest_clustering,
    summary,
    write_filter_suggestions_to_qc,
)
from picasso_workflow.qc_advisor.filter_advisor import (
    _min_net_gradient_from_ng,
)
from picasso_workflow.qc_advisor.notifier import (
    NotifierEngine,
    RegistryEvent,
)

import numpy as np

# ── the shared V0.8 demo fixture (imager clearly too high) ────────────────────
OVERCONC = dict(
    nena_zoom_nm=9.4,
    frc_resolution_nm=18.0,
    sbr=2.1,
    photons_per_loc=1800,
    background=650,
    localizations=120000,
    overlap_pct=17.0,
    on_time_frames=3.0,
    off_time_frames=22.0,
    drift_ptp_x_nm=12.0,
    conc_pm=1000,
    power_mw=40,
    exposure_ms=100,
)


# ══════════════════════════════════════════════════════════════════════════════
# diagnose / focus / summary
# ══════════════════════════════════════════════════════════════════════════════
class TestDiagnose:
    def test_findings_are_the_finding_contract(self):
        findings = diagnose(OVERCONC)
        assert findings and all(isinstance(f, Finding) for f in findings)
        f0 = findings[0]
        # the explicit contract WP-GUI consumes
        d = f0.as_dict()
        for key in (
            "metric",
            "severity",
            "message",
            "cause",
            "action",
            "value",
            "source",
            "detail",
        ):
            assert key in d

    def test_overconcentrated_run_known_findings(self):
        by_metric = {f.metric: f for f in diagnose(OVERCONC)}
        # the over-concentration case: poor NeNA (bad), overlap (bad),
        # low photons (bad), elevated background (warn), high duty (warn),
        # borderline SBR (warn) -- matches the V0.8 reference exactly.
        assert by_metric["nena"].severity == "bad"
        assert by_metric["overlap"].severity == "bad"
        assert by_metric["photons_per_loc"].severity == "bad"
        assert by_metric["background"].severity == "warn"
        assert by_metric["duty_cycle"].severity == "warn"
        assert by_metric["sbr"].severity == "warn"
        # the NeNA finding names the low-photon + overlap + duty causes
        assert "low photon count" in by_metric["nena"].cause
        assert "high spot overlap" in by_metric["nena"].cause

    def test_overconc_golden_full_list(self):
        # GOLDEN: lock the EXACT (metric, severity) sequence + count of the
        # composed findings for the overconcentrated fixture against the V0.8
        # reference output, so a dropped/added/re-ordered finding anywhere is
        # caught -- not just the individually-asserted metrics.
        golden = [
            ("nena", "bad"),
            ("overlap", "bad"),
            ("photons_per_loc", "bad"),
            ("background", "warn"),
            ("duty_cycle", "warn"),
            ("sbr", "warn"),
        ]
        got = [(f.metric, f.severity) for f in diagnose(OVERCONC)]
        assert got == golden

    def test_good_run_reports_ok(self):
        good = dict(
            nena_zoom_nm=3.1,
            frc_resolution_nm=6.2,
            sbr=8.0,
            photons_per_loc=9000,
            background=120,
            on_time_frames=2.0,
            off_time_frames=98.0,
        )
        sev = {f.metric: f.severity for f in diagnose(good)}
        assert sev["nena"] == "ok"
        assert sev["frc"] == "ok"
        assert sev["sbr"] == "ok"
        assert sev["photons_per_loc"] == "ok"
        assert sev["duty_cycle"] == "ok"

    def test_frc_vs_nena_undersampling_crosscheck(self):
        # FRC >> 3x NeNA -> flagged as sampling/drift-limited, not precision.
        m = dict(nena_zoom_nm=3.0, frc_resolution_nm=15.0)
        f = {x.metric: x for x in diagnose(m)}
        assert f["frc"].severity == "warn"
        assert "undersampling" in f["frc"].cause

    def test_laser_power_context_cells_vs_origami(self):
        # 45 mW is high for cells (>40) but not origami (<80).
        cells = diagnose(dict(background=500, power_mw=45, sample_type="cell"))
        assert any(x.metric == "laser_power" for x in cells)
        origami = diagnose(
            dict(background=500, power_mw=45, sample_type="origami")
        )
        assert not any(x.metric == "laser_power" for x in origami)

    def test_missing_keys_are_skipped_never_crash(self):
        assert diagnose({}) == []
        assert diagnose(None) == []
        # garbage values are coerced/skipped, not raised
        assert isinstance(diagnose({"nena_zoom_nm": "not a number"}), list)

    def test_findings_sorted_by_severity_then_metric(self):
        findings = diagnose(OVERCONC)
        keys = [f.sort_key for f in findings]
        assert keys == sorted(keys)

    def test_targets_override_thresholds(self):
        # loosening the NeNA-good threshold flips a 5 nm run from warn to ok
        m = dict(nena_zoom_nm=5.0)
        assert diagnose(m)[0].severity == "warn"
        loosened = diagnose(m, targets={"nena_good_nm": 6.0})
        assert loosened[0].severity == "ok"

    def test_focus_returns_only_topic_findings(self):
        nena = focus("nena", OVERCONC)
        allowed = {
            "nena",
            "photons_per_loc",
            "sbr",
            "overlap",
            "duty_cycle",
            "background",
        }
        assert nena and all(f.metric in allowed for f in nena)
        assert not any(f.metric == "frc" for f in nena)

    def test_summary_text_and_top_limit(self):
        assert summary(None) == "No metrics available to assess."
        text = summary(OVERCONC, top=2)
        assert text.count("[") == 2  # exactly two findings rendered

    def test_finding_rejects_bad_severity(self):
        with pytest.raises(ValueError):
            Finding("nena", "catastrophic", "boom")


# ══════════════════════════════════════════════════════════════════════════════
# frc_trend -- the net-new "you can stop" signal
# ══════════════════════════════════════════════════════════════════════════════
class TestFrcTrend:
    def test_insufficient_points(self):
        out = frc_trend([1000, 2000], [20, 18])
        assert out["status"] == "insufficient"

    def test_still_improving(self):
        # a clean power-law decay -> improving, keep acquiring
        frames = [1000, 2000, 4000, 8000, 16000]
        frc = [40.0, 28.3, 20.0, 14.1, 10.0]  # ~N^-0.5
        out = frc_trend(frames, frc)
        assert out["status"] == "improving"
        assert out["gain_if_doubled_pct"] > 3.0
        assert out["exponent_b"] > 0

    def test_plateau_when_flat_and_drift_corrected_and_late(self):
        frames = [1000, 2000, 4000, 8000, 16000, 32000]
        frc = [10.2, 10.1, 10.05, 10.02, 10.01, 10.0]
        out = frc_trend(frames, frc, undrift_done=True, progress_frac=0.9)
        assert out["status"] == "plateau"
        assert "you can stop" in out["action"]

    def test_flat_but_not_yet_drift_corrected_is_plateau_check(self):
        frames = [1000, 2000, 4000, 8000]
        frc = [10.1, 10.05, 10.02, 10.0]
        out = frc_trend(frames, frc, undrift_done=False)
        assert out["status"] == "plateau_check"

    def test_flat_with_default_args_does_not_say_stop(self):
        # DEVIATION from V0.8: with no guard context (default undrift_done=None,
        # progress_frac=None) a flat curve must NOT emit the confident stop
        # signal -- a false stop is a high-cost re-acquisition.
        frames = [1000, 2000, 4000, 8000, 16000, 32000]
        frc = [10.2, 10.1, 10.05, 10.02, 10.01, 10.0]
        out = frc_trend(frames, frc)  # default args
        assert out["status"] == "plateau_unconfirmed"
        assert "stop" not in out["action"].lower() or "only then" in (
            out["action"].lower()
        )
        assert "you can stop" not in out["action"].lower()

    def test_flat_needs_both_guards_for_plateau(self):
        frames = [1000, 2000, 4000, 8000]
        frc = [10.1, 10.05, 10.02, 10.0]
        # undrift confirmed but no progress -> still unconfirmed
        out = frc_trend(frames, frc, undrift_done=True)
        assert out["status"] == "plateau_unconfirmed"
        # both guards -> the confident stop
        out2 = frc_trend(frames, frc, undrift_done=True, progress_frac=0.9)
        assert out2["status"] == "plateau"
        assert "you can stop" in out2["action"]

    def test_flat_but_early_is_plateau_early(self):
        frames = [1000, 2000, 4000, 8000]
        frc = [10.1, 10.05, 10.02, 10.0]
        out = frc_trend(frames, frc, undrift_done=True, progress_frac=0.1)
        assert out["status"] == "plateau_early"

    def test_degrading_when_frc_rising(self):
        frames = [1000, 2000, 4000, 8000, 16000]
        frc = [10.0, 12.0, 15.0, 19.0, 24.0]  # getting worse
        out = frc_trend(frames, frc)
        assert out["status"] == "degrading"
        assert out["exponent_b"] < 0

    def test_target_eta_while_improving(self):
        frames = [1000, 2000, 4000, 8000]
        frc = [40.0, 28.3, 20.0, 14.1]
        out = frc_trend(frames, frc, target_nm=10.0)
        assert out["status"] == "improving"
        assert out.get("eta_frames", 0) > frames[-1]

    def test_empty_inputs_are_insufficient(self):
        # None / empty are iterable-safe -> "insufficient", not a crash.
        assert frc_trend(None, None)["status"] == "insufficient"
        assert frc_trend([], [])["status"] == "insufficient"

    def test_non_iterable_inputs_return_none(self):
        assert frc_trend(object(), object()) is None


# ══════════════════════════════════════════════════════════════════════════════
# center_edge_trend -- photodamage vs illumination profile
# ══════════════════════════════════════════════════════════════════════════════
class TestCenterEdgeTrend:
    def test_declining_centre_is_photodamage(self):
        frames = list(range(1, 13))
        center = [100, 98, 95, 90, 85, 78, 72, 66, 60, 55, 50, 45]
        edge = [100] * 12
        out = center_edge_trend(frames, center, edge)
        assert out["status"] == "declining"
        assert "photodamage" in out["cause"]

    def test_static_dip_is_illumination_profile(self):
        frames = list(range(1, 13))
        center = [70] * 12
        edge = [100] * 12
        out = center_edge_trend(frames, center, edge)
        assert out["status"] == "static_dip"

    def test_uniform(self):
        frames = list(range(1, 13))
        out = center_edge_trend(frames, [100] * 12, [100] * 12)
        assert out["status"] == "uniform"

    def test_insufficient_points(self):
        out = center_edge_trend([1, 2], [100, 100], [100, 100])
        assert out["status"] == "insufficient"

    def test_empty_inputs_are_insufficient(self):
        assert center_edge_trend(None, None, None)["status"] == "insufficient"

    def test_non_iterable_inputs_return_none(self):
        assert center_edge_trend(object(), object(), object()) is None


# ══════════════════════════════════════════════════════════════════════════════
# cohort_stats + db_anomalies -- live vs the user's OWN cohort (registry mocked)
# ══════════════════════════════════════════════════════════════════════════════
class TestCohortAndAnomalies:
    def _mock_cohort_runs(self):
        # a synthetic cohort (what the registry client's cohort(taxon_id) would
        # return): healthy runs at this setup.
        return [
            {
                "photons_per_loc": 8000,
                "background": 120,
                "sbr": 7.0,
                "nena_zoom_nm": 3.2,
                "frc_resolution_nm": 6.5,
            },
            {
                "photons_per_loc": 9000,
                "background": 130,
                "sbr": 7.5,
                "nena_zoom_nm": 3.0,
                "frc_resolution_nm": 6.0,
            },
            {
                "photons_per_loc": 8500,
                "background": 110,
                "sbr": 6.8,
                "nena_zoom_nm": 3.4,
                "frc_resolution_nm": 7.0,
            },
            {
                "photons_per_loc": 7500,
                "background": 140,
                "sbr": 6.5,
                "nena_zoom_nm": 3.6,
                "frc_resolution_nm": 6.8,
            },
            {
                "photons_per_loc": 8200,
                "background": 125,
                "sbr": 7.2,
                "nena_zoom_nm": 3.1,
                "frc_resolution_nm": 6.3,
            },
            {
                "photons_per_loc": 8800,
                "background": 115,
                "sbr": 7.1,
                "nena_zoom_nm": 3.3,
                "frc_resolution_nm": 6.6,
            },
        ]

    def test_cohort_stats_shape(self):
        stats = cohort_stats(self._mock_cohort_runs())
        assert "photons_per_loc" in stats
        st = stats["photons_per_loc"]
        assert set(st) == {"n", "median", "p20", "p80", "min", "max"}
        assert st["n"] == 6
        assert st["p20"] < st["median"] < st["p80"]

    def test_photons_far_below_cohort_flagged(self):
        stats = cohort_stats(self._mock_cohort_runs())
        # this run: photons 3x below usual -> laser/alignment early warning
        findings = db_anomalies({"photons_per_loc": 2500}, stats)
        photons = [f for f in findings if f.metric == "photons_per_loc"]
        assert photons and photons[0].severity in ("warn", "bad")
        assert photons[0].source == "db_anomalies"
        assert "laser" in photons[0].cause

    def test_unusually_good_reported_as_info(self):
        stats = cohort_stats(self._mock_cohort_runs())
        findings = db_anomalies({"photons_per_loc": 20000}, stats)
        photons = [f for f in findings if f.metric == "photons_per_loc"]
        assert photons and photons[0].severity == "info"
        assert "unusually good" in photons[0].message

    def test_in_range_value_not_flagged(self):
        stats = cohort_stats(self._mock_cohort_runs())
        assert db_anomalies({"photons_per_loc": 8300}, stats) == []

    def test_small_cohort_below_min_n_not_flagged(self):
        stats = cohort_stats(self._mock_cohort_runs()[:2])
        # n=2 < min_n=5 -> no anomaly even if the value is extreme
        assert db_anomalies({"photons_per_loc": 100}, stats) == []

    def test_collapsed_spread_cohort_not_over_flagged(self):
        # an all-equal cohort collapses the p20<->p80 band; a deviation must NOT
        # be reported as `bad` off a nonsensical "median X, p20-p80 X-X" band.
        allequal = [{"photons_per_loc": 8000} for _ in range(8)]
        stats = cohort_stats(allequal)
        assert (
            stats["photons_per_loc"]["p20"] == stats["photons_per_loc"]["p80"]
        )
        # even a large nominal deviation is skipped (insufficient spread)
        assert db_anomalies({"photons_per_loc": 2000}, stats) == []

    def test_zero_median_cohort_not_over_flagged(self):
        stats = cohort_stats([{"background": 0.0} for _ in range(8)])
        # a zero-median / zero-spread band must not fire a bad anomaly
        assert db_anomalies({"background": 5.0}, stats) == []

    def test_lower_better_metric_direction(self):
        # a NeNA far ABOVE the cohort (worse for a lower-better metric) is bad.
        stats = cohort_stats(self._mock_cohort_runs())
        findings = db_anomalies({"nena_zoom_nm": 12.0}, stats)
        nena = [f for f in findings if f.metric == "nena_zoom_nm"]
        assert nena and nena[0].severity in ("warn", "bad")

    def test_registry_client_cohort_mock_end_to_end(self):
        # emulate calling the picasso_registry client and bridging to stats.
        class MockRegistryClient:
            def cohort(self, taxon_id, **kw):
                return TestCohortAndAnomalies()._mock_cohort_runs()

        client = MockRegistryClient()
        runs = client.cohort("origami/grid")
        stats = cohort_stats(runs)
        findings = db_anomalies({"background": 400}, stats)
        assert any(f.metric == "background" for f in findings)


# ══════════════════════════════════════════════════════════════════════════════
# filter advisor + min-net-gradient + clustering
# ══════════════════════════════════════════════════════════════════════════════
def _make_locs(n_signal=500, n_noise=200, seed=0):
    rng = np.random.default_rng(seed)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("photons", "f4"),
        ("sx", "f4"),
        ("sy", "f4"),
        ("lpx", "f4"),
        ("lpy", "f4"),
    ]
    n = n_signal + n_noise
    locs = np.zeros(n, dtype=dtype)
    # signal: tight cluster near (30, 30), bright
    locs["x"][:n_signal] = 30 + rng.normal(0, 0.5, n_signal)
    locs["y"][:n_signal] = 30 + rng.normal(0, 0.5, n_signal)
    locs["photons"][:n_signal] = rng.normal(5000, 500, n_signal)
    locs["sx"][:n_signal] = rng.normal(1.2, 0.05, n_signal)
    locs["sy"][:n_signal] = rng.normal(1.2, 0.05, n_signal)
    locs["lpx"][:n_signal] = rng.normal(0.02, 0.003, n_signal)
    locs["lpy"][:n_signal] = rng.normal(0.02, 0.003, n_signal)
    # noise: spread out, dim, distorted PSFs
    locs["x"][n_signal:] = rng.uniform(0, 100, n_noise)
    locs["y"][n_signal:] = rng.uniform(0, 100, n_noise)
    locs["photons"][n_signal:] = rng.normal(500, 200, n_noise)
    locs["sx"][n_signal:] = rng.normal(2.5, 0.5, n_noise)
    locs["sy"][n_signal:] = rng.normal(1.0, 0.3, n_noise)
    locs["lpx"][n_signal:] = rng.normal(0.08, 0.02, n_noise)
    locs["lpy"][n_signal:] = rng.normal(0.08, 0.02, n_noise)
    names = list(locs.dtype.names)
    return locs, names


class TestFilterAdvisor:
    def test_filter_suggestions_from_zoom_roi(self):
        locs, names = _make_locs()
        fs = filter_suggestions(
            locs, names, vp_zoom=[(28, 28), (32, 32)], px=130.0
        )
        assert "photons_min" in fs and "photons_max" in fs
        assert fs["photons_min"] < fs["photons_max"]
        assert "ellipticity_max" in fs
        assert "precision_max_nm" in fs
        assert fs["source"] == "zoom ROI"

    def test_filter_too_few_signal_locs(self):
        locs, names = _make_locs(n_signal=500)
        fs = filter_suggestions(
            locs, names, vp_zoom=[(90, 90), (91, 91)], px=130.0
        )
        assert "error" in fs

    def test_filter_none_without_roi_or_picks(self):
        locs, names = _make_locs()
        assert filter_suggestions(locs, names) is None
        assert filter_suggestions(None, []) is None

    def test_keep_mask_removes_noise_keeps_signal(self):
        locs, names = _make_locs()
        fs = filter_suggestions(
            locs, names, vp_zoom=[(28, 28), (32, 32)], px=130.0
        )
        keep = filter_keep_mask(locs, names, fs, 130.0)
        # most of the bright tight signal survives; most noise is removed
        assert keep[:500].mean() > 0.8
        assert keep[500:].mean() < 0.5

    def test_filter_preview_counts_payload(self):
        locs, names = _make_locs()
        fs = filter_suggestions(
            locs, names, vp_zoom=[(28, 28), (32, 32)], px=130.0
        )
        payload = filter_preview_counts(locs, names, fs, 130.0)
        assert payload["n_total"] == len(locs)
        assert payload["n_kept"] + payload["n_removed"] == payload["n_total"]

    def test_write_filter_suggestions_to_qc(self, tmp_path):
        qc_path = str(tmp_path / "qc.json")
        # start with an existing qc.json holding other content
        with open(qc_path, "w") as fh:
            json.dump({"metrics": {"nena_zoom_nm": 3.1}}, fh)
        fs = {"photons_min": 1000, "photons_max": 9000}
        qc = write_filter_suggestions_to_qc(qc_path, fs)
        assert qc["recommendations"]["filter"] == fs
        # existing content preserved
        assert qc["metrics"]["nena_zoom_nm"] == 3.1
        # persisted to disk
        with open(qc_path) as fh:
            on_disk = json.load(fh)
        assert on_disk["recommendations"]["filter"]["photons_min"] == 1000


class TestMinNetGradient:
    def test_bimodal_valley(self):
        rng = np.random.default_rng(1)
        # two well-separated log-normal populations -> bimodal, cut in valley
        noise = 10 ** rng.normal(2.0, 0.12, 4000)  # ~100
        signal = 10 ** rng.normal(3.5, 0.12, 4000)  # ~3000
        ng = np.concatenate([noise, signal])
        out = _min_net_gradient_from_ng(ng, low=100, min_candidates=150)
        assert out["regime"] == "bimodal"
        # the recommendation sits between the two peaks
        assert (
            out["noise_peak_ng"] < out["recommended"] < out["signal_peak_ng"]
        )

    def test_noise_tail_shoulder(self):
        rng = np.random.default_rng(2)
        # one big noise peak + a long bright tail -> noise_tail regime
        noise = 10 ** rng.normal(2.0, 0.1, 5000)
        tail = 10 ** rng.uniform(2.5, 4.0, 300)
        ng = np.concatenate([noise, tail])
        out = _min_net_gradient_from_ng(ng, low=100, min_candidates=150)
        assert out["regime"] == "noise_tail"
        assert out["recommended"] > out["median_ng"]

    def test_too_few_candidates(self):
        out = _min_net_gradient_from_ng(
            np.array([100.0, 200.0]), low=100, min_candidates=150
        )
        assert "error" in out


class TestClustering:
    def test_eps_scales_with_nena(self):
        rec = suggest_clustering(nena_nm=4.0)
        assert rec["primary"]["parameters"]["eps_nm"] == 10.0  # 2.5 x 4
        assert rec["primary"]["algorithm"] == "DBSCAN"

    def test_eps_guidance_without_nena(self):
        rec = suggest_clustering()
        assert isinstance(rec["primary"]["parameters"]["eps_nm"], str)
        assert rec["alternatives"]


# ══════════════════════════════════════════════════════════════════════════════
# materials & methods -- deterministic, degrades to "not specified"
# ══════════════════════════════════════════════════════════════════════════════
class TestMaterialsMethods:
    def test_renders_with_no_hardware_profile(self):
        qc = {
            "software_version": "V0.8",
            "measurement": {"microscope": "Voyager"},
            "sample": {
                "sample_type": "DNA-Origami",
                "imager": "R1",
                "dye": "Cy3B",
                "conc_pm": 500,
                "buffer": "buffer C+",
                "target": "20 nm grid",
                "power_mw": 40,
            },
            "acquisition": {
                "exposure_ms": 100,
                "total_frames": 30000,
                "pixelsize_nm": 130,
                "imager_sequence": "R1",
            },
            "metrics": {
                "frc_resolution_nm": 6.2,
                "nena_zoom_nm": 3.1,
                "localizations": 1240000,
            },
        }
        text = materials_and_methods(qc)
        assert "DNA-Origami" in text
        assert "R1 (Cy3B)" in text
        assert "500 pM" in text
        assert "30000 frames" in text
        assert "Fourier ring correlation (6.2 nm)" in text
        assert "NeNA, 3.1 nm" in text
        # hardware fields degrade to "not specified" (no profile passed)
        assert "not specified" in text

    def test_empty_qc_still_renders(self):
        text = materials_and_methods({})
        assert isinstance(text, str) and text
        assert "not specified" in text

    def test_loads_hardware_profile_yaml(self, tmp_path):
        cfg = tmp_path / "microscope_config.yaml"
        cfg.write_text(
            "default_setup: Voyager\n"
            "setups:\n"
            "  Voyager:\n"
            "    stand: Nikon Ti2\n"
            "    illumination: TIRF\n"
            "    objective:\n"
            "      manufacturer: Nikon\n"
            "      model: CFI Apo\n"
            "      na: 1.49\n"
        )
        qc = {"measurement": {"microscope": "Voyager"}, "sample": {}}
        text = materials_and_methods(qc, config_path=str(cfg))
        assert "Nikon Ti2" in text
        assert "NA 1.49" in text
        assert "TIRF" in text

    def test_protocol_deviations_rendered(self):
        qc = {
            "sample": {"sample_type": "cells"},
            "protocol": {
                "name": "Standard NPC",
                "source": "confluence",
                "actual_values": {"deviations": "imager conc halved"},
            },
        }
        text = materials_and_methods(qc)
        assert "Standard NPC" in text
        assert "from Confluence" in text
        assert "imager conc halved" in text


# ══════════════════════════════════════════════════════════════════════════════
# notifier -- C35 deterministic lifecycle push + query surface
# ══════════════════════════════════════════════════════════════════════════════
class TestNotifier:
    def test_full_lifecycle_stream_pushes_under_one_thread(self):
        sent = []
        eng = NotifierEngine(sink=sent.append)
        eng.subscribe("alice", "run1", min_severity="info")
        stream = [
            RegistryEvent("run1", "acquisition_started"),
            RegistryEvent(
                "run1",
                "qc_update",
                seq=1,
                metrics={"nena_zoom_nm": 3.2, "frc_resolution_nm": 8.0},
            ),
            RegistryEvent("run1", "acquisition_done"),
            RegistryEvent("run1", "handoff"),
            RegistryEvent("run1", "analysis_started"),
            RegistryEvent(
                "run1", "drift_converged", detail={"drift_ptp_nm": 5.0}
            ),
            RegistryEvent(
                "run1",
                "analysis_done",
                metrics={"nena_zoom_nm": 3.0, "frc_resolution_nm": 6.5},
                detail={"thumbnail": "art://run1/recon.png"},
            ),
        ]
        for ev in stream:
            eng.handle_event(ev)
        # every message rode the ONE per-run thread
        assert sent
        assert {n.thread_key for n in sent} == {"run::run1"}
        # both live and cluster phases are represented
        kinds = {n.kind for n in sent}
        assert "acquisition_started" in kinds  # live
        assert "analysis_done" in kinds  # cluster

    def test_below_threshold_event_suppressed(self):
        eng = NotifierEngine()
        # bob only wants warnings and worse
        eng.subscribe("bob", "run2", min_severity="warn")
        info = eng.handle_event(RegistryEvent("run2", "acquisition_started"))
        assert info == []  # info suppressed for a warn-floor subscriber
        # but a failure always gets through regardless of threshold
        fail = eng.handle_event(
            RegistryEvent("run2", "analysis_failed", detail={"error": "OOM"})
        )
        assert fail and fail[0].severity == "bad"
        assert "OOM" in fail[0].message

    def test_unsubscribed_user_gets_nothing_for_routine_event(self):
        sent = []
        eng = NotifierEngine(sink=sent.append)
        # a routine info event with no subscribers -> nobody, no broadcast
        out = eng.handle_event(RegistryEvent("run3", "acquisition_started"))
        assert out == [] and sent == []

    def test_critical_with_no_subscribers_hits_fallback_sink(self):
        # a critical (analysis_failed) for a run NOBODY subscribed to must not be
        # silently dropped -- it broadcasts to the fallback sink exactly once.
        broadcast = []
        eng = NotifierEngine(fallback_sink=broadcast.append)
        out = eng.handle_event(
            RegistryEvent(
                "runX", "analysis_failed", detail={"error": "node died"}
            )
        )
        assert len(out) == 1
        assert out[0].user == NotifierEngine.FALLBACK_USER
        assert out[0].severity == "bad"
        assert len(broadcast) == 1
        # replay does not re-broadcast (deduped under the sentinel user)
        again = eng.handle_event(
            RegistryEvent(
                "runX", "analysis_failed", detail={"error": "node died"}
            )
        )
        assert again == [] and len(broadcast) == 1

    def test_fallback_defaults_to_main_sink(self):
        sent = []
        eng = NotifierEngine(sink=sent.append)  # no explicit fallback
        eng.handle_event(RegistryEvent("runY", "analysis_failed"))
        assert len(sent) == 1  # main sink used as the fallback

    def test_subscribed_critical_does_not_also_broadcast(self):
        sent, broadcast = [], []
        eng = NotifierEngine(sink=sent.append, fallback_sink=broadcast.append)
        eng.subscribe("alice", "runZ")
        out = eng.handle_event(RegistryEvent("runZ", "analysis_failed"))
        # delivered to the subscriber only; no redundant broadcast
        assert len(out) == 1 and out[0].user == "alice"
        assert broadcast == []

    def test_seqless_qc_updates_worsening_both_delivered(self):
        # two seq-less qc_updates whose picture worsens (healthy -> bad) must
        # BOTH be delivered; the 2nd must not be deduped away on `kind` alone.
        sent = []
        eng = NotifierEngine(sink=sent.append)
        eng.subscribe("alice", "runW", min_severity="info")
        n1 = eng.handle_event(
            RegistryEvent("runW", "qc_update", metrics={"nena_zoom_nm": 3.0})
        )
        n2 = eng.handle_event(
            RegistryEvent(
                "runW",
                "qc_update",
                metrics={"nena_zoom_nm": 9.0, "photons_per_loc": 1500},
            )
        )
        assert len(n1) == 1 and len(n2) == 1
        # the escalation is visible
        assert n2[0].severity in ("warn", "bad")

    def test_seqless_qc_update_identical_repeat_deduped(self):
        eng = NotifierEngine()
        eng.subscribe("alice", "runV")
        m = {"nena_zoom_nm": 3.0}
        first = eng.handle_event(RegistryEvent("runV", "qc_update", metrics=m))
        repeat = eng.handle_event(
            RegistryEvent("runV", "qc_update", metrics=dict(m))
        )
        assert len(first) == 1 and repeat == []

    def test_dedup_same_event_notifies_once(self):
        eng = NotifierEngine()
        eng.subscribe("alice", "run4")
        ev = RegistryEvent("run4", "analysis_done")
        first = eng.handle_event(ev)
        second = eng.handle_event(ev)  # replay
        assert len(first) == 1
        assert second == []  # deduped

    def test_qc_update_dedups_per_sequence(self):
        eng = NotifierEngine()
        eng.subscribe("alice", "run5")
        n1 = eng.handle_event(
            RegistryEvent(
                "run5", "qc_update", seq=1, metrics={"nena_zoom_nm": 3.2}
            )
        )
        n1b = eng.handle_event(
            RegistryEvent(
                "run5", "qc_update", seq=1, metrics={"nena_zoom_nm": 3.2}
            )
        )
        n2 = eng.handle_event(
            RegistryEvent(
                "run5", "qc_update", seq=2, metrics={"nena_zoom_nm": 3.3}
            )
        )
        assert len(n1) == 1 and n1b == [] and len(n2) == 1

    def test_bad_metrics_escalate_qc_update_severity(self):
        eng = NotifierEngine()
        eng.subscribe("bob", "run6", min_severity="warn")
        # a poor-NeNA / low-photons live update escalates info -> bad, so it
        # reaches a warn-floor subscriber.
        out = eng.handle_event(
            RegistryEvent(
                "run6",
                "qc_update",
                seq=1,
                metrics={"nena_zoom_nm": 9.4, "photons_per_loc": 1500},
            )
        )
        assert out and out[0].severity in ("warn", "bad")

    def test_status_query_read_only(self):
        eng = NotifierEngine()
        eng.subscribe("alice", "run7")
        eng.handle_event(RegistryEvent("run7", "analysis_started"))
        ans = eng.answer_query("run7", "how's my analysis?")
        assert ans is not None and "run7" in ans
        assert "cluster" in ans

    def test_quality_query_returns_registry_derived_findings(self):
        eng = NotifierEngine()
        eng.handle_event(
            RegistryEvent(
                "run8",
                "analysis_done",
                metrics={"nena_zoom_nm": 9.4, "photons_per_loc": 1800},
            )
        )
        ans = eng.answer_query("run8", "quality")
        assert ans and "Quality for run run8" in ans
        assert "NeNA" in ans

    def test_query_no_status_yet(self):
        eng = NotifierEngine()
        assert "No status" in eng.answer_query("unknown", "status")

    def test_non_query_text_returns_none(self):
        eng = NotifierEngine()
        eng.handle_event(RegistryEvent("run9", "analysis_done"))
        assert eng.answer_query("run9", "good morning") is None

    def test_classify_query_rules(self):
        assert classify_query("quality") == "quality"
        assert classify_query("show me the QC") == "quality"
        assert classify_query("what's the status?") == "status"
        assert classify_query("how's my run doing") == "status"
        # quality wins when both appear
        assert classify_query("status and quality") == "quality"
        assert classify_query("hello there") is None

    def test_unknown_event_kind_raises(self):
        eng = NotifierEngine()
        with pytest.raises(ValueError):
            eng.handle_event(RegistryEvent("r", "teleport"))


# ══════════════════════════════════════════════════════════════════════════════
# property tests
# ══════════════════════════════════════════════════════════════════════════════
class TestProperties:
    def test_sort_findings_is_severity_monotonic(self):
        raw = [
            Finding("a", "ok", "ok"),
            Finding("b", "bad", "bad"),
            Finding("c", "info", "info"),
            Finding("d", "warn", "warn"),
        ]
        ordered = sort_findings(raw)
        sevs = [SEV_ORDER[f.severity] for f in ordered]
        assert sevs == sorted(sevs)
        assert ordered[0].severity == "bad"

    def test_diagnose_never_crashes_on_partial_metrics(self):
        keys = [
            "nena_zoom_nm",
            "frc_resolution_nm",
            "sbr",
            "photons_per_loc",
            "background",
            "overlap_pct",
            "drift_ptp_x_nm",
        ]
        rng = np.random.default_rng(3)
        for _ in range(50):
            k = rng.integers(0, len(keys) + 1)
            chosen = rng.choice(keys, size=k, replace=False)
            m = {c: float(rng.uniform(0, 100)) for c in chosen}
            assert isinstance(diagnose(m), list)

    def test_frc_trend_monotone_gain_matches_status(self):
        # a strictly improving series never reports plateau/degrading
        frames = [2**i * 1000 for i in range(6)]
        frc = [50.0 * (f / 1000.0) ** -0.5 for f in frames]
        out = frc_trend(frames, frc)
        assert out["status"] == "improving"
        assert math.isfinite(out["exponent_b"])

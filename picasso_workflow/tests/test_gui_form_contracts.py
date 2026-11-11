#!/usr/bin/env python
"""Contract tests: GUI parameter forms must match ``analyse.py`` reads.

The hand-coded ``ModuleDescriptor`` parameter forms in ``picasso_workflow.gui``
drifted from what ``AutoPicasso.<module>`` in ``analyse.py`` actually reads
(dead fields, missing ``[]``-accessed params, misnamed and empty-``{}`` stub
keys). These tests pin each fixed form to the real analyse param contract so the
drift cannot silently return.

For each module, we assert the required analyse params are present and the dead
keys we removed are absent; for the fully-pinned forms we assert the exact key
set. Model: ``test_fit_csr_form_matches_analyse`` in ``test_gui_param_flags``.

Importing ``picasso_workflow.gui`` pulls in PyQt6, so the whole module is
skipped where that (or the GUI import) is unavailable.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

try:
    from picasso_workflow import gui  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"gui import failed: {exc}", allow_module_level=True)


def _keys(module_name):
    descriptor = gui.ModuleDescriptor()
    params_spec, _ = getattr(descriptor, module_name)()
    return set(params_spec)


# ---------------------------------------------------------------------------
# P0 — KeyError hazards
# ---------------------------------------------------------------------------


def test_find_cluster_motifs_form_matches_analyse():
    """analyse.find_cluster_motifs: cellfraction_threshold is required."""
    keys = _keys("find_cluster_motifs")
    required = {
        "fp_workflows",
        "report_names",
        "swkfl_dbscan_molint_key",
        "swkfl_CSR_sim_in_mask_key",
        "population_threshold",
        "ttest_pvalue_max",
        "cellfraction_threshold",
    }
    assert required <= keys
    # channel_colors is commented out in analyse (dead); motif_size and
    # similarity_threshold were never read.
    assert not keys & {"motif_size", "similarity_threshold", "channel_colors"}
    assert keys == required


def test_dbscan_molint_form_matches_analyse():
    """analyse.dbscan_molint reads only these 7 keys (all []-accessed)."""
    keys = _keys("dbscan_molint")
    required = {
        "fp_channel_map",
        "epsilon_nm",
        "minpts",
        "sigma_linker",
        "fp_merge_mask",
        "thresh_type",
        "cell_name",
    }
    assert not keys & {"radius", "min_samples", "interaction_threshold"}
    assert keys == required


# ---------------------------------------------------------------------------
# P1 — misnamed / migration renames and auto-stubbed forms
# ---------------------------------------------------------------------------


def test_ripleysk2_form_matches_analyse():
    """analyse.ripleysk2: metric is required; renamed/added FRC params."""
    keys = _keys("ripleysk2")
    assert "metric" in keys
    # Added/renamed real params.
    assert {
        "fraction_exclude_self",
        "normalize_to_bulkfraction",
        "aggfun",
    } <= keys
    # Invented / dead keys removed.
    assert not keys & {"algorithm", "mask_pixel_size", "fraction_exclude"}
    allowed = {
        "metric",
        "ripleys_n_random_controls",
        "ripleys_rmax",
        "ripleys_dr",
        "radii",
        "ripleys_threshold",
        "area",
        "fp_mask",
        "controltype",
        "randomization_radius",
        "shuffle_self",
        "relocate_self",
        "fraction_exclude_self",
        "normalize_to_bulkfraction",
        "significance_threshold",
        "normalization",
        "aggfun",
        "edge_correction",
        "showControlEnvelope",
    }
    assert keys == allowed


def test_link_locs_form_matches_analyse():
    """analyse.link_locs reads exactly d_max and tolerance."""
    keys = _keys("link_locs")
    assert not keys & {"max_distance", "max_frame_gap", "linking_algorithm"}
    assert keys == {"d_max", "tolerance"}


def test_resolution_analysis_form_matches_analyse():
    """analyse.resolution_analysis reads exactly these 6 optional keys."""
    keys = _keys("resolution_analysis")
    assert not keys & {"xlabel", "ylabel"}
    assert keys == {
        "delta_r",
        "r_max",
        "batch_size",
        "n_processes",
        "use_chunking",
        "use_sparse",
    }


def test_resolution_frc_spatial_form_matches_analyse():
    """analyse.resolution_frc_spatial reads exactly these 8 optional keys."""
    keys = _keys("resolution_frc_spatial")
    assert not keys & {"xlabel", "ylabel"}
    assert keys == {
        "pixelsize_render",
        "smoothing_sigma",
        "threshold",
        "region_size",
        "min_locs_per_region",
        "max_frc_range_nm",
        "n_processes",
        "smoothing_window",
    }


def test_create_mask_form_matches_analyse():
    """analyse.create_mask reads exactly these 7 required keys."""
    keys = _keys("create_mask")
    assert not keys & {"threshold", "smoothing_radius"}
    assert keys == {
        "fp_channel_map",
        "fp_combined_locs",
        "margin",
        "binsize",
        "sigma_mask_blur",
        "mask_resolution",
        "combine_col",
    }


def test_ripleysk_average2_form_matches_analyse():
    """analyse.ripleysk_average2 reads only these 3 keys from parameters.

    metric/controltype/ripleys_threshold are loaded from the sub-workflows'
    parameters, not from this module's own parameters.
    """
    keys = _keys("ripleysk_average2")
    assert not keys & {
        "statistical_method",
        "metric",
        "controltype",
        "randomization_radius",
    }
    assert keys == {"fp_workflows", "report_names", "swkfl_ripleysk_key"}


# ---------------------------------------------------------------------------
# P2 — dead-key removals / missing additions
# ---------------------------------------------------------------------------


def test_manual_form_matches_analyse():
    """analyse.manual reads prompt and filename (both []-accessed)."""
    keys = _keys("manual")
    assert not keys & {"message", "wait_for_input"}
    assert keys == {"prompt", "filename"}


def test_protein_interactions_form_matches_analyse():
    """analyse.protein_interactions reads these 10 required keys."""
    keys = _keys("protein_interactions")
    assert not keys & {"interaction_radius", "confidence_level"}
    assert keys == {
        "channel_map",
        "labeling_efficiency",
        "labeling_uncertainty",
        "n_simulate",
        "density",
        "nn_nth",
        "structure_distance",
        "res_factor",
        "sim_repeats",
        "interaction_pairs",
    }


def test_labeling_efficiency_analysis_form_matches_analyse():
    """n_simulate is no longer read; bootstrap is a real optional toggle."""
    keys = _keys("labeling_efficiency_analysis")
    assert "n_simulate" not in keys
    assert "bootstrap" in keys


def test_find_gold_form_matches_analyse():
    """analyse.find_gold reads remove_gold/diameter/std_range/mean_rmsd."""
    keys = _keys("find_gold")
    assert not keys & {"nlocs_threshold", "rms_threshold", "frame_window"}
    assert keys == {"remove_gold", "diameter", "std_range", "mean_rmsd"}


def test_find_similar_form_matches_analyse():
    """analyse.find_similar: dead reference/metric/tolerance removed."""
    keys = _keys("find_similar")
    assert not keys & {
        "reference_structure",
        "similarity_metric",
        "tolerance",
    }
    assert keys == {
        "diameter",
        "min_n_locs_per_frame",
        "max_n_locs_per_frame",
        "min_rmsd",
        "max_rmsd",
        "n_plot_structures",
        "display_pixelsize",
    }


def test_find_structures_form_matches_analyse():
    """analyse.find_structures: dead cluster_method/n_structures removed."""
    keys = _keys("find_structures")
    assert not keys & {"cluster_method", "n_structures"}
    assert keys == {
        "diameter",
        "min_n_locs_per_frame",
        "n_plot_structures",
        "display_pixelsize",
        "xi",
        "min_cluster_size",
    }


def test_interaction_graph_form_matches_analyse():
    """analyse.interaction_graph reads these 7 required keys."""
    keys = _keys("interaction_graph")
    assert not keys & {"graph_type", "edge_threshold", "fp_density"}
    assert keys == {
        "fp_workflows",
        "report_names",
        "swkfl_protint_key",
        "fp_ripleys_meanvals",
        "node_factor",
        "edge_factor",
        "channel_colors",
    }


def test_pick_origami_form_adds_missing_keys():
    """pick_origami was missing contour_threshold and n_plot_columns."""
    keys = _keys("pick_origami")
    assert {"contour_threshold", "n_plot_columns"} <= keys


def test_nneighbor_form_removes_dead_save_data():
    """analyse.nneighbor does not read save_data."""
    keys = _keys("nneighbor")
    assert "save_data" not in keys
    assert {"dims", "nth_NN", "nth_rdf"} <= keys


def test_create_mask2_form_removes_invented_keys():
    """create_mask2 had invented algorithm (required!) and parameters_dict."""
    keys = _keys("create_mask2")
    assert not keys & {"algorithm", "parameters_dict"}
    assert {"binsize", "blursize", "threshold", "mask_pixel_size"} <= keys


def test_pairwise_module_executor_form_matches_analyse():
    """analyse.pairwise_module_executor: dead sub_module* removed."""
    keys = _keys("pairwise_module_executor")
    assert not keys & {"sub_module", "sub_module_params"}
    assert {
        "module_name",
        "module_kwargs",
        "param_target1",
        "param_target2",
        "xlabel",
        "ylabel",
    } <= keys

#!/usr/bin/env python
"""Regression tests for the SPINNA batch wrapper.

Covers the result-directory naming fix: picasso writes the batch results to
``<basename>__fitting_results`` (DOUBLE underscore); the wrapper must
reconstruct that exact path (a single underscore points at a non-existent
directory), so the reconstructed ``result_dir`` and ``fp_summary`` contain
``__fitting_results``.
"""

import os
from unittest.mock import patch

from picasso_workflow.outpost.spinna import wrappers


@patch("picasso_workflow.outpost.spinna.wrappers.get_spinna_batch_analysis")
def test_spinna_batch_result_dir_double_underscore(mock_get_batch, tmp_path):
    """spinna_batch builds ``<basename>__fitting_results`` (2 underscores)."""
    # picasso's batch-analysis callable is a no-op here; we only check the
    # path reconstruction around it.
    mock_get_batch.return_value = lambda fp: None

    cfg_fp = os.path.join(str(tmp_path), "myparams.csv")
    with open(cfg_fp, "w") as f:
        f.write("col\n1\n")

    result_dir, fp_summary, fp_fig = wrappers.spinna_batch(cfg_fp)

    assert result_dir.endswith("__fitting_results")
    assert os.path.basename(result_dir) == "myparams__fitting_results"
    assert "__fitting_results" in fp_summary
    # Guard against a regression to the single-underscore directory name.
    assert "myparams_fitting_results" not in os.path.basename(result_dir)
    assert fp_summary == os.path.join(result_dir, "summary_results.csv")
    assert fp_fig == []

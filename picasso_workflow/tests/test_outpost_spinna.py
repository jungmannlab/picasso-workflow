#!/usr/bin/env python
"""Regression tests for the SPINNA batch wrapper.

picasso 1.0's public ``spinna.batch_analysis`` returns the result folder (and
the summary DataFrame), so the wrapper surfaces that folder's
``summary_results.csv`` and ``*_NND_*.png`` paths rather than reconstructing
the directory name.
"""

import os
from unittest.mock import patch

from picasso_workflow.outpost.spinna import wrappers


@patch("picasso_workflow.outpost.spinna.wrappers.spinna.batch_analysis")
def test_spinna_batch_surfaces_result_dir_paths(mock_batch, tmp_path):
    """spinna_batch returns batch_analysis's folder + the summary/fig paths."""
    result_dir = os.path.join(str(tmp_path), "myparams__fitting_results")
    os.makedirs(result_dir, exist_ok=True)
    # picasso 1.0 batch_analysis returns (result_dir, summary_df).
    mock_batch.return_value = (result_dir, None)

    cfg_fp = os.path.join(str(tmp_path), "myparams.csv")
    rdir, fp_summary, fp_fig = wrappers.spinna_batch(cfg_fp)

    mock_batch.assert_called_once_with(cfg_fp)
    assert rdir == result_dir
    assert fp_summary == os.path.join(result_dir, "summary_results.csv")
    # No NND figures in the empty folder.
    assert fp_fig == []

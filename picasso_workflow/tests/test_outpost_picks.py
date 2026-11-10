"""Regression tests for ``outpost/picks/picks.py``."""

import numpy as np

from picasso_workflow.outpost.picks.picks import _get_block_locs_at


def test_get_block_locs_at_includes_block_row_col_zero():
    """The retrieval scan must include block row/column 0.

    ``_get_block_locs_at`` mirrors picasso's inclusive ``0 <= k < K`` retrieval
    variant; the previous strict ``0 < k < K`` (copied from the count-only
    ``_n_block_locs_at``) skipped the first block row and column entirely, so
    picks whose centre landed there lost their own block.
    """
    # One localization living in block (row=0, col=0).
    locs_xy = np.array([[5.0], [5.0]])  # shape (2, 1): x row, y row
    K = L = 2
    block_starts = np.zeros((K, L), dtype=np.uint32)
    block_ends = np.zeros((K, L), dtype=np.uint32)
    block_ends[0, 0] = 1  # block (0, 0) owns locs indices [0, 1)

    result = _get_block_locs_at(0, 0, locs_xy, block_starts, block_ends, K, L)

    # The point in block (0, 0) is retrieved; strict 0 < k would have dropped
    # it (and left `indices` unset).
    assert result.shape[1] == 1
    np.testing.assert_array_equal(result[:, 0], [5.0, 5.0])

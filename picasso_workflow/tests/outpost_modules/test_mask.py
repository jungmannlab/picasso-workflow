#!/usr/bin/env python
"""
Module Name: test_mask.py
Author: Heinrich Grabmayr
Initial Date: Jan 22, 2025
Description: Test the module analyse.py
"""

import logging

import numpy as np
import pytest

from picasso_workflow.outpost.masking import mask
from picasso import io
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# The morphology now runs on picasso.masking (label_components / fill_holes /
# erode / dilate via the boundary); importorskip keeps the module collectable
# where picassosr is absent, matching the other outpost tests.
picasso_masking = pytest.importorskip(
    "picasso.masking",
    reason="picassosr required for the CellMask morphology tests",
)


def _cellmask_with_components():
    """Build a CellMask whose binary mask has three separate components
    whose area rank deliberately differs from their row-major scan order,
    so a label-id-based selection (the previous bug) cannot pass by luck.

    Layout (components are orthogonally separated, so picasso's 8-connected
    ``label_components`` keeps them distinct):
        comp A : area 12   (LARGEST)
        comp B : area  2    (smallest)
        comp C : area  9    (medium)

    Area rank therefore is A (0th) > C (1st) > B (2nd). picasso's
    ``label_components`` labels components largest-first (A=1, C=2, B=3),
    but ``filter_mask`` recomputes the ranking from ``np.unique`` so the
    selection is label-id-independent.
    """
    binary = np.zeros((10, 10), dtype=bool)
    comp_a = (slice(0, 2), slice(0, 6))  # 2 * 6 = 12
    comp_b = (slice(0, 1), slice(8, 10))  # 1 * 2 = 2
    comp_c = (slice(4, 7), slice(0, 3))  # 3 * 3 = 9
    binary[comp_a] = True
    binary[comp_b] = True
    binary[comp_c] = True

    cell_mask = mask.CellMask()
    cell_mask._binary_mask = binary
    # _recalc_density_mask_from_binary (called by filter_mask) needs a
    # finite initial density of the same shape.
    cell_mask._initial_density = np.ones_like(binary, dtype=np.float64)
    cell_mask._upsample = 10
    return cell_mask, {"a": comp_a, "b": comp_b, "c": comp_c}


def test_filter_mask_selects_nth_largest_by_area():
    """filter_mask(nth_largest=k) must select the kth-largest component
    by area (0 = largest), not by connected-component label id."""
    expected_by_rank = ["a", "c", "b"]  # largest, second, third by area
    for nth, key in enumerate(expected_by_rank):
        cell_mask, comps = _cellmask_with_components()
        cell_mask.filter_mask(nth_largest=nth, fill_holes=False)

        expected = np.zeros((10, 10), dtype=bool)
        expected[comps[key]] = True
        assert np.array_equal(
            cell_mask._binary_mask, expected
        ), f"nth_largest={nth} selected the wrong component"


def test_filter_mask_default_selects_largest():
    """The default (nth_largest=0) selects the single largest component."""
    cell_mask, comps = _cellmask_with_components()
    cell_mask.filter_mask(fill_holes=False)

    expected = np.zeros((10, 10), dtype=bool)
    expected[comps["a"]] = True
    assert np.array_equal(cell_mask._binary_mask, expected)


def _binary_cellmask(binary, upsample=10):
    """Wrap a given binary mask in a CellMask with a finite density."""
    cell_mask = mask.CellMask()
    cell_mask._binary_mask = binary.astype(np.bool_)
    cell_mask._initial_density = np.ones_like(binary, dtype=np.float64)
    cell_mask._upsample = upsample
    return cell_mask


def test_erode_matches_picasso_masking():
    """CellMask.erode routes through picasso.masking.erode (true-Euclidean,
    nm-accurate), so the result matches a direct picasso erode.

    An interior 7x7 block (not touching the array border) eroded by 10 nm at
    10 nm/px loses its one-pixel rim -> a 5x5 block (25 pixels).
    """
    upsample = 10
    binary = np.zeros((11, 11), dtype=bool)
    binary[2:9, 2:9] = True  # interior 7x7 block, 49 px

    cell_mask = _binary_cellmask(binary, upsample=upsample)
    cell_mask.erode(10)

    expected = picasso_masking.erode(binary, 10, pixel_size=upsample).astype(
        np.bool_
    )
    assert np.array_equal(cell_mask.binary_mask, expected)
    assert cell_mask.binary_mask.sum() == 25  # 5x5, picasso behaviour


def test_erode_treats_array_border_as_not_a_mask_edge():
    """picasso erode does NOT chamfer inward from the array border: a mask
    filling the whole array erodes nothing (behaviour change vs scipy, which
    eroded inward from border_value=0)."""
    upsample = 10
    binary = np.ones((7, 7), dtype=bool)  # fills the whole array

    cell_mask = _binary_cellmask(binary, upsample=upsample)
    cell_mask.erode(10)

    assert cell_mask.binary_mask.sum() == binary.sum()  # unchanged (49)


def test_erode_border_is_edge_shrinks_at_the_fov_border():
    """``border_is_edge=True`` treats the field-of-view border as a mask edge.

    A mask touching the array border is shrunk there too (what Ripley's edge
    correction needs, and what scipy did before the picasso-1.0 migration),
    while it stays a no-op for masks that do not reach the border.
    """
    upsample = 10

    # Fills the whole array -> with the flag, the 1-px border rim is removed
    # (5x5 = 25 px), unlike the default which leaves it unchanged (49).
    binary = np.ones((7, 7), dtype=bool)
    cell_mask = _binary_cellmask(binary, upsample=upsample)
    cell_mask.erode(10, border_is_edge=True)
    assert cell_mask.binary_mask.sum() == 25
    assert not cell_mask.binary_mask[0, :].any()  # top border row cleared

    # Interior mask (not touching the border): identical with or without it.
    interior = np.zeros((11, 11), dtype=bool)
    interior[2:9, 2:9] = True
    m_flag = _binary_cellmask(interior.copy(), upsample=upsample)
    m_plain = _binary_cellmask(interior.copy(), upsample=upsample)
    m_flag.erode(10, border_is_edge=True)
    m_plain.erode(10)
    assert np.array_equal(m_flag.binary_mask, m_plain.binary_mask)


def test_dilate_matches_picasso_masking():
    """CellMask.dilate routes through picasso.masking.dilate (true-Euclidean,
    nm-accurate), so the result matches a direct picasso dilate."""
    upsample = 10
    binary = np.zeros((15, 15), dtype=bool)
    binary[6:9, 6:9] = True  # interior 3x3 block, 9 px

    cell_mask = _binary_cellmask(binary, upsample=upsample)
    cell_mask.dilate(10)

    expected = picasso_masking.dilate(binary, 10, pixel_size=upsample).astype(
        np.bool_
    )
    assert np.array_equal(cell_mask.binary_mask, expected)
    assert cell_mask.binary_mask.sum() == 21  # picasso Euclidean 10 nm grow


def test_filter_mask_merges_diagonal_components_8connected():
    """picasso label_components is 8-connected: diagonal-touching pixels are
    one region (behaviour change vs scipy's 4-connected default)."""
    binary = np.zeros((6, 6), dtype=bool)
    binary[1, 1] = True
    binary[2, 2] = True  # diagonal neighbour of (1, 1)

    cell_mask = _binary_cellmask(binary, upsample=10)
    cell_mask.filter_mask(nth_largest=0, fill_holes=False)

    # 8-connectivity keeps both diagonal pixels as the single component.
    assert cell_mask.binary_mask.sum() == 2
    assert np.array_equal(cell_mask.binary_mask, binary)


def test_filter_mask_fills_holes_via_picasso():
    """filter_mask(fill_holes=True) fills an enclosed hole through
    picasso.masking.fill_holes."""
    binary = np.zeros((7, 7), dtype=bool)
    binary[1:6, 1:6] = True
    binary[3, 3] = False  # one enclosed hole

    cell_mask = _binary_cellmask(binary, upsample=10)
    cell_mask.filter_mask(nth_largest=0, fill_holes=True)

    expected = picasso_masking.fill_holes(binary).astype(np.bool_)
    assert np.array_equal(cell_mask.binary_mask, expected)
    assert cell_mask.binary_mask.sum() == 25  # hole filled -> full 5x5


def _uniform_cellmask(shape, total_count, upsample=10):
    """Build a CellMask with an all-True binary mask whose initial density
    holds a uniform, count-conserving per-pixel localization density.

    ``from_mol_coords`` divides the zoomed density by ``factor**2`` so the
    summed density conserves the localization count; ``_initial_density`` is
    therefore in units of localizations-per-mask-pixel. Replicate that here
    directly (no Otsu / zoom needed) by distributing ``total_count`` evenly
    over the mask.
    """
    cell_mask = mask.CellMask()
    cell_mask._upsample = upsample
    cell_mask._binary_mask = np.ones(shape, dtype=bool)
    n_pixels = shape[0] * shape[1]
    cell_mask._initial_density = np.full(
        shape, total_count / n_pixels, dtype=np.float64
    )
    cell_mask._recalc_density_mask_from_binary()
    return cell_mask


def test_mean_density_equals_count_over_area():
    """mean_density must equal the true localization count / area.

    Regression for the density fix: ``from_mol_coords`` now divides the
    zoomed density by ``factor**2`` (factor = binsize / upsample = 2 for the
    defaults), so the reported density no longer over-estimates by
    ``factor**2`` (4x for binsize=20 / upsample=10). With a count-conserving
    ``_initial_density``, ``mean_density`` (spots / nm^2) must equal
    ``total_count / area_nm2`` exactly, not 4x that.
    """
    upsample = 10  # nm per mask pixel
    shape = (30, 30)
    total_count = 900.0
    cell_mask = _uniform_cellmask(shape, total_count, upsample=upsample)

    area_nm2 = shape[0] * shape[1] * upsample**2
    expected = total_count / area_nm2  # 0.01 spots / nm^2

    assert np.isclose(cell_mask.mean_density, expected)
    # Not the pre-fix 4x over-estimate (factor**2 == 4 for the defaults).
    assert not np.isclose(cell_mask.mean_density, 4 * expected)

    # The densities array must likewise conserve the count: summing the
    # per-pixel density times the pixel area recovers total_count.
    pixel_area = upsample**2
    recovered = cell_mask.densities.sum() * pixel_area
    assert np.isclose(recovered, total_count)


def test_random_points_not_transposed_on_nonsquare_mask():
    """random_points on a NON-square mask must not swap x and y.

    Regression for the ``indexing="ij"`` meshgrid fix. Mask shape (40, 20)
    gives x_max (dim 0) = 400 nm wider than y_max (dim 1) = 200 nm. Simulated
    points must fall within [0, x_max] x [0, y_max] with x_max != y_max, and
    the x extent must match the (wider) x axis, not be capped at y_max as the
    transposed (default "xy") meshgrid produced.
    """
    upsample = 10
    nx, ny = 40, 20  # x range wider than y range
    cell_mask = mask.CellMask()
    cell_mask._upsample = upsample
    cell_mask._binary_mask = np.ones((nx, ny), dtype=bool)
    cell_mask._initial_density = np.ones((nx, ny), dtype=np.float64)
    cell_mask._recalc_density_mask_from_binary()

    x_max = nx * upsample  # 400
    y_max = ny * upsample  # 200
    assert x_max != y_max

    np.random.seed(1)
    X = cell_mask.random_points(5000, binary=False)

    # Points stay within the (correctly oriented) bounds.
    assert np.all((X[:, 0] >= 0) & (X[:, 0] <= x_max))
    assert np.all((X[:, 1] >= 0) & (X[:, 1] <= y_max))

    # The x extent reaches into the wider axis (> y_max): if x/y were swapped
    # the x coordinates would be capped at y_max = 200.
    assert X[:, 0].max() > y_max
    assert X[:, 1].max() <= y_max
    # x point cloud is wider than the y point cloud.
    assert X[:, 0].max() > X[:, 1].max()


if __name__ == "__main__":
    fps = []
    channel_locs = [io.load_locs(fp)[0] for fp in fps]

    # fig, ax = render.plot_scene(channel_locs, 100, 130)
    # plt.show()

    pixelsize = 130
    binsize = 20
    blursize = 400
    # blur = parameters["blursize"] / binsize
    threshold = 0.85
    mask_pixel_size = 10
    # binary = parameters["binary"]
    cell_mask = mask.CellMask.from_mol_coords(
        channel_locs,
        pixelsize,
        binsize,
        blursize,
        threshold,
        upsample=mask_pixel_size,
    )

    fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, sharey=True)
    ax[0, 0].imshow(cell_mask._initial_density)
    ax[0, 0].set_title("initial density")
    ax[0, 1].imshow(cell_mask._binary_mask)
    ax[0, 1].set_title("binary_mask")
    ax[0, 2].imshow(cell_mask._density_mask)
    ax[0, 2].set_title("density mask")

    cell_mask.filter_mask(fill_holes=True)
    ax[1, 0].imshow(cell_mask._initial_density)
    ax[1, 0].set_ylabel("filtered cell, fh")
    ax[1, 1].imshow(cell_mask._binary_mask)
    ax[1, 2].imshow(cell_mask._density_mask)

    cell_mask.dilate(dilate_nm=500)
    ax[2, 0].imshow(cell_mask._initial_density)
    ax[2, 0].set_ylabel("dilated")
    ax[2, 1].imshow(cell_mask._binary_mask)
    ax[2, 2].imshow(cell_mask._density_mask)

    plt.show()

    fig, ax = cell_mask.plot_mask(binary=True)
    plt.show()

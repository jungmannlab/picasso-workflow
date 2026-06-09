#!/usr/bin/env python
"""
Module Name: test_mask.py
Author: Heinrich Grabmayr
Initial Date: Jan 22, 2025
Description: Test the module analyse.py
"""

import logging

import numpy as np

from picasso_workflow.outpost_modules import mask
from picasso import io
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _cellmask_with_components():
    """Build a CellMask whose binary mask has three separate components
    whose scan-order label ids deliberately differ from their area rank,
    so a label-id-based selection (the previous bug) cannot pass by luck.

    Layout (row-major scan assigns labels in first-encounter order):
        comp A : area 12, scanned first  -> label 1   (LARGEST)
        comp B : area  2, scanned second -> label 2   (smallest)
        comp C : area  9, scanned third  -> label 3   (medium)

    Area rank therefore is A (0th) > C (1st) > B (2nd), while the label
    ids are A=1, B=2, C=3. The old `largest_label - nth_largest` logic
    would select label 0 (background) for nth_largest=1; the area-rank
    logic correctly selects C.
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

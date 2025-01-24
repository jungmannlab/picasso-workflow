#!/usr/bin/env python
"""
Module Name: test_mask.py
Author: Heinrich Grabmayr
Initial Date: Jan 22, 2025
Description: Test the module analyse.py
"""
import logging

from picasso_workflow.outpost_modules import mask
from picasso import io
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


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

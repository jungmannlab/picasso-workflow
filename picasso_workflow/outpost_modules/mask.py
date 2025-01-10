#!/usr/bin/env python
"""
Module Name: mask.py
Author: Heinrich Grabmayr
Initial Date: Dec 13, 2024
Description: This module provides a mask class for cell masking operations.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter, label
import pickle

logger = logging.getLogger(__name__)


class CellMask:
    # FOR NOW THIS IS ONLY ABOUT 2D MASKING

    # The internal mask (2D density mask)
    _mask = np.array(np.nan)
    # computed area in µm^2
    _area = 0
    # the camera pixel size in nm
    _pixelsize = 0
    # the mask bin size in nm
    _binsize = 0
    # the blur in nm
    _blursize = 0
    _threshold = 0
    _upsample = 0

    @classmethod
    def from_mol_coords(
        cls,
        locs,
        pixelsize,
        binsize=20,
        blursize=400,
        threshold=1 / 3,
        upsample=10,
    ):
        """Get a cell mask by histogramming to bins of size 20 nm, blur by
        a factor of 20 bins (400 nm) and set the threshold to one third of
        Otsu threshold. Lastly, upsample the mask to pixel size of 10 nm.

        OLD: Calculates the cell mask based on the molecular positions of all
        6 protein species.

        Parameters
        ----------
        mol_coords : list of np.2darrays
            List of molecular positions of each protein species in nm.
        pixelsize : float
            Pixel size of the camera in nm.
        binary : boolean
            whether to create a binary or density mask (default: False)

        Returns
        -------
        mask : np.2darray (bool)
            Binary mask of the cell.
        area : float
            Area of the cell in um^2.
        """
        instance = cls()
        instance._pixelsize = pixelsize
        instance._binsize = binsize
        instance._blursize = blursize
        instance._threshold = threshold
        instance._upsample = upsample

        mol_coords = picassolocs_to_coords(locs, pixelsize)
        # combine all coordinates into one array
        combined_coords = np.vstack(mol_coords) / binsize
        n_bins = int(np.ceil(512 * pixelsize / binsize))
        bins = np.arange(0, n_bins, 1, dtype=np.float64)
        mask = np.histogram2d(
            combined_coords[:, 0], combined_coords[:, 1], bins=bins
        )[0]
        mask = np.flipud(np.rot90(mask))
        blur = blursize / pixelsize
        mask = gaussian_filter(mask, blur)
        thresh = otsu(mask) * threshold
        mask[mask < thresh] = 0
        factor = int(binsize / upsample)
        mask_final = zoom(mask.astype(np.float64), factor)
        mask_final[mask_final < 0] = 0
        area = (mask_final > 0).sum() * upsample**2
        area = area / 1e6  # convert from nm^2 to um^2
        mask_final /= mask_final.sum()
        mask_final[np.isnan(mask_final)] = 0

        instance._mask = mask_final
        instance._area = area
        return instance

    def picassolocs_to_maskbins(self, locs):
        """Convert picasso localizations to bin locations in the mask
        Args:
            locs : np.rec.array (len N)
                the input localizations, must comprise 'x', 'y'
        Returns:
            mask_coords : np.array, int32 (2, N)
                the coordinates within the mask that correspond to the
                localization position, with out-of-bounds positions set to -1
        """
        mol_coords = picassolocs_to_coords(locs, self._pixelsize)
        factor = int(self._binsize / self._upsample)
        # combine all coordinates into one array
        mask_coords = np.vstack(mol_coords) / self._binsize * factor
        mask_coords = np.floor(mask_coords).astype(np.int32)
        # flip and turn (corresponding to the flipud, rot90 operations)
        # basically switches x and y coordinates
        mask_x = mask_coords[:, 0]
        mask_coords[:, 0] = mask_coords[:, 1]
        mask_coords[:, 1] = mask_x
        # set the coordinates that are out of bounds to -1
        mask_coords[mask_coords < 0] = -1
        mask_coords[mask_coords[:, 0] > self.mask.shape[0], 0] = -1
        mask_coords[mask_coords[:, 1] > self.mask.shape[1], 1] = -1
        return mask_coords

    def save(self, fp):
        save_dict = {
            "mask": self._mask,
            "area": self._area,
            "binsize": self._binsize,
            "blursize": self._blursize,
            "threshold": self._threshold,
            "upsample": self._upsample,
        }
        with open(fp, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, fp):
        with open(fp, "rb") as f:
            save_dict = pickle.load(f)
        instance = cls()
        instance._mask = save_dict["mask"]
        instance._area = save_dict["area"]
        instance._binsize = save_dict["binsize"]
        instance._blursize = save_dict["blursize"]
        instance._threshold = save_dict["threshold"]
        instance._upsample = save_dict["upsample"]
        return instance

    @property
    def density_mask(self):
        return self._mask

    @property
    def binary_mask(self):
        mask_final = self._mask.copy()
        mask_final[mask_final <= 0] = 0
        mask_final[mask_final > 0] = 1
        mask_final /= mask_final.sum()
        return mask_final

    @property
    def area(self):
        return self._area

    def filter_mask(self):
        """Select the largest connected area in the mask"""

        binary_mask = self.binary_mask
        labeled_array, num_features = label(binary_mask)
        sizes = np.bincount(labeled_array.ravel())
        largest_component_index = sizes[1:].argmax() + 1
        largest_component_mask = (
            labeled_array == largest_component_index
        ).astype(np.int8)

        self._mask[largest_component_mask == 0] = 0
        self._mask /= self._mask.sum()

    def apply_to_locs(self, locs):
        """Applies the binary mask to localizations: locs
        outside of the masked area are dropped

        Args:
            locs : np.rec.array
                localizations, must comprise 'x' and 'y'
        """
        mask_coords = self.picassolocs_to_maskbins(locs)
        # eliminate out of bound entries
        inbound = (mask_coords[:, 0] >= 0) | (mask_coords[:, 1] >= 0)
        mask_coords = mask_coords[inbound, :]
        bmask = self.binary_mask
        in_cell = bmask[mask_coords[:, 0], mask_coords[:, 1]]
        return locs[in_cell]

    def plot_mask(self, fp, binary=False):
        """plot binary or density version of the mask"""
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        ax.set_title("mask - final")
        # check if mask is binary
        if binary:
            mask_plot = self.density_mask
            mask_plot = mask_plot.astype(np.bool_)
            cmap = "binary"
        else:
            mask_plot = self._mask
            cmap = "hot"
        ax.imshow(
            mask_plot,
            extent=[
                0,
                self._upsample * mask_plot.shape[0],
                0,
                self._upsample * mask_plot.shape[1],
            ],
            cmap=cmap,
        )

        ax.set_xlabel("x [nm]")
        ax.set_ylabel("y [nm]")
        # ax.set_xticks()
        # ax.set_xlim(x0, x0 + length)
        # ax.set_ylim(y0, y0 + length)
        fig.savefig(fp)

    def random_points(self, n_points, binary=False, mask=None):
        """Simulates monomeric molecules based on density mask, see
        simulate_CSR to see the inputs.

        Returns
        -------
        X - np.array with simulated coordinates of shape (n_simulations,
            n_points, 2).
        """
        if mask is None:
            if binary:
                mask = self.binary_mask
            else:
                mask = self.density_mask
        x_min = y_min = 0
        x_max = y_max = mask.shape[0] * self._binsize
        X = np.zeros((n_points, 2))
        rng = np.random.default_rng()
        counts = rng.multinomial(n_points, pvals=mask.ravel())
        bins_x_left = np.arange(x_min, x_max, self._binsize)
        bins_y_left = np.arange(y_min, y_max, self._binsize)
        bins_x_left, bins_y_left = np.meshgrid(bins_x_left, bins_y_left)
        lows_x = np.repeat(bins_x_left.ravel(), counts)
        lows_y = np.repeat(bins_y_left.ravel(), counts)
        highs_x = lows_x + self._binsize
        highs_y = lows_y + self._binsize
        x = np.random.uniform(lows_x, highs_x)
        y = np.random.uniform(lows_y, highs_y)
        X = np.column_stack((x, y))
        return X

    def random_points_sets(self, n_sets, n_points_per_set, binary=False):
        if binary:
            mask = self.binary_mask
        else:
            mask = self.density_mask
        X = np.stack(
            [
                self.random_points(n_points_per_set, binary=binary, mask=mask)
                for _ in range(n_sets)
            ]
        )
        return X


def picassolocs_to_coords(mols, pixelsize):
    """Converts the Picasso-format np.rec.array to a 2D numpy array with
    spatial coordinates in nm."""
    return np.array([mols["x"], mols["y"]]).T * pixelsize


def otsu(image):
    """Simplified function from scikit-image so that i do not need to
    install the whole package."""

    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    thresh = bin_centers[idx]
    return thresh

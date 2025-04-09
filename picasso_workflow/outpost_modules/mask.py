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
from scipy.ndimage import (
    zoom,
    gaussian_filter,
    label,
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
)
import pickle

logger = logging.getLogger(__name__)


class CellMask:
    # FOR NOW THIS IS ONLY ABOUT 2D MASKING

    # The internal mask (2D density mask)
    _binary_mask = np.array(np.nan)
    _density_mask = np.array(np.nan)
    _initial_density = np.array(np.nan)
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
    _offset = 0

    @classmethod
    def from_mol_coords(
        cls,
        locs_list,
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

        mol_coords = [
            picassolocs_to_coords(locs, pixelsize) for locs in locs_list
        ]
        # combine all coordinates into one array
        combined_coords = np.vstack(mol_coords)  # / binsize
        # print(combined_coords.shape)
        x_min = np.floor(combined_coords[:, 0].min())
        x_max = np.ceil(combined_coords[:, 0].max())
        y_min = np.floor(combined_coords[:, 1].min())
        y_max = np.ceil(combined_coords[:, 1].max())
        instance._offset = x_min, y_min
        combined_coords[:, 0] -= x_min
        combined_coords[:, 1] -= y_min
        bins_x = np.arange(0, x_max - x_min, step=binsize, dtype=np.float64)
        bins_y = np.arange(0, y_max - y_min, step=binsize, dtype=np.float64)
        mask = np.histogram2d(
            combined_coords[:, 0], combined_coords[:, 1], bins=[bins_x, bins_y]
        )[0]
        # assuming: this is for display. Now using imshow origin "lower"
        # mask = np.flipud(np.rot90(mask))
        # mask = np.rot90(mask)

        blur = blursize / binsize
        factor = int(binsize / upsample)

        initial_density = gaussian_filter(mask, blur)
        initial_density = zoom(initial_density.astype(np.float64), factor)
        initial_density[initial_density < 0] = 0
        instance._initial_density = initial_density

        # create binary mask
        binary_mask = initial_density.copy()
        thresh = otsu(initial_density) * threshold
        binary_mask[binary_mask < thresh] = 0
        binary_mask[binary_mask >= thresh] = 1

        # blur and apply threshold to get more accurate mask
        # (remove added area on cell border, keep potential
        # low-density inside cell area - no 'holes')
        binary_mask = gaussian_filter(binary_mask, blur)
        reduction_threshold = 0.9
        binary_mask[binary_mask <= reduction_threshold] = 0
        binary_mask[binary_mask > reduction_threshold] = 1

        binary_mask = binary_mask.astype(np.bool_)
        instance._binary_mask = binary_mask

        instance._recalc_density_mask_from_binary()

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
        # ax[0, 0].imshow(instance._initial_density)
        # ax[0, 0].set_title('initial density')
        # ax[0, 1].imshow(instance._binary_mask)
        # ax[0, 1].set_title('binary_mask')
        # ax[1, 1].imshow(instance._density_mask)
        # ax[1, 1].set_title('density mask')
        # plt.show()

        return instance

    def _recalc_density_mask_from_binary(self):
        self._density_mask = self._initial_density.copy()
        self._density_mask[~self._binary_mask] = 0
        self._density_mask /= self._density_mask.sum()

    @property
    def mean_density(self):
        """Returns the mean density in the mask in spots/nm^2"""
        total_spots = np.sum(self._initial_density[self._binary_mask])
        n_pixels = np.sum(self._binary_mask)
        pixel_area = self._upsample**2
        return total_spots / n_pixels / pixel_area

    @property
    def densities(self):
        """Returns the densites of mask pixels (#spots/nm^2)"""
        total_spots = np.sum(self._initial_density[self._binary_mask])
        pixel_area = self._upsample**2
        return self.density_mask * total_spots / pixel_area

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
        mol_coords[:, 0] -= self._offset[0]
        mol_coords[:, 1] -= self._offset[1]
        factor = int(self._binsize / self._upsample)
        # combine all coordinates into one array
        mask_coords = np.vstack(mol_coords) / self._binsize * factor
        mask_coords = np.floor(mask_coords).astype(np.int32)
        # # flip and turn (corresponding to the flipud, rot90 operations)
        # # basically switches x and y coordinates
        # mask_x = mask_coords[:, 0]
        # mask_coords[:, 0] = mask_coords[:, 1]
        # mask_coords[:, 1] = mask_x
        # set the coordinates that are out of bounds to -1
        mask_coords[mask_coords < 0] = -1
        mask_coords[mask_coords[:, 0] > self.shape[0], 0] = -1
        mask_coords[mask_coords[:, 1] > self.shape[1], 1] = -1
        return mask_coords

    def save(self, fp):
        save_dict = {
            "initial_density": self._initial_density,
            "density_mask": self._density_mask,
            "binary_mask": self._binary_mask,
            "binsize": self._binsize,
            "blursize": self._blursize,
            "threshold": self._threshold,
            "upsample": self._upsample,
            "offset": self._offset,
            "pixelsize": self._pixelsize,
        }
        with open(fp, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, fp):
        with open(fp, "rb") as f:
            save_dict = pickle.load(f)
        instance = cls()
        instance._initial_density = save_dict["initial_density"]
        instance._density_mask = save_dict["density_mask"]
        instance._binary_mask = save_dict["binary_mask"]
        instance._binsize = save_dict["binsize"]
        instance._blursize = save_dict["blursize"]
        instance._threshold = save_dict["threshold"]
        instance._upsample = save_dict["upsample"]
        instance._offset = save_dict["offset"]
        instance._pixelsize = save_dict["pixelsize"]
        return instance

    @property
    def density_mask(self):
        return self._density_mask

    @property
    def binary_mask(self):
        return self._binary_mask

    @property
    def shape(self):
        """The mask(s) shape"""
        return self._binary_mask.shape

    @property
    def area(self):
        area = (self.binary_mask > 0).sum() * self._upsample**2
        area = area / 1e6  # convert from nm^2 to um^2
        return area

    def filter_mask(self, nth_largest=0, fill_holes=True):
        """Select the nth-largest connected area in the mask, and fill
        potential holes in this area.
        Args:
            nth_largest : int
                select the nth largest cell:
                0 - largest cell; 1 - scond to largest; etc
            fill_holes : bool
                whether to fill holes in the mask
        """
        binary_mask = self.binary_mask
        # print('binary mask shape in', binary_mask.shape)
        labeled_array, num_features = label(binary_mask)
        labeled_nobkg = labeled_array.ravel()
        labeled_nobkg = labeled_nobkg[labeled_nobkg > 0]
        feature, counts = np.unique(labeled_nobkg, return_counts=True)
        # sizes = np.bincount(labeled_nobkg)
        try:
            largest_component_index = feature[counts.argsort()[-nth_largest]]
            logger.debug(
                f"filtering {nth_largest} largest cell (starting 0)"
                + f"largest component_index: {largest_component_index}"
                + "features"
                + f": {feature}"
                + f"sizes uniques: {counts}"
            )
        except ValueError:
            largest_component_index = 1
        largest_component_mask = (
            labeled_array == largest_component_index
        ).astype(np.int8)

        if fill_holes:
            # fill holes in the mask
            largest_component_mask = binary_fill_holes(
                largest_component_mask
            ).astype(int)

        self._binary_mask = largest_component_mask.astype(np.bool_)

        self._recalc_density_mask_from_binary()

    def erode(self, erode_nm):
        """Focus the mask by a given number of nanometers"""
        erode_px = int(np.round(erode_nm / self._upsample))
        self._binary_mask = binary_erosion(
            self.binary_mask, iterations=erode_px
        ).astype(np.bool_)

        self._recalc_density_mask_from_binary()

    def dilate(self, dilate_nm):
        """Focus the mask by a given number of nanometers"""
        dilate_px = int(np.round(dilate_nm / self._upsample))
        self._binary_mask = binary_dilation(
            self.binary_mask, iterations=dilate_px
        ).astype(np.bool_)

        self._recalc_density_mask_from_binary()

    def apply_to_locs(self, locs):
        """Applies the binary mask to localizations: locs
        outside of the masked area are dropped

        Args:
            locs : np.rec.array
                localizations, must comprise 'x' and 'y'
        """
        mask_coords = self.picassolocs_to_maskbins(locs)
        # eliminate out of bound entries
        maskshape = self.shape
        inbound = (
            (mask_coords[:, 0] >= 0)
            & (mask_coords[:, 1] >= 0)
            & (mask_coords[:, 0] < maskshape[0])
            & (mask_coords[:, 1] < maskshape[1])
        )
        # mask_coords = mask_coords[inbound, :]
        bmask = self.binary_mask
        in_cell = np.zeros(locs.shape[0], dtype=np.bool_)
        in_cell[inbound] = bmask[
            mask_coords[inbound, 0], mask_coords[inbound, 1]
        ]
        return locs[in_cell]

    def plot_mask(self, fp=None, binary=False):
        """plot binary or density version of the mask"""
        fig, ax = plt.subplots()
        # ax.set_aspect("equal")
        # check if mask is binary
        if binary:
            mask_plot = self.binary_mask
            cmap = "binary"
            title = "binary mask"
        else:
            mask_plot = self.density_mask
            cmap = "hot"
            title = "density mask"
        ax.set_title(title)
        x_min = self._offset[0] / 1000
        x_max = x_min + self._upsample * mask_plot.shape[0] / 1000
        y_min = self._offset[1] / 1000
        y_max = y_min + self._upsample * mask_plot.shape[1] / 1000
        ax.imshow(
            mask_plot.T,  # np.rot90(mask_plot),
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            origin="lower",
            aspect="equal",
        )
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        # ax.set_xticks()
        # ax.set_xlim(x0, x0 + length)
        # ax.set_ylim(y0, y0 + length)
        if fp is not None:
            fig.savefig(fp)
        return fig, ax

    def random_points(self, n_points, binary=False, mask=None):
        """Simulates monomeric molecules based on density mask, see
        simulate_CSR to see the inputs.

        Returns
        -------
        X - np.array with simulated coordinates of shape (n_simulations,
            n_points, 2).
        """
        if binary:
            if mask is None:
                mask = self.binary_mask
        else:
            if mask is None:
                mask = self.density_mask
        mask = mask / mask.sum()
        x_min = y_min = 0
        x_max = mask.shape[0] * self._upsample
        y_max = mask.shape[1] * self._upsample
        X = np.zeros((n_points, 2))
        rng = np.random.default_rng()
        counts = rng.multinomial(n_points, pvals=mask.ravel())
        bins_x_left = np.arange(x_min, x_max, self._upsample)
        bins_y_left = np.arange(y_min, y_max, self._upsample)
        bins_x_left, bins_y_left = np.meshgrid(bins_x_left, bins_y_left)
        lows_x = np.repeat(bins_x_left.ravel(), counts)
        lows_y = np.repeat(bins_y_left.ravel(), counts)
        highs_x = lows_x + self._upsample
        highs_y = lows_y + self._upsample
        x = np.random.uniform(lows_x, highs_x)
        y = np.random.uniform(lows_y, highs_y)
        X = np.column_stack((x, y))
        return X

    # def random_points(self, n_points, binary=False, mask=None):
    #     """Simulates monomeric molecules based on density mask, see
    #     simulate_CSR to see the inputs.

    #     Returns
    #     -------
    #     X - np.array with simulated coordinates of shape (n_simulations,
    #         n_points, 2).
    #     """
    #     if binary:
    #         return self.random_points_binary(n_points, mask)
    #     else:
    #         return self.random_points_density(n_points, mask)

    # def random_points_density(self, n_points, mask=None):
    #     """Simulates monomeric molecules based on density mask, see
    #     simulate_CSR to see the inputs.

    #     Returns
    #     -------
    #     X - np.array with simulated coordinates of shape (n_simulations,
    #         n_points, 2).
    #     """
    #     if mask is None:
    #         mask = self.density_mask
    #     x_min = y_min = 0
    #     x_max = mask.shape[0] * self._upsample
    #     y_max = mask.shape[1] * self._upsample
    #     X = np.zeros((n_points, 2))
    #     rng = np.random.default_rng()
    #     counts = rng.multinomial(n_points, pvals=mask.ravel())
    #     bins_x_left = np.arange(x_min, x_max, self._upsample)
    #     bins_y_left = np.arange(y_min, y_max, self._upsample)
    #     bins_x_left, bins_y_left = np.meshgrid(bins_x_left, bins_y_left)
    #     lows_x = np.repeat(bins_x_left.ravel(), counts)
    #     lows_y = np.repeat(bins_y_left.ravel(), counts)
    #     highs_x = lows_x + self._upsample
    #     highs_y = lows_y + self._upsample
    #     x = np.random.uniform(lows_x, highs_x)
    #     y = np.random.uniform(lows_y, highs_y)
    #     X = np.column_stack((x, y))
    #     return X

    # def random_points_binary(self, n_points, mask=None):
    #     if mask is None:
    #         mask_area = self.area * 1e6  # convert from um^2 to nm^2
    #         mask = self.binary_mask
    #     else:
    #         mask_area = mask.sum() * self._upsample**2
    #     density = n_points / mask_area

    #     canvas_area = np.product(mask.shape) * self._upsample**2  # in nm^2
    #     n_sample = int(density * canvas_area)

    #     # create points on canvas
    #     X = np.column_stack([
    #         np.random.uniform(
    #             0, mask.shape[0] * self._upsample, size=n_sample),
    #         np.random.uniform(
    #             0, mask.shape[1] * self._upsample, size=n_sample),
    #     ])

    #     # Reject points outside of mask
    #     x_ind = (np.floor(X[:, 0] / self._upsample)).astype(int)
    #     y_ind = (np.floor(X[:, 1] / self._upsample)).astype(int)
    #     in_mask = mask[y_ind, x_ind].astype(bool)
    #     X = X[in_mask]
    #     return X

    def random_points_sets(self, n_sets, n_points_per_set, binary=False):
        if binary:
            mask = self.binary_mask
            mask = mask / np.sum(mask)
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
    return np.array([mols["x"].copy(), mols["y"].copy()]).T * pixelsize


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

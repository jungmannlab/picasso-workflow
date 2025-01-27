#!/usr/bin/env python
"""
Module Name: ripleys.py
Author: Rafal Kowalewski
Initial Date: Nov 21, 2024
Description: This module provides functionality for Ripley's K analysis,
    especially in the context of the DC Atlas paper.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.ndimage import zoom, gaussian_filter, label


logger = logging.getLogger(__name__)


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


def univariate_ripley_K(X, r, area):
    tree = KDTree(X)
    n = X.shape[0]
    NN = tree.count_neighbors(tree, r) - n
    lambda_inv = area / n
    K = lambda_inv * NN / n
    return K


def bivariate_ripley_K(X1, X2, r, area):
    tree1 = KDTree(X1)
    tree2 = KDTree(X2)
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    NN = tree1.count_neighbors(tree2, r)
    lambda_inv1 = area / n1
    lambda_inv2 = area / n2
    const_term = lambda_inv1 * lambda_inv2 / area
    K = const_term * NN
    return K


def ripley_K(X1, X2, r, area):
    """
    Alternatively, use spatial statistics packages, with
    Ripley's functions implemented, including edge correction
    e.g.
    https://docs.astropy.org/en/stable/stats/ripley.html
    https://docs.astropy.org/en/stable/api/astropy.stats.RipleysKEstimator.html#astropy.stats.RipleysKEstimator
    """
    if np.array_equal(X1, X2):
        return univariate_ripley_K(X1, r, area)
    else:
        return bivariate_ripley_K(X1, X2, r, area)


def ripley_H(X1, X2, r, area):
    """https://pmc.ncbi.nlm.nih.gov/articles/PMC2726315/"""
    rK = ripley_K(X1, X2, r, area)
    rL = np.sqrt(rK / np.pi)
    rH = rL - r
    return rH


def radial_distribution_function(X1, X2, r, univariate):
    """Calculates the density of X1 spots at annuli around
    X2 spots
    """
    tree1 = KDTree(X1)
    tree2 = KDTree(X2)
    n1 = X1.shape[0]
    # n2 = X2.shape[0]
    deltar = r[1] - r[2]
    rs = np.append(r, np.max(r) + deltar)
    n_means = tree1.count_neighbors(tree2, rs) / n1
    if univariate:
        n_means = n_means - 1  # subtract center point
    d_n_means = n_means[1:] - n_means[:-1]
    d_rs = rs[1:] - rs[:-1]
    r_means = (rs[1:] + rs[:-1]) / 2

    d_areas = 2 * np.pi * r_means * d_rs
    rdf = d_n_means / d_areas
    return rdf


def first_nn(X1, X2, r, univariate):
    """Calculates the first nearest neighbor histogram"""
    tree1 = KDTree(X1)
    tree2 = KDTree(X2)
    if univariate:
        k = 2
    else:
        k = 1
    alldist, indices = tree1.query(tree2.data, k=k)
    if len(alldist.shape) > 1:
        alldist = alldist[:, k - 1]
    bins = np.append(r, np.max(r) + (r[1] - r[0]))
    nnhist, _ = np.histogram(alldist, bins=bins)
    return nnhist


def overpopulation(X1, X2, r, univariate):
    """Calculates the "overpopulation": The mean number of X2 molecules
    between r[i] and r[i+1] from the X1 molecules surpassing the expected
    number, based on the density between r[-2] and r[-1]. This corrects
    for local density differences, and gives a meaningful measure if there
    is a mechanistic reason for X2 molecules to be at overrepresented around
    X1 molecules, and r extends to distances at which this effect is negligible.
    """
    pass


def fraction_types(
    Xall,
    types,
    type_self,
    r,
    nshuffle=20,
    shuffle_self=True,
    relocate_self=True,
    fraction_exclude=None,
):
    """Calculates the mean fraction of types within a given radius
    around type_self spots. This is done for original data, and data in which
    type identities are shuffled as a control
    Args:
        Xall : 2D array (2-3, N)
            x, y (, z) positions of spots
        types : 1D int
            spot types (represented as integers)
        type_self : int
            spot type to use as center
        r : 1D array
            the radii at which to evaluate
        nshuffle : int
            the number of times to shuffle type identities
        shuffle_self : bool
            whether to shuffle only other types or also the self type
        relocate_self : bool
            whether to relocate centerpoints to 'type_self' after
            shuffling.
        fraction_exclude : int or None
            the type to exclude from normalization (e.g. the
            'real' (not shuffled and relocated) type_self)
    Returns:
        fract_types : dict of 1D array
            the fractions of various types within the ball radii, for
            each value of r
        fract_types_ctrl : dict of 2D array
            the fractions of various types within the ball radii, for
            each value of r, and for each iteration of shuffling
    """
    types_present = np.unique(types)
    idx_self = np.argwhere(types == type_self).flatten()
    tree_all = KDTree(Xall)
    tree_self = KDTree(Xall[idx_self, :])

    neighbor_types_fract = [{} for radius in r]
    neighbor_types_ctrl_fract = [[{}] * nshuffle for radius in r]
    for ir, radius in enumerate(r):
        balls_indices = tree_self.query_ball_tree(tree_all, radius)

        neighbor_types_fract[ir] = type_fractions(
            balls_indices,
            types,
            types_present,
            type_self=type_self,
            fraction_exclude=fraction_exclude,
        )
        # do controls
        types_ctrl = types.copy().flatten()
        for ictrl in range(nshuffle):
            if shuffle_self:
                np.random.shuffle(types_ctrl)
                tp_self = types_ctrl[idx_self]
            else:
                idx_other = np.argwhere(types_ctrl != type_self)
                idx_other_shuffled = idx_other.copy()
                np.random.shuffle(idx_other_shuffled)
                types_ctrl[idx_other] = types_ctrl[idx_other_shuffled]
                tp_self = type_self
            if relocate_self:
                idx_self_ctrl = np.argwhere(types_ctrl == type_self).flatten()
                tree_self_ctrl = KDTree(Xall[idx_self_ctrl, :])
                balls_indices = tree_self_ctrl.query_ball_tree(
                    tree_all, radius
                )
                tp_self = types_ctrl[idx_self_ctrl]
            neighbor_types_ctrl_fract[ir][ictrl] = type_fractions(
                balls_indices,
                types_ctrl,
                types_present,
                type_self=tp_self,
                fraction_exclude=fraction_exclude,
            )

    # re shape from list of dicts to dict of array
    fract_types = {
        t: np.array([neighbor_types_fract[ir][t] for ir in range(len(r))])
        for t in types_present
    }
    fract_types_ctrl = {
        t: np.array(
            [
                [
                    neighbor_types_ctrl_fract[ir][ictrl][t]
                    for ir in range(len(r))
                ]
                for ictrl in range(nshuffle)
            ]
        )
        for t in types_present
    }
    return fract_types, fract_types_ctrl


def type_fractions(
    balls_indices, types, types_present, type_self=None, fraction_exclude=None
):
    """Counts types and calculates their fraction
    Args:
        balls_indices : list of lists of int
            the indices of types for each origin of balls
        types : 1D array of int
            the types of the spots indexed
        fraction_exclude : int or None
            the type to exclude from normalization (e.g. the
            'real' (not shuffled and relocated) type_self)
    """
    if fraction_exclude is None or fraction_exclude is False:
        total_neighbors = sum(
            [len(ball_indices) - 1 for ball_indices in balls_indices]
        )
    else:
        total_neighbors = 0
        for i, ball_indices in enumerate(balls_indices):
            ball_types = np.array(types[ball_indices])
            n_nonexcluded = np.sum(ball_types != fraction_exclude)
            # additionally exclude center point if it hasn't been excluded yet
            if isinstance(type_self, list) or isinstance(
                type_self, np.ndarray
            ):
                ctr_type = type_self[i]
            elif type_self is not None:
                ctr_type = type_self
            else:
                ctr_type = -1
            if ctr_type != fraction_exclude:
                n_nonexcluded -= 1
            total_neighbors += n_nonexcluded

    neighbor_types_fract = {type: 0 for type in types_present}
    for i, ball_indices in enumerate(balls_indices):
        # ball_indices is the list of tree_all indices within the ball
        # of one of the points of tree_self
        ball_types = types[ball_indices]
        if len(ball_types) > 1:
            # # leave out the origin spot (assuming the list is sorted by distance)
            # ball_types = ball_types[1:]
            pass
        else:
            # no spots apart from the origin spot
            continue
        tp, counts = np.unique(ball_types, return_counts=True)
        for t, c in zip(tp, counts):
            if isinstance(type_self, list) or isinstance(
                type_self, np.ndarray
            ):
                if t == type_self[i]:
                    c = c - 1
            elif type_self is not None:
                # leave out the origin spot (assuming the list is sorted by distance)
                if t == type_self:
                    c = c - 1
            if (fraction_exclude is None) or (t != fraction_exclude):
                neighbor_types_fract[t] += c / total_neighbors
    return neighbor_types_fract


def simulate_density_mask_CSR(n_points, mask, n_simulations, pixelsize):
    """Simulates monomeric molecules based on density mask, see
    simulate_CSR to see the inputs.

    Returns
    -------
    X - np.array with simulated coordinates of shape (n_simulations,
        n_points, 2).
    """
    x_min = y_min = 0
    x_max = y_max = mask.shape[0] * pixelsize
    X = np.zeros((n_simulations, n_points, 2))
    for i in range(n_simulations):
        rng = np.random.default_rng()
        counts = rng.multinomial(n_points, pvals=mask.ravel())
        bins_x_left = np.arange(x_min, x_max, pixelsize)
        bins_y_left = np.arange(y_min, y_max, pixelsize)
        bins_x_left, bins_y_left = np.meshgrid(bins_x_left, bins_y_left)
        lows_x = np.repeat(bins_x_left.ravel(), counts)
        lows_y = np.repeat(bins_y_left.ravel(), counts)
        highs_x = lows_x + pixelsize
        highs_y = lows_y + pixelsize
        x = np.random.uniform(lows_x, highs_x)
        y = np.random.uniform(lows_y, highs_y)
        X[i] = np.column_stack((x, y))
    return X


def simulate_CSR(n_points, mask, n_simulations, pixelsize):
    """Simulates CSR within using the density mask by simulating as in
    SPINNA.

    n_points - number of points to simulate (int for one species, tuple
               for two species)
    mask - binary mask of the cell
    area - area of the cell in um^2
    n_simulations - number of simulations to run
    pixelsize - mask pixel size in nm, see get_cell_mask - upsample

    returns X - two np.arrays with simulated coordinates of
        shape (n_simulations, n_points, 2).
    """

    # convert area to the units of mask bin size (from nm^2 to cam. pixels)
    # area /= pixelsize ** 2
    # image_area = mask.size # units: mask bin size
    if isinstance(n_points, int):
        X = simulate_density_mask_CSR(n_points, mask, n_simulations, pixelsize)
        return X, X
    else:
        X1 = simulate_density_mask_CSR(
            n_points[0], mask, n_simulations, pixelsize
        )
        X2 = simulate_density_mask_CSR(
            n_points[1], mask, n_simulations, pixelsize
        )
        return X1, X2


def randomize_data(X, randomization_radius):
    """Create uniform random data in a circle of radius randomization_radius,
    2D data is assumed.
    Args:
        X : np.array
            x and y values of localizations [nm]
        randomization_radius : float
            the radius to randomize data points by
    Returns:
        rnd : np.array of same shape as X
            the randomized dataset
    """
    N = X.shape[0]
    phase_rnd = np.exp(1j * 2 * np.pi * np.random.random(N))
    r_rnd = randomization_radius * np.random.power(a=3, size=N)  # quadratic
    cart_rnd = np.stack(
        [
            r_rnd * np.real(phase_rnd),
            r_rnd * np.imag(phase_rnd),
        ]
    ).T
    return X + cart_rnd


def randomize_data_ntimes(X, randomization_radius, n_randomizations):
    """Randomize data multiple times, to get normalization baseline.

    Args:
        X : np.array
            x and y values of localizations [nm]
        randomization_radius : float
            the radius to randomize data points by
        n_randomizations : int
            the number of separate randomizations to perform
    Returns:
        rnd_data : list of np.array of same shape as X
            the randomized datasets
    """
    rnd_data = [
        randomize_data(X, randomization_radius) for _ in n_randomizations
    ]
    return rnd_data


def ripley_K_CSR(
    n_points, mask, mask_pixel_size, area, radii, n_simulations=100
):
    # note that n_points is either a tuple of the number of points for
    # each of the 2 species (if we're doing cross-Ripley) or just the
    # number of points for one species (if we're doing univariate Ripley)
    X = simulate_CSR(n_points, mask, n_simulations, mask_pixel_size)
    K = []
    for i in range(n_simulations):
        K.append(ripley_K(X[0][i], X[1][i], radii, area))
    return np.array(K)


def normalize_to_CSR(K_exp, K_csr, ci=0.95):
    K_csr_mean = np.mean(K_csr, axis=0)
    K_exp_norm = K_exp - K_csr_mean

    quantile_low = (1 - ci) / 2
    quantile_high = 1 - quantile_low

    idx_pos = K_exp_norm >= 0
    quantiles_high = np.array(
        [np.quantile(x, quantile_high) for x in np.transpose(K_csr)]
    )
    divider_high = np.abs(quantiles_high - K_csr_mean)
    idx_pos_final = idx_pos & (divider_high != 0)
    K_exp_norm[idx_pos_final] /= divider_high[idx_pos_final]

    quantiles_low = np.array(
        [np.quantile(x, quantile_low) for x in np.transpose(K_csr)]
    )
    divider_low = np.abs(quantiles_low - K_csr_mean)
    idx_neg_final = ~idx_pos & (divider_low != 0)
    K_exp_norm[idx_neg_final] /= divider_low[idx_neg_final]
    return K_exp_norm


def get_cell_mask(
    mol_coords,
    pixelsize,
    binsize=20,
    blur=20,
    threshold=1 / 3,
    upsample=10,
    binary=False,
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

    # combine all coordinates into one array
    combined_coords = np.vstack(mol_coords) / binsize
    n_bins = int(np.ceil(512 * pixelsize / binsize))
    bins = np.arange(0, n_bins, 1, dtype=np.float64)
    mask = np.histogram2d(
        combined_coords[:, 0], combined_coords[:, 1], bins=bins
    )[0]
    mask = np.flipud(np.rot90(mask))
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
    if binary:
        mask_final[mask_final <= 0] = 0
        mask_final[mask_final > 0] = 1
        mask_final /= mask_final.sum()
    return mask_final, area


def filter_mask(mask):
    """Select the largest connected area in the mask"""
    binary_mask = mask.copy()
    binary_mask[binary_mask > 0] = 1

    labeled_array, num_features = label(binary_mask)
    sizes = np.bincount(labeled_array.ravel())
    largest_component_index = sizes[1:].argmax() + 1
    largest_component_mask = (labeled_array == largest_component_index).astype(
        np.int8
    )
    filtered_mask = mask.copy()
    filtered_mask[largest_component_mask == 0] = 0
    return filtered_mask


def plot_mask(mask, pixelsize, fp):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.set_title("mask - final")
    # check if mask is binary
    if len(np.unique(mask)) == 2:
        mask_plot = mask.copy()
        mask_plot[mask_plot > 0] = 1
        mask_plot[mask_plot < 1] = 0
        mask_plot = mask_plot.astype(np.bool_)
        cmap = "binary"
    else:
        mask_plot = mask
        cmap = "hot"
    ax.imshow(
        mask_plot,
        extent=[0, pixelsize * mask.shape[0], 0, pixelsize * mask.shape[1]],
        cmap=cmap,
    )

    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    # ax.set_xticks()
    # ax.set_xlim(x0, x0 + length)
    # ax.set_ylim(y0, y0 + length)
    fig.savefig(fp)


def convert_picasso_to_coords(mols, pixelsize):
    """Converts the Picasso-format np.rec.array to a 2D numpy array with
    spatial coordinates in nm."""
    return np.array([mols["x"], mols["y"]]).T * pixelsize


def analyze_2_channels(
    exp_X1,
    exp_X2,
    mask,
    mask_pixel_size,
    area,
    radii,
    n_simulations,
    ax_u,
    ax_n,
    name1="",
    name2="",
    controltype="CSR",
    metric="RK",
    normalization="zscore",
    randomization_radius=None,
    showControlEnvelope=True,
):
    """Runs the analysis of any two channels of the dataset (2 protein
    species).
    Args:
        metric : str
            the metric to calculate from the data
            "RK": Ripley's K
            "RH": Ripley's H
            "RDF": Radial Distribution function
            "1NN": first nearest neighbor distance distribution
        controltype : str
            the method of creating controls
            "CSRbin": draw random points from a previously generated mask,
                using binary mask information -> CSR
            "CSRdens": draw random points from a previously generated mask,
                with weighting density (of all targets combined)
            "RND": randomization of experimental data by uniformly relocating
                datapoints by a vector on a circle with randomization_radius
        normalization : str
            the method of normalizing experimental data
            "zscore" (units of  95% ci from mean of conrols)
            "diff" (difference to mean of controls)
            "deltaAprepeak" (only positive difference to mean of controls,
                before peak of mean of controls)
            "deltaAprepeakNorm" (only positive difference to mean of controls,
                before peak of mean of controls, divided by number of X2 spots)
            "deltaAprepeakPerc" (same as deltaAprepeakNorm, but in percent)
            "to_max_r" (divide by value of curve at maximum r)
    """
    if np.array_equal(exp_X1, exp_X2):
        n_points = len(exp_X1)
        univariate = True
    else:
        n_points = (len(exp_X1), len(exp_X2))
        univariate = False

    if normalization is None:
        normalization = "zscore"

    if metric == "RK":
        K_exp = ripley_K(exp_X1, exp_X2, radii, area)
    elif metric == "RH":
        K_exp = ripley_H(exp_X1, exp_X2, radii, area)
    elif metric == "RDF":
        K_exp = radial_distribution_function(exp_X1, exp_X2, radii, univariate)
    elif metric == "1NN":
        K_exp = first_nn(exp_X1, exp_X2, radii, univariate)
    else:
        raise NotImplementedError()

    K_csr = []
    for i in range(n_simulations):
        if "CSR" in controltype:
            # X_ctrl = simulate_CSR(n_points, mask, 1, mask_pixel_size)
            # X1_ctrl = X_ctrl[0][0]
            # X2_ctrl = X_ctrl[1][0]
            if "bin" in controltype:
                binary = True
            elif "dens" in controltype:
                binary = False

            if univariate:
                X1_ctrl = mask.random_points(n_points, binary=binary)
                X2_ctrl = X1_ctrl
            else:
                X1_ctrl = mask.random_points(n_points[0], binary=binary)
                X2_ctrl = mask.random_points(n_points[1], binary=binary)
        elif controltype == "RND":
            X1_ctrl = randomize_data(exp_X1, randomization_radius)
            # X1_ctrl = exp_X1
            if univariate:
                X2_ctrl = X1_ctrl
            else:
                X2_ctrl = randomize_data(exp_X2, randomization_radius)
        else:
            raise NotImplementedError()
        if metric == "RK":
            K_csr.append(ripley_K(X1_ctrl, X2_ctrl, radii, area))
        elif metric == "RH":
            K_csr.append(ripley_H(X1_ctrl, X2_ctrl, radii, area))
        elif metric == "RDF":
            K_csr.append(
                radial_distribution_function(
                    X1_ctrl, X2_ctrl, radii, univariate
                )
            )
        elif metric == "1NN":
            K_csr.append(first_nn(X1_ctrl, X2_ctrl, radii, univariate))
        else:
            raise NotImplementedError()
    K_csr = np.array(K_csr)

    def posdiffpreidx(K_data, K_ctrl_mean, idx):
        K_norm = K_data - K_ctrl_mean
        K_norm[idx:] = 0
        K_norm[K_norm < 0] = 0
        return K_norm

    if normalization == "zscore":
        K_exp_norm = normalize_to_CSR(K_exp, K_csr)
        K_csr_norm = np.array([normalize_to_CSR(K_c, K_csr) for K_c in K_csr])
    elif normalization == "diff":
        K_csr_mean = np.mean(K_csr, axis=0)
        K_exp_norm = K_exp - K_csr_mean
        K_csr_norm = np.array([K_c - K_csr_mean for K_c in K_csr])
    elif normalization == "deltaAprepeak":
        K_csr_mean = np.mean(K_csr, axis=0)
        peak_idx = np.argmax(K_csr_mean)

        K_exp_norm = posdiffpreidx(K_exp, K_csr_mean, peak_idx)
        K_csr_norm = np.array(
            [posdiffpreidx(K_c, K_csr_mean, peak_idx) for K_c in K_csr]
        )
    elif normalization == "deltaAprepeakNorm":
        N2 = len(exp_X2)
        K_csr_mean = np.mean(K_csr, axis=0)
        peak_idx = np.argmax(K_csr_mean)

        K_exp_norm = posdiffpreidx(K_exp, K_csr_mean, peak_idx) / N2
        K_csr_norm = (
            np.array(
                [posdiffpreidx(K_c, K_csr_mean, peak_idx) for K_c in K_csr]
            )
            / N2
        )
    elif normalization == "deltaAprepeakPerc":
        N2 = len(exp_X2)
        K_csr_mean = np.mean(K_csr, axis=0)
        peak_idx = np.argmax(K_csr_mean)

        K_exp_norm = 100 * posdiffpreidx(K_exp, K_csr_mean, peak_idx) / N2
        K_csr_norm = 100 * (
            np.array(
                [posdiffpreidx(K_c, K_csr_mean, peak_idx) for K_c in K_csr]
            )
            / N2
        )
    elif normalization == "to_max_r":

        def norm_to_max_r(K_data):
            K_norm = K_data / K_data[-1]
            return K_norm

        K_exp_norm = norm_to_max_r(K_exp)
        K_csr_norm = np.array([norm_to_max_r(K_c) for K_c in K_csr])

    # r_max = radii.max()
    # ripley_integral = np.trapz(K_exp_norm, radii) / r_max
    ripley_integral = np.trapz(K_exp_norm, radii)
    if ax_u is not None and ax_n is not None:
        plot_ripleys(
            radii,
            K_exp,
            K_csr,
            ci=0.95,
            normalized=False,
            showControls=True,
            title=f"{name1} -> {name2}",
            labelFontsize=30,
            axes=ax_u,
            metric=metric,
            showControlEnvelope=showControlEnvelope,
        )
        plot_ripleys(
            radii,
            K_exp_norm,
            K_csr_norm,
            ci=0.95,
            normalized=True,
            showControls=True,
            title=f"{name1} -> {name2}",
            labelFontsize=30,
            axes=ax_n,
            metric=metric,
            showControlEnvelope=showControlEnvelope,
        )
    return ripley_integral, K_exp, K_exp_norm


def analyze_all_channels(
    mol_coords,
    mask,
    mask_pixel_size,
    area,
    radii,
    n_simulations,
    do_plot=True,
    names="",
    controltype="CSR",  # CSR or RND
    metric="RK",  # RK or RDF
    randomization_radius=None,
    normalization="zscore",
    aggfun="mean",
    showControlEnvelope=True,
):
    """Do the neighborhood analysis of all channels with each other
    and generate a mean value matrix. This has been initially written for
    Ripley's K analysis, but can also be used for the radial distribution
    function.
    Args:
        mol_coords : list of np.rec.arrays
            the molecular coordinates of target types
        mask : 2D np array - outpost_modules.mask.CellMask
            the binary or density mask. Only needed if controltype is "CSR"
        mask_pixel_size : float
            the pixel size of the mask, in nm. Only needed if controltype is "CSR"
        area : float
            the area, in square nm (?). Only needed if metric is "RK"
        radii : np array 1D
            the radii to probe at
        n_simulations : int
            the number of controls to do
        do_plot : bool
            whether to create a plot of all the metric curves in a matrix subplot
        names : list of str
            the names of the molecular types
        randomization_radius : float
            defines the maximum length of the randomization vectors if controltype
            is "RND"
        metric : str
            the metric to calculate from the data
            "RK": Ripley's K
            "RH": Ripley's H
            "RDF": Radial Distribution function
            "1NN": first nearest neighbor distance distribution
        controltype : str
            the method of creating controls
            "CSRbin": draw random points from a previously generated mask,
                using binary mask information -> CSR
            "CSRdens": draw random points from a previously generated mask,
                with weighting density (of all targets combined)
            "RND": randomization of experimental data by uniformly relocating
                datapoints by a vector on a circle with randomization_radius
        normalization : str
            the method of normalizing experimental data
            "zscore" (units of  95% ci from mean of conrols)
            "diff" (difference to mean of controls)
            "deltaAprepeak" (only positive difference to mean of controls,
                before peak of mean of controls)
        aggfun : str
            the aggregation of normalized curves to values to put into the matrix
            "mean", "sum"
            default: "mean"
    """
    n_targets = len(mol_coords)
    curves = np.zeros((n_targets, n_targets, len(radii)))
    curves_norm = np.zeros((n_targets, n_targets, len(radii)))
    if do_plot:
        fig_n, ax_n = init_plot(n_targets, "normalized", controltype, metric)
        fig_u, ax_u = init_plot(
            n_targets, "un-normalized", controltype, metric
        )
    else:
        fig_u, ax_u = None, None
        fig_n, ax_n = None, None
    if not names:
        names = [""] * n_targets
    ripley_matrix = np.zeros((n_targets, n_targets), dtype=np.float64)
    for i, X1 in enumerate(mol_coords):
        for j, X2 in enumerate(mol_coords):
            # print(f"Analyzing interaction between receptor {i} and {j}...")
            ripley_integral, K_exp, K_exp_norm = analyze_2_channels(
                X1,
                X2,
                mask,
                mask_pixel_size,
                area,
                radii=radii,
                n_simulations=n_simulations,
                ax_u=ax_u[i, j],
                ax_n=ax_n[i, j],
                name1=names[i],
                name2=names[j],
                controltype=controltype,
                metric=metric,
                randomization_radius=randomization_radius,
                normalization=normalization,
                showControlEnvelope=showControlEnvelope,
            )
            curves[i, j, :] = K_exp
            curves_norm[i, j, :] = K_exp_norm
            if i < n_targets - 1:
                ax_u[i, j].xaxis.label.set_visible(False)
                ax_n[i, j].xaxis.label.set_visible(False)
                # ax_u[i, j].set_xticks([])
                # ax_n[i, j].set_xticks([])
                # due to sharex=True this is not necessary any more.
                pass
            if j > 0:
                ax_u[i, j].yaxis.label.set_visible(False)
                ax_n[i, j].yaxis.label.set_visible(False)

            if ripley_integral is np.nan:
                ripley_integral = 0
            if aggfun == "mean":
                ripley_mean = ripley_integral / (np.max(radii) - np.min(radii))
            elif aggfun == "sum":
                ripley_mean = ripley_integral
            else:
                raise NotImplementedError(f"aggfun {aggfun} not implemented.")
            ripley_matrix[i, j] = ripley_mean
    return ripley_matrix, fig_u, fig_n, curves, curves_norm


def typefraction_all_channels(
    mol_coords,
    radii,
    n_simulations,
    do_plot=True,
    names="",
    shuffle_self=True,
    relocate_self=False,
    fraction_exclude_self=False,
    normalize_to_bulkfraction=False,
    showControlEnvelope=None,
):
    """
    Args:
        mol_coords : list of np.rec.arrays
            the molecular coordinates of target types
        radii : np array 1D
            the radii to probe at
        n_simulations : int
            the number of controls to do
        do_plot : bool
            whether to create a plot of all the metric curves in a
            matrix subplot
        names : list of str, len N
            the names of the molecular types
        shuffle_self : bool
            whether to shuffle only other types or also the self type
        relocate_self : bool
            whether to relocate centerpoints to 'type_self' after
            shuffling.
        fraction_exclude_self : bool
            Whether to exclude self type from normalization when caluclating
            type fractions
        normalize_to_bulkfraction : bool
            if False, normalize to the controls,
            if True, normalize to the fraction at largest r. Sets
                n_simulations to 0
    Returns:
        curves : 3D array (N, N, len(radii))
            the fraction curves
        curves_norm : 3D array
            the normalized fraction curves
    """
    if normalize_to_bulkfraction:
        n_simulations = 0
    types = np.concatenate(
        [i * np.ones(len(coords)) for i, coords in enumerate(mol_coords)]
    )
    all_coords = np.concatenate(mol_coords)
    n_targets = len(mol_coords)

    curves = np.zeros((n_targets, n_targets, len(radii)))
    curves_norm = np.zeros((n_targets, n_targets, len(radii)))
    if do_plot:
        fig_n, ax_n = init_plot(
            n_targets, "normalized", "shuffle", "type fraction"
        )
        fig_u, ax_u = init_plot(
            n_targets, "un-normalized", "shuffle", "type fraction"
        )
    else:
        fig_u, ax_u = None, None
        fig_n, ax_n = None, None
    if not names:
        names = [""] * n_targets
    ripley_matrix = np.zeros((n_targets, n_targets), dtype=np.float64)
    for i, X1 in enumerate(mol_coords):
        if fraction_exclude_self:
            fraction_exclude = i
        else:
            fraction_exclude = None
        fract_types, fract_types_ctrl = fraction_types(
            all_coords,
            types,
            i,
            radii,
            nshuffle=n_simulations,
            shuffle_self=shuffle_self,
            relocate_self=relocate_self,
            fraction_exclude=fraction_exclude,
        )
        for j, X2 in enumerate(mol_coords):
            name1 = names[i]
            name2 = names[j]
            if not normalize_to_bulkfraction:
                K_exp = fract_types[j]
                K_csr = fract_types_ctrl[j]

                K_exp_norm = normalize_to_CSR(K_exp, K_csr)
                K_csr_norm = np.array(
                    [normalize_to_CSR(K_c, K_csr) for K_c in K_csr]
                )
                curves[i, j, :] = K_exp
                curves_norm[i, j, :] = K_exp_norm
                ripley_matrix[i, j] = np.mean(curves_norm[i, j, :])
                show_controls = True
                if showControlEnvelope is None:
                    showControlEnvelope = True
            else:
                curves[i, j, :] = fract_types[j]
                curves_norm[i, j, :] = fract_types[j] / fract_types[j][-1]
                ripley_matrix[i, j] = np.mean(curves_norm[i, j, :] - 1)
                K_exp = curves[i, j, :]
                K_exp_norm = curves_norm[i, j, :]
                K_csr = K_exp[np.newaxis, ...]
                K_csr_norm = K_csr
                show_controls = False
                if showControlEnvelope is None:
                    showControlEnvelope = False

            if ax_u is not None and ax_n is not None:
                plot_ripleys(
                    radii,
                    K_exp,
                    K_csr,
                    ci=0.95,
                    normalized=False,
                    showControls=show_controls,
                    showControlEnvelope=showControlEnvelope,
                    title=f"{name1} -> {name2}",
                    labelFontsize=30,
                    axes=ax_u[i, j],
                    metric="type fraction",
                )
                plot_ripleys(
                    radii,
                    K_exp_norm,
                    K_csr_norm,
                    ci=0.95,
                    normalized=True,
                    showControls=show_controls,
                    showControlEnvelope=showControlEnvelope,
                    title=f"{name1} -> {name2}",
                    labelFontsize=30,
                    axes=ax_n[i, j],
                    metric="type fraction",
                )

            if i < n_targets - 1:
                ax_u[i, j].xaxis.label.set_visible(False)
                ax_n[i, j].xaxis.label.set_visible(False)
                ax_u[i, j].set_xticks([])
                ax_n[i, j].set_xticks([])
            if j > 0:
                ax_u[i, j].yaxis.label.set_visible(False)
                ax_n[i, j].yaxis.label.set_visible(False)

    return ripley_matrix, fig_u, fig_n, curves, curves_norm


def analyze(mols, radii):
    """Run ripley's K analysis (as per the paper's methods) on one cell.

    Parameters
    ----------
    mols : list of np.rec.arrays
        Picasso-format molecule lists. Each list element contains the
        molecular positions of one species (e.g., CD80).

    Returns
    -------
    ripley_matrix : np.2darray
        Raw NxN array containing the Ripley's K integral values for each
        pair of molecular species. N is the number of targets, which
        is the length of the mols list.
    """
    mol_coords = [convert_picasso_to_coords(mol) for mol in mols]
    mask, area = get_cell_mask(mol_coords)
    ripley_matrix = analyze_all_channels(mol_coords, mask, area, radii)
    return ripley_matrix


def postprocess_ripley_matrix(ripley_matrix, radii):
    """Set values to zero if they lie within the 95% CI of the CSR
    simulations. Prior normalization sets 95% CI to +/- 1.
    Args:
        ripley_matrix : 2D np.array N x N
            matrix of normalized ripley's mean values between all
            N pairs of target molecules.
        radii : 1D np.array
            the radii probed [nm]
    """
    postprocessed = ripley_matrix.copy()
    ci = 1
    postprocessed[(postprocessed < ci) & (postprocessed > -ci)] = 0
    return postprocessed


def init_plot(n_targets, treatment, controltype, metric, figsize_per_target=5):
    fig, ax = plt.subplots(
        n_targets,
        n_targets,
        figsize=(
            int(n_targets * figsize_per_target),
            int(n_targets * figsize_per_target),
        ),
        sharey=True,
        sharex=True,
    )
    fig.suptitle(f"{metric}, {treatment} to {controltype}")
    return fig, ax


def plot_ripleys(
    radii,
    Kexp,
    Kctrl,
    ci=0.95,
    normalized=True,
    showControls=False,
    title=None,
    labelFontsize=14,
    axes=None,
    metric="",
    label_data="Observed data",
    showControlEnvelope=True,
):
    # Plot Ripley's K and confidence interval
    if axes is None:
        plt.figure()
        axes = plt.gca()

    # show controls
    if showControls:
        for k, Kct in enumerate(Kctrl):
            axes.plot(
                radii,
                Kct,
                c="lightgray",
                label="Random controls",
                linestyle="-",
            )
    axes.set_xlabel("d [nm]", fontsize=labelFontsize)
    if showControlEnvelope:
        if normalized:
            axes.plot(
                radii,
                np.zeros(len(radii)),
                c="k",
                label=f"{ci*100}% envelope",
                linestyle="--",
            )
            axes.plot(radii, np.ones(len(radii)), c="k", linestyle=":")
            axes.plot(radii, -np.ones(len(radii)), c="k", linestyle=":")
            axes.set_xlabel("d [nm]", fontsize=labelFontsize)
            axes.set_ylabel(f"Normalized {metric}", fontsize=labelFontsize)
        else:
            quantileLow = (1 - ci) / 2
            quantileHigh = 1 - (1 - ci) / 2
            axes.plot(
                radii,
                np.mean(Kctrl, axis=0),
                c="k",
                label="Mean of random controls",
                linestyle="--",
            )
            axes.plot(
                radii,
                np.quantile(Kctrl, quantileHigh, axis=0),
                c="k",
                label=f"{ci*100}% envelope",
                linestyle=":",
            )
            axes.plot(
                radii,
                np.quantile(Kctrl, quantileLow, axis=0),
                c="k",
                linestyle=":",
            )
            axes.set_ylabel(metric, fontsize=labelFontsize)

    # show data
    axes.plot(
        radii,
        Kexp,
        c="k",
        linewidth=2.0,
        label=label_data,
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=labelFontsize)

    if title is not None:
        axes.set_title(title, fontsize=labelFontsize)

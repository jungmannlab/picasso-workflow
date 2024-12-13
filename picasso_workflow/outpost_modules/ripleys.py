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
from scipy.ndimage import zoom, gaussian_filter


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
    if np.array_equal(X1, X2):
        return univariate_ripley_K(X1, r, area)
    else:
        return bivariate_ripley_K(X1, X2, r, area)


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


def ripley_K_CSR(n_points, mask, area, radii, n_simulations=100):
    # note that n_points is either a tuple of the number of points for
    # each of the 2 species (if we're doing cross-Ripley) or just the
    # number of points for one species (if we're doing univariate Ripley)
    X = simulate_CSR(n_points, mask, n_simulations)
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
    mask_final /= mask_final.sum()
    if binary:
        mask_final[mask_final > 0] = 1
    return mask_final, area


def plot_mask(mask, pixelsize, fp):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.set_title("mask - final")
    ax.imshow(
        mask,
        extent=[0, pixelsize * mask.shape[0], 0, pixelsize * mask.shape[1]],
        cmap="hot",
    )

    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    ax.set_xticks()
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
    area,
    radii,
    n_simulations,
    ax_u,
    ax_n,
    name1="",
    name2="",
):
    """Runs the analysis of any two channels of the dataset (2 protein
    species)."""

    if np.array_equal(exp_X1, exp_X2):
        n_points = len(exp_X1)
    else:
        n_points = (len(exp_X1), len(exp_X2))
    K_exp = ripley_K(exp_X1, exp_X2, radii, area)
    K_csr = ripley_K_CSR(
        n_points, mask, area, radii=radii, n_simulations=n_simulations
    )
    K_exp_norm = normalize_to_CSR(K_exp, K_csr)
    # r_max = radii.max()
    # ripley_integral = np.trapz(K_exp_norm, radii) / r_max
    ripley_integral = np.trapz(K_exp_norm, radii)
    if ax_u is not None and ax_n is not None:
        plot_ripleys(
            radii,
            K_exp,
            K_csr,
            ci=0.95,
            normalized=True,
            showControls=True,
            title=f"{name1} -> {name2}",
            labelFontsize=30,
            axes=ax_n,
        )
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
        )
    return ripley_integral


def analyze_all_channels(
    mol_coords, mask, area, radii, n_simulations, do_plot=True, names=""
):
    n_targets = len(mol_coords)
    if do_plot:
        fig_u, ax_u = init_plot(n_targets)
        fig_n, ax_n = init_plot(n_targets)
    else:
        fig_u, ax_u = None, None
        fig_n, ax_n = None, None
    if not names:
        names = [""] * n_targets
    ripley_matrix = np.zeros((n_targets, n_targets), dtype=np.float64)
    for i, X1 in enumerate(mol_coords):
        for j, X2 in enumerate(mol_coords):
            # print(f"Analyzing interaction between receptor {i} and {j}...")
            val = analyze_2_channels(
                X1,
                X2,
                mask,
                area,
                radii=radii,
                n_simulations=n_simulations,
                ax_u=ax_u,
                ax_n=ax_n,
                name1=names[i],
                name2=names[j],
            )
            if val is np.nan:
                val = 0
            ripley_matrix[i, j] = val
    return ripley_matrix, fig_u, fig_n


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
    # set values to zero if they lie withing the 95% CI of the CSR
    # simulations
    postprocessed = ripley_matrix.copy()
    ci = radii.max() - radii.min()
    postprocessed[(postprocessed < ci) & (postprocessed > -ci)] = 0
    return postprocessed


def init_plot(n_targets, figsize=30):
    fig, ax = plt.subplots(n_targets, n_targets, figsize=(figsize, figsize))
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
):
    # Plot Ripley's K and confidence interval
    if axes is None:
        plt.figure()
        axes = plt.gca()

    # show data
    axes.plot(
        radii,
        Kexp,
        c="k",
        linewidth=2.0,
        label="Observed data",
    )
    # show controls
    if showControls:
        for k in range(Kctrl):
            axes.plot(
                radii,
                Kctrl,
                c="lightgray",
                label="Random controls",
                linestyle="-",
            )
    axes.set_xlabel("d [nm]", fontsize=labelFontsize)
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
        axes.set_ylabel("Normalized K(d)", fontsize=labelFontsize)
    else:
        quantileLow = (1 - ci) / 2
        quantileHigh = 1 - (1 - ci) / 2
        axes.plot(
            radii,
            np.mean(Kctrl, axis=1),
            c="k",
            label="Mean of random controls",
            linestyle="--",
        )
        axes.plot(
            radii,
            np.quantile(Kctrl, quantileHigh, axis=1),
            c="k",
            label=f"{ci*100}% envelope",
            linestyle=":",
        )
        axes.plot(
            radii,
            np.quantile(Kctrl, quantileLow, axis=1),
            c="k",
            linestyle=":",
        )
        axes.set_ylabel("K(d)", fontsize=labelFontsize)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=labelFontsize)

    if title is not None:
        axes.set_title(title, fontsize=labelFontsize)

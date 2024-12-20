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
    alldist, indices = tree1.query(tree2, k=k)
    alldist = np.sort(alldist, axis=1)
    alldist = alldist[:, k - 1]
    nnhist, _ = np.histogram(alldist, bins=r)
    return nnhist


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


def plot_mask(mask, pixelsize, fp):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.set_title("mask - final")
    # check if mask is binary
    if len(np.unique(mask)) == 2:
        mask_plot = mask.copy()
        mask_plot[mask_plot > 0] = 1
        mask_plot[mask_plot < 1] = 0
        mask_plot = mask_plot.astype(np.int8)
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
    controltype="CSR",  # CSR or RND
    metric="RK",  # RK or RDF or 1NN
    randomization_radius=None,
):
    """Runs the analysis of any two channels of the dataset (2 protein
    species)."""

    if np.array_equal(exp_X1, exp_X2):
        n_points = len(exp_X1)
        univariate = True
    else:
        n_points = (len(exp_X1), len(exp_X2))
        univariate = False

    if metric == "RK":
        K_exp = ripley_K(exp_X1, exp_X2, radii, area)
    elif metric == "RDF":
        K_exp = radial_distribution_function(exp_X1, exp_X2, radii, univariate)
    elif metric == "1NN":
        K_exp = first_nn(exp_X1, exp_X2, radii, univariate)
    else:
        raise NotImplementedError()

    K_csr = []
    for i in range(n_simulations):
        if controltype == "CSR":
            X_ctrl = simulate_CSR(n_points, mask, 1, mask_pixel_size)
            X1_ctrl = X_ctrl[0][0]
            X2_ctrl = X_ctrl[1][0]
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
            K_csr.append(ripley_K(X1_ctrl, X1_ctrl, radii, area))
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

    K_exp_norm = normalize_to_CSR(K_exp, K_csr)
    K_csr_norm = np.array([normalize_to_CSR(K_c, K_csr) for K_c in K_csr])
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
        )
    return ripley_integral


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
):
    n_targets = len(mol_coords)
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
            ripley_integral = analyze_2_channels(
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
            )
            if j < n_targets - 1:
                ax_u[i, j].xaxis.label.set_visible(False)
                ax_n[i, j].xaxis.label.set_visible(False)
                ax_u[i, j].set_xticks([])
                ax_n[i, j].set_xticks([])
            if i > 0:
                ax_u[i, j].yaxis.label.set_visible(False)
                ax_n[i, j].yaxis.label.set_visible(False)

            if ripley_integral is np.nan:
                ripley_integral = 0
            ripley_mean = ripley_integral / (np.max(radii) - np.min(radii))
            ripley_matrix[i, j] = ripley_mean
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


def init_plot(n_targets, treatment, controltype, metric, figsize=30):
    fig, ax = plt.subplots(n_targets, n_targets, figsize=(figsize, figsize))
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
        for k, Kct in enumerate(Kctrl):
            axes.plot(
                radii,
                Kct,
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

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=labelFontsize)

    if title is not None:
        axes.set_title(title, fontsize=labelFontsize)

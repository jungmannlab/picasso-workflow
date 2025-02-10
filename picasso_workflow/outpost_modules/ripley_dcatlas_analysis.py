"""This script tries to recreate Magdalena's (and Valerio's) ripley 
analysis of the DC-Atlas data. However, we do it the way the god 
intended. This means, the software is reproducable!

This script provides the functions for analysis, it will not analyze
all the data (please refer to another script for that). The final 
function "analyze_cell" will analyze a single cell with 6 protein 
species.

Outline of the pipeline:
1. Extract the molecular positions of all 6 protein species from the
    Picasso-format files.
2. Calculate the binary mask based on the molecular positions of all 6
    protein species.
    * Bin molecule positions into bins of size of 20 nm, Gaussian blur
      by 400 nm, use 1/3 of otsu threshold on such an image and upsample
      to 10 nm bin size.
3. Run the Ripley's K analysis on all 6 protein species pairs:
    * Find Ripley's K for the experimental data.
    * Simulate 100 CSR datasets in the cell mask, with the same 
        (similar) number of points as the experimental data.
    * Find Ripley's K for each CSR dataset.
    * Normalize the experimental Ripley's K to the CSR datasets by
        subtracting the mean CSR Ripley's K and dividing by the
        difference between the 95th percentile of the CSR Ripley's K
        and the mean CSR Ripley's K.
    * Calculate the Ripley's K integral for the normalized Ripley's K.
4. Return the resulting 6x6 matrix of Ripley's K integrals.


author: Rafal Kowalewski
date: 21 Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from picasso import io
from picasso_workflow.outpost_modules.ripleys import init_plot, plot_ripleys

BLUR_FACTOR = 0.3 # gaussian blur factor for the mask, cam. pixels
PIXELSIZE = 130 # camera pixel size, nm (Picasso-format specifies positions in cam. pixels)
R_MAX = 200 # nm, maximum radius for Ripley's K analysis
RADII = np.concatenate((np.arange(4, 80, 2), np.arange(80, R_MAX+1, 12)))

def otsu(image):
    """Simplified function from scikit-image so that i do not need to 
    install the whole package."""

    # histogram the image and converts bin edges to bin centers
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype('float32', copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

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
    
def simulate_binary_mask_CSR(n_points, mask, area, n_simulations=100, pixelsize=10): #TODO: test that this works
    """Simulates CSR within the mask by simulating a CSR in a square of 
    equivalent area at the given density and then masking it, same
    as in Magdalena's script.

    n_points - number of points to simulate (int for one species, tuple
               for two species)
    mask - binary mask of the cell
    area - area of the cell in um^2
    n_simulations - number of simulations to run
    pixelsize - mask pixel size in nm, see get_cell_mask - upsample

    returns X - two lists with n_simulations elements, 1 for each 
                simulation. Each element is a 2D numpy array with the
                simulated points coordinates in nm. Note that n_points
                are not simulated, instead an approximate number of
                points is simulated. The first list is for the first
                species, the second for the second species.
    """

    # convert area to the units of mask bin size (from nm^2 to cam. pixels)
    area /= pixelsize ** 2
    image_area = mask.size # units: mask bin size
    if isinstance(n_points, int):
        density = (n_points / area)
        n_sample = int(density * image_area) # density points in mask, scale to whole image
        
        # create points in units of pixels
        X = np.random.uniform(0, mask.shape[0], size=(n_simulations, n_sample, 2)) 
        
        # Reject points outside of mask
        x_ind = (np.floor(X[:, :, 0])).astype(int)
        y_ind = (np.floor(X[:, :, 1])).astype(int)
        index = mask[y_ind, x_ind].astype(bool)
        X = [pixelsize * X[i][index[i]] for i in range(X.shape[0])]
        return X, X   
    else:
        density1 = (n_points[0] / area)
        density2 = (n_points[1] / area)
        n_sample1 = int(density1 * image_area) # density points in mask, scale to whole image
        n_sample2 = int(density2 * image_area) # density points in mask, scale to whole image
        X1 = np.random.uniform(0, mask.shape[0], size=(n_simulations, n_sample1, 2))
        X2 = np.random.uniform(0, mask.shape[0], size=(n_simulations, n_sample2, 2))
        x_ind1 = (np.floor(X1[:, :, 0])).astype(int)
        y_ind1 = (np.floor(X1[:, :, 1])).astype(int)
        x_ind2 = (np.floor(X2[:, :, 0])).astype(int)
        y_ind2 = (np.floor(X2[:, :, 1])).astype(int)
        index1 = mask[y_ind1, x_ind1].astype(bool)
        index2 = mask[y_ind2, x_ind2].astype(bool)
        X1 = [pixelsize * X1[i][index1[i]] for i in range(X1.shape[0])]
        X2 = [pixelsize * X2[i][index2[i]] for i in range(X2.shape[0])]
        return X1, X2

def simulate_density_mask_CSR_lower(n_points, mask, n_simulations, pixelsize):
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

def simulate_density_mask_CSR(n_points, mask, area, n_simulations=100, pixelsize=10): #TODO: test that this works
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
        X = simulate_density_mask_CSR_lower(n_points, mask, n_simulations, pixelsize)
        return X, X
    else:
        X1 = simulate_density_mask_CSR_lower(n_points[0], mask, n_simulations, pixelsize)
        X2 = simulate_density_mask_CSR_lower(n_points[1], mask, n_simulations, pixelsize)
        return X1, X2


def ripley_K_CSR(n_points, mask, area, radii=RADII, n_simulations=100):
    # note that n_points is either a tuple of the number of points for 
    # each of the 2 species (if we're doing cross-Ripley) or just the
    # number of points for one species (if we're doing univariate Ripley)
    n_mask_unique_vals = len(np.unique(mask))
    if n_mask_unique_vals < 5:
        binary = True
    else:
        binary = False
    if binary:
        X = simulate_binary_mask_CSR(n_points, mask, area, n_simulations)
    else:
        X = simulate_density_mask_CSR(n_points, mask, area, n_simulations)
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
    quantiles_high = np.array([np.quantile(x, quantile_high) for x in np.transpose(K_csr)])
    divider_high = np.abs(quantiles_high - K_csr_mean)
    idx_pos_final = idx_pos & (divider_high != 0)
    K_exp_norm[idx_pos_final] /= divider_high[idx_pos_final]

    quantiles_low = np.array([np.quantile(x, quantile_low) for x in np.transpose(K_csr)])
    divider_low = np.abs(quantiles_low - K_csr_mean)
    idx_neg_final = ~idx_pos & (divider_low != 0)
    K_exp_norm[idx_neg_final] /= divider_low[idx_neg_final]
    return K_exp_norm    

def get_cell_binary_mask(
    mol_coords, pixelsize=130, binsize=20, blur=20, threshold=1/3, upsample=10
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
    
    Returns
    -------
    mask : np.2darray (bool)
        Binary mask of the cell.
    area : float
        Area of the cell in um^2.
    """

    # # combine all coordinates into one array
    # combined_coords = np.vstack(mol_coords) / pixelsize#TODO: test this is correct
    # # first step is to histogram the coordinates (1 bin - 1 camera pixel)
    # bins = np.arange(0, 512, 1, dtype=np.float64)
    # mask = np.histogram2d(
    #     combined_coords[:, 0], combined_coords[:, 1], bins=bins, density=False
    # )[0]
    # # have the same shape as picasso's rendered image
    # mask = np.flipud(np.rot90(mask))
    # # then apply a gaussian blur
    # mask = gaussian_filter(mask, BLUR_FACTOR)
    # # threshold the mask - if less than 1 molecule per pixel is found, 
    # # the pixel is set to 0, otherwise to 1
    # mask = mask > 1
    # # calculate the area of the cell in nm^2
    # area = mask.sum() * pixelsize ** 2
    # return mask, area

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
    mask = mask > thresh
    # upsample with scipy.ndimage.zoom
    factor = int(binsize / upsample)
    mask_zoomed = zoom(np.array(mask, dtype=float), factor)
    mask_final = mask_zoomed > 0.5 # idk why Susanne used this in the end
    area = mask_final.astype(float).sum() * upsample ** 2 # area in nm^2, convert to float to avoid overflow
    return mask_final, area


def get_cell_density_mask(
    mol_coords, pixelsize=130, binsize=20, blur=20, threshold=1/3, upsample=10
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
    area = (mask_final > 0).sum() * upsample ** 2
    mask_final /= mask_final.sum()
    return mask_final, area
    # mask = mask > thresh
    # # upsample with scipy.ndimage.zoom
    # factor = int(binsize / upsample)
    # mask_zoomed = zoom(np.array(mask, dtype=float), factor)
    # mask_final = mask_zoomed > 0.5 # idk why Susanne used this in the end
    # area = mask_final.sum() * upsample ** 2 # area in nm^2
    # return mask_final, area


def convert_picasso_to_coords(mols, pixelsize=PIXELSIZE):
    """Converts the Picasso-format np.rec.array to a 2D numpy array with
    spatial coordinates in nm."""

    return np.array([mols['x'], mols['y']]).T * pixelsize

def analyze_2_channels(exp_X1, exp_X2, mask, area, radii=RADII, ax_n=None, ax_u=None, name1="", name2=""): 
    """Runs the analysis of any two channels of the dataset (2 protein
    species)."""
    
    if np.array_equal(exp_X1, exp_X2):
        n_points = len(exp_X1)
    else:
        n_points = (len(exp_X1), len(exp_X2))
    K_exp = ripley_K(exp_X1, exp_X2, radii, area)
    K_csr = ripley_K_CSR(n_points, mask, area, radii=radii, n_simulations=20)
    K_exp_norm = normalize_to_CSR(K_exp, K_csr)
    K_csr_norm = [normalize_to_CSR(K_c, K_csr) for K_c in K_csr]
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
            metric="RK",
            # showControlEnvelope=showControlEnvelope,
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
            metric="RK",
            # showControlEnvelope=showControlEnvelope,
        )
    # r_max = radii.max()
    # ripley_integral = np.trapz(K_exp_norm, radii) / r_max
    ripley_integral = np.trapz(K_exp_norm, radii)
    return ripley_integral

def analyze(mols, radii=RADII, binary=True):
    """Run ripley's K analysis (as per the paper's methods) on one cell.
    
    Parameters
    ----------
    mols : list of np.rec.arrays
        Picasso-format molecule lists. Each list element contains the
        molecular positions of one species (e.g., CD80).
        
    Returns
    -------
    ripley_matrix : np.2darray
        Raw 6x6 array containing the Ripley's K integral values for each 
        pair of molecular species.
    """

    ripley_matrix = np.zeros((6, 6), dtype=np.float64)
    mol_coords = [convert_picasso_to_coords(mol) for mol in mols]
    if binary:
        mask, area = get_cell_binary_mask(mol_coords)
        controltype = "CSRbin"
    else:
        mask, area = get_cell_density_mask(mol_coords)
        controltype = "CSRdens"
    # area = area * 1e6  # in nm^2
    fig_n, ax_n = init_plot(
        n_targets=6, treatment="normalized", controltype=controltype, metric="RK", figsize_per_target=5)
    fig_u, ax_u = init_plot(
        n_targets=6, treatment="unnormalized", controltype=controltype, metric="RK", figsize_per_target=5)
    for i, X1 in enumerate(mol_coords):
        for j, X2 in enumerate(mol_coords):
            # print(f"Analyzing interaction between receptor {i} and {j}...")
            val = analyze_2_channels(X1, X2, mask, area, radii=radii, ax_n=ax_n[i, j], ax_u=ax_u[i, j])
            if val is np.nan:
                val = 0
            ripley_matrix[i, j] = val
    return ripley_matrix, mask, area, fig_u, fig_n

def postprocess_ripley_matrix(ripley_matrix, radii):
    # set values to zero if they lie withing the 95% CI of the CSR
    # simulations
    postprocessed = ripley_matrix.copy()
    ci = radii.max() - radii.min()
    postprocessed[(postprocessed < ci) & (postprocessed > -ci)] = 0
    return postprocessed



### TESTING ###
if __name__ == "__main__":
    np.random.seed(42)
    import os
    proteins = ["MHC-I", "MHC-II", "CD86", "CD80", "PDL1", "PDL2"]
    cell_path = "/Users/kowalewski/Desktop/dcatlas_ripley_debugging/data/MutuDC_6h_stimulation/Cell_1"
    mols = [
        io.load_locs(os.path.join(cell_path, f"_Receptor_{p}.hdf5"))[0]
        for p in proteins
    ]
    ripley_matrix = analyze(mols, radii=RADII)
    print(ripley_matrix)
    plt.imshow(ripley_matrix, cmap='bwr_r', vmin=-2000, vmax=2000)
    plt.xticks(range(6), proteins)
    plt.yticks(range(6), proteins)
    plt.colorbar()
    plt.show()

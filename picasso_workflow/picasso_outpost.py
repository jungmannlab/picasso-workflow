#!/usr/bin/env python
"""
Module Name: picasso_outpost.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: This is a collection of exploratory DNA-PAINT analysis / picasso
    related functions which if useful should (potentially) be moved into the
    next picasso release. The reasoning to put them here is that it makes
    testing cycles faster.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from aicsimageio import AICSImage

from picasso import localize, render, imageprocess

from scipy.special import gamma as _gamma
from scipy.special import factorial as _factorial
from scipy.optimize import minimize


logger = logging.getLogger(__name__)


def align_channels(
    channel_locs, channel_info, max_iterations=5, convergence=0.001
):
    """This is taken from picasso.gui.render.View.align. As the code is not
    modular enough, it is replicated here. Potentially, this could go into
    a non-gui function in picasso.
    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels
        max_iterations : int
            the maximum number of iterations of alignment
        convergence : float
            convergence criterium when a shift is negligible and thus
            alignment convergence achieved. The value is in pixels.
    Returns:
        shift : list (len 2-3) of lists (len iterations)
            the shifts in x, y, (z) for each iteration, averaged over
            channels (?)
        cumulative_shift : np array (3, channels, iterations)
            the cumulative shift in the three dimensions, in all channels
            the total shift is the last value (in iterations) fo the cum shift
    """
    shift_x = []
    shift_y = []
    shift_z = []
    all_shift = np.zeros((3, len(channel_locs), max_iterations))

    logger.debug("Aligning datasets")

    for iteration in range(max_iterations):
        completed = True

        # find shift between channels
        shift = shift_from_rcc(channel_locs, channel_info)
        logger.debug("Shifting channels.")
        temp_shift_x = []
        temp_shift_y = []
        temp_shift_z = []
        for i, locs_ in enumerate(channel_locs):
            if (
                np.absolute(shift[0][i]) + np.absolute(shift[1][i])
                > convergence
            ):
                completed = False

            # shift each channel
            locs_.y -= shift[0][i]
            locs_.x -= shift[1][i]

            temp_shift_x.append(shift[1][i])
            temp_shift_y.append(shift[0][i])
            all_shift[0, i, iteration] = shift[1][i]
            all_shift[1, i, iteration] = shift[0][i]

            if len(shift) == 3:
                locs_.z -= shift[2][i]
                temp_shift_z.append(shift[2][i])
                all_shift[2, i, iteration] = shift[2][i]
        shift_x.append(np.mean(temp_shift_x))
        shift_y.append(np.mean(temp_shift_y))
        if len(shift) == 3:
            shift_z.append(np.mean(temp_shift_z))

        cumulative_shift = np.cumsum(all_shift, axis=2)

        # Skip when converged:
        if completed:
            break
    shift = [shift_x, shift_y]
    if shift_z != []:
        shift.append(shift_z)
    return shift, cumulative_shift


def plot_shift(shifts, cum_shifts, filepath):
    """Plot the sifts generated by align_channels
    Args:
        shifts : list of 1D array
            the shifts in x, y, and potentially z dimensions
        cum_shifts : 3 D array
            cumulative shifts (dimension, channel, iteration)
        filepath : str
            the filepath to save the plot
    """
    fig, ax = plt.subplots(nrows=1 + len(shifts), sharex=True)
    # ax[0].suptitle("Shift")
    for i, (shift, dim) in enumerate(zip(shifts, ["x", "y", "z"])):
        ax[0].plot(shift, "o-", label=f"{dim} shift")
        ax[1 + i].plot(cum_shifts[i, :, :])
        ax[1 + i].set_ylabel(f"{dim}-shift (Px)")
    ax[0].set_ylabel("Mean Shift (Px)")
    ax[-1].set_xlabel("Iteration")
    fig.set_size_inches((8, 8))
    ax[0].legend(loc="best")
    fig.savefig(filepath)


def shift_from_rcc(channel_locs, channel_info):
    """
    Used by align. Estimates image shifts based on whole images'
    rcc.

    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels

    Returns:
        shifts : tuple
            the channel shifts shape (2,) or (3,) (if z coordinate present)
    """
    n_channels = len(channel_locs)
    images = []
    logger.debug("Rendering localizations.")
    # render each channel and save it in images
    for i, (locs_, info_) in enumerate(zip(channel_locs, channel_info)):
        _, image = render.render(locs_, info_, blur_method="smooth")
        images.append(image)
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    logger.debug(f"Correlating {n_pairs} image pairs.")
    return imageprocess.rcc(images)


def convert_zeiss_file(filepath_czi, filepath_raw, info=None):
    """Convert Zeiss .czi file into a picasso-readable .raw file.
    Args:
        filepath_csi : str
            the filepath to the .czi file to load
        filepath_raw : str
            the filepath to the .raw file to write
        info : dict, default None
            the metadata to make the raw file picasso-readable.
            If None is given, dummy values are entered.
            Necesary keys:
                'Byte Order', 'Camera', 'Micro-Manager Metadata'
    """
    img = AICSImage(filepath_czi)

    with open(filepath_raw, "wb") as f:
        img.get_image_data().squeeze().tofile(f)

    if info is None:
        info = {"Byte Order": "<", "Camera": "FusionBT"}
        info["File"] = filepath_raw
        info["Height"] = img.get_image_data().shape[-2]
        info["Width"] = img.get_image_data().shape[-1]
        info["Frames"] = img.get_image_data().shape[0]
        info["Data Type"] = img.get_image_data().dtype.name
        info["Micro-Manager Metadata"] = {
            "FusionBT-ReadoutMode": 1,
            "Filter": 561,
        }

    filepath_info = os.path.splitext(filepath_raw)[0] + ".yaml"

    with open(filepath_info, "w") as f:
        yaml.dump(info, f)


#############################################################################
# for plotting single spots in analyse.AutoPicasso.
#############################################################################


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return localize._to_photons(spots, camera_info)


def _cut_spots(movie, ids, box):
    N = len(ids.frame)
    spots = np.zeros((N, box, box), dtype=movie.dtype)
    spots = _cut_spots_byrandomframe(
        movie, ids.frame, ids.x, ids.y, box, spots
    )
    return spots


def _cut_spots_byrandomframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie by non-sorted frames.

    Args:
        movie : AbstractPicassoMovie (t, x, y)
            the image data
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    for j, (fr, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        frame = movie[fr]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


def normalize_spot(spot, maxval=255, dtype=np.uint8):
    # logger.debug('spot input: ' + str(spot))
    sp = spot - np.min(spot)
    imgmax = np.max(sp)
    imgmax = 1 if imgmax == 0 else imgmax
    sp = sp.astype(np.float32) / imgmax * maxval
    # logger.debug('spot output: ' + str(sp.astype(dtype)))
    return sp.astype(dtype)


def spinna_temp(parameters_filename):
    """While SPINNA is under development (and the paper being written)
    it is not integrated in the regular picasso package. Here, the
    corresponding module is being loaded.

    Returns:
        result_dir : str
            folder containing the results
        fp_summary : str
            the filepath of the summary csv file
        fp_fig : list of str
            filepaths of the NND figures
    """
    print("importing spinna from outpost")
    from picasso_workflow.spinna_main import _spinna_batch_analysis

    print("starting spinna from outpost")

    result_dir, fp_summary, fp_fig = _spinna_batch_analysis(
        parameters_filename
    )
    print("result_dir", result_dir)
    print("fp_summary", fp_summary)
    print("fp_fig", fp_fig)
    return result_dir, fp_summary, fp_fig


########################################################################
# Begin Log likelihood CSR estimation
########################################################################


def estimate_density_from_neighbordists(
    nn_dists, rho_init, kmin=1, rho_bound_factor=100
):
    """For one point with k nearest neighbor distances (all assumed from
    a CSR distribution), do a maximum likelihood estimation for the
    density.
    Args:
        nn_dists : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
        rho_init : float
            the initial estimation of density
    Returns:
        mle_rho : float
            the maximum likelihood estimate for the local density
            based on the nearest neighbor distances
    """
    bounds = [
        (rho_init / rho_bound_factor, rho_init * rho_bound_factor)
    ]  # rho must be positive
    mle_rho = minimize(
        minimization_loglike,
        x0=[rho_init],
        args=(nn_dists, kmin),
        bounds=bounds,
        # tol=1e-8, options={'maxiter': 1e5}, method='Powell'
        # options={'maxiter': 1e5}, method='L-BFGS-B'
        # method='BFGS'#,
        # options={'maxiter': 1e5, 'gtol': 1e-6, 'eps': 1e-9},
        method="L-BFGS-B",
        options={
            "disp": None,
            "maxcor": 10,
            "ftol": 2e-15,
            "gtol": 1e-15,
            "eps": 1e-15,
            "maxfun": 150,
            "maxiter": 150,
            "iprint": -1,
            "maxls": 100,
            "finite_diff_rel_step": None,
        },
    )
    # print(mle_rho)
    return mle_rho.x[0], mle_rho


def minimization_loglike(rho, nndist_observed, kmin=1):
    """The minimization function for nndist loglikelihood fun
    based on k-th nearest neighbor CSR distributions
    Args:
        rho : list, len 1
            the estimated density
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
    Returns:
        loglike : float
            the log likelihood of finding the observed neighbor distances
            in the model of CSR and given rho
    """
    return -nndist_loglikelihood_csr(nndist_observed, rho[0], kmin)


def nndist_loglikelihood_csr(nndist_observed, rho, kmin=1):
    """get the Log-Likelihood of observed nearest neighbors assuming
    a CSR distribution with density rho.
    Args:
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of one or N spots)
        rho : float
            the density
    Returns:
        log_like : float
            the log likelihood of all distances observed being drawn
            from CSR
    """
    log_like = 0
    # print("nndist_obs shape", nndist_observed.shape)
    for i, dist in enumerate(nndist_observed):
        k = i + kmin
        # print(f"evaluating csr of {len(dist)} spots at k={k}, with rho={rho}")
        # assert False
        prob = nndistribution_from_csr(dist, k, rho)
        log_like += np.sum(np.log(prob))
    return log_like


def nndistribution_from_csr(r, k, rho, d=2):
    """The CSR Nearest Neighbor distribution of finding the k-th nearest
    neighbor at r. with the spatial randomness covering d dimensions
    Args:
        r : float or array of floats
            the distance(s) to evaluate the probability density at
        k : int
            evaluation of the k-th nearest neighbor
        rho : float
            the density
        d : int
            the dimensionality of the problem
    Returns:
        p : same as r
            the probability density of k-th nearest neighbor at r
    """
    # if k != 1:
    #     print(f'evaluating CSR not at k=1 but k={k}')

    # def gaussian_pdf(x, mean, std):
    #     factor = (1 / (np.sqrt(2 * np.pi) * std))
    #     return factor * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # pdf = gaussian_pdf(r, 4, k*rho*4)
    # # pdf = gaussian_pdf(r, 4+k*rho, .8)
    # return pdf #/ np.sum(pdf)
    lam = rho * np.pi ** (d / 2) / _gamma(d / 2 + 1)
    factor = d / _factorial(k - 1) * lam**k * r ** (d * k - 1)
    dist = factor * np.exp(-lam * r**d)
    return dist  # / np.sum(dist)


########################################################################
# End Log likelihood CSR estimation
########################################################################

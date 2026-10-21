"""Analytic CSR (complete spatial randomness) k-NN statistics.

Extracted from ``picasso_outpost.py`` (Phase 2 restructure, monolith split S2).
The analytic nearest-neighbour-distance model under CSR and the
maximum-likelihood density estimator built on it -- a `promote` candidate for
``picasso.postprocess`` (picasso lacks an analytic CSR k-NN model). Pure
numpy/scipy, no picasso coupling.
"""

import numpy as np
from loguru import logger
from scipy.special import gamma as _gamma
from scipy.special import factorial as _factorial
from scipy.optimize import minimize


def _check_distance_window(nn_dists, kmin, min_dist, max_dist):
    """Check that the [min_dist, max_dist] window retains distances.

    Filtering to a window that excludes every distance for some neighbour
    order leaves an empty array, which used to surface far downstream as
    ``ValueError: zero-size array to reduction operation maximum``, with a
    traceback deep inside scipy that never mentions the offending
    parameter. Fail here instead, naming it.

    Parameters
    ----------
    nn_dists : array
        Nearest-neighbour distances, shape ``(k,)`` or ``(k, N)``. Row
        ``r`` holds the distances for neighbour order ``k = r + kmin``.
    kmin : int
        The smallest neighbour order present in ``nn_dists``.
    min_dist, max_dist : float
        The filter window.

    Raises
    ------
    ValueError
        If any neighbour order has no distance inside the window.
    """
    rows = np.atleast_2d(np.asarray(nn_dists, dtype=float))
    for row, dist in enumerate(rows):
        k = row + kmin
        if dist.size == 0:
            continue
        n_keep = int(np.count_nonzero((dist >= min_dist) & (dist <= max_dist)))
        if n_keep == 0:
            raise ValueError(
                f"min_dist={min_dist}, max_dist={max_dist} leaves "
                f"{n_keep} of {dist.size} k={k} nearest-neighbour "
                f"distances (observed range {np.min(dist):.4g}-"
                f"{np.max(dist):.4g}). Widen the [min_dist, max_dist] "
                f"window or lower the number of neighbours fitted."
            )
        if n_keep < 2:
            logger.warning(
                f"min_dist={min_dist}, max_dist={max_dist} leaves only "
                f"{n_keep} of {dist.size} k={k} nearest-neighbour "
                f"distances; the fit for this neighbour order is "
                f"unlikely to be meaningful."
            )


def estimate_density_from_neighbordists(
    nn_dists,
    rho_init,
    kmin=1,
    rho_bound_factor=10,
    d=2,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
    fit_bkg=False,
):
    """Maximum-likelihood-estimate the density from k-NN distances.

    For one point with k nearest-neighbour distances (all assumed drawn from a
    CSR distribution).

    Parameters
    ----------
    nn_dists : array
        The k nearest-neighbour distances, of shape ``(k,)`` or ``(N, k)``
        for N spots.
    rho_init : float
        Initial density estimate.
    kmin : int, optional
        The smallest neighbour order in ``nn_dists``. Default is 1.
    rho_bound_factor : float, optional
        Factor setting the density bounds around ``rho_init``. Default is 10.
    d : int, optional
        Dimensionality. Default is 2.
    min_dist, max_dist : float, optional
        Ignore nn distances outside this range.
    bkg_fraction : float, optional
        Fraction of nn distances that are background (uniformly distributed,
        independent of spot density). Default is 0.
    fit_bkg : bool, optional
        Whether to fit ``bkg_fraction`` rather than take it as given. Default
        is False.

    Returns
    -------
    mle_rho : float
        The maximum-likelihood estimate for the local density.
    result : scipy.optimize.OptimizeResult
        The full minimizer result.
    """
    _check_distance_window(nn_dists, kmin, min_dist, max_dist)

    bounds = [
        (rho_init / rho_bound_factor, rho_init * rho_bound_factor)
    ]  # rho must be positive
    if fit_bkg:
        x0 = [rho_init, bkg_fraction]
    else:
        x0 = [rho_init]
    mle_rho = minimize(
        minimization_loglike,
        x0=x0,
        args=(nn_dists, d, kmin, min_dist, max_dist, bkg_fraction),
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


def minimization_loglike(
    parameters,
    nndist_observed,
    d=2,
    kmin=1,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
):
    """Objective for the NND log-likelihood fit (negative log-likelihood).

    Based on the k-th nearest-neighbour CSR distributions.

    Parameters
    ----------
    parameters : list
        The estimated density (and, if length 2, the ``bkg_fraction``).
    nndist_observed : array
        The k nearest-neighbour distances, shape ``(k,)`` or ``(N, k)``.
    d : int, optional
        Dimensionality. Default is 2.
    kmin : int, optional
        The smallest neighbour order. Default is 1.
    min_dist, max_dist : float, optional
        Ignore nn distances outside this range.
    bkg_fraction : float, optional
        The background fraction; ignored if a second entry of ``parameters``
        is given. Default is 0.

    Returns
    -------
    float
        The negative log-likelihood of the observed neighbour distances under
        the CSR model with the given density.
    """
    rho = parameters[0]
    if len(parameters) == 2:
        bkg_fraction = parameters[1]
    return -nndist_loglikelihood_csr(
        nndist_observed, rho, d, kmin, min_dist, max_dist, bkg_fraction
    )


def nndist_loglikelihood_csr(
    nndist_observed,
    rho,
    d=2,
    kmin=1,
    min_dist=0,
    max_dist=np.inf,
    bkg_fraction=0,
):
    """Log-likelihood of observed nearest neighbours under CSR.

    Parameters
    ----------
    nndist_observed : array
        The k nearest-neighbour distances, shape ``(k,)`` or ``(N, k)`` for
        one or N spots.
    rho : float
        The density.
    d : int, optional
        Dimensionality. Default is 2.
    kmin : int, optional
        The smallest neighbour order. Default is 1.
    min_dist, max_dist : float, optional
        Ignore nn distances outside this range.
    bkg_fraction : float, optional
        The background fraction. Default is 0.

    Returns
    -------
    log_like : float
        The log-likelihood of all observed distances being drawn from CSR.
    """
    log_like = 0
    # print("nndist_obs shape", nndist_observed.shape)
    for i, dist in enumerate(nndist_observed):
        k = i + kmin
        # assert False
        dist_consider = dist[(dist >= min_dist) & (dist <= max_dist)]
        prob = nndistribution_from_csr(
            dist_consider,
            k,
            rho,
            d=d,
            bkg_fraction=bkg_fraction,
            min_dist=min_dist,
            max_dist=max_dist,
        )
        # print(i, dist, prob, np.log(prob))
        log_like += np.sum(np.log(prob))
    return log_like


def nndistribution_from_csr(
    r,
    k,
    rho,
    d=2,
    bkg_fraction=0,
    min_dist=0,
    max_dist=np.inf,
    renormalize=True,
):
    """CSR nearest-neighbour distribution for the k-th neighbour at ``r``.

    For spatial randomness over ``d`` dimensions.

    Parameters
    ----------
    r : float or array of float
        The distance(s) to evaluate the probability density at.
    k : int
        Which nearest neighbour to evaluate.
    rho : float
        The density.
    d : int, optional
        The dimensionality of the problem. Default is 2.
    bkg_fraction : float, optional
        Uniform background fraction added across distances. Default is 0.
    min_dist : float, optional
        Minimum observable distance (e.g. due to technical limits); the model
        is cut off below it and renormalized. Default is 0.
    max_dist : float, optional
        Maximum observable distance. Default is ``np.inf``.
    renormalize : bool, optional
        Whether to renormalize after applying the distance cutoffs. Default is
        True.

    Returns
    -------
    p : same type as ``r``
        The probability density of the k-th nearest neighbour at ``r``.
    """
    # if k != 1:
    #     print(f'evaluating CSR not at k=1 but k={k}')

    # def gaussian_pdf(x, mean, std):
    #     factor = (1 / (np.sqrt(2 * np.pi) * std))
    #     return factor * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # pdf = gaussian_pdf(r, 4, k*rho*4)
    # # pdf = gaussian_pdf(r, 4+k*rho, .8)
    # return pdf #/ np.sum(pdf)
    # An empty r reaches np.max() below and raises an opaque
    # "zero-size array to reduction operation" - callers filtering by
    # [min_dist, max_dist] can easily produce one.
    if np.size(r) == 0:
        return np.asarray(r, dtype=float)

    lam = rho * np.pi ** (d / 2) / _gamma(d / 2 + 1)
    factor = d / _factorial(k - 1) * lam**k * r ** (d * k - 1)
    dist = factor * np.exp(-lam * r**d)
    # add am even background of observed nn distances, at all distances
    if len(r) > 1:
        dist += (bkg_fraction / (np.max(r) - np.min(r))) / (1 + bkg_fraction)
    dist[dist <= 0] = 1e-200  # np.finfo().eps
    # re-normalize
    if min_dist > 0 or max_dist < np.inf:
        r_temp = np.linspace(0, np.max(r), 300)
        prob = nndistribution_from_csr(
            r_temp,
            k,
            rho,
            d=d,
            bkg_fraction=bkg_fraction,
            min_dist=0,
            max_dist=np.inf,
        )
        renorm_fact = np.sum(
            prob[(r_temp >= min_dist) & (r_temp <= max_dist)]
        ) / np.sum(prob)
        # print(renorm_fact)
        dist[(r < min_dist) | (r > max_dist)] = 0
        if renormalize:
            dist = dist / renorm_fact
    # try:
    #     renorm_factor = np.sum(dist) / np.sum(dist[r >= min_dist])
    # except ZeroDivisionError:
    #     renorm_factor = 1
    # dist[r < min_dist] = 0
    # dist *= renorm_factor
    return dist  # / np.sum(dist)


def csr_cdf_for_ks_test(
    x, k, rho, d=2, min_dist=0, max_dist=np.inf, bkg_fraction=0
):
    """CDF of the theoretical CSR distribution for the k-th nearest neighbour.

    Used for Kolmogorov-Smirnov goodness-of-fit testing.

    Parameters
    ----------
    x : float or array-like
        Distance values to evaluate the CDF at.
    k : int
        The k-th nearest-neighbour order.
    rho : float
        Density parameter from the CSR fit.
    d : int, optional
        Dimensionality (2 or 3). Default is 2.
    min_dist, max_dist : float, optional
        Minimum/maximum observable distance.
    bkg_fraction : float, optional
        Background fraction parameter. Default is 0.

    Returns
    -------
    float or array
        CDF values at the input distances.
    """
    x = np.atleast_1d(x)
    cdf_values = np.zeros_like(x, dtype=float)

    for i, xi in enumerate(x):
        if xi <= min_dist:
            cdf_values[i] = 0.0
        elif xi >= max_dist:
            cdf_values[i] = 1.0
        else:
            # Calculate CDF by integrating PDF up to xi
            r_integrate = np.linspace(
                min_dist, xi, num=min(1000, int(xi * 10))
            )
            if len(r_integrate) > 1:
                pdf_vals = nndistribution_from_csr(
                    r_integrate,
                    k,
                    rho,
                    d=d,
                    min_dist=min_dist,
                    max_dist=max_dist,
                    bkg_fraction=bkg_fraction,
                    renormalize=True,
                )
                # Numerical integration using trapezoidal rule
                cdf_values[i] = np.trapezoid(pdf_vals, r_integrate)
            else:
                cdf_values[i] = 0.0

    return cdf_values if len(cdf_values) > 1 else cdf_values[0]

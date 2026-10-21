"""Point-pattern autocorrelation (PPAC) resolution estimator.

Third resolution estimator in this package (alongside FRC and
image-decorrelation). Extracted from ``picasso_outpost.py`` (Phase 2
restructure, monolith split S1).

.. warning::

   **Not promote-ready — this estimator never performed well.** It builds the
   G x G autocorrelation map by rebuilding a ``scipy.spatial.KDTree`` for
   *every* grid cell (see ``_resolution_ppac_sequential_optimized`` /
   ``_resolution_ppac_parallel_optimized``), i.e. O(G**2 . N log N); the
   ``use_sparse`` / chunked / thread-pool variants are band-aids on that, not a
   fix.

   **The fix, before promoting to picasso:** the whole map is a 2-D histogram of
   pairwise displacement vectors -- ``count_neighbors`` at shift (dx, dy) counts
   the point pairs whose displacement is approximately (dx, dy). So build *one*
   ``KDTree``, query all pairs within ``r_max`` once (``query_pairs`` /
   ``query_ball_tree``), and ``np.histogram2d`` their displacement vectors onto
   the grid -- one tree, one pass. Then drop the dead ``batch_size`` /
   ``use_chunking`` params and the redundant backends, add an equivalence test,
   and revisit promotion.
"""

import numpy as np
from scipy.spatial import KDTree


def resolution_ppac(
    locs,
    pixelsize,
    delta_r,
    r_max,
    batch_size=None,
    n_processes=None,
    use_chunking=False,
    use_sparse=False,
):
    """Calculate the resolution by 2D point-pattern autocorrelation.

    Parameters
    ----------
    locs : pd.DataFrame
        Localizations with ``x`` and ``y`` columns.
    pixelsize : float
        Pixel size in physical units (e.g. nm).
    delta_r : float
        Grid spacing for the autocorrelation calculation.
    r_max : float
        Maximum radius for the autocorrelation.
    batch_size : int or None, optional
        Deprecated; no longer used.
    n_processes : int or None, optional
        Number of parallel threads (auto-detected if None).
    use_chunking : bool, optional
        Deprecated and ignored (chunking was removed as mathematically
        incorrect). Default is False.
    use_sparse : bool, optional
        Whether to use sparse matrices for very large grids. Default is False.

    Returns
    -------
    numpy.ndarray
        2D autocorrelation intensity map normalized by the central value.

    Notes
    -----
    Uses a ``ThreadPoolExecutor`` for parallelization and exploits
    autocorrelation symmetry for a 2× speedup.
    """
    from multiprocessing import cpu_count

    # Warn if deprecated parameters are used
    if use_chunking:
        import warnings

        warnings.warn(
            "use_chunking parameter is deprecated and ignored. "
            "The chunking implementation was mathematically incorrect and has been removed.",
            DeprecationWarning,
            stacklevel=2,
        )

    r_max = (r_max // delta_r) * delta_r
    r_search = delta_r / 2
    rs = np.arange(
        -r_max, r_max + delta_r / 2, step=delta_r
    )  # Include endpoint
    idx_ctr = len(rs) // 2

    # Choose data structure based on grid size and use_sparse flag
    grid_size = len(rs)
    if use_sparse and grid_size > 50:
        from scipy import sparse

        intensities = sparse.lil_matrix((grid_size, grid_size))
        using_sparse = True
    else:
        intensities = np.zeros((grid_size, grid_size))
        using_sparse = False

    # Convert to physical coordinates once
    xy = np.column_stack([locs["x"] * pixelsize, locs["y"] * pixelsize])

    # Auto-detect number of threads
    if n_processes is None:
        n_processes = min(4, cpu_count())  # Cap at 4 to avoid memory pressure

    if n_processes > 1 and len(rs) > 4:
        # Use thread-based parallelization with symmetry optimization
        result_intensities = _resolution_ppac_parallel_optimized(
            xy, rs, r_search, n_processes
        )
        if using_sparse:
            intensities = sparse.lil_matrix(result_intensities)
        else:
            intensities = result_intensities
    else:
        # Use optimized sequential algorithm with symmetry
        intensities = _resolution_ppac_sequential_optimized(xy, rs, r_search)

    # Convert back to dense array if using sparse
    if using_sparse:
        intensities = intensities.toarray()

    # Normalize by maximum (no shift) and handle division by zero
    max_intensity = intensities[idx_ctr, idx_ctr]
    if max_intensity > 0:
        intensities = intensities / max_intensity
    # set the center to the mean of its 4-connected neighbors
    intensities[idx_ctr, idx_ctr] = np.mean(
        [
            intensities[idx_ctr - 1, idx_ctr],
            intensities[idx_ctr, idx_ctr - 1],
            intensities[idx_ctr + 1, idx_ctr],
            intensities[idx_ctr, idx_ctr + 1],
        ]
    )

    return intensities


def _resolution_ppac_sequential_optimized(xy, rs, r_search):
    """Optimized sequential PPAC exploiting autocorrelation symmetry.

    Parameters
    ----------
    xy : numpy.ndarray
        ``(N, 2)`` array of coordinates.
    rs : numpy.ndarray
        Array of shift values.
    r_search : float
        Search radius for neighbour counting.

    Returns
    -------
    numpy.ndarray
        2D autocorrelation intensity map.
    """
    import gc

    grid_size = len(rs)
    intensities = np.zeros((grid_size, grid_size))

    # Build base tree once
    tree_base = KDTree(xy)

    # Compute only half the grid using symmetry
    for i, delta_x in enumerate(rs):
        for j, delta_y in enumerate(rs):
            # Use symmetry: I(δx, δy) = I(-δx, -δy)
            # Only compute upper half + diagonal
            flat_idx = i * grid_size + j
            if flat_idx > grid_size * grid_size // 2:
                # Use symmetry from already computed point
                sym_i = grid_size - 1 - i
                sym_j = grid_size - 1 - j
                intensities[i, j] = intensities[sym_i, sym_j]
            else:
                # Compute this point
                xy_shift = xy + np.array([[delta_x, delta_y]])
                tree_probe = KDTree(xy_shift)
                intensities[i, j] = tree_base.count_neighbors(
                    tree_probe, r_search
                )
                del tree_probe

    del tree_base
    gc.collect()

    return intensities


def _resolution_ppac_parallel_optimized(xy, rs, r_search, n_threads):
    """Optimized parallel PPAC using a ThreadPoolExecutor with symmetry.

    Parameters
    ----------
    xy : numpy.ndarray
        ``(N, 2)`` array of coordinates.
    rs : numpy.ndarray
        Array of shift values.
    r_search : float
        Search radius for neighbour counting.
    n_threads : int
        Number of threads to use.

    Returns
    -------
    numpy.ndarray
        2D autocorrelation intensity map.
    """
    from concurrent.futures import ThreadPoolExecutor

    grid_size = len(rs)
    intensities = np.zeros((grid_size, grid_size))

    # Build base tree once (shared across threads)
    tree_base = KDTree(xy)

    # Generate only half the grid points (exploit symmetry)
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            flat_idx = i * grid_size + j
            if flat_idx <= grid_size * grid_size // 2:
                grid_points.append((i, j))

    # Worker function
    def compute_point(ij_tuple):
        """Compute autocorrelation at one grid point"""
        i, j = ij_tuple
        delta_x, delta_y = rs[i], rs[j]
        xy_shift = xy + np.array([[delta_x, delta_y]])
        tree_probe = KDTree(xy_shift)
        intensity = tree_base.count_neighbors(tree_probe, r_search)
        return (i, j, intensity)

    # Execute in parallel using threads (shares memory, avoids serialization)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = executor.map(compute_point, grid_points)

        # Fill in computed values and symmetric counterparts
        for i, j, intensity in results:
            intensities[i, j] = intensity

            # Fill symmetric point
            sym_i = grid_size - 1 - i
            sym_j = grid_size - 1 - j
            if sym_i != i or sym_j != j:  # Avoid overwriting center point
                intensities[sym_i, sym_j] = intensity

    return intensities


def _resolution_ppac_chunked(xy, rs, r_search, batch_size, n_processes):
    """DEPRECATED: Mathematically incorrect chunked autocorrelation

    This function is kept for backward compatibility but should not be used.
    The chunking approach used here produces incorrect results because it
    computes autocorrelation within data chunks rather than across the full
    dataset, then incorrectly averages them.

    Use _resolution_ppac_parallel_optimized() instead.
    """
    import gc

    n_points = len(xy)
    intensities = np.zeros((len(rs), len(rs)))

    # Process data in chunks to manage memory usage
    n_chunks = (n_points + batch_size - 1) // batch_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * batch_size
        end_idx = min((chunk_idx + 1) * batch_size, n_points)
        xy_chunk = xy[start_idx:end_idx]

        # Build KDTree for this chunk
        tree_chunk = KDTree(xy_chunk)

        # Process grid points in parallel for this chunk
        if n_processes > 1:
            chunk_intensities = _process_grid_parallel(
                xy_chunk, tree_chunk, rs, r_search, n_processes
            )
        else:
            chunk_intensities = _process_grid_sequential(
                xy_chunk, tree_chunk, rs, r_search
            )

        # Accumulate results (weighted by chunk size)
        weight = len(xy_chunk) / n_points
        intensities += chunk_intensities * weight

        # Cleanup
        del tree_chunk, xy_chunk, chunk_intensities
        gc.collect()

    return intensities


def _resolution_ppac_parallel(xy, rs, r_search, n_processes):
    """DEPRECATED: Inefficient multiprocessing-based parallel autocorrelation

    This function is kept for backward compatibility but should not be used.
    It uses multiprocessing.Pool which causes massive data serialization overhead
    and rebuilds KDTrees redundantly in each worker process.

    Use _resolution_ppac_parallel_optimized() instead, which uses ThreadPoolExecutor
    for true memory sharing and 3-7× better performance.
    """
    from multiprocessing import Pool

    tree_i = KDTree(xy)

    # Create grid point tasks
    tasks = [
        (i, j, xy, rs[i], rs[j], r_search)
        for i in range(len(rs))
        for j in range(len(rs))
    ]

    # Process in parallel
    with Pool(n_processes) as pool:
        results = pool.map(_compute_autocorr_point, tasks)

    # Reconstruct intensity matrix
    intensities = np.zeros((len(rs), len(rs)))
    for i, j, intensity in results:
        intensities[i, j] = intensity

    del tree_i
    return intensities


def _process_grid_parallel(xy_chunk, tree_chunk, rs, r_search, n_processes):
    """Process grid points in parallel for a data chunk"""
    from multiprocessing import Pool

    tasks = [
        (i, j, xy_chunk, rs[i], rs[j], r_search)
        for i in range(len(rs))
        for j in range(len(rs))
    ]

    with Pool(n_processes) as pool:
        results = pool.map(_compute_autocorr_point_chunk, tasks)

    intensities = np.zeros((len(rs), len(rs)))
    for i, j, intensity in results:
        intensities[i, j] = intensity

    return intensities


def _process_grid_sequential(xy_chunk, tree_chunk, rs, r_search):
    """Process grid points sequentially for a data chunk"""
    intensities = np.zeros((len(rs), len(rs)))

    for i, delta_x in enumerate(rs):
        for j, delta_y in enumerate(rs):
            xy_shift = xy_chunk + np.array([[delta_x, delta_y]])
            tree_probe = KDTree(xy_shift)
            intensities[i, j] = tree_chunk.count_neighbors(
                tree_probe, r_search
            )
            del tree_probe

    return intensities


def _compute_autocorr_point(task):
    """Compute autocorrelation for a single grid point (full dataset)"""
    i, j, xy, delta_x, delta_y, r_search = task
    tree_i = KDTree(xy)
    xy_shift = xy + np.array([[delta_x, delta_y]])
    tree_probe = KDTree(xy_shift)
    intensity = tree_i.count_neighbors(tree_probe, r_search)
    return (i, j, intensity)


def _compute_autocorr_point_chunk(task):
    """Compute autocorrelation for a single grid point (chunk-based)"""
    i, j, xy_chunk, delta_x, delta_y, r_search = task
    tree_chunk = KDTree(xy_chunk)
    xy_shift = xy_chunk + np.array([[delta_x, delta_y]])
    tree_probe = KDTree(xy_shift)
    intensity = tree_chunk.count_neighbors(tree_probe, r_search)
    return (i, j, intensity)


def analyse_resolution_ppac(intensities, delta_r):
    """Fit a 2D Gaussian to an autocorrelation map to extract resolution.

    Parameters
    ----------
    intensities : numpy.ndarray
        2D autocorrelation intensity map from :func:`resolution_ppac`.
    delta_r : float
        Grid spacing used for the autocorrelation calculation.

    Returns
    -------
    dict
        Resolution analysis results with keys ``sigma_x``/``sigma_y``
        (Gaussian std devs in physical units), ``resolution`` (``2.35 *
        mean(sigma_x, sigma_y)``), ``fwhm_x``/``fwhm_y``, ``amplitude``,
        ``background`` and ``fit_quality`` (R-squared).
    """
    from scipy.optimize import curve_fit

    def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, background):
        """2D Gaussian function for fitting"""
        x, y = coords
        return (
            amplitude
            * np.exp(
                -(
                    (x - x0) ** 2 / (2 * sigma_x**2)
                    + (y - y0) ** 2 / (2 * sigma_y**2)
                )
            )
            + background
        )

    # Create coordinate grids
    size = intensities.shape[0]
    center = size // 2
    x_grid = np.arange(size) * delta_r - center * delta_r
    y_grid = np.arange(size) * delta_r - center * delta_r
    X, Y = np.meshgrid(x_grid, y_grid)
    coords = (X.flatten(), Y.flatten())
    z_data = intensities.flatten()

    # Initial parameter guess
    amplitude_guess = np.max(intensities) - np.min(intensities)
    background_guess = np.min(intensities)
    sigma_guess = delta_r * 3  # Initial guess for sigma

    initial_guess = [
        amplitude_guess,
        0,
        0,
        sigma_guess,
        sigma_guess,
        background_guess,
    ]

    # Set parameter bounds
    bounds = (
        [
            0,
            -center * delta_r,
            -center * delta_r,
            delta_r / 2,
            delta_r / 2,
            0,
        ],  # Lower bounds
        [
            np.inf,
            center * delta_r,
            center * delta_r,
            center * delta_r,
            center * delta_r,
            np.inf,
        ],  # Upper bounds
    )

    try:
        # Perform the fit
        # Suppress numerical warnings from scipy optimizer
        # (divide by zero warnings are expected for ill-conditioned histograms)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            popt, pcov = curve_fit(
                gaussian_2d,
                coords,
                z_data,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )

        amplitude, x0, y0, sigma_x, sigma_y, background = popt

        # Calculate fit quality (R-squared)
        fitted_data = gaussian_2d(coords, *popt)
        ss_res = np.sum((z_data - fitted_data) ** 2)
        ss_tot = np.sum((z_data - np.mean(z_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate resolution metrics
        fwhm_x = 2.35 * sigma_x  # FWHM = 2.35 * sigma for Gaussian
        fwhm_y = 2.35 * sigma_y
        resolution = np.mean([fwhm_x, fwhm_y])  # Average resolution

        return {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "resolution": resolution,
            "fwhm_x": fwhm_x,
            "fwhm_y": fwhm_y,
            "amplitude": amplitude,
            "background": background,
            "center_x": x0,
            "center_y": y0,
            "fit_quality": r_squared,
            "fit_success": True,
            "fit_params": popt,
            "fit_covariance": pcov,
        }

    except Exception as e:
        # Return fallback values if fit fails
        return {
            "sigma_x": np.nan,
            "sigma_y": np.nan,
            "resolution": np.nan,
            "fwhm_x": np.nan,
            "fwhm_y": np.nan,
            "amplitude": np.nan,
            "background": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "fit_quality": np.nan,
            "fit_success": False,
            "error": str(e),
            "fit_params": None,
            "fit_covariance": None,
        }

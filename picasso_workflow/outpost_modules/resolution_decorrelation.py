"""Image decorrelation analysis for resolution measurement.

This module implements the image decorrelation analysis method described in:
Descloux et al., "Parameter-free image resolution estimation based on
decorrelation analysis", Nature Methods 16, 918-924 (2019).

The method measures resolution from a single image by analyzing how image
structure decorrelates across spatial frequencies.
"""

import numpy as np
# import logging
from loguru import logger
from concurrent.futures import ProcessPoolExecutor

# logger = logging.getLogger(__name__)


def _get_memory_usage_mb():
    """Get current process memory usage and available system memory in MB

    Returns:
        tuple: (process_rss_mb, available_mb)
    """
    import psutil
    import os
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        process_mb = mem_info.rss / 1024 / 1024

        virtual_mem = psutil.virtual_memory()
        available_mb = virtual_mem.available / 1024 / 1024

        return process_mb, available_mb
    except:
        return -1, -1


def apply_cosine_apodization(image, edge_width=20):
    """Apply cosine apodization to image edges to suppress FFT artifacts

    Args:
        image : ndarray
            2D image array
        edge_width : int
            Width of edge region to apodize in pixels

    Returns:
        image_apod : ndarray
            Apodized image
    """
    height, width = image.shape

    # Create 1D cosine windows
    y_window = np.ones(height)
    x_window = np.ones(width)

    # Apply cosine taper to edges
    for i in range(min(edge_width, height // 2)):
        weight = 0.5 * (1 + np.cos(np.pi * (edge_width - i) / edge_width))
        y_window[i] = weight
        y_window[-(i + 1)] = weight

    for i in range(min(edge_width, width // 2)):
        weight = 0.5 * (1 + np.cos(np.pi * (edge_width - i) / edge_width))
        x_window[i] = weight
        x_window[-(i + 1)] = weight

    # Create 2D window
    window_2d = np.outer(y_window, x_window)

    # Apply window
    image_apod = image * window_2d

    return image_apod


def compute_decorr_single(image, pixelsize_render, r_min=0.0, r_max=1.0,
                          n_r=50, n_gauss=10, apod_edge_width=20):
    """Compute decorrelation curve for a single image

    Implements the image decorrelation analysis algorithm from Descloux et al.

    Args:
        image : ndarray
            2D rendered image
        pixelsize_render : float
            Pixel size in nm
        r_min : float
            Minimum normalized frequency (default: 0.0)
        r_max : float
            Maximum normalized frequency (default: 1.0)
        n_r : int
            Number of radial sampling points (default: 50)
        n_gauss : int
            Number of Gaussian filter strengths (default: 10)
        apod_edge_width : int
            Edge apodization width in pixels (default: 20)

    Returns:
        decorr_curve : ndarray
            Decorrelation values at each radius
        r_values : ndarray
            Normalized radius values (0 to 1)
        kc_max : float
            Cutoff frequency in 1/nm
        resolution : float
            Resolution in nm (2 * pixelsize / kc_max_normalized)
    """
    # Apply edge apodization
    image_apod = apply_cosine_apodization(image, edge_width=apod_edge_width)

    # Compute FFT
    fft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image_apod)))

    # Normalize FFT
    fft_norm = np.sqrt(np.sum(np.abs(fft_image)**2))
    fft_normalized = fft_image / fft_norm if fft_norm > 0 else fft_image

    # Create radial frequency mask
    height, width = image.shape
    ky = np.fft.fftfreq(height)
    kx = np.fft.fftfreq(width)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kx_grid = np.fft.fftshift(kx_grid)
    ky_grid = np.fft.fftshift(ky_grid)
    R = np.sqrt(kx_grid**2 + ky_grid**2)

    # Normalize R to [0, 1]
    R_max = 0.5  # Nyquist frequency
    R_normalized = R / R_max

    # Radial sampling points
    r_values = np.linspace(r_min, r_max, n_r)

    # Gaussian filter strengths (logarithmic spacing)
    g_values = np.logspace(-1, 1, n_gauss)

    # Initialize decorrelation matrix
    decorr_matrix = np.zeros((n_r, n_gauss))

    logger.debug(f"    Computing decorrelation for {n_r} radii × {n_gauss} filters...")

    # Compute decorrelation for each Gaussian filter strength
    for g_idx, g in enumerate(g_values):
        # Apply Gaussian high-pass filter
        gauss_filter = 1.0 - np.exp(-2 * g**2 * R_normalized**2)
        fft_filtered = fft_normalized * gauss_filter

        # Normalization for filtered image
        c_filtered = np.sqrt(np.sum(np.abs(fft_filtered)**2))

        if c_filtered < 1e-10:
            continue

        # Compute correlation for each radius
        for r_idx, r in enumerate(r_values):
            if r <= 0:
                decorr_matrix[r_idx, g_idx] = 1.0
                continue

            # Bandpass mask (frequencies below r)
            bandpass_mask = (R_normalized**2 < r**2).astype(float)

            # Apply mask
            fft_bandpass = fft_normalized * bandpass_mask

            # Normalization for bandpass
            c_bandpass = np.sqrt(np.sum(np.abs(fft_bandpass)**2))

            if c_bandpass < 1e-10:
                decorr_matrix[r_idx, g_idx] = 0.0
                continue

            # Compute correlation coefficient
            cross_product = fft_filtered * np.conj(fft_bandpass)
            corr = np.real(np.sum(cross_product)) / (c_filtered * c_bandpass)

            decorr_matrix[r_idx, g_idx] = corr

    # Average across filter strengths
    decorr_curve = np.mean(decorr_matrix, axis=1)

    # Extract resolution
    kc_max, resolution = extract_resolution_decorr(
        decorr_curve, r_values, pixelsize_render
    )

    return decorr_curve, r_values, kc_max, resolution


def extract_resolution_decorr(decorr_curve, r_values, pixelsize_render,
                               threshold=0.5):
    """Extract resolution from decorrelation curve

    Args:
        decorr_curve : ndarray
            Decorrelation values
        r_values : ndarray
            Normalized radius values (0 to 1)
        pixelsize_render : float
            Pixel size in nm
        threshold : float
            Decorrelation threshold (default: 0.5)

    Returns:
        kc_max : float
            Cutoff frequency in 1/nm
        resolution : float
            Resolution in nm
    """
    # Find where decorrelation drops below threshold
    below_threshold = decorr_curve < threshold

    if not np.any(below_threshold):
        logger.warning("    Decorrelation never drops below threshold")
        return np.nan, np.nan

    # Find first crossing
    cutoff_idx = np.argmax(below_threshold)

    if cutoff_idx == 0:
        logger.warning("    Decorrelation starts below threshold")
        return np.nan, np.nan

    # Linear interpolation for better accuracy
    r1, r2 = r_values[cutoff_idx - 1], r_values[cutoff_idx]
    d1, d2 = decorr_curve[cutoff_idx - 1], decorr_curve[cutoff_idx]

    if d2 != d1:
        r_cutoff = r1 + (threshold - d1) * (r2 - r1) / (d2 - d1)
    else:
        r_cutoff = r1

    # Convert normalized frequency to 1/nm
    # r_cutoff is in normalized units [0, 1] where 1 = Nyquist = 0.5/pixel
    # So r_cutoff corresponds to (r_cutoff * 0.5) cycles/pixel
    # Converting to 1/nm: (r_cutoff * 0.5) / pixelsize_render
    kc_max = (r_cutoff * 0.5) / pixelsize_render

    # Resolution = 2 / kc_max (from Descloux paper)
    resolution = 2.0 / kc_max if kc_max > 0 else np.nan

    logger.debug(f"    Resolution: {resolution:.2f} nm (kc_max: {kc_max:.4f} 1/nm)")

    return kc_max, resolution


def _process_tile_decorr(tile_task):
    """Worker function to process a single tile

    Args:
        tile_task : dict
            Dictionary with keys:
                - id: tile identifier (i, j)
                - bounds: (x_min, x_max, y_min, y_max) in nm
                - locs: localization subset for this tile
                - pixelsize: camera pixel size in nm
                - pixelsize_render: rendering pixel size in nm
                - tile_dims: (width_pixels, height_pixels)
                - smoothing_sigma: optional Gaussian smoothing
                - r_min, r_max: frequency range
                - n_r: number of radial points
                - n_gauss: number of Gaussian filters
                - apod_edge_width: apodization width

    Returns:
        result : dict
            Success: {'tile_id', 'n_locs', 'decorr_curve', 'r_values',
                      'kc_max', 'resolution', 'success': True}
            Failure: {'tile_id', 'n_locs', 'error', 'success': False}
    """
    from picasso_workflow.outpost_modules.resolution_frc import (
        render_image_histogram
    )

    try:
        tile_id = tile_task['id']
        tile_locs = tile_task['locs']
        pixelsize = tile_task['pixelsize']
        pixelsize_render = tile_task['pixelsize_render']
        tile_dims = tile_task['tile_dims']
        smoothing_sigma = tile_task.get('smoothing_sigma', None)
        r_min = tile_task['r_min']
        r_max = tile_task['r_max']
        n_r = tile_task['n_r']
        n_gauss = tile_task['n_gauss']
        apod_edge_width = tile_task['apod_edge_width']
        bounds = tile_task['bounds']

        mem_start, avail_start = _get_memory_usage_mb()
        logger.debug(f"    Tile {tile_id}: Starting, memory = {mem_start:.1f} MB, "
                    f"available = {avail_start:.1f} MB, n_locs = {len(tile_locs)}")

        # Render image
        image, _ = render_image_histogram(
            tile_locs, pixelsize, pixelsize_render,
            bounds=bounds, smoothing_sigma=smoothing_sigma
        )

        mem_after_render, avail_after_render = _get_memory_usage_mb()
        logger.debug(f"    Tile {tile_id}: Rendered {image.shape}, "
                    f"memory = {mem_after_render:.1f} MB")

        # Compute decorrelation
        decorr_curve, r_values, kc_max, resolution = compute_decorr_single(
            image, pixelsize_render, r_min=r_min, r_max=r_max,
            n_r=n_r, n_gauss=n_gauss, apod_edge_width=apod_edge_width
        )

        mem_after_decorr, avail_after_decorr = _get_memory_usage_mb()
        logger.debug(f"    Tile {tile_id}: Resolution = {resolution:.2f} nm, "
                    f"memory = {mem_after_decorr:.1f} MB")

        return {
            'tile_id': tile_id,
            'n_locs': len(tile_locs),
            'decorr_curve': decorr_curve,
            'r_values': r_values,
            'kc_max': kc_max,
            'resolution': resolution,
            'success': True
        }

    except Exception as e:
        logger.error(f"    Tile {tile_task['id']}: Failed - {str(e)}")
        return {
            'tile_id': tile_task['id'],
            'n_locs': len(tile_task['locs']),
            'error': str(e),
            'success': False
        }


def compute_decorr_spatial(locs, pixelsize, pixelsize_render=5.0,
                           smoothing_sigma=None, region_size=10.0,
                           min_locs_per_region=500, n_processes=4,
                           r_min=0.0, r_max=1.0, n_r=50, n_gauss=10,
                           apod_edge_width=20):
    """Compute image decorrelation resolution using spatial tiling

    Divides the field-of-view into spatial regions, computes decorrelation
    for each region independently, and averages the results.

    Args:
        locs : structured array
            Localization data with 'x' and 'y' fields
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm (default: 5 nm)
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels
        region_size : float
            Size of each spatial region in micrometers (default: 10.0 µm)
        min_locs_per_region : int
            Minimum localizations per region (default: 500)
        n_processes : int
            Number of parallel processes (default: 4)
        r_min : float
            Minimum normalized frequency (default: 0.0)
        r_max : float
            Maximum normalized frequency (default: 1.0)
        n_r : int
            Number of radial sampling points (default: 50)
        n_gauss : int
            Number of Gaussian filter strengths (default: 10)
        apod_edge_width : int
            Edge apodization width in pixels (default: 20)

    Returns:
        results : dict
            Dictionary containing:
                - resolution : float (mean resolution in nm)
                - resolution_std : float (standard deviation)
                - resolutions_per_region : list
                - decorr_curve_mean : ndarray
                - decorr_curve_std : ndarray
                - r_values : ndarray
                - n_regions : int (number of valid regions)
                - n_regions_total : int
                - n_regions_x : int
                - n_regions_y : int
    """
    from concurrent.futures import ProcessPoolExecutor

    # Convert to physical coordinates
    x_nm = locs['x'] * pixelsize
    y_nm = locs['y'] * pixelsize

    # Determine overall bounds
    x_min, x_max = x_nm.min(), x_nm.max()
    y_min, y_max = y_nm.min(), y_nm.max()

    # Calculate region size
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Convert region_size from micrometers to nanometers
    region_size_nm = region_size * 1000.0

    # Calculate number of regions
    n_regions_x = max(1, int(np.ceil(x_range / region_size_nm)))
    n_regions_y = max(1, int(np.ceil(y_range / region_size_nm)))

    # Recalculate actual region dimensions
    region_width = x_range / n_regions_x
    region_height = y_range / n_regions_y

    logger.debug(f"Computing image decorrelation with {n_regions_x}×{n_regions_y} regions "
                f"(target size: {region_size:.1f} µm)...")
    logger.debug(f"  FOV: {x_range:.1f} × {y_range:.1f} nm")
    logger.debug(f"  Actual region size: {region_width:.1f} × {region_height:.1f} nm "
                f"({region_width/1000:.2f} × {region_height/1000:.2f} µm)")

    # Pre-calculate consistent tile dimensions
    tile_width_pixels = int(np.ceil(region_width / pixelsize_render))
    tile_height_pixels = int(np.ceil(region_height / pixelsize_render))

    logger.debug(f"  Tile dimensions: {tile_width_pixels}×{tile_height_pixels} pixels")

    # Generate spatial tiles with pre-filtered localizations
    tile_tasks = []
    n_total = n_regions_x * n_regions_y

    for i in range(n_regions_x):
        for j in range(n_regions_y):
            # Calculate tile bounds
            tile_x_min = x_min + i * region_width
            tile_x_max = tile_x_min + tile_width_pixels * pixelsize_render
            tile_y_min = y_min + j * region_height
            tile_y_max = tile_y_min + tile_height_pixels * pixelsize_render

            # Pre-filter to this tile
            mask = ((x_nm >= tile_x_min) & (x_nm < tile_x_max) &
                    (y_nm >= tile_y_min) & (y_nm < tile_y_max))
            tile_locs = locs[mask]

            if len(tile_locs) >= min_locs_per_region:
                tile_tasks.append({
                    'id': (i, j),
                    'bounds': (tile_x_min, tile_x_max, tile_y_min, tile_y_max),
                    'locs': tile_locs,
                    'pixelsize': pixelsize,
                    'pixelsize_render': pixelsize_render,
                    'tile_dims': (tile_width_pixels, tile_height_pixels),
                    'smoothing_sigma': smoothing_sigma,
                    'r_min': r_min,
                    'r_max': r_max,
                    'n_r': n_r,
                    'n_gauss': n_gauss,
                    'apod_edge_width': apod_edge_width
                })

    logger.debug(f"  Processing {len(tile_tasks)} valid regions (skipped "
                f"{n_total - len(tile_tasks)} sparse regions)")

    # Process tiles in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tile_results = list(executor.map(_process_tile_decorr, tile_tasks))

    # Separate successful and failed results
    successful_results = [r for r in tile_results if r['success']]
    failed_results = [r for r in tile_results if not r['success']]

    n_success = len(successful_results)
    n_failed = len(failed_results)

    logger.debug(f"  Completed: {n_success} successful, {n_failed} failed")

    if n_success == 0:
        raise RuntimeError("All tiles failed to process")

    # Aggregate results
    resolutions = np.array([r['resolution'] for r in successful_results])
    decorr_curves = np.array([r['decorr_curve'] for r in successful_results])
    r_values = successful_results[0]['r_values']

    # Calculate statistics
    valid_resolutions = resolutions[~np.isnan(resolutions)]

    if len(valid_resolutions) > 0:
        resolution_mean = np.mean(valid_resolutions)
        resolution_std = np.std(valid_resolutions) if len(valid_resolutions) > 1 else 0.0
    else:
        resolution_mean = np.nan
        resolution_std = np.nan

    decorr_curve_mean = np.nanmean(decorr_curves, axis=0)
    decorr_curve_std = np.nanstd(decorr_curves, axis=0)

    logger.debug(f"  Mean resolution: {resolution_mean:.2f} ± {resolution_std:.2f} nm")
    logger.debug(f"  Resolution range: {np.nanmin(resolutions):.2f} - "
                f"{np.nanmax(resolutions):.2f} nm")

    # Package results
    results = {
        'resolution': resolution_mean,
        'resolution_std': resolution_std,
        'resolutions_per_region': resolutions.tolist(),
        'decorr_curve_mean': decorr_curve_mean,
        'decorr_curve_std': decorr_curve_std,
        'r_values': r_values,
        'n_regions': n_success,
        'n_regions_total': n_total,
        'n_regions_x': n_regions_x,
        'n_regions_y': n_regions_y,
        'n_failed': n_failed,
        'tile_info': [(r['tile_id'], r['n_locs'], r['resolution'])
                      for r in successful_results],
        'failed_tiles': [(r['tile_id'], r.get('error', 'unknown'))
                        for r in failed_results] if n_failed > 0 else []
    }

    return results

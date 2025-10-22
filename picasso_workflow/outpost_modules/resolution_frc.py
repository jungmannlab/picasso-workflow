"""
Fourier Ring Correlation (FRC) for Resolution Estimation

This module implements Fourier Ring Correlation analysis for estimating
spatial resolution in super-resolution microscopy data.

Reference:
    Nieuwenhuizen et al. (2013). "Measuring image resolution in optical
    nanoscopy". Nature Methods 10, 557-562.
    DOI: 10.1038/nmeth.2448

Algorithm:
    1. Split localizations into two random subsets
    2. Render each subset into a super-resolution image
    3. Compute 2D Fourier transforms
    4. Calculate correlation in concentric rings in Fourier space
    5. Find spatial frequency where FRC drops below threshold (1/7)
    6. Resolution = 1 / cutoff_frequency

Author: Generated for picasso-workflow
"""

# import logging
from loguru import logger

import numpy as np
from scipy.ndimage import gaussian_filter

# logger = logging.getLogger(__name__)


def split_localizations_random(locs, seed=None):
    """Split localizations into two random independent subsets

    Args:
        locs : structured array or DataFrame
            Localization data with 'x' and 'y' fields
        seed : int or None
            Random seed for reproducibility

    Returns:
        locs_1 : structured array
            First subset (approximately half)
        locs_2 : structured array
            Second subset (approximately half)
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(locs)
    indices = np.random.permutation(n)
    split = n // 2

    locs_1 = locs[indices[:split]]
    locs_2 = locs[indices[split:]]

    logger.debug(
        f"  Split {n} localizations into {len(locs_1)} and {len(locs_2)}"
    )

    return locs_1, locs_2


def render_image_histogram(
    locs, pixelsize, pixelsize_render, bounds=None, smoothing_sigma=None
):
    """Render localizations into a super-resolution image using histogram

    Args:
        locs : structured array
            Localization data with 'x' and 'y' fields (in camera pixels)
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm
        bounds : tuple of 4 floats or None
            (x_min, x_max, y_min, y_max) in nm. If None, auto-calculate
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels. If None, no smoothing

    Returns:
        image : ndarray
            Rendered 2D image
        bounds : tuple
            Actual bounds used (x_min, x_max, y_min, y_max) in nm
    """
    # Convert to physical coordinates (nm)
    x_nm = locs["x"] * pixelsize
    y_nm = locs["y"] * pixelsize

    # Calculate bounds if not provided
    if bounds is None:
        x_min, x_max = x_nm.min(), x_nm.max()
        y_min, y_max = y_nm.min(), y_nm.max()

        # Add small margin
        margin = 10 * pixelsize_render
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
    else:
        x_min, x_max, y_min, y_max = bounds

    # Create bins
    x_bins = np.arange(x_min, x_max + pixelsize_render, pixelsize_render)
    y_bins = np.arange(y_min, y_max + pixelsize_render, pixelsize_render)

    # Render histogram
    image, _, _ = np.histogram2d(x_nm, y_nm, bins=[x_bins, y_bins])

    # Optional Gaussian smoothing
    if smoothing_sigma is not None:
        image = gaussian_filter(image, sigma=smoothing_sigma)

    # logger.debug(f"  Rendered image: {image.shape[0]}×{image.shape[1]} pixels")
    # logger.debug(f"  Pixel size: {pixelsize_render:.2f} nm")

    bounds = (x_min, x_max, y_min, y_max)

    return image, bounds


def compute_fft(image):
    """Compute 2D Fourier transform of image

    Args:
        image : ndarray
            2D input image

    Returns:
        fft_shifted : ndarray (complex)
            Shifted 2D FFT with DC component at center
    """
    # Compute FFT
    fft = np.fft.fft2(image)

    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft)

    logger.debug(f"  Computed FFT: {fft_shifted.shape}")

    return fft_shifted


def compute_frc_curve(fft1, fft2, pixelsize_render):
    """Compute Fourier Ring Correlation curve

    Args:
        fft1 : ndarray (complex)
            Shifted FFT of first image
        fft2 : ndarray (complex)
            Shifted FFT of second image
        pixelsize_render : float
            Pixel size of rendered images in nm

    Returns:
        frc_values : ndarray
            FRC values for each radial bin
        spatial_frequencies : ndarray
            Spatial frequencies in 1/nm
    """
    # Get image dimensions and center
    shape = fft1.shape
    center = np.array(shape) // 2

    # Compute distance matrix
    y, x = np.ogrid[: shape[0], : shape[1]]
    distances = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Define radial bins
    max_radius = min(center)
    radial_bins = np.arange(0, max_radius + 1)
    n_bins = len(radial_bins) - 1

    # Compute FRC for each ring
    frc_values = np.zeros(n_bins)

    for i in range(n_bins):
        r1, r2 = radial_bins[i], radial_bins[i + 1]
        mask = (distances >= r1) & (distances < r2)

        if np.sum(mask) == 0:
            frc_values[i] = np.nan
            continue

        # FRC formula: correlation normalized by intensities
        numerator = np.sum(fft1[mask] * np.conj(fft2[mask]))
        denom1 = np.sum(np.abs(fft1[mask]) ** 2)
        denom2 = np.sum(np.abs(fft2[mask]) ** 2)

        if denom1 > 0 and denom2 > 0:
            frc_values[i] = np.real(numerator) / np.sqrt(denom1 * denom2)
        else:
            frc_values[i] = np.nan

    # Calculate spatial frequencies
    # Frequency spacing = 1 / (N * pixel_size)
    freq_spacing = 1.0 / (shape[0] * pixelsize_render)
    spatial_frequencies = (
        (radial_bins[:-1] + radial_bins[1:]) / 2 * freq_spacing
    )

    logger.debug(f"  Computed FRC curve: {n_bins} frequency bins")

    return frc_values, spatial_frequencies


def extract_resolution(frc_values, spatial_frequencies, threshold=1 / 7):
    """Extract resolution from FRC curve

    Args:
        frc_values : ndarray
            FRC values
        spatial_frequencies : ndarray
            Spatial frequencies in 1/nm
        threshold : float
            FRC threshold for resolution cutoff (default: 1/7)

    Returns:
        resolution : float
            Resolution in nm
        cutoff_frequency : float
            Cutoff spatial frequency in 1/nm
    """
    # Remove NaN values
    valid = ~np.isnan(frc_values)
    frc_valid = frc_values[valid]
    freq_valid = spatial_frequencies[valid]

    if len(frc_valid) == 0:
        logger.warning("  No valid FRC values found")
        return np.nan, np.nan

    # Find first point below threshold
    below_threshold = frc_valid < threshold

    if not np.any(below_threshold):
        logger.warning("  FRC never drops below threshold")
        return np.nan, np.nan

    cutoff_idx = np.argmax(below_threshold)

    # Linear interpolation for better accuracy
    if cutoff_idx > 0:
        f1, f2 = freq_valid[cutoff_idx - 1], freq_valid[cutoff_idx]
        frc1, frc2 = frc_valid[cutoff_idx - 1], frc_valid[cutoff_idx]

        # Interpolate to find exact crossing point
        if frc2 != frc1:
            cutoff_frequency = f1 + (threshold - frc1) * (f2 - f1) / (
                frc2 - frc1
            )
        else:
            cutoff_frequency = f1
    else:
        cutoff_frequency = freq_valid[cutoff_idx]

    # Resolution = 1 / spatial_frequency
    resolution = 1.0 / cutoff_frequency

    logger.debug(
        f"  Resolution: {resolution:.2f} nm (cutoff: {cutoff_frequency:.4f} 1/nm)"
    )

    return resolution, cutoff_frequency


def compute_frc_resolution(
    locs,
    pixelsize,
    pixelsize_render=5.0,
    smoothing_sigma=None,
    threshold=1 / 7,
    seed=None,
    max_frc_range_nm=None,
):
    """Complete FRC resolution analysis pipeline

    Args:
        locs : structured array
            Localization data
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm (default: 5 nm)
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels
        threshold : float
            FRC threshold (default: 1/7)
        seed : int or None
            Random seed
        max_frc_range_nm : float or None
            Maximum resolution to compute (in nm). If specified, only compute
            FRC up to this resolution. Useful for speeding up computation.
            Default: None (compute full curve)

    Returns:
        results : dict
            Dictionary containing:
                - resolution : float (nm)
                - cutoff_frequency : float (1/nm)
                - frc_curve : ndarray
                - spatial_frequencies : ndarray
                - threshold : float
                - image_1 : ndarray
                - image_2 : ndarray
                - bounds : tuple
    """
    logger.debug("Computing FRC resolution...")

    # Step 1: Split localizations
    locs_1, locs_2 = split_localizations_random(locs, seed=seed)

    # Step 2: Render images
    logger.debug("  Rendering image 1...")
    image_1, bounds = render_image_histogram(
        locs_1, pixelsize, pixelsize_render, smoothing_sigma=smoothing_sigma
    )

    logger.debug("  Rendering image 2...")
    image_2, _ = render_image_histogram(
        locs_2,
        pixelsize,
        pixelsize_render,
        bounds=bounds,
        smoothing_sigma=smoothing_sigma,
    )

    # Step 3: Compute FFTs
    logger.debug("  Computing FFTs...")
    fft_1 = compute_fft(image_1)
    fft_2 = compute_fft(image_2)

    # Step 4: Compute FRC curve (vectorized)
    logger.debug("  Computing FRC curve...")
    frc_curve, spatial_frequencies = compute_frc_curve_vectorized(
        fft_1, fft_2, pixelsize_render, max_frc_range_nm=max_frc_range_nm
    )

    # Step 5: Extract resolution
    logger.debug("  Extracting resolution...")
    resolution, cutoff_frequency = extract_resolution(
        frc_curve, spatial_frequencies, threshold
    )

    # Package results
    results = {
        "resolution": resolution,
        "cutoff_frequency": cutoff_frequency,
        "frc_curve": frc_curve,
        "spatial_frequencies": spatial_frequencies,
        "threshold": threshold,
        "image_1": image_1,
        "image_2": image_2,
        "bounds": bounds,
        "pixelsize_render": pixelsize_render,
    }

    return results


########################################################################
# Phase 2: Optimizations (Memory & Performance)
########################################################################


def render_image_chunked_parallel(
    locs,
    pixelsize,
    pixelsize_render,
    bounds=None,
    smoothing_sigma=None,
    chunk_size_nm=10000,
    overlap_nm=500,
    n_processes=4,
):
    """Render large image using overlapping spatial chunks (memory-efficient)

    This method divides the field-of-view into overlapping spatial tiles,
    renders each tile independently in parallel, and stitches them together
    with smooth transitions in overlap regions.

    Args:
        locs : structured array
            Localization data with 'x' and 'y' fields (in camera pixels)
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm
        bounds : tuple of 4 floats or None
            (x_min, x_max, y_min, y_max) in nm. If None, auto-calculate
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels
        chunk_size_nm : float
            Size of each spatial chunk in nm (default: 10000 nm = 10 μm)
        overlap_nm : float
            Overlap between adjacent chunks in nm (default: 500 nm)
        n_processes : int
            Number of parallel processes

    Returns:
        image : ndarray
            Rendered 2D image
        bounds : tuple
            Actual bounds used (x_min, x_max, y_min, y_max) in nm
    """
    from concurrent.futures import ThreadPoolExecutor

    # Convert to physical coordinates (nm)
    x_nm = locs["x"] * pixelsize
    y_nm = locs["y"] * pixelsize

    # Calculate bounds if not provided
    if bounds is None:
        x_min, x_max = x_nm.min(), x_nm.max()
        y_min, y_max = y_nm.min(), y_nm.max()

        # Add small margin
        margin = 10 * pixelsize_render
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
    else:
        x_min, x_max, y_min, y_max = bounds

    # Calculate chunk grid
    x_range = x_max - x_min
    y_range = y_max - y_min

    n_chunks_x = max(1, int(np.ceil(x_range / chunk_size_nm)))
    n_chunks_y = max(1, int(np.ceil(y_range / chunk_size_nm)))

    logger.debug(f"  Using {n_chunks_x}×{n_chunks_y} spatial chunks")
    logger.debug(
        f"  Chunk size: {chunk_size_nm/1000:.1f} μm, overlap: {overlap_nm} nm"
    )

    # Calculate output image size
    n_pixels_x = int(np.ceil((x_max - x_min) / pixelsize_render))
    n_pixels_y = int(np.ceil((y_max - y_min) / pixelsize_render))

    # Initialize output arrays
    image = np.zeros((n_pixels_x, n_pixels_y), dtype=np.float32)
    weight_map = np.zeros((n_pixels_x, n_pixels_y), dtype=np.float32)

    # Prepare chunk tasks
    chunk_tasks = []
    for i in range(n_chunks_x):
        for j in range(n_chunks_y):
            # Calculate chunk bounds with overlap
            chunk_x_min = x_min + i * chunk_size_nm - overlap_nm
            chunk_x_max = chunk_x_min + chunk_size_nm + 2 * overlap_nm
            chunk_y_min = y_min + j * chunk_size_nm - overlap_nm
            chunk_y_max = chunk_y_min + chunk_size_nm + 2 * overlap_nm

            # Clip to overall bounds
            chunk_x_min = max(chunk_x_min, x_min)
            chunk_x_max = min(chunk_x_max, x_max)
            chunk_y_min = max(chunk_y_min, y_min)
            chunk_y_max = min(chunk_y_max, y_max)

            chunk_bounds = (chunk_x_min, chunk_x_max, chunk_y_min, chunk_y_max)
            chunk_tasks.append(
                (
                    locs,
                    pixelsize,
                    pixelsize_render,
                    chunk_bounds,
                    smoothing_sigma,
                    overlap_nm,
                )
            )

    # Render chunks in parallel
    def render_chunk_worker(task):
        """Worker function to render a single chunk"""
        (
            locs,
            pixelsize,
            pixelsize_render,
            chunk_bounds,
            smoothing_sigma,
            overlap_nm,
        ) = task

        # Extract localizations in this chunk
        x_nm = locs["x"] * pixelsize
        y_nm = locs["y"] * pixelsize

        mask = (
            (x_nm >= chunk_bounds[0])
            & (x_nm < chunk_bounds[1])
            & (y_nm >= chunk_bounds[2])
            & (y_nm < chunk_bounds[3])
        )

        chunk_locs = locs[mask]

        if len(chunk_locs) == 0:
            return None, None, None

        # Render chunk
        chunk_image, _ = render_image_histogram(
            chunk_locs,
            pixelsize,
            pixelsize_render,
            bounds=chunk_bounds,
            smoothing_sigma=smoothing_sigma,
        )

        # Create feathering weight map (distance from edge)
        chunk_shape = chunk_image.shape
        y_idx, x_idx = np.ogrid[: chunk_shape[0], : chunk_shape[1]]

        # Distance to edges (in pixels)
        overlap_pixels = int(overlap_nm / pixelsize_render)
        if overlap_pixels > 0:
            dist_left = np.minimum(x_idx, overlap_pixels)
            dist_right = np.minimum(chunk_shape[1] - 1 - x_idx, overlap_pixels)
            dist_top = np.minimum(y_idx, overlap_pixels)
            dist_bottom = np.minimum(
                chunk_shape[0] - 1 - y_idx, overlap_pixels
            )

            # Weight is minimum distance to any edge, normalized
            weight = (
                np.minimum(
                    np.minimum(dist_left, dist_right),
                    np.minimum(dist_top, dist_bottom),
                ).astype(np.float32)
                / overlap_pixels
            )
        else:
            weight = np.ones(chunk_shape, dtype=np.float32)

        return chunk_image, weight, chunk_bounds

    logger.debug(f"  Rendering {len(chunk_tasks)} chunks in parallel...")

    with ThreadPoolExecutor(max_workers=n_processes) as executor:
        chunk_results = list(executor.map(render_chunk_worker, chunk_tasks))

    # Stitch chunks together with feathering
    logger.debug("  Stitching chunks...")
    for chunk_image, weight, chunk_bounds in chunk_results:
        if chunk_image is None:
            continue

        # Calculate pixel indices for this chunk in output image
        px_x_start = int((chunk_bounds[0] - x_min) / pixelsize_render)
        px_x_end = px_x_start + chunk_image.shape[0]
        px_y_start = int((chunk_bounds[2] - y_min) / pixelsize_render)
        px_y_end = px_y_start + chunk_image.shape[1]

        # Clip to image bounds
        px_x_end = min(px_x_end, n_pixels_x)
        px_y_end = min(px_y_end, n_pixels_y)

        # Accumulate weighted contributions
        image[px_x_start:px_x_end, px_y_start:px_y_end] += chunk_image * weight
        weight_map[px_x_start:px_x_end, px_y_start:px_y_end] += weight

    # Normalize by weights
    valid_mask = weight_map > 0
    image[valid_mask] /= weight_map[valid_mask]

    logger.debug(f"  Stitched image: {image.shape[0]}×{image.shape[1]} pixels")

    bounds = (x_min, x_max, y_min, y_max)

    return image, bounds


def compute_frc_curve_vectorized(
    fft1, fft2, pixelsize_render, max_frc_range_nm=None
):
    """Compute Fourier Ring Correlation curve using vectorized operations

    This fully vectorized implementation uses np.bincount for radial averaging,
    providing 10-100× speedup compared to the loop-based approach.

    Args:
        fft1 : ndarray (complex)
            Shifted FFT of first image
        fft2 : ndarray (complex)
            Shifted FFT of second image
        pixelsize_render : float
            Pixel size of rendered images in nm
        max_frc_range_nm : float or None
            Maximum range to compute (in nm). If specified, only compute
            FRC up to this resolution, skipping high-frequency rings.
            Default: None (compute full curve)

    Returns:
        frc_values : ndarray
            FRC values for each radial bin
        spatial_frequencies : ndarray
            Spatial frequencies in 1/nm
    """
    # Get image dimensions and center
    shape = fft1.shape
    center = np.array(shape) // 2

    # Precompute distance matrix
    y, x = np.ogrid[: shape[0], : shape[1]]
    distances = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Define radial bins
    max_radius = min(center)

    # Optionally limit max radius based on max_resolution
    if max_frc_range_nm is not None:
        freq_spacing = 1.0 / (shape[0] * pixelsize_render)
        min_frequency = 1.0 / max_frc_range_nm  # 1/nm
        max_radius_for_resolution = min_frequency / freq_spacing
        max_radius = min(max_radius, int(np.ceil(max_radius_for_resolution)))
        logger.debug(
            f"  Limited FRC calculation to {max_frc_range_nm} nm "
            f"(max radius: {max_radius} pixels)"
        )

    # Convert distances to integer bins and clip to max_radius
    mem_start_frc, avail_start_frc = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: Starting bincount operations, memory = {mem_start_frc:.1f} MB, "
        f"available = {avail_start_frc:.1f} MB"
    )

    distance_bins = np.round(distances).astype(int)
    np.clip(distance_bins, 0, max_radius, out=distance_bins)  # In-place clip

    # Flatten arrays for bincount
    bins_flat = distance_bins.ravel()
    del distance_bins, distances  # Free memory immediately

    # Limit to max_radius + 1 bins
    max_bin = max_radius + 1

    # Compute cross-correlation (numerator) - process sequentially to save memory
    cross_product = fft1 * np.conj(fft2)

    mem_after_cross, avail_after_cross = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: After cross-product, memory = {mem_after_cross:.1f} MB "
        f"(+{mem_after_cross - mem_start_frc:.1f} MB), "
        f"available = {avail_after_cross:.1f} MB"
    )

    # Process real part
    cross_real = np.real(cross_product).ravel()
    cross_real_sum = np.bincount(
        bins_flat, weights=cross_real, minlength=max_bin
    )
    del cross_real

    # Process imaginary part
    cross_imag = np.imag(cross_product).ravel()
    cross_imag_sum = np.bincount(
        bins_flat, weights=cross_imag, minlength=max_bin
    )
    del cross_product, cross_imag

    mem_after_numerator, avail_after_numerator = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: After numerator, memory = {mem_after_numerator:.1f} MB "
        f"(freed {mem_after_cross - mem_after_numerator:.1f} MB), "
        f"available = {avail_after_numerator:.1f} MB"
    )

    # Compute power spectra (denominators) - one at a time to minimize peak memory
    power1 = np.abs(fft1)
    power1 *= power1  # In-place square
    power1_flat = power1.ravel()
    del power1
    power1_sum = np.bincount(bins_flat, weights=power1_flat, minlength=max_bin)
    del power1_flat

    mem_after_power1, avail_after_power1 = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: After power1, memory = {mem_after_power1:.1f} MB, "
        f"available = {avail_after_power1:.1f} MB"
    )

    power2 = np.abs(fft2)
    power2 *= power2  # In-place square
    power2_flat = power2.ravel()
    del power2
    power2_sum = np.bincount(bins_flat, weights=power2_flat, minlength=max_bin)
    del power2_flat

    mem_after_power2, avail_after_power2 = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: After power2, memory = {mem_after_power2:.1f} MB, "
        f"available = {avail_after_power2:.1f} MB"
    )

    pixel_counts = np.bincount(bins_flat, minlength=max_bin)
    del bins_flat

    mem_after_bincount, avail_after_bincount = _get_memory_usage_mb()
    logger.debug(
        f"      FRC: After all bincount ops, memory = {mem_after_bincount:.1f} MB "
        f"(total: +{mem_after_bincount - mem_start_frc:.1f} MB), "
        f"available = {avail_after_bincount:.1f} MB"
    )

    # Compute FRC for each ring (vectorized)
    # FRC = |sum(F1 * conj(F2))| / sqrt(sum(|F1|^2) * sum(|F2|^2))
    numerator = np.sqrt(cross_real_sum ** 2 + cross_imag_sum ** 2)
    denominator = np.sqrt(power1_sum * power2_sum)

    # Avoid division by zero
    valid = (denominator > 0) & (pixel_counts > 0)
    frc_values = np.full(max_bin, np.nan)
    frc_values[valid] = numerator[valid] / denominator[valid]

    # Calculate spatial frequencies for each bin
    freq_spacing = 1.0 / (shape[0] * pixelsize_render)
    radial_bins = np.arange(max_bin)
    spatial_frequencies = radial_bins * freq_spacing

    # Remove bin 0 (DC component)
    frc_values = frc_values[1:]
    spatial_frequencies = spatial_frequencies[1:]

    logger.debug(
        f"  Computed FRC curve: {len(frc_values)} frequency bins (vectorized)"
    )

    return frc_values, spatial_frequencies


def compute_frc_curve_parallel(fft1, fft2, pixelsize_render, n_processes=4):
    """DEPRECATED: Use compute_frc_curve_vectorized instead

    Compute Fourier Ring Correlation curve with parallel ring processing

    This function is deprecated and kept for backwards compatibility.
    Use compute_frc_curve_vectorized() for 10-100× better performance.

    Args:
        fft1 : ndarray (complex)
            Shifted FFT of first image
        fft2 : ndarray (complex)
            Shifted FFT of second image
        pixelsize_render : float
            Pixel size of rendered images in nm
        n_processes : int
            Number of parallel threads (ignored, kept for compatibility)

    Returns:
        frc_values : ndarray
            FRC values for each radial bin
        spatial_frequencies : ndarray
            Spatial frequencies in 1/nm
    """
    import warnings

    warnings.warn(
        "compute_frc_curve_parallel is deprecated. Use compute_frc_curve_vectorized "
        "for 10-100× better performance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_frc_curve_vectorized(fft1, fft2, pixelsize_render)


def compute_frc_averaged(
    locs,
    pixelsize,
    pixelsize_render=5.0,
    smoothing_sigma=None,
    threshold=1 / 7,
    n_splits=5,
    n_processes=4,
    use_chunking=False,
    chunk_size_nm=10000,
    max_frc_range_nm=None,
    parallel_splits=False,
):
    """Compute FRC resolution averaged over multiple random splits

    This provides more robust resolution estimates by averaging over multiple
    random data splits, with standard deviation as uncertainty estimate.

    Performance notes:
    - If use_chunking=True: Rendering is already parallelized, so parallel_splits
      should typically be False to avoid oversubscription
    - If use_chunking=False: Set parallel_splits=True for speedup with multiple cores

    Args:
        locs : structured array
            Localization data
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm (default: 5 nm)
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels
        threshold : float
            FRC threshold (default: 1/7)
        n_splits : int
            Number of random splits to average (default: 5)
        n_processes : int
            Number of parallel processes for rendering chunks (if use_chunking=True)
            or for parallel splits (if parallel_splits=True)
        use_chunking : bool
            Use chunked rendering for large images (default: False)
        chunk_size_nm : float
            Chunk size for chunked rendering (default: 10000 nm)
        max_frc_range_nm : float or None
            Maximum range to compute (in nm). If specified, only compute
            FRC up to this resolution. Useful for speeding up computation.
            Default: None (compute full curve)
        parallel_splits : bool
            Process splits in parallel (default: False). Only beneficial when
            use_chunking=False. If True and use_chunking=True, a warning is issued.

    Returns:
        results : dict
            Dictionary containing:
                - resolution : float (mean resolution in nm)
                - resolution_std : float (standard deviation)
                - resolutions_per_split : list (resolution for each split)
                - frc_curve_mean : ndarray (mean FRC curve)
                - frc_curve_std : ndarray (std of FRC curves)
                - spatial_frequencies : ndarray
                - threshold : float
                - n_splits : int
    """
    logger.debug(
        f"Computing FRC resolution with {n_splits} splits averaging..."
    )

    frc_curves = []
    resolutions = []
    cutoff_frequencies = []

    # Determine rendering function
    if use_chunking:
        render_func = lambda locs_subset, bounds: render_image_chunked_parallel(
            locs_subset,
            pixelsize,
            pixelsize_render,
            bounds=bounds,
            smoothing_sigma=smoothing_sigma,
            chunk_size_nm=chunk_size_nm,
            n_processes=n_processes,
        )
    else:
        render_func = lambda locs_subset, bounds: render_image_histogram(
            locs_subset,
            pixelsize,
            pixelsize_render,
            bounds=bounds,
            smoothing_sigma=smoothing_sigma,
        )

    # Determine bounds from full dataset (without rendering)
    # This is much faster than rendering the full image
    x_nm = locs["x"] * pixelsize
    y_nm = locs["y"] * pixelsize
    x_min, x_max = x_nm.min(), x_nm.max()
    y_min, y_max = y_nm.min(), y_nm.max()

    # Add small margin
    margin = 10 * pixelsize_render
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    common_bounds = (x_min, x_max, y_min, y_max)
    logger.debug(f"  Using common bounds: {common_bounds}")

    # Warn about potential oversubscription
    if parallel_splits and use_chunking:
        import warnings

        warnings.warn(
            "parallel_splits=True with use_chunking=True may cause CPU oversubscription. "
            "Consider using parallel_splits=False when chunked rendering is enabled.",
            stacklevel=2,
        )

    # Define worker function for a single split
    def process_single_split(split_idx):
        """Process one split: render, FFT, FRC"""
        # Split localizations
        locs_1, locs_2 = split_localizations_random(locs, seed=split_idx)

        # Render images with common bounds
        image_1, _ = render_func(locs_1, common_bounds)
        image_2, _ = render_func(locs_2, common_bounds)

        # Compute FFTs
        fft_1 = compute_fft(image_1)
        fft_2 = compute_fft(image_2)

        # Compute FRC curve
        frc_curve, spatial_frequencies = compute_frc_curve_vectorized(
            fft_1, fft_2, pixelsize_render, max_frc_range_nm=max_frc_range_nm
        )

        # Extract resolution
        resolution, cutoff_frequency = extract_resolution(
            frc_curve, spatial_frequencies, threshold
        )

        return frc_curve, spatial_frequencies, resolution, cutoff_frequency

    # Process splits (parallel or sequential)
    if parallel_splits and n_splits > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor

        # Limit processes to avoid oversubscription
        max_workers = min(n_processes, n_splits)
        logger.debug(
            f"  Processing {n_splits} splits in parallel ({max_workers} workers)..."
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            split_results = list(
                executor.map(process_single_split, range(n_splits))
            )

        # Unpack results
        for (
            frc_curve,
            spatial_frequencies,
            resolution,
            cutoff_frequency,
        ) in split_results:
            frc_curves.append(frc_curve)
            resolutions.append(resolution)
            cutoff_frequencies.append(cutoff_frequency)
    else:
        # Sequential processing
        for split_idx in range(n_splits):
            logger.debug(f"  Processing split {split_idx + 1}/{n_splits}...")

            (
                frc_curve,
                spatial_frequencies,
                resolution,
                cutoff_frequency,
            ) = process_single_split(split_idx)

            frc_curves.append(frc_curve)
            resolutions.append(resolution)
            cutoff_frequencies.append(cutoff_frequency)

    # Compute statistics - now all arrays have same length
    frc_curves = np.array(frc_curves)
    resolutions = np.array(resolutions)

    # Filter out NaN values for statistics
    valid_resolutions = resolutions[~np.isnan(resolutions)]

    if len(valid_resolutions) > 0:
        resolution_mean = np.mean(valid_resolutions)
        resolution_std = (
            np.std(valid_resolutions) if len(valid_resolutions) > 1 else 0.0
        )
    else:
        resolution_mean = np.nan
        resolution_std = np.nan

    frc_curve_mean = np.nanmean(frc_curves, axis=0)
    frc_curve_std = np.nanstd(frc_curves, axis=0)

    logger.debug(
        f"  Mean resolution: {resolution_mean:.2f} ± {resolution_std:.2f} nm"
    )

    # Package results
    results = {
        "resolution": resolution_mean,
        "resolution_std": resolution_std,
        "resolutions_per_split": resolutions.tolist(),
        "frc_curve_mean": frc_curve_mean,
        "frc_curve_std": frc_curve_std,
        "spatial_frequencies": spatial_frequencies,
        "threshold": threshold,
        "n_splits": n_splits,
        "cutoff_frequencies": cutoff_frequencies,
    }

    return results


def _get_memory_usage_mb():
    """Get current process memory usage and available system memory in MB

    Returns:
        tuple: (process_rss_mb, available_mb)
    """
    import os

    import psutil

    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        process_mb = mem_info.rss / 1024 / 1024  # Convert bytes to MB

        # Get system-wide available memory
        virtual_mem = psutil.virtual_memory()
        available_mb = virtual_mem.available / 1024 / 1024

        return process_mb, available_mb
    except:
        return -1, -1  # If psutil not available


def _process_tile_worker_minimal(tile_task):
    """Worker function for processing a single spatial tile

    This worker receives pre-filtered localizations for just this tile,
    minimizing serialization overhead.

    Args:
        tile_task : dict
            Contains:
                - id : tuple (i, j)
                - bounds : tuple (x_min, x_max, y_min, y_max)
                - locs : structured array (pre-filtered to this tile, or None)
                - pixelsize : float
                - pixelsize_render : float
                - smoothing_sigma : float or None
                - threshold : float
                - min_locs_per_region : int
                - max_frc_range_nm : float or None

    Returns:
        dict : Processing results with success flag
    """
    try:
        tile_id = tile_task["id"]
        tile_bounds = tile_task["bounds"]
        tile_locs = tile_task["locs"]
        pixelsize = tile_task["pixelsize"]
        pixelsize_render = tile_task["pixelsize_render"]
        smoothing_sigma = tile_task["smoothing_sigma"]
        threshold = tile_task["threshold"]
        min_locs_per_region = tile_task["min_locs_per_region"]
        max_frc_range_nm = tile_task["max_frc_range_nm"]

        mem_start, avail_start = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: Starting, memory = {mem_start:.1f} MB, "
            f"available = {avail_start:.1f} MB"
        )

        # Handle empty tile
        if tile_locs is None or len(tile_locs) == 0:
            return {
                "tile_id": tile_id,
                "error": "empty_tile",
                "n_locs": 0,
                "success": False,
            }

        # Validate: enough localizations
        if len(tile_locs) < min_locs_per_region:
            return {
                "tile_id": tile_id,
                "error": "insufficient_locs",
                "n_locs": len(tile_locs),
                "success": False,
            }

        # Random split within tile
        locs_1, locs_2 = split_localizations_random(
            tile_locs, seed=tile_id[0] * 1000 + tile_id[1]
        )

        # Validate: balanced split (at least 20% each)
        n_locs_1 = len(locs_1)
        n_locs_2 = len(locs_2)
        split_ratio = min(n_locs_1, n_locs_2) / max(n_locs_1, n_locs_2)

        if split_ratio < 0.2:
            return {
                "tile_id": tile_id,
                "error": "unbalanced_split",
                "n_locs": len(tile_locs),
                "split_ratio": split_ratio,
                "success": False,
            }

        # Render images (use tile bounds for consistent sizing)
        mem_before_render, avail_before_render = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: Before rendering, memory = {mem_before_render:.1f} MB, "
            f"available = {avail_before_render:.1f} MB"
        )

        image_1, _ = render_image_histogram(
            locs_1,
            pixelsize,
            pixelsize_render,
            bounds=tile_bounds,
            smoothing_sigma=smoothing_sigma,
        )
        image_shape = image_1.shape
        image_2, _ = render_image_histogram(
            locs_2,
            pixelsize,
            pixelsize_render,
            bounds=tile_bounds,
            smoothing_sigma=smoothing_sigma,
        )

        mem_after_render, avail_after_render = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: After rendering, memory = {mem_after_render:.1f} MB "
            f"(+{mem_after_render - mem_before_render:.1f} MB), "
            f"available = {avail_after_render:.1f} MB"
        )

        # Validate: non-empty images
        if image_1.sum() == 0 or image_2.sum() == 0:
            return {
                "tile_id": tile_id,
                "error": "empty_image",
                "n_locs": len(tile_locs),
                "success": False,
            }

        # Compute FFTs
        mem_before_fft, avail_before_fft = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: Before FFT, memory = {mem_before_fft:.1f} MB, "
            f"available = {avail_before_fft:.1f} MB"
        )

        fft_1 = compute_fft(image_1)
        fft_2 = compute_fft(image_2)

        mem_after_fft, avail_after_fft = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: After FFT, memory = {mem_after_fft:.1f} MB "
            f"(+{mem_after_fft - mem_before_fft:.1f} MB), "
            f"available = {avail_after_fft:.1f} MB"
        )

        del image_1, image_2

        # Compute FRC curve
        mem_before_frc, avail_before_frc = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: Before FRC computation, memory = {mem_before_frc:.1f} MB, "
            f"available = {avail_before_frc:.1f} MB"
        )

        frc_curve, spatial_frequencies = compute_frc_curve_vectorized(
            fft_1, fft_2, pixelsize_render, max_frc_range_nm=max_frc_range_nm
        )

        mem_after_frc, avail_after_frc = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: After FRC computation, memory = {mem_after_frc:.1f} MB "
            f"(+{mem_after_frc - mem_before_frc:.1f} MB), "
            f"available = {avail_after_frc:.1f} MB"
        )

        del fft_1, fft_2

        # Validate: FRC curve has valid values
        if np.all(np.isnan(frc_curve)):
            return {
                "tile_id": tile_id,
                "error": "invalid_frc",
                "n_locs": len(tile_locs),
                "success": False,
            }

        # Extract resolution
        resolution, cutoff_frequency = extract_resolution(
            frc_curve, spatial_frequencies, threshold
        )

        mem_end, avail_end = _get_memory_usage_mb()
        logger.debug(
            f"    Tile {tile_id}: Completed, final memory = {mem_end:.1f} MB "
            f"(total delta: {mem_end - mem_start:.1f} MB), "
            f"available = {avail_end:.1f} MB"
        )

        # Success - return results
        return {
            "tile_id": tile_id,
            "n_locs": len(tile_locs),
            "n_locs_1": n_locs_1,
            "n_locs_2": n_locs_2,
            "frc_curve": frc_curve,
            "spatial_frequencies": spatial_frequencies,
            "resolution": resolution,
            "cutoff_frequency": cutoff_frequency,
            "image_shape": image_shape,
            "success": True,
        }

    except Exception as e:
        # Catch any unexpected errors
        return {
            "tile_id": tile_task.get("id", "unknown"),
            "error": "exception",
            "error_message": str(e),
            "error_type": type(e).__name__,
            "success": False,
        }


def compute_frc_spatial(
    locs,
    pixelsize,
    pixelsize_render=5.0,
    smoothing_sigma=None,
    threshold=1 / 7,
    region_size=10.0,
    min_locs_per_region=500,
    max_frc_range_nm=None,
    n_processes=4,
    smoothing_window=0.005,
):
    """Compute FRC resolution using spatial tiling approach

    Divides the field-of-view into spatial regions, computes FRC for each
    region independently, and averages the FRC curves. This approach:
    - Reduces memory requirements (smaller images per region)
    - Provides better statistics through spatial averaging
    - Preserves high spatial frequencies within each region
    - Enables efficient multiprocessing

    Args:
        locs : structured array
            Localization data with 'x' and 'y' fields
        pixelsize : float
            Camera pixel size in nm
        pixelsize_render : float
            Rendered pixel size in nm (default: 5 nm)
        smoothing_sigma : float or None
            Gaussian smoothing sigma in pixels
        threshold : float
            FRC threshold (default: 1/7)
        region_size : float
            Size of each spatial region in micrometers (default: 10.0 µm)
        min_locs_per_region : int
            Minimum localizations per region (skip sparse regions, default: 500)
        max_frc_range_nm : float or None
            Maximum FRC range to compute in nm (default: None = full range)
        n_processes : int
            Number of parallel processes (default: 4)
        smoothing_window : float
            Moving average window size in spatial frequency units (1/nm)
            for smoothing the averaged FRC curve (default: 0.005 1/nm)

    Returns:
        results : dict
            Dictionary containing:
                - resolution : float (final resolution from smoothed curve in nm)
                - resolution_std : float (standard deviation)
                - resolution_unsmoothed : float (resolution from raw mean curve)
                - resolutions_per_region : list (resolution for each region)
                - frc_curve_mean : ndarray (mean FRC curve)
                - frc_curve_smoothed : ndarray (smoothed mean FRC curve)
                - frc_curve_std : ndarray (std of FRC curves)
                - spatial_frequencies : ndarray
                - threshold : float
                - n_regions : int (number of valid regions used)
                - n_regions_total : int (total regions attempted)
                - n_regions_x : int (number of regions along x)
                - n_regions_y : int (number of regions along y)
    """
    from concurrent.futures import ProcessPoolExecutor

    # Convert to physical coordinates
    x_nm = locs["x"] * pixelsize
    y_nm = locs["y"] * pixelsize

    # Determine overall bounds
    x_min, x_max = x_nm.min(), x_nm.max()
    y_min, y_max = y_nm.min(), y_nm.max()

    # Calculate region size (no overlap)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Convert region_size from micrometers to nanometers
    region_size_nm = region_size * 1000.0

    # Calculate number of regions based on FOV and region size
    n_regions_x = max(1, int(np.ceil(x_range / region_size_nm)))
    n_regions_y = max(1, int(np.ceil(y_range / region_size_nm)))

    # Recalculate actual region dimensions
    region_width = x_range / n_regions_x
    region_height = y_range / n_regions_y

    logger.debug(
        f"Computing spatial FRC with {n_regions_x}×{n_regions_y} regions "
        f"(target size: {region_size:.1f} µm)..."
    )
    logger.debug(f"  FOV: {x_range:.1f} × {y_range:.1f} nm")
    logger.debug(
        f"  Actual region size: {region_width:.1f} × {region_height:.1f} nm "
        f"({region_width/1000:.2f} × {region_height/1000:.2f} µm)"
    )

    # Pre-calculate consistent tile dimensions in pixels
    # All tiles will have exactly the same pixel dimensions
    tile_width_pixels = int(np.ceil(region_width / pixelsize_render))
    tile_height_pixels = int(np.ceil(region_height / pixelsize_render))

    logger.debug(
        f"  Tile dimensions: {tile_width_pixels}×{tile_height_pixels} pixels"
    )

    # Generate spatial tiles with pre-filtered localizations
    # This approach: filter locs for each tile BEFORE serialization
    # Memory: N_tiles × (locs_per_tile) instead of N_workers × (total_locs)
    logger.debug(f"  Pre-filtering localizations for each tile...")

    tile_tasks = []
    for i in range(n_regions_x):
        for j in range(n_regions_y):
            # Calculate tile bounds (no overlap, non-clamped)
            # Use exact pixel-aligned bounds to ensure consistent sizes
            tile_x_min = x_min + i * region_width
            tile_x_max = tile_x_min + tile_width_pixels * pixelsize_render
            tile_y_min = y_min + j * region_height
            tile_y_max = tile_y_min + tile_height_pixels * pixelsize_render

            # Pre-filter localizations to this tile
            mask = (
                (x_nm >= tile_x_min)
                & (x_nm < tile_x_max)
                & (y_nm >= tile_y_min)
                & (y_nm < tile_y_max)
            )
            tile_locs = locs[mask]

            # Only create task if tile has any localizations
            if len(tile_locs) > 0:
                tile_tasks.append(
                    {
                        "id": (i, j),
                        "bounds": (
                            tile_x_min,
                            tile_x_max,
                            tile_y_min,
                            tile_y_max,
                        ),
                        "locs": tile_locs,  # Only localizations in this tile
                        "pixelsize": pixelsize,
                        "pixelsize_render": pixelsize_render,
                        "smoothing_sigma": smoothing_sigma,
                        "threshold": threshold,
                        "min_locs_per_region": min_locs_per_region,
                        "max_frc_range_nm": max_frc_range_nm,
                    }
                )
            else:
                # Create failed result for empty tile
                tile_tasks.append(
                    {
                        "id": (i, j),
                        "bounds": (
                            tile_x_min,
                            tile_x_max,
                            tile_y_min,
                            tile_y_max,
                        ),
                        "locs": None,  # Marker for empty tile
                        "pixelsize": pixelsize,
                        "pixelsize_render": pixelsize_render,
                        "smoothing_sigma": smoothing_sigma,
                        "threshold": threshold,
                        "min_locs_per_region": min_locs_per_region,
                        "max_frc_range_nm": max_frc_range_nm,
                    }
                )

    logger.debug(f"  Generated {len(tile_tasks)} spatial tiles")

    # Log memory footprint
    total_locs_in_tiles = sum(
        len(t["locs"]) if t["locs"] is not None else 0 for t in tile_tasks
    )
    logger.debug(
        f"  Total localizations across tiles: {total_locs_in_tiles} "
        f"(vs {len(locs)} original)"
    )

    # Process tiles in parallel
    logger.debug(f"  Processing tiles with {n_processes} workers...")

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        tile_results = list(
            executor.map(_process_tile_worker_minimal, tile_tasks)
        )

    # Separate successful and failed tiles
    successful_results = [r for r in tile_results if r.get("success", False)]
    failed_results = [r for r in tile_results if not r.get("success", False)]

    n_success = len(successful_results)
    n_failed = len(failed_results)
    n_total = n_regions_x * n_regions_y

    logger.debug(f"  Successful tiles: {n_success}/{n_total}")

    # Log failures with details
    if n_failed > 0:
        error_counts = {}
        for r in failed_results:
            error_type = r.get("error", "unknown")
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        logger.warning(f"  Failed tiles: {n_failed}/{n_total}")
        for error_type, count in error_counts.items():
            logger.warning(f"    {error_type}: {count} tiles")

    if n_success == 0:
        logger.error("  No successful tiles found!")
        return {
            "resolution": np.nan,
            "resolution_std": np.nan,
            "resolutions_per_region": [],
            "frc_curve_mean": np.array([]),
            "frc_curve_std": np.array([]),
            "spatial_frequencies": np.array([]),
            "threshold": threshold,
            "n_regions": n_success,
            "n_regions_total": n_total,
            "n_failed": n_failed,
            "failed_tiles": failed_results,
        }

    # Validate: all successful tiles have same frequency array length
    freq_lengths = [len(r["spatial_frequencies"]) for r in successful_results]
    if len(set(freq_lengths)) > 1:
        logger.error(f"  Inconsistent FRC curve lengths: {set(freq_lengths)}")
        logger.error(
            "  This indicates inconsistent tile sizes - implementation bug!"
        )
        raise RuntimeError(
            f"Inconsistent FRC curve lengths across tiles: {set(freq_lengths)}. "
            "This should not happen with pixel-aligned bounds."
        )

    # Extract data from successful results
    frc_curves = [r["frc_curve"] for r in successful_results]
    resolutions = [r["resolution"] for r in successful_results]
    spatial_frequencies = successful_results[0][
        "spatial_frequencies"
    ]  # Same for all

    # Compute statistics
    frc_curves = np.array(frc_curves)
    resolutions = np.array(resolutions)

    # Filter out NaN resolutions
    valid_resolutions = resolutions[~np.isnan(resolutions)]

    if len(valid_resolutions) > 0:
        resolution_mean = np.mean(valid_resolutions)
        resolution_std = (
            np.std(valid_resolutions) if len(valid_resolutions) > 1 else 0.0
        )
    else:
        resolution_mean = np.nan
        resolution_std = np.nan

    frc_curve_mean = np.nanmean(frc_curves, axis=0)
    frc_curve_std = np.nanstd(frc_curves, axis=0)

    # Apply moving average smoothing to mean FRC curve
    if smoothing_window > 0:
        # Calculate spatial frequency spacing
        freq_spacing = spatial_frequencies[1] - spatial_frequencies[0]

        # Calculate window size in array indices
        window_size = int(np.round(smoothing_window / freq_spacing))
        window_size = max(1, window_size)  # At least 1

        # Ensure odd window size for symmetric smoothing
        if window_size % 2 == 0:
            window_size += 1

        logger.debug(
            f"  Smoothing FRC curve with window size {window_size} points "
            f"({smoothing_window:.4f} 1/nm)"
        )

        # Apply moving average using convolution
        window = np.ones(window_size) / window_size
        frc_curve_smoothed = np.convolve(frc_curve_mean, window, mode="same")

        # Handle edges where convolution is less accurate
        half_window = window_size // 2
        for i in range(half_window):
            # Left edge
            frc_curve_smoothed[i] = np.nanmean(
                frc_curve_mean[: i + half_window + 1]
            )
            # Right edge
            frc_curve_smoothed[-(i + 1)] = np.nanmean(
                frc_curve_mean[-(i + half_window + 1) :]
            )

        # Extract resolution from smoothed curve
        resolution_smoothed = extract_resolution(
            frc_curve_smoothed, spatial_frequencies, threshold
        )
        resolution_mean_fromcurve = extract_resolution(
            frc_curve_mean, spatial_frequencies, threshold
        )

        logger.debug(
            f"  Smoothed resolution: {resolution_smoothed[0]:.2f} nm "
            f"(unsmoothed: {resolution_mean:.2f} nm)"
            f"(mean from curve: {resolution_mean_fromcurve[0]:.2f} nm)"
        )
    else:
        frc_curve_smoothed = frc_curve_mean.copy()
        resolution_smoothed = resolution_mean

    logger.debug(
        f"  Mean resolution: {resolution_mean:.2f} ± {resolution_std:.2f} nm"
    )
    logger.debug(
        f"  Resolution range: {np.nanmin(resolutions):.2f} - {np.nanmax(resolutions):.2f} nm"
    )

    # Package results
    results = {
        "resolution": resolution_smoothed,
        "resolution_unsmoothed": resolution_mean_fromcurve,
        "resolution_std": resolution_std,
        "resolutions_per_region": resolutions.tolist(),
        "frc_curve_mean": frc_curve_mean,
        "frc_curve_smoothed": frc_curve_smoothed,
        "frc_curve_std": frc_curve_std,
        "spatial_frequencies": spatial_frequencies,
        "threshold": threshold,
        "n_regions": n_success,
        "n_regions_total": n_total,
        "n_regions_x": n_regions_x,
        "n_regions_y": n_regions_y,
        "n_failed": n_failed,
        "tile_info": [
            (r["tile_id"], r["n_locs"], r["resolution"])
            for r in successful_results
        ],  # For debugging
        "failed_tiles": (
            [(r["tile_id"], r.get("error", "unknown")) for r in failed_results]
            if n_failed > 0
            else []
        ),
    }

    return results


def create_frc_plot(frc_results, results_folder, threshold=1 / 7):
    """Create and save FRC curve plot

    Args:
        frc_results : dict
            Results dictionary from compute_frc_spatial containing:
                - frc_curve_mean : ndarray
                - frc_curve_smoothed : ndarray
                - frc_curve_std : ndarray
                - spatial_frequencies : ndarray
                - resolution : tuple or float
                - resolution_unsmoothed : float
                - n_regions : int
                - n_regions_x : int
                - n_regions_y : int
        results_folder : str
            Path to folder where plot should be saved
        threshold : float
            FRC threshold for resolution cutoff (default: 1/7)

    Returns:
        plot_path : str
            Path to saved plot file
    """
    import os

    import matplotlib.pyplot as plt

    # Create FRC curve plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot mean FRC curve with error band
    frc_curve_mean = frc_results["frc_curve_mean"]
    frc_curve_smoothed = frc_results["frc_curve_smoothed"]
    frc_curve_std = frc_results["frc_curve_std"]
    spatial_frequencies = frc_results["spatial_frequencies"]

    ax.plot(
        spatial_frequencies,
        frc_curve_mean,
        "b-",
        linewidth=1.5,
        alpha=0.75,
        label="Mean FRC (unsmoothed)",
    )
    ax.fill_between(
        spatial_frequencies,
        frc_curve_mean - frc_curve_std,
        frc_curve_mean + frc_curve_std,
        alpha=0.2,
        color="blue",
        label=f'±1 SD ({frc_results["n_regions"]} regions)',
    )

    # Plot smoothed curve
    ax.plot(
        spatial_frequencies,
        frc_curve_smoothed,
        "g-",
        linewidth=0.5,
        label="Smoothed FRC",
    )

    # Plot threshold line
    ax.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold (1/7)",
    )

    # Mark smoothed resolution
    resolution = frc_results["resolution"]
    # Handle both tuple and float return values
    if isinstance(resolution, tuple):
        resolution = resolution[0]
    resolution_unsmoothed = frc_results["resolution_unsmoothed"]
    if not np.isnan(resolution):
        resolution_freq = 1.0 / resolution
        ax.axvline(
            x=resolution_freq,
            color="g",
            linestyle=":",
            linewidth=2,
            # label=f"Resolution (smoothed): {resolution:.1f} nm",
            label=f"Resolution: {resolution:.1f} nm",
        )

    # # Optionally mark unsmoothed resolution if different
    # if (
    #     not np.isnan(resolution_unsmoothed)
    #     and abs(resolution - resolution_unsmoothed) > 1.0
    # ):
    #     resolution_freq_unsmoothed = 1.0 / resolution_unsmoothed
    #     ax.axvline(
    #         x=resolution_freq_unsmoothed,
    #         color="orange",
    #         linestyle=":",
    #         linewidth=1.5,
    #         alpha=0.7,
    #         label=f"Resolution (unsmoothed): {resolution_unsmoothed:.1f} nm",
    #     )

    ax.set_xlabel("Spatial Frequency (1/nm)", fontsize=12)
    ax.set_ylabel("FRC", fontsize=12)
    n_regions_x = frc_results["n_regions_x"]
    n_regions_y = frc_results["n_regions_y"]
    ax.set_title(
        f"Spatial FRC Analysis ({n_regions_x}×{n_regions_y} regions)",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()

    # Save plot with random code for unique filename
    import random
    import string

    rcode = "".join(random.choices(string.ascii_letters, k=6))
    plot_path = os.path.join(
        results_folder, f"resolution_frc_spatial_{rcode}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path

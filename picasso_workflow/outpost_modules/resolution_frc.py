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

import numpy as np
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


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

    logger.debug(f"  Split {n} localizations into {len(locs_1)} and {len(locs_2)}")

    return locs_1, locs_2


def render_image_histogram(locs, pixelsize, pixelsize_render, bounds=None,
                           smoothing_sigma=None):
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
    x_nm = locs['x'] * pixelsize
    y_nm = locs['y'] * pixelsize

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

    logger.debug(f"  Rendered image: {image.shape[0]}×{image.shape[1]} pixels")
    logger.debug(f"  Pixel size: {pixelsize_render:.2f} nm")

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
    y, x = np.ogrid[:shape[0], :shape[1]]
    distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)

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
        denom1 = np.sum(np.abs(fft1[mask])**2)
        denom2 = np.sum(np.abs(fft2[mask])**2)

        if denom1 > 0 and denom2 > 0:
            frc_values[i] = np.real(numerator) / np.sqrt(denom1 * denom2)
        else:
            frc_values[i] = np.nan

    # Calculate spatial frequencies
    # Frequency spacing = 1 / (N * pixel_size)
    freq_spacing = 1.0 / (shape[0] * pixelsize_render)
    spatial_frequencies = (radial_bins[:-1] + radial_bins[1:]) / 2 * freq_spacing

    logger.debug(f"  Computed FRC curve: {n_bins} frequency bins")

    return frc_values, spatial_frequencies


def extract_resolution(frc_values, spatial_frequencies, threshold=1/7):
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
            cutoff_frequency = f1 + (threshold - frc1) * (f2 - f1) / (frc2 - frc1)
        else:
            cutoff_frequency = f1
    else:
        cutoff_frequency = freq_valid[cutoff_idx]

    # Resolution = 1 / spatial_frequency
    resolution = 1.0 / cutoff_frequency

    logger.debug(f"  Resolution: {resolution:.2f} nm (cutoff: {cutoff_frequency:.4f} 1/nm)")

    return resolution, cutoff_frequency


def compute_frc_resolution(locs, pixelsize, pixelsize_render=5.0,
                           smoothing_sigma=None, threshold=1/7, seed=None,
                           max_frc_range_nm=None):
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
        locs_2, pixelsize, pixelsize_render, bounds=bounds,
        smoothing_sigma=smoothing_sigma
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
        'resolution': resolution,
        'cutoff_frequency': cutoff_frequency,
        'frc_curve': frc_curve,
        'spatial_frequencies': spatial_frequencies,
        'threshold': threshold,
        'image_1': image_1,
        'image_2': image_2,
        'bounds': bounds,
        'pixelsize_render': pixelsize_render,
    }

    return results


########################################################################
# Phase 2: Optimizations (Memory & Performance)
########################################################################


def render_image_chunked_parallel(locs, pixelsize, pixelsize_render, bounds=None,
                                   smoothing_sigma=None, chunk_size_nm=10000,
                                   overlap_nm=500, n_processes=4):
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
    x_nm = locs['x'] * pixelsize
    y_nm = locs['y'] * pixelsize

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
    logger.debug(f"  Chunk size: {chunk_size_nm/1000:.1f} μm, overlap: {overlap_nm} nm")

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
            chunk_tasks.append((locs, pixelsize, pixelsize_render, chunk_bounds,
                               smoothing_sigma, overlap_nm))

    # Render chunks in parallel
    def render_chunk_worker(task):
        """Worker function to render a single chunk"""
        locs, pixelsize, pixelsize_render, chunk_bounds, smoothing_sigma, overlap_nm = task

        # Extract localizations in this chunk
        x_nm = locs['x'] * pixelsize
        y_nm = locs['y'] * pixelsize

        mask = ((x_nm >= chunk_bounds[0]) & (x_nm < chunk_bounds[1]) &
                (y_nm >= chunk_bounds[2]) & (y_nm < chunk_bounds[3]))

        chunk_locs = locs[mask]

        if len(chunk_locs) == 0:
            return None, None, None

        # Render chunk
        chunk_image, _ = render_image_histogram(
            chunk_locs, pixelsize, pixelsize_render,
            bounds=chunk_bounds, smoothing_sigma=smoothing_sigma
        )

        # Create feathering weight map (distance from edge)
        chunk_shape = chunk_image.shape
        y_idx, x_idx = np.ogrid[:chunk_shape[0], :chunk_shape[1]]

        # Distance to edges (in pixels)
        overlap_pixels = int(overlap_nm / pixelsize_render)
        if overlap_pixels > 0:
            dist_left = np.minimum(x_idx, overlap_pixels)
            dist_right = np.minimum(chunk_shape[1] - 1 - x_idx, overlap_pixels)
            dist_top = np.minimum(y_idx, overlap_pixels)
            dist_bottom = np.minimum(chunk_shape[0] - 1 - y_idx, overlap_pixels)

            # Weight is minimum distance to any edge, normalized
            weight = np.minimum(
                np.minimum(dist_left, dist_right),
                np.minimum(dist_top, dist_bottom)
            ).astype(np.float32) / overlap_pixels
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


def compute_frc_curve_vectorized(fft1, fft2, pixelsize_render, max_frc_range_nm=None):
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
    y, x = np.ogrid[:shape[0], :shape[1]]
    distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Define radial bins
    max_radius = min(center)

    # Optionally limit max radius based on max_resolution
    if max_frc_range_nm is not None:
        freq_spacing = 1.0 / (shape[0] * pixelsize_render)
        min_frequency = 1.0 / max_frc_range_nm  # 1/nm
        max_radius_for_resolution = min_frequency / freq_spacing
        max_radius = min(max_radius, int(np.ceil(max_radius_for_resolution)))
        logger.debug(f"  Limited FRC calculation to {max_frc_range_nm} nm "
                    f"(max radius: {max_radius} pixels)")

    # Convert distances to integer bins
    distance_bins = np.round(distances).astype(int)

    # Flatten arrays for bincount
    bins_flat = distance_bins.ravel()

    # Compute cross-correlation (numerator)
    cross_product = fft1 * np.conj(fft2)
    cross_real = np.real(cross_product).ravel()
    cross_imag = np.imag(cross_product).ravel()

    # Compute power spectra (denominators)
    power1 = (np.abs(fft1)**2).ravel()
    power2 = (np.abs(fft2)**2).ravel()

    # Use bincount for vectorized radial averaging
    # Limit to max_radius + 1 bins
    max_bin = max_radius + 1
    valid_mask = bins_flat <= max_radius

    cross_real_sum = np.bincount(bins_flat[valid_mask],
                                  weights=cross_real[valid_mask],
                                  minlength=max_bin)
    cross_imag_sum = np.bincount(bins_flat[valid_mask],
                                  weights=cross_imag[valid_mask],
                                  minlength=max_bin)
    power1_sum = np.bincount(bins_flat[valid_mask],
                             weights=power1[valid_mask],
                             minlength=max_bin)
    power2_sum = np.bincount(bins_flat[valid_mask],
                             weights=power2[valid_mask],
                             minlength=max_bin)
    pixel_counts = np.bincount(bins_flat[valid_mask], minlength=max_bin)

    # Compute FRC for each ring (vectorized)
    # FRC = |sum(F1 * conj(F2))| / sqrt(sum(|F1|^2) * sum(|F2|^2))
    numerator = np.sqrt(cross_real_sum**2 + cross_imag_sum**2)
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

    logger.debug(f"  Computed FRC curve: {len(frc_values)} frequency bins (vectorized)")

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
        stacklevel=2
    )
    return compute_frc_curve_vectorized(fft1, fft2, pixelsize_render)


def compute_frc_averaged(locs, pixelsize, pixelsize_render=5.0,
                        smoothing_sigma=None, threshold=1/7,
                        n_splits=5, n_processes=4, use_chunking=False,
                        chunk_size_nm=10000, max_frc_range_nm=None):
    """Compute FRC resolution averaged over multiple random splits

    This provides more robust resolution estimates by averaging over multiple
    random data splits, with standard deviation as uncertainty estimate.

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
            Number of parallel processes for rendering/FRC (deprecated)
        use_chunking : bool
            Use chunked rendering for large images (default: False)
        chunk_size_nm : float
            Chunk size for chunked rendering (default: 10000 nm)
        max_frc_range_nm : float or None
            Maximum range to compute (in nm). If specified, only compute
            FRC up to this resolution. Useful for speeding up computation.
            Default: None (compute full curve)

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
    logger.debug(f"Computing FRC resolution with {n_splits} splits averaging...")

    frc_curves = []
    resolutions = []
    cutoff_frequencies = []

    # Determine rendering function
    if use_chunking:
        render_func = lambda locs_subset, bounds: render_image_chunked_parallel(
            locs_subset, pixelsize, pixelsize_render, bounds=bounds,
            smoothing_sigma=smoothing_sigma, chunk_size_nm=chunk_size_nm,
            n_processes=n_processes
        )
    else:
        render_func = lambda locs_subset, bounds: render_image_histogram(
            locs_subset, pixelsize, pixelsize_render, bounds=bounds,
            smoothing_sigma=smoothing_sigma
        )

    # Determine bounds from full dataset to ensure consistent image size
    if use_chunking:
        first_image, common_bounds = render_image_chunked_parallel(
            locs, pixelsize, pixelsize_render, bounds=None,
            smoothing_sigma=smoothing_sigma, chunk_size_nm=chunk_size_nm,
            n_processes=n_processes
        )
    else:
        first_image, common_bounds = render_image_histogram(
            locs, pixelsize, pixelsize_render, bounds=None,
            smoothing_sigma=smoothing_sigma
        )

    logger.debug(f"  Using common bounds: {common_bounds}")

    for split_idx in range(n_splits):
        logger.debug(f"  Processing split {split_idx + 1}/{n_splits}...")

        # Split localizations
        locs_1, locs_2 = split_localizations_random(locs, seed=split_idx)

        # Render images with common bounds to ensure same size
        logger.debug(f"  Rendering images")
        image_1, _ = render_func(locs_1, common_bounds)
        image_2, _ = render_func(locs_2, common_bounds)

        # Compute FFTs
        logger.debug(f"  Computing 2 FFTs")
        fft_1 = compute_fft(image_1)
        fft_2 = compute_fft(image_2)

        logger.debug(f"  Computing FRC Curve")
        # Compute FRC curve (use vectorized version)
        frc_curve, spatial_frequencies = compute_frc_curve_vectorized(
            fft_1, fft_2, pixelsize_render, max_frc_range_nm=max_frc_range_nm
        )

        # Extract resolution
        logger.debug(f"  Extracting Resolution")
        resolution, cutoff_frequency = extract_resolution(
            frc_curve, spatial_frequencies, threshold
        )

        # Store results
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
        resolution_std = np.std(valid_resolutions) if len(valid_resolutions) > 1 else 0.0
    else:
        resolution_mean = np.nan
        resolution_std = np.nan

    frc_curve_mean = np.nanmean(frc_curves, axis=0)
    frc_curve_std = np.nanstd(frc_curves, axis=0)

    logger.debug(f"  Mean resolution: {resolution_mean:.2f} ± {resolution_std:.2f} nm")

    # Package results
    results = {
        'resolution': resolution_mean,
        'resolution_std': resolution_std,
        'resolutions_per_split': resolutions.tolist(),
        'frc_curve_mean': frc_curve_mean,
        'frc_curve_std': frc_curve_std,
        'spatial_frequencies': spatial_frequencies,
        'threshold': threshold,
        'n_splits': n_splits,
        'cutoff_frequencies': cutoff_frequencies,
    }

    return results

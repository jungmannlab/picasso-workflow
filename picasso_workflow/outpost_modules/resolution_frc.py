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
                           smoothing_sigma=None, threshold=1/7, seed=None):
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

    # Step 4: Compute FRC curve
    logger.debug("  Computing FRC curve...")
    frc_curve, spatial_frequencies = compute_frc_curve(
        fft_1, fft_2, pixelsize_render
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

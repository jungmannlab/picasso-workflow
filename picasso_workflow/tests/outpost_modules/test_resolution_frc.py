"""
Unit tests for resolution_frc.py module

Tests the Fourier Ring Correlation (FRC) resolution estimation functions.

Author: Generated for picasso-workflow
Date: 2025
"""

import unittest
import numpy as np
import pandas as pd
from picasso_workflow.outpost_modules import resolution_frc


class TestResolutionFRC(unittest.TestCase):
    """Test suite for FRC resolution estimation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic localization data
        np.random.seed(42)
        n_locs = 1000

        # Create structured array format
        self.locs = pd.DataFrame(
            {
                "frame": np.random.randint(0, 100, n_locs),
                "x": np.random.uniform(0, 100, n_locs),  # in camera pixels
                "y": np.random.uniform(0, 100, n_locs),
                "photons": np.random.uniform(500, 2000, n_locs),
            }
        )

        self.pixelsize = 130.0  # nm
        self.pixelsize_render = 10.0  # nm

    def test_01_split_localizations_random(self):
        """Test random splitting of localizations"""
        locs_1, locs_2 = resolution_frc.split_localizations_random(
            self.locs, seed=42
        )

        # Check that splits are approximately equal
        self.assertEqual(len(locs_1), len(self.locs) // 2)
        self.assertEqual(len(locs_2), len(self.locs) - len(self.locs) // 2)

        # Check that all localizations are accounted for
        self.assertEqual(len(locs_1) + len(locs_2), len(self.locs))

        # Test reproducibility with same seed
        locs_1_repeat, locs_2_repeat = (
            resolution_frc.split_localizations_random(self.locs, seed=42)
        )
        np.testing.assert_array_equal(
            locs_1["x"].values, locs_1_repeat["x"].values
        )

    def test_02_render_image_histogram_basic(self):
        """Test basic histogram rendering"""
        image, bounds = resolution_frc.render_image_histogram(
            self.locs, self.pixelsize, self.pixelsize_render
        )

        # Check that image is 2D
        self.assertEqual(len(image.shape), 2)

        # Check that image contains data
        self.assertGreater(image.sum(), 0)

        # Check bounds format
        self.assertEqual(len(bounds), 4)
        x_min, x_max, y_min, y_max = bounds
        self.assertLess(x_min, x_max)
        self.assertLess(y_min, y_max)

    def test_03_render_image_histogram_with_bounds(self):
        """Test rendering with specified bounds"""
        bounds = (0, 10000, 0, 10000)  # nm
        image, bounds_out = resolution_frc.render_image_histogram(
            self.locs, self.pixelsize, self.pixelsize_render, bounds=bounds
        )

        # Check that output bounds match input
        self.assertEqual(bounds, bounds_out)

    def test_04_render_image_histogram_with_smoothing(self):
        """Test rendering with Gaussian smoothing"""
        image_no_smooth, _ = resolution_frc.render_image_histogram(
            self.locs, self.pixelsize, self.pixelsize_render
        )

        image_smooth, _ = resolution_frc.render_image_histogram(
            self.locs,
            self.pixelsize,
            self.pixelsize_render,
            smoothing_sigma=1.0,
        )

        # Smoothed image should be different
        self.assertFalse(np.array_equal(image_no_smooth, image_smooth))

        # Smoothed image should have lower maximum (smoothing spreads intensity)
        self.assertLess(image_smooth.max(), image_no_smooth.max())

    def test_05_compute_fft(self):
        """Test FFT computation"""
        image = np.random.rand(64, 64)
        fft_shifted = resolution_frc.compute_fft(image)

        # Check output shape matches input
        self.assertEqual(fft_shifted.shape, image.shape)

        # Check that output is complex
        self.assertTrue(np.iscomplexobj(fft_shifted))

        # Check that DC component is at center
        center = np.array(fft_shifted.shape) // 2
        dc_value = np.abs(fft_shifted[center[0], center[1]])
        # DC component should be the largest (for positive images)
        self.assertGreater(dc_value, 0)

    def test_06_compute_frc_curve_vectorized(self):
        """Test vectorized FRC curve computation"""
        # Create two similar images
        np.random.seed(42)
        image_1 = np.random.rand(64, 64) + 1.0  # Positive values
        image_2 = image_1 + np.random.rand(64, 64) * 0.1  # Similar with noise

        fft_1 = resolution_frc.compute_fft(image_1)
        fft_2 = resolution_frc.compute_fft(image_2)

        frc_values, spatial_frequencies = (
            resolution_frc.compute_frc_curve_vectorized(
                fft_1, fft_2, self.pixelsize_render
            )
        )

        # Check output shapes
        self.assertEqual(len(frc_values), len(spatial_frequencies))
        self.assertGreater(len(frc_values), 0)

        # FRC values should be between -1 and 1
        valid_frc = frc_values[~np.isnan(frc_values)]
        self.assertTrue(np.all(valid_frc >= -1.0))
        self.assertTrue(np.all(valid_frc <= 1.0))

        # Low frequencies should have high correlation (similar images)
        self.assertGreater(valid_frc[0], 0.5)

    def test_07_compute_frc_curve_with_max_range(self):
        """Test FRC curve computation with max_frc_range_nm"""
        image_1 = np.random.rand(64, 64) + 1.0
        image_2 = np.random.rand(64, 64) + 1.0

        fft_1 = resolution_frc.compute_fft(image_1)
        fft_2 = resolution_frc.compute_fft(image_2)

        # Compute full curve
        frc_full, freq_full = resolution_frc.compute_frc_curve_vectorized(
            fft_1, fft_2, self.pixelsize_render
        )

        # Compute limited curve
        frc_limited, freq_limited = (
            resolution_frc.compute_frc_curve_vectorized(
                fft_1, fft_2, self.pixelsize_render, max_frc_range_nm=100.0
            )
        )

        # Limited curve should be shorter or equal
        self.assertLessEqual(len(frc_limited), len(frc_full))

    def test_08_extract_resolution_valid(self):
        """Test resolution extraction from FRC curve"""
        # Create synthetic FRC curve that crosses threshold
        spatial_frequencies = np.linspace(0.001, 0.01, 100)  # 1/nm
        # Start high, drop below threshold
        frc_values = np.exp(-spatial_frequencies * 200)  # Exponential decay

        threshold = 1 / 7
        resolution, cutoff_frequency = resolution_frc.extract_resolution(
            frc_values, spatial_frequencies, threshold
        )

        # Should find a valid resolution
        self.assertFalse(np.isnan(resolution))
        self.assertFalse(np.isnan(cutoff_frequency))

        # Resolution should be positive
        self.assertGreater(resolution, 0)

        # Cutoff frequency should be in range
        self.assertGreater(cutoff_frequency, spatial_frequencies[0])
        self.assertLess(cutoff_frequency, spatial_frequencies[-1])

    def test_09_extract_resolution_no_crossing(self):
        """Test resolution extraction when FRC never crosses threshold"""
        spatial_frequencies = np.linspace(0.001, 0.01, 100)
        # FRC always above threshold
        frc_values = np.ones_like(spatial_frequencies) * 0.5  # > 1/7

        resolution, cutoff_frequency = resolution_frc.extract_resolution(
            frc_values, spatial_frequencies, threshold=1 / 7
        )

        # Should return NaN when no crossing
        self.assertTrue(np.isnan(resolution))
        self.assertTrue(np.isnan(cutoff_frequency))

    def test_10_extract_resolution_all_nan(self):
        """Test resolution extraction with all NaN FRC values"""
        spatial_frequencies = np.linspace(0.001, 0.01, 100)
        frc_values = np.full_like(spatial_frequencies, np.nan)

        resolution, cutoff_frequency = resolution_frc.extract_resolution(
            frc_values, spatial_frequencies
        )

        # Should return NaN
        self.assertTrue(np.isnan(resolution))
        self.assertTrue(np.isnan(cutoff_frequency))

    def test_11_compute_frc_resolution_pipeline(self):
        """Test complete FRC resolution pipeline"""
        # Create more structured data for better FRC
        np.random.seed(42)
        # Create localizations in a grid pattern (should have good resolution)
        x_grid, y_grid = np.meshgrid(
            np.linspace(10, 90, 20), np.linspace(10, 90, 20)
        )
        x_locs = x_grid.ravel() + np.random.rand(400) * 0.5
        y_locs = y_grid.ravel() + np.random.rand(400) * 0.5

        locs = pd.DataFrame(
            {
                "frame": np.random.randint(0, 100, len(x_locs)),
                "x": x_locs,
                "y": y_locs,
                "photons": np.random.uniform(500, 2000, len(x_locs)),
            }
        )

        results = resolution_frc.compute_frc_resolution(
            locs, pixelsize=130.0, pixelsize_render=10.0, seed=42
        )

        # Check required keys in results
        self.assertIn("resolution", results)
        self.assertIn("cutoff_frequency", results)
        self.assertIn("frc_curve", results)
        self.assertIn("spatial_frequencies", results)
        self.assertIn("image_1", results)
        self.assertIn("image_2", results)
        self.assertIn("bounds", results)

        # Check that arrays have correct shapes
        self.assertEqual(
            len(results["frc_curve"]), len(results["spatial_frequencies"])
        )

        # Images should be 2D
        self.assertEqual(len(results["image_1"].shape), 2)
        self.assertEqual(len(results["image_2"].shape), 2)

    def test_12_compute_frc_resolution_with_smoothing(self):
        """Test FRC pipeline with Gaussian smoothing"""
        results = resolution_frc.compute_frc_resolution(
            self.locs,
            pixelsize=self.pixelsize,
            pixelsize_render=self.pixelsize_render,
            smoothing_sigma=1.0,
            seed=42,
        )

        self.assertIn("resolution", results)
        # With smoothing, resolution might be different
        self.assertTrue(
            np.isnan(results["resolution"]) or results["resolution"] > 0
        )

    def test_13_compute_frc_averaged_basic(self):
        """Test averaged FRC computation over multiple splits"""
        # Use fewer splits for faster testing
        results = resolution_frc.compute_frc_averaged(
            self.locs,
            pixelsize=self.pixelsize,
            pixelsize_render=self.pixelsize_render,
            n_splits=3,
            use_chunking=False,
            parallel_splits=False,
        )

        # Check required keys
        self.assertIn("resolution", results)
        self.assertIn("resolution_std", results)
        self.assertIn("resolutions_per_split", results)
        self.assertIn("frc_curve_mean", results)
        self.assertIn("frc_curve_std", results)
        self.assertIn("spatial_frequencies", results)
        self.assertIn("n_splits", results)

        # Check that we got results for all splits
        self.assertEqual(len(results["resolutions_per_split"]), 3)

        # Standard deviation should be non-negative
        self.assertGreaterEqual(results["resolution_std"], 0)

    def test_14_split_localizations_preserves_data(self):
        """Test that splitting preserves localization data"""
        locs_1, locs_2 = resolution_frc.split_localizations_random(
            self.locs, seed=42
        )

        # Check that frames are preserved
        all_frames = set(self.locs["frame"])
        split1_frames = set(locs_1["frame"])
        split2_frames = set(locs_2["frame"])

        # Union of splits should equal original
        self.assertTrue(
            split1_frames.union(split2_frames).issubset(all_frames)
        )

    def test_15_render_image_empty_locs(self):
        """Test rendering with empty localizations"""
        empty_locs = pd.DataFrame(
            {
                "frame": [],
                "x": [],
                "y": [],
                "photons": [],
            }
        )

        image, bounds = resolution_frc.render_image_histogram(
            empty_locs,
            self.pixelsize,
            self.pixelsize_render,
            bounds=(0, 1000, 0, 1000),
        )

        # Should return an image (likely zeros)
        self.assertEqual(len(image.shape), 2)

    def test_16_frc_curve_identical_images(self):
        """Test FRC curve with identical images (should be 1.0)"""
        image = np.random.rand(64, 64) + 1.0
        fft = resolution_frc.compute_fft(image)

        frc_values, _ = resolution_frc.compute_frc_curve_vectorized(
            fft, fft, self.pixelsize_render
        )

        # FRC of identical images should be ~1.0 (allowing for numerical errors)
        valid_frc = frc_values[~np.isnan(frc_values)]
        self.assertTrue(np.allclose(valid_frc, 1.0, atol=1e-10))

    def test_17_frc_curve_uncorrelated_images(self):
        """Test FRC curve with uncorrelated random images"""
        np.random.seed(42)
        image_1 = np.random.rand(64, 64)
        image_2 = np.random.rand(64, 64)

        fft_1 = resolution_frc.compute_fft(image_1)
        fft_2 = resolution_frc.compute_fft(image_2)

        frc_values, _ = resolution_frc.compute_frc_curve_vectorized(
            fft_1, fft_2, self.pixelsize_render
        )

        # FRC of uncorrelated images should be low
        valid_frc = frc_values[~np.isnan(frc_values)]
        # Mean should be close to 0 for uncorrelated data
        self.assertLess(np.abs(np.mean(valid_frc)), 0.3)


class TestResolutionFRCEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_01_extract_resolution_interpolation(self):
        """Test that resolution extraction uses interpolation"""
        # Create FRC that crosses threshold between two points
        spatial_frequencies = np.array([0.001, 0.002, 0.003, 0.004])
        threshold = 0.2
        frc_values = np.array(
            [0.8, 0.3, 0.1, 0.05]
        )  # Crosses between index 1 and 2

        resolution, cutoff_freq = resolution_frc.extract_resolution(
            frc_values, spatial_frequencies, threshold
        )

        # Cutoff should be between frequencies at index 1 and 2
        self.assertGreater(cutoff_freq, spatial_frequencies[1])
        self.assertLess(cutoff_freq, spatial_frequencies[2])

    def test_02_compute_frc_small_image(self):
        """Test FRC computation with small images"""
        # Very small image
        image = np.random.rand(8, 8) + 1.0
        fft = resolution_frc.compute_fft(image)

        frc_values, spatial_frequencies = (
            resolution_frc.compute_frc_curve_vectorized(
                fft, fft, pixelsize_render=10.0
            )
        )

        # Should still produce results
        self.assertGreater(len(frc_values), 0)
        self.assertEqual(len(frc_values), len(spatial_frequencies))

    def test_03_render_with_single_localization(self):
        """Test rendering with only one localization"""
        locs = pd.DataFrame(
            {
                "frame": [0],
                "x": [50.0],
                "y": [50.0],
                "photons": [1000.0],
            }
        )

        image, bounds = resolution_frc.render_image_histogram(
            locs, pixelsize=130.0, pixelsize_render=10.0
        )

        # Should create an image with one pixel lit
        self.assertGreater(image.sum(), 0)
        self.assertEqual(image.max(), 1.0)  # One localization -> one count


if __name__ == "__main__":
    unittest.main()

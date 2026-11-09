"""
Unit tests for resolution_decorrelation.py module

Tests the image-decorrelation resolution estimation (Descloux et al.).

Author: Generated for picasso-workflow
"""

import unittest
import numpy as np
from picasso_workflow.outpost.resolution import resolution_decorrelation


class TestExtractResolutionDecorr(unittest.TestCase):
    """Test suite for extract_resolution_decorr."""

    def test_decorr_resolution_is_inverse_cutoff(self):
        """Resolution equals 1/kc_max == 2*pixelsize_render/r_cutoff.

        The decorrelation curve steps from 0.8 to 0.2 so that it crosses
        the 0.5 threshold at a normalized radius r_cutoff ≈ 0.5. With
        pixelsize_render=10 nm:

            kc_max = (r_cutoff * 0.5) / pixelsize_render = 0.025 1/nm
            resolution = 1 / kc_max = 2 * pixelsize_render / r_cutoff = 40 nm

        The previous buggy formula used ``2 / kc_max`` and reported ~80 nm
        (twice the Descloux resolution); this guards against regressing to
        that double-counted value.
        """
        r_values = np.linspace(0, 1, 50)
        # Step from 0.8 to 0.2 at the normalized radius 0.5.
        decorr_curve = np.where(r_values < 0.5, 0.8, 0.2)

        kc_max, resolution = (
            resolution_decorrelation.extract_resolution_decorr(
                decorr_curve, r_values, pixelsize_render=10.0
            )
        )

        # Threshold crossing is near r=0.5, so kc_max ≈ 0.025 1/nm.
        self.assertAlmostEqual(kc_max, 0.025, places=3)

        # resolution == 1 / kc_max (the fixed formula), i.e. 40 nm.
        self.assertAlmostEqual(resolution, 40.0, places=3)
        self.assertAlmostEqual(resolution, 1.0 / kc_max, places=9)

        # The old ``2 / kc_max`` would have returned ~80 nm.
        self.assertNotAlmostEqual(resolution, 2.0 / kc_max, places=3)

    def test_decorr_no_crossing_returns_nan(self):
        """A curve that never drops below threshold returns NaN."""
        r_values = np.linspace(0, 1, 50)
        decorr_curve = np.full_like(r_values, 0.8)  # always > 0.5

        kc_max, resolution = (
            resolution_decorrelation.extract_resolution_decorr(
                decorr_curve, r_values, pixelsize_render=10.0
            )
        )

        self.assertTrue(np.isnan(kc_max))
        self.assertTrue(np.isnan(resolution))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Standalone testing script for AutoPicasso modules with custom data.

This script allows you to test individual AutoPicasso modules with your own
localization data without running a full workflow or integrating into unit tests.

Usage:
1. Edit the configuration section below
2. Run: python test_autopicasso_module.py

Author: Claude Code
"""

import os
import sys
import numpy as np
from pathlib import Path
from picasso import io
import traceback
from picasso_workflow.analyse import AutoPicasso


# Add the package to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Path to your localization data (HDF5 or CSV format)
LOCS_FILE_PATH = "/path/to/locs.hdf5"  # CHANGE THIS

# Module to test (method name from AutoPicasso)
MODULE_TO_TEST = "undrift_rsso"  # CHANGE THIS

# Parameters for the module (adjust based on your module)
MODULE_PARAMETERS = {
    "ton": 10.0,  # Half-life of localization in frames
    "toff": 100.0,  # Time for spot reappearance in frames
    "max_shift": 2.0,  # Maximum expected drift per frame in pixels
    "processing_chunk_size": 50,  # Frames per processing chunk for memory efficiency
    "min_locs_per_frame": 5,  # Minimum localizations per frame
    "min_locs_per_block": 50,  # Minimum localizations per toff-scale block
    "plot_drift": True,  # Whether to save drift plots
    "save_locs": True,  # Whether to save undrifted localizations
}

# Optional: Camera parameters (will use defaults if not specified)
CAMERA_PARAMS = {
    "Pixelsize": 130.0,  # Camera pixel size in nm
    # Add other camera parameters as needed
}

# Output directory for results and plots
OUTPUT_DIR = "./autopicasso_test_results"

# Enable detailed logging
VERBOSE = True

# =============================================================================
# TEST CONFIGURATION EXAMPLES FOR DIFFERENT MODULES
# =============================================================================

# Example configurations for other common modules:
"""
# For localize module:
MODULE_TO_TEST = "localize"
MODULE_PARAMETERS = {
    "box": 7,
    "gradient": 5000.0,
    "min_net_gradient": 0.0,
    "fit_method": "mle",
    "save_locs": True
}

# For cluster_dbscan module:
MODULE_TO_TEST = "cluster_dbscan"
MODULE_PARAMETERS = {
    "radius": 0.05,
    "min_density": 10,
    "save_cluster_info": True
}

# For undrift module:
MODULE_TO_TEST = "undrift"
MODULE_PARAMETERS = {
    "segmentation": 1000,
    "display": True,
    "save_locs": True
}

# For render module:
MODULE_TO_TEST = "render"
MODULE_PARAMETERS = {
    "oversampling": 10,
    "viewport": [0, 0, 100, 100],
    "blur_method": "gaussian",
    "save_render": True
}
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_localization_data(file_path):
    """Load localization data from HDF5 or CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Localization file not found: {file_path}")

    print(f"Loading localization data from: {file_path}")

    if file_path.suffix.lower() == ".hdf5":
        # Load HDF5 format (picasso standard)
        locs, info = io.load_locs(file_path)
    elif file_path.suffix.lower() == ".csv":
        # Load CSV format
        import pandas as pd

        df = pd.read_csv(file_path)
        # Convert to structured array (picasso format)
        locs = df.to_records(index=False)
        info = []
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    print(f"Loaded {len(locs)} localizations")
    if VERBOSE:
        print(f"Data columns: {locs.dtype.names}")
        print(f"Frame range: {locs['frame'].min()} - {locs['frame'].max()}")
        if "x" in locs.dtype.names and "y" in locs.dtype.names:
            print(f"X range: {locs['x'].min():.2f} - {locs['x'].max():.2f}")
            print(f"Y range: {locs['y'].min():.2f} - {locs['y'].max():.2f}")

    return locs, info


def setup_autopicasso(locs, info, result_location, camera_info):
    """Initialize AutoPicasso instance with data."""
    print("Setting up AutoPicasso instance...")

    analysis_config = {
        "result_location": result_location,
        "camera_info": camera_info,
        "gpufit_installed": False,
        "always_save": True,
    }
    # Create AutoPicasso instance
    ap = AutoPicasso(result_location, analysis_config)

    # Load localization data
    ap.locs = locs
    ap.info = info

    return ap


def run_module_test(ap, module_name, parameters, output_dir):
    """Run a specific module test."""
    print(f"\nTesting module: {module_name}")
    print(f"Parameters: {parameters}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results dictionary
    results = {"folder": str(output_path)}

    try:
        # Get the module method
        if not hasattr(ap, module_name):
            raise AttributeError(f"AutoPicasso has no module '{module_name}'")

        module_method = getattr(ap, module_name)

        # Run the module
        print(f"Running {module_name}...")
        returned_params, returned_results = module_method(
            i=0,  # Module index
            parameters=parameters.copy(),
            # results=results
        )

        print(f"✓ Module {module_name} completed successfully!")

        # Print results summary
        if VERBOSE and returned_results:
            print("\nResults summary:")
            for key, value in returned_results.items():
                if isinstance(value, (int, float, bool, str)):
                    print(f"  {key}: {value}")
                elif isinstance(value, np.ndarray) and value.size < 10:
                    print(f"  {key}: {value}")
                else:
                    print(
                        f"  {key}: {type(value)} "
                        + f"(size: {getattr(value, 'shape', 'N/A')})"
                    )

        return returned_params, returned_results, True

    except Exception as e:
        print(f"✗ Module {module_name} failed!")
        print(f"Error: {str(e)}")
        if VERBOSE:
            print("\nFull traceback:")
            traceback.print_exc()
        return parameters, results, False


def save_results(ap, output_dir, module_name):
    """Save results and data to output directory."""
    output_path = Path(output_dir)

    # Save updated localization data if it changed
    if hasattr(ap, "locs") and ap.locs is not None:
        locs_output = output_path / f"{module_name}_output_locs.hdf5"
        print(f"Saving updated localizations to: {locs_output}")

        import h5py

        with h5py.File(locs_output, "w") as f:
            f.create_dataset("locs", data=ap.locs)

            # Save info if available
            if hasattr(ap, "info") and ap.info:
                info_str = str(ap.info)
                f.attrs["info"] = info_str

    # Save drift data if available
    if hasattr(ap, "drift") and ap.drift is not None:
        drift_output = output_path / f"{module_name}_drift.csv"
        print(f"Saving drift data to: {drift_output}")
        np.savetxt(
            drift_output,
            ap.drift,
            delimiter=",",
            header="drift_x,drift_y",
            comments="",
        )

    # List generated files
    generated_files = list(output_path.glob("*"))
    if generated_files:
        print(f"\nGenerated files in {output_path}:")
        for file in sorted(generated_files):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name} ({size_mb:.2f} MB)")


def main():
    """Main testing function."""
    print("AutoPicasso Module Tester")
    print("=" * 50)

    try:
        # Load data
        locs, info = load_localization_data(LOCS_FILE_PATH)

        # Setup AutoPicasso
        ap = setup_autopicasso(locs, info, OUTPUT_DIR, CAMERA_PARAMS)

        # Run module test
        params, results, success = run_module_test(
            ap, MODULE_TO_TEST, MODULE_PARAMETERS, OUTPUT_DIR
        )

        if success:
            # Save results
            save_results(ap, OUTPUT_DIR, MODULE_TO_TEST)

            print("\n✓ Test completed successfully!")
            print(f"Check results in: {OUTPUT_DIR}")
        else:
            print("\n✗ Test failed!")
            return 1

    except Exception as e:
        print(f"\n✗ Test setup failed: {str(e)}")
        if VERBOSE:
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Validate configuration before running
    if LOCS_FILE_PATH == "/path/to/your/locs_data.hdf5":
        print("ERROR: Please edit LOCS_FILE_PATH to point to your data file!")
        sys.exit(1)

    exit_code = main()
    sys.exit(exit_code)

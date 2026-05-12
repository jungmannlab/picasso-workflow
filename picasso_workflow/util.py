#!/usr/bin/env python
"""
Module Name: util.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Utility functions for the package
"""
import abc
import copy
import inspect

# import logging
from loguru import logger
import os
import re

import numpy as np

# logger = logging.getLogger(__name__)


class AbstractModuleCollection(abc.ABC):
    """Describes the modules an analysis and reporting pipeline
    must support. This needs to be implemented
    in classes in analyse.py and confluence.py,
    such that the workflow class can call the other's methods
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def dummy_module(self, i, parameters, results):
        """A module that does nothing, for quickly removing
        modules in a workflow without having to renumber the
        following result idcs. Only for workflow debugging,
        remove when done.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results (unchanged)
        """
        pass

    @abc.abstractmethod
    def analysis_documentation(self, i, parameters, results):
        """This module documents where and how analysis is being performed
        Args:
            parameters : dict
                This module does not use any parameters
        Returns:
            parameters : dict
                as input, unchanged
            results : dict
                the analysis results, updated with:
                    picasso version : str
                        version of picasso library used
                    picasso-workflow version : str
                        version of picasso-workflow
                    Architecture : str
                        machine architecture
                    OS : str
                        operating system
                    host : str
                        hostname of machine
                    processor : str
                        processor information
                    CPU Frequency [MHz] : float
                        current CPU frequency
                    CPU cores : int
                        number of CPU cores
                    Memory total [GB] : int
                        total system memory in GB
                    Memory available [GB] : int
                        available system memory in GB
                    GPU : str
                        GPU name or "N/A"
                    GPU memory [GB] : int
                        GPU memory in GB or 0 if no GPU
        """
        pass

    @abc.abstractmethod
    def conditional_branch(self):
        """Execute different sub-module sequences based on a condition.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    condition : dict
                        condition dictionary with keys:
                            - "left": value or parameter command tuple
                            - "operator": str (>, <, >=, <=, ==, !=)
                            - "right": value or parameter command tuple
                        or logical condition with "and"/"or" keys
                    if_true : list of tuples
                        list of (module_name, module_parameters) tuples
                        to execute if condition is True
                    if_false : list of tuples
                        list of (module_name, module_parameters) tuples
                        to execute if condition is False
                optional keys:
                    parameter_command_executor : ParameterCommandExecutor
                        if provided, will be used for resolving parameter
                        commands in condition values
            results : dict
                the results this function generates

        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results including:
                    - condition_result : bool
                    - branch_taken : str ("if_true" or "if_false")
                    - if_branch : dict of sub-module results
                    - branch_modules : dict of flat-indexed results
        """
        pass

    ##########################################################################
    # Single-dataset workflow modules
    ##########################################################################

    @abc.abstractmethod
    def convert_zeiss_movie(self, i, parameters, results):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.
        Args:
            parameters : dict
                necessary items:
                    filepath : str
                        the czi file name to load.
                optional items:
                    filename_raw : str
                        the raw file name to write to
                    info : dict, information as used by picasso
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    filepath_raw : str
                        full path to the output raw file
                    filename_raw : str
                        name of the output raw file
        """
        pass

    @abc.abstractmethod
    def load_dataset_movie(self, i, parameters, results):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads DNA-PAINT movie data and metadata into memory for subsequent
        analysis. Optionally creates sample movies and loads camera
        configuration. The data is saved in self.movie and self.info.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filename : str
                        Path to the movie file to load
                Optional keys:
                    sample_movie : dict
                        Parameters for creating a subsampled movie
                    load_camera_info : bool
                        Whether to load camera configuration from
                        picasso.CONFIG
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    picasso version : str
                        Version of picasso library used
                    movie.shape : tuple
                        Movie dimensions (frames, width, height)
                    sample_movie : dict
                        Results from subsampled movie creation (if requested)

        Returns:
            parameters : dict
                Input parameters, potentially modified (sample_movie paths
                updated)
            results : dict
                Input results with added movie information and metadata
        """
        pass

    @abc.abstractmethod
    def load_dataset_localizations(self, i, parameters, results):
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        The data is saved in
            self.locs
            self.info
        Args:
            parameters : dict
                necessary items:
                    filename : str
                        the (main) file name to load. This can be image files,
                        or hdf5.
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    picasso version : str
                        version of picasso library used
                    nlocs : int
                        number of localizations loaded
        """
        pass

    @abc.abstractmethod
    def identify(self, i, parameters, results):
        """Identifies localizations in a loaded dataset.

        Identifies potential localization sites in the loaded movie using
        net gradient thresholding. Optionally performs automatic net gradient
        detection and creates identification vs frame plots.
        The data is saved in self.identifications.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    box_size : int
                        Size of the detection box in pixels
                    min_gradient : float
                        Minimum net gradient threshold for detection
                        (required unless auto_netgrad is provided)
                Optional keys:
                    auto_netgrad : dict
                        Parameters for automatic net gradient detection:
                            box_size : int
                                Box size for auto detection
                            frame_numbers : list or int
                                Frame range for analysis
                            filename : str
                                Output filename for auto-detection plot
                            start_ng : float
                                Starting net gradient value
                            zscore : float
                                Z-score threshold for detection
                            bins : int
                                Number of histogram bins
                    ids_vs_frame : dict
                        Parameters for plotting identifications vs time:
                            filename : str
                                Output filename for plot
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    num_identifications : int
                        Total number of identifications found
                    auto_netgrad : dict
                        Results from automatic net gradient detection (if
                        requested)
                    ids_vs_frame : dict
                        Results from identifications vs frame analysis (if
                        requested)

        Returns:
            parameters : dict
                Input parameters, potentially with updated min_gradient
            results : dict
                Input results with identification statistics and optional plots
        """
        pass

    @abc.abstractmethod
    def localize(self):
        """Localizes Spots previously identified.
        The data is saved in
            self.locs
        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    box_size : as always
                    fit_parallel : bool
                        whether to fit on multiple cores
                optional items:
                    locs_vs_frame : dict
                        for plotting locs vs time
                        items correspond to arguments of _plot_locs_vs_frame
                    save_locs : dict
                        if saving localizations is requested.
                        Items correpsond to arguments of save_locs
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    locs_vs_frame : dict
                        plot results if locs_vs_frame parameter was provided
                    locs_columns : list
                        list of column names in the localizations array
        """
        pass

    @abc.abstractmethod
    def zfit(self):
        """
        Fits z coordinates to localized spots using an astigmatic calibration.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    calibration : str or dict
                        filepath to a calibration file or the calibration itself.
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
        """
        pass

    @abc.abstractmethod
    def load_picassoconfig(self):
        """
        Loads a specific picasso configuration file, as opposed to the default
        version residing in the picasso installation folder.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    fp_config : str
                        filepath to a config file.
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
        """
        pass

    @abc.abstractmethod
    def export_brightfield(self):
        """Opens a single-plane tiff image and saves it to png with
        contrast adjustment.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    filepath : str or list of str or dict
                        the tiff file(s) to load. The converted file(s) will
                        have the same name, but with .png extension
                        if dict: keys are labels
                optional items:
                    min_quantile : float, default: 0
                        the quantile below which pixels are shown black
                    max_quantile : float, default: 1
                        the quantile above which pixels are shown white
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    labeled filepaths : dict
                        keys : labels
                        values : filepaths
                    success : bool
                        whether the export was successful
        """
        pass

    @abc.abstractmethod
    def render(self):
        """Renders localizations on the whole field of view, and on
        a zoom in around the center of mass of localizations.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                optional items:
                    ctrmass_fov_nm : Field of view of the zoom in rendering
                        around the center of mass in nm
                    fullfov_pixelsize : The rendered pixel size [nm] of the
                        full FOV rendering
                    ctrmass_pixelsize : The rendered pixel size [nm] of the
                        zoom in rendering around the center of mass
                    ctrmass_blur_method : Blur method
                    ctrmass_min_blur_width : min blur with
                    ctrmass_ang : angle
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    fp_scene_fullfov : str
                        filepath to full FOV rendering
                    fp_scene_ctrmass : str
                        filepath to center of mass zoom rendering (conditional, only if ctrmass_fov_nm provided)
        """
        pass

    @abc.abstractmethod
    def undrift_rcc(self):
        """Undrifts localized data using redundant cross correlation.
        drift is saved in
        self.drift

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    segmentation : int
                        the number of frames segmented for RCC
                optional items:
                    max_iter_segmentations : int, default: 3
                        maximum number of iterations to adaptively increase segmentation if RCC fails
                    filename : str
                        the drift txt file name
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
                Note: dimensions parameter is set to ['x', 'y'] by this module
            results : dict
                the analysis results, updated with:
                    success : bool
                        whether undrifting was successful
                    message : str
                        error or warning messages if any
                    filepath_driftfile : str
                        filepath to drift txt file (conditional, only if undrifting succeeded)
                    filepath_plot : str
                        filepath to drift plot png (conditional, only if undrifting succeeded)
        """
        pass

    @abc.abstractmethod
    def undrift_rsso(self):
        """Undrift localized data using iterative RSSO-based drift correction

        This method applies an iterative RSSO (Redundant Spot Shift
        Overrepresentation) approach where each frame is compared against
        the whole dataset to compute total drift for that frame. The process
        is repeated iteratively with the undrifted dataset to improve accuracy.
        Includes uncertainty analysis, confidence evaluation, windowing and
        outlier detection.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    ton : float
                        Half-life of localization in frames (how long a spot
                        stays visible)
                    toff : float
                        Time in frames for a spot to reappear after
                        disappearing
                    max_shift : float
                        Maximum expected drift per frame in pixels
                optional items:
                    min_locs_per_frame : int
                        Minimum localizations per frame for reliable drift
                        estimation (default: 10)
                    max_iterations : int
                        Maximum number of iterative refinement rounds (default: 5)
                    convergence_threshold : float
                        RMS drift change threshold for convergence in nm (default: 0.1)
                    plot_drift : bool
                        Whether to save drift plots (default: True)
                    save_locs : bool
                        Whether to save undrifted localizations (default: True)
                    n_processes : int or None
                        Number of processes for parallel computation (default: auto)
                    confidence_threshold : float
                        Confidence threshold for windowing analysis (default: 0.8)
                    outlier_detection_enabled : bool
                        Enable RSSO failure and outlier detection (default: True)
                    outlier_z_threshold : float
                        Z-score threshold for temporal outlier detection (default: 3.5)
                    min_signal_to_noise : float
                        Minimum signal-to-noise ratio for drift measurements (default: 0.5)
                    windowing_enabled : bool
                        Enable adaptive windowing for low-confidence frames (default: True)
                    window_size_range : tuple
                        Min and max window sizes for adaptive windowing (default: (3, 20))

        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results including:
                    success : bool
                        whether drift correction succeeded
                    drift_x, drift_y : ndarray
                        total drift trajectories in nm for each frame
                    uncertainty_x, uncertainty_y : ndarray
                        uncertainty estimates for drift measurements
                    drift_quality : ndarray
                        quality/confidence metrics per frame
                    n_iterations : int
                        number of iterations performed
                    convergence_rms : float
                        final RMS change indicating convergence
                    drift_plots : str
                        path to drift visualization plots
        """

    @abc.abstractmethod
    def undrift_aim(self):
        """Unrift localized data using the AIM algorithm
        drift is saved in
        self.drift

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    segmentation : int
                        the number of frames segmented
                    intersect_d : float
                        Intersect distance in nanometers.
                    roi_r : float
                        Radius of the local search region in nanometers.
                        Should be larger than the maximum expected drift wihtin
                        segmentation.
                    dimensions : list of str
                        the dimensions undrifted, typically ['x', 'y'].
                optional items:
                    progress : callback function
                        progress callback for status updates
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    success : bool
                        whether undrifting was successful
                    fp_driftfile : str
                        filepath to drift txt file
                    fp_fig : str
                        filepath to drift plot png
        """
        pass

    @abc.abstractmethod
    def manual(self):
        """Handles a manual step: if the files required are not
        present, prompt the user to provide them. if they are, move
        to the next step.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    prompt : str
                        the user prompt
                    filename : str
                        the file the user should provide.
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def summarize_dataset(self):
        """Summarize dataset using various analysis methods

        Computes dataset quality metrics such as NeNa (Nearest Neighbor Analysis)
        and median localization precision.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    methods : dict
                        Dictionary of analysis methods to run. Keys are method names,
                        values are method-specific parameter dicts.
                        Supported methods:
                            "nena" : dict (no parameters)
                                Performs Nearest Neighbor Analysis to estimate localization precision
                            "median-loc-precision" : dict
                                Calculates median localization precision
                                Optional keys:
                                    qe_correction : float
                                        Quantum efficiency correction factor (default: 1)
            results : dict
                the results dict, created by the module_decorator
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results, updated with:
                    nena : dict (if nena method used)
                        Dictionary with keys:
                            res : str - all best fit values
                            NeNa : str - formatted NeNa result
                            nena-px : float - NeNa value in pixels
                            nena-nm : float - NeNa value in nanometers
                            filepath_plot : str - path to NeNa plot
                    median-loc-precision : dict (if median-loc-precision method used)
                        Dictionary with keys:
                            median_lp-px : float - median localization precision in pixels
                            median_lp-nm : float - median localization precision in nanometers
        """
        pass

    # @abc.abstractmethod
    # def aggregate_cluster(self):
    #     """Aggregate along the cluster column.
    #     Uses picasso.postprocess.cluster_combine"""
    #     pass

    @abc.abstractmethod
    def density(self):
        """Calculate local localization density
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    radius : float
                        the radius for calculating local density
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def dbscan(self, i, parameters, results):
        """Perform clustering using dbscan.

        Applies DBSCAN clustering algorithm to localizations, optionally
        replacing localizations with cluster centers for subsequent analysis.
        After this module, the standard locs will be the cluster centers.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        The DBSCAN radius parameter in nm
                    min_samples : int
                        Minimum number of samples required for a cluster
                    continue_with_centers : bool
                        Whether to replace localizations with cluster centers
                Optional keys:
                    save_locs : bool
                        Whether to save clustered localization data to results
                        folder
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    fp_fig_clustersizes : str
                        Filepath to cluster size distribution figure
                    fp_centers : str
                        Filepath to cluster centers file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with clustering outputs and file paths
        """
        pass

    @abc.abstractmethod
    def hdbscan(self):
        """Perform hdbscan clustering. After this module, the standard
        locs will be the cluster centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    min_cluster : float
                        the hdbscan min_cluster
                    min_samples : float
                        the hdbscan min_sample
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def binding_event_analysis(self):
        """Evaluate binding events according to Philipp Steen's methods

        Steen, P.R., Unterauer, E.M., Masullo, L.A. et al.
        The DNA-PAINT palette: a comprehensive performance analysis
        of fluorescent dyes.
        Nat Methods (2024).
        https://doi.org/10.1038/s41592-024-02374-8

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_locs : str
                        file path to input locs
                    n_frames
        """
        pass

    @abc.abstractmethod
    def resolution_analysis(self):
        """Perform resolution analysis using point pattern autocorrelation

        This method calculates the spatial resolution of localizations
        by computing a 2D autocorrelation function and fitting a Gaussian to
        extract resolution metrics. The analysis includes 2D Gaussian fitting,
        radial profile computation, and 1D Gaussian fitting to the radial profile.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                with optional keys:
                    delta_r : float
                        grid spacing for autocorrelation (default: 5 nm)
                    r_max : float
                        maximum radius for autocorrelation (default: 100 nm)
                    batch_size : int or None
                        number of data points per batch for chunking (auto-calculated if None)
                    n_processes : int or None
                        number of parallel processes (auto-detected if None, capped at 4)
                    use_chunking : bool
                        enable memory-efficient chunking for large datasets (default: True)
                    use_sparse : bool
                        use sparse matrices for very large grids (default: False)

        Results:
            resolution : float
                average resolution in nm (FWHM)
            sigma_x, sigma_y : float
                Gaussian standard deviations in x,y directions
            fwhm_x, fwhm_y : float
                Full-width half-maximum in x,y directions
            fit_quality : float
                R-squared goodness of fit
            autocorr_map : ndarray
                2D autocorrelation intensity map
            radial_profile : ndarray
                radial profile of autocorrelation
            radial_distances : ndarray
                distance values for radial profile
            resolution_radial : float
                resolution from radial Gaussian fit (FWHM)
            resolution_dblradial : float
                resolution from double Gaussian fit (FWHM)
            fig_resolution : str
                path to resolution plot
            fig_radial : str
                path to radial profile plot
        """

    @abc.abstractmethod
    def resolution_frc_spatial(self):
        """Calculate resolution using spatial FRC approach

        This method divides the FOV into spatial regions, computes FRC for each
        region independently, and averages the results. Benefits:
        - Lower memory usage (smaller images per region)
        - Better statistics through spatial averaging
        - Efficient multiprocessing (fully independent regions)
        - Preserves high spatial frequencies

        Args:
            i : int
                the index of the module
            parameters: dict
                with optional keys:
                    pixelsize_render : float
                        pixel size for rendered images in nm (default: 5 nm)
                    smoothing_sigma : float or None
                        Gaussian smoothing sigma in pixels (default: None)
                    threshold : float
                        FRC threshold for resolution cutoff (default: 1/7 ≈ 0.143)
                    region_size : float
                        size of each spatial region in micrometers (default: 10.0 µm)
                    min_locs_per_region : int
                        minimum localizations per region to process (default: 500)
                    max_frc_range_nm : float or None
                        maximum FRC range in nm (default: None = full range)
                    n_processes : int
                        number of parallel processes (default: 4)
                    smoothing_window : float
                        moving average window size for FRC smoothing in 1/nm
                        (default: 0.005)

        Results:
            resolution_frc_spatial : float
                mean FRC-based resolution in nm
            resolution_std : float
                standard deviation across regions
            n_regions : int
                number of valid regions processed
            cutoff_frequency : float
                mean spatial frequency at resolution cutoff (1/nm)
            frc_curve_mean : ndarray
                mean FRC curve across regions
            frc_curve_std : ndarray
                std of FRC curves
            spatial_frequencies : ndarray
                spatial frequency values (1/nm)
            threshold : float
                threshold used
            fig_frc : str
                path to FRC curve plot
        """

    @abc.abstractmethod
    def smlm_clusterer(self):
        """Perform smlm clustering. After this module, the standard
        locs will be the cluster centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    radius : float
                        the smlm radius, in nm
                    min_locs : float
                        the smlm min_locs
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
                    basic_fa : bool
                        the smlm basic fa, default: False
                    radius_z : float
                        the smlm radius_z, default: None
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def gaussian_mixture_cluster(self):
        """Perform clustering using gaussian mixture modelsAfter this module,
        the standard locs will be the Gaussian centers.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    locs : np.recarray
                        Localizations.
                    info : list
                        Information dictionaries.
                    min_locs : int
                        Minimum number of localizations per component. Used
                        to filter out components with too few localizations
                        that likely represent background.
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
                    max_rounds_without_best_bic : int
                        (default=3)
                        Maximum number of rounds without BIC improvement to
                        terminate the optimal GMM search.
                    bootstrap_check : bool (default=False)
                        If True, the standard error of the means (SEM) is
                        calculated using bootstrapping. If False, the
                        standard, single Gaussian SEM is used as
                        approximation.
                    calibration : dict (default=None)
                        Calibration dictionary with x and y coefficients, z
                        step size and the number of frames. Only required for
                        3D data.
                    asynch : bool (default=True)
                        If True, the GMM search is run in parallel using
                        multiprocessing. If False, the GMM search is run
                        without multiprocessing.
                    callback_parent : function (default='silent')
                        Callback function's parent object for displaying
                        progress bar. If None, the progress bar displayed
                        directly to the console. If 'silent', no progress
                        is displayed
                    sigma_bounds : float (not recommended)
                        Minimum standard deviation of the Gaussian components
                        in nanometers. Useful for avoiding overfitting within
                        a single localization cloud. Now using individual
                        loc precision, so min_sigma is not recommended.
                    loc_prec_handle : Literal["local", "global", "abs"]
                        default: local
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def nneighbor(self):
        """Perform nearest neighbor calculation
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    dims : list of str
                        the distance dimensions, e.g. ['x', 'y']
                        or ['x', 'y', 'z']
                    nth_NN : int
                        calculate the 1st to nth nearest neighbor distances
                    nth_rdf : int
                        calculate distances up to the 95th percile of the
                        nth_rdf nearest neighbor
                    subsample_1stNN : int
                        by how much fold to subsample distances from the
                        median of the 1st nearest nteighbor. Default is 20
                    add_column : bool
                        whether to add a column of nearest neighbor distance
                        to the locs
                and optional keys:
                    save_locs : bool
                        whether to save the locs into the results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def fit_csr(self, i, parameters, results):
        """Fit a Completely Spatially Random Distribution to nearest neighbors.

        Fits CSR model to nearest neighbor distance distributions and evaluates
        goodness-of-fit using statistical measures and visualization.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    nneighbors : str or numpy.ndarray or list
                        If str: filepath to nearest neighbor data file
                        If array: 2D array (N, k) of kth nearest neighbor
                        distances
                        If list: multiple datasets or file paths
                    dimensionality : int
                        Spatial dimensionality (2 or 3) for CSR model
                Optional keys:
                    kmin : int
                        Minimum k-th nearest neighbor order to fit (default: 1)
                    min_dist : float
                        Minimum observable distance in nm due to technical
                        limits
                    max_dist : float
                        Maximum distance for filtering analysis
                    bkg_fraction : float
                        Background fraction for fitting
                    fit_bkg : bool
                        Whether to fit background (default: False)
            results : dict
                Automatic keys (provided by decorator):
                    start time : str
                        Module execution start timestamp
                    end time : str
                        Module execution end timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for module results
                    folder : str
                        Output folder for generated files
                Results updated with:
                    density : float or list
                        Fitted spatial density value(s) in units^(-d)
                    bkg_fraction : list
                        Background fraction values
                    fp_fig : str or list
                        Filepath(s) to CSR fit visualization figure(s)
                    wasserstein_distances_per_k : list
                        Wasserstein distances for each k-th nearest neighbor
                        order
                    mean_wasserstein_distance : float or list
                        Mean Wasserstein distance across all k orders
                    ks_pvalues_per_k : list
                        Kolmogorov-Smirnov p-values for each k-th NN order

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with CSR fitting results and goodness-of-fit
                metrics
        """
        pass

    # @abs.abstractmethod
    # def radial_distribution_function(self):
    #     """Generate the Radial Distribution Function,
    #     Whis is the sum of nearest neighbors with geometry factor.
    #     At long radii, its value is the overall density.
    #     """
    # pass

    @abc.abstractmethod
    def save_single_dataset(self):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                        filename : str
                            the name of the dataset
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    @abc.abstractmethod
    def load_datasets_to_aggregate(self):
        """Loads the results of single-dataset workflows
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    filepaths : list of str
                        the hdf5 files to load.
                    tags : list of str
                        the tags to name the datasets
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def align_channels(self):
        """Aligns multiple channels to each other (part of an aggregation
        workflow)
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    filepaths : list of str
                        the previously saved hdf5 files to be loaded and
                        aligned. if not given, the last processed data is used
                    align_pars : dict
                        kwargs of picasso_outpost.align_channels
                            max_iterations, convergence
                    fp_fiducials : list of str
                        the previously saved hdf5 files of fiducial markers
                        to be loaded and aligned.
                    fig_filename : str
                        the location to save the drift figure to
                    crop_boundaries : bool
                        whether to crop the localizations according to the
                        image boundaries (after shifting)
                    fp_co_shift_channel_locs : list of str
                        hdf5 files not in the 'main workflow' that should
                        be shifted as well. This could e.g. be clustered
                        localizations when the workflow has continued with
                        cluster centers
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def combine_channels(self):
        """Combines multiple channels into one dataset. This is relevant
        e.g. for RESI.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    tag : str
                        the tag / name of the combined dataset
                    combine_col : str
                        the column name for the IDs to the different datasets
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def save_datasets_aggregated(self):
        """Save data of multiple single-dataset workflows from one
        aggregation workflow.

        Saves all channel localization data and metadata from the aggregated
        workflow to individual files in the results folder.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                The results dictionary, updated with:
                    filepaths : list
                        List of all saved file paths from the aggregated
                        datasets

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Updated results dictionary with saved file paths
        """
        pass

    # @abc.abstractmethod
    # def spinna_manual(self):
    #     """Direct implementation of spinna batch analysis.
    #     The current locs file(s) are saved into the results folder, and
    #     a template csv file is created. This csv needs to be filled out by the
    #     user in a manual step before the spinna analysis is carried out.

    #     Args:
    #         i : int
    #             the index of the module
    #         parameters: dict
    #             with required keys:
    #                 proposed_labeling_efficiency : float, range 0-100
    #                     labeling efficiency percentage, default for all targets
    #                     used proposed value in spinna_config.csv and can be
    #                     altered manually after the first run of this module
    #                 proposed_labeling_uncertainty : float
    #                     labeling uncertainty [nm]; good value is e.g. 5
    #                     used proposed value in spinna_config.csv and can be
    #                      alteredmanually after the first run of this module
    #                 proposed_n_simulate : int
    #                     number of target molecules to simulated;
    #                     good value is e.g. 50000
    #                     used proposed value in spinna_config.csv and can be
    #                     altered manually after the first run of this module
    #                 proposed_density : int
    #                     density to simulate;
    #                     area density if 2D; volume density if 3D
    #                     used proposed value in spinna_config.csv and can be
    #                     altered manually after the first run of this module
    #                 proposed_nn_plotted : int
    #                     number of nearest neighbors to plot
    #                     used proposed value in spinna_config.csv and can be
    #                      alteredmanually after the first run of this module
    #             and optional keys:
    #                 structures : list of dict
    #                     SPINNA structures. Each structure dict has
    #                         "Molecular targets": list of str,
    #                         "Structure title": str,
    #                         "TARGET_x": list of float,
    #                         "TARGET_y": list of float,
    #                         "TARGET_z": list of float,
    #                     where TARGET is one each of the target names in
    #                     "Molecular targets"
    #                 structures_d : float
    #                     distance between molecules within auto-generated
    #                     structures, in nm. Only necessary if 'structures'
    #                     is not given.
    #         results : dict
    #             the results this function generates. This is created
    #             in the decorator wrapper
    #     """
    #     pass

    @abc.abstractmethod
    def spinna(self):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    labeling_efficiency : dict of float, range 0-1
                        labeling efficiency, for all targets
                    labeling_uncertainty : float or dict of floats
                        labeling uncertainty [nm]; good value is e.g. 5
                        assumed the same value for all targets
                    n_simulate : int
                        number of target molecules to simulated;
                        good value is e.g. 50000
                    structures : str or list of dict
                        if str: filepath to a yaml file with the structures.
                        if list of dict:
                        SPINNA structures. Each structure dict has
                            "Molecular targets": list of str,
                            "Structure title": str,
                            "TARGET_x": list of float,
                            "TARGET_y": list of float,
                            "TARGET_z": list of float,
                        where TARGET is one each of the target names in
                        "Molecular targets"
                    fp_mask_dict : str
                        the filepath to the mask_dict file
                    density : list of float
                        density to simulate in 1/nm^d;
                        area density if 2D; volume density if 3D
                        (required: either density or density_app)
                    random_rot_mode : '2D', or '3D'
                        Mode of molecule rotation in simulation
                    sim_repeats : int
                        number of simulation repeats
                    fit_NND_bin : float
                        bin size of fits
                    fit_NND_maxdist : float
                        max of histogram
                    n_nearest_neighbors : int
                        number of nearest neighbors to evaluate
                    granularity : float
                    the spinna granularity
                optional keys:
                    density_app : list of float
                        apparent density in 1/nm^2;
                        this is the product of 'real' density & lbl efficiency
        """
        pass

    @abc.abstractmethod
    def spinna(self):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_spinna_batch_config : str
                        path to the spinna batch analysis config file.
                and optional keys:
        """
        pass

    @abc.abstractmethod
    def spinna_batch(self):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_spinna_batch_config : str
                        path to the spinna batch analysis config file.
                and optional keys:
        """
        pass

    @abc.abstractmethod
    def ripleysk(self):
        pass

    # @abc.abstractmethod
    # def ripleysk_rafal(self):
    #     pass

    @abc.abstractmethod
    def ripleysk2(self):
        pass

    @abc.abstractmethod
    def ripleysk_average(self):
        pass

    @abc.abstractmethod
    def ripleysk_average2(self):
        pass

    @abc.abstractmethod
    def protein_interactions(self):
        pass

    @abc.abstractmethod
    def protein_interactions_average(self):
        pass

    @abc.abstractmethod
    def create_mask(self):
        """
        This is Susanne's implementation of calculating a cell mask,
        written (ni part?) for the initial version of the DC-Atlas.
        May be obsolete with create_mask2, but kept for backwards
        compatibility. To be deprecated on the long run.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    fp_combined_locs : str
                        filepath to the locs combined in 'combine_channels'
                        module
                    margin : float
                        Size of the added empty margin to the FOV, in nm
                    binsize : float
                        Size o fthe 2D histogram bins of the first step, in nm
                    sigma_mask_blur : int
                        parameter of the gaussian blur in binsize units
                    mask_resolution : float
                        Controls the digital resolution of the mask, in nm
                    combine_col : str
                        the name of the combine column, e.g. 'combine_id'
                        or 'protein'. Same as used in 'combine_channels' module
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def create_mask2(self):
        """
        This is Rafal's implementation of cell masking, written for the
        3rd version of the DC Atlas. It is (mostly?) identical with an
        implementation of it in spinna, which will be integrated into
        picasso soon. Evaluate deprecation (or moving source from
        outpost_modules/ripleys to picasso/spinna) at that time.

        the locs must be protein positions at this stage.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    binsize : float
                        the bin size in nanometers. A good value is 20
                    blursize : float
                        the gaussian blur to apply in nanometers.
                        A good value is 400
                    mask_pixel_size : float
                        the pixelsize of the final mask, in nanometers.
                        Often used: 10
                    threshold : float
                        the threshold value below which the mask is set
                        to zero. For example 1 / 3
                    binary : boolean
                        whether to create a binary or density mask
                    select_cell : boolean
                        whether to select the largest connected component,
                        assumed to be the cell of interest.
                    fill_holes : boolean
                        whether to fill holes in the cell mask
                    dilate_nm : float
                        the nanometers to dilate the mask (useful if a large
                        threshold has been used)
                    apply_to_locs : boolean
                        whether to drop all localizations outside the area
                and optional keys:
                    fp_combined_locs : str default: None or ''
                        filepath to the locs combined in 'combine_channels'
                        module. If None or '', loaded channel_locs is used
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    combine_col : str
                        the name of the combine column, e.g. 'combine_id'
                        or 'protein'. Same as used in 'combine_channels' module
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def refine_mask_by_density(self):
        """
        This module analyses and refines a previously created mask.
        Particularly, the density histogram of the mask bins are plotted,
        and an area of homogeneous density can be selected

        the locs must be protein positions at this stage.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_mask : str
                        the file path to the mask
                    min_density, max_density : float
                        the density range to select
                and optional keys:
                    nbins : int
                        the number of bins for plotting
                    nth_largest : int
                        select the nth largest area in density range.
                        set 0 for largest.
                    apply_to_locs : bool
                        whether to apply the created mask to the locs
                    smoothe_nm : float
                        the number of nanometers to dilate and erode
                        the mask. This can be useful to remove excessive
                        holes and ragging in the mask due to the
                        density thresholding
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def dbscan_molint(self):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    epsilon_nm : float
                        dbscan epsilon in nm
                    minpts : int
                        minimum number of points
                    sigma_linker : float
                        ... in nm
                    fp_merge_mask : str
                        filepath to the merge mask (generated in module
                        'create_mask')
                    thresh_type : str
                        ...
                    cell_name : str
                        the name of the cell currently analyzed
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def CSR_sim_in_mask(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask, and perform dbscan as well
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_channel_map : str
                        filepath to the map from 'combine_channels' module,
                        which is a dict from channel name to ID int in the
                        locs['combine_id']
                    fp_mask_dict : str
                        filepath to the mask_dict.pkl file generated in
                        the 'create_mask' module
                    N_repeats : int
                        number of simulation repeats
                    epsilon_nm : float
                        dbscan epsilon in nm
                    minpts : int
                        minimum number of points
                    sigma_linker : float
                        ... in nm
                    fp_merge_mask : str
                        filepath to the merge mask (generated in module
                        'create_mask')
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def find_cluster_motifs(self):
        """Analyses the binary barcode results of _do_dbscan_molint.
        Compares experimental to CSR data.
        Merged for multiple cells
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_workflows : list of str
                        the paths to the folders of separate workflows
                        where the separate ripleys analyses have been done
                    report_names : list of str
                        the report names of those worklfows
                    swkfl_dbscan_molint_key : str
                        the results key of the dbscan module.
                        e.g. '09_dbscan_molint'
                    swkfl_CSR_sim_in_mask_key : str
                        the results key of the CSR dbscan module.
                        e.g. '10_CSR_sim_in_mask'
                    population_threshold : float, 0 - 1
                        only select barcodes with a relative population
                        larger than this
                    ttest_pvalue_max : float, < 0
                        the pvalue below which the difference between number
                        of clusters found for a barcode between exp and csr
                        is deemed significant
                    channel_colors : list of str
                        colors to describe the receptors with
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def interaction_graph(self):
        """Plot the interaction graph, displaying the different targets
        and their interactions in a graph. The node sizes denote the
        density, and the ripley interaction matrix is represented in the
        edges.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_workflows : list of str
                        the paths to the folders of separate workflows
                        where the separate ripleys analyses have been done
                    report_names : list of str
                        the report names of those worklfows
                    swkfl_protint_key : str
                        the results key of the protein_interactions module.
                        e.g. '09_protein_interactions'
                    fp_density : str
                        fp to the denfsities of the channels.
                    fp_ripleys_meanvals : str
                        the filepath to the interaction matrix
                    edge_factor : float
                        factor to display useful sizes
                    node_factor : float
                        factor to display useful sizes
                    channel_colors : list of str
                        colors to describe the receptors with
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def plot_densities(self):
        """Aggregate densities and cell areas of multiple datasets and
        plot them
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_workflows : list of str
                        the paths to the folders of separate workflows
                        where the separate ripleys analyses have been done
                    report_names : list of str
                        the report names of those worklfows
                    swkfl_create_mask_key : str
                        the results key of the dbscan module.
                        e.g. '11_create_mask'
                    swkfl_protint_key : str
                        the results key of the protein_interactions module.
                        e.g. '09_protein_interactions'
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def find_gold(self):
        """Find localizations stemming from gold beads based on blinking
        kinetics.
        The metrics used are number of locs and rms deviation from mean
        frame
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    remove_gold : bool
                        if present and set to True, the gold locs
                        are discarded and self.locs is set to the
                        nongold-locs
                    diameter : float
                        the pick similar diameter for identifying gold
                    std_range, mean_rmsd : float
                        the pick similar parameters identifying gold
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def find_similar(self):
        """pick similar in nlocs/rmsd space (with specified limits in
        that space).
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    diameter : float
                        the pick similar diameter for identifying gold
                and optional keys:
                    min_n_locs_per_frame : float, range 0-1
                        the min percentage of frames with events in the pick
                        region to pick. default: 0.01
                    max_n_locs_per_frame : float, range 0-1
                        the max percentage of frames with events in the pick
                        region to pick. default: 0.01
                    min_rmsd : float
                        the minimum root mean square distance from pick center
                        to pick
                    max_rmsd : float
                        the maximum root mean square distance from pick center
                        to pick
                    n_plot_structures : int
                        the number of structures to plot
                    display_pixelsize : float
                        the pixelsize for display in nm, default: 1
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def find_structures(self):
        """pick similar on clusters in nlocs/rmsd space.
        This may be useful for automated picking of origamis, and may
        help for defining parameters for finding gold
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    diameter : float
                        the pick similar diameter for identifying gold
                and optional keys:
                    min_n_locs_per_frame : float
                        the percentage of frames with events in the pick
                        region below which there is noise. default: 0.01
                    n_plot_structures : int
                        the number of structures to plot
                    display_pixelsize : float
                        the pixelsize for display in nm, default: 1
                    xi : float
                        the xi parameter for clustering. default 0.05
                    min_cluster_size : float
                        the minimun cluster size (fract). default .05
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def undrift_from_picked(self):
        """Performs undrift from piced locs.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    fp_picked_locs : str
                        filepath to the picked locs to undrift from
                        (.hdf5 file of list of locs, with 'group' column
                         to describe picks)
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def filter_locs(self):
        """Filter localizations to lie within a min-max range of a metric.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    field : str or list of str
                        the field(s) to filter on
                and optional keys:
                    minval : dtype of field (or list of it)
                        the minimum value(s) to accept
                    maxval : dtype of field (or list of it)
                        the maximum value(s) to accept
                    mode : str
                        the mode of threshold application:
                         - absolute: minval and maxval are values
                            in units of the field
                         - zscore: minval and maxval are in units of
                            standard deviations from the mean
                            (-2, 2 means cut off at 2*std from mean)
                         - quantile: minval and maxval are quantiles
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def filter_transient_binding(self, i, parameters, results):
        """Filter molecule positions (after clustering or Gaussian Mixture)
        for those who show transient binding. Specifically, the mean frame
        should not be at extreme positions
        (default, 0.1 > mean frame / nframes > 0.9), and std of frames
        (default: 0.3 > std frame).
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                and optional keys:
                    meanframe_cutoff : float (0-1, default .1)
                        filter out positions at more extreme temporal positions
                    stdframe_cutoff : float
                        filter out positions with lower std than .16
                    fp_locs : str
                        the filepath to the underlying localizations
                        (self.locs are centers). If given, these are filtered
                        as well and saved with the same filename in the current
                        results folder
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def link_locs(self):
        """Link localizations.
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    d_max : int
                        maximum distance to link [px]
                    tolerance : int
                        maximum transient dark time [frames]
                and optional keys:
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def pairwise_module_executor(self):
        """Calls another module (as a sub-module) for all pairs in the
        channel_locs
        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    module_name : str
                        the module to call
                    param_target1 : str
                        parameter name of the first target to set for the
                        module
                    param_target2 : str
                        parameter name of the second target to set for the
                        module
                    module_kwargs : dict
                        the other arguments to the module
                and optional keys:
                    result_scalar : str
                        the key to display in a heatmap as main result
                    scalar_threshold : float
                        the saturation value in the heatmap
                    scalar_minval : float
                        the minimum value for color in the heatmap
                    result_fpfig : str or list of str
                        the key to the filepath of one or more figures
                        generated to display for documentation
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass

    @abc.abstractmethod
    def random_val(self):
        """Generate random values and plot for debugging and testing the
        pairwise module.

        Creates a random value and generates a test plot with random data
        for debugging purposes in pairwise module workflows.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    xlabel : str
                        Label for the x-axis of the test plot
                    ylabel : str
                        Label for the y-axis of the test plot
                Optional keys: (none)
            results : dict
                The results dictionary, updated with:
                    random_val : float
                        A random value between 0 and 1
                    fp_fig : str
                        Filepath to the generated test figure

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Updated results dictionary with random value and figure path
        """
        pass

    @abc.abstractmethod
    def labeling_efficiency_analysis(self):
        """Analyse for labeling efficiency.
        Perform 3 component SPINNA analysis for monomers and heterodimers
        of target (A) and reference (B). For the analysis, we enter a
        labeling efficiency of 1, yielding proportions of monomers and
        dimers as seen in the data. The real labeling efficiency is then

        Model:
        Binders A and B bind to an engineered construct A*-anchor-B*.
            A <-> A*-anchor-B* <-> B
        There are four possible configurations:
            A_only: AA*-anchor-B*
            AB: AA*-anchor-B*B
            B_only: A*-anchor-B*B
            None (invisible in data): A*-anchor-B*
        Number of total constructs with A, or B, respectively:
            #A_tot = #A_only + #AB
            #B_tot = #B_only + #AB

        Proportions can be given in terms of #structures, or in terms
        of #molecules, e.g.
        with proportions given in terms of #structures
         10 monomers, 10 dimers (20molecules in dimers) -> p_m = 50%, p_d=50%

        with proportions given in terms of #molecules
         10 monomers, 10 dimers (20molecules in dimers) -> p_m = 33%, p_d=66%

        in terms of #structures
        prop_A^S = #A_only / (#A_only + #B_only + #AB)
        prop_B^S = #B_only / (#A_only + #B_only + #AB)
        prop_AB^S = #AB / (#A_only + #B_only + #AB)
        in terms of #molecules
        prop_A^S = #A_only / (#A_only + #B_only + 2 #AB)
        prop_B^S = #B_only / (#A_only + #B_only + 2 #AB)
        prop_AB^S = 2 #AB / (#A_only + #B_only + 2 #AB)

        #AB = #anchor * LE_A * LE_B
        #A_tot = #anchor * LE_A
        #B_tot = #anchor * LE_B
        #A_only = #A_tot - #AB = #anchor * LE_A * (1 - LE_B)
        #B_only = #B_tot - #AB = #anchor * LE_B * (1 - LE_A)

        THUS, finally, the labeling efficiency can be calculated by

        with proportions given in terms of #structures
        LE_A = prop(AB) / (prop(B) + prop(AB))
        LE_B = prop(AB) / (prop(A) + prop(AB))

        with proportions given in terms of #molecules
        LE_A = prop(AB) / (2 * prop(B) + prop(AB))
        LE_B = prop(AB) / (2 * prop(A) + prop(AB))

        SPINNA outputs propportions in terms of #molecules, so the last
        formulae are used below.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    reference_name : str
                        the channgel_tag of the reference
                    target_name : str
                        the channel_tag of the target queried for LE
                    pair_distance: 10 # real distance of pair of tags in nm
                    labeling_uncertainty : dict, channel tag to float
                        labeling uncertainty [nm]; good value is e.g. 5
                    n_simulate : int
                        number of target molecules to be simulated;
                        good value is e.g. 50000
                    density : dict, channel tag to float
                        density to simulate [nm^2 or nm^3];
                        area density if 2D; volume density if 3D
                    granularity : float
                        the spinna res_factor
                    sim_repeats : int
                        number of simulation repeats, for noise reduction
                and optional keys:
                    nn_nth : int
                        number of nearest neighbors to analyse
                        default: 1
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        pass


class DictSimpleTyper:
    """Scans a complex dictionary and converts numpy arrays and
    tuples to lists"""

    def __init__(self, to_simple_type=True):
        """
        Args:
            to_simple_type : bool
                converts numpy arrays and tuples to lists, numpy scalars to
                python scalars
        """
        self.to_simple_type = to_simple_type
        self.curr_rootidx = 0

    def run(self, parameters):
        """Scan a parameter set for commands to execute prior to module
        execution.
        commands: '$get_prior_result'
        Args:
            parameters : dict
                the parameters for a module
        """
        logger.debug("Running DictSimpleTyper")
        return self.scan(parameters)

    def scan(self, itrbl, root_level=False):
        """Scan a level in a dict.
        Args:
            itrbl : usually an iterable
                the value to scan
            root_level : bool
                whether the value is in root level.
                If it is, its index will be stored.
        """
        if isinstance(itrbl, dict):
            res = self.scan_dict(itrbl)
        elif isinstance(itrbl, list):
            res = self.scan_list(itrbl, root_level)
        elif isinstance(itrbl, tuple):
            res = self.scan_tuple(itrbl)
        elif isinstance(itrbl, np.ndarray):
            res = self.scan_ndarray(itrbl)
        # elif isinstance(itrbl, np.core.multiarray):
        #     res = self.scan_ndarray(itrbl)
        elif isinstance(itrbl, np.generic):
            if self.to_simple_type:
                res = float(itrbl)
        else:
            res = itrbl
        return res

    def scan_dict(self, d):
        addl_items = {}
        del_keys = []
        for k, v in d.items():
            result = self.scan(v)
            # not elegant, but should work: In case this is a
            # ParameterCommandExecutor, keep the original commands
            # commands are always tuples and reside in dicts
            # and return dicts with 'original' and 'parsed' keys
            if (
                isinstance(v, tuple)
                and isinstance(result, dict)
                and "parsed" in result.keys()
                and "original" in result.keys()
            ):
                d[k] = result["parsed"]
                addl_items[f"{k}_originalnocmd"] = result["original"]
            else:
                d[k] = result

            # also keys may be commands
            if isinstance(k, tuple):
                k_result = self.scan(k)
                if isinstance(k_result, dict):
                    addl_items[k_result["parsed"]] = d[k]
                    addl_items[f"{k}_originalnocmd"] = d[k]
                    del_keys.append(k)
        for k, v in addl_items.items():
            d[k] = v
        for k in del_keys:
            del d[k]
        return d

    def scan_list(self, li, root_level=False):
        for i, it in enumerate(li):
            if root_level:
                self.curr_rootidx = i
            li[i] = self.scan(it)
        return li

    def scan_ndarray(self, itrbl):
        if self.to_simple_type:
            return itrbl.tolist()
        else:
            return itrbl

    def scan_tuple(self, t):
        # it's just a normal tuple
        tout = []
        for i, it in enumerate(t):
            # logger.debug(f"{i}: {it}")
            tout.append(self.scan(it))
        return tuple(tout)
        # if self.to_simple_type:
        #     return tout
        # else:
        #     return tuple(tout)


class ParameterCommandExecutor(DictSimpleTyper):
    """Scans parameter sets for commands and executes them.
    This is useful e.g. in the picasso-workflow.workflow.WorkflowRunner
    where some parameters of later modules depend on results of previous
    modules. These can be retrieved with this ParameterCommandExecutor.
    """

    def __init__(
        self,
        parent_object=None,
        map_dict={},
        to_simple_type=False,
        command_sign="$",
    ):
        """
        Args:
            parent_object : object
                the object to execute the command on.
                e.g. the WorkflowRunner itself
            map_dict : dict
                a dictionary to map values using the $map command
            to_simple_type : bool
                converts numpy arrays and tuples to lists, numpy scalars to
                python scalars
            command_sign : str
                the command sign to execute on. In aggregation workflow
                preparation (Using the ParameterTiler), the single-workflow
                commands should not be executed, therefore different
                signs are used.
        """
        super().__init__(to_simple_type)
        self.parent_object = parent_object
        self.map = map_dict
        self.command_sign = command_sign

    def run(self, parameters, curr_rootidx=None):
        """Scan a parameter set for commands to execute prior to module
        execution.
        commands: '$get_prior_result'
        Args:
            parameters : dict
                the parameters for a module
            curr_rootidx : int or None
                if int, this is the current module index
        """
        logger.debug("Running ParameterCommandExecutor")
        if curr_rootidx is not None:
            self.curr_rootidx = curr_rootidx
        return self.scan(parameters, root_level=True)

    def scan_tuple(self, t):
        """Firstly, this scans normal tuples. Secondly, tuples of len 2
        can be commands, e.g.
            $get_prior_result
                retreive a result of a prior module, e.g.
                ("$get_prior_result", "results, 04_manual, filepath")
            $get_previous_module_result
                retreive a result of the module directly before the current one
                ("$get_previous_module_result",
                 "sample_movie, sample_frame_idx")
            $map
                use self.map dictionary to map values, e.g.
                ("$$map", "filepath")
            $sum
                sum up values, e.g.
                ("$sum", ($get_prior_result, ...), ($get_prior_result, ...))
                or
                ("$sum", ("$$get_prior_result, all_results,
                          single_dataset, $$all, 03_nneighbor, density_rdf")
            $product
                analog to sum
            $min, $max
                analog to sum
        The commands can be combined with numeric operations:
            ('$get_previous_module_result *2', 'nena')
            The arithmetic expression must not contain any spaces.
        """
        if (
            len(t) > 1
            and isinstance(t[0], str)
            and t[0][: len(self.command_sign)] == self.command_sign
        ):
            originals = []
            # this is a parameter command
            if " " in t[0]:
                cmd = t[0].split(" ")[0]
                aritexp = t[0].split(" ")[1]
            else:
                cmd = t[0]
                aritexp = None
            if cmd == f"{self.command_sign}get_prior_result":
                logger.debug(f"Getting prior result from {t[1]}.")
                res = self.get_prior_result(t[1])
                logger.debug(f"Prior result is {res}.")
            elif cmd == f"{self.command_sign}get_previous_module_result":
                logger.debug(f"Getting previous module result {t[1]}.")
                res = self.get_previous_module_result(t[1])
                logger.debug(f"Previous module result is {res}.")
            elif cmd == f"{self.command_sign}map":
                res = self.map[t[1]]
                logger.debug(f"Mapping {t[1]}: {res}")
            elif cmd == f"{self.command_sign}index":
                idx = int(aritexp)
                res = self.map[t[1]][idx]
                logger.debug(f"Indexing map {t[1]}[{idx}]: {res}")
                aritexp = None  # avoid arithmetic expression below
            elif cmd == f"{self.command_sign}sum":
                logger.debug(f"summing up {t[1:]}.")
                components = []
                for arg in t[1:]:
                    if isinstance(arg, dict) and "parsed" in arg.keys():
                        components.append(arg["parsed"])
                    elif (
                        isinstance(arg, (tuple, list))
                        and len(arg) > 1
                        and isinstance(arg[0], str)
                        and arg[0][: len(self.command_sign)]
                        == self.command_sign
                    ):
                        sub_res = self.scan_tuple(arg)
                        components.append(sub_res["parsed"])
                        originals.append(sub_res["original"])
                    else:
                        components.append(arg)
                logger.debug(f"summing up {components}.")
                res = self.sum(components)
                logger.debug(f"Sum result is {res}.")
            elif cmd == f"{self.command_sign}product":
                logger.debug(f"multiplying {t[1:]}.")
                components = []
                for arg in t[1:]:
                    if isinstance(arg, dict) and "parsed" in arg.keys():
                        components.append(arg["parsed"])
                    elif (
                        isinstance(arg, (tuple, list))
                        and len(arg) > 1
                        and isinstance(arg[0], str)
                        and arg[0][: len(self.command_sign)]
                        == self.command_sign
                    ):
                        sub_res = self.scan_tuple(arg)
                        components.append(sub_res["parsed"])
                        originals.append(sub_res["original"])
                    else:
                        components.append(arg)
                logger.debug(f"multiplying {components}.")
                res = self.product(components)
                logger.debug(f"Product result is {res}.")
            elif cmd == f"{self.command_sign}min":
                logger.debug(f"min of {t[1:]}.")
                components = []
                for arg in t[1:]:
                    if isinstance(arg, dict) and "parsed" in arg.keys():
                        components.append(arg["parsed"])
                    elif (
                        isinstance(arg, (tuple, list))
                        and len(arg) > 1
                        and isinstance(arg[0], str)
                        and arg[0][: len(self.command_sign)]
                        == self.command_sign
                    ):
                        sub_res = self.scan_tuple(arg)
                        components.append(sub_res["parsed"])
                        originals.append(sub_res["original"])
                    else:
                        components.append(arg)
                logger.debug(f"min of {components}.")
                res = self.min(components)
                logger.debug(f"Min result is {res}.")
            elif cmd == f"{self.command_sign}max":
                logger.debug(f"max of {t[1:]}.")
                components = []
                for arg in t[1:]:
                    if isinstance(arg, dict) and "parsed" in arg.keys():
                        components.append(arg["parsed"])
                    elif (
                        isinstance(arg, (tuple, list))
                        and len(arg) > 1
                        and isinstance(arg[0], str)
                        and arg[0][: len(self.command_sign)]
                        == self.command_sign
                    ):
                        sub_res = self.scan_tuple(arg)
                        components.append(sub_res["parsed"])
                        originals.append(sub_res["original"])
                    else:
                        components.append(arg)
                logger.debug(f"max of {components}.")
                res = self.max(components)
                logger.debug(f"Max result is {res}.")
            else:
                msg = (
                    "Found undefined command for current command "
                    + f"sign {self.command_sign}: {t}"
                )
                logger.debug(msg)
                raise NotImplementedError(msg)
            # elif add more parameter commands

            # check for arithmetic expression:
            if aritexp is not None:
                if isinstance(res, str):
                    if aritexp[0] == "+":
                        res = res + aritexp[1:]
                    else:
                        raise NotImplementedError(
                            f"Cannot operate '{aritexp}' on '{res}' (str)"
                        )
                elif isinstance(res, (int, float)):
                    if not is_valid_expression(aritexp):
                        raise PriorResultError(
                            f"'{aritexp}' is not a valid numeric "
                            + "arithmetic expression."
                        )
                    res = eval(str(res) + aritexp)
            # to deactivate, leave out ocmmand sign
            t_out = tuple([t[0][len(self.command_sign) :], t[1]])
            total_result = {"parsed": res, "original": t_out}
            if originals:
                total_result["originals"] = originals
            return total_result
        else:
            # it's just a normal tuple
            tout = []
            for i, it in enumerate(t):
                # logger.debug(f"{i}: {it}")
                tout.append(self.scan(it))
            if self.to_simple_type:
                return tout
            else:
                return tuple(tout)

    def get_prior_result(self, locator):
        """In some cases, input parameters for a module should be taken from
        prior results. This is performed here
        Args:
            locator : str
                the chain of attributes for finding the prior result, comma
                separated. They all need to be obtainable with getattr,
                starting from this class e.g. "results, 02_load, sample_movie,
                sample_frame_idx" obtains
                self.results['02_load']['sample_movie']['sample_frame_idx']
        Returns:
            the last attribute in the chain.
        """
        root_att = self.parent_object
        attribute_levels = [it.strip() for it in locator.split(",")]
        for i, att_name in enumerate(attribute_levels):
            if att_name == f"{self.command_sign}all":
                # root_att is a list, and all items should be equally processed
                # in the next rounds
                # logger.debug(f"Leaving {root_att}, to get all.")
                pass
            else:
                try:
                    if isinstance(root_att, list):
                        # logger.debug(
                        #     f"Getting all {att_name} attributes of {root_att}"
                        # )
                        root_att = [
                            self.get_attribute(list_att, att_name)
                            for list_att in root_att
                        ]
                    else:
                        root_att = self.get_attribute(root_att, att_name)
                except PriorResultError:
                    raise PriorResultError(
                        f'"{attribute_levels[i - 1]}" of "{locator}" not '
                        + f"present. Cannot get {att_name}. Check your "
                        + f"workflow {self.command_sign}get_prior_result "
                        + "argument."
                    )
        # logger.debug(f"Prior Result of {locator} is {root_att}")
        return root_att

    def get_previous_module_result(self, locator):
        """This is a convenience function for get_prior_result. It
        automatically prepends the previous module to the command.
        Args:
            locator : str
                the chain of attributes for finding the result from within
                the module; e.g. "sample_movie, sample_frame_idx". Called from
                module 3, this will obtain
                self.results['02_load']['sample_movie']['sample_frame_idx']
        Returns:
            the last attribute in the chain.
        """
        prev_module_idx = self.curr_rootidx - 1
        all_module_ids = list(self.parent_object.results.keys())
        prev_module_id = [
            mid
            for mid in all_module_ids
            if mid.startswith(f"{prev_module_idx:02d}_")
        ]
        prev_module_id = prev_module_id[0]
        locator = f"results, {prev_module_id}, {locator}"
        return self.get_prior_result(locator)

    def get_attribute(self, root_att, att_name):
        if isinstance(root_att, dict):
            att = root_att.get(att_name.strip())
        elif isinstance(root_att, object):
            try:
                att = getattr(root_att, att_name.strip())
            except AttributeError as e:
                if root_att is None:
                    raise PriorResultError()
                else:
                    logger.error(
                        f"Could not get attribute {att_name} "
                        + f"from {str(root_att)}."
                    )
                    raise e
        # logger.debug(f'From {root_att}, extracting "{att_name}": {att}')
        return att

    def sum(self, *args):
        """sum up components that may be given as iterables, or separate
        arguments.
        """
        components = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for ar in arg:
                    components.append(ar)
            else:
                components.append(arg)
        return np.sum(components)

    def product(self, *args):
        """Multiply components that may be given as iterables, or separate
        arguments.
        """
        components = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for ar in arg:
                    components.append(ar)
            else:
                components.append(arg)
        return np.product(components)

    def max(self, *args):
        """Take the maximum of components that may be given as iterables,
        or separate arguments.
        """
        components = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for ar in arg:
                    components.append(ar)
            else:
                components.append(arg)
        return np.max(components)

    def min(self, *args):
        """Take the minimum of components that may be given as iterables,
        or separate arguments.
        """
        components = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for ar in arg:
                    components.append(ar)
            else:
                components.append(arg)
        return np.min(components)


def is_valid_expression(expression):
    """Check for validity of a numeric expression, e.g. '* 3.1415"""
    # pattern = r"^[\d+\-*/\s()]+$"
    pattern = r"^[*-+/][0-9]*(\.[0-9]*)?"
    return re.match(pattern, expression) is not None


class ConditionEvaluator:
    """Evaluates conditions for conditional branching in workflows.

    Supports comparison operators (>, <, >=, <=, ==, !=) and logical
    operators (and, or) for combining conditions.
    """

    COMPARISON_OPERATORS = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, parameter_command_executor=None):
        """
        Args:
            parameter_command_executor : ParameterCommandExecutor or None
                if provided, will be used to resolve parameter commands
                in condition values (e.g., $get_prior_result)
        """
        self.parameter_command_executor = parameter_command_executor

    def evaluate(self, condition):
        """Evaluate a condition dictionary.

        Args:
            condition : dict
                Either a comparison condition with keys:
                    - "left": value or parameter command tuple
                    - "operator": str, one of >, <, >=, <=, ==, !=
                    - "right": value or parameter command tuple
                Or a logical condition with keys:
                    - "and": list of conditions (all must be True)
                    - "or": list of conditions (at least one must be True)

        Returns:
            bool : the result of the condition evaluation

        Raises:
            ValueError : if the condition format is invalid or operator
                        is unsupported
        """
        # Handle logical operators
        if "and" in condition:
            return all(self.evaluate(c) for c in condition["and"])
        elif "or" in condition:
            return any(self.evaluate(c) for c in condition["or"])

        # Handle comparison operators
        if not all(k in condition for k in ["left", "operator", "right"]):
            raise ValueError(
                "Condition must have 'left', 'operator', and 'right' keys, "
                f"or 'and'/'or' keys. Got: {condition.keys()}"
            )

        left = self._resolve_value(condition["left"])
        operator = condition["operator"]
        right = self._resolve_value(condition["right"])

        if operator not in self.COMPARISON_OPERATORS:
            raise ValueError(
                f"Unsupported operator: {operator}. "
                f"Supported operators: {list(self.COMPARISON_OPERATORS.keys())}"
            )

        result = self.COMPARISON_OPERATORS[operator](left, right)
        logger.debug(
            f"Condition evaluation: {left} {operator} {right} = {result}"
        )
        return result

    def _resolve_value(self, value):
        """Resolve a value, which may be a parameter command or a literal.

        Args:
            value : any
                the value to resolve. If it's a tuple starting with $,
                it will be resolved as a parameter command.

        Returns:
            any : the resolved value
        """
        # Check if it's a parameter command tuple
        if (
            isinstance(value, tuple)
            and len(value) > 1
            and isinstance(value[0], str)
            and value[0].startswith("$")
        ):
            if self.parameter_command_executor is None:
                raise ValueError(
                    "Cannot resolve parameter command without "
                    "ParameterCommandExecutor"
                )
            # Use the parameter command executor to resolve
            result = self.parameter_command_executor.scan_tuple(value)
            if isinstance(result, dict) and "parsed" in result:
                return result["parsed"]
            return result
        return value


class PriorResultError(AttributeError):
    pass


class ParameterTiler:
    """Multiplies a set of parameters according to a tile command.
    This has the usecase of e.g. doing multiple analogue analyses
    for different datasets, which are then aggregated.
    Uses the ParameterCommandExecutor, so the same commands will
    be used.
    """

    def __init__(
        self, parent_object, tile_entries, map_dict={}, command_sign="$$"
    ):
        """
        Args:
            parent_object : object
                the object to execute the command on.
                e.g. the WorkflowRunner itself
            tile_entries : dict
                one or multiple key-list pairs, where the lists
                have identical length. One parameter set will be
                generated for each item in the list. The keys should
                be used in a $map command in the parameters in 'run'.
                In addtition to the mapped variables, tile_entries
                may comprise '#tags', which are keyword tags for the
                list of parameter sets.
                for example:
                    tile_entries = {'file_name': ['a1.tiff', 'a2.tiff']}
                    parameters = {'load': {'filename': ('$map', 'file_name')}}
            map_dict : dict
                a dictionary to map values using the $map command
                the tile_entries will be added to the map_dict
            command_sign : str
                the command sign to execute on. In aggregation workflow
                preparation (Using the ParameterTiler), the single-workflow
                commands should not be executed, therefore different
                signs are used.
        """
        logger.debug("Initializeing ParameterTiler")
        self.tile_entries = tile_entries
        self.ntiles = len(list(tile_entries.values())[0])
        self.map_dict = map_dict
        self.parent_object = parent_object
        self.command_sign = command_sign

    def run(self, parameters):
        """Creates the tile set of parameters.
        Args:
            parameters : dict
                the parameters for a module
        Returns:
            result_parameters : list of dict
                the tiles of parameters
            tags : list of str
                if the map_dict contains the key '#tags', its value is
                returned (supposed to be tags to use for naming),
                otherwise list of empty strings
        """
        logger.debug("Running ParameterTiler.")
        result_parameters = []
        for i in range(self.ntiles):
            # set the tile parameters according to the iteration
            for k, v in self.tile_entries.items():
                self.map_dict[k] = v[i]
            logger.debug(f"Map for tile {i}: {self.map_dict}")
            pce = ParameterCommandExecutor(
                self.parent_object,
                self.map_dict,
                command_sign=self.command_sign,
            )
            # logger.debug(f"Running with parameters {parameters}")
            result_parameters.append(pce.run(copy.deepcopy(parameters)))
        if (tags := self.tile_entries.get("#tags")) is None:
            tags = [""] * len(result_parameters)

        return result_parameters, tags


def correct_path_separators(file_path):
    """Ensure correct path separators ('/' or '\') in a file path.
    Args:
        file_path : str
            input file path with any of the two separators
    Returns:
        file_path : str
            the file path with separators according to operating system
    """
    path_components = re.split(r"[\\/]", file_path)
    file_path = os.path.join(*path_components)
    if path_components[0] == "":
        file_path = os.sep + file_path
    return file_path


def get_caller_name(levels_back=1):
    """Get the name of a function in the trackeback (the caller,
    or the caller of the caller, ..).
    Args:
        levels_back : int
            the number of levels in the trace back.
            e.g. if you want a function name within that function,
            call: get_caller_name(1)
            if you want a the name of the caller, use
            get_caller_name(2)
    Returns:
        function_name : str
            the function name
    """
    # Get the current frame
    frame = inspect.currentframe()
    # Get the frames of the caller function enough levels back
    for i in range(levels_back):
        frame = frame.f_back
    # Get the name of that function
    function_name = frame.f_code.co_name
    return function_name


def multiply_recarray(ra, factor):
    columns = [it[0] for it in ra.dtype.descr if it[0] != ""]

    column_dtypes = [it[1] for it in ra.dtype.descr if it[0] in columns]
    dt = column_dtypes[0]
    if not all([it == dt for it in column_dtypes]):
        raise AttributeError("Cannot multiply, not all dtypes are the same")
    for i, col in enumerate(columns):
        nda = ra[col].astype(dt)
        ra[col] = nda * factor
    return ra


def stripplot(data, positions, jitter, ax, color, alpha=1):
    """Plot jittered data onto an axis. This can be a useful addition to
    a violin or boxplot, especially for sparse data.
    Args:
        data : list of 1D array, or 2D array
            the example datapoints to plot for each position
        positions : list of numeric
            the positions to plot the data at
        jitter : float
            the amount of jitter to add along x, to separate the data points
        ax : plt.axes
            the axes to plot in
        color : str or whatever matplotlib understands
            the color to plot with
        alpha : flot
            the transparency to plot with
    """
    for pos, d in zip(positions, data):
        x = pos * np.ones(len(d))
        x += np.random.uniform(-jitter / 2, jitter / 2, size=len(d))
        ax.scatter(x, d, color=color, alpha=alpha)

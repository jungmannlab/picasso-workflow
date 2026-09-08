#!/usr/bin/env python
"""Utility classes and functions for the package.

Defines :class:`AbstractModuleCollection` (the contract every analysis and
reporting module must implement) and the parameter-handling helpers
(:class:`DictSimpleTyper`, :class:`ParameterCommandExecutor`,
:class:`ParameterTiler`).

Author: Heinrich Grabmayr
Initial date: March 7, 2024
"""

from __future__ import annotations

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
    """Contract of the modules an analysis/reporting pipeline must support.

    Implemented by classes in ``analyse.py`` and ``confluence.py`` so the
    workflow class can call each side's matching methods.

    Notes
    -----
    Every module method takes ``(i, parameters, results)`` and returns the
    (possibly updated) ``parameters`` and ``results``. The ``results`` dict is
    pre-populated by the module decorator with ``start time``, ``end time``,
    ``duration`` and ``folder``; module docstrings below list only the keys
    they add on top of those.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def dummy_module(self, i, parameters, results):
        """Do nothing; a placeholder to disable a module without renumbering.

        Lets a module be removed from a workflow without renumbering the
        following result indices. For workflow debugging only; remove when
        done.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Uses no keys.
        results : dict
            Module results (decorator-provided keys only; see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Input results, unchanged.
        """

    @abc.abstractmethod
    def analysis_documentation(self, i, parameters, results):
        """Document where and how the analysis is being performed.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Uses no keys.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Results updated with: ``picasso version``, ``picasso-workflow
            version``, ``Architecture``, ``OS``, ``host``, ``processor``,
            ``CPU Frequency [MHz]``, ``CPU cores``, ``Memory total [GB]``,
            ``Memory available [GB]``, ``GPU`` (name or ``"N/A"``) and ``GPU
            memory [GB]`` (0 if no GPU).
        """

    @abc.abstractmethod
    def conditional_branch(self):
        """Execute different sub-module sequences based on a condition.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``condition`` : dict
                Either a comparison with keys ``"left"`` (value or parameter
                command tuple), ``"operator"`` (one of ``>``, ``<``, ``>=``,
                ``<=``, ``==``, ``!=``) and ``"right"`` (value or parameter
                command tuple), or a logical condition with ``"and"``/``"or"``
                keys.
            ``if_true`` : list of tuple
                ``(module_name, module_parameters)`` tuples to run if the
                condition is True.
            ``if_false`` : list of tuple
                ``(module_name, module_parameters)`` tuples to run if the
                condition is False.

            Optional keys:

            ``parameter_command_executor`` : ParameterCommandExecutor
                If provided, used to resolve parameter commands in condition
                values.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``condition_result`` (bool), ``branch_taken``
            (``"if_true"`` or ``"if_false"``), ``if_branch`` (dict of
            sub-module results) and ``branch_modules`` (dict of flat-indexed
            results).
        """

    ##########################################################################
    # Single-dataset workflow modules
    ##########################################################################

    @abc.abstractmethod
    def convert_zeiss_movie(self, i, parameters, results):
        """Convert a DNA-PAINT movie into picasso-supported ``.raw``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filepath`` : str
                The czi file to load.

            Optional keys:

            ``filename_raw`` : str
                The raw file to write to.
            ``info`` : dict
                Metadata as used by picasso.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``filepath_raw`` (full path to the output raw
            file) and ``filename_raw`` (its name).
        """

    @abc.abstractmethod
    def load_dataset_movie(self, i, parameters, results):
        """Load a DNA-PAINT movie dataset in a picasso-supported format.

        Loads movie data and metadata into ``self.movie`` and ``self.info``
        for subsequent analysis. Optionally creates sample movies and loads
        camera configuration.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filename`` : str
                Path to the movie file to load.

            Optional keys:

            ``sample_movie`` : dict
                Parameters for creating a subsampled movie.
            ``load_camera_info`` : bool
                Whether to load camera configuration from ``picasso.CONFIG``.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly modified (``sample_movie`` paths
            updated).
        results : dict
            Results updated with ``picasso version``, ``movie.shape`` (frames,
            width, height) and, if requested, ``sample_movie``.
        """

    @abc.abstractmethod
    def load_dataset_localizations(self, i, parameters, results):
        """Load a DNA-PAINT localizations dataset in a picasso format.

        The data is saved in ``self.locs`` and ``self.info``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filename`` : str
                The (main) file to load (image files or HDF5).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``picasso version`` and ``nlocs`` (number of
            localizations loaded).
        """

    @abc.abstractmethod
    def identify(self, i, parameters, results):
        """Identify localization sites in a loaded movie.

        Detects candidate sites by net-gradient thresholding, optionally
        performing automatic net-gradient detection and identification-vs-frame
        plots. The result is saved in ``self.identifications``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``box_size`` : int
                Size of the detection box in pixels.
            ``min_gradient`` : float
                Minimum net-gradient detection threshold (required unless
                ``auto_netgrad`` is provided).

            Optional keys:

            ``auto_netgrad`` : dict
                Automatic net-gradient detection parameters: ``box_size``,
                ``frame_numbers`` (list or int), ``filename``, ``start_ng``,
                ``zscore`` and ``bins``.
            ``ids_vs_frame`` : dict
                Identifications-vs-time plot parameters: ``filename``.
            ``identify_parallel`` : bool
                Run identification on multiple cores. Default is True.
            ``temporal_median_window`` : int
                Window (frames) of the picasso 0.11 temporal-median
                background filter. Omit to disable.
            ``temporal_median_stride`` : int
                Stride (frames) for the temporal-median filter.
            ``gaussian_filter_sigma`` : float
                Sigma of a spatial Gaussian pre-filter. Omit to disable.
            ``roi`` : tuple or list
                One or more rectangular ROIs to restrict detection to.
            ``frame_bounds`` : tuple or list
                One or more ``(start, end)`` frame ranges to detect within.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly with an updated ``min_gradient``.
        results : dict
            Results updated with ``num_identifications`` and, if requested,
            ``auto_netgrad`` and ``ids_vs_frame``.
        """

    @abc.abstractmethod
    def localize(self):
        """Localize the spots previously identified.

        The result is saved in ``self.locs``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``box_size`` : int
                Detection box size in pixels.

            Optional keys:

            ``fit_parallel`` : bool
                Whether to fit on multiple cores. Default is True.
            ``fitting_method`` : str
                picasso 0.11 fitting model (e.g. ``"gausslq"`` (default),
                ``"gaussmle"``, ``-rotated`` / ``-spherical`` variants,
                ``"spline"``, or their ``-gpu`` counterparts).
            ``spline_calibration`` : dict or str
                Spline-PSF calibration (dict or path); required for the
                ``spline`` methods and yields z directly.
            ``camera_calibration`` : dict or str
                Per-pixel sCMOS camera calibration (dict or path).
            ``eps`` : float
                Fitter convergence criterion.
            ``max_it`` : int
                Maximum number of fit iterations.
            ``locs_vs_frame`` : dict
                Plot-vs-time parameters (arguments of ``_plot_locs_vs_frame``).
            ``save_locs`` : dict
                If saving localizations is requested (arguments of
                ``save_locs``).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``locs_vs_frame`` (if requested) and
            ``locs_columns`` (column names of the localizations array).
        """

    @abc.abstractmethod
    def zfit(self):
        """Fit z coordinates of localized spots via astigmatic calibration.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``calibration`` : str or dict
                Filepath to a calibration file, or the calibration itself.

            Optional keys:

            ``fitting_method`` : str
                2D fitter the localizations came from (``"gausslq"`` or
                ``"gaussmle"``); used to compute the axial precision. Default
                ``"auto"`` infers it from the localize step's recorded method.
            ``gpu`` : bool
                Fit z on a CUDA-capable GPU. Default False.
            ``filter`` : int
                picasso z-fit RMSD filter (0 = none, default here; 2 =
                picasso's default).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Module results, unchanged apart from decorator-provided keys.
        """

    @abc.abstractmethod
    def load_picassoconfig(self):
        """Load a specific picasso configuration file.

        Used instead of the default config residing in the picasso
        installation folder.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_config`` : str
                Filepath to a config file.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Module results, unchanged apart from decorator-provided keys.
        """

    @abc.abstractmethod
    def export_brightfield(self):
        """Open single-plane tiff image(s) and save as PNG with contrast.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filepath`` : str or list of str or dict
                The tiff file(s) to load; converted files keep the name with a
                ``.png`` extension. If a dict, its keys are labels.

            Optional keys:

            ``min_quantile`` : float
                Quantile below which pixels are shown black. Default is 0.
            ``max_quantile`` : float
                Quantile above which pixels are shown white. Default is 1.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``labeled filepaths`` (label -> filepath) and
            ``success`` (whether the export succeeded).
        """

    @abc.abstractmethod
    def render(self):
        """Render localizations on the full FOV and a center-of-mass zoom.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``ctrmass_fov_nm`` : float
                Field of view (nm) of the zoom rendering around the center of
                mass.
            ``fullfov_pixelsize`` : float
                Rendered pixel size (nm) of the full-FOV rendering.
            ``ctrmass_pixelsize`` : float
                Rendered pixel size (nm) of the center-of-mass zoom.
            ``ctrmass_blur_method`` : str
                Blur method.
            ``ctrmass_min_blur_width`` : float
                Minimum blur width.
            ``ctrmass_ang`` : tuple
                Rotation angle.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``fp_scene_fullfov`` (full-FOV rendering) and,
            only if ``ctrmass_fov_nm`` was provided, ``fp_scene_ctrmass``
            (center-of-mass zoom rendering).
        """

    @abc.abstractmethod
    def undrift_rcc(self):
        """Undrift localized data using redundant cross-correlation (RCC).

        The drift is saved in ``self.drift``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``segmentation`` : int
                Number of frames per segment for RCC.

            Optional keys:

            ``max_iter_segmentations`` : int
                Max iterations to adaptively increase segmentation if RCC
                fails. Default is 3.
            ``filename`` : str
                The drift txt file name.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency. This module
            sets ``dimensions`` to ``['x', 'y']``.
        results : dict
            Results updated with ``success``, ``message`` and, only if
            undrifting succeeded, ``filepath_driftfile`` and ``filepath_plot``.
        """

    @abc.abstractmethod
    def undrift_rsso(self):
        """Undrift localized data using iterative RSSO drift correction.

        Applies an iterative RSSO (Redundant Spot Shift Overrepresentation)
        approach in which each frame is compared against the whole dataset to
        compute that frame's total drift, repeated on the undrifted dataset to
        improve accuracy. Includes uncertainty analysis, confidence evaluation,
        windowing and outlier detection.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``ton`` : float
                Half-life of a localization in frames (how long a spot stays
                visible).
            ``toff`` : float
                Frames for a spot to reappear after disappearing.
            ``max_shift`` : float
                Maximum expected drift per frame in pixels.

            Optional keys (defaults in parentheses):

            ``min_locs_per_frame`` : int
                Min localizations per frame for reliable estimation (10).
            ``max_iterations`` : int
                Max iterative refinement rounds (5).
            ``convergence_threshold`` : float
                RMS drift-change convergence threshold in nm (0.1).
            ``plot_drift`` : bool
                Whether to save drift plots (True).
            ``save_locs`` : bool
                Whether to save undrifted localizations (True).
            ``n_processes`` : int or None
                Processes for parallel computation (auto).
            ``confidence_threshold`` : float
                Confidence threshold for windowing analysis (0.8).
            ``outlier_detection_enabled`` : bool
                Enable RSSO failure and outlier detection (True).
            ``outlier_z_threshold`` : float
                Z-score threshold for temporal outlier detection (3.5).
            ``min_signal_to_noise`` : float
                Min signal-to-noise ratio for drift measurements (0.5).
            ``windowing_enabled`` : bool
                Enable adaptive windowing for low-confidence frames (True).
            ``window_size_range`` : tuple
                Min and max window sizes for adaptive windowing ((3, 20)).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``success``, ``drift_x``/``drift_y`` (total
            drift trajectories in nm per frame), ``uncertainty_x``/
            ``uncertainty_y``, ``drift_quality`` (per-frame confidence),
            ``n_iterations``, ``convergence_rms`` and ``drift_plots`` (path to
            the visualization).
        """

    @abc.abstractmethod
    def undrift_aim(self):
        """Undrift localized data using the AIM algorithm.

        The drift is saved in ``self.drift``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``segmentation`` : int
                Number of frames per segment.
            ``intersect_d`` : float
                Intersect distance in nanometers.
            ``roi_r`` : float
                Radius of the local search region in nm; should exceed the
                maximum expected drift within a segment.
            ``dimensions`` : list of str
                The dimensions to undrift, typically ``['x', 'y']``.

            Optional keys:

            ``progress`` : callable
                Progress callback for status updates.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``success``, ``fp_driftfile`` (drift txt file)
            and ``fp_fig`` (drift plot PNG).
        """

    @abc.abstractmethod
    def manual(self):
        """Handle a manual step that waits for user-provided files.

        If the required files are not present, prompt the user to provide
        them; if they are, move on to the next step.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``prompt`` : str
                The user prompt.
            ``filename`` : str
                The file the user should provide.

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def summarize_dataset(self):
        """Summarize a dataset using various quality-metric methods.

        Computes metrics such as NeNa (nearest-neighbour analysis) and median
        localization precision.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``methods`` : dict
                Analysis methods to run, mapping method name to a
                method-specific parameter dict. Supported methods:

                ``"nena"`` : dict
                    Nearest-neighbour analysis (no parameters).
                ``"median-loc-precision"`` : dict
                    Median localization precision; optional ``qe_correction``
                    (quantum-efficiency correction factor, default 1).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated, depending on the methods used, with:

            ``nena`` : dict
                Keys ``res`` (all best-fit values), ``NeNa`` (formatted
                result), ``nena-px``, ``nena-nm`` and ``filepath_plot``.
            ``median-loc-precision`` : dict
                Keys ``median_lp-px`` and ``median_lp-nm``.
        """

    # @abc.abstractmethod
    # def aggregate_cluster(self):
    #     """Aggregate along the cluster column.
    #     Uses picasso.postprocess.cluster_combine"""
    #     pass

    @abc.abstractmethod
    def density(self):
        """Calculate the local localization density.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                The radius for calculating local density.

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def dbscan(self, i, parameters, results):
        """Cluster localizations using DBSCAN.

        Optionally replaces localizations with cluster centers for subsequent
        analysis; after this module the standard locs are the cluster centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                The DBSCAN radius parameter in nm.
            ``min_samples`` : int
                Minimum number of samples required for a cluster.
            ``continue_with_centers`` : bool
                Whether to replace localizations with cluster centers.

            Optional keys:

            ``save_locs`` : bool
                Whether to save clustered localization data to the results
                folder.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Results updated with ``fp_fig_clustersizes`` (cluster-size
            distribution figure) and ``fp_centers`` (cluster centers file).
        """

    @abc.abstractmethod
    def hdbscan(self):
        """Cluster localizations using HDBSCAN.

        After this module the standard locs are the cluster centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``min_cluster`` : float
                The HDBSCAN ``min_cluster_size``.
            ``min_samples`` : float
                The HDBSCAN ``min_samples``.

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def binding_event_analysis(self):
        """Evaluate binding events following Steen et al.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_locs`` : str
                File path to the input locs.
            ``n_frames`` : int
                Number of frames in the acquisition.
        results : dict
            Module results (see class docstring).

        References
        ----------
        Steen, P.R., Unterauer, E.M., Masullo, L.A. et al. The DNA-PAINT
        palette: a comprehensive performance analysis of fluorescent dyes.
        Nat Methods (2024). https://doi.org/10.1038/s41592-024-02374-8
        """

    @abc.abstractmethod
    def resolution_analysis(self):
        """Estimate spatial resolution via point-pattern autocorrelation.

        Computes a 2D autocorrelation function and fits a Gaussian to extract
        resolution metrics, including 2D Gaussian fitting, radial profile
        computation and a 1D Gaussian fit to the radial profile.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``delta_r`` : float
                Grid spacing for autocorrelation (default 5 nm).
            ``r_max`` : float
                Maximum radius for autocorrelation (default 100 nm).
            ``batch_size`` : int or None
                Data points per batch for chunking (auto-calculated if None).
            ``n_processes`` : int or None
                Number of parallel processes (auto-detected if None, capped
                at 4).
            ``use_chunking`` : bool
                Memory-efficient chunking for large datasets (default True).
            ``use_sparse`` : bool
                Use sparse matrices for very large grids (default False).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``resolution`` (average FWHM, nm),
            ``sigma_x``/``sigma_y``, ``fwhm_x``/``fwhm_y``, ``fit_quality``
            (R-squared), ``autocorr_map``, ``radial_profile``,
            ``radial_distances``, ``resolution_radial`` (radial-fit FWHM),
            ``resolution_dblradial`` (double-Gaussian FWHM), ``fig_resolution``
            and ``fig_radial``.
        """

    @abc.abstractmethod
    def resolution_frc_spatial(self):
        """Calculate resolution using a spatial FRC approach.

        Divides the FOV into spatial regions, computes FRC for each region
        independently and averages the results. This lowers memory usage
        (smaller images per region), improves statistics through spatial
        averaging, parallelises efficiently (fully independent regions) and
        preserves high spatial frequencies.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``pixelsize_render`` : float
                Pixel size for rendered images in nm (default 5 nm).
            ``smoothing_sigma`` : float or None
                Gaussian smoothing sigma in pixels (default None).
            ``threshold`` : float
                FRC threshold for the resolution cutoff (default 1/7 ≈ 0.143).
            ``region_size`` : float
                Size of each spatial region in micrometers (default 10.0 µm).
            ``min_locs_per_region`` : int
                Minimum localizations per region to process (default 500).
            ``max_frc_range_nm`` : float or None
                Maximum FRC range in nm (default None = full range).
            ``n_processes`` : int
                Number of parallel processes (default 4).
            ``smoothing_window`` : float
                Moving-average window size for FRC smoothing in 1/nm
                (default 0.005).
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, possibly updated for consistency.
        results : dict
            Results updated with ``resolution_frc_spatial`` (mean FRC-based
            resolution, nm), ``resolution_std``, ``n_regions``,
            ``cutoff_frequency`` (mean spatial frequency at cutoff, 1/nm),
            ``frc_curve_mean``, ``frc_curve_std``, ``spatial_frequencies``,
            ``threshold`` and ``fig_frc``.
        """

    @abc.abstractmethod
    def smlm_clusterer(self):
        """Cluster localizations using the SMLM clusterer.

        After this module the standard locs are the cluster centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                The SMLM radius in nm.
            ``min_locs`` : float
                The SMLM ``min_locs``.

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
            ``basic_fa`` : bool
                The SMLM ``basic_fa`` (default False).
            ``radius_z`` : float
                The SMLM ``radius_z`` (default None).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def gaussian_mixture_cluster(self):
        """Cluster localizations using Gaussian mixture models.

        After this module the standard locs are the Gaussian centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``locs`` : np.recarray
                Localizations.
            ``info`` : list
                Information dictionaries.
            ``min_locs`` : int
                Minimum localizations per component, used to filter out
                components with too few localizations (likely background).

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
            ``max_rounds_without_best_bic`` : int
                Max rounds without BIC improvement before terminating the
                optimal-GMM search (default 3).
            ``bootstrap_check`` : bool
                If True, compute the standard error of the means via
                bootstrapping; otherwise use the single-Gaussian SEM
                approximation (default False).
            ``calibration`` : dict
                Calibration with x/y coefficients, z step size and number of
                frames. Required only for 3D data (default None).
            ``asynch`` : bool
                If True, run the GMM search in parallel via multiprocessing
                (default True).
            ``callback_parent`` : function
                Parent object for the progress-bar callback. If None, the bar
                is shown on the console; if ``'silent'``, nothing is shown
                (default ``'silent'``).
            ``sigma_bounds`` : float
                Minimum Gaussian-component standard deviation in nm (not
                recommended now that individual loc precision is used).
            ``loc_prec_handle`` : {"local", "global", "abs"}
                How to handle localization precision (default ``"local"``).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def nneighbor(self):
        """Compute nearest-neighbour distances.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``dims`` : list of str
                Distance dimensions, e.g. ``['x', 'y']`` or
                ``['x', 'y', 'z']``.
            ``nth_NN`` : int
                Compute the 1st to nth nearest-neighbour distances.
            ``nth_rdf`` : int
                Compute distances up to the 95th percentile of the
                ``nth_rdf`` nearest neighbour.
            ``subsample_1stNN`` : int
                Fold by which to subsample distances from the median of the
                1st nearest neighbour (default 20).
            ``add_column`` : bool
                Whether to add a nearest-neighbour-distance column to the locs.

            Optional keys:

            ``save_locs`` : bool
                Whether to save the locs into the results folder.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def fit_csr(self, i, parameters, results):
        """Fit a complete-spatial-randomness model to nearest neighbours.

        Fits a CSR model to nearest-neighbour distance distributions and
        evaluates goodness-of-fit with statistical measures and visualization.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``nneighbors`` : str or numpy.ndarray or list
                A filepath to a nearest-neighbour data file, a 2D ``(N, k)``
                array of kth nearest-neighbour distances, or a list of
                multiple datasets / file paths.
            ``dimensionality`` : int
                Spatial dimensionality (2 or 3) for the CSR model.

            Optional keys:

            ``kmin`` : int
                Minimum kth nearest-neighbour order to fit (default 1).
            ``min_dist`` : float
                Minimum observable distance in nm due to technical limits.
            ``max_dist`` : float
                Maximum distance for filtering analysis. Bounds the fit
                only, not the plotting range (see ``plot_max_dist``).
            ``bkg_fraction`` : float
                Background fraction for fitting.
            ``fit_bkg`` : bool
                Whether to fit the background (default False).
            ``plot_max_dist`` : float
                Maximum distance shown on the plot's distance axis, in nm.
                Only affects display, not the fit. Defaults to the 95th
                percentile of the largest-k neighbour distances.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Results updated with ``density`` (fitted spatial density in
            units^(-d)), ``bkg_fraction``, ``fp_fig`` (CSR fit figure(s)),
            ``wasserstein_distances_per_k``, ``mean_wasserstein_distance`` and
            ``ks_pvalues_per_k`` (Kolmogorov-Smirnov p-values per kth order).
        """

    # @abs.abstractmethod
    # def radial_distribution_function(self):
    #     """Generate the Radial Distribution Function,
    #     Whis is the sum of nearest neighbors with geometry factor.
    #     At long radii, its value is the overall density.
    #     """
    # pass

    @abc.abstractmethod
    def save_single_dataset(self):
        """Save the locs and info of a single dataset.

        Makes loading for the aggregation workflow more straightforward.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filename`` : str
                The name of the dataset.
        results : dict
            Module results (see class docstring).
        """

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    @abc.abstractmethod
    def load_datasets_to_aggregate(self):
        """Load the results of single-dataset workflows for aggregation.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``filepaths`` : list of str
                The hdf5 files to load.
            ``tags`` : list of str
                The tags naming the datasets.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def align_channels(self):
        """Align multiple channels to each other (aggregation workflow).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``filepaths`` : list of str
                Previously saved hdf5 files to load and align. If omitted, the
                last processed data is used.
            ``align_pars`` : dict
                Kwargs of ``picasso_outpost.align_channels``
                (``max_iterations``, ``convergence``).
            ``fp_fiducials`` : list of str
                Previously saved hdf5 files of fiducial markers to load and
                align.
            ``fig_filename`` : str
                Where to save the drift figure.
            ``crop_boundaries`` : bool
                Whether to crop localizations to the image boundaries after
                shifting.
            ``fp_co_shift_channel_locs`` : list of str
                hdf5 files outside the main workflow to shift as well (e.g.
                clustered localizations when the workflow continued with
                cluster centers).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def register_channels(self):
        """Register channels via picasso 0.11 bead-based transforms.

        Fits a higher-degree-of-freedom transform (affine / projective /
        polynomial) between channels from fiducial-bead images using
        ``picasso.registration`` and warps each channel's localizations into
        the reference frame. Complements the translation-only
        :meth:`align_channels`.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``bead_movies`` : list of str
                One bead-calibration movie file per channel, in channel order.
            ``box_size`` : int
                Box size used to detect and fit the beads.
            ``min_gradient`` : float or list
                Minimum net gradient for a bead candidate.

            Optional keys:

            ``model`` : str
                Transform model: ``"affine"`` (default), ``"projective"``,
                ``"polynomial2"`` or ``"polynomial3"``.
            ``reference`` : int
                Index of the reference channel. Default 0.
            ``filepaths`` : list of str
                Channel hdf5 files to load first (as in ``align_channels``).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def combine_channels(self):
        """Combine multiple channels into one dataset (e.g. for RESI).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``tag`` : str
                The tag / name of the combined dataset.
            ``combine_col`` : str
                The column name for the IDs of the different datasets.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def save_datasets_aggregated(self):
        """Save data of all single-dataset workflows in an aggregation.

        Saves all channel localization data and metadata from the aggregated
        workflow to individual files in the results folder.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Uses no keys.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Results updated with ``filepaths`` (all saved file paths from the
            aggregated datasets).
        """

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
        """Run a direct SPINNA batch analysis.

        The current locs file(s) are saved into the results folder and a
        template csv is created, which the user fills out in a manual step
        before the SPINNA analysis is carried out.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``labeling_efficiency`` : dict of float
                Labeling efficiency (range 0-1) for all targets.
            ``labeling_uncertainty`` : float or dict of float
                Labeling uncertainty in nm (e.g. 5); a scalar is applied to
                all targets.
            ``n_simulate`` : int
                Number of target molecules to simulate (e.g. 50000).
            ``structures`` : str or list of dict
                A filepath to a structures YAML, or SPINNA structures as a
                list of dicts, each with ``"Molecular targets"`` (list of
                str), ``"Structure title"`` (str) and ``"TARGET_x"`` /
                ``"TARGET_y"`` / ``"TARGET_z"`` (lists of float) for each
                target named in ``"Molecular targets"``.
            ``fp_mask_dict`` : str
                Filepath to the mask_dict file.
            ``density`` : list of float
                Density to simulate in 1/nm^d (area density in 2D, volume
                density in 3D). Either ``density`` or ``density_app`` is
                required.
            ``random_rot_mode`` : {"2D", "3D"}
                Mode of molecule rotation in the simulation.
            ``sim_repeats`` : int
                Number of simulation repeats.
            ``fit_NND_bin`` : float
                Bin size of the fits.
            ``fit_NND_maxdist`` : float
                Maximum of the histogram.
            ``n_nearest_neighbors`` : int
                Number of nearest neighbours to evaluate.
            ``granularity`` : float
                The SPINNA granularity.

            Optional keys:

            ``density_app`` : list of float
                Apparent density in 1/nm^2 (the product of the real density
                and the labeling efficiency).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def spinna_batch(self):
        """Run a SPINNA batch analysis from a pre-existing config file.

        File-path columns of the config csv (``structures_filename``,
        ``exp_data_*`` and ``mask_filename_*``) are converted to the
        current machine using the Drivepaths config. The modified
        config is written to a copy inside the module's results folder
        -- the user's original csv is not changed -- and that copy is
        passed on to picasso's batch analysis.

        If ``use_workflow_locs`` is True, the current locs file(s) are
        additionally saved as .hdf5 into the module's results folder
        and their filepaths are written into the SPINNA batch config
        csv as one ``exp_data_<tag>`` column per channel. When False
        (the default) the ``exp_data_*`` columns from the user-provided
        csv are used as-is (after path conversion).

        The config csv must already be prepared by the user. See
        ``picasso.__main__._spinna_batch_analysis`` for the columns
        expected in the config file.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_spinna_batch_config`` : str
                Path to the user-prepared SPINNA batch-analysis config csv.

            Optional keys:

            ``use_workflow_locs`` : bool
                If True, save this workflow's current locs and inject their
                paths into the batch config. Default is False.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def ripleysk(self):
        """Compute Ripley's K spatial statistics for the dataset."""

    # @abc.abstractmethod
    # def ripleysk_rafal(self):
    #     pass

    @abc.abstractmethod
    def ripleysk2(self):
        """Compute Ripley's K statistics (second implementation)."""

    @abc.abstractmethod
    def ripleysk_average(self):
        """Average Ripley's K curves across datasets."""

    @abc.abstractmethod
    def ripleysk_average2(self):
        """Average Ripley's K curves across datasets (second variant)."""

    @abc.abstractmethod
    def protein_interactions(self):
        """Quantify protein-protein interactions from the localizations."""

    @abc.abstractmethod
    def protein_interactions_average(self):
        """Average protein-interaction metrics across datasets."""

    @abc.abstractmethod
    def create_mask(self):
        """Calculate a cell mask (Susanne's original DC-Atlas implementation).

        Kept for backwards compatibility; may be obsolete given
        :meth:`create_mask2` and is slated for eventual deprecation.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_channel_map`` : str
                Filepath to the channel map from the ``combine_channels``
                module (channel name -> ID int in ``locs['combine_id']``).
            ``fp_combined_locs`` : str
                Filepath to the locs combined in the ``combine_channels``
                module.
            ``margin`` : float
                Size of the empty margin added to the FOV, in nm.
            ``binsize`` : float
                Size of the first-step 2D histogram bins, in nm.
            ``sigma_mask_blur`` : int
                Gaussian-blur parameter, in binsize units.
            ``mask_resolution`` : float
                Digital resolution of the mask, in nm.
            ``combine_col`` : str
                Name of the combine column (e.g. ``'combine_id'`` or
                ``'protein'``), as used in the ``combine_channels`` module.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def create_mask2(self):
        """Calculate a cell mask (Rafal's DC-Atlas v3 implementation).

        Largely identical to an implementation in spinna that will be
        integrated into picasso; evaluate deprecation (or moving the source
        from ``outpost_modules/ripleys`` to ``picasso/spinna``) at that time.
        The locs must be protein positions at this stage.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``binsize`` : float
                Bin size in nm (a good value is 20).
            ``blursize`` : float
                Gaussian blur to apply in nm (a good value is 400).
            ``mask_pixel_size`` : float
                Pixel size of the final mask in nm (often 10).
            ``threshold`` : float
                Threshold below which the mask is set to zero (e.g. 1/3).
            ``binary`` : bool
                Whether to create a binary (vs density) mask.
            ``select_cell`` : bool
                Whether to keep the largest connected component (assumed to be
                the cell of interest).
            ``fill_holes`` : bool
                Whether to fill holes in the cell mask.
            ``dilate_nm`` : float
                Nanometers to dilate the mask (useful with a large threshold).
            ``apply_to_locs`` : bool
                Whether to drop all localizations outside the area.

            Optional keys:

            ``fp_combined_locs`` : str
                Filepath to the locs combined in the ``combine_channels``
                module. If None or ``''``, the loaded ``channel_locs`` is used.
            ``fp_channel_map`` : str
                Filepath to the channel map from the ``combine_channels``
                module (channel name -> ID int in ``locs['combine_id']``).
            ``combine_col`` : str
                Name of the combine column (e.g. ``'combine_id'`` or
                ``'protein'``).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def refine_mask_by_density(self):
        """Analyse and refine a previously created mask by density.

        Plots the density histogram of the mask bins so an area of homogeneous
        density can be selected. The locs must be protein positions at this
        stage.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_mask`` : str
                Filepath to the mask.
            ``min_density``, ``max_density`` : float
                The density range to select.

            Optional keys:

            ``nbins`` : int
                Number of bins for plotting.
            ``nth_largest`` : int
                Select the nth largest area in the density range (1-based; 1
                for the largest).
            ``apply_to_locs`` : bool
                Whether to apply the created mask to the locs.
            ``smoothe_nm`` : float
                Nanometers to dilate and erode the mask, useful to remove
                excessive holes and ragging from density thresholding.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def dbscan_molint(self):
        """Run DBSCAN for the molecular-interactions workflow.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_channel_map`` : str
                Filepath to the channel map from the ``combine_channels``
                module (channel name -> ID int in ``locs['combine_id']``).
            ``epsilon_nm`` : float
                DBSCAN epsilon in nm.
            ``minpts`` : int
                Minimum number of points.
            ``sigma_linker`` : float
                Linker size in nm.
            ``fp_merge_mask`` : str
                Filepath to the merge mask (from the ``create_mask`` module).
            ``thresh_type`` : str
                Threshold type.
            ``cell_name`` : str
                Name of the cell currently analyzed.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def CSR_sim_in_mask(self):
        """Simulate CSR within a density mask and run DBSCAN on it.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_channel_map`` : str
                Filepath to the channel map from the ``combine_channels``
                module (channel name -> ID int in ``locs['combine_id']``).
            ``fp_mask_dict`` : str
                Filepath to the ``mask_dict.pkl`` from the ``create_mask``
                module.
            ``N_repeats`` : int
                Number of simulation repeats.
            ``epsilon_nm`` : float
                DBSCAN epsilon in nm.
            ``minpts`` : int
                Minimum number of points.
            ``sigma_linker`` : float
                Linker size in nm.
            ``fp_merge_mask`` : str
                Filepath to the merge mask (from the ``create_mask`` module).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def find_cluster_motifs(self):
        """Analyse the binary barcode results of the molint DBSCAN.

        Compares experimental to CSR data, merged over multiple cells.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_workflows`` : list of str
                Paths to the folders of the separate workflows where the
                individual Ripley's analyses were done.
            ``report_names`` : list of str
                The report names of those workflows.
            ``swkfl_dbscan_molint_key`` : str
                Results key of the DBSCAN module (e.g. ``'09_dbscan_molint'``).
            ``swkfl_CSR_sim_in_mask_key`` : str
                Results key of the CSR DBSCAN module
                (e.g. ``'10_CSR_sim_in_mask'``).
            ``population_threshold`` : float
                Only select barcodes with a relative population above this
                (range 0-1).
            ``ttest_pvalue_max`` : float
                The p-value below which the experiment-vs-CSR difference in
                cluster count for a barcode is deemed significant.
            ``channel_colors`` : list of str
                Colors describing the receptors.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def interaction_graph(self):
        """Plot the target-interaction graph.

        Displays the targets and their interactions as a graph: node sizes
        denote density and the Ripley interaction matrix is represented in the
        edges.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_workflows`` : list of str
                Paths to the folders of the separate workflows where the
                individual Ripley's analyses were done.
            ``report_names`` : list of str
                The report names of those workflows.
            ``swkfl_protint_key`` : str
                Results key of the ``protein_interactions`` module
                (e.g. ``'09_protein_interactions'``).
            ``fp_density`` : str
                Filepath to the channel densities.
            ``fp_ripleys_meanvals`` : str
                Filepath to the interaction matrix.
            ``edge_factor``, ``node_factor`` : float
                Scaling factors for useful display sizes.
            ``channel_colors`` : list of str
                Colors describing the receptors.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def plot_densities(self):
        """Aggregate and plot densities and cell areas across datasets.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_workflows`` : list of str
                Paths to the folders of the separate workflows where the
                individual Ripley's analyses were done.
            ``report_names`` : list of str
                The report names of those workflows.
            ``swkfl_create_mask_key`` : str
                Results key of the mask module (e.g. ``'11_create_mask'``).
            ``swkfl_protint_key`` : str
                Results key of the ``protein_interactions`` module
                (e.g. ``'09_protein_interactions'``).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def find_gold(self):
        """Find localizations from gold beads via blinking kinetics.

        The metrics used are the number of locs and the RMS deviation from the
        mean frame.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``remove_gold`` : bool
                If True, discard the gold locs and set ``self.locs`` to the
                non-gold locs.
            ``diameter`` : float
                The pick-similar diameter for identifying gold.
            ``std_range``, ``mean_rmsd`` : float
                The pick-similar parameters identifying gold.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def find_similar(self):
        """Pick-similar in nlocs/rmsd space within specified limits.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``diameter`` : float
                The pick-similar diameter for identifying gold.

            Optional keys:

            ``min_n_locs_per_frame``, ``max_n_locs_per_frame`` : float
                Min/max percentage (range 0-1) of frames with events in the
                pick region to pick. Default 0.01.
            ``min_rmsd``, ``max_rmsd`` : float
                Minimum/maximum RMS distance from the pick center to pick.
            ``n_plot_structures`` : int
                Number of structures to plot.
            ``display_pixelsize`` : float
                Pixel size for display in nm (default 1).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def find_structures(self):
        """Pick-similar on clusters in nlocs/rmsd space.

        Useful for automated picking of origamis, and to help define
        parameters for finding gold.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``diameter`` : float
                The pick-similar diameter for identifying gold.

            Optional keys:

            ``min_n_locs_per_frame`` : float
                Percentage of frames with events in the pick region below
                which there is noise (default 0.01).
            ``n_plot_structures`` : int
                Number of structures to plot.
            ``display_pixelsize`` : float
                Pixel size for display in nm (default 1).
            ``xi`` : float
                The OPTICS ``xi`` clustering parameter (default 0.05).
            ``min_cluster_size`` : float
                Minimum cluster size as a fraction (default 0.05).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def undrift_from_picked(self):
        """Undrift using picked localizations.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``fp_picked_locs`` : str
                Filepath to the picked locs to undrift from (an hdf5 file of
                locs with a ``'group'`` column describing the picks).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def filter_locs(self):
        """Filter localizations to a min-max range of a metric.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``field`` : str or list of str
                The field(s) to filter on.

            Optional keys:

            ``minval``, ``maxval`` : dtype of field (or list thereof)
                The minimum/maximum value(s) to accept.
            ``mode`` : str
                How thresholds are applied: ``"absolute"`` (values in the
                field's units), ``"zscore"`` (standard deviations from the
                mean; ``-2, 2`` cuts at 2*std) or ``"quantile"`` (quantiles).
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def filter_transient_binding(self, i, parameters, results):
        """Filter molecule positions for transient binding.

        Keeps positions (after clustering or Gaussian mixture) whose mean
        frame is not at extreme temporal positions (default
        ``0.1 > mean_frame / nframes`` or ``> 0.9``) and whose frame standard
        deviation is large enough (default ``std_frame > 0.3``).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``meanframe_cutoff`` : float
                Filter out positions at more extreme temporal positions
                (range 0-1, default 0.1).
            ``stdframe_cutoff`` : float
                Filter out positions with a lower frame std than this.
            ``fp_locs`` : str
                Filepath to the underlying localizations (``self.locs`` are
                centers). If given, these are filtered as well and saved under
                the same filename in the current results folder.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def link_locs(self):
        """Link localizations across frames.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``d_max`` : int
                Maximum distance to link, in pixels.
            ``tolerance`` : int
                Maximum transient dark time, in frames.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def pairwise_module_executor(self):
        """Call another module as a sub-module for all channel pairs.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``module_name`` : str
                The module to call.
            ``param_target1``, ``param_target2`` : str
                Parameter names of the first and second targets to set on the
                sub-module.
            ``module_kwargs`` : dict
                The other arguments to the sub-module.

            Optional keys:

            ``result_scalar`` : str
                Results key to display in a heatmap as the main result.
            ``scalar_threshold`` : float
                Saturation value in the heatmap.
            ``scalar_minval`` : float
                Minimum value for color in the heatmap.
            ``result_fpfig`` : str or list of str
                Results key(s) of figure filepath(s) to display for
                documentation.
        results : dict
            Module results (see class docstring).
        """

    @abc.abstractmethod
    def random_val(self):
        """Generate a random value and test plot for debugging.

        Used to debug and test the pairwise-module machinery.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``xlabel``, ``ylabel`` : str
                Axis labels for the test plot.
        results : dict
            Module results (see class docstring).

        Returns
        -------
        parameters : dict
            Input parameters, unchanged.
        results : dict
            Results updated with ``random_val`` (a value in [0, 1]) and
            ``fp_fig`` (filepath to the generated test figure).
        """

    @abc.abstractmethod
    def labeling_efficiency_analysis(self):
        """Analyse labeling efficiency via a 3-component SPINNA analysis.

        Performs a 3-component SPINNA analysis for monomers and heterodimers
        of target (A) and reference (B). The analysis is run with a labeling
        efficiency of 1, yielding the proportions of monomers and dimers seen
        in the data; the real labeling efficiency is then derived as in the
        Notes.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``reference_name`` : str
                Channel tag of the reference.
            ``target_name`` : str
                Channel tag of the target queried for labeling efficiency.
            ``pair_distance`` : float
                Real distance of a pair of tags, in nm (e.g. 10).
            ``labeling_uncertainty`` : dict
                Channel tag -> labeling uncertainty in nm (e.g. 5).
            ``n_simulate`` : int
                Number of target molecules to simulate (e.g. 50000).
            ``density`` : dict
                Channel tag -> density to simulate (area density in 2D,
                volume density in 3D).
            ``granularity`` : float
                The SPINNA ``res_factor``.
            ``sim_repeats`` : int
                Number of simulation repeats, for noise reduction.

            Optional keys:

            ``nn_nth`` : int
                Number of nearest neighbours to analyse (default 1).
        results : dict
            Module results (see class docstring).

        Notes
        -----
        Binders A and B bind to an engineered construct ``A*-anchor-B*``::

            A <-> A*-anchor-B* <-> B

        with four possible configurations: ``A_only`` (``AA*-anchor-B*``),
        ``AB`` (``AA*-anchor-B*B``), ``B_only`` (``A*-anchor-B*B``) and
        ``None`` (``A*-anchor-B*``, invisible in the data). The totals are
        ``#A_tot = #A_only + #AB`` and ``#B_tot = #B_only + #AB``.

        Proportions may be expressed per #structures or per #molecules; e.g.
        10 monomers and 10 dimers give ``p_m = 50%, p_d = 50%`` per
        #structures but ``p_m = 33%, p_d = 66%`` per #molecules. With

        ::

            #AB     = #anchor * LE_A * LE_B
            #A_only = #anchor * LE_A * (1 - LE_B)
            #B_only = #anchor * LE_B * (1 - LE_A)

        the labeling efficiency follows, per #structures::

            LE_A = prop(AB) / (prop(B) + prop(AB))
            LE_B = prop(AB) / (prop(A) + prop(AB))

        and per #molecules::

            LE_A = prop(AB) / (2 * prop(B) + prop(AB))
            LE_B = prop(AB) / (2 * prop(A) + prop(AB))

        SPINNA outputs proportions per #molecules, so the latter are used.
        """


class DictSimpleTyper:
    """Scan a nested structure, converting numpy arrays/tuples to lists."""

    def __init__(self, to_simple_type: bool = True):
        """Initialize the typer.

        Parameters
        ----------
        to_simple_type : bool, optional
            If True, convert numpy arrays and tuples to lists and numpy
            scalars to Python scalars. Default is True.
        """
        self.to_simple_type = to_simple_type
        self.curr_rootidx = 0

    def run(self, parameters: dict):
        """Scan a parameter set, applying simple-type conversion.

        Parameters
        ----------
        parameters : dict
            The parameters for a module.

        Returns
        -------
        dict
            The scanned parameters.
        """
        logger.debug("Running DictSimpleTyper")
        return self.scan(parameters)

    def scan(self, itrbl, root_level: bool = False):
        """Scan one value, recursing into containers.

        Parameters
        ----------
        itrbl : object
            The value to scan (usually an iterable).
        root_level : bool, optional
            Whether the value is at the root level; if so, its index is
            stored. Default is False.

        Returns
        -------
        object
            The scanned (and possibly type-converted) value.
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
    """Scan parameter sets for commands and execute them.

    Useful e.g. in :class:`~picasso_workflow.workflow.WorkflowRunner`, where
    some parameters of later modules depend on results of previous modules and
    can be retrieved via the commands this class understands.
    """

    def __init__(
        self,
        parent_object=None,
        map_dict: dict = {},
        to_simple_type: bool = False,
        command_sign: str = "$",
    ):
        """Initialize the command executor.

        Parameters
        ----------
        parent_object : object, optional
            The object to execute commands on, e.g. the
            :class:`~picasso_workflow.workflow.WorkflowRunner` itself.
        map_dict : dict, optional
            A dictionary used to map values via the ``$map`` command.
        to_simple_type : bool, optional
            If True, convert numpy arrays and tuples to lists and numpy
            scalars to Python scalars. Default is False.
        command_sign : str, optional
            The command sign to execute on. During aggregation-workflow
            preparation (via the :class:`ParameterTiler`), the single-workflow
            commands must not be executed, so a different sign is used.
            Default is ``"$"``.
        """
        super().__init__(to_simple_type)
        self.parent_object = parent_object
        self.map = map_dict
        self.command_sign = command_sign

    def run(self, parameters: dict, curr_rootidx: int | None = None):
        """Scan a parameter set, executing commands before module execution.

        Parameters
        ----------
        parameters : dict
            The parameters for a module.
        curr_rootidx : int or None, optional
            If an int, the current module index.

        Returns
        -------
        dict
            The parameters with commands resolved.
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
                An optional third element is a default value used when the
                key is absent from self.map, so a single (reusable) spec can
                run whether or not a deployment defines that per-tile column:
                ("$$map", "min_locs", 10)
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
                # An optional third tuple element is a default, applied when
                # the key is missing so a reusable spec need not define the
                # per-tile column on every deployment.
                if len(t) > 2:
                    res = self.map.get(t[1], t[2])
                else:
                    res = self.map[t[1]]
                logger.debug(f"Mapping {t[1]}: {res}")
            elif cmd == f"{self.command_sign}index":
                idx = int(aritexp)
                # As for $map, an optional third element is a default used
                # when the key is missing or the sequence is too short.
                if len(t) > 2:
                    seq = self.map.get(t[1])
                    if seq is None or idx >= len(seq):
                        res = t[2]
                    else:
                        res = seq[idx]
                else:
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

    def get_prior_result(self, locator: str):
        """Retrieve a prior module's result by an attribute-chain locator.

        Parameters
        ----------
        locator : str
            Comma-separated chain of attributes locating the prior result,
            each obtainable via ``getattr``/item access starting from the
            parent object. E.g. ``"results, 02_load, sample_movie,
            sample_frame_idx"`` obtains
            ``self.results['02_load']['sample_movie']['sample_frame_idx']``.

        Returns
        -------
        object
            The last attribute in the chain.
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

    def get_previous_module_result(self, locator: str):
        """Retrieve a result from the immediately preceding module.

        A convenience wrapper around :meth:`get_prior_result` that
        automatically prepends the previous module to the locator.

        Parameters
        ----------
        locator : str
            Attribute chain locating the result within the module, e.g.
            ``"sample_movie, sample_frame_idx"``. Called from module 3, this
            obtains ``self.results['02_load']['sample_movie']
            ['sample_frame_idx']``.

        Returns
        -------
        object
            The last attribute in the chain.
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

    def get_attribute(self, root_att, att_name: str):
        """Get ``att_name`` from a dict (by key) or object (by attribute).

        Parameters
        ----------
        root_att : dict or object
            The container to read from.
        att_name : str
            The key or attribute name (whitespace is stripped).

        Returns
        -------
        object
            The retrieved value.

        Raises
        ------
        PriorResultError
            If ``root_att`` is None.
        """
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


def is_valid_expression(expression: str) -> bool:
    """Check whether a string is a valid numeric expression (e.g. ``*3.14``).

    Parameters
    ----------
    expression : str
        The arithmetic expression to validate.

    Returns
    -------
    bool
        Whether the expression matches the allowed numeric pattern.
    """
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
        """Initialize the evaluator.

        Parameters
        ----------
        parameter_command_executor : ParameterCommandExecutor or None, optional
            If provided, used to resolve parameter commands in condition
            values (e.g. ``$get_prior_result``).
        """
        self.parameter_command_executor = parameter_command_executor

    def evaluate(self, condition: dict) -> bool:
        """Evaluate a condition dictionary.

        Parameters
        ----------
        condition : dict
            Either a comparison condition with keys ``"left"`` (value or
            parameter command tuple), ``"operator"`` (one of ``>``, ``<``,
            ``>=``, ``<=``, ``==``, ``!=``) and ``"right"`` (value or
            parameter command tuple); or a logical condition with an ``"and"``
            key (list of conditions, all must hold) or ``"or"`` key (list of
            conditions, at least one must hold).

        Returns
        -------
        bool
            The result of the condition evaluation.

        Raises
        ------
        ValueError
            If the condition format is invalid or the operator is unsupported.
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
        """Resolve a value that may be a parameter command or a literal.

        Parameters
        ----------
        value : object
            The value to resolve. If it is a tuple starting with ``$``, it is
            resolved as a parameter command.

        Returns
        -------
        object
            The resolved value.
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
    """Raised when a ``$get_prior_result`` locator cannot be resolved."""


class ParameterTiler:
    """Multiply a set of parameters according to a tile command.

    Used e.g. to run multiple analogous analyses on different datasets that
    are then aggregated. Uses the :class:`ParameterCommandExecutor`, so the
    same commands apply.
    """

    def __init__(
        self,
        parent_object,
        tile_entries: dict,
        map_dict: dict = {},
        command_sign: str = "$$",
    ):
        """Initialize the tiler.

        Parameters
        ----------
        parent_object : object
            The object to execute commands on, e.g. the
            :class:`~picasso_workflow.workflow.WorkflowRunner` itself.
        tile_entries : dict
            One or more key-list pairs whose lists are of equal length. One
            parameter set is generated per list item; the keys are referenced
            in ``$map`` commands in the parameters passed to :meth:`run`. In
            addition to the mapped variables, ``tile_entries`` may contain
            ``'#tags'``, keyword tags for the list of parameter sets. For
            example::

                tile_entries = {'file_name': ['a1.tiff', 'a2.tiff']}
                parameters = {'load': {'filename': ('$map', 'file_name')}}
        map_dict : dict, optional
            A dictionary to map values via the ``$map`` command; the
            ``tile_entries`` are added to it.
        command_sign : str, optional
            The command sign to execute on. During aggregation-workflow
            preparation the single-workflow commands must not be executed, so
            a different sign is used. Default is ``"$$"``.
        """
        logger.debug("Initializeing ParameterTiler")
        self.tile_entries = tile_entries
        self.ntiles = len(list(tile_entries.values())[0])
        self.map_dict = map_dict
        self.parent_object = parent_object
        self.command_sign = command_sign

    def run(self, parameters: dict) -> tuple[list[dict], list[str]]:
        """Create the tiled set of parameters.

        Parameters
        ----------
        parameters : dict
            The parameters for a module.

        Returns
        -------
        result_parameters : list of dict
            The tiles of parameters.
        tags : list of str
            The value of the ``'#tags'`` entry (names to use), or a list of
            empty strings if ``'#tags'`` is absent.
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


def correct_path_separators(file_path: str) -> str:
    r"""Normalize path separators (``/`` or ``\``) for the current OS.

    Parameters
    ----------
    file_path : str
        Input file path with either separator.

    Returns
    -------
    str
        The file path with OS-appropriate separators.
    """
    path_components = re.split(r"[\\/]", file_path)
    file_path = os.path.join(*path_components)
    if path_components[0] == "":
        file_path = os.sep + file_path
    return file_path


def get_caller_name(levels_back: int = 1) -> str:
    """Get a function name from the traceback (the caller, or further back).

    Parameters
    ----------
    levels_back : int, optional
        Number of levels back in the traceback. Use 1 for the current
        function's name, 2 for the name of its caller, and so on. Default
        is 1.

    Returns
    -------
    str
        The function name.
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
    """Multiply every (same-dtype) column of a recarray by a factor.

    Parameters
    ----------
    ra : numpy.recarray
        The record array to scale (all named columns must share a dtype).
    factor : float
        The factor to multiply each column by.

    Returns
    -------
    numpy.recarray
        The scaled record array.

    Raises
    ------
    AttributeError
        If the columns do not all share the same dtype.
    """
    columns = [it[0] for it in ra.dtype.descr if it[0] != ""]

    column_dtypes = [it[1] for it in ra.dtype.descr if it[0] in columns]
    dt = column_dtypes[0]
    if not all([it == dt for it in column_dtypes]):
        raise AttributeError("Cannot multiply, not all dtypes are the same")
    for i, col in enumerate(columns):
        nda = ra[col].astype(dt)
        ra[col] = nda * factor
    return ra


def stripplot(data, positions, jitter, ax, color, alpha: float = 1):
    """Plot jittered data points onto an axis.

    A useful addition to a violin or box plot, especially for sparse data.

    Parameters
    ----------
    data : list of 1D array, or 2D array
        The example data points to plot for each position.
    positions : list of numeric
        The positions to plot the data at.
    jitter : float
        The amount of jitter to add along x, to separate the data points.
    ax : matplotlib.axes.Axes
        The axes to plot in.
    color : str
        The color to plot with (anything matplotlib understands).
    alpha : float, optional
        The transparency to plot with. Default is 1.
    """
    for pos, d in zip(positions, data):
        x = pos * np.ones(len(d))
        x += np.random.uniform(-jitter / 2, jitter / 2, size=len(d))
        ax.scatter(x, d, color=color, alpha=alpha)


def convert_filepath_for_machine(
    path: str, dest_machine: str | None = None
) -> str:
    """Convert a file path to the layout valid on the current/given machine.

    Uses the ``Drivepaths`` section of the picasso-workflow config. This is
    the entry point for analysis modules that need to adjust user-provided
    file paths to the machine the analysis runs on.
    :class:`metaworkflow.PathParser` is imported lazily to avoid a circular
    import (metaworkflow imports the workflow runners, which import
    analyse/util). Non-string or empty values, and paths not located under any
    known drive root, are returned unchanged (see
    :meth:`PathParser.convert_path`).

    Parameters
    ----------
    path : str
        The file path to convert.
    dest_machine : str or None, optional
        Target machine pattern key (e.g. ``'hpcl8XXX'``); None auto-detects
        the current machine via ``platform.node()``.

    Returns
    -------
    str
        The converted path, or the input value unchanged.
    """
    if not isinstance(path, str) or not path:
        return path
    from picasso_workflow.metaworkflow import PathParser

    return PathParser().convert_path(path, dest_machine)


def get_movie_groups(paths: list[str], extension: str) -> dict:
    """Group files by basename and index, supporting variable extensions.

    Parameters
    ----------
    paths : list of str
        Filenames to group.
    extension : str
        The extension to match (e.g. ``'.tif'`` or ``'.ome.tif'``).

    Returns
    -------
    dict
        Mapping from base name to a list of file paths sorted by index.
    """
    import re

    groups = {}
    if not paths:
        return groups

    ext_pattern = re.escape(extension)
    pattern = re.compile(rf"(.*?)(?:_(\d+))?{ext_pattern}$")

    match_infos = []
    for path in paths:
        match = pattern.match(path)
        if match:
            base, index = match.groups()
            match_infos.append(
                {
                    "path": path,
                    "base": base,
                    "index": int(index) if index else 0,
                }
            )

    # Grouping logic
    basenames = {m["base"] for m in match_infos}
    for base in basenames:
        group_items = [m for m in match_infos if m["base"] == base]
        # Sort by index
        group_items.sort(key=lambda x: x["index"])
        groups[base] = [item["path"] for item in group_items]

    return groups


def find_raw_movies(working_folder: str) -> dict:
    """Recursively find raw movie files (``.tif``, ``.ome.tif``, ``.nd2``).

    Parameters
    ----------
    working_folder : str
        Path to search.

    Returns
    -------
    dict
        Mapping from dataset name to a path or list of paths.
    """
    import os
    import fnmatch
    from pathlib import Path

    datasets = {}

    for root, dirs, files in os.walk(working_folder):
        p = Path(root)

        # Check what extensions exist here
        has_nd2 = list(p.glob("*.nd2"))
        has_ometif = list(p.glob("*.ome.tif"))
        # .tif check needs to exclude .ome.tif to be accurate
        has_tif = [
            f for f in p.glob("*.tif") if not f.name.endswith(".ome.tif")
        ]

        found_types = []
        if has_nd2:
            found_types.append(".nd2")
        if has_ometif:
            found_types.append(".ome.tif")
        if has_tif:
            found_types.append(".tif")

        if not found_types:
            continue

        # Priority: .nd2 > .ome.tif > .tif
        ext = found_types[0]

        if ext == ".nd2":
            nd2_files = sorted(fnmatch.filter(os.listdir(root), "*.nd2"))
            for f in nd2_files:
                stem = Path(f).stem
                # Use parent folder name as key if unique, otherwise include stem
                key = p.name if p.name else stem
                if len(nd2_files) > 1:
                    key = f"{p.name}_{stem}"
                datasets[key] = str(p / f)
        else:
            # tif or ome.tif
            tif_files = sorted([f.name for f in p.glob(f"*{ext}")])
            groups = get_movie_groups(tif_files, ext)
            for base, group_paths in groups.items():
                # Use directory name as key if it's the only group, otherwise use base
                if len(groups) == 1:
                    key = p.name
                else:
                    key = base

                full_paths = [str(p / fname) for fname in group_paths]
                datasets[key] = full_paths[0]

    # Natural sort the datasets by key
    import re

    def natsort_key(s):
        return [
            int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
        ]

    sorted_keys = sorted(datasets.keys(), key=natsort_key)
    return {k: datasets[k] for k in sorted_keys}

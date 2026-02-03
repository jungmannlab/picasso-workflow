#!/usr/bin/env python
"""
Module Name: analyse.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: This is the picasso interface of picasso-workflow
"""
from picasso import lib, io, localize, gausslq, postprocess, clusterer
from picasso import aim, spinna
from picasso_workflow.outpost_modules import g5m
from picasso import __version__ as picassoversion
from picasso import CONFIG as pCONFIG
import picasso
import copy
import gc

# import logging
from loguru import logger
import multiprocessing as mp
import os
import pickle
import platform
import random
import string
import sys
import time
from datetime import datetime
from functools import wraps

import matplotlib.pyplot as plt

# from tqdm import tqdm
import numpy as np
import pandas as pd
import psutil
import yaml
from matplotlib import cm
from memory_profiler import memory_usage
from picasso import CONFIG as pCONFIG
from picasso import __version__ as picassoversion
from picasso import (
    aim,
    clusterer,
    # g5m,
    gausslq,
    io,
    lib,
    localize,
    postprocess,
    spinna,
)
from picasso_workflow.outpost_modules import g5m
from scipy.ndimage import label
from scipy.spatial import KDTree, distance
from scipy.stats import kstest, norm, poisson

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    # Fallback implementation if scipy version issues
    def wasserstein_distance(u_values, v_values):
        """Manual implementation of 1D Wasserstein distance"""
        u_sorted = np.sort(u_values)
        v_sorted = np.sort(v_values)
        n = len(u_sorted)
        m = len(v_sorted)

        # Create cumulative distributions (not used in this implementation)
        # u_cdf = np.arange(1, n + 1) / n
        # v_cdf = np.arange(1, m + 1) / m

        # Merge and compute distance
        all_values = np.concatenate([u_sorted, v_sorted])
        all_values = np.sort(all_values)

        distance = 0
        for i in range(len(all_values) - 1):
            u_cum = np.searchsorted(u_sorted, all_values[i], side="right") / n
            v_cum = np.searchsorted(v_sorted, all_values[i], side="right") / m
            distance += abs(u_cum - v_cum) * (
                all_values[i + 1] - all_values[i]
            )

        return distance


from picasso_workflow import (
    outpost_modules,
    picasso_outpost,
    process_brightfield,
    util,
)
from picasso_workflow.outpost_modules import render
from picasso_workflow.ripleys_analysis import run_ripleysAnalysis

# logger = logging.getLogger(__name__)


def generate_random_code(length):
    letters = string.ascii_letters
    random_code = "".join(random.choices(letters, k=length))
    return random_code


def create_unique_filename(folder, fn, len_code=6):
    rcode = generate_random_code(len_code)
    fparts = os.path.split(fn)
    return os.path.join(folder, f"{fparts[0]}_{rcode}{fparts[1]}")


def get_memory_of(obj):
    """
    Recursively calculate the memory usage of an object, including its
    contents.
    Returns:
        size : int
            size in bytes
    """
    seen_ids = set()

    def inner(o):
        if id(o) in seen_ids:
            return 0
        seen_ids.add(id(o))
        size = sys.getsizeof(o)

        if isinstance(o, dict):
            size += sum(inner(k) + inner(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(inner(i) for i in o)

        return size

    return inner(obj)


def profile_resource_usage(method):
    """
        Decorator that profiles peak memory and CPU usage of a function.
        Returns results in a dictionary with values in GB and core count.
        Compatible with Linux, MacOS, and Windows.

        Intended to be used "outside" the module_wrapper, e.g.

    #    @profile_resource_usage
        @module_decorator
        def undrift(self, i, parameters, results):
            pass
    """

    @wraps(method)
    def wrapper(self, i, parameters, calling_module_dir=None, suffix=""):
        profiling_results = {"peak_memory_gb": 0.0, "peak_cpu_cores": 0.0}

        # Memory profiling function
        def memory_measure(self, i, parameters, calling_module_dir, suffix):
            start_cpu = time.process_time()
            start_real = time.time()
            _ = psutil.cpu_percent()
            # Call the actual function
            results = method(self, i, parameters, calling_module_dir, suffix)
            cpu_time = time.process_time() - start_cpu
            real_time = time.time() - start_real
            cpu_percent_end = psutil.cpu_percent()
            # Store CPU usage for access in wrapper
            wrapper.cpu_usage = cpu_time / max(real_time, 0.001)
            wrapper.mean_cpu_percent = cpu_percent_end
            return results

        try:
            # Measure memory usage
            mem_usage = memory_usage(
                proc=(
                    memory_measure,
                    [self, i, parameters, calling_module_dir, suffix],
                ),
                interval=0.5,
                timeout=None,
                max_usage=True,
                retval=True,
            )

            # Extract memory results
            peak_memory = mem_usage[0]  # First element is peak memory
            # Convert MiB to GiB
            profiling_results["peak_memory_gb"] = peak_memory / 1024.0

            # Calculate peak CPU usage in terms of cores
            total_cores = psutil.cpu_count(logical=True)  # total logical cores
            profiling_results["peak_cpu_cores"] = min(
                wrapper.cpu_usage * total_cores, total_cores
            )
            profiling_results["peak_cpu_usage"] = wrapper.cpu_usage
            profiling_results["mean_cpu_usage"] = wrapper.mean_cpu_percent
            pcpuc = profiling_results["peak_cpu_cores"]
            logger.debug(
                f"profiled cpu usage: {wrapper.cpu_usage}, "
                + f"total cores: {total_cores}, "
                + f"peak_cpu_cores: {pcpuc}"
            )

            # Get the results from the memory_measure function
            # Second element is the actual results
            parameters, results = mem_usage[1]

        except Exception as e:
            raise e
            # print(f"Profiling error: {e}")
            # return None, profiling_results  # Return None for results on
            # error

        results["peak_memory_gb"] = profiling_results["peak_memory_gb"]
        try:
            if self.locs is not None:
                locs_size = len(self.locs)
            else:
                channel_locs_size = sum(
                    [len(locs) for locs in self.channel_locs]
                )
                locs_size = max([locs_size, channel_locs_size])
            results["peak_memory_gb_per_locs"] = (
                profiling_results["peak_memory_gb"] / locs_size
            )
        except Exception:
            results["peak_memory_per_locs"] = 0
        results["peak_cpu_cores"] = profiling_results["peak_cpu_cores"]
        results["peak_cpu_usage"] = profiling_results["peak_cpu_usage"]
        results["mean_cpu_usage"] = profiling_results["mean_cpu_usage"]
        if self.channel_locs is not None:
            nlocs = sum([len(locs) for locs in self.channel_locs])
        elif self.locs is not None:
            nlocs = len(self.locs)
        else:
            nlocs = 0
        results["nlocs"] = nlocs

        return parameters, results

    wrapper.cpu_usage = 0.0  # Initialize CPU usage storage
    return wrapper


def module_decorator(method):
    def module_wrapper(
        self, i, parameters, calling_module_dir=None, suffix=""
    ):
        # create the results direcotry
        # method_name = get_caller_name(2)
        method_name = method.__name__

        if calling_module_dir is None:
            module_result_dir = os.path.join(
                self.results_folder, f"{i:02d}_" + method_name + suffix
            )
        else:
            module_result_dir = os.path.join(
                calling_module_dir, f"{i:02d}_" + method_name + suffix
            )
        os.makedirs(module_result_dir, exist_ok=True)

        results = {
            "folder": os.path.normpath(module_result_dir),
            "start time": datetime.now().strftime("%y-%m-%d %H:%M:%S"),
        }

        # call the module
        parameters, results = method(self, i, parameters, results)

        # post-actions
        # modules only need to specifically set an error.
        if results.get("success") is None:
            results["success"] = True
            # save locs if desired
            if parameters.get("save_locs") is True or self.analysis_config.get(
                "always_save"
            ):
                if hasattr(self, "locs") and self.locs is not None:
                    self._save_locs(
                        os.path.join(results["folder"], "locs.hdf5")
                    )
                if (
                    hasattr(self, "channel_locs")
                    and self.channel_locs is not None
                ):
                    self._save_datasets_agg(results["folder"])
        results["end time"] = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        td = datetime.strptime(
            results["end time"], "%y-%m-%d %H:%M:%S"
        ) - datetime.strptime(results["start time"], "%y-%m-%d %H:%M:%S")
        results["duration"] = td.total_seconds()
        # logger.debug(f"RESULTS: {results}")

        # close all figures potentially still open
        plt.close("all")
        return parameters, results

    return module_wrapper


class AutoPicasso(util.AbstractModuleCollection):
    """A class to automatically evaluate datasets.
    Each module that runs saves their results into a separate folder.
    """

    # for single-dataset analysis
    movie = None
    info = []
    identifications = None
    locs = None
    drift = None

    # for multi-dataset analysis (aggregation)
    channel_locs = None
    channel_info = None
    channel_tags = None

    def __init__(self, results_folder, analysis_config):
        """
        Args:
            results_folder : str
                the folder all analysis modules save their respective results
                to.
            analysis_config : dict
                the general configuration. necessary items:
                    gpufit_installed : bool
                        whether the machine has gpufit installed
                (potentially) optional items
                    camera_info : dict
                        as used by picasso. Only necessary if not loaded by
                        module load_dataset
                    always_save : bool
                        whether every module should end in saving the current
                        locs
        """
        self.results_folder = os.path.normpath(results_folder)
        self.analysis_config = analysis_config

    @property
    def info_mm_entry(self):
        try:
            infofirst = self.info[0]
        except IndexError:
            infofirst = self.channel_info[0][0]
        except Exception:
            raise AttributeError(
                "Cannot load camera name from info. Load data first."
            )
        return infofirst

    @property
    def camera_name(self):
        infofirst = self.info_mm_entry
        try:
            cam_name = infofirst["Camera"]
        except KeyError:
            logger.debug(
                "Cannot find camera entry. Probably this is a simulation."
            )
            cam_name = None
        return cam_name

    @property
    def em_wavelength(self):
        cam_name = self.camera_name
        filter_config = pCONFIG["Cameras"][cam_name].get("Channel Device")
        filterturret_label = filter_config["Name"]

        infofirst = self.info_mm_entry
        filter_label = infofirst["Micro-Manager Metadata"][filterturret_label]

        em_wl = filter_config["Emission Wavelengths"][filter_label]

        return em_wl

    @property
    def camera_info(self):
        if camera_info := self.analysis_config.get("camera_info"):
            return camera_info
        else:
            try:
                infofirst = self.info[0]
            except IndexError:
                infofirst = self.channel_info[0][0]
            except Exception:
                raise AttributeError(
                    "Cannot load pixelsize from info. Load data first."
                )
            try:
                cam_name = infofirst["Camera"]
            except KeyError:
                logger.debug(
                    "cannot find camera entry. Probably this is a simulation."
                )
                return {"Pixelsize": self.channel_info[0][1]["Pixelsize"]}
            if cam_config := pCONFIG.get("Cameras", {}).get(cam_name):
                # # find quantum efficiency
                # filter_name = cam_config.get("Channel Device", {}).get(
                #     "Name")
                # filter_used = self.info.get(filter_name)
                # emission_wavelength = cam_config.get(
                #     "Channel Device", {}).get(
                #     "Emission Wavelengths", {}).get(filter_used)
                # qe = cam_config.get("Quantum Efficiency", {}).get(
                #     emission_wavelength, 1)

                # find camera sensitivity
                sensitivity = cam_config.get("Sensitivity")
                if isinstance(sensitivity, dict):
                    # sensitivity starts being a dict, and ends as a value
                    cat_vals = ""
                    for category in cam_config.get("Sensitivity Categories"):
                        category_value = infofirst[
                            "Micro-Manager Metadata"
                        ].get(f"{cam_name}-{category}")
                        cat_vals += f"{category}: {category_value}; "
                        sensitivity = sensitivity.get(category_value, {})
                    if isinstance(sensitivity, dict):
                        raise PicassoConfigError(
                            f"""Could not find sensitivity value for camera
                            {cam_name} with category values {cat_vals} in
                            picasso CONFIG."""
                        )

                camera_info = {
                    "Baseline": cam_config["Baseline"],
                    "Gain": cam_config.get("Gain", 1),
                    "Sensitivity": sensitivity,
                    "Qe": 1,  # relevant are detected, not incident photons
                    "Pixelsize": cam_config["Pixelsize"],
                }
                self.analysis_config["camera_info"] = camera_info
                return camera_info
            else:
                raise PicassoConfigError(
                    f"Cannot load camera {cam_name} from picasso CONFIG."
                )
                raise AttributeError(
                    f"Cannot find camera '{cam_name}' in info."
                )

    @property
    def pixelsize(self):
        try:
            for infopart in self.info:
                pixelsize = infopart.get("Pixelsize")
                if pixelsize is not None:
                    return pixelsize
        except Exception:
            pass
        try:
            for sgl_info in self.channel_info:
                for infopart in sgl_info:
                    pixelsize = infopart.get("Pixelsize")
                    if pixelsize is not None:
                        return pixelsize
        except Exception:
            pass

        camera_info = self.camera_info
        pixelsize = camera_info["Pixelsize"]
        return pixelsize

    @property
    def frames(self):
        try:
            for infopart in self.info:
                frames = infopart.get("Frames")
                if frames is not None:
                    return frames
        except Exception:
            pass
        try:
            for sgl_info in self.channel_info:
                for infopart in sgl_info:
                    frames = infopart.get("Frames")
                    if frames is not None:
                        return frames
        except Exception:
            pass

        raise KeyError("Could not determine #Frames.")

    #    @profile_resource_usage
    @module_decorator
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
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def conditional_branch(self, i, parameters, results):
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
        # Get the parameter command executor from workflow runner if available
        pce = parameters.get("parameter_command_executor", None)

        # Create condition evaluator
        condition_evaluator = util.ConditionEvaluator(pce)

        # Evaluate the condition
        condition = parameters["condition"]
        condition_result = condition_evaluator.evaluate(condition)

        logger.info(f"Condition evaluated to: {condition_result}")
        results["condition_result"] = condition_result
        results["condition"] = condition

        # Determine which branch to execute
        if condition_result:
            branch_to_execute = parameters["if_true"]
            branch_name = "if_true"
        else:
            branch_to_execute = parameters["if_false"]
            branch_name = "if_false"

        results["branch_taken"] = branch_name
        logger.info(f"Executing branch: {branch_name}")

        # Execute sub-modules in the selected branch
        branch_results = {}
        flat_results = {}

        for sub_idx, (module_name, module_parameters) in enumerate(
            branch_to_execute
        ):
            logger.info(
                f"Executing sub-module {sub_idx}: {module_name} "
                f"in branch {branch_name}"
            )

            # Get the module method
            if not hasattr(self, module_name):
                raise AttributeError(
                    f"Module '{module_name}' not found in AutoPicasso"
                )

            module_method = getattr(self, module_name)

            # Create flat folder key for compatibility with existing modules
            flat_key = f"{i:02d}_{sub_idx:02d}_{module_name}"

            # Prepare module parameters with parameter command executor
            if pce is not None:
                # Temporarily update the current root index for parameter
                # resolution within sub-modules
                original_rootidx = pce.curr_rootidx
                # Sub-modules should reference previous modules correctly
                pce.curr_rootidx = i
                module_parameters_resolved = pce.run(
                    copy.deepcopy(module_parameters), curr_rootidx=i
                )
                pce.curr_rootidx = original_rootidx
            else:
                module_parameters_resolved = module_parameters

            # Execute the sub-module
            # The module will be called with the decorators already applied
            # We pass the results folder as calling_module_dir to create
            # a nested structure
            try:
                sub_params, sub_results = module_method(
                    sub_idx,
                    module_parameters_resolved,
                    calling_module_dir=results["folder"],
                    suffix="",
                )

                # Store results in branch structure
                sub_key = f"{sub_idx:02d}_{module_name}"
                branch_results[sub_key] = sub_results

                # Also store in flat structure for easier access
                flat_results[flat_key] = sub_results

                logger.info(
                    f"Sub-module {module_name} completed successfully. "
                    f"Result folder: {sub_results.get('folder', 'N/A')}"
                )

            except Exception as e:
                logger.error(
                    f"Error executing sub-module {module_name}: {str(e)}"
                )
                # Store error information
                sub_key = f"{sub_idx:02d}_{module_name}"
                branch_results[sub_key] = {
                    "success": False,
                    "error": str(e),
                }
                flat_results[flat_key] = branch_results[sub_key]
                # Propagate the error
                raise

        # Store branch results
        results["if_branch"] = branch_results
        results["branch_modules"] = flat_results

        # Store information about skipped branch for reference
        skipped_branch = "if_false" if condition_result else "if_true"
        results["skipped_branch"] = skipped_branch
        results["skipped_modules"] = [
            module_name
            for module_name, _ in parameters.get(skipped_branch, [])
        ]

        logger.info(
            f"Conditional branch completed. "
            f"Executed {len(branch_results)} sub-modules in {branch_name} branch."
        )

        return parameters, results

    ##########################################################################
    # Single dataset modules
    ##########################################################################

    #    @profile_resource_usage
    @module_decorator
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
        results["picasso version"] = picassoversion
        results["picasso-workflow version"] = "N/A"
        results["Architecture"] = platform.machine()
        results["OS"] = platform.system()
        results["host"] = platform.node()
        results["processor"] = platform.processor()
        results["CPU Frequency [MHz]"] = psutil.cpu_freq().current
        results["CPU cores"] = psutil.cpu_count()
        results["Memory total [GB]"] = psutil.virtual_memory().total // (
            1024**3
        )
        results["Memory available [GB]"] = (
            psutil.virtual_memory().available // (1024**3)
        )
        try:
            gpu_info = psutil.virtual_memory().gpu
        except AttributeError:
            gpu_info = None
        if gpu_info:
            results["GPU"] = gpu_info.name
            results["GPU memory"] = gpu_info.memory_total // (1024**3)
        else:
            results["GPU"] = "N/A"
            results["GPU memory [GB]"] = 0
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
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
        filepath_czi = parameters["filepath"]
        filename_raw = parameters.get("filename_raw")
        if filename_raw is None:
            filename_raw = os.path.split(
                (os.path.splitext(filepath_czi)[0] + ".raw")
            )[1]
        filepath_raw = os.path.join(results["folder"], filename_raw)
        picasso_outpost.convert_zeiss_file(
            filepath_czi, filepath_raw, parameters.get("info")
        )

        results["filepath_raw"] = filepath_raw
        results["filename_raw"] = filename_raw
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
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
        results["picasso version"] = picassoversion
        self.movie, self.info = io.load_movie(parameters["filename"])
        results["movie.shape"] = self.movie.shape

        if parameters.get("load_camera_info"):
            cam_name = self.info[0]["Camera"]
            if cam_config := pCONFIG.get("Cameras", {}).get(cam_name):
                # # find quantum efficiency
                # filter_name = cam_config.get("Channel Device", {}).get(
                #     "Name")
                # filter_used = self.info.get(filter_name)
                # emission_wavelength = cam_config.get(
                #     "Channel Device", {}).get(
                #     "Emission Wavelengths", {}).get(filter_used)
                # qe = cam_config.get("Quantum Efficiency", {}).get(
                #     emission_wavelength, 1)

                # find camera sensitivity
                sensitivity = cam_config.get("Sensitivity")
                # sensitivity starts being a dict, and ends as a value
                cat_vals = ""
                for category in cam_config.get("Sensitivity Categories"):
                    category_value = self.info[0].get(f"{cam_name}-{category}")
                    cat_vals += f"{category}: {category_value}; "
                    sensitivity = sensitivity.get(category_value, {})
                if isinstance(sensitivity, dict):
                    raise PicassoConfigError(
                        f"""Could not find sensitivity value for camera
                        {cam_name} with category values {cat_vals} in picasso
                        CONFIG."""
                    )

                camera_info = {
                    "Baseline": cam_config["Baseline"],
                    "Gain": cam_config.get("Gain", 1),
                    "Sensitivity": sensitivity,
                    "qe": 1,  # relevant are detected, not incident photons
                    "pixelsize": cam_config["Pixelsize"],
                }
                self.analysis_config["camera_info"] = camera_info
            else:
                raise PicassoConfigError(
                    f"Cannot load camera {cam_name} from picasso CONFIG."
                )

        # check this is actually a DNA-PAINT analysable movie
        if results["movie.shape"][0] < 10:
            results["sample_movie"] = self.movie
            raise AutoPicassoError(
                "Movie loaded has less than 10 frames."
                + " Unsuitable for DNA-PAINT."
            )
            # results["success"] = False

        # create sample movie
        if (samplemov_pars := parameters.get("sample_movie")) is not None:
            samplemov_pars["filename"] = os.path.join(
                results["folder"], samplemov_pars["filename"]
            )
            res = self._create_sample_movie(**samplemov_pars)
            results["sample_movie"] = res

        return parameters, results

    def _create_sample_movie(
        self,
        filename,
        start_sample_pct=0,
        n_sample=30,
        min_quantile=0,
        max_quantile=0.9998,
        fps=1,
    ):
        """Create a subsampled movie of the movie loaded. The movie is saved
        to disk and referenced by filename.
        Args:
            filename : str
                the file name to save the subsamled movie as (.mp4)
            start_sample_pct : float
                percentage of movie frames from which to start sampling
                This can be useful if the first frames are different due to
                residual autofluorescence or such.
            rest: as in save_movie
        """
        results = {}
        if len(self.movie) < n_sample:
            n_sample = len(self.movie)

        start_idx = int(start_sample_pct / 100 * len(self.movie))
        len_subsample = len(self.movie) - start_idx
        dn = int(len_subsample / (n_sample - 1))
        frame_numbers = np.arange(start_idx, len_subsample, dn)
        results["sample_frame_idx"] = frame_numbers

        subsampled_frames = np.array([self.movie[i] for i in frame_numbers])
        process_brightfield.save_movie(
            filename,
            subsampled_frames,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
            fps=fps,
        )
        results["filename"] = filename
        return results

    #    @profile_resource_usage
    @module_decorator
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
        results["picasso version"] = picassoversion
        self.locs, self.info = io.load_locs(parameters["filename"])
        results["nlocs"] = len(self.locs)

        return parameters, results

    def _auto_min_netgrad(
        self,
        box_size,
        frame_numbers,
        filename=None,
        start_ng=-3000,
        zscore=5,
        bins=None,
    ):
        """Calculate histograms of the net gradient at local maxima of n
        frames. For the automatic calculation of a threshold net_gradient for
        localizations, assume the background (of random local maxima without a
        localization signal) to be Gaussian distributed. Assume the background
        peak in the histogram is the highest value. The threshold net_gradient
        will be determined as zscore bacground standard deviations above the
        peak.

        Args:
            box_size : int
                the box size for evaluation
            frame_numbers : list
                the frame indexes to analyze
            filename : str, default None
                the file name of the plot to be created
                no plot generated if None
            start_ng : float
                the minimum net gradient to accept for the histogram.
                this should be below zero, to capture all net gradient
                values that exist in the data
            zscore : float
                the number of sigmas above the background net gradient peak
                to set as the estimated min net gradient threshold
            bins : None, int or array
                specify the bins of the histogram
        Returns:
            results : dict; with
                filename : string
                    filename of the generated plot
                estd_net_grad : float
                    the estimated min net gradient
        """
        results = {}
        identifications = []

        if isinstance(frame_numbers, int):
            n_sample = frame_numbers
            start_sample_pct = 0
            if len(self.movie) < n_sample:
                n_sample = len(self.movie)

            start_idx = int(start_sample_pct / 100 * len(self.movie))
            len_subsample = len(self.movie) - start_idx
            dn = int(len_subsample / (n_sample - 1))
            frame_numbers = np.arange(start_idx, len_subsample, dn)

        for frame_number in frame_numbers:
            identifications.append(
                localize.identify_by_frame_number(
                    self.movie, start_ng, box_size, frame_number
                )
            )
        # id_list = identifications
        # identifications = np.hstack(identifications).view(np.recarray)
        # identifications.sort(kind="mergesort", order="frame")
        identifications = pd.concat(identifications)
        identifications.sort_values("frame", kind="mergesort")

        # calculate histogram
        if bins is None:
            hi = np.quantile(identifications["net_gradient"], 0.9995)
            bins = np.linspace(start_ng, hi, num=500)
        hist, edges = np.histogram(
            identifications["net_gradient"], bins=bins, density=True
        )

        # find the background peak, assume it to be Gaussian and the
        # highest peak in the histogram: find max and FWHM
        # FWHM as the most robust detection for peak width
        # only use the lower half for FWHM calculation, as the higher
        # tail is confounded by non-background spots
        bkg_peak_height, bkg_peak_pos = np.max(hist), np.argmax(hist)
        logger.debug(f"bkg_peak_height: {bkg_peak_height}")
        logger.debug(f"bkg_peak_pos: {bkg_peak_pos}")
        bkg_half_lo = np.argsort(
            np.abs(hist[:bkg_peak_pos] - bkg_peak_height / 2)
        )
        logger.debug(f"bkg_half_lo: {bkg_half_lo}")
        bkg_fwhm = 2 * np.abs(bkg_peak_pos - bkg_half_lo[0])
        bkg_sigma = bkg_fwhm / np.sqrt(4 * np.log(2))
        # threshold at zscore * bkg_sigma
        ng_est_idx = int(zscore * bkg_sigma) + bkg_peak_pos
        if ng_est_idx >= len(edges):
            ng_est_idx = len(edges) - 1
        results["estd_net_grad"] = edges[ng_est_idx]
        bkg_peak = edges[bkg_peak_pos]
        lo_idx = int(bkg_peak_pos - bkg_sigma)
        if lo_idx < 0:
            lo_idx = 0
        bkg_sigma = bkg_peak - edges[lo_idx]

        # plot results
        if filename:
            fig, ax = plt.subplots(nrows=2)
            ax[0].plot(edges[:-1], hist, color="b", label="combined histogram")
            # for i, frame_number in enumerate(frame_numbers):
            #     hi, ed = np.histogram(
            #         id_list[i]['net_gradient'], bins=bins, density=True)
            #     ax.plot(ed[:-1], hi, color='gray')
            ylims = ax[0].get_ylim()
            ax[0].set_title("Net Gradient histogram of subsampled frames")
            ax[0].set_xlabel("net gradient")
            ax[0].set_yscale("log")
            ax[0].plot(
                [results["estd_net_grad"], results["estd_net_grad"]],
                ylims,
                color="r",
                label="estimated min net gradient: {:.0f}".format(
                    results["estd_net_grad"]
                ),
            )
            ax[0].plot(
                [edges[bkg_peak_pos], edges[bkg_peak_pos]],
                ylims,
                color="gray",
                label=f"background: {bkg_peak:.0f}+/-{bkg_sigma:.0f}",
            )
            ax[0].legend()
            # plt.show()

            sample_spots, ng_start, ng_end = self._draw_sample_spots(
                identifications, results["estd_net_grad"], box_size
            )
            ax[1].imshow(sample_spots, cmap="gray", interpolation="nearest")
            ax[1].grid(visible=False)
            ax[1].tick_params(bottom=False, left=False)
            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_title(
                "spots with net_gradient " + f"{ng_start:.0f} to {ng_end:.0f}"
            )

            results["filename"] = filename
            plt.tight_layout()
            fig.savefig(results["filename"])
        return results

    def _draw_sample_spots(
        self,
        identifications,
        estd_net_grad,
        box_size,
        sample_spots_rows=4,
        sample_spots_cols=12,
    ):
        """Pull up example spots at the threshold net_grad for
        visualizing the automatically found min net gradient.
        Args:
            identifications : np recarray
                identifications from subsampled frames with
                very low min net gradient.
            estd_net_grad : float
                the estimated min net gradient
            box_size : uneven int
                the box size to display
            sample_spots_rows : int
                the number of rows of spots to display
            sample_spots_cols : int
                the number of cols of spots to display
        Returns:
            canvas : 2D array
                the canvas with spots to display
            ng_start : float
                the lowest net gradient shown (upper left spot)
            ng_end : float
                the highest net gradient shown (lower right spot)
        """
        n_spots = sample_spots_cols * sample_spots_rows
        sample_idxs = np.argsort(
            np.abs(identifications["net_gradient"] - estd_net_grad)
        )[:n_spots]
        # sample_identifications = identifications[sample_idxs]
        # sample_identifications = sample_identifications[
        #     np.argsort(sample_identifications["net_gradient"])
        # ]
        sample_identifications = identifications.iloc[sample_idxs]
        sample_identifications = sample_identifications.sort_values(
            "net_gradient"
        ).reset_index(drop=True)

        # sample_spots = localize.get_spots(
        sample_spots = picasso_outpost.get_spots(
            self.movie,
            sample_identifications,
            box_size,
            self.camera_info,
        )
        ng_start = np.min(sample_identifications["net_gradient"])
        ng_end = np.max(sample_identifications["net_gradient"])

        border_width = 2
        canvas_size = (
            box_size * sample_spots_rows
            + border_width * (sample_spots_rows - 1),
            box_size * sample_spots_cols
            + border_width * (sample_spots_cols - 1),
        )

        canvas = np.zeros(canvas_size, dtype=np.uint8)
        for i, spot in enumerate(sample_spots):
            ix, iy = i // sample_spots_cols, i % sample_spots_cols
            pix = ix * (box_size + border_width)
            piy = iy * (box_size + border_width)
            # logger.debug(f"drawing spot {i} at ({pix}, {piy}: {str(spot)}")
            canvas[pix : pix + box_size, piy : piy + box_size] = (
                picasso_outpost.normalize_spot(spot)
            )
        return canvas, ng_start, ng_end

    #    @profile_resource_usage
    @module_decorator
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
        # auto-detect net grad if required:
        if (autograd_pars := parameters.get("auto_netgrad")) is not None:
            if "filename" in autograd_pars.keys() and autograd_pars["filename"]:
                autograd_pars["filename"] = os.path.join(
                    results["folder"], autograd_pars["filename"]
                )
            else:
                autograd_pars["filename"] = os.path.join(
                    results["folder"], "auto_identification.png")

            potential_pars = [
                "box_size",
                "frame_numbers",
                "filename",
                "start_ng",
                "zscore",
                "bins",
            ]
            pars_to_pass = {
                k: autograd_pars[k]
                for k in potential_pars
                if k in autograd_pars.keys()
            }
            res = self._auto_min_netgrad(**pars_to_pass)
            results["auto_netgrad"] = res
            parameters["min_gradient"] = res["estd_net_grad"]

        curr, futures = localize.identify_async(
            self.movie,
            parameters["min_gradient"],
            parameters["box_size"],
            roi=None,
        )
        self.identifications = localize.identifications_from_futures(futures)
        results["num_identifications"] = len(self.identifications)

        if (pars := parameters.get("ids_vs_frame")) is not None:
            if "filename" in pars.keys() and pars["filename"]:
                filename = pars["filename"]
            else:
                filename = "id_vs_frame.png"
            pars["filename"] = os.path.join(results["folder"], filename)
            results["ids_vs_frame"] = self._plot_ids_vs_frame(**pars)

        # add info
        new_info = {
            "Generated by": "picasso-workflow : identify",
            "Box Size": parameters["box_size"],
            "Min. Net Gradient": float(parameters["min_gradient"]),
            # "Width": ,
            # "Height": ,
            # "Frames": len(self.movie),
            # "Data Type": ,
            # "parameters": parameters,
        }
        self.info = self.info + [new_info]

        return parameters, results

    def _plot_ids_vs_frame(self, filename):
        """Plot identifications vs frame index"""
        results = {}
        frames = np.arange(len(self.movie))
        bins = np.arange(len(self.movie) + 1) - 0.5
        locs, _ = np.histogram(self.identifications["frame"], bins=bins)
        fig, ax = plt.subplots()
        ax.plot(frames, locs)
        ax.set_xlabel("frame")
        ax.set_ylabel("number of identifications")
        results["filename"] = filename
        fig.savefig(results["filename"])
        plt.close(fig)
        return results

    #    @profile_resource_usage
    @module_decorator
    def localize(self, i, parameters, results):
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
        em = self.camera_info["Gain"] > 1
        spots = localize.get_spots(
            self.movie,
            self.identifications,
            parameters["box_size"],
            self.camera_info,
        )
        if self.analysis_config["gpufit_installed"]:
            theta = gausslq.fit_spots_gpufit(spots)
            self.locs = gausslq.locs_from_fits_gpufit(
                self.identifications, theta, parameters["box_size"], em
            )
        else:
            if parameters["fit_parallel"]:
                # theta = gausslq.fit_spots_parallel(spots, asynch=False)
                fs = gausslq.fit_spots_parallel(spots, asynch=True)
                # n_tasks = len(fs)
                # with tqdm(total=n_tasks, unit="task") as progress_bar:
                #     for f in _futures.as_completed(fs):
                #         progress_bar.update()
                theta = gausslq.fits_from_futures(fs)
                em = self.camera_info["Gain"] > 1
                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters["box_size"], em
                )
            else:
                theta = np.empty((len(spots), 6), dtype=np.float32)
                theta.fill(np.nan)
                # for i in tqdm(range(len(spots))):
                for i in range(len(spots)):
                    theta[i] = gausslq.fit_spot(spots[i])

                self.locs = gausslq.locs_from_fits(
                    self.identifications, theta, parameters["box_size"], em
                )

        if pars := parameters.get("locs_vs_frame"):
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            results["locs_vs_frame"] = self._plot_locs_vs_frame(
                pars["filename"]
            )

        # add info
        localize_info = {
            "Generated by": "Picasso Localize",
            "ROI": None,
            "Box Size": int(parameters["box_size"]),
            # "Min. Net Gradient": min_net_gradient,
            # "Convergence Criterion": convergence,
            # "Max. Iterations": max_iterations,
            "Pixelsize": float(self.pixelsize),
            "Fit method": "gausslq",
            "Wrapped by": "picasso-workflow : localize",
            # "parameters": parameters,
        }
        self.info = self.info + [localize_info]

        # save locs
        if pars := parameters.get("save_locs"):
            if "filename" in pars.keys():
                pars["filename"] = os.path.join(
                    results["folder"], pars["filename"]
                )
            self._save_locs(pars["filename"])

        results["locs_columns"] = list(self.locs.columns)
        return parameters, results

    @module_decorator
    def zfit(self, i, parameters, results):
        """Fits z positions to previously localized spots.

        Args:
            i : int
                the module index in the protocol
            parameters : dict
                necessary items:
                    magnification_factor : float
                        the magnification factor for z calibration
                optional items:
                    fp_calibration : str
                        filepath to the 3D calibration yaml file
                        if not given
                    save_locs : dict
                        if saving localizations is requested.
                        Items correpsond to arguments of save_locs
            results : dict
                the results dict, created by the module_decorator
                    fp_calibration : str
                        the filepath to the calibration that has been used.
                        Either as given, or as loaded from config
                    fp_calibration_fig : str
                        filepath to the calbiration graph, if loaded
        Returns:
            parameters : dict
                as input, potentially changed values, for consistency
            results : dict
                the analysis results
        """
        import shutil
        from picasso import zfit

        # fp_cfg = os.path.join(results["folder"], "config.yaml")
        # with open(fp_cfg, "w") as config_file:
        #     yaml.dump(pCONFIG, config_file)

        path = parameters.get("fp_calibration")
        if path is None or path == "":
            # fp_calib_lam = CONFIG["z-calibrations"].get(camera)
            # if fp_calib_lam is not None:
            #     em_combo = self.emission_combos[camera]
            #     wavelength = em_combo.currentText()
            #     fp_calib = fp_calib_lam.get(wavelength)
            camera = self.camera_name
            em_wl = self.em_wavelength
            path = pCONFIG.get("z-calibrations").get(camera).get(em_wl)

        results["fp_calibration"] = path

        # try loading the calibration graphs, for documentation
        fp_fig_src, _ = os.path.splitext(path)
        fp_fig_src += ".png"
        if os.path.exists(fp_fig_src):
            _, fn_fig = os.path.split(fp_fig_src)
            fp_fig_dst = os.path.join(results["folder"], fn_fig)
            shutil.copyfile(fp_fig_src, fp_fig_dst)
            results["fp_calibration_fig"] = fp_fig_dst

        magnification_factor = parameters["magnification_factor"]

        with open(path, "r") as f:
            z_calibration = yaml.full_load(f)

        N = len(self.locs)
        fs = zfit.fit_z_parallel(
            self.locs,
            self.info,
            z_calibration,
            magnification_factor,
            filter=0,
            asynch=True,
        )
        n_tasks = len(fs)
        while lib.n_futures_done(fs) < n_tasks:
            time.sleep(0.2)
        self.locs = zfit.locs_from_futures(fs, filter=0)

        # generate a z coordinate histogram
        fig, ax = plt.subplots()
        ax.hist(self.locs["z"], bins=50)
        ax.set_xlabel("z [nm]")
        ax.set_title("Histogram of z coordinate distribution")
        fp_fig = os.path.join(results["folder"], "z_histogram.png")
        fig.savefig(fp_fig)
        results["fp_fig_zhist"] = fp_fig

        # add info
        zfit_info = {
            "Generated by": "Picasso zfit",
            "Calibration filepath": results["fp_calibration"],
            "magnification factor": f"{magnification_factor}",
            "Wrapped by": "picasso-workflow : zfit",
            # "parameters": parameters,
        }
        self.info = self.info + [zfit_info]

        return parameters, results

    def _plot_locs_vs_frame(self, filename):
        results = {}
        frames = np.arange(len(self.movie))
        # bins = np.arange(len(self.movie) + 1) - .5

        df_locs = pd.DataFrame(self.locs)
        gbframe = df_locs.groupby("frame")
        photons_mean = gbframe["photons"].mean()
        photons_std = gbframe["photons"].std()
        sx_mean = gbframe["sx"].mean()
        sx_std = gbframe["sx"].std()
        sy_mean = gbframe["sy"].mean()
        sy_std = gbframe["sy"].std()

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(frames, photons_mean, color="b", label="mean photons")
        xhull = np.concatenate([frames, frames[::-1]])
        yhull = np.concatenate(
            [
                photons_mean + photons_std,
                photons_mean[::-1] - photons_std[::-1],
            ]
        )
        ax[0].fill_between(
            xhull, yhull, color="b", alpha=0.2, label="std photons"
        )
        ax[0].set_xlabel("frame")
        ax[0].set_ylabel("photons")
        ax[0].legend()
        ax[1].plot(frames, sx_mean, color="c", label="mean sx")
        yhull = np.concatenate(
            [sx_mean + sx_std, sx_mean[::-1] - sx_std[::-1]]
        )
        ax[1].fill_between(xhull, yhull, color="c", alpha=0.2, label="std sx")
        ax[1].plot(frames, sy_mean, color="m", label="mean sy")
        yhull = np.concatenate(
            [sy_mean + sy_std, sy_mean[::-1] - sy_std[::-1]]
        )
        ax[1].fill_between(xhull, yhull, color="m", alpha=0.2, label="std sy")
        ax[1].set_xlabel("frame")
        ax[1].set_ylabel("width")
        ax[1].legend()
        results["filename"] = filename
        fig.savefig(results["filename"])
        plt.close(fig)
        return results

    @module_decorator
    def load_picassoconfig(self, i, parameters, results):
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
        global pCONFIG

        with open(parameters["fp_config"], "r") as config_file:
            new_config = yaml.full_load(config_file)

        # if new_config is not None:
        picasso.CONFIG = new_config
        pCONFIG = new_config
        print(new_config)

        fp_cfg = os.path.join(results["folder"], "config.yaml")
        with open(fp_cfg, "w") as config_file:
            yaml.dump(pCONFIG, config_file)
        results["fp_config"] = fp_cfg

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def export_brightfield(self, i, parameters, results):
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
        fps_in = parameters["filepath"]
        if isinstance(fps_in, str):
            fps_in = [fps_in]
        if isinstance(fps_in, list):
            d = {}
            for i, fp in enumerate(fps_in):
                d[str(i)] = fp
            fps_in = d
        fps_out = {}
        for lbl, fp in fps_in.items():
            mov, _ = io.load_movie(fp)
            frame = mov[0]
            fn = os.path.split(fp)[1]
            fn = os.path.splitext(fn)[0] + ".png"
            fp = os.path.join(results["folder"], fn)
            fps_out[lbl] = fp
            min_quantile = parameters.get("min_quantile", 0)
            max_quantile = parameters.get("max_quantile", 1)
            process_brightfield.save_frame(
                fp, frame, min_quantile, max_quantile
            )
        results["labeled filepaths"] = fps_out
        results["success"] = True
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def render(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        rcode = generate_random_code(6)

        if self.channel_locs is not None:
            render_locs = self.channel_locs
            x_mean = np.mean([np.mean(lcs["x"]) for lcs in self.channel_locs])
            y_mean = np.mean([np.mean(lcs["y"]) for lcs in self.channel_locs])
        else:
            render_locs = self.locs
            x_mean = np.mean(self.locs["x"])
            y_mean = np.mean(self.locs["y"])

        # render whole field of view
        fullfov_pixelsize = parameters.get("fullfov_pixelsize", pixelsize)
        results["fp_scene_fullfov"] = os.path.join(
            results["folder"], f"locs_fullfov_{rcode}.png"
        )
        render.plot_scene(
            render_locs,
            fullfov_pixelsize,
            pixelsize,
            fp=results["fp_scene_fullfov"],
        )

        # render zoom into the center of mass
        if parameters.get("ctrmass_fov_nm"):
            ctrmass_pixelsize = parameters.get("ctrmass_pixelsize", pixelsize)
            fov_half = parameters.get("ctrmass_fov_nm") / 2
            x_min = x_mean - fov_half / pixelsize
            x_max = x_mean + fov_half / pixelsize
            y_min = y_mean - fov_half / pixelsize
            y_max = y_mean + fov_half / pixelsize

            render_kwargs = {
                "oversampling": pixelsize / ctrmass_pixelsize,
                "viewport": [(y_min, x_min), (y_max, x_max)],
                "blur_method": parameters.get("ctrmass_blur_method"),
                "min_blur_width": parameters.get("ctrmass_min_blur_width", 0),
                "ang": parameters.get("ctrmass_ang"),
            }
            results["fp_scene_ctrmass"] = os.path.join(
                results["folder"], f"locs_ctrmass_{rcode}.png"
            )
            render.plot_scene(
                render_locs,
                ctrmass_pixelsize,
                pixelsize,
                fp=results["fp_scene_ctrmass"],
                render_kwargs=render_kwargs,
                # x_offset=x_min * pixelsize,
                # y_offset=y_min * pixelsize,
            )
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def undrift_aim(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        progress = parameters.get("progress", None)
        # progress = lib.MockProgress().set_value,  # parameters.get("progress", None)

        # dirty debug: picasso.aim.aim expects the existence of
        # info[1]["Pixelsize"]
        self.info[1]["Pixelsize"] = pixelsize

        self.locs, self.info, self.drift = aim.aim(
            self.locs,
            self.info,
            segmentation=parameters["segmentation"],
            intersect_d=parameters["intersect_d"] / pixelsize,
            roi_r=parameters["roi_r"] / pixelsize,
            progress=progress,
        )

        results["success"] = True
        results["fp_driftfile"] = create_unique_filename(
            results["folder"], "drift.txt"
        )
        np.savetxt(results["fp_driftfile"], self.drift, delimiter=",")
        results["fp_fig"] = (
            os.path.splitext(results["fp_driftfile"])[0] + ".png"
        )
        self._plot_drift(
            results["fp_fig"], parameters["dimensions"], pixelsize
        )

        # # save locs
        # if pars := parameters.get("save_locs"):
        #     if "filename" in pars.keys():
        #         pars["filename"] = os.path.join(
        #             results["folder"], pars["filename"]
        #         )
        #     self._save_locs(pars["filename"])

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def undrift_rcc(self, i, parameters, results):
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
        pixelsize = self.pixelsize

        seg_init = parameters["segmentation"]
        for i in range(parameters.get("max_iter_segmentations", 3)):
            # if the segmentation is too low, the process raises an error
            # adaptively increase the value.
            try:
                self.drift, self.locs = postprocess.undrift(
                    self.locs,
                    self.info,
                    segmentation=parameters["segmentation"],
                    display=False,
                    segmentation_callback=lib.MockProgress().set_value,
                    rcc_callback=lib.MockProgress().set_value,
                )
                results["success"] = True
                break
            except ValueError:
                parameters["segmentation"] = 2 * parameters["segmentation"]
                logger.debug(
                    f"""RCC with segmentation {parameters["segmentation"]}
                    raised an error. Doubling."""
                )
                results[
                    "message"
                ] = f"""Initial Segmentation of {seg_init}
                    was too low."""
        else:  # did not work until the end
            logger.error(
                f"""RCC failed up to segmentation {parameters["segmentation"]}.
                Aborting."""
            )
            max_segmentation = parameters["segmentation"]
            # initial segmentation
            parameters["segmentation"] = int(
                parameters["segmentation"]
                / 2 ** parameters["max_iter_segmentations"]
            )
            results[
                "message"
            ] = f"""
                    Undrifting did not work in
                    {parameters['max_iter_segmentations']} iterations
                    up to a segmentation of {max_segmentation}."""
            results["success"] = False

        parameters["dimensions"] = ["x", "y"]

        if parameters.get("filename"):
            results["filepath_driftfile"] = os.path.join(
                results["folder"], parameters["filename"]
            )
            np.savetxt(
                results["filepath_driftfile"], self.drift, delimiter=","
            )
            results["filepath_plot"] = (
                os.path.splitext(results["filepath_driftfile"])[0] + ".png"
            )
            self._plot_drift(
                results["filepath_plot"],
                parameters["dimensions"],
                pixelsize,
                method="RCC",
            )

        # add info
        new_info = {
            "Generated by": "picasso-workflow : undrift_rcc",
            "parameters": parameters,
        }
        self.info = self.info + [new_info]

        # # save locs
        # if pars := parameters.get("save_locs"):
        #     if "filename" in pars.keys():
        #         pars["filename"] = os.path.join(
        #             results["folder"], pars["filename"]
        #         )
        #     self._save_locs(pars["filename"])

        return parameters, results

    # Multiprocessing helper functions for undrift_rsso
    @staticmethod
    def _process_drift_block(args):
        """Stateless helper function for processing drift blocks in parallel"""
        (
            locs_data,
            frames,
            block_start,
            block_end,
            toff_block_size,
            max_shift,
            min_locs_per_block,
            save_all_rsso_plots,
            plot_dir,
        ) = args

        from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

        if block_start == 0:
            return None  # Skip first block (reference)

        # Get frame ranges for current and previous blocks
        current_min_frame = frames[block_start]
        current_max_frame = frames[block_end - 1]
        prev_min_frame = frames[max(0, block_start - toff_block_size)]
        prev_max_frame = (
            frames[block_start - 1] if block_start > 0 else frames[0]
        )

        # Filter localizations efficiently
        current_block_locs = locs_data[
            (locs_data["frame"] >= current_min_frame)
            & (locs_data["frame"] <= current_max_frame)
        ]
        prev_block_locs = locs_data[
            (locs_data["frame"] >= prev_min_frame)
            & (locs_data["frame"] <= prev_max_frame)
        ]

        # Estimate drift between blocks
        if (
            len(prev_block_locs) < min_locs_per_block
            or len(current_block_locs) < min_locs_per_block
        ):
            return (block_start, block_end, None, None, 0, None, None)

        shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
            prev_block_locs,
            current_block_locs,
            max_shift,
            plot_histogram=save_all_rsso_plots,
            plot_dir=plot_dir,
        )
        logger.debug(
            f"""
            coarse undrift of block {current_min_frame} to {current_max_frame}
            shift x: {shift_x}; shift y: {shift_y};
            uncertainty info: {uncertainty_info}
            """
        )

        if shift_x is not None and shift_y is not None:
            quality = len(prev_block_locs) + len(current_block_locs)
            uncertainty_x = (
                uncertainty_info["sigma_x"]
                if uncertainty_info["fit_successful"]
                else uncertainty_info["shift_x_error"]
            )
            uncertainty_y = (
                uncertainty_info["sigma_y"]
                if uncertainty_info["fit_successful"]
                else uncertainty_info["shift_y_error"]
            )
            return (
                block_start,
                block_end,
                shift_x,
                shift_y,
                quality,
                uncertainty_x,
                uncertainty_y,
            )
        else:
            return (block_start, block_end, None, None, 0, None, None)

    def _adaptive_drift_correction(
        self,
        frames,
        max_shift,
        min_locs_per_frame,
        min_window_size,
        max_window_size,
        confidence_threshold,
        outlier_detection_enabled,
        outlier_z_threshold,
        change_point_sensitivity,
        min_signal_to_noise,
        n_processes,
        save_all_rsso_plots,
        plot_dir,
        use_spline_interpolation,
    ):
        """Adaptive drift correction with variable window sizes based on local statistics

        This method uses time series cross validation principles to adaptively
        choose window sizes, with larger windows in stable regions and smaller
        windows in rapidly changing regions.

        Returns:
            tuple: (drift_x_fine, drift_y_fine, uncertainty_x_fine,
                   uncertainty_y_fine, drift_quality)
        """
        from scipy import stats
        from scipy.signal import find_peaks

        n_frames = len(frames)

        # Initialize drift arrays
        drift_x_fine = np.zeros(n_frames)
        drift_y_fine = np.zeros(n_frames)
        uncertainty_x_fine = np.zeros(n_frames)
        uncertainty_y_fine = np.zeros(n_frames)
        drift_quality = np.zeros(n_frames)

        logger.debug(
            f"Adaptive drift correction: {n_frames} frames, "
            f"window size range: {min_window_size}-{max_window_size}"
        )

        # Phase 1: Initial coarse estimation to detect change points
        coarse_estimates = self._get_coarse_drift_estimates(
            frames, max_shift, min_locs_per_frame, max_window_size
        )

        # Phase 2: Temporal outlier filtering
        if outlier_detection_enabled:
            coarse_estimates_filtered = self._filter_temporal_outliers(
                coarse_estimates, outlier_z_threshold
            )
        else:
            coarse_estimates_filtered = coarse_estimates

        # Phase 3: Change point detection (on filtered data)
        change_points = self._detect_change_points(
            coarse_estimates_filtered, change_point_sensitivity
        )

        logger.debug(
            f"Detected {len(change_points)} change points: {change_points}"
        )

        # Phase 4: Adaptive window sizing
        optimal_windows = self._calculate_adaptive_windows(
            frames,
            change_points,
            coarse_estimates_filtered,
            min_window_size,
            max_window_size,
            confidence_threshold,
        )

        # Phase 5: High-precision drift estimation with optimal windows
        (
            drift_x_fine,
            drift_y_fine,
            uncertainty_x_fine,
            uncertainty_y_fine,
            drift_quality,
        ) = self._estimate_drift_with_adaptive_windows(
            frames,
            optimal_windows,
            max_shift,
            min_locs_per_frame,
            save_all_rsso_plots,
            plot_dir,
            outlier_detection_enabled,
            min_signal_to_noise,
        )

        # Log performance statistics
        avg_window_size = np.mean(optimal_windows[1:])  # Exclude first frame
        avg_uncertainty_x = np.mean(uncertainty_x_fine[1:])
        avg_uncertainty_y = np.mean(uncertainty_y_fine[1:])
        avg_quality = np.mean(drift_quality[1:])

        logger.info(
            f"Adaptive drift correction completed: avg_window_size={avg_window_size:.1f}, "
            f"avg_uncertainty=({avg_uncertainty_x:.3f}, {avg_uncertainty_y:.3f}), "
            f"avg_quality={avg_quality:.0f}"
        )

        # Generate comprehensive drift analysis plot
        if plot_dir:
            self._plot_adaptive_drift_analysis(
                frames,
                drift_x_fine,
                drift_y_fine,
                uncertainty_x_fine,
                uncertainty_y_fine,
                drift_quality,
                optimal_windows,
                change_points,
                coarse_estimates,
                coarse_estimates_filtered,
                plot_dir,
                use_spline_interpolation,
            )

        return (
            drift_x_fine,
            drift_y_fine,
            uncertainty_x_fine,
            uncertainty_y_fine,
            drift_quality,
        )

    def _get_coarse_drift_estimates(
        self, frames, max_shift, min_locs_per_frame, window_size
    ):
        """Get initial coarse drift estimates for change point detection"""
        n_frames = len(frames)
        coarse_estimates = {
            "drift_x": [],
            "drift_y": [],
            "uncertainty_x": [],
            "uncertainty_y": [],
            "quality": [],
            "frame_indices": [],
        }

        # Use sliding windows with fixed size for initial estimation
        for center_idx in range(window_size // 2, n_frames - window_size // 2):
            start_idx = max(0, center_idx - window_size // 2)
            end_idx = min(n_frames, center_idx + window_size // 2)

            # Aggregate localizations from window
            ref_frames = frames[start_idx:center_idx]
            target_frames = frames[center_idx:end_idx]

            ref_locs = self.locs[np.isin(self.locs["frame"], ref_frames)]
            target_locs = self.locs[np.isin(self.locs["frame"], target_frames)]

            if (
                len(ref_locs) < min_locs_per_frame
                or len(target_locs) < min_locs_per_frame
            ):
                # Insufficient data, use nan
                coarse_estimates["drift_x"].append(np.nan)
                coarse_estimates["drift_y"].append(np.nan)
                coarse_estimates["uncertainty_x"].append(np.nan)
                coarse_estimates["uncertainty_y"].append(np.nan)
                coarse_estimates["quality"].append(0)
            else:
                (
                    shift_x,
                    shift_y,
                    quality,
                    uncertainty_x,
                    uncertainty_y,
                ) = self._estimate_drift_between_frame_groups(
                    ref_locs,
                    target_locs,
                    max_shift,
                    min_locs_per_frame,
                    outlier_detection_enabled=False,
                    min_signal_to_noise=0.5,  # Disable for coarse estimates
                )
                coarse_estimates["drift_x"].append(
                    shift_x if shift_x is not None else np.nan
                )
                coarse_estimates["drift_y"].append(
                    shift_y if shift_y is not None else np.nan
                )
                coarse_estimates["uncertainty_x"].append(
                    uncertainty_x if uncertainty_x is not None else np.nan
                )
                coarse_estimates["uncertainty_y"].append(
                    uncertainty_y if uncertainty_y is not None else np.nan
                )
                coarse_estimates["quality"].append(quality)

            coarse_estimates["frame_indices"].append(center_idx)

        return coarse_estimates

    def _filter_temporal_outliers(self, coarse_estimates, z_threshold=3.5):
        """Filter temporal outliers from coarse drift estimates using local consistency

        This method identifies and removes drift measurements that are inconsistent
        with their temporal neighbors, which often result from RSSO fit failures.
        """
        drift_x = np.array(coarse_estimates["drift_x"])
        drift_y = np.array(coarse_estimates["drift_y"])
        uncertainties_x = np.array(coarse_estimates["uncertainty_x"])
        uncertainties_y = np.array(coarse_estimates["uncertainty_y"])
        qualities = np.array(coarse_estimates["quality"])

        # Create filtered copy
        filtered_estimates = {
            key: coarse_estimates[key].copy() for key in coarse_estimates
        }

        # Find valid measurements
        valid_mask = ~(np.isnan(drift_x) | np.isnan(drift_y))
        if np.sum(valid_mask) < 3:  # Need at least 3 points for filtering
            return filtered_estimates

        # Method 1: Local consistency check using moving median
        window_size = min(5, max(3, len(drift_x) // 4))

        for i in range(len(drift_x)):
            if not valid_mask[i]:
                continue

            # Define local window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(drift_x), i + window_size // 2 + 1)

            local_drift_x = drift_x[start_idx:end_idx]
            local_drift_y = drift_y[start_idx:end_idx]
            local_valid = valid_mask[start_idx:end_idx]

            if np.sum(local_valid) < 2:
                continue

            # Calculate local statistics (excluding current point)
            local_x_others = local_drift_x[
                local_valid & (np.arange(start_idx, end_idx) != i)
            ]
            local_y_others = local_drift_y[
                local_valid & (np.arange(start_idx, end_idx) != i)
            ]

            if len(local_x_others) < 2:
                continue

            # Use robust statistics (median and MAD)
            median_x = np.median(local_x_others)
            median_y = np.median(local_y_others)
            mad_x = np.median(np.abs(local_x_others - median_x))
            mad_y = np.median(np.abs(local_y_others - median_y))

            # Convert MAD to approximate standard deviation
            std_x = 1.4826 * mad_x if mad_x > 0 else np.std(local_x_others)
            std_y = 1.4826 * mad_y if mad_y > 0 else np.std(local_y_others)

            # Check if current measurement is an outlier (Modified Z-score > threshold)
            z_score_x = abs(drift_x[i] - median_x) / (std_x + 1e-10)
            z_score_y = abs(drift_y[i] - median_y) / (std_y + 1e-10)

            is_outlier = (z_score_x > z_threshold) or (z_score_y > z_threshold)

            # Additional check: inconsistent with local trend
            if not is_outlier and len(local_x_others) >= 3:
                # Check if measurement is inconsistent with local linear trend
                trend_inconsistency = self._check_trend_inconsistency(
                    i, start_idx, end_idx, drift_x, drift_y, valid_mask
                )
                is_outlier = is_outlier or trend_inconsistency

            if is_outlier:
                logger.debug(
                    f"Filtering temporal outlier at frame {coarse_estimates['frame_indices'][i]}: "
                    f"drift=({drift_x[i]:.3f}, {drift_y[i]:.3f}), "
                    f"z_scores=({z_score_x:.2f}, {z_score_y:.2f})"
                )

                # Mark as invalid
                filtered_estimates["drift_x"][i] = np.nan
                filtered_estimates["drift_y"][i] = np.nan
                filtered_estimates["uncertainty_x"][i] = np.nan
                filtered_estimates["uncertainty_y"][i] = np.nan
                filtered_estimates["quality"][i] = 0

        return filtered_estimates

    def _check_trend_inconsistency(
        self, current_idx, start_idx, end_idx, drift_x, drift_y, valid_mask
    ):
        """Check if a measurement is inconsistent with local trend"""
        try:
            # Get local valid indices
            local_indices = np.arange(start_idx, end_idx)
            local_valid = valid_mask[start_idx:end_idx] & (
                local_indices != current_idx
            )

            if np.sum(local_valid) < 2:
                return False

            # Extract local data (excluding current point)
            local_x = drift_x[start_idx:end_idx][local_valid]
            local_y = drift_y[start_idx:end_idx][local_valid]
            local_frame_indices = local_indices[local_valid]

            # Fit linear trend
            if len(local_x) >= 2:
                trend_x = np.polyfit(local_frame_indices, local_x, 1)
                trend_y = np.polyfit(local_frame_indices, local_y, 1)

                # Predict what current measurement should be
                expected_x = np.polyval(trend_x, current_idx)
                expected_y = np.polyval(trend_y, current_idx)

                # Calculate residuals for trend fitting
                residuals_x = local_x - np.polyval(
                    trend_x, local_frame_indices
                )
                residuals_y = local_y - np.polyval(
                    trend_y, local_frame_indices
                )

                # Estimate trend uncertainty
                trend_std_x = (
                    np.std(residuals_x) if len(residuals_x) > 1 else 0
                )
                trend_std_y = (
                    np.std(residuals_y) if len(residuals_y) > 1 else 0
                )

                # Check if current measurement deviates significantly from trend
                deviation_x = abs(drift_x[current_idx] - expected_x)
                deviation_y = abs(drift_y[current_idx] - expected_y)

                # Use 3-sigma rule for trend consistency
                inconsistent_x = deviation_x > 3 * (trend_std_x + 1e-10)
                inconsistent_y = deviation_y > 3 * (trend_std_y + 1e-10)

                return inconsistent_x or inconsistent_y

        except Exception as e:
            logger.debug(f"Trend inconsistency check failed: {e}")
            return False

        return False

    def _detect_change_points(self, coarse_estimates, sensitivity):
        """Detect change points in drift patterns using statistical methods"""
        drift_x = np.array(coarse_estimates["drift_x"])
        drift_y = np.array(coarse_estimates["drift_y"])
        frame_indices = np.array(coarse_estimates["frame_indices"])

        # Remove nan values
        valid_mask = ~(np.isnan(drift_x) | np.isnan(drift_y))
        if np.sum(valid_mask) < 5:  # Need at least 5 points
            return []

        drift_x_valid = drift_x[valid_mask]
        drift_y_valid = drift_y[valid_mask]
        frame_indices_valid = frame_indices[valid_mask]

        change_points = []

        # Method 1: Detect large changes in drift magnitude
        drift_magnitude = np.sqrt(drift_x_valid**2 + drift_y_valid**2)
        if len(drift_magnitude) > 3:
            # Use moving average and standard deviation for change detection
            window = min(5, len(drift_magnitude) // 3)
            rolling_mean = np.convolve(
                drift_magnitude, np.ones(window) / window, mode="same"
            )
            rolling_std = np.array(
                [
                    np.std(
                        drift_magnitude[
                            max(0, i - window // 2) : i + window // 2 + 1
                        ]
                    )
                    for i in range(len(drift_magnitude))
                ]
            )

            # Detect outliers (change points)
            z_scores = np.abs(
                (drift_magnitude - rolling_mean) / (rolling_std + 1e-10)
            )
            change_indices = np.where(z_scores > sensitivity)[0]

            change_points.extend(frame_indices_valid[change_indices].tolist())

        # Method 2: Confidence interval intersection analysis
        uncertainties_x = np.array(coarse_estimates["uncertainty_x"])[
            valid_mask
        ]
        uncertainties_y = np.array(coarse_estimates["uncertainty_y"])[
            valid_mask
        ]

        if not np.any(np.isnan(uncertainties_x)) and not np.any(
            np.isnan(uncertainties_y)
        ):
            for i in range(1, len(drift_x_valid)):
                # Check if confidence intervals don't overlap (significant change)
                prev_ci_x = [
                    drift_x_valid[i - 1]
                    - sensitivity * uncertainties_x[i - 1],
                    drift_x_valid[i - 1]
                    + sensitivity * uncertainties_x[i - 1],
                ]
                curr_ci_x = [
                    drift_x_valid[i] - sensitivity * uncertainties_x[i],
                    drift_x_valid[i] + sensitivity * uncertainties_x[i],
                ]

                prev_ci_y = [
                    drift_y_valid[i - 1]
                    - sensitivity * uncertainties_y[i - 1],
                    drift_y_valid[i - 1]
                    + sensitivity * uncertainties_y[i - 1],
                ]
                curr_ci_y = [
                    drift_y_valid[i] - sensitivity * uncertainties_y[i],
                    drift_y_valid[i] + sensitivity * uncertainties_y[i],
                ]

                # Check for non-overlapping confidence intervals
                if (
                    curr_ci_x[0] > prev_ci_x[1]
                    or curr_ci_x[1] < prev_ci_x[0]
                    or curr_ci_y[0] > prev_ci_y[1]
                    or curr_ci_y[1] < prev_ci_y[0]
                ):
                    change_points.append(frame_indices_valid[i])

        # Remove duplicates and sort
        change_points = sorted(list(set(change_points)))

        return change_points

    def _calculate_adaptive_windows(
        self,
        frames,
        change_points,
        coarse_estimates,
        min_window_size,
        max_window_size,
        confidence_threshold,
    ):
        """Calculate optimal window sizes for each frame based on local stability"""
        n_frames = len(frames)
        optimal_windows = np.full(n_frames, min_window_size)

        # Convert change points to segments
        segment_boundaries = [0] + change_points + [n_frames]
        segments = [
            (segment_boundaries[i], segment_boundaries[i + 1])
            for i in range(len(segment_boundaries) - 1)
        ]

        for start, end in segments:
            segment_length = end - start

            # Analyze local noise level in this segment
            local_uncertainties = []
            local_qualities = []

            for i, frame_idx in enumerate(coarse_estimates["frame_indices"]):
                if start <= frame_idx < end:
                    if not np.isnan(coarse_estimates["uncertainty_x"][i]):
                        local_uncertainties.append(
                            np.sqrt(
                                coarse_estimates["uncertainty_x"][i] ** 2
                                + coarse_estimates["uncertainty_y"][i] ** 2
                            )
                        )
                    local_qualities.append(coarse_estimates["quality"][i])

            if local_uncertainties:
                # Calculate confidence metric for this segment
                mean_uncertainty = np.mean(local_uncertainties)
                mean_quality = (
                    np.mean(local_qualities) if local_qualities else 0
                )

                # Normalize confidence (higher quality, lower uncertainty = higher confidence)
                max_quality = (
                    max(coarse_estimates["quality"])
                    if coarse_estimates["quality"]
                    else 1
                )
                confidence = (mean_quality / max_quality) / (
                    1 + mean_uncertainty
                )

                # Determine optimal window size based on confidence
                if confidence >= confidence_threshold:
                    # High confidence: can use larger windows
                    target_window_size = min(
                        max_window_size, segment_length // 2
                    )
                else:
                    # Low confidence: use smaller windows for better tracking
                    target_window_size = max(
                        min_window_size,
                        min(max_window_size // 2, segment_length // 3),
                    )

                # Set window sizes for this segment
                for frame_idx in range(start, end):
                    optimal_windows[frame_idx] = target_window_size

                logger.debug(
                    f"Segment [{start}:{end}] - Confidence: {confidence:.3f}, "
                    f"Window size: {target_window_size}"
                )

        return optimal_windows

    def _estimate_drift_with_adaptive_windows(
        self,
        frames,
        optimal_windows,
        max_shift,
        min_locs_per_frame,
        save_all_rsso_plots,
        plot_dir,
        outlier_detection_enabled=True,
        min_signal_to_noise=0.5,
    ):
        """Estimate drift using adaptive window sizes for each frame"""
        n_frames = len(frames)

        drift_x_fine = np.zeros(n_frames)
        drift_y_fine = np.zeros(n_frames)
        uncertainty_x_fine = np.zeros(n_frames)
        uncertainty_y_fine = np.zeros(n_frames)
        drift_quality = np.zeros(n_frames)

        for frame_idx in range(1, n_frames):
            window_size = int(optimal_windows[frame_idx])

            # Define reference and target windows
            ref_start = max(0, frame_idx - window_size)
            ref_end = frame_idx
            target_start = frame_idx
            target_end = min(n_frames, frame_idx + window_size)

            # Get localizations for reference and target windows
            ref_frame_range = frames[ref_start:ref_end]
            target_frame_range = frames[target_start:target_end]

            ref_locs = self.locs[np.isin(self.locs["frame"], ref_frame_range)]
            target_locs = self.locs[
                np.isin(self.locs["frame"], target_frame_range)
            ]

            # Estimate drift for this frame
            (
                shift_x,
                shift_y,
                quality,
                uncertainty_x,
                uncertainty_y,
            ) = self._estimate_drift_between_frame_groups(
                ref_locs,
                target_locs,
                max_shift,
                min_locs_per_frame,
                outlier_detection_enabled,
                min_signal_to_noise,
            )

            if shift_x is not None and shift_y is not None:
                # Scale by window overlap to get per-frame drift
                overlap_factor = (
                    min(window_size, target_end - target_start) / window_size
                )
                per_frame_shift_x = shift_x * overlap_factor / window_size
                per_frame_shift_y = shift_y * overlap_factor / window_size

                # Accumulate drift
                drift_x_fine[frame_idx] = (
                    drift_x_fine[frame_idx - 1] + per_frame_shift_x
                )
                drift_y_fine[frame_idx] = (
                    drift_y_fine[frame_idx - 1] + per_frame_shift_y
                )

                # Scale uncertainties appropriately
                scaled_uncertainty_x = (
                    uncertainty_x * overlap_factor / window_size
                    if uncertainty_x
                    else 0
                )
                scaled_uncertainty_y = (
                    uncertainty_y * overlap_factor / window_size
                    if uncertainty_y
                    else 0
                )

                uncertainty_x_fine[frame_idx] = np.sqrt(
                    uncertainty_x_fine[frame_idx - 1] ** 2
                    + scaled_uncertainty_x**2
                )
                uncertainty_y_fine[frame_idx] = np.sqrt(
                    uncertainty_y_fine[frame_idx - 1] ** 2
                    + scaled_uncertainty_y**2
                )

                drift_quality[frame_idx] = quality
            else:
                # Failed estimation - use previous values
                drift_x_fine[frame_idx] = drift_x_fine[frame_idx - 1]
                drift_y_fine[frame_idx] = drift_y_fine[frame_idx - 1]
                uncertainty_x_fine[frame_idx] = uncertainty_x_fine[
                    frame_idx - 1
                ]
                uncertainty_y_fine[frame_idx] = uncertainty_y_fine[
                    frame_idx - 1
                ]
                drift_quality[frame_idx] = 0

        return (
            drift_x_fine,
            drift_y_fine,
            uncertainty_x_fine,
            uncertainty_y_fine,
            drift_quality,
        )

    def _estimate_drift_between_frame_groups(
        self,
        locs_ref,
        locs_target,
        max_shift,
        min_locs,
        outlier_detection_enabled=True,
        min_signal_to_noise=0.5,
    ):
        """Helper function to estimate drift between two groups of localizations using RSSO

        Enhanced with RSSO fit failure detection and outlier filtering.

        Returns:
            shift_x, shift_y: drift measurements (None if failed/outlier)
            quality: quality metric based on number of localizations
            uncertainty_x, uncertainty_y: estimated uncertainties in shift measurements
        """
        from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

        if len(locs_ref) < min_locs or len(locs_target) < min_locs:
            return None, None, 0, None, None

        shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
            locs_ref, locs_target, max_shift, plot_histogram=False
        )

        if shift_x is not None and shift_y is not None:
            # Check for RSSO fit failure indicators
            fit_successful = uncertainty_info.get("fit_successful", False)

            # Extract uncertainty estimates
            uncertainty_x = (
                uncertainty_info["sigma_x"]
                if fit_successful
                else uncertainty_info.get("shift_x_error", np.inf)
            )
            uncertainty_y = (
                uncertainty_info["sigma_y"]
                if fit_successful
                else uncertainty_info.get("shift_y_error", np.inf)
            )

            # Quality metrics for outlier detection
            quality = len(locs_ref) + len(locs_target)
            shift_magnitude = np.sqrt(shift_x**2 + shift_y**2)

            # Detect RSSO failure and outliers
            if outlier_detection_enabled:
                is_outlier = self._detect_rsso_failure_and_outliers(
                    shift_x,
                    shift_y,
                    uncertainty_x,
                    uncertainty_y,
                    shift_magnitude,
                    max_shift,
                    fit_successful,
                    quality,
                    min_signal_to_noise,
                )
            else:
                is_outlier = False

            if is_outlier:
                logger.debug(
                    f"Detected RSSO outlier/failure: shift=({shift_x:.3f}, {shift_y:.3f}), "
                    f"magnitude={shift_magnitude:.3f}, fit_successful={fit_successful}, "
                    f"uncertainty=({uncertainty_x:.3f}, {uncertainty_y:.3f})"
                )
                return None, None, 0, None, None

            return shift_x, shift_y, quality, uncertainty_x, uncertainty_y
        else:
            return None, None, 0, None, None

    def _detect_rsso_failure_and_outliers(
        self,
        shift_x,
        shift_y,
        uncertainty_x,
        uncertainty_y,
        shift_magnitude,
        max_shift,
        fit_successful,
        quality,
        min_signal_to_noise=0.5,
    ):
        """Detect RSSO fit failures and false drift outliers

        Args:
            shift_x, shift_y: Estimated drift values
            uncertainty_x, uncertainty_y: Uncertainty estimates
            shift_magnitude: Magnitude of drift vector
            max_shift: Maximum expected drift per comparison
            fit_successful: Whether the Gaussian fit was successful
            quality: Quality metric (number of localizations)

        Returns:
            bool: True if measurement should be considered an outlier/failure
        """

        # Criterion 1: Gaussian fit failure
        if not fit_successful:
            return True

        # Criterion 2: Excessive shift magnitude (likely spurious correlation)
        # Allow for accumulated drift but flag extreme outliers
        if shift_magnitude > 3 * max_shift:  # 3x safety margin
            return True

        # Criterion 3: Excessive uncertainty relative to signal
        if uncertainty_x is not None and uncertainty_y is not None:
            # Flag if uncertainty is comparable to or larger than the shift
            uncertainty_magnitude = np.sqrt(
                uncertainty_x**2 + uncertainty_y**2
            )
            signal_to_noise = shift_magnitude / (uncertainty_magnitude + 1e-10)

            if (
                signal_to_noise < min_signal_to_noise
            ):  # Signal less than minimum threshold
                return True

            # Flag if uncertainty is unrealistically large (fit instability)
            if uncertainty_magnitude > 2 * max_shift:
                return True

        # Criterion 4: Very low quality (insufficient data for reliable estimation)
        if quality < 20:  # Minimum threshold for reliable cross-correlation
            return True

        # Criterion 5: Detect common RSSO failure patterns
        # Check for suspiciously round numbers (often artifacts)
        if abs(shift_x) < 1e-10 and abs(shift_y) < 1e-10:  # Exactly zero drift
            return True

        # Check for extreme aspect ratios (likely spurious)
        aspect_ratio = max(abs(shift_x), abs(shift_y)) / (
            min(abs(shift_x), abs(shift_y)) + 1e-10
        )
        if aspect_ratio > 10:  # One direction dominates unrealistically
            return True

        return False

    def _plot_adaptive_drift_analysis(
        self,
        frames,
        drift_x_fine,
        drift_y_fine,
        uncertainty_x_fine,
        uncertainty_y_fine,
        drift_quality,
        optimal_windows,
        change_points,
        coarse_estimates,
        coarse_estimates_filtered,
        plot_dir,
        use_spline_interpolation,
    ):
        """Generate comprehensive drift analysis plot showing local drift, confidence, and window sizes"""
        import os

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        stage2_method = (
            "Cubic Spline" if use_spline_interpolation else "Linear Blocks"
        )
        fig.suptitle(
            f"Adaptive Drift Correction Analysis (Stage 2: {stage2_method})",
            fontsize=16,
            fontweight="bold",
        )

        frame_indices = np.arange(len(frames))

        # Plot 1: Drift trajectory with confidence intervals
        ax1 = axes[0]
        ax1.plot(
            frame_indices, drift_x_fine, "b-", label="Drift X", linewidth=1.5
        )
        ax1.plot(
            frame_indices, drift_y_fine, "r-", label="Drift Y", linewidth=1.5
        )

        # Add confidence intervals
        ax1.fill_between(
            frame_indices,
            drift_x_fine - uncertainty_x_fine,
            drift_x_fine + uncertainty_x_fine,
            alpha=0.3,
            color="blue",
            label="X uncertainty",
        )
        ax1.fill_between(
            frame_indices,
            drift_y_fine - uncertainty_y_fine,
            drift_y_fine + uncertainty_y_fine,
            alpha=0.3,
            color="red",
            label="Y uncertainty",
        )

        # # Mark change points
        # for cp in change_points:
        #     if cp < len(frames):
        #         ax1.axvline(
        #             x=cp,
        #             color="orange",
        #             linestyle="--",
        #             alpha=0.7,
        #             linewidth=2,
        #         )

        # Mark filtered outliers
        coarse_frame_indices = np.array(coarse_estimates["frame_indices"])
        original_drift_x = np.array(coarse_estimates["drift_x"])
        original_drift_y = np.array(coarse_estimates["drift_y"])
        filtered_drift_x = np.array(coarse_estimates_filtered["drift_x"])
        filtered_drift_y = np.array(coarse_estimates_filtered["drift_y"])

        # Find outliers (points that were valid originally but filtered out)
        original_valid = ~(
            np.isnan(original_drift_x) | np.isnan(original_drift_y)
        )
        filtered_valid = ~(
            np.isnan(filtered_drift_x) | np.isnan(filtered_drift_y)
        )
        outliers = original_valid & ~filtered_valid

        if np.any(outliers):
            outlier_frames = coarse_frame_indices[outliers]
            outlier_x = original_drift_x[outliers]
            outlier_y = original_drift_y[outliers]
            ax1.scatter(
                outlier_frames,
                outlier_x,
                color="red",
                marker="x",
                s=50,
                label=f"Filtered Outliers ({np.sum(outliers)})",
                alpha=0.7,
            )
            ax1.scatter(
                outlier_frames,
                outlier_y,
                color="red",
                marker="x",
                s=50,
                alpha=0.7,
            )

        ax1.set_ylabel("Cumulative Drift (pixels)")
        ax1.set_title("Drift Trajectory with Confidence Intervals")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Local drift rate and quality
        ax2 = axes[1]

        # Calculate instantaneous drift rate
        drift_rate_x = np.diff(drift_x_fine)
        drift_rate_y = np.diff(drift_y_fine)
        drift_rate_magnitude = np.sqrt(drift_rate_x**2 + drift_rate_y**2)

        # Plot drift rate
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(
            frame_indices[1:],
            drift_rate_magnitude,
            "g-",
            linewidth=1.5,
            label="Drift Rate Magnitude",
        )

        # Plot quality as bars
        quality_normalized = (
            drift_quality / np.max(drift_quality)
            if np.max(drift_quality) > 0
            else drift_quality
        )
        line2 = ax2_twin.bar(
            frame_indices,
            quality_normalized,
            alpha=0.4,
            color="purple",
            label="Normalized Quality",
        )

        ax2.set_ylabel("Drift Rate (pixels/frame)", color="g")
        ax2_twin.set_ylabel("Normalized Quality", color="purple")
        ax2.set_title("Local Drift Rate and Measurement Quality")
        ax2.tick_params(axis="y", labelcolor="g")
        ax2_twin.tick_params(axis="y", labelcolor="purple")
        ax2.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Plot 3: Adaptive window sizes and confidence
        ax3 = axes[2]

        # Plot window sizes as step function
        ax3.step(
            frame_indices,
            optimal_windows,
            where="mid",
            linewidth=2,
            color="navy",
            label="Window Size",
        )

        # Add confidence metric overlay
        ax3_twin = ax3.twinx()

        # Calculate local confidence from coarse estimates
        confidence_series = np.full(len(frames), np.nan)
        coarse_frame_indices = np.array(coarse_estimates["frame_indices"])
        coarse_uncertainties = np.array(coarse_estimates["uncertainty_x"])
        coarse_qualities = np.array(coarse_estimates["quality"])

        valid_coarse = ~np.isnan(coarse_uncertainties) & (coarse_qualities > 0)
        if np.any(valid_coarse):
            max_quality = np.max(coarse_qualities[valid_coarse])
            for i, frame_idx in enumerate(coarse_frame_indices):
                if valid_coarse[i] and frame_idx < len(frames):
                    uncertainty_combined = (
                        np.sqrt(
                            coarse_estimates["uncertainty_x"][i] ** 2
                            + coarse_estimates["uncertainty_y"][i] ** 2
                        )
                        if not np.isnan(coarse_estimates["uncertainty_y"][i])
                        else coarse_uncertainties[i]
                    )

                    confidence = (coarse_qualities[i] / max_quality) / (
                        1 + uncertainty_combined
                    )
                    confidence_series[frame_idx] = confidence

        # Interpolate confidence for plotting
        valid_conf_mask = ~np.isnan(confidence_series)
        if np.any(valid_conf_mask):
            conf_interp = np.interp(
                frame_indices,
                frame_indices[valid_conf_mask],
                confidence_series[valid_conf_mask],
            )
            ax3_twin.plot(
                frame_indices,
                conf_interp,
                "orange",
                linewidth=1.5,
                label="Local Confidence",
                alpha=0.8,
            )

        # Add segments with different window sizes
        segment_boundaries = [0] + change_points + [len(frames)]
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_boundaries) - 1))

        for i, (start, end) in enumerate(
            zip(segment_boundaries[:-1], segment_boundaries[1:])
        ):
            if end > start:
                rect = Rectangle(
                    (start, 0),
                    end - start,
                    np.max(optimal_windows),
                    alpha=0.1,
                    color=colors[i % len(colors)],
                )
                ax3.add_patch(rect)

        ax3.set_ylabel("Window Size (frames)", color="navy")
        ax3_twin.set_ylabel("Confidence", color="orange")
        ax3.set_title("Adaptive Window Sizes and Local Confidence")
        ax3.tick_params(axis="y", labelcolor="navy")
        ax3_twin.tick_params(axis="y", labelcolor="orange")
        ax3.grid(True, alpha=0.3)

        # Legend
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines3_twin, labels3_twin = ax3_twin.get_legend_handles_labels()
        ax3.legend(
            lines3 + lines3_twin, labels3 + labels3_twin, loc="upper right"
        )

        # Plot 4: Uncertainty evolution
        ax4 = axes[3]

        total_uncertainty = np.sqrt(
            uncertainty_x_fine**2 + uncertainty_y_fine**2
        )
        ax4.plot(
            frame_indices,
            uncertainty_x_fine,
            "b-",
            linewidth=1.5,
            label="X Uncertainty",
            alpha=0.7,
        )
        ax4.plot(
            frame_indices,
            uncertainty_y_fine,
            "r-",
            linewidth=1.5,
            label="Y Uncertainty",
            alpha=0.7,
        )
        ax4.plot(
            frame_indices,
            total_uncertainty,
            "k-",
            linewidth=2,
            label="Total Uncertainty",
        )

        # # Mark change points
        # for cp in change_points:
        #     if cp < len(frames):
        #         ax4.axvline(x=cp, color="orange", linestyle="--", alpha=0.7)

        ax4.set_ylabel("Uncertainty (pixels)")
        ax4.set_xlabel("Frame Index")
        ax4.set_title("Cumulative Uncertainty Evolution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_path = os.path.join(plot_dir, "adaptive_drift_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.debug(f"Saved adaptive drift analysis plot: {plot_path}")

        # Additional summary statistics plot
        self._plot_drift_statistics_summary(
            frames,
            drift_x_fine,
            drift_y_fine,
            uncertainty_x_fine,
            uncertainty_y_fine,
            optimal_windows,
            change_points,
            plot_dir,
        )

    def _plot_drift_statistics_summary(
        self,
        frames,
        drift_x,
        drift_y,
        uncertainty_x,
        uncertainty_y,
        window_sizes,
        change_points,
        plot_dir,
    ):
        """Generate summary statistics plot for drift correction performance"""
        import os

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            "Drift Correction Performance Summary",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Drift magnitude histogram
        ax1 = axes[0, 0]
        drift_magnitude = np.sqrt(drift_x**2 + drift_y**2)
        ax1.hist(
            drift_magnitude[1:],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax1.set_xlabel("Total Drift Magnitude (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Cumulative Drift")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Window size distribution
        ax2 = axes[0, 1]
        window_counts, window_bins = np.histogram(window_sizes[1:], bins=20)
        ax2.bar(
            window_bins[:-1],
            window_counts,
            width=np.diff(window_bins),
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        ax2.set_xlabel("Window Size (frames)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Adaptive Window Sizes")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Uncertainty vs Window Size
        ax3 = axes[1, 0]
        total_uncertainty = np.sqrt(uncertainty_x**2 + uncertainty_y**2)
        ax3.scatter(
            window_sizes[1:],
            total_uncertainty[1:],
            alpha=0.6,
            c=np.arange(1, len(window_sizes)),
            cmap="viridis",
        )
        ax3.set_xlabel("Window Size (frames)")
        ax3.set_ylabel("Total Uncertainty (pixels)")
        ax3.set_title("Uncertainty vs Window Size")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Change point analysis
        ax4 = axes[1, 1]
        if change_points:
            # Calculate inter-change-point intervals
            intervals = []
            prev_cp = 0
            for cp in change_points:
                intervals.append(cp - prev_cp)
                prev_cp = cp
            intervals.append(len(frames) - prev_cp)

            ax4.hist(
                intervals,
                bins=min(15, len(intervals)),
                alpha=0.7,
                color="orange",
                edgecolor="black",
            )
            ax4.set_xlabel("Segment Length (frames)")
            ax4.set_ylabel("Frequency")
            ax4.set_title(
                f"Stable Segment Lengths ({len(change_points)} change points)"
            )
        else:
            ax4.text(
                0.5,
                0.5,
                "No Change Points Detected",
                transform=ax4.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax4.set_title("Change Point Analysis")

        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        summary_path = os.path.join(plot_dir, "drift_correction_summary.png")
        plt.savefig(
            summary_path, dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.close()

        logger.debug(f"Saved drift correction summary plot: {summary_path}")

    def _spline_based_drift_correction(
        self,
        frames,
        toff,
        max_shift,
        min_locs_per_block,
        n_processes,
        save_all_rsso_plots,
        plot_dir,
        smoothing_factor,
        min_blocks_for_spline,
    ):
        """Cubic spline-based long-timescale drift correction using block centers as anchor points

        This method estimates drift at the center of each toff-sized time block, then uses
        cubic spline interpolation to create a smooth drift trajectory over all frames.

        Returns:
            tuple: (drift_x_coarse, drift_y_coarse, uncertainty_x_coarse, uncertainty_y_coarse)
        """
        from scipy.interpolate import UnivariateSpline

        n_frames = len(frames)
        toff_block_size = int(toff)

        # Initialize arrays
        drift_x_coarse = np.zeros(n_frames)
        drift_y_coarse = np.zeros(n_frames)
        uncertainty_x_coarse = np.zeros(n_frames)
        uncertainty_y_coarse = np.zeros(n_frames)

        # Calculate block centers and estimate drift at each center
        block_centers = []
        block_center_frames = []
        block_drifts_x = []
        block_drifts_y = []
        block_uncertainties_x = []
        block_uncertainties_y = []
        block_weights = []

        logger.debug(
            f"Processing {int(np.ceil(n_frames / toff_block_size))} time blocks for spline anchors"
        )

        # Process blocks to get anchor points
        block_args = []
        for block_start in range(0, n_frames, toff_block_size):
            block_end = min(block_start + toff_block_size, n_frames)

            # Calculate block center frame
            center_frame_idx = (block_start + block_end) // 2
            center_frame = frames[center_frame_idx]

            block_args.append(
                (
                    self.locs,
                    frames,
                    block_start,
                    block_end,
                    toff_block_size,
                    max_shift,
                    min_locs_per_block,
                    save_all_rsso_plots,
                    plot_dir,
                )
            )
            block_centers.append(center_frame_idx)
            block_center_frames.append(center_frame)

        # Process blocks in parallel to get drift estimates at centers
        try:
            if n_processes == 1 or len(block_args) <= 1:
                logger.debug(
                    "Using single-threaded processing for spline anchor points"
                )
                block_results = [
                    self._process_drift_block(args) for args in block_args
                ]
            else:
                logger.debug(
                    f"Using {n_processes} processes for spline anchor points "
                    + f"({len(block_args)} blocks)"
                )
                from multiprocessing import Pool

                with Pool(processes=n_processes) as pool:
                    block_results = pool.map(
                        self._process_drift_block, block_args
                    )
        except Exception as e:
            logger.warning(
                "Multiprocessing failed for spline anchor points, falling back to "
                + f"single-threaded: {e}"
            )
            block_results = [
                self._process_drift_block(args) for args in block_args
            ]

        # Extract valid anchor points for spline fitting
        valid_anchors = []
        cumulative_drift_x = 0
        cumulative_drift_y = 0

        for i, result in enumerate(block_results):
            if result is None:
                continue  # Skip first block (no comparison possible)

            (
                block_start,
                block_end,
                shift_x,
                shift_y,
                quality,
                uncertainty_x,
                uncertainty_y,
            ) = result

            if shift_x is not None and shift_y is not None and quality > 0:
                # Accumulate drift to get absolute position
                cumulative_drift_x += shift_x
                cumulative_drift_y += shift_y

                # Store anchor point data
                center_idx = block_centers[i]
                block_drifts_x.append(cumulative_drift_x)
                block_drifts_y.append(cumulative_drift_y)
                block_uncertainties_x.append(
                    uncertainty_x if uncertainty_x else 1.0
                )
                block_uncertainties_y.append(
                    uncertainty_y if uncertainty_y else 1.0
                )

                # Weight based on quality and inverse uncertainty
                weight_x = quality / (block_uncertainties_x[-1] ** 2 + 1e-10)
                weight_y = quality / (block_uncertainties_y[-1] ** 2 + 1e-10)
                block_weights.append((weight_x, weight_y))

                valid_anchors.append(center_idx)

                logger.debug(
                    f"Anchor point at frame {center_idx}: drift=({cumulative_drift_x:.3f}, {cumulative_drift_y:.3f}), "
                    f"quality={quality:.0f}"
                )

        # Check if we have enough anchor points for spline fitting
        if len(valid_anchors) < min_blocks_for_spline:
            logger.warning(
                f"Only {len(valid_anchors)} valid anchor points, need at least {min_blocks_for_spline}. "
                "Using linear interpolation instead."
            )
            return self._linear_interpolation_fallback(
                valid_anchors,
                block_drifts_x,
                block_drifts_y,
                block_uncertainties_x,
                block_uncertainties_y,
                n_frames,
            )

        # Fit cubic splines to anchor points
        anchor_frames = np.array(valid_anchors)
        drifts_x = np.array(block_drifts_x)
        drifts_y = np.array(block_drifts_y)
        weights_x = np.array([w[0] for w in block_weights])
        weights_y = np.array([w[1] for w in block_weights])

        # Determine smoothing factor if not specified
        if smoothing_factor is None:
            # Auto-calculate based on data quality and number of points
            mean_uncertainty_x = np.mean(block_uncertainties_x)
            mean_uncertainty_y = np.mean(block_uncertainties_y)
            smoothing_factor = (
                len(valid_anchors)
                * (mean_uncertainty_x + mean_uncertainty_y)
                / 2
            )
            logger.debug(
                f"Auto-calculated smoothing factor: {smoothing_factor:.3f}"
            )

        try:
            # Fit univariate splines for X and Y drift
            spline_x = UnivariateSpline(
                anchor_frames, drifts_x, w=weights_x, s=smoothing_factor
            )
            spline_y = UnivariateSpline(
                anchor_frames, drifts_y, w=weights_y, s=smoothing_factor
            )

            # Interpolate drift for all frames
            frame_indices = np.arange(n_frames)
            drift_x_coarse = spline_x(frame_indices)
            drift_y_coarse = spline_y(frame_indices)

            # Estimate uncertainties from spline residuals and propagation
            spline_residuals_x = drifts_x - spline_x(anchor_frames)
            spline_residuals_y = drifts_y - spline_y(anchor_frames)

            # Calculate uncertainty estimates
            base_uncertainty_x = (
                np.std(spline_residuals_x)
                if len(spline_residuals_x) > 1
                else np.mean(block_uncertainties_x)
            )
            base_uncertainty_y = (
                np.std(spline_residuals_y)
                if len(spline_residuals_y) > 1
                else np.mean(block_uncertainties_y)
            )

            # Interpolate uncertainties (conservative approach)
            uncertainty_x_coarse = np.full(n_frames, base_uncertainty_x)
            uncertainty_y_coarse = np.full(n_frames, base_uncertainty_y)

            # Adjust uncertainties based on distance from anchor points
            for frame_idx in range(n_frames):
                # Find closest anchor points
                distances = np.abs(anchor_frames - frame_idx)
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]

                # Increase uncertainty for frames far from anchor points
                distance_factor = 1 + min_distance / toff_block_size
                uncertainty_x_coarse[frame_idx] *= distance_factor
                uncertainty_y_coarse[frame_idx] *= distance_factor

            logger.info(
                f"Spline fitting successful: {len(valid_anchors)} anchor points, "
                f"smoothing={smoothing_factor:.3f}, "
                f"residual_std=({base_uncertainty_x:.3f}, {base_uncertainty_y:.3f})"
            )

            # Save spline diagnostics if plotting is enabled
            if plot_dir:
                self._plot_spline_diagnostics(
                    anchor_frames,
                    drifts_x,
                    drifts_y,
                    spline_x,
                    spline_y,
                    frame_indices,
                    drift_x_coarse,
                    drift_y_coarse,
                    block_uncertainties_x,
                    block_uncertainties_y,
                    plot_dir,
                )

            return (
                drift_x_coarse,
                drift_y_coarse,
                uncertainty_x_coarse,
                uncertainty_y_coarse,
            )

        except Exception as e:
            logger.warning(
                f"Spline fitting failed: {e}. Using linear interpolation fallback."
            )
            return self._linear_interpolation_fallback(
                valid_anchors,
                block_drifts_x,
                block_drifts_y,
                block_uncertainties_x,
                block_uncertainties_y,
                n_frames,
            )

    def _linear_interpolation_fallback(
        self,
        anchor_frames,
        drifts_x,
        drifts_y,
        uncertainties_x,
        uncertainties_y,
        n_frames,
    ):
        """Fallback to linear interpolation when spline fitting fails"""
        from scipy.interpolate import interp1d

        logger.debug("Using linear interpolation fallback")

        if len(anchor_frames) == 0:
            # No valid data - return zeros
            return (
                np.zeros(n_frames),
                np.zeros(n_frames),
                np.ones(n_frames),
                np.ones(n_frames),
            )

        if len(anchor_frames) == 1:
            # Single point - constant drift
            drift_x = np.full(n_frames, drifts_x[0])
            drift_y = np.full(n_frames, drifts_y[0])
            uncertainty_x = np.full(n_frames, uncertainties_x[0])
            uncertainty_y = np.full(n_frames, uncertainties_y[0])
        else:
            # Linear interpolation/extrapolation
            frame_indices = np.arange(n_frames)

            interp_x = interp1d(
                anchor_frames,
                drifts_x,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            interp_y = interp1d(
                anchor_frames,
                drifts_y,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            interp_unc_x = interp1d(
                anchor_frames,
                uncertainties_x,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )
            interp_unc_y = interp1d(
                anchor_frames,
                uncertainties_y,
                kind="linear",
                fill_value="extrapolate",
                bounds_error=False,
            )

            drift_x = interp_x(frame_indices)
            drift_y = interp_y(frame_indices)
            uncertainty_x = interp_unc_x(frame_indices)
            uncertainty_y = interp_unc_y(frame_indices)

        return drift_x, drift_y, uncertainty_x, uncertainty_y

    def _plot_spline_diagnostics(
        self,
        anchor_frames,
        anchor_drifts_x,
        anchor_drifts_y,
        spline_x,
        spline_y,
        frame_indices,
        drift_x_interpolated,
        drift_y_interpolated,
        anchor_uncertainties_x,
        anchor_uncertainties_y,
        plot_dir,
    ):
        """Generate diagnostic plots for spline fitting quality"""
        import os

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Cubic Spline Drift Correction Diagnostics",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: X-direction spline fit
        ax1 = axes[0, 0]
        ax1.plot(
            frame_indices,
            drift_x_interpolated,
            "b-",
            linewidth=2,
            label="Spline Fit",
        )
        ax1.errorbar(
            anchor_frames,
            anchor_drifts_x,
            yerr=anchor_uncertainties_x,
            fmt="ro",
            markersize=6,
            capsize=3,
            label="Anchor Points",
        )
        ax1.set_xlabel("Frame Index")
        ax1.set_ylabel("Cumulative Drift X (pixels)")
        ax1.set_title("X-Direction Spline Interpolation")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Y-direction spline fit
        ax2 = axes[0, 1]
        ax2.plot(
            frame_indices,
            drift_y_interpolated,
            "r-",
            linewidth=2,
            label="Spline Fit",
        )
        ax2.errorbar(
            anchor_frames,
            anchor_drifts_y,
            yerr=anchor_uncertainties_y,
            fmt="ro",
            markersize=6,
            capsize=3,
            label="Anchor Points",
        )
        ax2.set_xlabel("Frame Index")
        ax2.set_ylabel("Cumulative Drift Y (pixels)")
        ax2.set_title("Y-Direction Spline Interpolation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Spline residuals
        ax3 = axes[1, 0]
        residuals_x = anchor_drifts_x - spline_x(anchor_frames)
        residuals_y = anchor_drifts_y - spline_y(anchor_frames)

        ax3.scatter(
            anchor_frames,
            residuals_x,
            c="blue",
            alpha=0.6,
            label="X Residuals",
        )
        ax3.scatter(
            anchor_frames, residuals_y, c="red", alpha=0.6, label="Y Residuals"
        )
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Frame Index")
        ax3.set_ylabel("Spline Residual (pixels)")
        ax3.set_title(
            f"Fit Residuals (RMS: X={np.std(residuals_x):.3f}, Y={np.std(residuals_y):.3f})"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Drift trajectory comparison
        ax4 = axes[1, 1]
        drift_magnitude = np.sqrt(
            drift_x_interpolated**2 + drift_y_interpolated**2
        )
        anchor_magnitude = np.sqrt(
            np.array(anchor_drifts_x) ** 2 + np.array(anchor_drifts_y) ** 2
        )

        ax4.plot(
            frame_indices,
            drift_magnitude,
            "g-",
            linewidth=2,
            label="Spline Magnitude",
        )
        ax4.scatter(
            anchor_frames,
            anchor_magnitude,
            c="orange",
            s=50,
            label="Anchor Magnitude",
            zorder=5,
        )
        ax4.set_xlabel("Frame Index")
        ax4.set_ylabel("Total Drift Magnitude (pixels)")
        ax4.set_title("Drift Magnitude Trajectory")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        spline_path = os.path.join(plot_dir, "spline_drift_diagnostics.png")
        plt.savefig(
            spline_path, dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.close()

        logger.debug(f"Saved spline diagnostics plot: {spline_path}")

    def _process_linear_block_results(
        self, block_results, n_frames, toff_block_size, drift_quality
    ):
        """Process block results using original linear distribution approach"""
        # Initialize arrays
        drift_x_coarse = np.zeros(n_frames)
        drift_y_coarse = np.zeros(n_frames)
        uncertainty_x_coarse = np.zeros(n_frames)
        uncertainty_y_coarse = np.zeros(n_frames)

        # Process results from parallel computation (original implementation)
        for result in block_results:
            if result is None:
                continue  # Skip first block

            (
                block_start,
                block_end,
                shift_x,
                shift_y,
                quality,
                uncertainty_x,
                uncertainty_y,
            ) = result

            if shift_x is not None and shift_y is not None:
                # Distribute block drift across frames in the block
                block_drift_x = shift_x / toff_block_size  # Per-frame drift
                block_drift_y = shift_y / toff_block_size
                # Uncertainty also scales with block size
                block_uncertainty_x = (
                    uncertainty_x / toff_block_size
                    if uncertainty_x is not None
                    else 0
                )
                block_uncertainty_y = (
                    uncertainty_y / toff_block_size
                    if uncertainty_y is not None
                    else 0
                )

                for frame_idx in range(block_start, block_end):
                    if frame_idx > 0:
                        drift_x_coarse[frame_idx] = (
                            drift_x_coarse[frame_idx - 1] + block_drift_x
                        )
                        drift_y_coarse[frame_idx] = (
                            drift_y_coarse[frame_idx - 1] + block_drift_y
                        )
                        # Accumulate uncertainties (assuming independence)
                        uncertainty_x_coarse[frame_idx] = np.sqrt(
                            uncertainty_x_coarse[frame_idx - 1] ** 2
                            + block_uncertainty_x**2
                        )
                        uncertainty_y_coarse[frame_idx] = np.sqrt(
                            uncertainty_y_coarse[frame_idx - 1] ** 2
                            + block_uncertainty_y**2
                        )
                    else:
                        uncertainty_x_coarse[frame_idx] = block_uncertainty_x
                        uncertainty_y_coarse[frame_idx] = block_uncertainty_y
                    drift_quality[frame_idx] += quality / (
                        block_end - block_start
                    )
            else:
                # Use previous block's drift rate if estimation failed
                for frame_idx in range(block_start, block_end):
                    if frame_idx > 0:
                        drift_x_coarse[frame_idx] = drift_x_coarse[
                            frame_idx - 1
                        ]
                        drift_y_coarse[frame_idx] = drift_y_coarse[
                            frame_idx - 1
                        ]
                        uncertainty_x_coarse[frame_idx] = uncertainty_x_coarse[
                            frame_idx - 1
                        ]
                        uncertainty_y_coarse[frame_idx] = uncertainty_y_coarse[
                            frame_idx - 1
                        ]

        return (
            drift_x_coarse,
            drift_y_coarse,
            uncertainty_x_coarse,
            uncertainty_y_coarse,
        )

    @staticmethod
    def _process_fine_drift_chunk(args):
        """Stateless helper function for processing fine drift chunks
        in parallel"""
        (
            locs_data,
            frames,
            chunk_frames,
            max_shift,
            min_locs_per_frame,
            save_all_rsso_plots,
            plot_dir,
        ) = args
        logger.debug(
            f"undrifting chunk {min(chunk_frames)} to {max(chunk_frames)}"
        )

        from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

        results = []

        for frame_idx in chunk_frames:
            if frame_idx == 0:
                continue  # Skip first frame

            current_frame = frames[frame_idx]
            prev_frame = frames[frame_idx - 1]

            # Get localizations for current and previous frame
            current_locs = locs_data[locs_data["frame"] == current_frame]
            prev_locs = locs_data[locs_data["frame"] == prev_frame]

            # Estimate fine drift between consecutive frames
            if (
                len(prev_locs) < min_locs_per_frame
                or len(current_locs) < min_locs_per_frame
            ):
                results.append((frame_idx, None, None, 0, None, None))
                continue

            shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
                prev_locs,
                current_locs,
                max_shift,
                plot_histogram=save_all_rsso_plots,
                plot_dir=plot_dir,
            )

            if shift_x is not None and shift_y is not None:
                quality = len(prev_locs) + len(current_locs)
                uncertainty_x = (
                    uncertainty_info["sigma_x"]
                    if uncertainty_info["fit_successful"]
                    else uncertainty_info["shift_x_error"]
                )
                uncertainty_y = (
                    uncertainty_info["sigma_y"]
                    if uncertainty_info["fit_successful"]
                    else uncertainty_info["shift_y_error"]
                )
                results.append(
                    (
                        frame_idx,
                        shift_x,
                        shift_y,
                        quality,
                        uncertainty_x,
                        uncertainty_y,
                    )
                )
            else:
                results.append((frame_idx, None, None, 0, None, None))

        return results

    #    @profile_resource_usage
    @module_decorator
    def undrift_rsso(self, i, parameters, results):
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
        from picasso_workflow.outpost_modules.undrift_rsso import (
            compute_undrift_rsso,
        )

        # Call external computation function
        self.locs, self.drift, results_data = compute_undrift_rsso(
            locs=self.locs,
            pixelsize=self.pixelsize,
            info=self.info,
            parameters=parameters,
            results_folder=results["folder"],
        )

        # Merge results from computation into results dict
        results.update(results_data)

        return parameters, results

    def _plot_drift_with_confidence(
        self,
        filename,
        dimensions,
        pixelsize,
        method="",
        drift=None,
        uncertainty=None,
    ):
        """Plot drift with confidence intervals"""
        if drift is None:
            drift = self.drift
        if uncertainty is None:
            uncertainty = getattr(self, "drift_uncertainty", None)

        fig, ax = plt.subplots(figsize=(10, 6))
        frames = np.arange(drift.shape[0])

        colors = ["blue", "red", "green", "orange"]

        for i, dim in enumerate(dimensions):
            color = colors[i % len(colors)]

            if isinstance(drift, np.recarray):
                drift_values = drift[dim] * pixelsize
                uncertainty_values = (
                    uncertainty[dim] * pixelsize
                    if uncertainty is not None
                    else None
                )
            else:
                drift_values = drift[:, i] * pixelsize
                uncertainty_values = (
                    uncertainty[:, i] * pixelsize
                    if uncertainty is not None
                    else None
                )

            # Plot drift trajectory
            ax.plot(
                frames,
                drift_values,
                label=f"{dim} drift",
                color=color,
                linewidth=2,
            )

            # Plot confidence intervals if available
            if uncertainty_values is not None:
                # 1-sigma confidence interval
                ax.fill_between(
                    frames,
                    drift_values - uncertainty_values,
                    drift_values + uncertainty_values,
                    alpha=0.3,
                    color=color,
                    label=f"{dim} ±1σ",
                )
                # 2-sigma confidence interval
                ax.fill_between(
                    frames,
                    drift_values - 2 * uncertainty_values,
                    drift_values + 2 * uncertainty_values,
                    alpha=0.1,
                    color=color,
                    label=f"{dim} ±2σ",
                )

        ax.set_xlabel("Frame")
        ax.set_ylabel("Drift [nm]")
        ax.set_title(f"Undrift by {method} (with confidence intervals)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_drift(
        self, filename, dimensions, pixelsize, method="", drift=None
    ):
        """Legacy drift plotting method for compatibility"""
        if drift is None:
            drift = self.drift
        fig, ax = plt.subplots()
        frames = np.arange(drift.shape[0])
        for i, dim in enumerate(dimensions):
            factor = 1e-3 if dim == "z" else 1
            if isinstance(drift, pd.DataFrame):
                ax.plot(frames, drift[dim] * pixelsize * factor, label=dim)
            else:
                ax.plot(frames, drift[:, i] * pixelsize * factor, label=dim)
        ax.set_xlabel("frame")
        ax.set_ylabel("drift [nm]")
        ax.set_title(f"undrift by {method}")
        ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    def _plot_channel_alignment_with_confidence(
        self, shifts, shift_uncertainties, filename
    ):
        """Plot channel alignment shifts with confidence intervals"""
        import matplotlib.pyplot as plt

        n_channels = shifts.shape[1]
        if n_channels < 2:
            return

        channel_indices = np.arange(n_channels)

        # Extract uncertainties
        shift_x_uncertainties = shift_uncertainties.get(
            "shift_x_uncertainties", np.zeros(n_channels)
        )
        shift_y_uncertainties = shift_uncertainties.get(
            "shift_y_uncertainties", np.zeros(n_channels)
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # X shifts plot
        shifts_x = shifts[1, :]  # X shifts are in second row
        ax1.errorbar(
            channel_indices,
            shifts_x,
            yerr=shift_x_uncertainties,
            fmt="o-",
            capsize=5,
            capthick=2,
            label="X shifts",
        )
        ax1.fill_between(
            channel_indices,
            shifts_x - 2 * shift_x_uncertainties,
            shifts_x + 2 * shift_x_uncertainties,
            alpha=0.2,
            label="2σ confidence",
        )
        ax1.fill_between(
            channel_indices,
            shifts_x - shift_x_uncertainties,
            shifts_x + shift_x_uncertainties,
            alpha=0.3,
            label="1σ confidence",
        )
        ax1.set_xlabel("Channel Index")
        ax1.set_ylabel("X Shift [pixels]")
        ax1.set_title("Channel X Alignment Shifts")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Y shifts plot
        shifts_y = shifts[0, :]  # Y shifts are in first row
        ax2.errorbar(
            channel_indices,
            shifts_y,
            yerr=shift_y_uncertainties,
            fmt="o-",
            capsize=5,
            capthick=2,
            label="Y shifts",
        )
        ax2.fill_between(
            channel_indices,
            shifts_y - 2 * shift_y_uncertainties,
            shifts_y + 2 * shift_y_uncertainties,
            alpha=0.2,
            label="2σ confidence",
        )
        ax2.fill_between(
            channel_indices,
            shifts_y - shift_y_uncertainties,
            shifts_y + shift_y_uncertainties,
            alpha=0.3,
            label="1σ confidence",
        )
        ax2.set_xlabel("Channel Index")
        ax2.set_ylabel("Y Shift [pixels]")
        ax2.set_title("Channel Y Alignment Shifts")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    #    @profile_resource_usage
    @module_decorator
    def manual(self, i, parameters, results):
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
        filepath = os.path.join(results["folder"], parameters["filename"])
        if os.path.exists(filepath):
            results["filepath"] = filepath
            results["success"] = True
        else:
            msg = "This is a manual step. Please provide input, "
            msg += "and re-execute the workflow. "
            msg += parameters["prompt"]
            msg += f" The resulting file should be {filepath}."
            results["message"] = msg
            logger.debug(msg)
            print(msg)
            results["success"] = False
            # raise ManualInputLackingError(f'{filepath} missing.')
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def summarize_dataset(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        for meth, meth_pars in parameters["methods"].items():
            if meth.lower() == "nena":
                try:
                    res, best_val = postprocess.nena(self.locs, self.info)
                    fp_plot = os.path.join(results["folder"], "nena.png")
                    self._plot_nena(res, fp_plot, pixelsize)
                    all_best_vals = {
                        "delta_a": res["best_values"]["delta_a"],
                        "s": res["best_values"]["s"],
                        "ac": res["best_values"]["ac"],
                        "dc": res["best_values"]["dc"],
                        "sc": res["best_values"]["sc"],
                    }
                    results["nena"] = {
                        "res": str(all_best_vals),
                        # "chisqr": res.chisqr,
                        "NeNa": (
                            f"{best_val:.3f} px;"
                            + f" {pixelsize*best_val:.3f} nm "
                        ),
                        "nena-px": best_val,
                        "nena-nm": pixelsize * best_val,
                        "filepath_plot": fp_plot,
                    }
                except ValueError as e:
                    logger.error(e)
                    results["nena"] = {"res": "Fitting Error", "best_vals": ""}
                except Exception as e:
                    logger.error(e)
                    results["nena"] = {
                        "res": str(e),
                        "best_vals": "Error.",
                    }
            elif meth.lower() == "median-loc-precision":
                median_lp = np.median(
                    np.stack(
                        (self.locs["lpx"], self.locs["lpy"]), axis=1
                    ).mean(1)
                )
                median_lp = median_lp / meth_pars.get("qe_correction", 1)
                results["median-loc-precision"] = {
                    "median_lp-px": median_lp,
                    "median_lp-nm": median_lp * pixelsize,
                }
            else:
                raise NotImplementedError(
                    f"Description method {meth} not implemented."
                )
        return parameters, results

    def _plot_nena(self, nena_result, filepath_plot, pixelsize=None):
        fig, ax = plt.subplots()
        d = nena_result["d"]
        if pixelsize is None:
            xlabel = "Distance [px]"
        else:
            d = d * pixelsize
            xlabel = "Distance [nm]"
        ax.set_title("Next frame neighbor distance histogram")
        ax.plot(d, nena_result["data"], label="Data")
        ax.plot(d, nena_result["best_fit"], label="Fit")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.legend(loc="best")
        fig.savefig(filepath_plot)

    # @module_decorator
    # def aggregate_cluster(self, i, parameters, results):
    #     """Aggregate along the 'cluster' column.
    #     Uses picasso.postprocess.cluster_combine
    #     Args:
    #         i : int
    #             the index of the module
    #         parameters: dict
    #             with required keys:
    #             and optional keys:
    #                 save_locs : bool
    #                     whether to save the locs into the results folder
    #         results : dict
    #             the results this function generates. This is created
    #             in the decorator wrapper
    #     """
    #     self.locs = postprocess.cluster_combine(self.locs)
    #     combined_info = {"Generated by": "Picasso Combine"}
    #     self.info.append(combined_info)
    #     results["nlocs"] = np.len(self.locs)
    #     if parameters.get("save"):
    #         self._save_locs(os.path.join(results["folder"], "locs.hdf5"))
    #     return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def density(self, i, parameters, results):
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
        self.locs = postprocess.compute_local_density(
            self.locs, self.info, parameters["radius"]
        )
        density_info = {
            "Generated by": "Picasso Density",
            "Wrapped by": "picasso-workflow : density",
            "Radius": float(parameters["radius"]),
        }
        self.info.append(density_info)
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
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
                        Number of localizations within radius to consider a given point
                        a core sample.
                    min_locs : int
                        Minimum number of localizations in a cluster. Clusters with
                        fewer localizations will be removed. Default is 0.
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
        pixelsize = int(self.pixelsize)
        radius = parameters["radius"] / pixelsize
        min_samples = parameters["min_samples"]
        min_locs = parameters["min_locs"]
        # label locs according to clusters
        self.locs = clusterer.dbscan(
            self.locs, radius, min_samples, min_locs, pixelsize)
        dbscan_info = {
            "Generated by": "Picasso DBSCAN",
            "Radius": radius,
            "Minimum number of locs": min_samples,
            "Wrapped by": "picasso-workflow : dbscan",
        }
        self.info.append(dbscan_info)
        # results["fp_locs"] = os.path.join(
        #     results["folder"], "locs_dbscan.hdf5"
        # )
        # self._save_locs(results["fp_locs"])

        # plot: histogram of cluster sizes
        fig, ax = plt.subplots()
        uniques, counts = np.unique(self.locs["group"], return_counts=True)
        maxbin = int(np.quantile(counts, 0.95))
        ax.hist(counts, bins=np.arange(maxbin))
        ax.set_xlabel("cluster size [locs]")
        ax.set_ylabel("Frequency")
        results["fp_fig_clustersizes"] = os.path.join(
            results["folder"], "fig_dbscan_clustersize.png"
        )
        fig.savefig(results["fp_fig_clustersizes"])

        cluster_centers = clusterer.find_cluster_centers(self.locs, pixelsize)
        results["fp_centers"] = os.path.join(
            results["folder"], "centers_dbscan.hdf5"
        )
        io.save_locs(results["fp_centers"], cluster_centers, self.info)
        if parameters["continue_with_centers"]:
            self.locs = cluster_centers

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def hdbscan(self, i, parameters, results):
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
        min_cluster = parameters["min_cluster"]
        min_samples = parameters["min_samples"]
        pixelsize = self.pixelsize

        # label locs according to clusters
        self.locs = clusterer.hdbscan(
            self.locs, min_cluster, min_samples, pixelsize
        )
        hdbscan_info = {
            "Generated by": "Picasso HDBSCAN",
            "Min. cluster": min_cluster,
            "Min. samples": min_samples,
            "Wrapped by": "picasso-workflow : hdbscan",
        }
        self.info.append(hdbscan_info)
        filepath = os.path.join(results["folder"], "locs_hdbscan.hdf5")
        self._save_locs(filepath)

        self.locs = clusterer.find_cluster_centers(self.locs, pixelsize)
        logger.warning("saving cluster centeras as locs. Is that intended?")

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def binding_event_analysis(self, i, parameters, results):
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
        folder_in, file_in = os.path.split(parameters["fp_locs"])
        file_nameonly, _ = os.path.splitext(file_in)
        Meas = outpost_modules.binding_event_analysis.Measurement(
            in_path=parameters["fp_locs"],
            save_path=results["folder"],
            saving_path=file_nameonly,
            total_n_frames=self.frames,
        )
        Meas.Begin()
        Meas.FileSaver()

        Plotter = outpost_modules.binding_event_plotting.Plotting(
            table_g=Meas.table_g,
            table_k=Meas.table_k,
            show=False,
            save=True,
            saving_name=file_nameonly,
            total_n_frames=self.frames,
        )
        _, fp = Plotter.Plot_photons()
        results["fig_photons"] = fp
        _, fp = Plotter.Plot_bg()
        results["fig_bg"] = fp
        _, fp = Plotter.Plot_sbr()
        results["fig_sbr"] = fp
        _, fp = Plotter.Plot_tb()
        results["fig_tb"] = fp
        _, fp = Plotter.plot_td()
        results["fig_td"] = fp
        _, fp = Plotter.plot_r()
        results["fig_r"] = fp
        _, fp = Plotter.Plot_locs()
        results["fig_locs"] = fp
        config = Plotter.saveAllResults()
        for k, v in config.items():
            results[k] = v

    #    @profile_resource_usage
    @module_decorator
    def resolution_analysis(self, i, parameters, results):
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
        from picasso_workflow.picasso_outpost import (
            analyse_resolution_ppac,
            resolution_ppac,
        )

        # Get parameters with defaults
        delta_r = parameters.get("delta_r", 5.0)  # 5 nm default
        r_max = parameters.get("r_max", 100.0)  # 100 nm default

        # Performance parameters
        batch_size = parameters.get(
            "batch_size", None
        )  # Auto-calculated if None
        n_processes = parameters.get(
            "n_processes", None
        )  # Auto-detected if None
        use_chunking = parameters.get(
            "use_chunking", True
        )  # Enable chunking by default
        use_sparse = parameters.get(
            "use_sparse", False
        )  # Sparse matrices for very large grids

        # Calculate autocorrelation with performance optimizations
        autocorr_map = resolution_ppac(
            self.locs,
            self.pixelsize,
            delta_r,
            r_max,
            batch_size=batch_size,
            n_processes=n_processes,
            use_chunking=use_chunking,
            use_sparse=use_sparse,
        )

        # Analyze autocorrelation with Gaussian fitting
        analysis_results = analyse_resolution_ppac(autocorr_map, delta_r)

        # Store 2D fit results
        results["resolution"] = analysis_results["resolution"]
        results["sigma_x"] = analysis_results["sigma_x"]
        results["sigma_y"] = analysis_results["sigma_y"]
        results["fwhm_x"] = analysis_results["fwhm_x"]
        results["fwhm_y"] = analysis_results["fwhm_y"]
        results["fit_quality"] = analysis_results["fit_quality"]
        results["autocorr_map"] = autocorr_map

        # Compute radial profile (optimized vectorized version)
        def compute_radial_profile(autocorr_map, sampling_resolution):
            """Compute radial profile of autocorrelation map using vectorized binning

            This optimized version uses np.digitize and np.bincount for 5-10x
            speedup compared to loop-based binning.

            Args:
                autocorr_map : ndarray
                    2D autocorrelation map
                sampling_resolution : float
                    pixel size in nm

            Returns:
                radial_profile : ndarray
                    averaged intensity values at each radial distance
                radial_distances : ndarray
                    distance values in nm
            """
            center = np.array(autocorr_map.shape) // 2
            y, x = np.ogrid[: autocorr_map.shape[0], : autocorr_map.shape[1]]

            # Compute distances once
            distances = (
                np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
                * sampling_resolution
            )
            max_radius = min(center) * sampling_resolution

            # Use np.digitize for efficient binning
            n_bins = min(center)
            radial_bins = np.linspace(0, max_radius, n_bins + 1)

            # Flatten arrays for vectorized operations
            distances_flat = distances.ravel()
            autocorr_flat = autocorr_map.ravel()

            # Bin the distances
            bin_indices = np.digitize(distances_flat, radial_bins)

            # Vectorized computation of bin averages using bincount
            bin_sums = np.bincount(
                bin_indices, weights=autocorr_flat, minlength=n_bins + 2
            )
            bin_counts = np.bincount(bin_indices, minlength=n_bins + 2)

            # Avoid division by zero
            valid_bins = bin_counts > 0
            radial_profile = np.zeros(n_bins + 2)
            radial_profile[valid_bins] = (
                bin_sums[valid_bins] / bin_counts[valid_bins]
            )

            # Calculate bin centers (exclude first and last bins which are outside range)
            radial_distances = (radial_bins[:-1] + radial_bins[1:]) / 2

            # Return valid bins (bins 1 through n_bins, excluding boundary bins)
            return radial_profile[1 : n_bins + 1], radial_distances

        radial_profile, radial_distances = compute_radial_profile(
            autocorr_map, delta_r
        )

        # Store radial profile
        results["radial_profile"] = radial_profile
        results["radial_distances"] = radial_distances

        # Fit Gaussian to radial profile (reuse from resolution_autocorr)
        from scipy.optimize import curve_fit

        def gaussian_1d(x, amplitude, sigma, background):
            """1D Gaussian function for radial profile fitting"""
            return (
                amplitude * np.exp(-((x) ** 2) / (2 * sigma**2)) + background
            )

        def dblgaussian_1d(
            x, amplitude_1, amplitude_2, sigma_1, sigma_2, background
        ):
            """Double Gaussian function for radial profile fitting"""
            return (
                amplitude_1 * np.exp(-((x) ** 2) / (2 * sigma_1**2))
                + amplitude_2 * np.exp(-((x) ** 2) / (2 * sigma_2**2))
            ) + background

        # Fit 1D Gaussian to radial profile
        try:
            center_peak = radial_profile[0]
            background_est = (
                np.mean(radial_profile[-5:]) if len(radial_profile) > 5 else 0
            )
            p0_radial = [center_peak - background_est, 1.0, background_est]

            fit_range = radial_distances < r_max / 2
            if np.sum(fit_range) < 3:
                fit_range = slice(min(8, len(radial_distances)))

            popt_radial, _ = curve_fit(
                gaussian_1d,
                radial_distances[fit_range],
                radial_profile[fit_range],
                p0=p0_radial,
                maxfev=2000,
            )

            sigma_radial = abs(popt_radial[1])
            resolution_radial = sigma_radial * 2.355
            radial_fit_success = True
            # Cache the fit curve to avoid recomputation in plotting
            radial_fit_curve = gaussian_1d(radial_distances, *popt_radial)
            logger.debug(
                f"  Radial fit: σ = {sigma_radial:.2f} nm, FWHM = {resolution_radial:.2f} nm"
            )

        except Exception as e:
            logger.debug(f"  Radial fit failed: {e}")
            radial_fit_success = False
            sigma_radial = np.nan
            resolution_radial = np.nan
            popt_radial = [np.nan] * 3
            radial_fit_curve = None

        # Fit double Gaussian to radial profile
        try:
            center_peak = radial_profile[0]
            background_est = (
                np.mean(radial_profile[-5:]) if len(radial_profile) > 5 else 0
            )
            total_amp = center_peak - background_est
            p0_dblradial = [
                0.9 * total_amp,
                0.1 * total_amp,
                1.0,
                20,
                background_est,
            ]
            bounds_lo = [0.6 * total_amp, 0, 0.5, 15, 0]
            bounds_hi = [
                1.1 * total_amp,
                0.4 * total_amp,
                10,
                50,
                1.5 * background_est,
            ]

            fit_range = radial_distances < r_max / 2
            if np.sum(fit_range) < 3:
                fit_range = slice(min(8, len(radial_distances)))

            popt_dblradial, _ = curve_fit(
                dblgaussian_1d,
                radial_distances[fit_range],
                radial_profile[fit_range],
                p0=p0_dblradial,
                bounds=(bounds_lo, bounds_hi),
                maxfev=2000,
            )

            sigma_dblradial = abs(popt_dblradial[2])
            resolution_dblradial = sigma_dblradial * 2.355
            dblradial_fit_success = True
            # Cache the fit curve to avoid recomputation in plotting
            dblradial_fit_curve = dblgaussian_1d(
                radial_distances, *popt_dblradial
            )
            logger.debug(
                f"  Double Radial fit: σ = {sigma_dblradial:.2f} nm, FWHM = {resolution_dblradial:.2f} nm"
            )

        except Exception as e:
            logger.debug(f"  Double Radial fit failed: {e}")
            dblradial_fit_success = False
            sigma_dblradial = np.nan
            resolution_dblradial = np.nan
            popt_dblradial = [np.nan] * 5
            dblradial_fit_curve = None

        # Store radial fit results
        results["resolution_radial"] = resolution_radial
        results["sigma_radial"] = sigma_radial
        results["resolution_dblradial"] = resolution_dblradial
        results["sigma_dblradial"] = sigma_dblradial

        # Create main figure with 2D autocorrelation and radial profile
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: 2D autocorrelation map
        extent = [-r_max, r_max, -r_max, r_max]
        im1 = ax1.imshow(
            autocorr_map, extent=extent, origin="lower", cmap="hot"
        )
        ax1.set_xlabel("Δx (nm)")
        ax1.set_ylabel("Δy (nm)")
        ax1.set_title("2D Autocorrelation")
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Plot 2: Radial profile with fits
        ax2.plot(
            radial_distances, radial_profile, "b-", linewidth=2, label="Data"
        )

        # Add radial Gaussian fit if successful (using cached curve)
        if radial_fit_success and radial_fit_curve is not None:
            ax2.plot(
                radial_distances,
                radial_fit_curve,
                "r--",
                linewidth=2,
                label="Fit",
            )

        # Add double Gaussian fit if successful (using cached curve)
        if dblradial_fit_success and dblradial_fit_curve is not None:
            ax2.plot(
                radial_distances,
                dblradial_fit_curve,
                "m--",
                linewidth=2,
                label="Double Fit",
            )

        # Set title with resolution values
        title_parts = ["Radial Profile"]
        if radial_fit_success and dblradial_fit_success:
            title_parts.append(
                f"FWHM: {resolution_radial:.2f} nm | {resolution_dblradial:.2f} nm"
            )
        elif radial_fit_success:
            title_parts.append(f"FWHM: {resolution_radial:.2f} nm")
        elif dblradial_fit_success:
            title_parts.append(f"FWHM: {resolution_dblradial:.2f} nm")

        ax2.set_title("\n".join(title_parts))
        ax2.set_xlabel("Distance (nm)")
        ax2.set_ylabel("Autocorrelation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save main plot
        plot_path = os.path.join(results["folder"], "resolution_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        results["fig_resolution"] = plot_path

        # Create separate radial profile plot (similar to resolution_autocorr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(radial_distances, radial_profile, "b-", linewidth=2)

        if radial_fit_success and radial_fit_curve is not None:
            ax.plot(radial_distances, radial_fit_curve, "r--", linewidth=2)
            ax.set_title(
                f"Radial Autocorr (Resolution: {resolution_radial:.2f} nm FWHM)"
            )
        else:
            ax.set_title("Radial Autocorrelation")

        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)

        plot_path_radial = os.path.join(
            results["folder"], "resolution_analysis_radial.png"
        )
        plt.savefig(plot_path_radial, dpi=300, bbox_inches="tight")
        plt.close()

        results["fig_radial"] = plot_path_radial

        return parameters, results

    @staticmethod
    def _compute_frame_to_dataset_shift(frame_data):
        """Compute RSSO shift of single frame against whole dataset (multiprocessing worker)

        Args:
            frame_data : tuple
                (frame_idx, frame_locs, dataset_locs, max_shift, min_locs_per_frame, ton, toff)

        Returns:
            tuple : (frame_idx, shift_x, shift_y, uncertainty_x, uncertainty_y, confidence, quality)
        """
        from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

        (
            frame_idx,
            frame_locs,
            dataset_locs,
            max_shift,
            min_locs_per_frame,
            ton,
            toff,
        ) = frame_data

        # Skip frames with insufficient localizations
        if len(frame_locs) < min_locs_per_frame:
            return (frame_idx, None, None, None, None, 0.0, 0.0, None)

        try:
            # Calculate RSSO shift between frame and whole dataset
            shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
                dataset_locs, frame_locs, max_shift, plot_histogram=False
            )

            if shift_x is not None and shift_y is not None:
                # Extract uncertainty information from _calculate_pairwise_shift
                uncertainty_x = (
                    uncertainty_info.get("shift_x_error", np.nan)
                    if uncertainty_info
                    else np.nan
                )
                uncertainty_y = (
                    uncertainty_info.get("shift_y_error", np.nan)
                    if uncertainty_info
                    else np.nan
                )

                # Calculate confidence based on number of localizations and uncertainty
                n_locs_frame = len(frame_locs)
                n_locs_dataset = len(dataset_locs)

                # Simple confidence metric based on localization count and uncertainty
                if not (np.isnan(uncertainty_x) or np.isnan(uncertainty_y)):
                    confidence = min(
                        1.0,
                        (n_locs_frame / 50.0)
                        * (
                            1.0
                            / (
                                1.0
                                + np.sqrt(uncertainty_x**2 + uncertainty_y**2)
                            )
                        ),
                    )
                else:
                    confidence = min(1.0, n_locs_frame / 50.0)

                quality = n_locs_frame * confidence

                return (
                    frame_idx,
                    shift_x,
                    shift_y,
                    uncertainty_x,
                    uncertainty_y,
                    confidence,
                    quality,
                )
            else:
                return (frame_idx, None, None, None, None, 0.0, 0.0, None)

        except Exception as e:
            print(f"      Frame {frame_idx} RSSO failed: {e}")
            return (frame_idx, None, None, None, None, 0.0, 0.0, None)

    @staticmethod
    def _compute_frame_to_dataset_shift_memory_efficient(frame_data):
        """Memory-efficient RSSO shift computation using frame indices instead of data copies

        Args:
            frame_data : tuple
                (frame_idx, locs_array, target_frame, max_shift, min_locs_per_frame, ton, toff)

        Returns:
            tuple : (frame_idx, shift_x, shift_y, uncertainty_x, uncertainty_y, confidence, quality)
        """
        import numpy as np

        from picasso_workflow.picasso_outpost import _calculate_pairwise_shift

        (
            frame_idx,
            locs_array,
            target_frame,
            max_shift,
            min_locs_per_frame,
            ton,
            toff,
        ) = frame_data

        try:
            # Extract frame localizations on-demand (no pre-created copies)
            frame_mask = locs_array["frame"] == target_frame
            frame_locs = locs_array[frame_mask]

            # Skip frames with insufficient localizations
            if len(frame_locs) < min_locs_per_frame:
                return (frame_idx, None, None, None, None, 0.0, 0.0, None)

            # Extract dataset (all other frames) on-demand
            dataset_mask = locs_array["frame"] != target_frame
            dataset_locs = locs_array[dataset_mask]

            # Calculate RSSO shift between frame and whole dataset
            shift_x, shift_y, _, uncertainty_info = _calculate_pairwise_shift(
                dataset_locs, frame_locs, max_shift, plot_histogram=False
            )

            if shift_x is not None and shift_y is not None:
                # Extract uncertainty information from _calculate_pairwise_shift
                uncertainty_x = (
                    uncertainty_info.get("shift_x_error", np.nan)
                    if uncertainty_info
                    else np.nan
                )
                uncertainty_y = (
                    uncertainty_info.get("shift_y_error", np.nan)
                    if uncertainty_info
                    else np.nan
                )

                # Calculate confidence based on number of localizations and uncertainty
                n_locs_frame = len(frame_locs)
                n_locs_dataset = len(dataset_locs)

                # Simple confidence metric based on localization count and uncertainty
                if not (np.isnan(uncertainty_x) or np.isnan(uncertainty_y)):
                    uncertainty_magnitude = np.sqrt(
                        uncertainty_x**2 + uncertainty_y**2
                    )
                    confidence = min(
                        1.0,
                        (n_locs_frame / 100.0) / (1.0 + uncertainty_magnitude),
                    )
                else:
                    confidence = min(1.0, n_locs_frame / 100.0)

                quality = (
                    n_locs_frame + n_locs_dataset
                )  # Simple quality metric

                return (
                    frame_idx,
                    shift_x,
                    shift_y,
                    uncertainty_x,
                    uncertainty_y,
                    confidence,
                    quality,
                )
            else:
                return (frame_idx, None, None, None, None, 0.0, 0.0, None)

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            return (frame_idx, None, None, None, None, 0.0, 0.0, None)

    @staticmethod
    def _process_autocorr_chunk(chunk_data):
        """Process a single spatial chunk for autocorrelation analysis (multiprocessing worker)

        Args:
            chunk_data : tuple
                (chunk_bounds, x_coords, y_coords, sampling_res, max_shift_pixels, min_locs_per_chunk, chunk_idx)

        Returns:
            dict or None : Chunk result with autocorr, n_locs, bounds, and chunk_idx
        """
        (
            chunk_bounds,
            chunk_x,  # x_coords,
            chunk_y,  # y_coords,
            sampling_res,
            max_shift_pixels,
            min_locs_per_chunk,
            chunk_idx,
        ) = chunk_data
        x_min, x_max, y_min, y_max = chunk_bounds

        # # Extract localizations in this chunk
        # mask = (
        #     (x_coords >= x_min)
        #     & (x_coords < x_max)
        #     & (y_coords >= y_min)
        #     & (y_coords < y_max)
        # )

        # chunk_x = x_coords[mask]
        # chunk_y = y_coords[mask]
        n_locs = len(chunk_x)

        if n_locs < min_locs_per_chunk:
            return None

        try:
            # Create histogram for this chunk
            x_bins = np.arange(x_min, x_max + sampling_res, sampling_res)
            y_bins = np.arange(y_min, y_max + sampling_res, sampling_res)

            chunk_hist, _, _ = np.histogram2d(
                chunk_x, chunk_y, bins=[x_bins, y_bins]
            )
            chunk_hist = chunk_hist.astype(np.float32)

            if np.sum(chunk_hist) == 0:
                return None

            # Compute autocorrelation using efficient FFT
            F_hist = np.fft.fft2(chunk_hist)
            autocorr_full = np.fft.fftshift(
                np.real(np.fft.ifft2(F_hist * np.conj(F_hist)))
            )

            # Extract central autocorr region
            center = np.array(autocorr_full.shape) // 2
            safe_shift = min(max_shift_pixels, min(center))

            autocorr_chunk = autocorr_full[
                center[0] - safe_shift : center[0] + safe_shift + 1,
                center[1] - safe_shift : center[1] + safe_shift + 1,
            ].copy()

            # Normalize
            if autocorr_chunk.max() > 0:
                autocorr_chunk = autocorr_chunk / autocorr_chunk.max()
                # remove center point
                center = np.array(autocorr_chunk.shape) // 2
                meanmax = np.mean(
                    [
                        autocorr_chunk[center[0], center[1] - 1],
                        autocorr_chunk[center[0] - 1, center[1]],
                        autocorr_chunk[center[0], center[1] + 1],
                        autocorr_chunk[center[0] + 1, center[1]],
                    ]
                )
                autocorr_chunk[center[0], center[1]] = meanmax

            return {
                "autocorr": autocorr_chunk,
                "n_locs": n_locs,
                "bounds": (x_min, x_max, y_min, y_max),
                "chunk_idx": chunk_idx,
            }

        except Exception as e:
            print(f"      Chunk {chunk_idx} failed: {e}")
            return None

    #    @profile_resource_usage
    @module_decorator
    def resolution_autocorr(self, i, parameters, results):
        """Calculate resolution using 2D autocorrelation analysis

        This method estimates spatial resolution by creating small spatial chunks,
        computing autocorrelation on each chunk, then combining them weighted by
        localization count. This preserves exact sampling resolution while managing memory.

        Args:
            i : int
                the index of the module
            parameters: dict
                with optional keys:
                    sampling_res : float
                        histogram sampling resolution in nm (default: 0.5)
                    max_shift : float
                        maximum autocorrelation shift in nm (default: 10.0)
                    chunk_size_nm : float
                        spatial chunk size in nm (default: 5000, i.e., 5 μm)
                    min_locs_per_chunk : int
                        minimum localizations per chunk (default: 500)
                    max_memory_gb : float
                        maximum memory limit in GB (default: 2.0)
                    n_processes : int or None
                        number of parallel processes (auto-detected if None)

        Results:
            resolution : float
                estimated resolution in nm (FWHM)
            sigma_x, sigma_y : float
                Gaussian standard deviations in x,y directions
            fwhm_x, fwhm_y : float
                Full-width half-maximum in x,y directions
            autocorr_2d : ndarray
                2D autocorrelation map (weighted average)
            radial_profile : ndarray
                radial profile of autocorrelation
            radial_distances : ndarray
                distance values for radial profile
            fig_autocorr_2d : str
                path to 2D autocorrelation plot
            fig_radial : str
                path to radial profile plot
        """
        from scipy.optimize import curve_fit

        # Get parameters with defaults
        sampling_res = parameters.get("sampling_res", 0.5)  # 0.5 nm sampling
        max_shift = parameters.get("max_shift", 10.0)  # 10 nm max shift
        chunk_size_nm = parameters.get("chunk_size_nm", 5000)  # 5 μm chunks
        min_locs_per_chunk = parameters.get(
            "min_locs_per_chunk", 500
        )  # Min locs per chunk
        max_memory_gb = parameters.get(
            "max_memory_gb", 2.0
        )  # Smaller memory limit
        n_processes = parameters.get("n_processes", None)

        if n_processes is None:
            n_processes = min(mp.cpu_count(), 4)  # Limit processes for memory

        logger.debug(f"Computing resolution using autocorrelation analysis...")
        logger.debug(
            f"  Sampling resolution: {sampling_res} nm (preserved exactly)"
        )
        logger.debug(f"  Maximum shift: {max_shift} nm")
        logger.debug(f"  Chunk size: {chunk_size_nm/1000:.1f} μm")
        logger.debug(f"  Memory limit: {max_memory_gb} GB per chunk")

        # Extract coordinates
        x_coords = self.locs["x"] * self.pixelsize  # Convert to nm
        y_coords = self.locs["y"] * self.pixelsize

        logger.debug(f"  Processing {len(x_coords)} localizations")
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        logger.debug(
            f"  Field size: {x_range/1000:.1f} × {y_range/1000:.1f} μm"
        )

        # Calculate chunking grid
        n_chunks_x = max(1, int(np.ceil(x_range / chunk_size_nm)))
        n_chunks_y = max(1, int(np.ceil(y_range / chunk_size_nm)))
        total_chunks = n_chunks_x * n_chunks_y

        logger.debug(
            f"  Using {n_chunks_x} × {n_chunks_y} = {total_chunks} spatial chunks"
        )

        # Memory estimate per chunk
        chunk_pixels = int(chunk_size_nm / sampling_res)
        max_shift_pixels = int(np.ceil(max_shift / sampling_res))
        autocorr_size = 2 * max_shift_pixels + 1

        chunk_memory_gb = (chunk_pixels**2 * 4 * 8) / (
            1024**3
        )  # float32 * 4 arrays
        logger.debug(f"  Estimated memory per chunk: {chunk_memory_gb:.2f} GB")

        if chunk_memory_gb > max_memory_gb:
            # Reduce chunk size to fit memory
            new_chunk_size = (
                np.sqrt(max_memory_gb * (1024**3) / (4 * 8)) * sampling_res
            )
            chunk_size_nm = max(2000, new_chunk_size)  # At least 2 μm
            logger.debug(
                f"  ⚠ Reducing chunk size to {chunk_size_nm/1000:.1f} μm to fit memory"
            )
            n_chunks_x = max(1, int(np.ceil(x_range / chunk_size_nm)))
            n_chunks_y = max(1, int(np.ceil(y_range / chunk_size_nm)))
            total_chunks = n_chunks_x * n_chunks_y

            logger.debug(
                f"  Using {n_chunks_x} × {n_chunks_y} = {total_chunks} spatial chunks"
            )

        # Determine number of processes
        n_processes = parameters.get("n_processes", min(mp.cpu_count(), 8))
        logger.debug(f"  Using {n_processes} processes for chunk processing")

        # Generate chunk boundaries and prepare data for multiprocessing
        x_min_global, y_min_global = x_coords.min(), y_coords.min()

        logger.debug(
            f"  Preparing {total_chunks} chunks for parallel processing..."
        )

        # Prepare all chunk data for multiprocessing
        chunk_data_list = []
        for i in range(n_chunks_x):
            for j in range(n_chunks_y):
                chunk_idx = i * n_chunks_y + j + 1

                # Define chunk boundaries
                x_chunk_min = x_min_global + i * chunk_size_nm
                x_chunk_max = min(x_chunk_min + chunk_size_nm, x_coords.max())
                y_chunk_min = y_min_global + j * chunk_size_nm
                y_chunk_max = min(y_chunk_min + chunk_size_nm, y_coords.max())

                chunk_bounds = (
                    x_chunk_min,
                    x_chunk_max,
                    y_chunk_min,
                    y_chunk_max,
                )
                # Extract localizations in this chunk
                mask = (
                    (x_coords >= x_chunk_min)
                    & (x_coords < x_chunk_max)
                    & (y_coords >= y_chunk_min)
                    & (y_coords < y_chunk_max)
                )
                chunk_x = x_coords[mask]
                chunk_y = y_coords[mask]

                chunk_data = (
                    chunk_bounds,
                    chunk_x,
                    chunk_y,
                    sampling_res,
                    max_shift_pixels,
                    min_locs_per_chunk,
                    chunk_idx,
                )
                chunk_data_list.append(chunk_data)

        logger.debug(
            f"  Processing chunks with multiprocessing ({n_processes} processes)..."
        )

        # Process chunks in parallel
        chunk_results = []
        valid_chunks = 0

        with mp.Pool(processes=n_processes) as pool:
            # Submit all chunk processing jobs
            results_mp = pool.map(
                self._process_autocorr_chunk, chunk_data_list
            )

            # Collect valid results
            for result in results_mp:
                if result is not None:
                    chunk_results.append(result)
                    valid_chunks += 1
                    chunk_idx = result["chunk_idx"]
                    logger.debug(
                        f"      Chunk {chunk_idx}: {result['n_locs']} locs, peak: {result['autocorr'].max():.3f}"
                    )

        logger.debug(f"  Parallel processing completed.")
        gc.collect()  # Clean up after multiprocessing

        total_locs_processed = sum(r["n_locs"] for r in chunk_results)
        logger.debug(f"  Processed {valid_chunks}/{total_chunks} chunks")
        logger.debug(f"  Total localizations: {total_locs_processed:,}")

        if valid_chunks == 0:
            logger.debug("  ⚠ No valid chunks found!")
            results["resolution"] = np.nan
            results["sigma_x"] = results["sigma_y"] = np.nan
            results["fwhm_x"] = results["fwhm_y"] = np.nan
            results["autocorr_2d"] = np.zeros((21, 21))
            results["radial_profile"] = np.array([])
            results["radial_distances"] = np.array([])
            return parameters, results

        # Combine autocorrelations with proper weighting
        logger.debug(f"  Combining {valid_chunks} chunk autocorrelations...")

        # Calculate weights based on localization count
        weights = np.array(
            [r["n_locs"] for r in chunk_results], dtype=np.float64
        )
        weights = weights / weights.sum()

        logger.debug(
            f"    Chunk weights range: {weights.min():.3f} - {weights.max():.3f}"
        )

        # Find common autocorr size (use smallest)
        autocorr_sizes = [r["autocorr"].shape[0] for r in chunk_results]
        min_size = min(autocorr_sizes)

        # Combine weighted autocorrelations
        combined_autocorr = np.zeros((min_size, min_size), dtype=np.float64)

        for i, result in enumerate(chunk_results):
            chunk_autocorr = result["autocorr"]

            # Crop to common size if needed
            if chunk_autocorr.shape[0] > min_size:
                center = chunk_autocorr.shape[0] // 2
                half_min = min_size // 2
                chunk_autocorr = chunk_autocorr[
                    center - half_min : center - half_min + min_size,
                    center - half_min : center - half_min + min_size,
                ]

            combined_autocorr += weights[i] * chunk_autocorr

        # Clean up chunk data
        del chunk_results, x_coords, y_coords
        gc.collect()

        autocorr_2d = combined_autocorr
        logger.debug(f"  Combined autocorr peak: {autocorr_2d.max():.3f}")

        # Calculate radial profile
        def compute_radial_profile(autocorr_map, sampling_resolution):
            center = np.array(autocorr_map.shape) // 2
            y, x = np.ogrid[: autocorr_map.shape[0], : autocorr_map.shape[1]]

            distances = (
                np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
                * sampling_resolution
            )
            max_radius = min(center) * sampling_resolution
            radial_bins = np.linspace(0, max_radius, min(center))

            radial_profile = np.zeros(len(radial_bins) - 1)
            radial_distances = np.zeros(len(radial_bins) - 1)

            for i in range(len(radial_bins) - 1):
                r1, r2 = radial_bins[i], radial_bins[i + 1]
                mask = (distances >= r1) & (distances < r2)
                if np.sum(mask) > 0:
                    radial_profile[i] = np.mean(autocorr_map[mask])
                radial_distances[i] = (r1 + r2) / 2

            return radial_profile, radial_distances

        radial_profile, radial_distances = compute_radial_profile(
            autocorr_2d, sampling_res
        )

        # Fit Gaussian to extract resolution
        def gaussian_1d(x, amplitude, sigma, background):
            return (
                amplitude * np.exp(-((x) ** 2) / (2 * sigma**2)) + background
            )

        # Fit double Gaussian to extract resolution
        def dblgaussian_1d(
            x, amplitude_1, amplitude_2, sigma_1, sigma_2, background
        ):
            return (
                amplitude_1 * np.exp(-((x) ** 2) / (2 * sigma_1**2))
                + amplitude_2 * np.exp(-((x) ** 2) / (2 * sigma_2**2))
            ) + background

        def gaussian_2d_fit(
            xy, amplitude, x0, y0, sigma_x, sigma_y  # , background
        ):
            x, y = xy
            return (
                amplitude
                * np.exp(
                    -(
                        (x - x0) ** 2 / (2 * sigma_x**2)
                        + (y - y0) ** 2 / (2 * sigma_y**2)
                    )
                )
                # + background
            ).ravel()

        # Fit 1D Gaussian to radial profile
        try:
            center_peak = radial_profile[0]
            background_est = (
                np.mean(radial_profile[-5:]) if len(radial_profile) > 5 else 0
            )
            # p0_radial = [center_peak - background_est, 0, 1.0, background_est]
            p0_radial = [center_peak - background_est, 1.0, background_est]

            fit_range = radial_distances < max_shift / 2
            if np.sum(fit_range) < 3:
                fit_range = slice(min(8, len(radial_distances)))

            popt_radial, _ = curve_fit(
                gaussian_1d,
                radial_distances[fit_range],
                radial_profile[fit_range],
                p0=p0_radial,
                maxfev=2000,
            )

            sigma_radial = abs(popt_radial[1])
            resolution_radial = sigma_radial * 2.355
            radial_fit_success = True
            logger.debug(
                f"  Radial fit: σ = {sigma_radial:.2f} nm, FWHM = {resolution_radial:.2f} nm"
            )

        except Exception as e:
            logger.debug(f"  Radial fit failed: {e}")
            radial_fit_success = False
            sigma_radial = np.nan
            resolution_radial = np.nan
            popt_radial = [np.nan] * 3

        # Fit 1D double Gaussian to radial profile
        try:
            center_peak = radial_profile[0]
            background_est = (
                np.mean(radial_profile[-5:]) if len(radial_profile) > 5 else 0
            )
            # p0_radial = [center_peak - background_est, 0, 1.0, background_est]
            total_amp = center_peak - background_est
            p0_dblradial = [
                0.9 * total_amp,
                0.1 * total_amp,
                1.0,
                20,
                background_est,
            ]
            bounds_lo = [0.6 * total_amp, 0, 0.5, 15, 0]
            bounds_hi = [
                1.1 * total_amp,
                0.4 * total_amp,
                10,
                50,
                1.5 * background_est,
            ]

            fit_range = radial_distances < max_shift / 2
            if np.sum(fit_range) < 3:
                fit_range = slice(min(8, len(radial_distances)))

            popt_dblradial, _ = curve_fit(
                dblgaussian_1d,
                radial_distances[fit_range],
                radial_profile[fit_range],
                p0=p0_dblradial,
                bounds=(bounds_lo, bounds_hi),
                maxfev=2000,
            )

            sigma_dblradial = abs(popt_dblradial[2])
            resolution_dblradial = sigma_dblradial * 2.355
            dblradial_fit_success = True
            logger.debug(
                f"  Double Radial fit: σ = {sigma_dblradial:.2f} nm, FWHM = {resolution_dblradial:.2f} nm"
            )

        except Exception as e:
            logger.debug(f"  Radial fit failed: {e}")
            radial_fit_success = False
            sigma_radial = np.nan
            resolution_radial = np.nan
            popt_radial = [np.nan] * 3

        # Fit 2D Gaussian
        try:
            extent = (min_size // 2) * sampling_res
            x_2d = np.linspace(-extent, extent, autocorr_2d.shape[1])
            y_2d = np.linspace(-extent, extent, autocorr_2d.shape[0])
            X_2d, Y_2d = np.meshgrid(x_2d, y_2d)

            peak_val = autocorr_2d.max()
            background_2d = np.quantile(autocorr_2d, 0.1)
            background_2d = 0
            p0_2d = [
                peak_val - background_2d,
                0,
                0,
                1.0,
                1.0,
            ]  # , background_2d]

            fit_size = min(autocorr_2d.shape[0] // 3, 15)
            center_2d = autocorr_2d.shape[0] // 2
            fit_region = slice(center_2d - fit_size, center_2d + fit_size + 1)

            popt_2d, _ = curve_fit(
                gaussian_2d_fit,
                (X_2d[fit_region, fit_region], Y_2d[fit_region, fit_region]),
                autocorr_2d[fit_region, fit_region].ravel(),
                p0=p0_2d,
                maxfev=2000,
            )

            sigma_x = abs(popt_2d[3])
            sigma_y = abs(popt_2d[4])
            fwhm_x = sigma_x * 2.355
            fwhm_y = sigma_y * 2.355
            resolution_2d = np.sqrt(fwhm_x * fwhm_y)
            fit_2d_success = True
            logger.debug(
                f"  2D fit: σx = {sigma_x:.2f}, σy = {sigma_y:.2f} nm, resolution = {resolution_2d:.2f} nm"
            )

        except Exception as e:
            logger.debug(f"  2D fit failed: {e}")
            fit_2d_success = False
            sigma_x = sigma_y = np.nan
            fwhm_x = fwhm_y = np.nan
            resolution_2d = np.nan
            popt_2d = [np.nan] * 6

        # Store results
        if fit_2d_success:
            results["resolution"] = resolution_2d
            results["sigma_x"] = sigma_x
            results["sigma_y"] = sigma_y
            results["fwhm_x"] = fwhm_x
            results["fwhm_y"] = fwhm_y
        elif radial_fit_success:
            results["resolution"] = resolution_radial
            results["sigma_x"] = sigma_radial
            results["sigma_y"] = sigma_radial
            results["fwhm_x"] = resolution_radial
            results["fwhm_y"] = resolution_radial
        else:
            results["resolution"] = np.nan
            results["sigma_x"] = np.nan
            results["sigma_y"] = np.nan
            results["fwhm_x"] = np.nan
            results["fwhm_y"] = np.nan

        results["autocorr_2d"] = autocorr_2d
        results["radial_profile"] = radial_profile
        results["radial_distances"] = radial_distances

        # Create plots (simplified to save memory)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Plot 1: 2D autocorrelation
        extent_nm = (min_size // 2) * sampling_res
        im1 = ax1.imshow(
            autocorr_2d,
            extent=[-extent_nm, extent_nm, -extent_nm, extent_nm],
            origin="lower",
            cmap="hot",
        )
        ax1.set_xlabel("Δx (nm)")
        ax1.set_ylabel("Δy (nm)")
        ax1.set_title("2D Autocorrelation")
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Plot 2: Radial profile
        ax2.plot(
            radial_distances, radial_profile, "b-", linewidth=2, label="Data"
        )
        if radial_fit_success:
            radial_fit = gaussian_1d(radial_distances, *popt_radial)
            ax2.plot(
                radial_distances, radial_fit, "r--", linewidth=2, label="Fit"
            )
            ax2.set_title(f"Radial Profile (FWHM: {resolution_radial:.2f} nm)")
        else:
            ax2.set_title("Radial Profile")
        if dblradial_fit_success:
            dblradial_fit = dblgaussian_1d(radial_distances, *popt_dblradial)
            ax2.plot(
                radial_distances,
                dblradial_fit,
                "m--",
                linewidth=2,
                label="Double Fit",
            )
            ax2.set_title(
                f"Radial Profile (FWHM: {resolution_radial:.2f} nm \
                | {resolution_dblradial:.2f})"
            )
        else:
            ax2.set_title("Radial Profile")

        ax2.set_xlabel("Distance (nm)")
        ax2.set_ylabel("Autocorrelation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plots
        plot_path_2d = os.path.join(
            results["folder"], "resolution_autocorr_2d.png"
        )
        plt.savefig(plot_path_2d, dpi=300, bbox_inches="tight")
        plt.close()

        # Simple radial plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(radial_distances, radial_profile, "b-", linewidth=2)
        if radial_fit_success:
            radial_fit = gaussian_1d(radial_distances, *popt_radial)
            ax.plot(radial_distances, radial_fit, "r--", linewidth=2)
            ax.set_title(
                f"Radial Autocorr (Resolution: {resolution_radial:.2f} nm FWHM)"
            )
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)

        plot_path_radial = os.path.join(
            results["folder"], "resolution_autocorr_radial.png"
        )
        plt.savefig(plot_path_radial, dpi=300, bbox_inches="tight")
        plt.close()

        results["fig_autocorr_2d"] = plot_path_2d
        results["fig_radial"] = plot_path_radial

        gc.collect()
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def resolution_frc(self, i, parameters, results):
        """Calculate resolution using Fourier Ring Correlation (FRC)

        This method splits localizations into two random subsets, renders them
        into images, and computes their Fourier ring correlation to estimate
        spatial resolution.

        Phase 2 optimizations:
        - use_chunking: Enables memory-efficient chunked rendering for large FOVs
        - n_splits: Averages over multiple random splits for robustness
        - max_frc_range_nm: Limit FRC calculation range for speedup
        - parallel_splits: Parallelize split processing (best when use_chunking=False)

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
                    seed : int or None
                        random seed for reproducibility (default: None)
                    use_chunking : bool
                        use chunked rendering for large images (default: False)
                    chunk_size_nm : float
                        chunk size in nm for chunked rendering (default: 10000)
                    n_splits : int
                        number of splits to average (default: 1, set >1 for robustness)
                    n_processes : int
                        number of parallel processes (default: 4)
                    max_frc_range_nm : float or None
                        maximum range to compute in nm (default: None = full range)
                        Setting this (e.g., 25 nm) speeds up FRC calculation
                    parallel_splits : bool
                        process splits in parallel (default: False)
                        Only beneficial when use_chunking=False

        Results:
            resolution_frc : float
                FRC-based resolution in nm
            resolution_std : float
                standard deviation (only if n_splits > 1)
            cutoff_frequency : float
                spatial frequency at resolution cutoff (1/nm)
            frc_curve : ndarray
                FRC values as function of spatial frequency
            frc_curve_std : ndarray
                std of FRC curve (only if n_splits > 1)
            spatial_frequencies : ndarray
                spatial frequency values (1/nm)
            threshold : float
                threshold used
            fig_frc : str
                path to FRC curve plot
            fig_images : str
                path to split images comparison plot (only if n_splits=1)
        """
        from picasso_workflow.outpost_modules.resolution_frc import (
            compute_frc_averaged,
            compute_frc_resolution,
        )

        # Get parameters with defaults
        pixelsize_render = parameters.get("pixelsize_render", 5.0)
        smoothing_sigma = parameters.get("smoothing_sigma", None)
        threshold = parameters.get("threshold", 1.0 / 7.0)
        seed = parameters.get("seed", None)
        use_chunking = parameters.get("use_chunking", False)
        chunk_size_nm = parameters.get("chunk_size_nm", 10000)
        n_splits = parameters.get("n_splits", 1)
        n_processes = parameters.get("n_processes", 4)
        max_frc_range_nm = parameters.get("max_frc_range_nm", None)
        parallel_splits = parameters.get("parallel_splits", False)

        # Choose pipeline based on n_splits
        if n_splits > 1:
            # Use averaging pipeline
            logger.debug(f"Using averaged FRC with {n_splits} splits")
            frc_results = compute_frc_averaged(
                self.locs,
                self.pixelsize,
                pixelsize_render=pixelsize_render,
                smoothing_sigma=smoothing_sigma,
                threshold=threshold,
                n_splits=n_splits,
                n_processes=n_processes,
                use_chunking=use_chunking,
                chunk_size_nm=chunk_size_nm,
                max_frc_range_nm=max_frc_range_nm,
                parallel_splits=parallel_splits,
            )
        else:
            # Use single-split pipeline
            frc_results = compute_frc_resolution(
                self.locs,
                self.pixelsize,
                pixelsize_render=pixelsize_render,
                smoothing_sigma=smoothing_sigma,
                threshold=threshold,
                seed=seed,
                max_frc_range_nm=max_frc_range_nm,
            )

        # Store results
        results["resolution_frc"] = frc_results["resolution"]
        results["cutoff_frequency"] = frc_results.get(
            "cutoff_frequency", np.nan
        )
        results["spatial_frequencies"] = frc_results["spatial_frequencies"]
        results["threshold"] = frc_results["threshold"]

        # Store multi-split specific results
        if n_splits > 1:
            results["resolution_std"] = frc_results["resolution_std"]
            results["frc_curve"] = frc_results["frc_curve_mean"]
            results["frc_curve_std"] = frc_results["frc_curve_std"]
            results["resolutions_per_split"] = frc_results[
                "resolutions_per_split"
            ]
        else:
            results["frc_curve"] = frc_results["frc_curve"]

        # Create plots
        # Plot 1: FRC curve
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine which FRC curve to plot
        if n_splits > 1:
            frc_curve = frc_results["frc_curve_mean"]
            frc_curve_std = frc_results["frc_curve_std"]

            # Plot with error band
            ax.plot(
                frc_results["spatial_frequencies"],
                frc_curve,
                "b-",
                linewidth=2,
                label=f"FRC (mean, n={n_splits})",
            )
            ax.fill_between(
                frc_results["spatial_frequencies"],
                frc_curve - frc_curve_std,
                frc_curve + frc_curve_std,
                alpha=0.3,
                color="b",
                label="±1 std",
            )

            resolution_label = f"Resolution: {frc_results['resolution']:.1f} ± {frc_results['resolution_std']:.1f} nm"
        else:
            frc_curve = frc_results["frc_curve"]

            ax.plot(
                frc_results["spatial_frequencies"],
                frc_curve,
                "b-",
                linewidth=2,
                label="FRC",
            )

            resolution_label = (
                f"Resolution: {frc_results['resolution']:.1f} nm"
            )

        # Plot threshold line
        ax.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.3f})",
        )

        # Plot resolution point
        if not np.isnan(frc_results["resolution"]):
            cutoff_freq = frc_results.get("cutoff_frequency")
            if cutoff_freq is None and n_splits > 1:
                # Calculate mean cutoff frequency
                cutoff_freqs = frc_results.get("cutoff_frequencies", [])
                cutoff_freq = np.nanmean(
                    [c for c in cutoff_freqs if not np.isnan(c)]
                )

            if cutoff_freq is not None and not np.isnan(cutoff_freq):
                ax.axvline(
                    x=cutoff_freq,
                    color="g",
                    linestyle="--",
                    linewidth=1.5,
                    label=resolution_label,
                )

        ax.set_xlabel("Spatial Frequency (1/nm)")
        ax.set_ylabel("FRC")
        ax.set_title("Fourier Ring Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

        plt.tight_layout()

        # Save FRC plot
        plot_path_frc = os.path.join(results["folder"], "resolution_frc.png")
        plt.savefig(plot_path_frc, dpi=300, bbox_inches="tight")
        plt.close()

        results["fig_frc"] = plot_path_frc

        # Plot 2: Split images comparison (only for single split)
        if n_splits == 1 and "image_1" in frc_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            image_1 = frc_results["image_1"]
            image_2 = frc_results["image_2"]
            bounds = frc_results["bounds"]

            extent = [bounds[0], bounds[1], bounds[2], bounds[3]]

            # Plot image 1
            im1 = ax1.imshow(
                image_1.T, extent=extent, origin="lower", cmap="hot"
            )
            ax1.set_xlabel("x (nm)")
            ax1.set_ylabel("y (nm)")
            ax1.set_title(f"Image 1 ({len(self.locs)//2} locs)")
            plt.colorbar(im1, ax=ax1, shrink=0.8)

            # Plot image 2
            im2 = ax2.imshow(
                image_2.T, extent=extent, origin="lower", cmap="hot"
            )
            ax2.set_xlabel("x (nm)")
            ax2.set_ylabel("y (nm)")
            ax2.set_title(f"Image 2 ({len(self.locs)//2} locs)")
            plt.colorbar(im2, ax=ax2, shrink=0.8)

            plt.tight_layout()

            # Save images plot
            plot_path_images = os.path.join(
                results["folder"], "resolution_frc_images.png"
            )
            plt.savefig(plot_path_images, dpi=300, bbox_inches="tight")
            plt.close()

            results["fig_images"] = plot_path_images

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def resolution_frc_spatial(self, i, parameters, results):
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
        from picasso_workflow.outpost_modules.resolution_frc import (
            compute_frc_spatial,
        )

        # Get parameters with defaults
        pixelsize_render = parameters.get("pixelsize_render", 5.0)
        smoothing_sigma = parameters.get("smoothing_sigma", None)
        threshold = parameters.get("threshold", 1.0 / 7.0)
        region_size = parameters.get("region_size", 10.0)
        min_locs_per_region = parameters.get("min_locs_per_region", 500)
        max_frc_range_nm = parameters.get("max_frc_range_nm", None)
        n_processes = parameters.get("n_processes", 4)
        smoothing_window = parameters.get("smoothing_window", 0.005)

        # Compute spatial FRC
        logger.debug("Using spatial FRC approach")
        frc_results = compute_frc_spatial(
            self.locs,
            self.pixelsize,
            pixelsize_render=pixelsize_render,
            smoothing_sigma=smoothing_sigma,
            threshold=threshold,
            region_size=region_size,
            min_locs_per_region=min_locs_per_region,
            max_frc_range_nm=max_frc_range_nm,
            n_processes=n_processes,
            smoothing_window=smoothing_window,
        )

        # Store results
        results["resolution_frc_spatial"] = frc_results["resolution"]
        results["resolution_unsmoothed"] = frc_results["resolution_unsmoothed"]
        results["resolution_std"] = frc_results["resolution_std"]
        results["n_regions"] = frc_results["n_regions"]
        results["n_regions_total"] = frc_results["n_regions_total"]
        results["frc_curve_mean"] = frc_results["frc_curve_mean"]
        results["frc_curve_smoothed"] = frc_results["frc_curve_smoothed"]
        results["frc_curve_std"] = frc_results["frc_curve_std"]
        results["spatial_frequencies"] = frc_results["spatial_frequencies"]
        results["threshold"] = frc_results["threshold"]
        results["resolutions_per_region"] = frc_results[
            "resolutions_per_region"
        ]

        # Create plot using external function
        from picasso_workflow.outpost_modules.resolution_frc import (
            create_frc_plot,
        )

        plot_path = create_frc_plot(frc_results, results["folder"], threshold)
        results["fig_frc"] = plot_path

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def resolution_decorr_spatial(self, i, parameters, results):
        """Calculate resolution using spatial image decorrelation approach

        This method implements the image decorrelation analysis from
        Descloux et al., Nature Methods 2019. It divides the FOV into spatial
        regions and computes decorrelation-based resolution for each region.

        Args:
            i : int
                the index of the module
            parameters: dict
                with optional keys:
                    pixelsize_render : float
                        pixel size for rendered images in nm (default: 5 nm)
                    smoothing_sigma : float or None
                        Gaussian smoothing sigma in pixels (default: None)
                    region_size : float
                        size of each spatial region in micrometers (default: 10.0 µm)
                    min_locs_per_region : int
                        minimum localizations per region to process (default: 500)
                    n_processes : int
                        number of parallel processes (default: 4)
                    r_min : float
                        minimum normalized frequency (default: 0.0)
                    r_max : float
                        maximum normalized frequency (default: 1.0)
                    n_r : int
                        number of radial sampling points (default: 50)
                    n_gauss : int
                        number of Gaussian filter strengths (default: 10)
                    apod_edge_width : int
                        edge apodization width in pixels (default: 20)

        Results:
            resolution_decorr_spatial : float
                mean decorrelation-based resolution in nm
            resolution_std : float
                standard deviation across regions
            n_regions : int
                number of valid regions processed
            decorr_curve_mean : ndarray
                mean decorrelation curve across regions
            decorr_curve_std : ndarray
                std of decorrelation curves
            r_values : ndarray
                normalized radius values
            fig_decorr : str
                path to decorrelation curve plot
        """
        from picasso_workflow.outpost_modules.resolution_decorrelation import (
            compute_decorr_spatial,
        )

        # Get parameters with defaults
        pixelsize_render = parameters.get("pixelsize_render", 5.0)
        smoothing_sigma = parameters.get("smoothing_sigma", None)
        region_size = parameters.get("region_size", 10.0)
        min_locs_per_region = parameters.get("min_locs_per_region", 500)
        n_processes = parameters.get("n_processes", 4)
        r_min = parameters.get("r_min", 0.0)
        r_max = parameters.get("r_max", 1.0)
        n_r = parameters.get("n_r", 50)
        n_gauss = parameters.get("n_gauss", 10)
        apod_edge_width = parameters.get("apod_edge_width", 20)

        # Compute spatial decorrelation
        logger.debug("Using spatial image decorrelation approach")
        decorr_results = compute_decorr_spatial(
            self.locs,
            self.pixelsize,
            pixelsize_render=pixelsize_render,
            smoothing_sigma=smoothing_sigma,
            region_size=region_size,
            min_locs_per_region=min_locs_per_region,
            n_processes=n_processes,
            r_min=r_min,
            r_max=r_max,
            n_r=n_r,
            n_gauss=n_gauss,
            apod_edge_width=apod_edge_width,
        )

        # Store results
        results["resolution_decorr_spatial"] = decorr_results["resolution"]
        results["resolution_std"] = decorr_results["resolution_std"]
        results["n_regions"] = decorr_results["n_regions"]
        results["n_regions_total"] = decorr_results["n_regions_total"]
        results["decorr_curve_mean"] = decorr_results["decorr_curve_mean"]
        results["decorr_curve_std"] = decorr_results["decorr_curve_std"]
        results["r_values"] = decorr_results["r_values"]
        results["resolutions_per_region"] = decorr_results[
            "resolutions_per_region"
        ]

        # Create decorrelation curve plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot mean decorrelation curve with error band
        decorr_curve_mean = decorr_results["decorr_curve_mean"]
        decorr_curve_std = decorr_results["decorr_curve_std"]
        r_values = decorr_results["r_values"]

        ax.plot(
            r_values,
            decorr_curve_mean,
            "b-",
            linewidth=2,
            label="Mean Decorrelation",
        )
        ax.fill_between(
            r_values,
            decorr_curve_mean - decorr_curve_std,
            decorr_curve_mean + decorr_curve_std,
            alpha=0.3,
            color="blue",
            label=f'±1 SD ({decorr_results["n_regions"]} regions)',
        )

        # Plot threshold line
        ax.axhline(
            y=0.5,
            color="r",
            linestyle="--",
            linewidth=2,
            label="Threshold (0.5)",
        )

        # Mark resolution
        resolution = decorr_results["resolution"]
        if not np.isnan(resolution):
            # Convert resolution to normalized frequency
            # resolution = 2 * pixelsize / kc_max
            # kc_max = 2 * pixelsize / resolution (in 1/nm)
            # normalized kc = kc_max * pixelsize * 2 (since r=1 corresponds to Nyquist = 0.5/pixel)
            r_cutoff = resolution / (4 * pixelsize_render)
            if np.any(r_cutoff < r_values):
                ax.axvline(
                    x=r_cutoff,
                    color="g",
                    linestyle=":",
                    linewidth=2,
                    label=f"Resolution: {resolution:.1f} nm",
                )

        ax.set_xlabel("Normalized Frequency", fontsize=12)
        ax.set_ylabel("Decorrelation", fontsize=12)
        n_regions_x = decorr_results["n_regions_x"]
        n_regions_y = decorr_results["n_regions_y"]
        ax.set_title(
            f"Image Decorrelation Analysis ({n_regions_x}×{n_regions_y} regions)",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            results["folder"], "resolution_decorr_spatial.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        results["fig_decorr"] = plot_path

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def smlm_clusterer(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        radius = parameters["radius"] / pixelsize
        min_locs = parameters["min_locs"]
        basic_fa = parameters.get("basic_fa", False)
        parameters["basic_fa"] = basic_fa
        radius_z = parameters.get("radius_z", None)
        # make sure pxsiz is ok here
        if radius_z is not None:
            radius_z = radius_z / pixelsize
        parameters["radius_z"] = radius_z
        pixelsize = self.pixelsize

        # cluster locs or channel_locs,
        # depending on whether this is a single target workflow
        if self.locs is not None:
            kwargs = {
                "locs": self.locs,
                "radius_xy": radius,
                "min_locs": min_locs,
                "frame_analysis": basic_fa,
                "pixelsize": pixelsize,
            }
            if radius_z is not None:  # 3D
                kwargs["radius_z"] = radius_z

            # label locs according to clusters
            # logger.debug(
            #     f"starting clusterer on self.locs with kwargs {kwargs}"
            # )
            self.locs = clusterer.cluster(**kwargs)
            smlm_cluster_info = {
                "Generated by": "Picasso SMLM clusterer",
                "Radius_xy": radius,
                "Radius_z": radius_z,
                "Min locs": min_locs,
                "Basic frame analysis": basic_fa,
                "Wrapped by": "picasso-workflow : smlm_clusterer",
            }
            self.info.append(smlm_cluster_info)
            filepath = os.path.join(
                results["folder"], "cluster_smlm_locs.hdf5"
            )
            self._save_locs(filepath)
            results["fp_clustered_locs"] = filepath

            self.locs = clusterer.find_cluster_centers(self.locs, pixelsize)
            logger.warning(
                "saving cluster centeras as locs. Is that intended?"
            )

            filepath = os.path.join(
                results["folder"], "cluster_smlm_centers.hdf5"
            )
            self._save_locs(filepath)
            results["fp_cluster_centers"] = filepath
        else:
            logger.debug("smlm clustering channel_locs")
            new_channel_locs = []
            new_channel_infos = []
            for tag, info, locs in zip(
                self.channel_tags, self.channel_info, self.channel_locs
            ):
                kwargs = {
                    "locs": locs,
                    "radius_xy": radius,
                    "min_locs": min_locs,
                    "frame_analysis": basic_fa,
                    "pixelsize": pixelsize,
                }
                if radius_z is not None:  # 3D
                    kwargs["radius_z"] = radius_z

                # label locs according to clusters
                clustered_locs = clusterer.cluster(**kwargs)
                smlm_cluster_info = {
                    "Generated by": "Picasso SMLM clusterer",
                    "Radius_xy": radius,
                    "Radius_z": radius_z,
                    "Min locs": min_locs,
                    "Basic frame analysis": basic_fa,
                    "Wrapped by": "picasso-workflow : smlm_clusterer",
                }
                info.append(smlm_cluster_info)
                filepath = os.path.join(
                    results["folder"], f"{tag}_cluster_smlm_locs.hdf5"
                )
                io.save_locs(filepath, clustered_locs, info)

                cc_locs = clusterer.find_cluster_centers(
                    clustered_locs, pixelsize
                )

                filepath = os.path.join(
                    results["folder"], f"{tag}_cluster_smlm_centers.hdf5"
                )
                io.save_locs(filepath, cc_locs, info)

                new_channel_locs.append(cc_locs)
                new_channel_infos.append(info)
            self.channel_locs = new_channel_locs
            self.channel_info = new_channel_infos

        return parameters, results

    @profile_resource_usage
    @module_decorator
    def gaussian_mixture_cluster(self, i, parameters, results):
        """Perform clustering using gaussian mixture models. After this module,
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
                    sigma_bounds : tuple of float (not recommended)
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
        pixelsize = self.pixelsize
        required_args = ["min_locs"]
        optional_args = [
            ("max_rounds_without_best_bic", g5m.MAX_ROUNDS_WITHOUT_BEST_BIC),
            ("bootstrap_check", False),
            ("calibration", None),
            ("pixelsize", pixelsize),
            ("asynch", True),
            ("callback_parent", None),  # "silent"),
            ("sigma_bounds", (g5m.MIN_SIGMA_FACTOR, g5m.MAX_SIGMA_FACTOR)),
            ("loc_prec_handle", "local"),
        ]
        try:
            kwargs = {k: parameters[k] for k in required_args}
        except KeyError as e:
            logger.error(
                f"""All of the following arguments are required for
                picasso.g5m.run_g5m: {required_args}"""
            )
            raise e
        # sigma values are given in nm in parameters but px in gmm
        if "min_sigma" in kwargs.keys():
            kwargs["min_sigma"] = kwargs["min_sigma"] / pixelsize
            kwargs["max_sigma"] = kwargs["max_sigma"] / pixelsize
        for oa, default in optional_args:
            setval = parameters.get(oa, default)
            if oa == "calibration" and setval == "":
                setval = default
            elif oa == "calibration" and isinstance(setval, str):
                fp_calib = setval
                with open(fp_calib, "r") as f:
                    z_calibration = yaml.full_load(f)
                setval = z_calibration
            elif oa == "callback_parent" and (
                setval == "silent" or setval == "None"
            ):
                setval = None
            kwargs[oa] = setval

        print("G5M arguments")
        print(kwargs)

        results["g5m_args"] = str(kwargs)

        center_locs, clustered_locs, gmm_info = g5m.run_g5m(
            self.locs, self.info, **kwargs
        )

        if parameters.get("save_locs"):
            fp_centers = os.path.join(results["folder"], "gmm_centers.hdf5")
            io.save_locs(fp_centers, center_locs, gmm_info)
            fp_centers = os.path.join(
                results["folder"], "gmm_clustered_locs.hdf5"
            )
            io.save_locs(fp_centers, clustered_locs, gmm_info)

        # plot: histogram of cluster sizes
        fig, ax = plt.subplots()
        maxbin = int(np.quantile(center_locs["n_events"], 0.95))
        ax.hist(center_locs["n_events"], bins=np.arange(maxbin))
        ax.set_xlabel("cluster size [locs]")
        ax.set_ylabel("Frequency")
        results["fp_fig_clustersizes"] = os.path.join(
            results["folder"], "fig_gmm_clustersize.png"
        )
        fig.savefig(results["fp_fig_clustersizes"])

        # test for subclustering
        results["fp_fig_subclustering"] = os.path.join(
            results["folder"], "subcluster_test.png"
        )
        g5m.test_subclustering(center_locs, results["fp_fig_subclustering"])

        results["n_locs_in"] = len(self.locs)
        results["n_locs_clustered"] = len(clustered_locs)
        results["n_centers"] = len(center_locs)

        self.locs = copy.copy(center_locs)
        self.info = gmm_info

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def nneighbor(self, i, parameters, results):
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
        if self.locs is not None:
            locs_list = [self.locs]
            tags = ["locs"]
            sgl_stage = True
        elif self.channel_locs is not None:
            locs_list = self.channel_locs
            tags = self.channel_tags
            sgl_stage = False
        else:
            raise KeyError("No locs loaded")

        density_results = []
        nn_fps = []
        fp_figs = []
        for i, (tag, locs) in enumerate(zip(tags, locs_list)):
            fig, ax = plt.subplots(nrows=2)
            # points = np.array(
            #     locs[parameters["dims"]].tolist()
            # )  # c-locs[0] only for now, before sgl/agg workflow refactoring!!
            points = locs[list(parameters["dims"])].to_numpy()
            # convert all dimensions to nanometers
            pixelsize = self.pixelsize
            for i, dim in enumerate(parameters["dims"]):
                if dim in ["x", "y"]:
                    points[:, i] = points[:, i] * pixelsize

            # logger.debug(points)
            # logger.debug(points.shape)
            if len(locs) < 10000:
                alldist = distance.cdist(points, points)
                logger.debug("found all distances")
                if parameters.get("add_column", False):
                    # print(alldist.shape)
                    if sgl_stage:
                        self.locs["NNdist"] = alldist[:, 1]
                        # print(locs.dtype)
                    else:
                        self.channel_locs[i]["NNdist"] = alldist[:, 1]
                        # print(locs.dtype)
                alldist = np.sort(alldist, axis=1)
                logger.debug("sorted all distances")
            else:
                k = max([parameters["nth_NN"] + 3, parameters["nth_rdf"] + 3])
                tree = KDTree(points)
                alldist, indices = tree.query(points, k=k)
                if parameters.get("add_column", False):
                    print(alldist.shape)
                    if sgl_stage:
                        self.locs["NNdist"] = alldist[:, 1]
                        print(self.locs.dtypes)
                    else:
                        self.channel_locs[i]["NNdist"] = alldist[:, 1]
                        print(self.channel_locs[i].dtypes)
                alldist = np.sort(alldist, axis=1)

            # calculate bins
            NN_median = np.median(alldist[:, 1])
            deltar = NN_median / parameters.get("subsample_1stNN", 20)
            rmax_NN = np.quantile(alldist[:, parameters["nth_NN"]], 0.95)
            rmax_rdf = np.quantile(alldist[:, parameters["nth_rdf"]], 0.95)

            logger.debug("calculated radial distribution function")
            # print(alldist)
            # print(alldist.shape)
            # out_path = os.path.join(results["folder"], "nneighbors_all.txt")
            # np.savetxt(out_path, np.sort(alldist, axis=1), newline="\r\n")
            # alldist[alldist == 0] = float("inf")
            # nneighbors = np.sort(alldist, axis=1)[:, : parameters["nth"]]
            nneighbors = alldist[:, 1 : parameters["nth_NN"] + 1]
            out_path = os.path.join(results["folder"], f"{tag}_nneighbors.txt")
            np.savetxt(out_path, nneighbors, newline="\r\n")
            nn_fps.append(out_path)

            # logger.debug("calculated bin parameters")
            # # as alldist can be large, reduce it here already, so memory
            # can be freed
            nspots = alldist.shape[0]
            # idx_ = np.min(alldist, axis=1) <= (rmax_rdf + deltar)
            # alldist = alldist[:, idx_]
            # logger.debug("cropped 2d alldist")
            # alldist = np.sort(alldist.flatten())
            # logger.debug("flattened alldist")
            # # distarray = alldist.flatten()
            # # logger.debug('flattened distarray')
            # alldist = alldist[alldist <= (rmax_rdf + deltar)]
            # logger.debug("prepared alldist")
            _, _, density = self._calc_radial_distribution_function(
                # alldist,
                points,
                deltar,
                rmax_rdf,
                nspots,
                d=len(parameters["dims"]),
                ax=ax[0],
            )
            density_results.append(density)

            # plot results
            colors = cm.get_cmap("viridis", nneighbors.shape[1]).colors
            bins = np.arange(0, rmax_NN, step=deltar)
            nnhist_obs = np.zeros((len(bins), nneighbors.shape[1]))
            for i in range(nnhist_obs.shape[1]):
                k = i + 1
                _ = ax[1].hist(
                    nneighbors[:, i],
                    bins=bins,
                    color=colors[i],
                    alpha=0.2,
                    label=f"k={k}",
                )
            ax[1].legend()
            ax[1].set_xlabel("Distance [nm]")
            ax[1].set_ylabel("Frequency")
            ax[1].set_title(f"{tag} Nearest Neighbor Histogram")
            rcode = generate_random_code(6)
            fp_fig = os.path.join(
                results["folder"], f"{tag}_nndist_{rcode}.png"
            )
            fp_figs.append(fp_fig)
            plt.tight_layout()
            fig.savefig(fp_fig)

        if len(tags) == 1:
            results["density_rdf"] = density_results[0]
            results["nneighbors"] = nn_fps[0]
            results["fp_fig"] = fp_figs[0]
        else:
            results["density_rdf"] = density_results
            results["nneighbors"] = nn_fps
            results["fp_fig"] = fp_figs

        return parameters, results

    def _calc_radial_distribution_function_legacy(
        self, alldist, deltar, rmax, nspots, d=2, ax=None
    ):
        rs = np.arange(
            0,
            rmax + deltar,
            deltar,
        )
        # n_means = np.zeros_like(rs)
        # d_areas = np.zeros_like(rs)

        # logger.debug(f"calculating {len(rs)} rdf points")

        # for i, r in enumerate(rs):
        #     # area = 2 * np.pi * r**2
        #     # n_mean = np.sum(alldist < r) / len(locs)
        #     # crdf[i] = n_mean / area
        #     d_areas[i] = 2 * np.pi * r * deltar
        #     # n_means[i] = np.sum(distarray <= r) / nspots
        #     n_means[i] = np.sum(alldist <= r) / nspots
        # d_n_means = n_means[1:] - n_means[:-1]

        d_areas = 2 * np.pi * rs * deltar
        d_n_means, _ = np.histogram(alldist, bins=rs)
        d_n_means = d_n_means / nspots
        rdf = d_n_means[1:] / d_areas[2:]
        rs = rs[2:]
        # rdf = crdf[1:] - crdf[:-1]

        # assuming the RDF converged to the bulk density in
        # its second half
        density = np.median(rdf[int(len(rs) / 2) :])

        # plot results
        ax.plot(rs, rdf * 1e3**d)
        ax.set_xlabel("Radius [nm]")
        ax.set_ylabel(f"density [µm^{-d}]")
        ax.set_title("Radial Distribution Function")
        return rs, rdf, density

    def _calc_radial_distribution_function(  # _KD(
        self, locs, deltar, rmax, nspots, d=2, ax=None
    ):
        rs = np.arange(
            0,
            rmax + deltar,
            deltar,
        )

        tree = KDTree(locs)
        n_means = tree.count_neighbors(tree, rs) / nspots - 1
        d_n_means = n_means[1:] - n_means[:-1]

        d_areas = 2 * np.pi * rs * deltar
        rdf = d_n_means / d_areas[1:]
        rs = rs[1:]
        # rdf = crdf[1:] - crdf[:-1]

        # assuming the RDF converged to the bulk density in
        # its second half
        density = np.median(rdf[int(len(rs) / 2) :])

        # plot results
        ax.plot(rs, rdf * 1e3**d)
        ax.set_xlabel("Radius [nm]")
        ax.set_ylabel(f"density [µm^{-d}]")
        ax.set_title("Radial Distribution Function")
        return rs, rdf, density

    #    @profile_resource_usage
    @module_decorator
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
        if isinstance(parameters["nneighbors"], str):
            nneighbor_list = [np.loadtxt(parameters["nneighbors"])]
            tags = ["locs"]
            sgl_stage = True
        elif isinstance(parameters["nneighbors"], list):
            if isinstance(parameters["nneighbors"][0], str):
                nneighbor_list = [
                    np.loadtxt(fp) for fp in parameters["nneighbors"]
                ]
            else:
                nneighbor_list = [parameters["nneighbors"]]
            tags = self.channel_tags
            sgl_stage = False
        else:
            nneighbor_list = [parameters["nneighbors"]]
            tags = ["locs"]
            sgl_stage = True
        nneighbors = nneighbor_list[0]
        # print(nneighbors.shape)
        # return
        kmin = int(parameters.get("kmin", 1))
        k_max = nneighbors.shape[1]
        # nspots = nneighbors.shape[0]
        d = int(parameters.get("dimensionality", 2))
        kwargs = {
            # "nn_dists": nneighbors.T[kmin - 1 :, :],
            "kmin": kmin,
            "rho_bound_factor": 10,
        }
        if min_dist := parameters.get("min_dist"):
            kwargs["min_dist"] = float(min_dist)
        if max_dist := parameters.get("max_dist"):
            kwargs["max_dist"] = float(max_dist)
        if bkg_fraction := parameters.get("bkg_fraction"):
            kwargs["bkg_fraction"] = float(bkg_fraction)
        kwargs["fit_bkg"] = bool(parameters.get("fit_bkg", False))
        kwargs["d"] = d
        rho_init = 2 / (2 * d * np.pi * np.median(nneighbors[:, 0]) ** d)
        kwargs["rho_init"] = rho_init

        densities = []
        bkgs = []
        wasserstein_distances = []
        mean_wasserstein_distances = []
        ks_pvalues = []
        results["fp_fig"] = []
        for tag, nneighbors in zip(tags, nneighbor_list):
            kwargs["nn_dists"] = nneighbors.T[kmin - 1 :, :]
            (
                rho_mle,
                fitresult,
            ) = picasso_outpost.estimate_density_from_neighbordists(**kwargs)
            # print(fitresult)
            logger.debug(str(fitresult))
            densities.append(rho_mle)
            if len(fitresult.x) > 1:
                bkgs.append(fitresult.x[1])
            else:
                bkgs.append(float(parameters.get("bkg_fraction", 0)))

            # Calculate goodness-of-fit using Wasserstein distance and KS tests
            k_wasserstein_distances = []
            k_ks_pvalues = []
            for k_idx in range(nneighbors.shape[1]):
                k = k_idx + 1
                if k >= kmin:  # Only calculate for fitted k values
                    observed_distances = nneighbors[:, k_idx]
                    # Filter distances within bounds
                    observed_filtered = observed_distances[
                        (observed_distances >= kwargs.get("min_dist", 0))
                        & (
                            observed_distances
                            <= float(kwargs.get("max_dist", np.inf))
                        )
                    ]

                    if len(observed_filtered) > 0:
                        # Generate theoretical CSR samples with same size
                        n_samples = len(observed_filtered)
                        max_dist_theory = (
                            np.max(observed_filtered) * 1.2
                        )  # Extend range slightly
                        r_theory = np.linspace(
                            float(kwargs.get("min_dist", 0)),
                            max_dist_theory,
                            n_samples,
                        )

                        # Get theoretical CSR probability distribution
                        csr_pdf = picasso_outpost.nndistribution_from_csr(
                            r_theory,
                            k,
                            rho_mle,
                            d=d,
                            min_dist=float(kwargs.get("min_dist", 0)),
                            max_dist=float(kwargs.get("max_dist", np.inf)),
                            bkg_fraction=float(bkgs[-1]),
                            renormalize=True,
                        )

                        # Convert PDF to samples by inverse transform sampling
                        # Set seed for reproducibility
                        np.random.seed(42 + k)
                        cdf = np.cumsum(csr_pdf)
                        if cdf[-1] > 0:
                            cdf = cdf / cdf[-1]  # Normalize to [0,1]
                        else:
                            # Handle edge case where all probabilities are zero
                            cdf = np.linspace(0, 1, len(cdf))
                        uniform_samples = np.random.uniform(0, 1, n_samples)
                        theoretical_samples = np.interp(
                            uniform_samples, cdf, r_theory
                        )

                        # Calculate Wasserstein distance
                        w_dist = wasserstein_distance(
                            observed_filtered, theoretical_samples
                        )
                        k_wasserstein_distances.append(w_dist)

                        # Calculate Kolmogorov-Smirnov test
                        # Use CDF function from picasso_outpost
                        def csr_cdf(x):
                            return picasso_outpost.csr_cdf_for_ks_test(
                                x,
                                k,
                                rho_mle,
                                d=d,
                                min_dist=kwargs.get("min_dist", 0),
                                max_dist=kwargs.get("max_dist", np.inf),
                                bkg_fraction=bkgs[-1],
                            )

                        # Perform KS test
                        ks_stat, ks_pvalue = kstest(observed_filtered, csr_cdf)
                        k_ks_pvalues.append(ks_pvalue)

                        logger.debug(
                            f"k={k}, Wasserstein distance: {w_dist:.3f}, "
                            f"KS p-value: {ks_pvalue:.3f}"
                        )

            wasserstein_distances.append(k_wasserstein_distances)
            ks_pvalues.append(k_ks_pvalues)
            if k_wasserstein_distances:
                mean_wasserstein_distances.append(
                    np.mean(k_wasserstein_distances)
                )
            else:
                mean_wasserstein_distances.append(np.nan)

            # plot results
            fig, ax = plt.subplots()
            colors = cm.get_cmap("viridis", k_max).colors
            bin_max = np.quantile(nneighbors[:, -1], 0.95)
            median_1stNN = np.median(nneighbors[:, 0])
            # sample bins such that there are 5 bins from 0 to middle of 1stNN
            nbins = int(5 * bin_max / median_1stNN)
            bins = np.linspace(0, bin_max, num=nbins)
            rvals = np.linspace(0, bin_max, num=3 * nbins)
            nnhist_obs = np.zeros((len(bins), k_max))
            nnhist_an = np.zeros_like(nnhist_obs)
            for i in range(nnhist_an.shape[1]):
                k = i + 1
                # nnhist_obs, edges = np.histogram(nneighbors[:, i], bins=bins)
                nnhist_an = picasso_outpost.nndistribution_from_csr(
                    rvals,
                    k,
                    rho_mle,
                    d=d,
                    min_dist=kwargs.get("min_dist", 0),
                    max_dist=kwargs.get("max_dist", np.inf),
                    renormalize=False,
                )
                if i == 0:
                    lbl = f"rho_init {1E6*rho_init:.1f} um^-2"
                else:
                    lbl = f"observed k={k}"
                _ = ax.hist(
                    nneighbors[:, i],
                    bins=bins,
                    density=True,
                    color=colors[i],
                    alpha=0.2,
                    label=lbl,
                )
                if k < kmin:
                    linestyle = ":"
                    lblf = f"k={k} not fitted"
                elif k == kmin:
                    linestyle = "--"
                    lblf = f"fit {1E6*rho_mle:.1f} µm^-2"
                else:
                    linestyle = "--"
                    lblf = f"fit k={k}"
                # do not plot the model cutoff
                if kwargs.get("min_dist", 0) > 0:
                    nnhist_an[rvals <= kwargs["min_dist"]] = np.nan
                ax.plot(
                    rvals,  # + (bins[1] - bins[0]) / 2,
                    nnhist_an,
                    color=colors[i],
                    linestyle=linestyle,
                    label=lblf,
                )
                # now, plot the histogram below cutoff in white for shading
                if (kwargs.get("min_dist", 0) > 0) and (i == 0):
                    xlim = ax.get_xlim()
                    x_fill = [xlim[0], kwargs["min_dist"]]
                    y_fill1 = [ax.get_ylim()[0]] * 2
                    y_fill2 = [ax.get_ylim()[1]] * 2
                    ax.fill_between(
                        x_fill, y_fill1, y_fill2, color="grey", alpha=0.2
                    )
                    ax.set_xlim(xlim)
            ax.legend(loc="upper right")
            ax.set_xlabel("Distance [nm]")
            ax.set_ylabel("probability density")
            ax.set_title(f"Nearest Neighbor Distribution {tag}")
            fp_fig = os.path.join(results["folder"], f"{tag}_nndist_fit.png")
            fig.savefig(fp_fig)
            results["fp_fig"].append(fp_fig)
        results["density"] = densities
        results["bkg_fraction"] = bkgs
        results["wasserstein_distances_per_k"] = wasserstein_distances
        results["mean_wasserstein_distance"] = mean_wasserstein_distances
        results["ks_pvalues_per_k"] = ks_pvalues

        if sgl_stage:
            results["density"] = results["density"][0]
            results["fp_fig"] = results["fp_fig"][0]
            results["wasserstein_distances_per_k"] = results[
                "wasserstein_distances_per_k"
            ][0]
            results["mean_wasserstein_distance"] = results[
                "mean_wasserstein_distance"
            ][0]
            results["ks_pvalues_per_k"] = results["ks_pvalues_per_k"][0]

        return parameters, results

    # @module_decorator
    # def radial_distribution_function(self, i, parameters, results):
    #     """Generate the Radial Distribution Function,
    #     Whis is the sum of nearest neighbors with geometry factor.
    #     At long radii, its value is the overall density.

    #     Every spot is picked, pick radii are altered and the density
    #     calculated. The RDF is the difference between those densities.
    #     Args:
    #         i : int
    #             the index of the module
    #         parameters: dict
    #             with required keys:
    #                 dims : list of str
    #                     the distance dimensions, e.g. ['x', 'y']
    #                     or ['x', 'y', 'z']
    #                 rmax : float
    #                     the maximum r to evaluate
    #                 deltar : float
    #                     the step size in r
    #             and optional keys:
    #                 save_locs : bool
    #                     whether to save the locs into the results folder
    #         results : dict
    #             the results this function generates. This is created
    #             in the decorator wrapper
    #     """
    #     rs = np.arange(
    #         0,
    #         parameters["rmax"] + 2 * parameters["deltar"],
    #         parameters["deltar"])
    #     n_means = np.zeros_like(rs)
    #     d_areas = np.zeros_like(rs)
    #     locs = self.channel_locs[
    #         0
    #     ]  # c-locs[0] only for now, before sgl/agg workflow refactoring!!

    #     points = np.array(locs[parameters["dims"]].tolist())
    #     # convert all dimensions to nanometers
    #     pixelsize = self.analysis_config["camera_info"]["pixelsize"]
    #     for i, dim in enumerate(parameters["dims"]):
    #         if dim in ["x", "y"]:
    #             points[:, i] = points[:, i] * pixelsize

    #     alldist = distance.cdist(points, points)
    #     # alldist[alldist == 0] = float("inf")
    #     alldist = np.sort(alldist, axis=1)

    #     for i, r in enumerate(rs):
    #         # area = 2 * np.pi * r**2
    #         # n_mean = np.sum(alldist < r) / len(locs)
    #         # crdf[i] = n_mean / area
    #         d_areas[i] = 2 * np.pi * r * parameters["deltar"]
    #         n_means[i] = np.sum(alldist < r) / len(locs)
    #     d_n_means = n_means[1:] - n_means[:-1]
    #     rdf = d_n_means / d_areas[1:]
    #     # rdf = crdf[1:] - crdf[:-1]

    #     results["density"] = np.median(rdf[int(len(rs) / 2):])

    #     # plot results
    #     fig, ax = plt.subplots()
    #     ax.plot(rs[1:], rdf)
    #     ax.set_xlabel("Radius [nm]")
    #     ax.set_ylabel("probability density")
    #     ax.set_title("Radial Distribution Function")
    #     results["fp_fig"] = os.path.join(results["folder"], "rdf.png")
    #     fig.savefig(results["fp_fig"])

    #     return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def save_single_dataset(self, i, parameters, results):
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
        results["filepath"] = os.path.join(
            results["folder"], parameters["filename"]
        )
        root, ext = os.path.splitext(results["filepath"])
        if ext != ".hdf5":
            results["filepath"] = root + ".hdf5"

        results["nlocs"] = len(self.locs)
        res = self._save_locs(results["filepath"])
        for k, v in res.items():
            results[k] = v
        return parameters, results

    def _save_locs(self, filename):
        t00 = time.time()

        io.save_locs(filename, self.locs, self.info)
        # # when the paths get long, the hdf5 library throws an error, so chdir
        # # but apparently, the issue is the length of the filename itself
        # previous_dir = os.getcwd()
        # parent_dir, fn = os.path.split(filename)
        # os.chdir(parent_dir)
        # io.save_locs(fn, self.locs, self.info)
        # os.chdir(previous_dir)

        dt = np.round(time.time() - t00, 2)
        results_save = {"duration": dt}
        return results_save

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    #    @profile_resource_usage
    @module_decorator
    def load_datasets_to_aggregate(self, i, parameters, results):
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
        self.channel_locs = []
        self.channel_info = []
        self.channel_tags = []
        for i, (fp, tag) in enumerate(
            zip(parameters["filepaths"], parameters["tags"])
        ):
            locs, info = io.load_locs(fp)
            # locs = lib.append_to_rec(
            #     locs,
            #     np.full(len(locs), tag, dtype='U10'),
            #     "channel",
            # )
            locs["channel"] = i * np.ones(len(locs), dtype=np.int8)
            self.channel_locs.append(locs)
            self.channel_info.append(info)
            self.channel_tags.append(tag)
        results["filepaths"] = parameters["filepaths"]
        results["tags"] = parameters["tags"]
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def align_channels(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        rcode = generate_random_code(6)
        if parameters.get("filepaths"):
            self.channel_locs = []
            self.channel_info = []
            self.channel_tags = []
            for fp in parameters["filepaths"]:
                locs, info = io.load_locs(fp)
                self.channel_locs.append(locs)
                self.channel_info.append(info)
                self.channel_tags.append(os.path.split(fp)[1])
        if parameters.get("fp_fiducials"):
            fiducial_locs = []
            fiducial_info = []
            for fp in parameters["fp_fiducials"]:
                locs, info = io.load_locs(fp)
                fiducial_locs.append(locs)
                fiducial_info.append(info)
        else:
            fiducial_locs = None

        results["fp_scene_locs_before"] = os.path.join(
            results["folder"], f"locs_before_{rcode}.png"
        )
        render.plot_scene(
            self.channel_locs,
            pixelsize,
            pixelsize,
            fp=results["fp_scene_locs_before"],
        )
        if fiducial_locs is not None:
            results["fp_scene_fids_before"] = os.path.join(
                results["folder"], f"fiducials_before_{rcode}.png"
            )
            fid_render_kwargs = {
                "blur_method": "gaussian",
                "min_blur_width": 2,
            }
            render.plot_scene(
                fiducial_locs,
                pixelsize,
                pixelsize,
                fp=results["fp_scene_fids_before"],
                render_kwargs=fid_render_kwargs,
            )

        align_pars = parameters.get("align_pars", {})
        align_pars["plot_dir"] = results["folder"]
        (
            shifts,
            cum_shifts,
            used_fiducials,
            algo_used,
            fp_figs,
            shift_uncertainties,
        ) = picasso_outpost.align_channels(
            self.channel_locs,
            self.channel_info,
            self.channel_tags,
            fiducial_locs=fiducial_locs,
            **align_pars,
        )
        results["shifts"] = cum_shifts[:, :, -1]
        results["alignment_algorithm"] = algo_used
        results["used_fiducials"] = used_fiducials
        results["fp_figs"] = fp_figs
        results["shift_uncertainties"] = shift_uncertainties

        fp_shifts = os.path.join(results["folder"], "shifts.txt")
        np.savetxt(fp_shifts, results["shifts"])
        fp_cumshifts = os.path.join(results["folder"], "cum_shifts.npy")
        np.save(fp_cumshifts, cum_shifts)

        # shift other localizations as well
        if fp_co_shift_list := parameters.get("fp_co_shift_channel_locs"):
            fp_co_shift_locs_out = []
            new_info = {
                "Generated by": "picasso-workflow : align_channels-coshift",
                "cumulative shifts": str(results["shifts"]),
                # "parameters": parameters,
            }
            for i, (tag, fp) in enumerate(
                zip(self.channel_tags, fp_co_shift_list)
            ):
                co_shift_locs, co_shift_info = io.load_locs(fp)
                co_shift_locs["x"] -= cum_shifts[0, i, -1]
                co_shift_locs["y"] -= cum_shifts[1, i, -1]
                if len(shifts) == 3:
                    co_shift_locs["z"] -= cum_shifts[2, i, -1]
                _, fn_coshift_locs = os.path.split(fp)
                fp_co_shift_locs_out.append(
                    os.path.join(results["folder"], f"{tag}_{fn_coshift_locs}")
                )
                co_shift_info.append(new_info)
                io.save_locs(
                    fp_co_shift_locs_out[-1], co_shift_locs, co_shift_info
                )
            results["fp_co_shift_locs_out"] = fp_co_shift_locs_out

        if fn := parameters.get("fig_filename"):
            fig_filepath = os.path.join(results["folder"], fn)
            picasso_outpost.plot_shift(shifts, cum_shifts, fig_filepath)
            results["fig_filepath"] = fig_filepath

            # Add confidence interval plotting for RSSO method
            if algo_used == "RSSO" and shift_uncertainties:
                fn_confidence = fn.replace(".png", "_with_confidence.png")
                fig_confidence_filepath = os.path.join(
                    results["folder"], fn_confidence
                )
                self._plot_channel_alignment_with_confidence(
                    results["shifts"],
                    shift_uncertainties,
                    fig_confidence_filepath,
                )
                results["fig_confidence_filepath"] = fig_confidence_filepath

        if parameters.get("crop_boundaries"):
            max_xmin, min_xmax = -np.inf, np.inf
            max_ymin, min_ymax = -np.inf, np.inf
            for locs in self.channel_locs:
                max_xmin = max(max_xmin, locs["x"].min())
                min_xmax = min(min_xmax, locs["x"].max())
                max_ymin = max(max_ymin, locs["y"].min())
                min_ymax = min(min_ymax, locs["y"].max())
            for i, locs in enumerate(self.channel_locs):
                self.channel_locs[i] = locs[
                    (locs["x"] > max_xmin) & (locs["x"] < min_xmax)
                ]
                self.channel_locs[i] = locs[
                    (locs["y"] > max_ymin) & (locs["y"] < min_ymax)
                ]

        results["fp_scene_locs_after"] = os.path.join(
            results["folder"], f"locs_after_{rcode}.png"
        )
        render.plot_scene(
            self.channel_locs, 100, 130, fp=results["fp_scene_locs_after"]
        )

        if fiducial_locs is not None:
            results["fp_scene_fids_after"] = os.path.join(
                results["folder"], f"fiducials_after_{rcode}.png"
            )
            render.plot_scene(
                fiducial_locs,
                130,
                130,
                fp=results["fp_scene_fids_after"],
                render_kwargs=fid_render_kwargs,
            )
            # save the potentially changed fiducials
            fp_fiducials = []
            for tag, flocs, finfo in zip(
                self.channel_tags, fiducial_locs, fiducial_info
            ):
                fp_fiducials.append(
                    os.path.join(results["folder"], f"{tag}_fiducials.hdf5")
                )
                io.save_locs(fp_fiducials[-1], flocs, finfo)
            results["fp_fiducials"] = fp_fiducials

        # add info
        new_info = {
            "Generated by": "picasso-workflow : align_channels",
            "shifts": str(results["shifts"]),
            # "parameters": parameters,
        }
        for i in range(len(self.channel_info)):
            self.channel_info[i].append(new_info)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def combine_channels(self, i, parameters, results):
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
        combine_map = {tag: i for i, tag in enumerate(self.channel_tags)}
        results["combine_map"] = combine_map
        fp_combinemap = os.path.join(results["folder"], "combine_map.yaml")
        with open(fp_combinemap, "w") as f:
            yaml.dump(combine_map, f)
        results["fp_combinemap"] = fp_combinemap
        combine_col = parameters.get("combine_col", "combine_id")
        for i in range(len(self.channel_locs)):
            locs = self.channel_locs[i]
            # Add combine_id column using DataFrame column assignment
            locs[combine_col] = i * np.ones(len(locs))
            self.channel_locs[i] = locs
        # Concatenate all DataFrames
        combined_locs = pd.concat(self.channel_locs, ignore_index=True)
        # sort like all Picasso localization lists
        combined_locs.sort_values("frame", kind="mergesort", inplace=True)

        # replace the channel_locs with the one combined dataset
        self.channel_locs = [combined_locs]
        new_info = {
            "Generated by": "picasso-workflow : combine_channels",
            "files combined": self.channel_tags,
        }
        info = self.channel_info[0] + [new_info]
        self.channel_info = [info]
        tag = parameters.get("tag", "combined-channels")
        self.channel_tags = [tag]

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def save_datasets_aggregated(self, i, parameters, results):
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
        allfps = self._save_datasets_agg(results["folder"])
        results["filepaths"] = allfps

        return parameters, results

    def _save_datasets_agg(self, folder):
        """Save aggregated channel datasets to individual HDF5 files.

        Internal helper method that iterates through all channels and saves
        their localization data and metadata to separate files.

        Args:
            folder : str
                Target folder path where files will be saved

        Returns:
            allfps : list of str
                List of all saved file paths
        """
        allfps = []
        for locs, info, tag in zip(
            self.channel_locs, self.channel_info, self.channel_tags
        ):
            filepath = os.path.join(folder, tag + ".hdf5")
            io.save_locs(filepath, locs, info)
            allfps.append(filepath)
        return allfps

    #    @profile_resource_usage
    @module_decorator
    def spinna(self, i, parameters, results):
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
        if isinstance(parameters["structures"], str):
            structures = io.load_info(parameters["structures"])
        else:
            structures = parameters["structures"]

        if isinstance(parameters["labeling_uncertainty"], (int, float)):
            labeling_uncertainty = {
                tgt: parameters["labeling_uncertainty"]
                for tgt in self.channel_tags
            }
        else:
            labeling_uncertainty = parameters["labeling_uncertainty"]

        if parameters.get("fp_mask_dict"):
            with open(parameters["fp_mask_dict"], "rb") as f:
                mask_dict = pickle.load(f)
        else:
            mask_dict = None

        if parameters.get("density") is not None:
            density = parameters["density"]
        else:
            density = parameters["density_app"]
            for i, (tgt, le) in enumerate(parameters["labeling_efficiency"]):
                density[i] = density[i] / le

        # locs, but as np.ndarray
        pixelsize = self.pixelsize
        exp_data = {}
        exp_n_targets = np.zeros(len(self.channel_tags))
        for i, target in enumerate(self.channel_tags):
            locs = self.channel_locs[i]
            exp_n_targets[i] = len(locs)
            if "z" in locs.columns:
                exp_data[target] = np.stack(
                    (locs["x"] * pixelsize, locs["y"] * pixelsize, locs["z"])
                ).T
                # dim = 3
            else:
                exp_data[target] = np.stack(
                    (locs["x"] * pixelsize, locs["y"] * pixelsize)
                ).T

        data_2d = "z" not in self.channel_locs[0].columns
        if not data_2d:
            if self.locs is not None:
                z_range = int(self.locs["z"].max() - self.locs["z"].min())
            else:
                z_maxs = [int(locs["z"].max()) for locs in self.channel_locs]
                z_mins = [int(locs["z"].min()) for locs in self.channel_locs]
                z_range = max(z_maxs) - min(z_mins)
            if z_range <= 0:
                data_2d = True
        if data_2d:
            area = parameters["n_simulate"] / sum(density)
            width = np.sqrt(area)
            height = np.sqrt(area)
            depth = None
            # d = 2
        else:
            if self.locs is not None:
                z_range = int(self.locs["z"].max() - self.locs["z"].min())
            else:
                z_maxs = [int(locs["z"].max()) for locs in self.channel_locs]
                z_mins = [int(locs["z"].min()) for locs in self.channel_locs]
                z_range = max(z_maxs) - min(z_mins)
            volume = parameters["n_simulate"] / sum(density)
            width = np.sqrt(volume / z_range)
            height = np.sqrt(volume / z_range)
            depth = z_range
            # d = 3

        structures, targets = picasso_outpost.load_structures_from_dict(
            structures
        )

        exp_frac_targets = exp_n_targets / np.sum(exp_n_targets)
        n_sim_targets = {
            tgt: exp_frac_targets[i] * parameters["n_simulate"]
            for i, tgt in enumerate(self.channel_tags)
        }
        # N_structures = spinna.generate_N_structures(
        N_structures = picasso_outpost.generate_N_structures(
            structures, n_sim_targets, parameters["granularity"]
        )

        spinna_pars = {
            "structures": structures,
            "label_unc": labeling_uncertainty,
            "le": parameters["labeling_efficiency"],
            "mask_dict": mask_dict,
            "width": width,
            "height": height,
            "depth": depth,
            "random_rot_mode": parameters["random_rot_mode"],
            "exp_data": exp_data,
            "sim_repeats": parameters["sim_repeats"],
            "NND_bin": parameters["fit_NND_bin"],
            "NND_maxdist": parameters["fit_NND_maxdist"],
            "N_structures": N_structures,
            "save_filename": os.path.join(results["folder"], "spinna-run"),
            "asynch": True,
            "targets": self.channel_tags,
            "apply_mask": mask_dict is not None,
            "nn_plotted": parameters["n_nearest_neighbors"],
            "result_dir": results["folder"],
            "n_simulated": n_sim_targets,
        }

        spinna_results, fp_figs = picasso_outpost.single_spinna_run(
            **spinna_pars
        )
        plt.close("all")
        results["fp_figs"] = fp_figs
        results["spinna_results"] = spinna_results

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def spinna_manual(self, i, parameters, results):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Args:
            i : int
                the index of the module
            parameters: dict
                with required keys:
                    proposed_labeling_efficiency : float, range 0-100
                        labeling efficiency percentage, default for all targets
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_labeling_uncertainty : float
                        labeling uncertainty [nm]; good value is e.g. 5
                        used proposed value in spinna_config.csv and can be
                         alteredmanually after the first run of this module
                    proposed_n_simulate : int
                        number of target molecules to simulated;
                        good value is e.g. 50000
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_density : float
                        density to simulate;
                        area density if 2D; volume density if 3D
                        used proposed value in spinna_config.csv and can be
                        altered manually after the first run of this module
                    proposed_nn_plotted : int
                        number of nearest neighbors to plot
                        used proposed value in spinna_config.csv and can be
                         alteredmanually after the first run of this module
                and optional keys:
                    structures : list of dict
                        SPINNA structures. Each structure dict has
                            "Molecular targets": list of str,
                            "Structure title": str,
                            "TARGET_x": list of float,
                            "TARGET_y": list of float,
                            "TARGET_z": list of float,
                        where TARGET is one each of the target names in
                        "Molecular targets"
                    structures_d : float
                        distance between molecules within auto-generated
                        structures, in nm. Only necessary if 'structures'
                        is not given.
            results : dict
                the results this function generates. This is created
                in the decorator wrapper
        """
        cfg_fp = os.path.join(results["folder"], "spinna_config.csv")
        if os.path.exists(cfg_fp):
            prepped = True
        else:
            prepped = False

        if not prepped:
            spinna_config = {}
            data_2d = "z" not in self.channel_locs[0].columns
            if data_2d:
                spinna_config["rotation_mode"] = ["2D"]
                area = (
                    parameters["proposed_n_simulate"]
                    / parameters["proposed_density"]
                )
                spinna_config["area"] = [area]
                d = 2
            else:
                spinna_config["rotation_mode"] = ["3D"]
                z_range = int(self.locs["z"].max() - self.locs["z"].min())
                volume = (
                    parameters["proposed_n_simulate"]
                    / parameters["proposed_density"]
                )
                spinna_config["volume"] = [volume]
                spinna_config["z_range"] = [z_range]
                d = 3

            # prepare input files for the user to edit, with default values
            spinna_structs = parameters.get("structures")
            if spinna_structs is None:
                spinna_structs = self._create_spinna_structure(
                    self.channel_tags,
                    [[1, 2]] * len(self.channel_tags),
                    distance=parameters["structures_d"],
                    dimensionality=d,
                )
            structs_fn = "spinna_structs.yaml"
            structs_fp = os.path.join(results["folder"], structs_fn)
            with open(structs_fp, "w") as f:
                yaml.dump_all(spinna_structs, f)

            spinna_config["structures_filename"] = [structs_fp]
            for locs, info, tag in zip(
                self.channel_locs, self.channel_info, self.channel_tags
            ):
                locs_fn = tag + ".hdf5"
                locs_fp = os.path.join(results["folder"], locs_fn)
                io.save_locs(locs_fp, locs, info)

                spinna_config[f"exp_data_{tag}"] = [locs_fp]
                spinna_config[f"le_{tag}"] = [
                    parameters["proposed_labeling_efficiency"]
                ]
                spinna_config[f"label_unc_{tag}"] = [
                    parameters["proposed_labeling_uncertainty"]
                ]
                spinna_config[f"n_simulated_{tag}"] = [
                    parameters["proposed_n_simulate"]
                ]
            spinna_config["granularity"] = [100]
            spinna_config["save_filename"] = ["spinna_results"]
            spinna_config["nn_plotted"] = [parameters["proposed_nn_plotted"]]

            # bin size: more than Nyquist subsampling
            expected_1stNN_peak = (
                2 / (2 * d * np.pi * parameters["proposed_density"])
            ) ** (1 / d)
            spinna_config["NND_bin"] = [expected_1stNN_peak / 10]
            spinna_config["density"] = parameters["proposed_density"]
            # max dist: a few times the first NN distance peak
            spinna_config["NND_maxdist"] = [20 * expected_1stNN_peak]
            spinna_config["sim_repeats"] = [2]

            # save config to file
            pd.DataFrame.from_dict(spinna_config).to_csv(cfg_fp)

            msg = "This is a manual step. Please provide input, "
            msg += "and re-execute the workflow. "
            msg += f" The file {cfg_fp} has been prepared for you"
            msg += ", based on the parameters given."
            results["message"] = msg
            logger.debug(msg)
            print(msg)
            results["success"] = False
        else:
            # kick off SPINNA analysis
            print("starting spinna batch analysis")
            result_dir, fp_summary, fp_fig = picasso_outpost.spinna_batch(
                cfg_fp
            )

            results["message"] = "Successfully performed SPINNA analysis."
            results["result_dir"] = result_dir
            results["fp_summary"] = fp_summary
            results["fp_fig"] = fp_fig
            results["success"] = True

        return parameters, results

    def _create_spinna_structure(
        self, names, multimers, distance, dimensionality=2
    ):
        """
        Args:
            names : list of str
                the names of proteins
            multimers : list of list of int
                for each name, the homo-multimers to implement
            distance : float
                distance between entities, in nm
        """
        spinna_structs = []
        for tag, name_multimers in zip(names, multimers):
            for n in name_multimers:
                # create positions on a cubic lattice
                positions = np.zeros((3, n))
                ux = np.array([1, 0, 0])
                uy = np.array([0, 1, 0])
                uz = np.array([0, 0, 1])
                edgelength = int(np.ceil(n ** (1 / dimensionality)))
                for i in range(n):
                    iz = i // (edgelength**2)
                    iy = i % (edgelength**2)
                    ix = i % edgelength
                    positions[:, i] = ix * ux + iy * uy + iz * uz
                positions[0, :] -= np.mean(positions[0, :])
                positions[1, :] -= np.mean(positions[1, :])
                positions[2, :] -= np.mean(positions[2, :])

                # create structure
                struct = {
                    "Molecular targets": [tag],
                    "Structure title": f"{tag}-{n}-mer",
                    f"{tag}_x": [float(x) for x in positions[0, :]],
                    f"{tag}_y": [float(x) for x in positions[1, :]],
                    f"{tag}_z": [float(x) for x in positions[2, :]],
                }
                spinna_structs.append(struct)
        return spinna_structs

    #    @profile_resource_usage
    @module_decorator
    def ripleysk(self, i, parameters, results):
        """Perforn Ripley's K analysis between the channels using
        Magdalena's code.
        Args:
            parameters:
                ripleys_n_random_controls : int
                    number of random controls, default: 100
                ripleys_rmax : int
                    the maximum radius, default 200
                ripleys_dr : float
                    the radius interval, default 5
                radii : 1D np array
                    the radius values. If given, ripleys_rmax and
                    ripleys_dr are ignored.
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                fp_combined_locs : str
                    filepath to the combined locs of all channel_locs
                atype : str
                    the type of analysis: 'Ripleys' for the standard
                    Ripley's K analysis, or 'RDF' for calculation of the
                    radial distribution function instead of K, and random
                    controls by relocating each point by a random x/y in a
                    circle with the currently investigated r, which preserves
                    the density fluctuations (instead of CSR simulation)
        """
        nRandomControls = parameters.get("ripleys_n_random_controls", 100)
        # radii = np.concatenate(
        #     (
        #         np.arange(0, 100, 2),
        #         np.arange(100, parameters.get("ripleys_rmax", 200), 12),
        #     )
        # )
        if (radii := parameters.get("radii")) is not None:
            radii = np.array(radii)
        else:
            radii = np.concatenate(
                (
                    np.arange(
                        0,
                        parameters.get("ripleys_rmax", 200),
                        parameters.get("ripleys_dr", 5),
                    ),
                )
            )

        if isinstance(parameters["fp_combined_locs"], list):
            fp_combined_locs = parameters["fp_combined_locs"][0]
        else:
            fp_combined_locs = parameters["fp_combined_locs"]
        combined_locs, _ = io.load_locs(fp_combined_locs)

        (
            ripleysResults,
            ripleysIntegrals,
            ripleysMeanVal,
        ) = run_ripleysAnalysis.performRipleysMultiAnalysis(
            path=results["folder"],
            filename="",
            fileIDs=self.channel_tags,
            radii=radii,
            nRandomControls=nRandomControls,
            channel_locs=self.channel_locs,
            combined_locs=combined_locs,
            pixelsize=self.pixelsize,
            atype=parameters["atype"],
        )

        results["fp_ripleys_meanval"] = os.path.join(
            results["folder"], "Ripleys_IntegralsMean.txt"
        )
        np.savetxt(results["fp_ripleys_meanval"], ripleysMeanVal)

        results["fp_fig_ripleys_meanval"] = self._plot_ripleys_integrals(
            ripleysMeanVal,
            results["folder"],
            self.channel_tags,
            parameters["atype"],
        )
        results["fp_fig_unnormalized"] = os.path.join(
            results["folder"], f"{parameters['atype']}_unnormalized.png"
        )
        results["fp_fig_normalized"] = os.path.join(
            results["folder"], f"{parameters['atype']}_normalized.png"
        )

        results["ripleys_significant"] = self._find_ripleys_significant(
            ripleysMeanVal,
            parameters["ripleys_threshold"],
            self.channel_tags,
        )

        return parameters, results

    # @module_decorator
    # def ripleysk_rafal(self, i, parameters, results):
    #     """Exactly along Rafal's code"""
    #     from picasso_workflow.outpost_modules.ripley_dcatlas_analysis \
    #         import analyze as analyze_whole_cell
    #     from picasso_workflow.outpost_modules.ripley_dcatlas_analysis \
    #         import postprocess_ripley_matrix

    #     rcode = generate_random_code(6)

    #     R_MAX = 200  # nm, maximum radius for Ripley's K analysis
    #     RADII = np.concatenate(
    #         (np.arange(4, 80, 2), np.arange(80, R_MAX + 1, 12))
    #     )
    #     # boundaries for plotting final ripley matrices (as described in
    #     # methods)
    #     VMIN, VMAX = [
    #         -2000,
    #         2000,
    #     ]

    #     # first: binary
    #     ripley_matrix, mask, area, fig_u, fig_n = analyze_whole_cell(
    #         self.channel_locs, RADII, binary=True
    #     )
    #     postprocessed = postprocess_ripley_matrix(ripley_matrix, RADII)
    #     path_save_integral_raw = os.path.join(
    #         results["folder"],
    #         f"raw_ripley_integral_binary-{rcode}.npy",
    #     )
    #     path_save_integral_postprocessed = os.path.join(
    #         results["folder"],
    #         f"postprocessed_ripley_integral_binary-{rcode}.npy",
    #     )
    #     np.save(path_save_integral_raw, ripley_matrix)
    #     # save in excel format
    #     df_raw = pd.DataFrame(
    #         ripley_matrix, index=self.channel_tags, columns=self.channel_tags
    #     )
    #     df_raw.to_excel(path_save_integral_raw.replace(".npy", ".xlsx"))
    #     df_pp = pd.DataFrame(
    #         postprocessed,
    #         index=self.channel_tags,
    #         columns=self.channel_tags,
    #     )
    #     df_pp.to_excel(
    #         path_save_integral_postprocessed.replace(".npy", ".xlsx")
    #     )

    #     def plot_and_save(matrix, savepath, vmin, vmax):
    #         plt.figure()
    #         plt.imshow(matrix, cmap="bwr_r", vmin=vmin, vmax=vmax)
    #         plt.xticks(range(6), self.channel_tags)
    #         plt.yticks(range(6), self.channel_tags)
    #         plt.colorbar()
    #         plt.savefig(savepath, dpi=150)
    #         plt.close()

    #     results["ripley_matrix_raw_binary"] = ripley_matrix
    #     results["mask area_binary"] = area
    #     results["fp_fig_mask_binary"] = os.path.join(
    #         results["folder"], f"mask_binary-{rcode}.png"
    #     )
    #     fig, ax = plt.subplots()
    #     ax.set_box_aspect(1)
    #     ax.set_title("mask")
    #     cmap = "hot"
    #     ax.imshow(
    #         mask,
    #         extent=[
    #             0,
    #             mask.shape[0] * 10 / 1000,  # 10 nm mask pixel size
    #             0,
    #             mask.shape[1] * 10 / 1000,
    #         ],
    #         cmap=cmap,
    #         origin="lower",
    #     )
    #     ax.set_xlabel("x [µm]")
    #     ax.set_ylabel("y [µm]")
    #     fig.savefig(results["fp_fig_mask_binary"])

    #     results["fp_fig_unnormalized_binary"] = os.path.join(
    #         results["folder"], f"unnormalized_binary-{rcode}.png"
    #     )
    #     fig_u.savefig(results["fp_fig_unnormalized_binary"])
    #     results["fp_fig_normalized_binary"] = os.path.join(
    #         results["folder"], f"normalized_binary-{rcode}.png"
    #     )
    #     fig_n.savefig(results["fp_fig_normalized_binary"])

    #     results["fp_fig_raw_binary"] = path_save_integral_raw.replace(
    #         ".npy", ".png"
    #     )
    #     plot_and_save(
    #         matrix=ripley_matrix,
    #         savepath=results["fp_fig_raw_binary"],
    #         vmin=-np.max(np.abs(ripley_matrix)),
    #         vmax=np.max(np.abs(ripley_matrix)),
    #     )
    #     results["fp_fig_postprocessed_binary"] = (
    #         path_save_integral_postprocessed.replace(".npy", ".png")
    #     )
    #     plot_and_save(
    #         matrix=postprocessed,
    #         savepath=results["fp_fig_postprocessed_binary"],
    #         vmin=VMIN,
    #         vmax=VMAX,
    #     )

    #     # second: density
    #     ripley_matrix, mask, area, fig_u, fig_n = analyze_whole_cell(
    #         self.channel_locs, RADII, binary=False
    #     )
    #     postprocessed = postprocess_ripley_matrix(ripley_matrix, RADII)
    #     path_save_integral_raw = os.path.join(
    #         results["folder"],
    #         f"raw_ripley_integral_density-{rcode}.npy",
    #     )
    #     path_save_integral_postprocessed = os.path.join(
    #         results["folder"],
    #         f"postprocessed_ripley_integral_density-{rcode}.npy",
    #     )
    #     np.save(path_save_integral_raw, ripley_matrix)
    #     # save in excel format
    #     df_raw = pd.DataFrame(
    #         ripley_matrix, index=self.channel_tags, columns=self.channel_tags
    #     )
    #     df_raw.to_excel(path_save_integral_raw.replace(".npy", ".xlsx"))
    #     df_pp = pd.DataFrame(
    #         postprocessed,
    #         index=self.channel_tags,
    #         columns=self.channel_tags,
    #     )
    #     df_pp.to_excel(
    #         path_save_integral_postprocessed.replace(".npy", ".xlsx")
    #     )

    #     def plot_and_save(matrix, savepath, vmin, vmax):
    #         plt.figure()
    #         plt.imshow(matrix, cmap="bwr_r", vmin=vmin, vmax=vmax)
    #         plt.xticks(range(6), self.channel_tags)
    #         plt.yticks(range(6), self.channel_tags)
    #         plt.colorbar()
    #         plt.savefig(savepath, dpi=150)
    #         plt.close()

    #     results["ripley_matrix_raw_density"] = ripley_matrix
    #     results["mask area_density"] = area
    #     results["fp_fig_mask_density"] = os.path.join(
    #         results["folder"], f"mask_density-{rcode}.png"
    #     )
    #     fig, ax = plt.subplots()
    #     ax.set_box_aspect(1)
    #     ax.set_title("mask")
    #     cmap = "hot"
    #     ax.imshow(
    #         mask,
    #         extent=[
    #             0,
    #             mask.shape[0] / 1000,
    #             0,
    #             mask.shape[1] / 1000,
    #         ],
    #         cmap=cmap,
    #         origin="lower",
    #     )
    #     ax.set_xlabel("x [µm]")
    #     ax.set_ylabel("y [µm]")
    #     fig.savefig(results["fp_fig_mask_density"])

    #     results["fp_fig_unnormalized_density"] = os.path.join(
    #         results["folder"], f"unnormalized_density-{rcode}.png"
    #     )
    #     fig_u.savefig(results["fp_fig_unnormalized_density"])
    #     results["fp_fig_normalized_density"] = os.path.join(
    #         results["folder"], f"normalized_density-{rcode}.png"
    #     )
    #     fig_n.savefig(results["fp_fig_normalized_density"])

    #     results["fp_fig_raw_density"] = path_save_integral_raw.replace(
    #         ".npy", ".png"
    #     )
    #     plot_and_save(
    #         matrix=ripley_matrix,
    #         savepath=results["fp_fig_raw_density"],
    #         vmin=-np.max(np.abs(ripley_matrix)),
    #         vmax=np.max(np.abs(ripley_matrix)),
    #     )
    #     results["fp_fig_postprocessed_density"] = (
    #         path_save_integral_postprocessed.replace(".npy", ".png")
    #     )
    #     plot_and_save(
    #         matrix=postprocessed,
    #         savepath=results["fp_fig_postprocessed_density"],
    #         vmin=VMIN,
    #         vmax=VMAX,
    #     )

    #     return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def ripleysk2(self, i, parameters, results):
        """Perforn Ripley's K analysis between the channels using
        Rafal's code.
        Args:
            parameters:
                ripleys_n_random_controls : int
                    number of random controls, default: 100
                ripleys_rmax : int
                    the maximum radius, default 200
                ripleys_dr : float
                    the radius interval, default 5
                radii : 1D np array
                    the radius values. If given, ripleys_rmax and
                    ripleys_dr are ignored.
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                area : float
                    the cell area in µm^2
                    optional. only used with controltype=CSR
                fp_mask : str
                    the filepath to the cell mask.
                    optional, only used with CSR. can be binary or density mask
                mask_pixel_size : float
                    the pixel size of mask pixels (move to mask class which
                    internally keeps this information)
                    optional, only used with controltype=CSR
                metric : str
                    the type of analysis: 'RK' for the standard
                    Ripley's K analysis, or 'RDF' for calculation of the
                    radial distribution function instead of K, and random
                    controls by relocating each point by a random x/y in a
                    circle with the currently investigated r, which preserves
                    the density fluctuations (instead of CSR simulation)
                    Alternatively, "FRC" for fraction of molecular types
                    within the radii.
                controltype : str
                    "CSR" or "RND". Control n_random_controls by either
                    CSR simulation within the density mask, or randomizing
                    the real data
                randomization_radius : float
                    for controltype "RND", the radius [nm] by which
                    to randomize.
                    optional.
                shuffle_self : bool
                    for metric "FRC", whether to shuffle only other types or
                    also the self type
                relocate_self : bool
                    for metric "FRC", whether to relocate centerpoints to
                    'type_self' after shuffling.
                fraction_exclude
                significance_threshold : float
                    threshold above which heatmap entries are colored
                normalization : str
                edge_correction : bool
                    if True, only locs further from mask edges than max radius
                    are used for evaluation
                showControlEnvelope : bool

        """
        nRandomControls = parameters.get("ripleys_n_random_controls", 100)
        # radii = np.concatenate(
        #     (
        #         np.arange(0, 100, 2),
        #         np.arange(100, parameters.get("ripleys_rmax", 200), 12),
        #     )
        # )
        if (radii := parameters.get("radii")) is not None:
            radii = np.array(radii)
        else:
            radii = np.concatenate(
                (
                    np.arange(
                        0,
                        parameters.get("ripleys_rmax", 200),
                        parameters.get("ripleys_dr", 5),
                    ),
                )
            )

        # if isinstance(parameters["fp_combined_locs"], list):
        #     fp_combined_locs = parameters["fp_combined_locs"][0]
        # else:
        #     fp_combined_locs = parameters["fp_combined_locs"]
        # combined_locs, _ = io.load_locs(fp_combined_locs)

        if fp_mask := parameters.get("fp_mask"):
            # mask = np.load(fp_mask)
            # mask = mask / np.sum(mask)
            mask = outpost_modules.mask.CellMask.load(fp_mask)
            area = mask.area * 1e6  # in nm^2
            mask_pixel_size = mask._upsample
        else:
            mask = None
            area = parameters.get("area", 1) * 1e6  # in nm^2
            mask_pixel_size = 1
        # mask_pixel_size = parameters.get("mask_pixel_size")

        pixelsize = self.pixelsize

        if parameters.get("edge_correction"):
            # make the mask smaller by the maximum radius, and apply
            max_r = np.max(radii)
            ec_mask = copy.copy(mask)
            ec_mask.erode(max_r)
            area = ec_mask.area * 1e6  # in nm^2
            locs_used = [
                ec_mask.apply_to_locs(locs) for locs in self.channel_locs
            ]
        else:
            locs_used = self.channel_locs

        mol_coords = [
            outpost_modules.ripleys.convert_picasso_to_coords(mol, pixelsize)
            for mol in locs_used
        ]

        if parameters["metric"] == "FRC":
            (
                ripley_matrix,
                fig_u,
                fig_n,
                curves,
                curves_norm,
            ) = outpost_modules.ripleys.typefraction_all_channels(
                mol_coords,
                radii,
                nRandomControls,
                names=self.channel_tags,
                shuffle_self=parameters.get("shuffle_self", False),
                relocate_self=parameters.get("relocate_self", False),
                fraction_exclude_self=parameters.get(
                    "fraction_exclude_self", False
                ),
                normalize_to_bulkfraction=parameters.get(
                    "normalize_to_bulkfraction", None
                ),
                showControlEnvelope=parameters.get(
                    "showControlEnvelope", None
                ),
            )
        else:
            (
                ripley_matrix,
                fig_u,
                fig_n,
                curves,
                curves_norm,
            ) = outpost_modules.ripleys.analyze_all_channels(
                mol_coords,
                mask,
                mask_pixel_size,
                area,
                radii,
                nRandomControls,
                names=self.channel_tags,
                metric=parameters["metric"],
                controltype=parameters.get("controltype"),
                randomization_radius=parameters.get("randomization_radius"),
                normalization=parameters.get("normalization"),
                aggfun=parameters.get("aggfun"),
                showControlEnvelope=parameters.get(
                    "showControlEnvelope", None
                ),
            )

        results["fp_curves"] = os.path.join(results["folder"], "curves.npy")
        np.save(results["fp_curves"], curves)
        results["fp_curves_norm"] = os.path.join(
            results["folder"], "curves_norm.npy"
        )
        np.save(results["fp_curves_norm"], curves_norm)

        # ripley_matrix = outpost_modules.ripleys.postprocess_ripley_matrix(
        #     ripley_matrix, radii
        # )

        results["fp_ripleys_meanval"] = os.path.join(
            results["folder"], "Ripleys_IntegralsMean.txt"
        )
        np.savetxt(results["fp_ripleys_meanval"], ripley_matrix)

        rcode = generate_random_code(6)

        results["fp_fig_ripleys_meanval"] = self._plot_ripleys_integrals(
            ripley_matrix,
            results["folder"],
            self.channel_tags,
            parameters["metric"],
            parameters.get("controltype", "None"),
            parameters.get("ripleys_threshold", None),
            suffix=rcode,
            significance_threshold=parameters.get(
                "significance_threshold", None
            ),
        )
        results["fp_fig_unnormalized"] = os.path.join(
            results["folder"],
            f"{parameters['metric']}_{parameters.get('controltype', 'None')}"
            + f"_unnormalized_{rcode}.png",
        )
        fig_u.savefig(results["fp_fig_unnormalized"])
        results["fp_fig_normalized"] = os.path.join(
            results["folder"],
            f"{parameters['metric']}_{parameters.get('controltype', 'None')}_"
            + f"normalized_{rcode}.png",
        )
        fig_n.savefig(results["fp_fig_normalized"])

        if parameters.get("ripleys_threshold"):
            r_sig = self._find_ripleys_significant(
                ripley_matrix,
                parameters.get("ripleys_threshold", 1),
                self.channel_tags,
            )
        else:
            r_sig = []
        results["ripleys_significant"] = r_sig

        return parameters, results

    def _plot_ripleys_integrals(
        self,
        ripleysMeanVal,
        folder,
        channel_tags,
        metric,
        controltype,
        threshold=None,
        std=None,
        suffix="",
        significance_threshold=None,
    ):
        fig, ax = plt.subplots()
        plot_ripleysMeanVal = ripleysMeanVal.copy()
        if threshold is None:
            threshold = np.abs(plot_ripleysMeanVal).max()
        if significance_threshold is not None:
            plot_ripleysMeanVal[
                np.abs(plot_ripleysMeanVal) <= significance_threshold
            ] = 0
        heatmap = ax.imshow(
            plot_ripleysMeanVal,
            cmap="coolwarm_r",
            vmin=-threshold,
            vmax=threshold,
        )
        ax.grid(False)
        ax.set_xticks(np.arange(plot_ripleysMeanVal.shape[0]))
        ax.set_yticks(np.arange(plot_ripleysMeanVal.shape[1]))
        # Add number annotations to cells
        for i in range(ripleysMeanVal.shape[0]):
            for j in range(ripleysMeanVal.shape[1]):
                txt = f"{ripleysMeanVal[i, j]:.2f}"
                if std is not None:
                    txt += f"\n+-{std[i, j]:.2f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="black",
                    size=8,
                )
        ax.set_xticklabels(channel_tags, rotation=45)
        ax.set_yticklabels(channel_tags, rotation=45)
        ax.set_title(f"Mean Value - {metric} normalized to {controltype}")
        cbar = plt.colorbar(heatmap, format="%.2f")
        cbar.set_label("z-score [95% ci intervals]", rotation=90, labelpad=15)
        fp_integrals = os.path.join(
            folder, f"{metric}_{controltype}_ripleysMeanVal_{suffix}.png"
        )
        fig.set_size_inches((9, 7))
        fig.savefig(fp_integrals)
        return fp_integrals

    def _find_ripleys_significant(
        self, ripleysIntegrals, threshold, channel_tags
    ):
        # elucidate significant pairs
        significant_pairs = []
        if threshold is None:
            return significant_pairs
        for i in range(len(channel_tags)):
            for j in range(i, len(channel_tags)):
                if ripleysIntegrals[i, j] > threshold:
                    significant_pairs.append(
                        (channel_tags[i], channel_tags[j])
                    )
        return significant_pairs

    #    @profile_resource_usage
    @module_decorator
    def ripleysk_average(self, i, parameters, results):
        """Average the results of multiple Ripley's K Analyses, analyse
        the significant pairs after averaging, and save them into the
        separate workflow manual folders (for further analysis there)
        Args:
            parameters:
                # fp_ripleys_integrals : list of str
                #     the various single analyses to average, e.g. of
                #     different workflows
                fp_workflows : list of str
                    the paths to the folders of separate workflows
                    where the separate ripleys analyses have been done
                report_names : list of str
                    the report names of those worklfows
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                atype : str
                    "Ripleys" or "RDF"
                # output_folders : list of str
                #     folders to write the significant pairs into. This can
                #     e.g. be the 'manual' results folders of the
                #     workflows, so these can proceed.
            optional:
                swkfl_ripleysk_key : str
                    the results key of the ripleysk module.
                    e.g. '05_ripleysk'
                swkfl_manual_key : str
                    the results key of the manual module to save the
                    integrals to
                if those two are not given, saving is not performed
        """
        # from picasso_workflow.workflow import WorkflowRunner

        # all_integrals = np.concat(
        #     [np.loadtxt(fp) for fp in parameters["fp_ripleys_integrals"]])
        # averaged_integrals = np.mean(all_integrals, axis=0)

        # check single intregals based on workflow file
        fp_ripleys_meanvals = []  # [""] * len(parameters["fp_workflows"])
        output_folders = []  # [""] * len(parameters["fp_workflows"])

        channel_tags = None
        search_dict = {
            (
                parameters["swkfl_ripleysk_key"],
                "fp_ripleys_meanval",
            ): fp_ripleys_meanvals,
            (parameters["swkfl_manual_key"], "folder"): output_folders,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # for i, (wkflfolder, report_name) in enumerate(
        #     zip(parameters["fp_workflows"], parameters["report_names"])
        # ):
        #     # find analysis folder
        #     postfix = WorkflowRunner._check_previous_runner(
        #         wkflfolder, report_name
        #     )
        #     # find aggregation WorkflowRunner config
        #     fp_wr_cfg = os.path.join(
        #         wkflfolder,
        #         report_name + "_" + postfix,
        #         report_name + "_aggregation_" + postfix,
        #         "WorkflowRunner.yaml",
        #     )
        #     with open(fp_wr_cfg, "r") as f:
        #         data = yaml.load(f, Loader=yaml.FullLoader)
        #     # check for results of 'ripleysk' module
        #     for mod_key, mod_res in data["results"].items():
        #         if mod_key == parameters.get("swkfl_ripleysk_key"):
        #             print(mod_key, mod_res)
        #             print(parameters.get("swkfl_ripleysk_key"))
        #             fp_ripleys_integrals[i] = mod_res["fp_ripleys_integrals"]
        #         elif mod_key == parameters.get("swkfl_manual_key"):
        #             print(mod_key, mod_res)
        #             print(parameters.get("swkfl_manual_key"))
        #             output_folders[i] = mod_res["folder"]
        #     # find AggregationWorkflowRunner config
        #     fp_wr_cfg = os.path.join(
        #         wkflfolder,
        #         report_name + "_" + postfix,
        #         "AggregationWorkflowRunner.yaml",
        #     )
        #     with open(fp_wr_cfg, "r") as f:
        #         data = yaml.load(f, Loader=yaml.FullLoader)
        #     channel_tags = data["aggregation_workflow"][
        #         "single_dataset_tileparameters"
        #     ]["#tags"]
        # fp_ripleys_integrals = [
        #     fp for fp in fp_ripleys_integrals if fp != ""
        # ]
        # output_folders = [fp for fp in output_folders if fp != ""]
        results["output_folders"] = output_folders

        # load and average the integrals
        all_integrals = np.stack(
            [np.loadtxt(fp) for fp in fp_ripleys_meanvals]
        )
        averaged_integrals = np.nanmean(all_integrals, axis=0)
        std_integrals = np.nanstd(all_integrals, axis=0)

        # save into own results folder
        results["fp_ripleys_meanvals"] = os.path.join(
            results["folder"], "Ripleys_MeanVals.txt"
        )
        np.savetxt(results["fp_ripleys_meanvals"], averaged_integrals)

        results["fp_figmeanvals"] = self._plot_ripleys_integrals(
            averaged_integrals,
            results["folder"],
            channel_tags,
            parameters["atype"],
            std=std_integrals,
        )

        significant_pairs = self._find_ripleys_significant(
            averaged_integrals, parameters["ripleys_threshold"], channel_tags
        )
        results["ripleys_significant"] = significant_pairs

        # save significant pairs into given folders
        results["fp_ripleys_significant"] = os.path.join(
            results["folder"], "significant_pairs.txt"
        )
        save_fp = [
            os.path.join(fol, "significant_pairs.yaml")
            for fol in output_folders
        ]
        save_fp.append(results["fp_ripleys_significant"])
        for fp in save_fp:
            with open(fp, "w") as f:
                yaml.dump(significant_pairs, f)
            # np.savetxt(fp, significant_pairs)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def ripleysk_average2(self, i, parameters, results):
        """Average the results of multiple Ripley's K Analyses, analyse
        the significant pairs after averaging, and save them into the
        separate workflow manual folders (for further analysis there)
        Args:
            parameters:
                # fp_ripleys_integrals : list of str
                #     the various single analyses to average, e.g. of
                #     different workflows
                fp_workflows : list of str
                    the paths to the folders of separate workflows
                    where the separate ripleys analyses have been done
                report_names : list of str
                    the report names of those worklfows
                ripleys_threshold : float
                    the threshold of ripleys integrals above which the
                    interaction is deemed significant.
                metric : str
                    the type of analysis: 'RK' for the standard
                    Ripley's K analysis, or 'RDF' for calculation of the
                    radial distribution function instead of K, and random
                    controls by relocating each point by a random x/y in a
                    circle with the currently investigated r, which preserves
                    the density fluctuations (instead of CSR simulation)
                controltype : str
                    "CSR" or "RND". Control n_random_controls by either
                    CSR simulation within the density mask, or randomizing
                    the real data
                randomization_radius : float
                    for controltype "RND", the radius [nm] by which to
                    randomize.
                # output_folders : list of str
                #     folders to write the significant pairs into. This can
                #     e.g. be the 'manual' results folders of the
                #     workflows, so these can proceed.
            optional:
                swkfl_ripleysk_key : str
                    the results key of the ripleysk module.
                    e.g. '05_ripleysk'
                swkfl_manual_key : str
                    the results key of the manual module to save the
                    integrals to
                if those two are not given, saving is not performed
        """
        # from picasso_workflow.workflow import WorkflowRunner

        # all_integrals = np.concat(
        #     [np.loadtxt(fp) for fp in parameters["fp_ripleys_integrals"]])
        # averaged_integrals = np.mean(all_integrals, axis=0)

        # check single intregals based on workflow file
        fp_ripleys_meanvals = []  # [""] * len(parameters["fp_workflows"])
        fp_curves = []
        fp_curves_norm = []

        channel_tags = None

        # load single dataset results
        search_dict = {
            (
                parameters["swkfl_ripleysk_key"],
                "fp_ripleys_meanval",
            ): fp_ripleys_meanvals,
            (
                parameters["swkfl_ripleysk_key"],
                "fp_curves",
            ): fp_curves,
            (
                parameters["swkfl_ripleysk_key"],
                "fp_curves_norm",
            ): fp_curves_norm,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # load single dataset parameters
        ripleys_thresholds = []
        ripleys_metrics = []
        ripleys_controltypes = []
        ripleys_radii = []
        significance_thresholds = []
        search_dict = {
            (
                parameters["swkfl_ripleysk_key"],
                "ripleys_threshold",
            ): ripleys_thresholds,
            (
                parameters["swkfl_ripleysk_key"],
                "metric",
            ): ripleys_metrics,
            (
                parameters["swkfl_ripleysk_key"],
                "controltype",
            ): ripleys_controltypes,
            (
                parameters["swkfl_ripleysk_key"],
                "radii",
            ): ripleys_radii,
            (
                parameters["swkfl_ripleysk_key"],
                "significance_threshold",
            ): significance_thresholds,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_parameter_keys=search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)
        # check that all thresholds, metrics and controltypes are the same
        ripleys_threshold = set(ripleys_thresholds)
        if len(ripleys_threshold) > 1:
            raise ValueError(
                "All ripleys_threshold values should be the same, but "
                + f"got: {ripleys_thresholds}"
            )
        ripleys_threshold = ripleys_thresholds[0]
        ripleys_metric = set(ripleys_metrics)
        if len(ripleys_metric) > 1:
            raise ValueError(
                "All ripleys_metric values should be the same, but "
                + f"got: {ripleys_metrics}"
            )
        ripleys_metric = ripleys_metrics[0]
        ripleys_controltype = set(ripleys_controltypes)
        if len(ripleys_controltype) > 1:
            raise ValueError(
                "All ripleys_controltype values should be the same, but "
                + f"got: {ripleys_controltypes}"
            )
        ripleys_controltype = ripleys_controltypes[0]

        ripleys_radii = ripleys_radii[0]

        significance_threshold = significance_thresholds[0]

        # load and plot the single curves
        fig_curves, ax_curves = outpost_modules.ripleys.init_plot(
            len(channel_tags),
            "un-normalized",
            ripleys_controltype,
            ripleys_metric,
            figsize_per_target=5,
        )
        fig_curves_norm, ax_curves_norm = outpost_modules.ripleys.init_plot(
            len(channel_tags),
            "normalized",
            ripleys_controltype,
            ripleys_metric,
            figsize_per_target=5,
        )
        for fp_curve, fp_curve_norm, reportname in zip(
            fp_curves, fp_curves_norm, parameters["report_names"]
        ):
            curves = np.load(fp_curve)
            curves_norm = np.load(fp_curve_norm)
            for i, name1 in enumerate(channel_tags):
                for j, name2 in enumerate(channel_tags):
                    outpost_modules.ripleys.plot_ripleys(
                        ripleys_radii,
                        curves[i, j, :],
                        None,
                        ci=0.95,
                        normalized=False,
                        showControls=False,
                        title=f"{name1} -> {name2}",
                        labelFontsize=30,
                        axes=ax_curves[i, j],
                        metric=ripleys_metric,
                        label_data=reportname,
                        showControlEnvelope=False,
                    )
                    outpost_modules.ripleys.plot_ripleys(
                        ripleys_radii,
                        curves_norm[i, j, :],
                        None,
                        ci=0.95,
                        normalized=True,
                        showControls=False,
                        title=f"{name1} -> {name2}",
                        labelFontsize=30,
                        axes=ax_curves_norm[i, j],
                        metric=ripleys_metric,
                        label_data=reportname,
                        showControlEnvelope=False,
                    )
                    if i < len(channel_tags) - 1:
                        ax_curves[i, j].xaxis.label.set_visible(False)
                        ax_curves_norm[i, j].xaxis.label.set_visible(False)
                        ax_curves[i, j].set_xticks([])
                        ax_curves_norm[i, j].set_xticks([])
                    if j > 0:
                        ax_curves[i, j].yaxis.label.set_visible(False)
                        ax_curves_norm[i, j].yaxis.label.set_visible(False)

        rcode = generate_random_code(6)
        results["fp_fig_unnormalized"] = os.path.join(
            results["folder"],
            f"{ripleys_metric}_{ripleys_controltype}_{rcode}"
            + "_unnormalized.png",
        )
        fig_curves.savefig(results["fp_fig_unnormalized"])
        results["fp_fig_normalized"] = os.path.join(
            results["folder"],
            f"{ripleys_metric}_{ripleys_controltype}_{rcode}_"
            + "normalized.png",
        )
        fig_curves_norm.savefig(results["fp_fig_normalized"])

        # load and average the integrals
        all_integrals = np.stack(
            [np.loadtxt(fp) for fp in fp_ripleys_meanvals]
        )
        averaged_integrals = np.nanmean(all_integrals, axis=0)
        std_integrals = np.nanstd(all_integrals, axis=0)

        # save into own results folder
        results["fp_ripleys_meanvals"] = os.path.join(
            results["folder"], "Ripleys_MeanVals.txt"
        )
        np.savetxt(results["fp_ripleys_meanvals"], averaged_integrals)

        results["fp_figmeanvals"] = self._plot_ripleys_integrals(
            averaged_integrals,
            results["folder"],
            channel_tags,
            ripleys_metric,
            ripleys_controltype,
            ripleys_threshold,
            std=std_integrals,
            suffix=rcode,
            significance_threshold=significance_threshold,
        )

        significant_pairs = self._find_ripleys_significant(
            averaged_integrals, ripleys_threshold, channel_tags
        )
        results["ripleys_significant"] = significant_pairs

        # save significant pairs into given folders
        results["fp_ripleys_significant"] = os.path.join(
            results["folder"], "significant_pairs.txt"
        )
        with open(results["fp_ripleys_significant"], "w") as f:
            yaml.dump(significant_pairs, f)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def protein_interactions(self, i, parameters, results):
        """Perform interaction analysis on those dataset pairs that showed
        significance in Ripley's K analysis. The interaction analysis consists
        of
        (1) calculating proportion of singly or doubly co-occurring instances
            of the single receptors (in clusters)
        (2) calculating the co-occurrence of these single or double events of
            one receptor with single or double events of another receptor
            within a cluster (where Ripley's K showed significance)
        This approach stems from a time of early development of SPINNA.
        Nowadays, this could be done directly but potentially with slightly
        different results.
        Fixed to 2D. Fixed to only using 1st nearest neighbor
        Args:
            parameters:
                channel_map : dict
                    maps between channels (protein names, tags before
                    combining) and index in the combine_id column of combined
                    locs
                labeling_efficiency : dict, channel tag to float, range 0-100
                    labeling efficiency percentage, default for all targets
                labeling_uncertainty : dict, channel tag to float
                    labeling uncertainty [nm]; good value is e.g. 5
                n_simulate : int
                    number of target molecules to be simulated;
                    good value is e.g. 50000
                density : dict, channel tag to float
                    density to simulate [nm^2 or nm^3];
                    area density if 2D; volume density if 3D
                nn_nth : int
                    number of nearest neighbors to analyse
                structure_distance : float
                    the protein distance between each other in nm
                res_factor : float
                    the spinna res_factor
                sim_repeats : int
                    number of simulation repeats, for noise reduction
                interaction_pairs: list of list of two strings, or str
                    pairs that are able to interact
                    if str: filepath to a yaml file with list of tuples
        """
        logger.debug("Molecular interactions")

        # # homo-analysis (proportions of 1- or 2-mers of the same kind)
        # props = {}
        dimensionality = 2
        pixelsize = self.pixelsize
        if isinstance(parameters["density"], list):
            density = {
                tag: parameters["density"][cid]
                for tag, cid in parameters["channel_map"].items()
            }
        elif isinstance(parameters["density"], dict):
            density = parameters["density"]
        else:
            raise KeyError("density parameter must be list of dict.")
        results["fp_density"] = os.path.join(results["folder"], "density.yaml")
        with open(results["fp_density"], "w") as f:
            yaml.dump(density, f)

        # ground thruth density, adjusted by labeling efficiency
        density_gt = {
            tag: density[tag] / parameters["labeling_efficiency"][tag]
            for tag in density.keys()
        }

        # compound_density = sum(parameters["density"].values())
        # area = parameters["n_simulate"] / (compound_density / 1e6)
        # n_sim_targets = {
        #     tag:
        #     int(parameters["n_simulate"] * compound_density / den)
        #     for tag, den in parameters["density"].items()
        # }
        # pixelsize = self.analysis_config["camera_info"].get("Pixelsize")
        # structures = self._create_spinna_structure(
        #     self.channel_tags, [1, 2], distance=parameters["distance"])
        # N_structures = picasso_outpost.generate_N_structures(
        #     structures, n_sim_targets, parameters["res_factor"]
        # )
        # # bin size: more than Nyquist subsampling
        # expected_1stNN_peak = (
        #     2 / (2 * dimensionality * np.pi * parameters["density"])
        # ) ** (1 / dimensionality)
        # fit_NND_bin = expected_1stNN_peak / 10
        # # max dist: a few times the first NN distance peak
        # fit_NND_maxdist = 20 * expected_1stNN_peak
        # for tag, locs in zip(self.channel_tags, self.channel_locs):
        #     spinna_parameters = {
        #         "structures": self._create_spinna_structure(
        #             [tag], [[1, 2]], parameters["structure_distance"]),
        #         "label_unc": parameters["labeling_uncertainty"],
        #         "le": parameters["labeling_efficiency"],
        #         "mask_dict": None,
        #         "width": np.sqrt(area * 1e6),
        #         "height": np.sqrt(area * 1e6),
        #         "depth": None,
        #         "random_rot_mode": "2D",
        #         "exp_data": {tag: np.stack((locs[['x', 'y']] * pixelsize))},
        #         "sim_repeats": parameters["sim_repeats"],
        #         "fit_NND_bin": [fit_NND_bin],
        #         "fit_NND_maxdist": [fit_NND_maxdist],
        #         "N_structures": N_structures,
        #         "save_filename": (
        #              os.path.join(results["folder"], "homo-{tag}")
        #         ),
        #         "asynch": True,
        #         "targets": [tag],
        #         "apply_mask": False,
        #         "nn_plotted": parameters["nn_nth"],
        #         "result_dir": results["folder"],
        #     }
        #     result, fp_fig = (
        #         picasso_outpost.spinna_sgl_temp(spinna_parameters)
        #     )
        #     props[tag] = result["Fitted proportions of structures"]
        # logger.debug(f'found proportions of {props}')

        if isinstance(parameters["interaction_pairs"], str):
            with open(parameters["interaction_pairs"], "r") as f:
                interaction_pairs = yaml.safe_load(f)
        else:
            interaction_pairs = parameters["interaction_pairs"]
            # np.savetxt(fp, significant_pairs)

        # hetero-analysis (pairwise up to 2+2-mers)
        # structures: A, B, AA, BB, AB, AABB
        props = {}
        fp_allfigs = []
        for A, B in interaction_pairs:
            # if A == B:  # or should we include homotetramers?
            #     continue
            logger.debug(
                f"analysing interaction between {A} and {B} with SPINNA."
            )
            # find index of A and B in self.channel_locs
            ia = self.channel_tags.index(A)
            ib = self.channel_tags.index(B)

            # locs, but as np.ndarray
            exp_data = {}
            for i, target in zip([ia, ib], [A, B]):
                locs = self.channel_locs[i]
                if "z" in locs.columns:
                    exp_data[target] = np.stack(
                        (
                            locs["x"] * pixelsize,
                            locs["y"] * pixelsize,
                            locs["z"],
                        )
                    ).T
                    # dim = 3
                else:
                    exp_data[target] = np.stack(
                        (locs["x"] * pixelsize, locs["y"] * pixelsize)
                    ).T
                    # dim = 2
            structures = self._create_spinna_structure(
                [A], [[1, 2]], parameters["structure_distance"]
            )
            if A != B:
                structures += self._create_spinna_structure(
                    [B], [[1, 2]], parameters["structure_distance"]
                )
                # heterodimer
                struct = {
                    "Molecular targets": [A, B],
                    "Structure title": f"{A}-{B}-heterodimer",
                    f"{A}_x": [-parameters["structure_distance"] / 2],
                    f"{A}_y": [0],
                    f"{A}_z": [0],
                    f"{B}_x": [parameters["structure_distance"] / 2],
                    f"{B}_y": [0],
                    f"{B}_z": [0],
                }
                structures.append(struct)
                # heterotetramer, in a square
                struct = {
                    "Molecular targets": [A, B],
                    "Structure title": f"{A}-{B}-heterotetramer",
                    f"{A}_x": [
                        -parameters["structure_distance"] / 2,
                        parameters["structure_distance"] / 2,
                    ],
                    f"{A}_y": [
                        -parameters["structure_distance"] / 2,
                        -parameters["structure_distance"] / 2,
                    ],
                    f"{A}_z": [0, 0],
                    f"{B}_x": [
                        -parameters["structure_distance"] / 2,
                        parameters["structure_distance"] / 2,
                    ],
                    f"{B}_y": [
                        parameters["structure_distance"] / 2,
                        parameters["structure_distance"] / 2,
                    ],
                    f"{B}_z": [0, 0],
                }
                structures.append(struct)

                compound_density = (
                    density_gt[A] / parameters["labeling_efficiency"][A]
                    + density_gt[B] / parameters["labeling_efficiency"][B]
                )
                # area = parameters["n_simulate"] / (compound_density / 1e6)
                # area = parameters["n_simulate"] / (compound_density)
                area = parameters["n_simulate"] / (compound_density * 1e6)
                n_sim_targets = {
                    tag: int(
                        parameters["n_simulate"]
                        * density_gt[tag]
                        / compound_density
                    )
                    for tag in [A, B]
                }
            else:
                compound_density = density_gt[A]
                area = parameters["n_simulate"] / (density_gt[A] * 1e6)
                n_sim_targets = {A: int(parameters["n_simulate"])}
            structures, targets = picasso_outpost.load_structures_from_dict(
                structures
            )

            N_structures = picasso_outpost.generate_N_structures(
                structures, n_sim_targets, parameters["res_factor"]
            )

            # bin size: more than Nyquist subsampling
            expected_1stNN_peak = (
                2 / (2 * dimensionality * np.pi * (compound_density / 2))
            ) ** (1 / dimensionality)
            fit_NND_bin = expected_1stNN_peak / 3
            # max dist: a few times the first NN distance peak
            fit_NND_maxdist = 20 * expected_1stNN_peak

            spinna_parameters = {
                "structures": structures,
                "label_unc": parameters["labeling_uncertainty"],
                "le": parameters["labeling_efficiency"],
                "mask_dict": None,
                "width": np.sqrt(area * 1e6),
                "height": np.sqrt(area * 1e6),
                "depth": None,
                "random_rot_mode": "2D",
                "exp_data": exp_data,
                "sim_repeats": parameters["sim_repeats"],
                "fit_NND_bin": fit_NND_bin,
                "fit_NND_maxdist": fit_NND_maxdist,
                "N_structures": N_structures,
                "save_filename": os.path.join(
                    results["folder"], f"interaction-{A}-{B}"
                ),
                "asynch": True,
                "targets": [A, B],
                "apply_mask": False,
                "nn_plotted": parameters["nn_nth"],
                "result_dir": results["folder"],
                "n_simulated": n_sim_targets,
            }

            result, fp_fig = picasso_outpost.single_spinna_run(
                spinna_parameters
            )
            plt.close("all")
            props[f"{A},{B}"] = result["Fitted proportions of structures"]
            fp_allfigs.append(fp_fig)
            # break

        logger.debug(f"proportions: {props}")
        results["fp_allfigs"] = fp_allfigs
        results["Interaction proportions"] = props
        results["fp_proportions"] = os.path.join(
            results["folder"], "interaction_proportions.yaml"
        )
        with open(results["fp_proportions"], "w") as f:
            yaml.dump(props, f)

        results["fp_proportions"] = os.path.join(
            results["folder"], "interaction_proportions.pkl"
        )
        with open(results["fp_proportions"], "wb") as f:
            pickle.dump(props, f)

        # import json
        # results["fp_proportions"] = os.path.join(
        #     results["folder"], "interaction_proportions.json")
        # with open(results["fp_proportions"], 'w') as f:
        #     json.dump(props, f)

        cols = ["A", "AA", "B", "BB", "AB", "AABB"]
        df = pd.DataFrame(columns=cols, index=props.keys())
        for k, v in props.items():
            if len(v) == len(df.columns):
                df.loc[k, :] = v
            elif len(v) == 2:
                # this is homo-analysis, only A, and AA
                df.loc[k, ["A", "AA"]] = v
            else:
                raise NotImplementedError("")
        results["fp_proportions"] = os.path.join(
            results["folder"], "interaction_proportions.xlsx"
        )
        df.to_excel(results["fp_proportions"])

        # from these results, calculate the proportion of direct
        # interaction, so AB or AABB vs all other (A, AA, B, BB);
        # for self-interactions: AA vs A
        df_di = pd.DataFrame(
            index=self.channel_tags, columns=self.channel_tags, data=np.nan
        )
        for pair, row in df.iterrows():
            A, B = pair.split(",")
            if A == B:
                prop = 2 * row["AA"] / (row["A"] + 2 * row["AA"])
                df_di.loc[A, B] = 100 * prop
            else:
                # proportion of A interacting with any number of B
                prop = (row["AB"] + 2 * row["AABB"]) / (
                    row["A"] + 2 * row["AA"] + row["AB"] + 2 * row["AABB"]
                )
                df_di.loc[A, B] = 100 * prop
                # proportion of B interacting with any number of A
                prop = (row["AB"] + 2 * row["AABB"]) / (
                    row["B"] + 2 * row["BB"] + row["AB"] + 2 * row["AABB"]
                )
                df_di.loc[B, A] = 100 * prop
        results["fp_interaction_map"] = os.path.join(
            results["folder"], "interaction_map.xlsx"
        )
        df_di.to_excel(results["fp_interaction_map"])
        results["fp_fig_imap"] = self._plot_direct_interaction(
            df_di, results["folder"]
        )

        return parameters, results

    def _plot_direct_interaction(self, direct_interaction, folder, std=None):
        """
        Args:
            direct_interaction : DataFrame
                index, columns: channel_tags
                values: percentage of interaction
        """
        fig, ax = plt.subplots()
        heatmap = ax.imshow(
            direct_interaction.values, cmap="Blues", vmin=0, vmax=100
        )
        ax.grid(False)
        ax.set_xticks(np.arange(len(direct_interaction.columns)))
        ax.set_yticks(np.arange(len(direct_interaction.index)))
        # Add number annotations to cells
        for i, A in enumerate(direct_interaction.columns):
            for j, B in enumerate(direct_interaction.index):
                txt = f"{direct_interaction.loc[A, B]:.2f}"
                if std is not None:
                    txt += f"\n+-{std.loc[A, B]:.2f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="black",
                    size=8,
                )
        ax.set_xticklabels(direct_interaction.columns, rotation=45)
        ax.set_yticklabels(direct_interaction.index, rotation=45)
        ax.set_title("Percentage of [row] interacting at 10 nm with [col]")
        plt.colorbar(heatmap, format="%.2f")
        fp_imap = os.path.join(folder, "interaction_map.png")
        fig.set_size_inches((9, 7))
        fig.savefig(fp_imap)
        return fp_imap

    #    @profile_resource_usage
    @module_decorator
    def protein_interactions_average(self, i, parameters, results):
        """Average the results of multiple "protein_interactions" analyses.
        Create a bar plot with mean and stddev of the different proportions
        of interaction partners.
        Args:
            parameters:
                fp_workflows : list of str
                    the paths to the folders of separate workflows
                    where the separate ripleys analyses have been done
                report_names : list of str
                    the report names of those worklfows
                swkfl_protint_key : str
                    the results key of the protein interactions module.
                    e.g. '05_protein_interactions'
            optional:
        """
        # check single intregals based on workflow file
        fp_proportions = []
        fp_interaction_map = []

        channel_tags = None
        search_dict = {
            (
                parameters["swkfl_protint_key"],
                "fp_proportions",
            ): fp_proportions,
            (
                parameters["swkfl_protint_key"],
                "fp_interaction_map",
            ): fp_interaction_map,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # load the interaction maps
        all_imap = []
        for fp in fp_interaction_map:
            all_imap.append(pd.read_excel(fp, index_col=0, header=0))
        mean_imap = np.mean(np.stack(all_imap), axis=0)
        df_mean = pd.DataFrame(
            index=all_imap[0].index,
            columns=all_imap[0].columns,
            data=mean_imap,
        )
        std_imap = np.std(np.stack(all_imap), axis=0)
        df_std = pd.DataFrame(
            index=all_imap[0].index,
            columns=all_imap[0].columns,
            data=std_imap,
        )
        results["fp_fig_imap"] = self._plot_direct_interaction(
            df_mean, results["folder"], df_std
        )

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def create_mask(self, i, parameters, results):
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
        from picasso_workflow.dbscan_molint import mask

        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)
        # locs for the mask are the combined locs
        if isinstance(parameters["fp_combined_locs"], list):
            fp_combined_locs = parameters["fp_combined_locs"][0]
        else:
            fp_combined_locs = parameters["fp_combined_locs"]
        combined_locs, combined_info = io.load_locs(fp_combined_locs)
        # self.channel_locs = [combined_locs]
        multi_filename = "multi_ID.hdf5"
        pixelsize = self.pixelsize
        mask_dict = mask.gen_mask(
            combined_locs["x"],
            combined_locs["y"],
            parameters["margin"],
            parameters["binsize"],
            parameters["sigma_mask_blur"],
            parameters["mask_resolution"],
            pixelsize,
            results["folder"],
            filename=multi_filename,
            plot_figures=True,
        )

        # get exp coordinates in mask
        new_info = combined_info + [
            {
                "Generated by": "picasso-workflow: create_mask",
            }
        ]
        # self.channel_info = [new_info]
        df_merge_mask, mask_dict = mask.exp_data_in_mask(
            pd.DataFrame(combined_locs),
            mask_dict,
            pixelsize,
            results["folder"],
            multi_filename,
            new_info,
            plot_figures=True,
        )
        results["fp_fig_blur"] = os.path.join(
            results["folder"], "mask", "multi_ID_blurred_exp_data.png"
        )
        results["fp_fig_mask"] = os.path.join(
            results["folder"], "mask", "multi_ID_mask_final.png"
        )

        results["fp_merge_mask"] = os.path.join(
            results["folder"], "merge_mask.hdf5"
        )
        df_merge_mask.to_hdf(results["fp_merge_mask"], key="locs")

        # Get densities of individual proteins:
        N_proteins = df_merge_mask.groupby(parameters["combine_col"]).size()

        for protein, protein_ID in channel_map.items():
            N = N_proteins.loc[protein_ID]
            area = mask_dict["area"]
            density = N / area

            mask_dict["N_exp_" + protein] = N
            mask_dict["density_exp_" + protein + " (/um^2)"] = density

        mask_dict["info"] = new_info
        mask_dict["filename"] = results["fp_merge_mask"]

        fp_mask_dict = os.path.join(results["folder"], "mask_dict.pkl")
        results["fp_mask_dict"] = fp_mask_dict
        with open(fp_mask_dict, "wb+") as f:
            pickle.dump(mask_dict, f)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def create_mask2(self, i, parameters, results):
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
        logger.debug("before creating mask")
        xmin = [locs["x"].min() for locs in self.channel_locs]
        logger.debug(f"min locs x vals [px] are {xmin}")
        ymin = [locs["y"].min() for locs in self.channel_locs]
        logger.debug(f"min locs y vals [px] are {ymin}")
        xmax = [locs["x"].max() for locs in self.channel_locs]
        logger.debug(f"max locs x vals [px] are {xmax}")
        ymax = [locs["y"].max() for locs in self.channel_locs]
        logger.debug(f"max locs y vals [px] are {ymax}")
        pixelsize = self.pixelsize
        # # get map
        # with open(parameters["fp_channel_map"], "r") as f:
        #     channel_map = yaml.safe_load(f)
        # locs for the mask are the combined locs
        rcode = generate_random_code(6)
        results["fp_scene_locs_before"] = os.path.join(
            results["folder"], f"locs_before_{rcode}.png"
        )
        render.plot_scene(
            self.channel_locs,
            pixelsize,
            pixelsize,
            fp=results["fp_scene_locs_before"],
        )
        fp_combined_locs = parameters.get("fp_combined_locs", None)
        if fp_combined_locs:
            if isinstance(parameters["fp_combined_locs"], list):
                fp_combined_locs = parameters["fp_combined_locs"][0]
            else:
                fp_combined_locs = parameters["fp_combined_locs"]
            combined_locs, combined_info = io.load_locs(fp_combined_locs)
            mols = [combined_locs]
        else:
            mols = [locs for locs in self.channel_locs]

        # mol_coords = [
        #     outpost_modules.ripleys.convert_picasso_to_coords(mol, pixelsize)
        #     for mol in mols
        # ]
        binsize = parameters["binsize"]
        blursize = parameters["blursize"]
        # blur = parameters["blursize"] / binsize
        threshold = parameters["threshold"]
        mask_pixel_size = parameters["mask_pixel_size"]
        # binary = parameters["binary"]
        cell_mask = outpost_modules.mask.CellMask.from_mol_coords(
            mols,
            pixelsize,
            binsize,
            blursize,
            threshold,
            upsample=mask_pixel_size,
        )
        if parameters.get("select_cell"):
            kwargs = {}
            if fill_holes := parameters.get("fill_holes"):
                kwargs["fill_holes"] = fill_holes
            if fill_holes := parameters.get("nth_largest_cell"):
                kwargs["nth_largest"] = fill_holes
            cell_mask.filter_mask(**kwargs)
        if dilate_nm := parameters.get("dilate_nm"):
            cell_mask.dilate(dilate_nm)
        if parameters.get("apply_to_locs"):
            self.channel_locs = [
                cell_mask.apply_to_locs(locs) for locs in self.channel_locs
            ]
            results["fp_scene_locs_after"] = os.path.join(
                results["folder"], f"locs_after_{rcode}.png"
            )
            render.plot_scene(
                self.channel_locs,
                pixelsize,
                pixelsize,
                fp=results["fp_scene_locs_after"],
            )
            logger.debug("after applying mask to locs")
            xmin = [locs["x"].min() for locs in self.channel_locs]
            logger.debug(f"min locs x vals [px] are {xmin}")
            ymin = [locs["y"].min() for locs in self.channel_locs]
            logger.debug(f"min locs y vals [px] are {ymin}")
            xmax = [locs["x"].max() for locs in self.channel_locs]
            logger.debug(f"max locs x vals [px] are {xmax}")
            ymax = [locs["y"].max() for locs in self.channel_locs]
            logger.debug(f"max locs y vals [px] are {ymax}")
        area = cell_mask.area

        # mask, area = outpost_modules.ripleys.get_cell_mask(
        #     mol_coords,
        #     pixelsize,
        #     binsize=binsize,
        #     blur=blur,
        #     threshold=threshold,
        #     upsample=mask_pixel_size,
        #     binary=binary,
        # )
        # if parameters.get("select_cell"):
        #     mask = outpost_modules.ripleys.filter_mask(mask)

        results["area"] = area
        # results["fp_mask"] = os.path.join(
        #     results["folder"], f"mask_binary-{binary}.npy"
        # )
        # np.save(results["fp_mask"], mask)
        results["fp_mask"] = os.path.join(results["folder"], "mask.pkl")
        cell_mask.save(results["fp_mask"])
        results["mask_pixel_size"] = mask_pixel_size

        # results["fp_fig_mask"] = os.path.join(
        #     results["folder"], f"mask_binary-{binary}_{rcode}.png"
        # )
        # outpost_modules.ripleys.plot_mask(
        #     mask, mask_pixel_size, results["fp_fig_mask"]
        # )
        results["fp_fig_mask_binary"] = os.path.join(
            results["folder"], f"mask_binary_{rcode}.png"
        )
        cell_mask.plot_mask(results["fp_fig_mask_binary"], binary=True)
        results["fp_fig_mask_density"] = os.path.join(
            results["folder"], f"mask_density_{rcode}.png"
        )
        cell_mask.plot_mask(results["fp_fig_mask_density"], binary=False)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def refine_mask_by_density(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        nth_largest = parameters.get("nth_largest", 0)
        mask = outpost_modules.mask.CellMask.load(parameters["fp_mask"])
        mask_pixel_area = mask._upsample**2
        densities = mask.densities

        nbins = parameters.get("nbins", 20)
        fig, ax = plt.subplots()
        densities_to_plot = densities.copy().ravel()
        densities_to_plot = densities_to_plot[densities_to_plot > 0]

        if (
            "min_density" in parameters.keys()
            and "max_density" in parameters.keys()
            and parameters["max_density"] > 0
        ):
            min_density = parameters["min_density"]
            max_density = parameters["max_density"]
        elif std_cutoff := parameters["density_std_cutoff"]:
            median_nlocs = np.median(densities_to_plot) * mask_pixel_area
            min_nlocs = median_nlocs - std_cutoff * np.sqrt(median_nlocs)
            max_nlocs = median_nlocs + std_cutoff * np.sqrt(median_nlocs)
            median_density = median_nlocs / mask_pixel_area
            min_density = min_nlocs / mask_pixel_area
            max_density = max_nlocs / mask_pixel_area
            logger.debug(
                f"median nlocs {median_nlocs}, poisson std\
                {np.sqrt(median_nlocs)}"
            )
            logger.debug(f"min nlocs {min_nlocs}, max nlocs {max_nlocs}")
            logger.debug(
                f"median density {median_density}, poisson std \
                {np.sqrt(median_density)}"
            )
            logger.debug(
                f"min density {min_density}, max density {max_density}"
            )
        else:
            raise KeyError(
                "Either 'min_density' and 'max_density or "
                + "'density_std_cutoff' need to be given."
            )

        n, bins, patches = ax.hist(
            densities_to_plot * 1e6,
            bins=nbins,
            color="b",
            label="densities in mask",
        )
        ylim = ax.get_ylim()
        logger.debug(f"xlims: {ax.get_xlim()}")
        ax.plot(
            [
                min_density * 1e6,
                min_density * 1e6,
                np.nan,
                max_density * 1e6,
                max_density * 1e6,
            ],
            [0, 0.9 * ylim[1], np.nan, 0, 0.9 * ylim[1]],
            color="r",
            label="selection boundaries",
        )
        ax.set_xlabel("Density [µm^(-2)]")
        ax.set_ylabel("#bins")
        ax.set_title("Density Histogram of input mask")
        rcode = generate_random_code(6)
        results["fp_density_hist_before"] = os.path.join(
            results["folder"], f"density_hist_before_{rcode}.png"
        )
        fig.savefig(results["fp_density_hist_before"])

        # select requested densities
        densities[(densities < min_density) | (densities > max_density)] = 0
        assert densities.shape != (0,)
        # select nth largest connected area
        labeled_array, num_features = label(densities > 0)
        sizes = np.bincount(labeled_array.ravel())
        try:
            component_index = sizes[1:].argmax() + 1 - nth_largest
        except ValueError:
            component_index = 1
        component_mask = (labeled_array == component_index).astype(np.int8)
        mask._binary_mask = component_mask.astype(np.bool_)
        mask._recalc_density_mask_from_binary()

        if smoothe_nm := parameters.get("smoothe_nm"):
            # smoothe the mask by dilating and eroding
            # this removes e.g. single-pixel holes that are due to
            # local density variations. it is better to keep those
            # to reduce boundary effects
            mask.dilate(smoothe_nm)
            mask.erode(smoothe_nm)
            # secondly, first erode, select largest component,
            # and then enlarge again
            mask.erode(smoothe_nm)
            mask.filter_mask(nth_largest=0, fill_holes=False)
            mask.dilate(smoothe_nm)

        results["area_um^2"] = mask.area
        results["fp_mask"] = os.path.join(results["folder"], "mask.pkl")
        mask.save(results["fp_mask"])

        results["fp_fig_mask_density"] = os.path.join(
            results["folder"], f"mask_density_{rcode}.png"
        )
        mask.plot_mask(results["fp_fig_mask_density"], binary=False)
        results["fp_fig_mask_binary"] = os.path.join(
            results["folder"], f"mask_binary_{rcode}.png"
        )
        mask.plot_mask(results["fp_fig_mask_binary"], binary=True)

        # plot histogram of selected area
        fig, ax = plt.subplots()
        densities_to_plot = densities.ravel()
        densities_to_plot = densities_to_plot[densities_to_plot > 0]
        n, bins, patches = ax.hist(
            densities_to_plot * 1e6,
            bins=bins,
            color="b",
            label="densities in mask",
        )
        ylim = ax.get_ylim()
        ax.plot(
            [
                min_density * 1e6,
                min_density * 1e6,
                np.nan,
                max_density * 1e6,
                max_density * 1e6,
            ],
            [0, 0.9 * ylim[1], np.nan, 0, 0.9 * ylim[1]],
            color="r",
            label="selection boundaries",
        )
        mask_pixel_area = mask._upsample**2
        xlim = ax.get_xlim()
        x_density = np.linspace(xlim[0], xlim[1], 100) * 1e-6
        x_nlocs = x_density * mask_pixel_area
        mean_n_locs = np.median(densities_to_plot) * mask_pixel_area
        nbins = len(densities_to_plot)
        std_n_locs = np.sqrt(mean_n_locs)
        if mean_n_locs > 30:
            poisson_densities = norm.pdf(
                x_nlocs, scale=std_n_locs, loc=mean_n_locs
            )
        else:
            x_nlocs = x_nlocs.astype(np.int32)
            x_density = x_nlocs / mask_pixel_area
            poisson_densities = poisson.pmf(
                x_nlocs, mu=int(np.round(mean_n_locs))
            )
        poisson_densities = (
            poisson_densities / np.max(poisson_densities) * 0.9 * ylim[1]
        )
        ax.plot(
            x_density * 1e6,
            poisson_densities,
            color="k",
            label="poisson process with mask bins",
        )
        ax.set_xlabel("Density [µm^(-2)]")
        ax.set_ylabel("#bins")
        ax.set_title("Density Histogram of sleected area")
        ax.legend()
        results["fp_density_hist_after"] = os.path.join(
            results["folder"], f"density_hist_after_{rcode}.png"
        )
        fig.savefig(results["fp_density_hist_after"])

        if parameters.get("apply_to_locs"):
            results["fp_scene_locs_before"] = os.path.join(
                results["folder"], f"locs_before_{rcode}.png"
            )
            render.plot_scene(
                self.channel_locs,
                pixelsize,
                pixelsize,
                fp=results["fp_scene_locs_before"],
            )
            self.channel_locs = [
                mask.apply_to_locs(locs) for locs in self.channel_locs
            ]
            results["fp_scene_locs_after"] = os.path.join(
                results["folder"], f"locs_after_{rcode}.png"
            )
            render.plot_scene(
                self.channel_locs,
                pixelsize,
                pixelsize,
                fp=results["fp_scene_locs_after"],
            )

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def dbscan_molint(self, i, parameters, results):
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
        # from picasso_workflow.dbscan_molint import dbscan
        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)

        pixelsize = self.pixelsize
        epsilon_nm = parameters["epsilon_nm"]
        df_mask = pd.read_hdf(parameters["fp_merge_mask"], key="locs")
        fp_out_base = os.path.join(results["folder"], "dbscan.hdf5")
        filepaths = picasso_outpost._do_dbscan_molint(
            results["folder"],
            fp_out_base,
            df_mask,
            self.channel_info[0],
            pixelsize,
            epsilon_nm,
            parameters["minpts"],
            parameters["sigma_linker"],
            parameters["thresh_type"],
            parameters["cell_name"],
            channel_map,
        )
        for k, v in filepaths.items():
            results[k] = v

        return parameters, results

    def _load_other_workflow_data(
        self,
        fp_workflow,
        report_name,
        search_keys=None,
        search_parameter_keys=None,
    ):
        """Load result data from a different workflow
        Args:
            fp_workflow : str
                the root folder of the other workflow
            report_name : str
                the report name of the other workflow. The
                workflow result data will be in
                fp_workflow/report_name_[postfix]
            search_keys : tuple of
                1st : str
                    the module keys (e.g. '04_manual')
                2nd : str
                    the result entries (e.g. 'filepath')
                set this None or search_parameter_keys None
            search_parameter_keys: tuple of
                1st : str
                    the module keys (e.g. '04_manual')
                2nd : str
                    the result entries (e.g. 'filepath')
        Returns:
            loaded_data : dict
                keys : tuple of (str, str)
                    tuple of search key & value
                values : the corresponding loaded data
            channel_tags : list of str
                the channel tags
        """
        from picasso_workflow.workflow import WorkflowRunner

        loaded_data = {}

        # find analysis folder
        postfix = WorkflowRunner._check_previous_runner(
            fp_workflow, report_name
        )
        # find aggregation WorkflowRunner config
        fp_wr_cfg = os.path.join(
            fp_workflow,
            report_name + "_" + postfix,
            report_name + "_aggregation_" + postfix,
            "WorkflowRunner.yaml",
        )
        with open(fp_wr_cfg, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        if search_keys is not None:
            # check for results of the modules
            for mod_key, mod_res in data["results"].items():
                for search_mod, search_res in search_keys:
                    if mod_key == search_mod:
                        res = mod_res[search_res]
                        loaded_data[(search_mod, search_res)] = res
        elif search_parameter_keys is not None:
            # check for parameters of the modules
            for i, (module_name, module_pars) in enumerate(
                data["workflow_modules"]
            ):
                for search_module, search_parname in search_parameter_keys:
                    search_i, search_name = search_module.split("_")
                    search_i = int(search_i)
                    if search_i == i and search_name == module_name:
                        parameter_val = module_pars.get(search_parname)
                        loaded_data[(search_module, search_parname)] = (
                            parameter_val
                        )

        # find AggregationWorkflowRunner config
        fp_wr_cfg = os.path.join(
            fp_workflow,
            report_name + "_" + postfix,
            "AggregationWorkflowRunner.yaml",
        )
        with open(fp_wr_cfg, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        channel_tags = data["aggregation_workflow"][
            "single_dataset_tileparameters"
        ]["#tags"]

        return loaded_data, channel_tags

    #    @profile_resource_usage
    @module_decorator
    def CSR_sim_in_mask(self, i, parameters, results):
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
        from picasso_workflow.dbscan_molint import mask

        # from picasso_workflow.dbscan_molint import dbscan

        pixelsize = self.pixelsize
        epsilon_nm = parameters["epsilon_nm"]

        # get map
        with open(parameters["fp_channel_map"], "r") as f:
            channel_map = yaml.safe_load(f)
        with open(parameters["fp_mask_dict"], "rb") as f:
            mask_dict = pickle.load(f)
        info = mask_dict["info"]
        # filename_base = mask_dict['filename']

        all_filepaths = []
        for s in range(1, parameters["N_repeats"] + 1):
            # print()
            # print('repeat', s)
            # CSR simulation in mask:
            #     first: for each channel
            #     second: create multi file

            filename = os.path.join(
                results["folder"], f"CSR_in_mask_rep_{s}.hdf5"
            )
            df_CSR_mask, info_CSR = mask.CSR_sim_in_mask_multi_channel(
                channel_map,
                mask_dict,
                pixelsize,
                results["folder"],
                filename,
                info,
                plot_figures=True,
            )
            fp_out_base = os.path.join(results["folder"], f"dbscan_{s}.hdf5")
            filepaths = picasso_outpost._do_dbscan_molint(
                results["folder"],
                fp_out_base,
                df_CSR_mask,
                info,
                pixelsize,
                epsilon_nm,
                parameters["minpts"],
                parameters["sigma_linker"],
                parameters["thresh_type"],
                parameters["cell_name"],
                channel_map,
                it=s,
            )
            all_filepaths.append(filepaths)
        # re-organize: save the list of filepath dicts as
        # different dict values of lists of strings
        for k in all_filepaths[0].keys():
            results[k] = [fp[k] for fp in all_filepaths]

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def plot_densities(self, i, parameters, results):
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
        # get density and channel tags
        fp_density = []  # workflow, multiple CSR sims are done
        fp_maskdict = []
        channel_tags = None
        protint_key = parameters["swkfl_protint_key"]
        crmask_key = parameters["swkfl_create_mask_key"]
        search_dict = {
            (protint_key, "fp_density"): fp_density,
            (crmask_key, "fp_mask_dict"): fp_maskdict,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # load densities from nneighbor analysis
        all_densities_rdf = {k: [] for k in channel_tags}
        for fp in fp_density:
            with open(fp, "r") as f:
                d = yaml.safe_load(f)
                for k, v in d.items():
                    all_densities_rdf[k].append(v)

        # load mask parameters
        all_densities_mask = {k: [] for k in channel_tags}
        all_areas_mask = []
        for fp in fp_maskdict:
            with open(fp, "rb") as f:
                mask_dict = pickle.load(f)
            all_areas_mask.append(mask_dict["area"])
            for tgt in channel_tags:
                all_densities_mask[tgt].append(
                    mask_dict["density_exp_" + tgt + " (/um^2)"]
                )

        fig, ax = plt.subplots(nrows=2, sharex=True)
        data = [all_densities_rdf[k] for k in channel_tags]
        ax[0].violinplot(data, showmedians=True)
        util.stripplot(
            data,
            np.arange(1, 1 + len(channel_tags)),
            0.3,
            ax[0],
            "k",
            alpha=0.5,
        )
        ax[0].set_ylabel("RDF density")
        data = [all_densities_mask[k] for k in channel_tags]
        ax[1].violinplot(data, showmedians=True)
        util.stripplot(
            data,
            np.arange(1, 1 + len(channel_tags)),
            0.3,
            ax[1],
            "k",
            alpha=0.5,
        )
        ax[1].set_ylabel("density from mask")
        ax[1].set_xticks(np.arange(1, 1 + len(channel_tags)))
        ax[1].set_xticklabels(channel_tags, rotation=90)
        fp_fig_density = os.path.join(results["folder"], "density.png")
        fig.savefig(fp_fig_density)
        results["fp_fig_density"] = fp_fig_density

        # save data into results folder
        fp_density_rdf = os.path.join(results["folder"], "density_rdf.pkl")
        with open(fp_density_rdf, "wb") as f:
            pickle.dump(all_densities_rdf, f)
        results["fp_density_rdf"] = fp_density_rdf
        fp_density_mask = os.path.join(results["folder"], "density_mask.pkl")
        with open(fp_density_mask, "wb") as f:
            pickle.dump(all_densities_mask, f)
        results["fp_density_mask"] = fp_density_mask

        fig, ax = plt.subplots()
        ax.violinplot(all_areas_mask, showmedians=True)
        util.stripplot([all_areas_mask], [1], 0.3, ax, "k", alpha=0.5)
        ax.set_ylabel("area")
        ylim = ax.get_ylim()
        ax.set_ylim([0, 1.3 * ylim[1]])
        ax.set_xticklabels([])
        fp_fig_area = os.path.join(results["folder"], "area.png")
        fig.savefig(fp_fig_area)
        results["fp_fig_area"] = fp_fig_area

        fp_area_mask = os.path.join(results["folder"], "area_mask.pkl")
        with open(fp_area_mask, "wb") as f:
            pickle.dump(all_areas_mask, f)
        results["fp_area_mask"] = fp_area_mask

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def find_cluster_motifs(self, i, parameters, results):
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
        channel_tags = None
        fp_exp_bc = []  # will be a list of strings (1 for each cell)
        fp_exp_bcagg = []  # same as above
        fp_exp_bcmap = []  # same as above
        fp_csr_bc = []  # will be list of list of strings as in each
        fp_csr_bcagg = []  # same as above
        fp_csr_bcmap = []  # same as above
        fp_cluster_info_exp = []
        fp_cluster_info_csr = []
        exp_key = parameters["swkfl_dbscan_molint_key"]
        csr_key = parameters["swkfl_CSR_sim_in_mask_key"]
        search_dict = {
            (
                exp_key,
                "fp_barcode",
            ): fp_exp_bc,
            (
                exp_key,
                "fp_barcode_agg",
            ): fp_exp_bcagg,
            (
                exp_key,
                "fp_barcode_map",
            ): fp_exp_bcmap,
            (
                exp_key,
                "fp_cluster_info",
            ): fp_cluster_info_exp,
            (
                csr_key,
                "fp_barcode",
            ): fp_csr_bc,
            (
                csr_key,
                "fp_barcode_agg",
            ): fp_csr_bcagg,
            (
                csr_key,
                "fp_barcode_map",
            ): fp_csr_bcmap,
            (
                csr_key,
                "fp_cluster_info",
            ): fp_cluster_info_csr,
        }
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # load all data
        barcode_map = None
        barcodes_exp = None
        barcodes_exp_agg = None
        barcodes_csr = None
        barcodes_csr_agg = None

        for fp in fp_exp_bcmap:
            df = pd.read_excel(fp, index_col=0, header=0)
            if barcode_map is None:
                barcode_map = df
            else:
                if not barcode_map.equals(df):
                    # raise KeyError(
                    #     "The different workflows used "
                    #     + "different barcode maps"
                    # )
                    logger.error(
                        "The different workflows used "
                        + "different barcode maps"
                    )
                    print(
                        "ERROR: The different workflows used "
                        + "different barcode maps"
                    )
        for fplist in fp_csr_bcmap:
            for fp in fplist:
                df = pd.read_excel(fp, index_col=0, header=0)
                if barcode_map is None:
                    barcode_map = df
                else:
                    if not barcode_map.equals(df):
                        # raise KeyError(
                        #     "The different workflows used "
                        #     + "different barcode maps"
                        # )
                        logger.error(
                            "The different workflows used "
                            + "different barcode maps"
                        )
                        print(
                            "ERROR: The different workflows used "
                            + "different barcode maps"
                        )

        for fp, name in zip(fp_exp_bc, parameters["report_names"]):
            df = pd.read_excel(fp, index_col=0, header=0)
            df["name"] = name
            df["iter"] = 0
            if barcodes_exp is None:
                barcodes_exp = df
            else:
                barcodes_exp = pd.concat([barcodes_exp, df], ignore_index=True)
        for fp, name in zip(fp_exp_bcagg, parameters["report_names"]):
            df = pd.read_excel(fp, index_col=0, header=[0, 1])
            df["name"] = name
            df["metric"] = df.index
            df = df.reset_index()
            if barcodes_exp_agg is None:
                barcodes_exp_agg = df
            else:
                barcodes_exp_agg = pd.concat(
                    [barcodes_exp_agg, df], ignore_index=True
                )
        for fplist, name in zip(fp_csr_bc, parameters["report_names"]):
            for i, fp in enumerate(fplist):
                df = pd.read_excel(fp, index_col=0, header=[0])
                df["name"] = name
                df["iter"] = i
                if barcodes_csr is None:
                    barcodes_csr = df
                else:
                    barcodes_csr = pd.concat(
                        [barcodes_csr, df], ignore_index=True
                    )
        for fplist, name in zip(fp_csr_bcagg, parameters["report_names"]):
            for i, fp in enumerate(fplist):
                df = pd.read_excel(fp, index_col=0, header=[0, 1])
                df["name"] = name
                df["metric"] = df.index
                if barcodes_csr_agg is None:
                    barcodes_csr_agg = df
                else:
                    barcodes_csr_agg = pd.concat(
                        [barcodes_csr_agg, df], ignore_index=True
                    )
        cluster_info_exp = {}
        for fp in fp_cluster_info_exp:
            with open(fp, "r") as f:
                cluster_info = yaml.safe_load(f)
            for k, v in cluster_info.items():
                if k in cluster_info_exp.keys():
                    cluster_info_exp[k].append(v)
                else:
                    cluster_info_exp[k] = [v]

        cluster_info_csr = {}
        # iterate through cells
        for fplist in fp_cluster_info_csr:
            # fplist is list over multiple csr simulations
            # prepare dict wiht lists of iteration values
            cluster_info_lists = {}
            for k in cluster_info_exp.keys():
                cluster_info_lists[k] = []
            for fp in fplist:
                with open(fp, "r") as f:
                    cluster_info = yaml.safe_load(f)
                for k, v in cluster_info.items():
                    cluster_info_lists[k].append(v)
            # for each cell, add the mean over all simulations
            for k, v in cluster_info_lists.items():
                if k in cluster_info_csr.keys():
                    cluster_info_csr[k].append(np.mean(v))
                else:
                    cluster_info_csr[k] = [np.mean(v)]

        targets = channel_tags
        # target_colors = parameters["channel_colors"]
        origin_colors = ["blue", "gray"]

        # plot degree of clustering
        fp_figs = picasso_outpost.degree_of_clustering(
            cluster_info_exp,
            cluster_info_csr,
            origin_colors,
            results["folder"],
        )
        results["fp_fig_degreeofclustering"] = fp_figs[0]
        results["fp_fig_fracdegreeofclustering"] = fp_figs[1]

        # analyse the barcodes
        barcodes_exp["origin"] = "exp"
        barcodes_csr["origin"] = "csr"
        bc_all = pd.concat([barcodes_exp, barcodes_csr], ignore_index=True)

        results["fp_barcodes"] = os.path.join(
            results["folder"], "barcodes.hdf5"
        )
        bc_all.to_hdf(results["fp_barcodes"], key="barcodes")

        # number of barcodes
        barcode_numbers = pd.pivot_table(
            bc_all[["barcode", "origin", "name", "iter"]],
            index="barcode",
            columns=["origin", "name", "iter"],
            aggfunc=len,
            fill_value=0,
        )
        barcode_numbers.to_excel(
            os.path.join(results["folder"], "barcodes_numbers.xlsx")
        )
        # average over 'iter'
        barcode_numbers = (
            barcode_numbers.T.groupby(level=["origin", "name"]).mean().T
        )
        barcode_numbers.to_excel(
            os.path.join(results["folder"], "barcodes_numbers_iteravg.xlsx")
        )
        results["fp_fig_nbarcodesbox"] = os.path.join(
            results["folder"], "n_barcodes_boxplot.png"
        )
        (
            significant_barcodes,
            p_values,
        ) = picasso_outpost._plot_and_compare_barcodes(
            barcode_numbers,
            origin_colors,
            targets,
            parameters["ttest_pvalue_max"],
            parameters["population_threshold"],
            parameters["cellfraction_threshold"],
            results["fp_fig_nbarcodesbox"],
            title="Barcode Occurrence",
            ylabel="# barcodes found",
        )
        # results["significant_barcodes"] = significant_barcodes
        # results["ttest_pvalues"] = p_values

        # area of barcodes
        barcode_areas = pd.pivot_table(
            bc_all[["barcode", "origin", "name", "iter", "area (nm^2)"]],
            index="barcode",
            columns=["origin", "name", "iter"],
            values="area (nm^2)",
            aggfunc="sum",
            fill_value=0,
        )
        barcode_areas.to_excel(
            os.path.join(results["folder"], "barcodes_areas.xlsx")
        )
        # average over 'iter'
        barcode_areas = (
            barcode_areas.T.groupby(level=["origin", "name"]).mean().T
        )
        barcode_areas.to_excel(
            os.path.join(results["folder"], "barcodes_areas_iteravg.xlsx")
        )
        results["fp_fig_abarcodesbox"] = os.path.join(
            results["folder"], "a_barcodes_boxplot.png"
        )
        (
            significant_barcodes,
            p_values,
        ) = picasso_outpost._plot_and_compare_barcodes(
            barcode_areas,
            origin_colors,
            targets,
            parameters["ttest_pvalue_max"],
            parameters["population_threshold"],
            parameters["cellfraction_threshold"],
            results["fp_fig_abarcodesbox"],
            title="Barcode Areas",
            ylabel="total cluster area (nm^2)",
        )
        results["significant_barcodes"] = significant_barcodes
        results["ttest_pvalues"] = p_values

        # plot number of targets for each significant barcode
        fp_fig_ntargets = []
        for bc in significant_barcodes:
            df = bc_all.loc[bc_all["barcode"] == bc, :]
            fp_fig = os.path.join(
                results["folder"], f"ntargets_barcode_{bc[2:]}.png"
            )
            picasso_outpost._plot_and_compare_ntargets_in_barcodes(
                df, bc, origin_colors, targets, fp_fig
            )
            fp_fig_ntargets.append(fp_fig)
        results["fp_fig_ntargets"] = fp_fig_ntargets

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def interaction_graph(self, i, parameters, results):
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
        # get density and channel tags
        fp_density = []  # workflow, multiple CSR sims are done
        channel_tags = None
        protint_key = parameters["swkfl_protint_key"]
        search_dict = {(protint_key, "fp_density"): fp_density}
        for folder, name in zip(
            parameters["fp_workflows"], parameters["report_names"]
        ):
            loaded_data, wf_channel_tags = self._load_other_workflow_data(
                folder, name, search_dict.keys()
            )
            for key, res in loaded_data.items():
                search_dict[key].append(res)

            # make sure all channel tags (e.g. protein names)
            # are the same across workflows to be merged
            if channel_tags is None:
                channel_tags = wf_channel_tags
            else:
                if channel_tags != wf_channel_tags:
                    raise KeyError(
                        "Loaded datasets have different channel tags!"
                    )

        # load densities and average
        all_densities = {k: [] for k in channel_tags}
        for fp in fp_density:
            with open(fp, "r") as f:
                d = yaml.safe_load(f)
                for k, v in d.items():
                    all_densities[k].append(v)

        mean_densities = {k: np.mean(v) for k, v in all_densities.items()}
        densities = np.array([mean_densities[tgt] for tgt in channel_tags])

        targets = channel_tags
        meanvals = np.loadtxt(parameters["fp_ripleys_meanvals"])
        fig, ax = picasso_outpost._plot_interaction_graph(
            densities * parameters["node_factor"],
            meanvals * parameters["edge_factor"],
            parameters["channel_colors"],
            targets,
        )
        results["fp_fig"] = os.path.join(
            results["folder"], f"interaction_graph_mod{i}.png"
        )
        fig.set_size_inches((7, 7))
        fig.savefig(results["fp_fig"])
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def find_gold(self, i, parameters, results):
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
        logger.debug(f"# locs: {len(self.locs)}")
        # search for xy positions that look like gold ('pick similar')
        kwargs = {}
        for prop in ["diameter", "std_range", "mean_rmsd"]:
            if val := parameters.get(prop):
                kwargs[prop] = val
        gold_picks = picasso_outpost.pick_gold(self.locs, self.info, **kwargs)

        results["n_gold"] = len(gold_picks)
        logger.debug(f"# gold particles found: {len(gold_picks)}")

        if len(gold_picks) <= 2:
            logger.debug(
                """
                Not engouh gold particles found. Skipping further undrifting
                steps for this file" continue without gold undrifting"""
            )
            gold_locs = pd.DataFrame(columns=self.locs.columns)
            nongold_locs = self.locs
        else:
            # function needs to return the locs in a r radius around the gold
            # coordinates
            gold_locs, nongold_locs = picasso_outpost.picked_locs(
                self.locs,
                self.info,
                gold_picks,
                pick_diameter=parameters.get("diameter", 2.5),
                return_nonpicked=True,
            )

        # save gold locs
        fp_gold = os.path.join(results["folder"], "gold.hdf5")
        # fp_gold = os.path.join(results["folder"], "gold.pkl")
        results["fp_gold"] = fp_gold
        gold_info = self.info
        gold_info.append(
            {
                "Generated by": "picasso-workflow.analyse.find_gold",
                "data": "gold",
            }
        )
        io.save_locs(fp_gold, gold_locs, gold_info)
        # with open(fp_gold, "wb") as f:
        #     pickle.dump(gold_locs, f)

        fp_nogold = os.path.join(results["folder"], "nogold.hdf5")
        results["fp_nogold"] = fp_nogold
        nogold_info = self.info
        nogold_info.append(
            {
                "Generated by": "picasso-workflow.analyse.find_gold",
                "data": "no gold",
            }
        )
        io.save_locs(fp_nogold, nongold_locs, nogold_info)

        if parameters.get("remove_gold"):
            self.locs = nongold_locs
            self.info = nogold_info
            logger.debug("removing gold from attribute locs.")

        # logger.debug(f"# locs kept as attribute: {len(self.locs)}")

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def find_similar(self, i, parameters, results):
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
        logger.debug(f"# locs: {len(self.locs)}")
        # search for xy positions that look like gold ('pick similar')
        diameter = parameters["diameter"]
        kwargs = {
            "diameter": diameter,
            "min_n_locs_per_frame": parameters["min_n_locs_per_frame"],
            "max_n_locs_per_frame": parameters["max_n_locs_per_frame"],
            "min_rmsd": parameters["min_rmsd"],
            "max_rmsd": parameters["max_rmsd"],
        }
        # print(self.locs.dtype)
        (picks, nlocs, rmsds, labels) = picasso_outpost.pick_similar(
            self.locs, self.info, **kwargs
        )
        # print(self.locs.dtype)

        results["n_picks"] = len(picks)

        # show clustering results
        fig, ax = plt.subplots()
        # plot non-selected picks
        ax.scatter(
            nlocs[labels == -1],
            rmsds[labels == -1],
            color="k",
            alpha=0.1,
            label="non-selected",
        )
        ax.scatter(
            nlocs[labels == 0],
            rmsds[labels == 0],
            color="r",
            alpha=0.3,
            label="selected",
        )
        ax.set_xlabel("# localizations in pick")
        ax.set_ylabel("root mean square distance in pick")
        ax.set_title("Selection of picks")
        ax.legend()
        rcode = generate_random_code(6)
        results["fp_phasespace"] = os.path.join(
            results["folder"], f"rawcluster-{rcode}.png"
        )
        fig.set_size_inches((9, 9))
        fig.savefig(results["fp_phasespace"])

        fig, ax = plt.subplots()
        extent = [
            np.quantile(nlocs, 0.02),
            np.quantile(nlocs, 0.98),
            np.quantile(rmsds, 0.02),
            np.quantile(rmsds, 0.98),
        ]
        gridsize = int(extent[1] - extent[0]) + 1
        if gridsize > 100:
            gridsize = 50
        ax.hexbin(nlocs, rmsds, extent=extent, gridsize=gridsize)
        ax.set_xlabel("# localizations in pick")
        ax.set_ylabel("root mean square distance in pick")
        ax.set_title("Phase Space")
        results["fp_phasespace_hexbin"] = os.path.join(
            results["folder"], f"phsp-hexbin-{rcode}.png"
        )
        fig.set_size_inches((9, 9))
        fig.savefig(results["fp_phasespace_hexbin"])

        # all xy coords found for the picks
        if len(picks) > 2:
            picked_locs = picasso_outpost.picked_locs(
                self.locs,
                self.info,
                picks,
                pick_diameter=diameter,
                return_nonpicked=False,
            )
            fullfov_pixelsize = 1000
            results["fp_picked_fullfov"] = os.path.join(
                results["folder"], f"picked_locs_fullfov_{rcode}.png"
            )
            render.plot_scene(
                picked_locs,
                fullfov_pixelsize,
                self.pixelsize,
                fp=results["fp_picked_fullfov"],
            )
        else:
            logger.debug(
                """
                Not many picks found in specified phase space."""
            )
            try:
                # dt_orig = self.locs.dtype
                # if not isinstance(dt_orig, list) and len(dt_orig) == 2:
                #     dt_orig = dt_orig[1]
                # dtypes = self.locs.dtype + [("group", "<i4")]
                column_names = [
                    "frame",
                    "x",
                    "y",
                    "photons",
                    "sx",
                    "sy",
                    "bg",
                    "lpx",
                    "lpy",
                    "ellipticity",
                    "net_gradient",
                    "group",
                ]
            except Exception as e:
                raise e
            picked_locs = pd.DataFrame(columns=column_names)
        results["n_picked_locs"] = len(picked_locs)
        results["n_locs"] = len(self.locs)

        # save picked locs
        fp_locs = os.path.join(results["folder"], "picked_locs.hdf5")
        cluster_info = self.info
        cluster_info.append(
            {
                "Generated by": "picasso-workflow.analyse.find_similar",
                "data": "similar picks",
            }
        )
        io.save_locs(fp_locs, picked_locs, cluster_info)
        results["fp_picked_locs"] = fp_locs

        # plot representative structures
        n_plot = parameters.get("n_plot_structures")
        fp_renderings = []
        if n_plot is not None:
            pixelsize = self.pixelsize
            pixelsize_display = parameters.get("display_pixelsize", 1)
            for idx, pick_i in enumerate(
                np.random.choice(len(picks), size=n_plot, replace=False)
            ):
                x_min = picks[pick_i][0] - diameter / 2
                y_min = picks[pick_i][1] - diameter / 2
                render_kwargs = {
                    "oversampling": pixelsize / pixelsize_display,
                    "viewport": [
                        (y_min, x_min),
                        (
                            picks[pick_i][1] + diameter / 2,
                            picks[pick_i][0] + diameter / 2,
                        ),
                    ],
                }
                fp_renderings.append(
                    os.path.join(
                        results["folder"],
                        f"render_structure_{idx}_{pick_i}-{rcode}.png",
                    )
                )
                render.plot_scene(
                    picked_locs,
                    pixelsize_display,
                    pixelsize,
                    fp=fp_renderings[-1],
                    render_kwargs=render_kwargs,
                    # x_offset=x_min * pixelsize,
                    # y_offset=y_min * pixelsize,
                    title=f"pick {pick_i}",
                )

        results["fp_cluster_locs"] = fp_locs
        results["fp_renderings"] = [fp_renderings]

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def find_structures(self, i, parameters, results):
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
        logger.debug(f"# locs: {len(self.locs)}")
        # search for xy positions that look like gold ('pick similar')
        diameter = parameters["diameter"]
        kwargs = {"diameter": diameter}
        for prop in ["min_n_locs_per_frame", "xi", "min_cluster_size"]:
            if val := parameters.get(prop):
                kwargs[prop] = val
        (
            cluster_picks,
            nlocs,
            rmsds,
            labels,
            newlabels,
        ) = picasso_outpost.find_structures(self.locs, self.info, **kwargs)

        results["n_clusters"] = len(cluster_picks)
        results["n_picks"] = [len(picks) for picks in cluster_picks]

        # show clustering results
        fig, ax = plt.subplots()
        colors = ["k", "g", "r", "b", "y", "c"]
        for lbl, color in zip(range(-1, 1 + np.max(labels)), colors):
            if lbl < 0:
                alpha = 0.1
            else:
                alpha = 0.3
            ax.scatter(
                nlocs[labels == lbl],
                rmsds[labels == lbl],
                color=color,
                alpha=alpha,
                label=f"cluster {lbl:02d}",
            )
        ax.set_xlabel("# localizations in pick")
        ax.set_ylabel("root mean square distance in pick")
        ax.set_title("Raw Clustering of picks")
        ax.legend()
        rcode = generate_random_code(6)
        results["fp_rawcluster"] = os.path.join(
            results["folder"], f"rawcluster-{rcode}.png"
        )
        fig.set_size_inches((9, 9))
        fig.savefig(results["fp_rawcluster"])

        fig, ax = plt.subplots()
        for lbl, color in zip(range(-1, 1 + np.max(newlabels)), colors):
            if lbl < 0:
                alpha = 0.1
            else:
                alpha = 0.3
            ax.scatter(
                nlocs[newlabels == lbl],
                rmsds[newlabels == lbl],
                color=color,
                alpha=alpha,
                label=f"cluster {lbl:02d}",
            )
        ax.set_xlabel("# localizations in pick")
        ax.set_ylabel("root mean square distance in pick")
        ax.set_title("Pick-similar Clustering of picks")
        ax.legend()
        results["fp_picksimcluster"] = os.path.join(
            results["folder"], f"picksimcluster-{rcode}.png"
        )
        fig.set_size_inches((9, 9))
        fig.savefig(results["fp_picksimcluster"])

        # save the pick locs
        fp_locs = []
        fp_renderings = []
        for cluster_id, picks in enumerate(cluster_picks):
            # all xy coords found for this cluster of structure
            if len(picks) > 2:
                cluster_locs = picasso_outpost.picked_locs(
                    self.locs,
                    self.info,
                    picks,
                    pick_diameter=diameter,
                    return_nonpicked=False,
                )

            # save gold locs
            fp = os.path.join(
                results["folder"], f"structure_cluster_{cluster_id:02d}.hdf5"
            )
            # fp_gold = os.path.join(results["folder"], "gold.pkl")
            fp_locs.append(fp)
            cluster_info = self.info
            cluster_info.append(
                {
                    "Generated by": "picasso-workflow.analyse.find_structures",
                    "data": f"strucutre {cluster_id:02d}",
                }
            )
            io.save_locs(fp, cluster_locs, cluster_info)

            # plot representative structures
            n_plot = parameters.get("n_plot_structures")
            if n_plot is None:
                continue
            fp_renderings_cluster = []
            pixelsize = self.pixelsize
            pixelsize_display = parameters.get("display_pixelsize", 1)
            for idx, pick_i in enumerate(
                np.random.choice(len(picks), size=n_plot, replace=False)
            ):
                x_min = picks[pick_i][0] - diameter / 2
                y_min = picks[pick_i][1] - diameter / 2
                render_kwargs = {
                    "oversampling": pixelsize / pixelsize_display,
                    "viewport": [
                        (y_min, x_min),
                        (
                            picks[pick_i][1] + diameter / 2,
                            picks[pick_i][0] + diameter / 2,
                        ),
                    ],
                }
                fp_renderings_cluster.append(
                    os.path.join(
                        results["folder"],
                        f"render_structure_{idx}_{pick_i}-{rcode}.png",
                    )
                )
                render.plot_scene(
                    cluster_locs,
                    pixelsize_display,
                    pixelsize,
                    fp=fp_renderings_cluster[-1],
                    render_kwargs=render_kwargs,
                    # x_offset=x_min * pixelsize,
                    # y_offset=y_min * pixelsize,
                    title=f"pick {pick_i}",
                )
            fp_renderings.append(fp_renderings_cluster)

        results["fp_cluster_locs"] = fp_locs
        results["fp_renderings"] = fp_renderings

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def undrift_from_picked(self, i, parameters, results):
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
        pixelsize = self.pixelsize
        picked_locs, info = io.load_locs(parameters["fp_picked_locs"])
        # with open(parameters["fp_picked_locs"], "rb") as f:
        #     result = pickle.load(f)

        if not isinstance(picked_locs, list):
            # picked locs are saved as one recarray, with the 'group' the pick
            groups = np.unique(picked_locs["group"])
            picked_locs = [
                picked_locs[picked_locs["group"] == group] for group in groups
            ]
        # print(result)
        # picked_locs, picked_info = io.load_locs(parameters["fp_picked_locs"])
        self.locs, self.info, drift = picasso_outpost._undrift_from_picked(
            self.locs, self.info, picked_locs
        )
        drift = pd.DataFrame(drift)

        dims = ["x", "y"]
        if "z" in picked_locs[0].columns:
            dims.append("z")
        fp_fig = os.path.join(results["folder"], "undrift_from_picked.png")
        results["fp_fig"] = fp_fig
        self._plot_drift(fp_fig, dims, pixelsize, method="picked", drift=drift)
        # fig, ax = plt.subplots(nrows=1)
        # ax.plot(drift[0], label="x drift")
        # ax.plot(drift[1], label="y drift")
        # ax.set_title("undrift from picked")
        # ax.set_ylabel("drift")
        # ax.set_xlabel("frame")
        # fig.savefig(fp_fig)
        # plt.close(fig)

        # fp_locs = os.path.join(results["folder"], "locs.hdf5")
        # results["fp_locs"] = fp_locs
        # self._save_locs(fp_locs)
        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def filter_locs(self, i, parameters, results):
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
        all_field = parameters["field"]
        all_xmin = parameters.get("minval")
        all_xmax = parameters.get("maxval")
        if isinstance(all_field, str):
            all_field = [all_field]
            all_xmin = [float(all_xmin)]
            all_xmax = [float(all_xmax)]
        else:
            if all_xmin is None:
                all_xmin = [None] * len(all_field)
            if all_xmax is None:
                all_xmax = [None] * len(all_field)
        if parameters.get("mode") == "zscore":
            # Turn from zscores into absolute values
            for i, field in enumerate(all_field):
                mean = np.mean(self.locs[field])
                std = np.std(self.locs[field])
                all_xmin = [
                    None if mi is None else mean + mi * std for mi in all_xmin
                ]
                all_xmax = [
                    None if ma is None else mean + ma * std for ma in all_xmax
                ]
        elif parameters.get("mode") == "quantile":
            # Turn from quantiles into absolute values
            for i, field in enumerate(all_field):
                all_xmin = [
                    None if mi is None else np.quantile(self.locs[field], mi)
                    for mi in all_xmin
                ]
                all_xmax = [
                    None if mi is None else np.quantile(self.locs[field], mi)
                    for mi in all_xmin
                ]

        results["nlocs_before"] = len(self.locs)
        # plot heatmaps before filtering
        fig, ax = self.plot_heatmaps(all_field)
        results["fp_fig_before"] = os.path.join(
            results["folder"], f"hist_before_{i:02d}.png"
        )
        fig.savefig(results["fp_fig_before"])

        # filter
        for field, xmin, xmax in zip(all_field, all_xmin, all_xmax):
            # self.locs = self.locs[
            #     (self.locs[field] >= xmin) & (self.locs[field] <= xmax)
            # ]
            if xmin is not None:
                self.locs = self.locs[self.locs[field] >= xmin]
            if xmax is not None:
                self.locs = self.locs[self.locs[field] <= xmax]

        results["nlocs_after"] = len(self.locs)
        # plot heatmaps after filtering
        fig, ax = self.plot_heatmaps(all_field)
        results["fp_fig_after"] = os.path.join(
            results["folder"], f"hist_after_{i:02d}.png"
        )
        fig.savefig(results["fp_fig_after"])

        # fp_locs = os.path.join(results["folder"], "locs.hdf5")
        # results["fp_locs"] = fp_locs
        # self._save_locs(fp_locs)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
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
        results["nlocs_before"] = len(self.locs)

        nframes = self.frames
        fields = ["frame", "std_frame"]
        all_xmin = [
            parameters.get("meanframe_cutoff", 0.1) * nframes,
            parameters.get("stdframe_cutoff", 0.16) * nframes,
        ]
        all_xmax = [
            (1 - parameters.get("meanframe_cutoff", 0.1)) * nframes,
            None,
        ]
        results["fields_filtered"] = fields
        results["all_xmin"] = all_xmin
        results["all_xmax"] = all_xmax
        # plot heatmaps before filtering
        fig, ax = self.plot_heatmaps(fields)
        results["fp_fig_before"] = os.path.join(
            results["folder"], f"hist_before_{i:02d}.png"
        )
        fig.savefig(results["fp_fig_before"])

        # filter
        for field, xmin, xmax in zip(fields, all_xmin, all_xmax):
            # self.locs = self.locs[
            #     (self.locs[field] >= xmin) & (self.locs[field] <= xmax)
            # ]
            if xmin is not None:
                self.locs = self.locs[self.locs[field] >= xmin]
            if xmax is not None:
                self.locs = self.locs[self.locs[field] <= xmax]

        results["nlocs_after"] = len(self.locs)
        # plot heatmaps after filtering
        fig, ax = self.plot_heatmaps(fields)
        results["fp_fig_after"] = os.path.join(
            results["folder"], f"hist_after_{i:02d}.png"
        )
        fig.savefig(results["fp_fig_after"])

        # Load Locs
        if fp_locs := parameters.get("fp_locs"):
            groups_kept = self.locs["group"]
            locs, info = io.load_locs(fp_locs)
            mask = np.isin(locs["group"], groups_kept)
            locs = locs[mask]
            results["fp_locs"] = os.path.join(
                results["folder"], os.path.split(fp_locs)[1]
            )
            io.save_locs(results["fp_locs"], locs, info)

        return parameters, results

    def plot_heatmaps(self, fields):
        """Plot headmaps between all tuples of fields given

        Args:
            fields : list of str

        Returns:
            fig, ax: fig and ax of all tuples of fields
        """
        if len(fields) == 1:
            fig, ax = plt.subplots()
            picasso_outpost.plot_1dhist(self.locs, fields[0], fig, ax)
        else:
            fig, ax = plt.subplots(
                nrows=len(fields) - 1, ncols=len(fields) - 1, squeeze=False
            )
            for i, field_x in enumerate(fields[:-1]):
                for j, field_y in enumerate(fields[i + 1 :]):
                    picasso_outpost.plot_2dhist(
                        self.locs, field_x, field_y, fig, ax[i, j]
                    )
                # if i > 0:
                #     for j in range(len(fields) - i, len(fields) - 1):
                #         ax[i, j].axis("off")

        return fig, ax

    #    @profile_resource_usage
    @module_decorator
    def link_locs(self, i, parameters, results):
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
        self.locs = postprocess.link(
            self.locs, self.info, parameters["d_max"], parameters["tolerance"]
        )
        link_info = {
            "Maximum Distance": parameters["d_max"],
            "Maximum Transient Dark Time": parameters["tolerance"],
            "Generated by": "Picasso Link",
        }
        self.info.append(link_info)

        # fp_locs = os.path.join(results["folder"], "locs.hdf5")
        # results["fp_locs"] = fp_locs
        # self._save_locs(fp_locs)

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def pairwise_module_executor(self, i, parameters, results):
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
        fun_name = parameters["module_name"]
        sub_results = {}
        sub_params_general = parameters.get("module_kwargs", {})
        param_target1 = parameters["param_target1"]
        param_target2 = parameters["param_target2"]

        key_scalar = parameters.get("result_scalar")
        key_fpfigs = parameters.get("result_fpfig")
        n_channels = len(self.channel_tags)
        result_matrix = np.zeros([n_channels] * 2)
        fp_figs = [["" for c in range(n_channels)] for _ in range(n_channels)]
        if isinstance(key_fpfigs, list):
            fp_figs = [fp_figs.copy() for _ in len(key_fpfigs)]
        else:
            fp_figs = [fp_figs]
            key_fpfigs = [key_fpfigs]

        rollover = int(10 ** (np.ceil((len(self.channel_tags)) ** (0.1))))
        for i, tag1 in enumerate(self.channel_tags):
            for j, tag2 in enumerate(self.channel_tags):
                idx = int(rollover * i + j)
                suffix = f"{tag1}-{tag2}"
                sub_params = sub_params_general.copy()
                sub_params[param_target1] = tag1
                sub_params[param_target2] = tag2
                logger.debug(f"Working on {fun_name}: {suffix}")
                fun = getattr(self, fun_name)
                try:
                    sub_module_pars, sub_results[suffix] = fun(
                        idx, sub_params, results["folder"], suffix
                    )
                except AutoPicassoError as e:
                    logger.error(e)
                    raise e
                except Exception as e:
                    logger.error(e)
                    raise e

                # save the results
                logger.debug(f"results: {sub_results[suffix]}")
                logger.debug(f"key_fpfigs {key_fpfigs}")
                logger.debug(f"result matrix {result_matrix}")
                result_matrix[i, j] = sub_results[suffix].get(key_scalar)
                for k, key_fpfig in enumerate(key_fpfigs):
                    fp_figs[k][i][j] = sub_results[suffix].get(key_fpfig)

        results["sub_module_results"] = sub_results
        if key_scalar is None:
            results["result_matrix"] = None
        else:
            results["result_matrix"] = result_matrix
            results["fp_fig_matrix"] = self._plot_ripleys_integrals(
                result_matrix,
                results["folder"],
                self.channel_tags,
                metric=fun_name,
                controltype="",
                threshold=parameters.get("scalar_threshold"),
                std=None,
                suffix="",
                significance_threshold=parameters.get("scalar_minval"),
            )

        if key_fpfigs is None:
            results["fp_figs"] = None
        else:
            results["fp_figs"] = fp_figs

        return parameters, results

    #    @profile_resource_usage
    @module_decorator
    def random_val(self, i, parameters, results):
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
        results["random_val"] = np.random.rand()
        fig, ax = plt.subplots()
        x = np.arange(100)
        ax.plot(x, np.random.rand(len(x)))
        ax.set_xlabel(parameters["xlabel"])
        ax.set_ylabel(parameters["ylabel"])
        rcode = generate_random_code(6)
        results["fp_fig"] = os.path.join(
            results["folder"], f"myfig_{rcode}.png"
        )
        fig.savefig(results["fp_fig"])

        return parameters, results

    def create_le_structures(self, target, reference, pair_distance):
        # target monomer
        structures = self._create_spinna_structure(
            [target], [[1]], pair_distance
        )
        # reference monomer
        structures += self._create_spinna_structure(
            [reference], [[1]], pair_distance
        )
        # heterodimer
        struct = {
            "Molecular targets": [target, reference],
            "Structure title": f"{target}-{reference}-heterodimer",
            f"{target}_x": [-pair_distance / 2],
            f"{target}_y": [0],
            f"{target}_z": [0],
            f"{reference}_x": [pair_distance / 2],
            f"{reference}_y": [0],
            f"{reference}_z": [0],
        }
        structures.append(struct)
        return structures

    #    @profile_resource_usage
    @module_decorator
    def labeling_efficiency_analysis(self, i, parameters, results):
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
        if not parameters.get("nn_nth"):
            parameters["nn_nth"] = 2
        target = parameters["target_name"]
        reference = parameters["reference_name"]
        labeling_efficiency = {
            target: float(1.0),
            reference: float(1.0),
        }

        pair_distance = parameters["pair_distance"]

        channel_map = {tag: i for i, tag in enumerate(self.channel_tags)}

        logger.debug("Labeling Efficiency determination using SPINNA")

        # # homo-analysis (proportions of 1- or 2-mers of the same kind)
        # props = {}
        dimensionality = 2
        pixelsize = self.pixelsize
        width = max([locs["x"].max() for locs in self.channel_locs]) - min(
            [locs["x"].min() for locs in self.channel_locs]
        )
        height = max([locs["y"].max() for locs in self.channel_locs]) - min(
            [locs["y"].min() for locs in self.channel_locs]
        )
        try:
            depth = max([locs["z"].max() for locs in self.channel_locs]) - min(
                [locs["z"].min() for locs in self.channel_locs]
            )
        except (ValueError, KeyError):
            depth = None

        if isinstance(parameters["density"], list):
            density = {
                tag: parameters["density"][cid]
                for tag, cid in channel_map.items()
            }
        elif isinstance(parameters["density"], dict):
            density = parameters["density"]
        else:
            raise KeyError("density parameter must be list or dict.")
        results["fp_density"] = os.path.join(results["folder"], "density.yaml")
        with open(results["fp_density"], "w") as f:
            yaml.dump(density, f)

        # ground thruth density, adjusted by labeling efficiency
        # assume le=1 for target here, it is calculated in the end.
        density_gt = {reference: density[reference], target: density[target]}

        logger.debug(f"analysing labeling efficency of {target} using SPINNA.")
        # find index of A and B in self.channel_locs
        i_target = self.channel_tags.index(target)
        i_reference = self.channel_tags.index(reference)

        # locs, but as np.ndarray
        exp_data = {}
        for i, tgt in zip([i_target, i_reference], [target, reference]):
            locs = self.channel_locs[i]
            if "z" in locs.columns:
                exp_data[tgt] = np.stack(
                    (locs["x"] * pixelsize, locs["y"] * pixelsize, locs["z"])
                ).T
                # dim = 3
            else:
                exp_data[tgt] = np.stack(
                    (locs["x"] * pixelsize, locs["y"] * pixelsize)
                ).T
                # dim = 2

        compound_density = density_gt[target] / 1 + density_gt[reference] / 1
        # area = parameters["n_simulate"] / (compound_density / 1e6)
        # area = parameters["n_simulate"] / (compound_density)
        area = parameters["n_simulate"] / (compound_density * 1e6)
        n_sim_targets = {
            tag: int(
                parameters["n_simulate"] * density_gt[tag] / compound_density
            )
            for tag in [target, reference]
        }

        if isinstance(pair_distance, list):
            all_test_structures = []
            for test_distance in pair_distance:
                structures = self.create_le_structures(
                    target, reference, test_distance
                )
                (
                    structures,
                    targets,
                ) = picasso_outpost.load_structures_from_dict(structures)
                all_test_structures.append(structures)
                logger.debug(f"pair distance: {test_distance}")
                tgts = []
                for structure in structures:
                    for tgt in structure.targets:
                        if tgt not in tgts:
                            tgts.append(tgt)

                # number of molecular targets in each structure; each
                # row gives one target species and each column gives one
                # structure
                n_t = len(tgts)
                n_s = len(structures)
                logger.debug(f"{n_s} structures: {str(structures)}")
                logger.debug(f"{n_t} targets: {str(tgts)}")

            logger.debug(str(all_test_structures))
            label_unc = {
                k: v if isinstance(v, list) else [v]
                for k, v in parameters["labeling_uncertainty"].items()
            }
            (
                best_score,
                best_idx,
                label_unc,
                best_mixer,
                best_props,
            ) = spinna.compare_models(
                models=all_test_structures,
                exp_data=exp_data,
                granularity=parameters["granularity"],
                label_unc=parameters["labeling_uncertainty"],
                le=labeling_efficiency,
                width=width,
                height=height,
                depth=depth,
            )
            pair_distance = pair_distance[best_idx]
            structures = all_test_structures[best_idx]
            labeling_uncertainty = label_unc
            results["best_pair_distance"] = pair_distance
        else:
            structures = self.create_le_structures(
                target, reference, pair_distance
            )
            labeling_uncertainty = {
                k: v[0] if isinstance(v, list) else v
                for k, v in parameters["labeling_uncertainty"].items()
            }

            structures, targets = picasso_outpost.load_structures_from_dict(
                structures
            )

        N_structures = picasso_outpost.generate_N_structures(
            structures, n_sim_targets, parameters["granularity"]
        )

        # bin size: more than Nyquist subsampling
        expected_1stNN_peak = (
            2 / (2 * dimensionality * np.pi * (compound_density / 2))
        ) ** (1 / dimensionality)
        fit_NND_bin = pair_distance / 3
        # max dist: a few times the first NN distance peak
        fit_NND_maxdist = 4 * parameters["nn_nth"] * expected_1stNN_peak

        spinna_parameters = {
            "structures": structures,
            "label_unc": labeling_uncertainty,
            "le": labeling_efficiency,
            "mask_dict": None,
            "width": np.sqrt(area * 1e6),
            "height": np.sqrt(area * 1e6),
            "depth": None,
            "random_rot_mode": "2D",
            "exp_data": exp_data,
            "sim_repeats": parameters["sim_repeats"],
            "NND_bin": fit_NND_bin,
            "NND_maxdist": fit_NND_maxdist,
            "N_structures": N_structures,
            "save_filename": os.path.join(
                results["folder"], f"interaction-{target}-{reference}"
            ),
            "asynch": True,
            "targets": [target, reference],
            "apply_mask": False,
            "nn_plotted": parameters["nn_nth"],
            "result_dir": results["folder"],
            "n_simulated": n_sim_targets,
            "bootstrap": parameters.get("bootstrap"),
        }

        result, fp_fig = picasso_outpost.single_spinna_run(**spinna_parameters)
        plt.close("all")

        # rename figures with random code
        rcode = generate_random_code(6)
        fp_fig_out = []
        for fp in fp_fig:
            fparts = os.path.splitext(fp)
            fp_out = f"{fparts[0]}_{rcode}{fparts[1]}"
            try:
                os.rename(fp, fp_out)
            except FileNotFoundError:
                pass
            fp_fig_out.append(fp_out)
        results["fp_fig"] = fp_fig_out
        props = result["props"]  # given in percent
        prop_t = props[0]
        prop_r = props[1]
        prop_tr = props[2]
        std_t = result["props_std"][0]
        std_r = result["props_std"][1]
        std_tr = result["props_std"][2]
        results["spinna_props_std"] = result["props_std"]
        results["spinna_props_std"] = result["props_std"]

        # SPINNA outputs proportions in terms of #molecules
        le_target = prop_tr / (2 * prop_r + prop_tr)
        le_reference = prop_tr / (2 * prop_t + prop_tr)

        # error propagation for std
        def le_std(prop_sglo, prop_dbl, std_sglo, std_dbl):
            """Calculate the standard deviation of le,
            by error propagation: sum of derivatives
            with respect to both variables multiplied by their std
            """
            deriv_sglo = -2 * prop_dbl / ((2 * prop_sglo + prop_dbl) ** 2)
            deriv_dbl = (2 * prop_sglo + prop_dbl) ** (-1) - prop_dbl / (
                2 * prop_sglo + prop_dbl
            ) ** 2
            return np.abs(deriv_sglo * std_sglo) + np.abs(deriv_dbl * std_dbl)

        le_target_std = le_std(prop_r, prop_tr, std_r, std_tr)
        le_reference_std = le_std(prop_t, prop_tr, std_t, std_tr)

        results["labeling_efficiency"] = {
            parameters["target_name"]: le_target,
            parameters["reference_name"]: le_reference,
        }
        results["labeling_efficiency_std"] = {
            parameters["target_name"]: le_target_std,
            parameters["reference_name"]: le_reference_std,
        }
        # results for compatibility with pairwise analysis
        results["labeling_efficiency_target"] = le_target
        results["labeling_efficiency_reference"] = le_reference
        results["labeling_efficiency_std_target"] = le_target_std
        results["labeling_efficiency_std_reference"] = le_reference_std
        results["fp_fig_AA"] = fp_fig_out[0]
        results["fp_fig_AB"] = fp_fig_out[1]
        results["fp_fig_BA"] = fp_fig_out[2]
        results["fp_fig_BB"] = fp_fig_out[3]

        return parameters, results


class AutoPicassoError(Exception):
    pass


class ManualInputLackingError(AutoPicassoError):
    pass


class PicassoConfigError(AutoPicassoError):
    pass

#!/usr/bin/env python
"""GUI descriptor module for picasso-workflow.

Module Name: gui.py
Author: Heinrich Grabmayr
Initial Date: August 4, 2024
"""

from __future__ import annotations

from picasso_workflow import util, CONFIG
from loguru import logger
import subprocess
import os
import re
import sys
import yaml
import importlib.util

# import pkgutil
# import importlib
import traceback
import tempfile
import functools
import inspect
import textwrap
from picasso import lib
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QEvent

try:
    from picasso_workflow._version import __version__ as __GUIVERSION__
except ImportError:
    __GUIVERSION__ = "unknown"


def _read_text_safe(path):
    """Read a text file for logging. Returns a sentinel on any failure
    so a logging side-effect can never raise into the caller."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:  # pragma: no cover - defensive
        return f"(could not read {path!r}: {exc!r})"


class ModuleDescriptor(util.AbstractModuleCollection):
    """Module descriptor class that provides GUI-friendly parameter and result
    specifications.

    This class extracts parameter and result information from the standardized
    docstrings to provide structured metadata that can be used by downstream
    GUI applications to create appropriate input elements (text fields,
    dropdowns, number fields, etc.).

    Each method returns a dictionary with 'parameters' and 'results' keys,
    where the values contain GUI-friendly specifications including types,
    ranges, options, and validation rules.

    Usage Example:
        from picasso_workflow.gui import ModuleDescriptor

        # Create descriptor instance
        descriptor = ModuleDescriptor()

        # Get parameter specifications for a module
        specs = descriptor.identify(0, {}, {})

        # Access GUI-friendly metadata
        param_specs = specs['parameters']
        result_specs = specs['results']

        # Use in GUI framework to automatically generate input widgets
        for param_name, param_info in param_specs.items():
            widget_type = param_info['type']
            is_required = param_info.get('required', False)
            description = param_info.get('description', '')

            if widget_type == 'file':
                # Create file browser widget
                widget = create_file_browser(
                    extensions=param_info.get('extensions', []),
                    mode=param_info.get('mode', 'open'),  # 'open' or 'save'
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'int':
                # Create number input widget
                widget = create_number_input(
                    min_val=param_info.get('min'),
                    max_val=param_info.get('max'),
                    step=param_info.get('step', 1),
                    default=param_info.get('default'),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'float':
                # Create decimal number input widget
                widget = create_float_input(
                    min_val=param_info.get('min'),
                    max_val=param_info.get('max'),
                    default=param_info.get('default'),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'str' and 'options' in param_info:
                # Create dropdown/combobox widget
                widget = create_dropdown(
                    options=param_info['options'],
                    default=param_info.get('default'),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'str':
                # Create text input widget
                widget = create_text_input(
                    multiline=param_info.get('multiline', False),
                    default=param_info.get('default', ''),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'bool':
                # Create checkbox widget
                widget = create_checkbox(
                    default=param_info.get('default', False),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'list':
                # Create list editor widget
                element_type = param_info.get('element_type', 'str')
                widget = create_list_editor(
                    element_type=element_type,
                    min_items=param_info.get('min_items', 0),
                    max_items=param_info.get('max_items'),
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'dict':
                # Create nested parameter group
                properties = param_info.get('properties', {})
                widget = create_parameter_group(
                    properties=properties,
                    collapsible=True,
                    required=is_required,
                    tooltip=description
                )

            elif widget_type == 'tuple':
                # Create tuple input widget
                length = param_info.get('length', 2)
                element_type = param_info.get('element_type', 'float')
                widget = create_tuple_input(
                    length=length,
                    element_type=element_type,
                    min_val=param_info.get('min'),
                    max_val=param_info.get('max'),
                    default=param_info.get('default'),
                    required=is_required,
                    tooltip=description
                )

            elif isinstance(widget_type, list):
                # Handle union types (multiple acceptable types)
                widget = create_union_input(
                    types=widget_type,
                    default_type=widget_type[0],
                    required=is_required,
                    tooltip=description
                )

            # Add widget to form with proper layout
            form.add_widget(param_name, widget, required=is_required)

        # Validation example
        def validate_parameters(param_values):
            errors = []
            for param_name, param_info in param_specs.items():
                value = param_values.get(param_name)

                # Check required parameters
                if param_info.get('required', False) and value is None:
                    errors.append(f"{param_name} is required")
                    continue

                if value is not None:
                    # Type validation
                    expected_type = param_info['type']
                    if not validate_type(value, expected_type):
                        errors.append(
                        f"{param_name} must be of type {expected_type}")

                    # Range validation for numbers
                    if expected_type in ['int', 'float']:
                        min_val = param_info.get('min')
                        max_val = param_info.get('max')
                        if min_val is not None and value < min_val:
                            errors.append(f"{param_name} must be >= {min_val}")
                        if max_val is not None and value > max_val:
                            errors.append(f"{param_name} must be <= {max_val}")

                    # Option validation for dropdowns
                    if (
                        'options' in param_info
                        and value not in param_info['options']
                    ):
                        errors.append(
                            f"{param_name} must be one "
                            + f"of {param_info['options']}")

            return errors

        # Result handling example
        def handle_results(result_values):
            for result_name, result_info in result_specs.items():
                result_type = result_info['type']
                description = result_info.get('description', '')

                if result_name in result_values:
                    value = result_values[result_name]

                    if (
                        result_type == 'str'
                        and result_name.endswith('_filepath')
                    ):
                        # Handle file outputs - could open, display, or link
                        display_file_result(value, description)

                    elif result_type in ['int', 'float']:
                        # Handle numeric results - display in status or plot
                        display_numeric_result(result_name, value, description)

                    elif result_type == 'numpy.ndarray':
                        # Handle array results - could plot or save
                        display_array_result(result_name, value, description)

                    elif result_type == 'dict':
                        # Handle complex results - could create summary table
                        display_dict_result(result_name, value, description)

    Type Specifications:
        The parameter and result specifications use the following type system:

        Basic Types:
            - 'str': String input (text field or dropdown if options provided)
            - 'int': Integer input (number field with step=1)
            - 'float': Decimal input (number field with decimal precision)
            - 'bool': Boolean input (checkbox)

        Complex Types:
            - 'file': File path input (file browser with extension filtering)
            - 'list': List input (dynamic list editor)
            - 'dict': Dictionary input (nested parameter group)
            - 'tuple': Fixed-length sequence (tuple editor)
            - 'numpy.ndarray': Array data (not for input, results only)

        Union Types:
            - ['type1', 'type2']: Multiple acceptable types
                (union input widget)

        Attributes
        ----------
        - 'required': bool - Whether parameter is mandatory
        - 'default': any - Default value for parameter
        - 'min', 'max': number - Range constraints for numeric types
        - 'step': number - Step size for numeric inputs
        - 'options': list - Valid choices for dropdown selection
        - 'extensions': list - File extensions for file inputs
        - 'mode': str - File dialog mode ('open' or 'save')
        - 'multiline': bool - Multi-line text input
        - 'length': int - Fixed length for tuples
        - 'element_type': str - Type of list/tuple elements
        - 'properties': dict - Sub-parameters for nested dictionaries
        - 'description': str - Human-readable description for tooltips
    """

    def __init__(self):
        super().__init__()

    def get_module_names(self):
        """Get a list of all available module names.

        Returns a list of all method names that correspond to workflow modules,
        excluding internal methods like __init__ and this method itself.

        Returns
        -------
        list of str: List of module names that can be used in workflows.
            These correspond to all the abstract methods implemented from
            AbstractModuleCollection.

        Examples
        --------
        descriptor = ModuleDescriptor()
        modules = descriptor.get_module_names()
        print(f"Available modules: {modules}")
        # Output: ['dummy_module', 'analysis_documentation',
                   'convert_zeiss_movie', ...]

        # Use in GUI to populate module selection dropdown
        module_dropdown.add_items(modules)

        # Get specifications for a specific module
        selected_module = 'identify'
        if selected_module in modules:
            specs = getattr(descriptor, selected_module)(0, {}, {})
            param_specs = specs['parameters']
            result_specs = specs['results']
        """
        # Get all methods of this class
        all_methods = [
            method for method in dir(self) if callable(getattr(self, method))
        ]

        # Filter out private methods, special methods, and non-module methods
        excluded_methods = {
            "__init__",
            "get_module_names",
            "get_docstring",
            "__class__",
            "__delattr__",
            "__dict__",
            "__dir__",
            "__doc__",
            "__eq__",
            "__format__",
            "__ge__",
            "__getattribute__",
            "__gt__",
            "__hash__",
            "__init_subclass__",
            "__le__",
            "__lt__",
            "__module__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__setattr__",
            "__sizeof__",
            "__str__",
            "__subclasshook__",
            "__weakref__",
        }

        # Get module methods
        # (exclude private/special methods and utility methods)
        module_methods = [
            method
            for method in all_methods
            if not method.startswith("_") and method not in excluded_methods
        ]

        # Sort alphabetically for consistent ordering
        return sorted(module_methods)

    def get_docstring(self, module):
        """Reads and returns the docstring of a module"""
        fun = getattr(self, module)
        return fun.__doc__

    def dummy_module(self):
        """A module that does nothing, for quickly removing
        modules in a workflow without having to renumber the
        following result idcs. Only for workflow debugging,
        remove when done.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters (unchanged)
        results : dict
            Input results (unchanged)
        """
        parameters_spec = {}

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
        }

        return parameters_spec, results_spec

    def analysis_documentation(self):
        """This module documents where and how analysis is being performed

        Parameters
        ----------
        parameters : dict
            This module does not use any parameters

        Returns
        -------
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
        parameters_spec = {}

        results_spec = {
            "picasso version": {
                "type": "str",
                "description": "version of picasso library used",
            },
            "picasso-workflow version": {
                "type": "str",
                "description": "version of picasso-workflow",
            },
            "Architecture": {
                "type": "str",
                "description": "machine architecture",
            },
            "OS": {
                "type": "str",
                "description": "operating system",
            },
            "host": {
                "type": "str",
                "description": "hostname of machine",
            },
            "processor": {
                "type": "str",
                "description": "processor information",
            },
            "CPU Frequency [MHz]": {
                "type": "float",
                "description": "current CPU frequency",
            },
            "CPU cores": {
                "type": "int",
                "description": "number of CPU cores",
            },
            "Memory total [GB]": {
                "type": "int",
                "description": "total system memory in GB",
            },
            "Memory available [GB]": {
                "type": "int",
                "description": "available system memory in GB",
            },
            "GPU": {
                "type": "str",
                "description": "GPU name or N/A",
            },
            "GPU memory [GB]": {
                "type": "int",
                "description": "GPU memory in GB or 0 if no GPU",
            },
        }

        return parameters_spec, results_spec

    def convert_zeiss_movie(self):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.

        Parameters
        ----------
        parameters : dict
            necessary items:
                filepath : str
                    the czi file name to load.
            optional items:
                filename_raw : str
                    the raw file name to write to
                info : dict, information as used by picasso

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results, updated with:
                filepath_raw : str
                    full path to the output raw file
                filename_raw : str
                    name of the output raw file
        """
        parameters_spec = {
            "filepath": {
                "type": "path",
                "description": "the czi file name to load",
                "extensions": [".czi"],
                "required": True,
            },
            "filename_raw": {
                "type": "str",
                "description": "the raw file name to write to",
                "required": False,
            },
            "info": {
                "type": "dict",
                "description": "information as used by picasso",
                "required": False,
            },
        }

        results_spec = {
            "filepath_raw": {
                "type": "str",
                "description": "full path to the output raw file",
            },
            "filename_raw": {
                "type": "str",
                "description": "name of the output raw file",
            },
        }

        return parameters_spec, results_spec

    def load_dataset_movie(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads DNA-PAINT movie data and metadata into memory for subsequent
        analysis. Optionally creates sample movies and loads camera
        configuration. The data is saved in self.movie and self.info.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters, potentially modified (sample_movie paths
            updated)
        results : dict
            Input results with added movie information and metadata
        """
        parameters_spec = {
            "filename": {
                "type": "path",
                "description": "Path to the movie file to load",
                "extensions": [".raw", ".tif", ".tiff", ".ome.tif"],
                "required": True,
            },
            "sample_movie": {
                "type": "dict",
                "description": "Parameters for creating a subsampled movie",
                "required": False,
                "properties": {
                    "filename": {
                        "type": "str",
                        "description": "Output filename for sample movie",
                        "default": "selected_frames.mp4",
                    },
                    "n_sample": {
                        "type": "int",
                        "description": "Number of frames to sample",
                        "min": 1,
                        "default": 40,
                    },
                    "max_quantile": {
                        "type": "float",
                        "description": "max quantile for display",
                        "min": 0,
                        "max": 1,
                        "default": 0.9998,
                    },
                    "fps": {
                        "type": "float",
                        "description": "frames per second for replay of \
                            subsampled movie",
                        "min": 1,
                        "max": 100,
                        "default": 2,
                    },
                },
            },
            "load_camera_info": {
                "type": "bool",
                "description": (
                    "Whether to load camera configuration from "
                    + "picasso.CONFIG"
                ),
                "required": False,
                "default": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for generated files",
            },
            "picasso version": {
                "type": "str",
                "description": "Version of picasso library used",
            },
            "movie.shape": {
                "type": "tuple",
                "description": "Movie dimensions (frames, width, height)",
                "length": 3,
                "element_type": "int",
            },
            "sample_movie": {
                "type": "dict",
                "description": (
                    "Results from subsampled movie creation (if requested)"
                ),
                "required": False,
            },
            "sample_movie, sample_frame_idx": {
                "type": "str",
                "description": ("sample movie frame indices"),
                "required": False,
            },
        }

        return parameters_spec, results_spec

    def load_dataset_localizations(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        The data is saved in
            self.locs
            self.info

        Parameters
        ----------
        parameters : dict
            necessary items:
                filename : str
                    the (main) file name to load. This can be image files,
                    or hdf5.

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results, updated with:
                picasso version : str
                    version of picasso library used
                nlocs : int
                    number of localizations loaded
        """
        parameters_spec = {
            "filename": {
                "type": "str",
                "description": "the (main) file name to load. This can be \
                    image files, or hdf5",
                "extensions": [".hdf5", ".h5"],
                "required": True,
            },
        }

        results_spec = {
            "picasso version": {
                "type": "str",
                "description": "version of picasso library used",
            },
            "nlocs": {
                "type": "int",
                "description": "number of localizations loaded",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def identify(self):
        """Identifies localizations in a loaded dataset.

        Identifies potential localization sites in the loaded movie using
        net gradient thresholding. Optionally performs automatic net gradient
        detection and creates identification vs frame plots.
        The data is saved in self.identifications.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters, potentially with updated min_gradient
        results : dict
            Input results with identification statistics and optional plots
        """
        parameters_spec = {
            "box_size": {
                "type": "int",
                "description": "Size of the detection box in pixels",
                "min": 3,
                "max": 21,
                "step": 2,
                "default": 7,
                "required": True,
            },
            "min_gradient": {
                "type": "float",
                "description": "Minimum net gradient threshold for detection",
                "min": 0.0,
                "max": 100000.0,
                "default": 20000,
                "required": True,
                "note": "Required unless auto_netgrad is provided",
            },
            "auto_netgrad": {
                "type": "dict",
                "description": (
                    "Parameters for automatic net gradient detection"
                ),
                "required": False,
                "properties": {
                    "box_size": {
                        "type": "int",
                        "description": "Box size for auto detection",
                        "min": 3,
                        "max": 21,
                        "step": 2,
                        "default": 7,
                    },
                    "frame_numbers": {
                        "type": ["list", "int"],
                        "description": "Frame range for analysis",
                        "default": 40,
                    },
                    "filename": {
                        "type": "str",
                        "description": (
                            "Output filename for auto-detection plot"
                        ),
                        "default": "auto-id.png",
                    },
                    "start_ng": {
                        "type": "float",
                        "description": "Starting net gradient value",
                        "min": -10000.0,
                        "default": -3000,
                    },
                    "zscore": {
                        "type": "float",
                        "description": "Z-score threshold for detection",
                        "min": 0.0,
                        "default": 2.0,
                    },
                    "bins": {
                        "type": "int",
                        "description": "Number of histogram bins",
                        "min": 10,
                        "max": 1000,
                        "default": 100,
                    },
                },
            },
            "ids_vs_frame": {
                "type": "dict",
                "description": (
                    "Parameters for plotting identifications vs time (dict)"
                ),
                "required": False,
                "properties": {
                    "filename": {
                        "type": "str",
                        "description": "Output filename for plot",
                    }
                },
                "default": "{'filename': 'ids_vs_frame.png'}",
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for generated files",
            },
            "num_identifications": {
                "type": "int",
                "description": "Total number of identifications found",
                "min": 0,
            },
            "auto_netgrad": {
                "type": "dict",
                "description": (
                    "Results from automatic net gradient detection"
                    + " (if requested)"
                ),
                "required": False,
            },
            "ids_vs_frame": {
                "type": "dict",
                "description": (
                    "Results from identifications vs frame analysis"
                    + " (if requested)"
                ),
                "required": False,
            },
        }

        return parameters_spec, results_spec

    def localize(self):
        """Localizes Spots previously identified.
        The data is saved in
            self.locs

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results, updated with:
                locs_vs_frame : dict
                    plot results if locs_vs_frame parameter was provided
                locs_columns : list
                    list of column names in the localizations array
        """
        parameters_spec = {
            "box_size": {
                "type": "int",
                "description": "as always",
                "min": 3,
                "max": 21,
                "step": 2,
                "default": 7,
                "required": True,
            },
            "fit_parallel": {
                "type": "bool",
                "description": "whether to fit on multiple cores",
                "default": False,
                "required": True,
            },
            "locs_vs_frame": {
                "type": "dict",
                "description": "Dictionary for plotting locs vs time. e.g."
                "{'filename': 'locsvsframe.png'}",
                "default": "{'filename': 'locsvsframe.png'}",
                "required": False,
            },
            # "save_locs": {
            #     "type": "dict",
            #     "description": "if saving localizations is requested. Items \
            #         correpsond to arguments of save_locs",
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "locs_vs_frame": {
                "type": "dict",
                "description": "plot results if locs_vs_frame parameter was \
                    provided",
            },
            "locs_columns": {
                "type": "list",
                "description": "list of column names in the localizations \
                    array",
            },
        }

        return parameters_spec, results_spec

    def zfit(self):
        """Fits z positions to previously localized spots.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results
        """
        parameters_spec = {
            "magnification_factor": {
                "type": "float",
                "description": "The magnification factor to compensate stage scanning "
                "calibration vs in-sample measurement.",
                "default": 0.79,
                "min": 0,
                "max": 1e6,
                "required": True,
            },
            "fp_calibration": {
                "type": "path",
                "description": "The calibration file path to use. If not given, the filepath"
                "from config is loaded for the microscope and emission wavelength. "
                "Keep in mind this must be a path on the cluster for now"
                " (i.e. /fs/mpib/pool-miblab5/... instead of /Volumes/pool...)",
                "default": "",
                "required": False,
            },
            # "save_locs": {
            #     "type": "dict",
            #     "description": "if saving localizations is requested. Items \
            #         correpsond to arguments of save_locs",
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "fp_calibration": {
                "type": "str",
                "description": "The calibration file path used",
            },
            "fp_calibration_fig": {
                "type": "str",
                "description": "The calibration graph copied to the results folder",
            },
        }

        return parameters_spec, results_spec

    def load_picassoconfig(self):
        """
        Loads a specific picasso configuration file, as opposed to the default
        version residing in the picasso installation folder.

        Parameters
        ----------
        i : int
            the module index in the protocol
        parameters : dict
            necessary items:
                fp_config : str
                    filepath to a config file.
        results : dict
            the results dict, created by the module_decorator

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results, updated with:
        """
        parameters_spec = {
            "fp_config": {
                "type": "path",
                "description": "Filepath to a specific picasso config. "
                "Keep in mind this must be a path on the cluster for now"
                " (i.e. /fs/mpib/pool-miblab5/... instead of /Volumes/pool...)",
                "required": True,
            },
            # "save_locs": {
            #     "type": "dict",
            #     "description": "If saving localizations is requested. Items \
            #         correpsond to arguments of save_locs",
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
        }

        return parameters_spec, results_spec

    def export_brightfield(self):
        """Opens a single-plane tiff image and saves it to png with
        contrast adjustment.

        Parameters
        ----------
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

        Returns
        -------
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
        parameters_spec = {
            "filepath": {
                "type": ["str", "list", "dict"],
                "description": "the tiff file(s) to load. The converted \
                    file(s) will have the same name, but with .png extension. \
                    If dict: keys are labels",
                "extensions": [".tif", ".tiff"],
                "required": True,
            },
            "min_quantile": {
                "type": "float",
                "description": "the quantile below which pixels are shown \
                    black",
                "min": 0.0,
                "max": 1.0,
                "default": 0,
                "required": False,
            },
            "max_quantile": {
                "type": "float",
                "description": "the quantile above which pixels are shown \
                    white",
                "min": 0.0,
                "max": 1.0,
                "default": 1,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "labeled filepaths": {
                "type": "dict",
                "description": "keys : labels, values : filepaths",
            },
            "success": {
                "type": "bool",
                "description": "whether the export was successful",
            },
            # "keys": {
            #     # TODO: Add type, description, min, max, default, required, step, extensions, properties
            #     # Hint: type appears to be labels
            # },
            # "values": {
            #     # TODO: Add type, description, min, max, default, required, step, extensions, properties
            #     # Hint: type appears to be filepaths
            # },
        }

        return parameters_spec, results_spec

    def render(self):
        """Renders localizations on the whole field of view, and on
        a zoom in around the center of mass of localizations.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results, updated with:
                fp_scene_fullfov : str
                    filepath to full FOV rendering
                fp_scene_ctrmass : str
                    filepath to center of mass zoom rendering (conditional,
                    only if ctrmass_fov_nm provided)
        """
        parameters_spec = {
            "ctrmass_fov_nm": {
                "type": "float",
                "description": "Field of view of the zoom in rendering around \
                    the center of mass in nm",
                "min": 0,
                "default": 10000,
                "required": False,
            },
            "fullfov_pixelsize": {
                "type": "float",
                "description": "The rendered pixel size [nm] of the full FOV \
                    rendering",
                "min": 0,
                "default": 130,
                "required": False,
            },
            "ctrmass_pixelsize": {
                "type": "float",
                "description": "The rendered pixel size [nm] of the zoom in \
                    rendering around the center of mass",
                "min": 0,
                "default": 100,
                "required": False,
            },
            "ctrmass_blur_method": {
                "type": "str",
                "description": "Blur method",
                "options": ["gaussian", "gaussian_iso", "smooth", "convolve"],
                "required": False,
            },
            "ctrmass_min_blur_width": {
                "type": "float",
                "description": "min blur with",
                "min": 0,
                "required": False,
            },
            "ctrmass_ang": {
                "type": "float",
                "description": "angle",
                "required": False,
            },
            "generate_active_rois": {
                "type": "bool",
                "description": "Whether to generate density-driven active site zoom-in previews next to the overview image",
                "default": True,
                "required": False,
            },
            "n_active_rois": {
                "type": "int",
                "description": "Number of high-density active sites (ROIs) to generate",
                "min": 1,
                "max": 25,
                "default": 4,
                "required": False,
            },
            "colormap": {
                "type": "str",
                "description": "Colormap of the generated images",
                "options": ["magma", "hot", "inferno", "viridis", "gray"],
                "default": "magma",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "fp_scene_fullfov": {
                "type": "str",
                "description": "filepath to full FOV rendering",
            },
            "fp_scene_fullfov_unmarked": {
                "type": "str",
                "description": "filepath to full FOV rendering without outlines",
            },
            "fp_scene_ctrmass": {
                "type": "str",
                "description": "filepath to center of mass zoom rendering \
                    (conditional, only if ctrmass_fov_nm provided)",
            },
            "fp_scene_rois": {
                "type": "list",
                "description": "list of filepaths to the density-driven active site Zoom-In previews",
            },
        }

        return parameters_spec, results_spec

    def undrift_rcc(self):
        """Undrifts localized data using redundant cross correlation.
        drift is saved in
        self.drift

        Parameters
        ----------
        i : int
            the module index in the protocol
        parameters : dict
            necessary items:
                segmentation : int
                    the number of frames segmented for RCC
            optional items:
                max_iter_segmentations : int, default: 3
                    maximum number of iterations to adaptively increase
                    segmentation if RCC fails
                filename : str
                    the drift txt file name
        results : dict
            the results dict, created by the module_decorator

        Returns
        -------
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
                    filepath to drift txt file (conditional, only if
                    undrifting succeeded)
                filepath_plot : str
                    filepath to drift plot png (conditional, only if
                    undrifting succeeded)
        """
        parameters_spec = {
            "segmentation": {
                "type": "int",
                "description": "the number of frames segmented for RCC",
                "min": 2,
                "max": 1000,
                "default": 50,
                "required": True,
            },
            "max_iter_segmentations": {
                "type": "int",
                "description": "maximum number of iterations to adaptively \
                    increase segmentation if RCC fails",
                "min": 1,
                "max": 10,
                "default": 3,
                "required": False,
            },
            "filename": {
                "type": "str",
                "description": "the drift txt file name",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "success": {
                "type": "bool",
                "description": "whether undrifting was successful",
            },
            "message": {
                "type": "str",
                "description": "error or warning messages if any",
            },
            "filepath_driftfile": {
                "type": "str",
                "description": "filepath to drift txt file (conditional, \
                only if undrifting succeeded)",
            },
            "filepath_plot": {
                "type": "str",
                "description": "filepath to drift plot png (conditional, \
                only if undrifting succeeded)",
            },
        }

        return parameters_spec, results_spec

    def undrift_aim(self):
        """Unrift localized data using the AIM algorithm
        drift is saved in
        self.drift

        Parameters
        ----------
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
                    Should be larger than the maximum expected drift
                    wihtin segmentation.
                dimensions : list of str
                    the dimensions undrifted, typically ['x', 'y'].
            optional items:
                progress : callback function
                    progress callback for status updates
        results : dict
            the results dict, created by the module_decorator

        Returns
        -------
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
        parameters_spec = {
            "segmentation": {
                "type": "int",
                "description": "the number of frames segmented",
                "min": 2,
                "max": 1000,
                "default": 50,
                "required": True,
            },
            "intersect_d": {
                "type": "float",
                "description": "Intersect distance in nanometers.",
                "min": 0,
                "required": True,
            },
            "roi_r": {
                "type": "float",
                "description": "Radius of the local search region in \
                nanometers. Should be larger than the maximum expected drift \
                wihtin segmentation.",
                "min": 0,
                "required": True,
            },
            "dimensions": {
                "type": "list",
                "description": "the dimensions undrifted, typically \
                    ['x', 'y'].",
                "element_type": "str",
                "default": ["x", "y"],
                "required": True,
            },
            # "progress": {
            #     "type": "function",
            #     "description": "progress callback for status updates",
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "success": {
                "type": "bool",
                "description": "whether undrifting was successful",
            },
            "fp_driftfile": {
                "type": "str",
                "description": "filepath to drift txt file",
            },
            "fp_fig": {
                "type": "str",
                "description": "filepath to drift plot png",
            },
        }

        return parameters_spec, results_spec

    def manual(self):
        """Handles a manual step: if the files required are not
        present, prompt the user to provide them. if they are, move
        to the next step.

        Parameters
        ----------
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
        parameters_spec = {
            "message": {
                "type": "str",
                "description": "Message to display to user",
                "required": False,
                "multiline": True,
            },
            "wait_for_input": {
                "type": "bool",
                "description": "Whether to wait for user confirmation",
                "default": True,
                "required": False,
            },
            # "prompt": {
            #     # TODO: Add type, description, min, max, default,
            #     #     required, step, extensions, properties
            #     # Hint: type appears to be str
            #     # Hint: required = True
            # },
            # "filename": {
            #     # TODO: Add type, description, min, max, default,
            #     # required, step, extensions, properties
            #     # Hint: type appears to be str
            #     # Hint: required = True
            # },
            # "save_locs": {
            #     # TODO: Add type, description, min, max, default,
            #     # required, step, extensions, properties
            #     # Hint: type appears to be bool
            #     # Hint: required = False (optional)
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
        }

        return parameters_spec, results_spec

    def summarize_dataset(self):
        """Summarize dataset using various analysis methods

        Computes dataset quality metrics such as NeNa (Nearest Neighbor
        Analysis) and median localization precision.

        Parameters
        ----------
        i : int
            The index of the module in the workflow
        parameters : dict
            Required keys:
                methods : dict
                    Dictionary of analysis methods to run. Keys are
                    method names, values are method-specific parameter
                    dicts.
                    Supported methods:
                        "nena" : dict (no parameters)
                            Performs Nearest Neighbor Analysis to estimate
                            localization precision
                        "median-loc-precision" : dict
                            Calculates median localization precision
                            Optional keys:
                                qe_correction : float
                                    Quantum efficiency correction factor
                                    (default: 1)
        results : dict
            the results dict, created by the module_decorator

        Returns
        -------
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
                median-loc-precision : dict (if median-loc-precision method
                        used)
                    Dictionary with keys:
                        median_lp-px : float - median localization
                            precision in pixels
                        median_lp-nm : float - median localization
                            precision in nanometers
        """
        parameters_spec = {
            "include_plots": {
                "type": "bool",
                "description": "Whether to include visualization plots",
                "default": True,
                "required": False,
            },
            "methods": {
                "type": "dict",
                "description": ("Methods to summarize"),
                "required": True,
                "properties": {
                    "nena": {
                        "type": "dict",
                        "description": "NeNa calculation",
                        "required": False,
                    },
                    "median-loc-precision": {
                        "type": "dict",
                        "description": "median localizatino precision \
                            calculation",
                        "required": False,
                        "properties": {
                            "qe_correction": {
                                "type": "float",
                                "description": "QE correction factor",
                                "default": 1,
                            },
                        },
                    },
                },
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "summary_report": {
                "type": "dict",
                "description": "Comprehensive analysis summary",
            },
            "nena, nena-nm": {
                "type": "float",
                "description": "NeNa value in nanometers",
            },
            "nena, nena-px": {
                "type": "float",
                "description": "NeNa value in pixels",
            },
            "nena, filepath_plot": {
                "type": "str",
                "description": "File path of the graph",
            },
            "median-loc-precision, median_lp-px": {
                "type": "float",
                "description": "Median localization precision in pixels",
            },
            "median-loc-precision, median_lp-nm": {
                "type": "float",
                "description": "Median localization precision in nanometers",
            },
        }

        return parameters_spec, results_spec

    def density(self):
        """Calculate local localization density

        Parameters
        ----------
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
        parameters_spec = {
            "radius": {
                "type": "float",
                "description": "Radius for local density calculation in nm",
                "min": 1.0,
                "max": 1000.0,
                "default": 50.0,
                "required": True,
            },
            # "save_locs": {
            #     "type": "bool",
            #     "description": (
            #         "Whether to save density-annotated localizations"
            #     ),
            #     "default": False,
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "nlocs": {
                "type": "int",
                "description": "Number of localizations processed",
                "min": 0,
            },
            "density_stats": {
                "type": "dict",
                "description": "Statistical summary of density calculations",
                "required": False,
            },
        }

        return parameters_spec, results_spec

    def dbscan(self):
        """Perform clustering using dbscan.

        Applies DBSCAN clustering algorithm to localizations, optionally
        replacing localizations with cluster centers for subsequent analysis.
        After this module, the standard locs will be the cluster centers.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters (unchanged)
        results : dict
            Input results with clustering outputs and file paths
        """
        parameters_spec = {
            "radius": {
                "type": "float",
                "description": "The DBSCAN radius parameter in nm",
                "min": 1.0,
                "max": 1000.0,
                "default": 50.0,
                "required": True,
            },
            "min_samples": {
                "type": "int",
                "description": (
                    "Number of localizations within radius to consider a "
                    "given point a core sample."
                ),
                "min": 1,
                "max": 100,
                "default": 3,
                "required": True,
            },
            "min_locs": {
                "type": "int",
                "description": (
                    "Minimum number of localizations in a cluster. Clusters with"
                    "fewer localizations will be removed. Default is 0."
                ),
                "min": 0,
                "max": 100,
                "default": 0,
                "required": True,
            },
            "continue_with_centers": {
                "type": "bool",
                "description": (
                    "Whether to replace localizations with cluster centers"
                ),
                "default": True,
                "required": True,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for generated files",
            },
            "fp_fig_clustersizes": {
                "type": "str",
                "description": "Filepath to cluster size distribution figure",
            },
            "fp_centers": {
                "type": "str",
                "description": "Filepath to cluster centers file",
            },
        }

        return parameters_spec, results_spec

    def hdbscan(self):
        """Perform hdbscan clustering. After this module, the standard
        locs will be the cluster centers.

        Parameters
        ----------
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
        parameters_spec = {
            "min_cluster": {
                "type": "int",
                "description": "Minimum cluster size for HDBSCAN",
                "min": 2,
                "max": 1000,
                "default": 5,
                "required": True,
            },
            "min_samples": {
                "type": "int",
                "description": "Minimum samples parameter for HDBSCAN",
                "min": 1,
                "max": 100,
                "default": 3,
                "required": True,
            },
            "continue_with_centers": {
                "type": "bool",
                "description": (
                    "Whether to use cluster centers for subsequent analysis"
                ),
                "default": True,
                "required": False,
            },
            # "save_locs": {
            #     "type": "bool",
            #     "description": "Whether to save clustered localization data",
            #     "default": False,
            #     "required": False,
            # },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "n_clusters": {
                "type": "int",
                "description": "Number of clusters identified",
                "min": 0,
                "required": False,
            },
            "cluster_centers": {
                "type": "numpy.ndarray",
                "description": "Coordinates of cluster centers",
                "required": False,
            },
        }

        return parameters_spec, results_spec

    def binding_event_analysis(self):
        """Evaluate binding events according to Philipp Steen's methods

        Steen, P.R., Unterauer, E.M., Masullo, L.A. et al.
        The DNA-PAINT palette: a comprehensive performance analysis
        of fluorescent dyes.
        Nat Methods (2024).
        https://doi.org/10.1038/s41592-024-02374-8

        Parameters
        ----------
        i : int
            the index of the module
        parameters: dict
            with required keys:
                fp_locs : str
                    file path to input locs
                n_frames
        """
        parameters_spec = {
            "radius": {
                "type": "float",
                "description": "Clustering radius in nm",
                "min": 1.0,
                "max": 1000.0,
                "default": 20.0,
                "required": True,
            },
            "min_binding_time": {
                "type": "float",
                "description": "Minimum binding time threshold",
                "min": 0.0,
                "max": 10.0,
                "default": 0.1,
                "required": False,
            },
            "fp_locs": {
                "type": "path",
                "description": "The filepath of the .hdf5 file to write",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "binding_events": {
                "type": "int",
                "description": "Number of binding events detected",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def smlm_clusterer(self):
        """Perform smlm clustering. After this module, the standard
        locs will be the cluster centers.

        Parameters
        ----------
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
        parameters_spec = {
            # "method": {
            #     "type": "str",
            #     "description": "Clustering method to use",
            #     "options": ["voronoi", "dbscan_like", "hierarchical"],
            #     "required": True,
            # },
            "radius": {
                "type": "float",
                "description": "Clustering radius parameter [nm]",
                "min": 0.0,
                "max": 1000.0,
                "default": 50.0,
                "required": False,
            },
            "min_locs": {
                "type": "int",
                "description": "Minimum number of localizations in a cluster",
                "required": True,
            },
            "basic_fa": {
                "type": "bool",
                "description": "Whether to perform basic frame analysis",
                "default": False,
                "required": False,
            },
            "radius_z": {
                "type": "float",
                "description": "The smlm radius_z [nm]",
                "default": None,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "n_clusters": {
                "type": "int",
                "description": "Number of clusters found",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def gaussian_mixture_cluster(self):
        """Perform clustering using gaussian mixture modelsAfter this module,
        the standard locs will be the Gaussian centers.

        Parameters
        ----------
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
        parameters_spec = {
            "min_locs": {
                "type": "int",
                "description": "Minimum number of localizations in a \
                    component",
                "min": 0,
                "max": 50,
                "default": 3,
                "required": True,
            },
            "max_rounds_without_best_bic": {
                "type": "int",
                "description": "Maximum number of rounds without BIC \
                    improvement to\
                    terminate the optimal GMM search.",
                "default": 3,
                "required": False,
            },
            "bootstrap_check": {
                "type": "bool",
                "description": "If True, the standard error of the means \
                        (SEM) is calculated using bootstrapping. If False, \
                        the standard, single Gaussian SEM is used as\
                        approximation.",
                "default": False,
                "required": False,
            },
            "calibration": {
                "type": "dict",
                "description": "Calibration dictionary with x and y \
                        coefficients, z step size and the number of frames.\
                        Only required for 3D data.",
                "default": None,
                "required": False,
            },
            # "asynch": {
            #     "type": "bool",
            #     "description": "If True, the GMM search is run in parallel using\
            #             multiprocessing. If False, the GMM search is run\
            #             without multiprocessing.",
            #     "default": True,
            #     "required": False,
            # },
            # "sigma_bounds": {
            #     "type": "float",
            #     "description": "(not recommended)\
            #             Minimum standard deviation of the Gaussian components\
            #             in nanometers. Useful for avoiding overfitting within\
            #             a single localization cloud. Now using individual\
            #             loc precision, so min_sigma is not recommended.",
            #     "default": None,
            #     "required": False,
            # },
            "loc_prec_handle": {
                "type": "str",
                "description": 'One of ["local", "global", "abs"]',
                "default": "local",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "cluster_assignments": {
                "type": "numpy.ndarray",
                "description": "Cluster assignment probabilities",
            },
        }

        return parameters_spec, results_spec

    def nneighbor(self):
        """Perform nearest neighbor calculation. Plot NN histogram
        and radial distribution function.

        Parameters
        ----------
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
        parameters_spec = {
            "save_data": {
                "type": "bool",
                "description": "Whether to save nearest neighbor data",
                "default": True,
                "required": False,
            },
            "dims": {
                "type": "list",
                "description": "the distance dimensions, e.g. ['x', 'y']",
                "default": ["x", "y"],
                "required": False,
            },
            "nth_NN": {
                "type": "int",
                "description": "Number of nearest neighbors to calculate",
                "min": 1,
                "max": 50,
                "default": 5,
                "required": True,
            },
            "nth_rdf": {
                "type": "int",
                "description": "Calculate distances up to the 95th percile \
                    of the nth_rdf nearest neighbor",
                "min": 1,
                "required": True,
            },
            "subsample_1stNN": {
                "type": "float",
                "description": "By how much fold to subsample distances from \
                the median of the 1st nearest neighbor.",
                "min": 1,
                "default": 20,
                "required": False,
            },
            "add_column": {
                "type": "bool",
                "description": "Whether to add a column of nearest neighbor \
                    distance to the locs.",
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "nneighbor": {
                "type": "Path",
                "description": "File path to the nearest neighbor data",
            },
        }

        return parameters_spec, results_spec

    def fit_csr(self):
        """Fit a Completely Spatially Random Distribution to nearest neighbors.

        Fits CSR model to nearest neighbor distance distributions and evaluates
        goodness-of-fit using statistical measures and visualization.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters (unchanged)
        results : dict
            Input results with CSR fitting results and goodness-of-fit
            metrics
        """
        parameters_spec = {
            "nneighbors": {
                "type": ["str", "numpy.ndarray", "list"],
                "description": (
                    "Nearest neighbor data (file path, array, or list)"
                ),
                "extensions": [".txt", ".npy"],
                "required": True,
            },
            "dimensionality": {
                "type": "int",
                "description": "Spatial dimensionality (2 or 3) for CSR model",
                "options": [2, 3],
                "default": 2,
                "required": True,
            },
            "kmin": {
                "type": "int",
                "description": "Minimum k-th nearest neighbor order to fit",
                "min": 1,
                "max": 20,
                "default": 1,
                "required": False,
            },
            "min_dist": {
                "type": "float",
                "description": (
                    "Minimum observable distance in nm due to technical"
                    + " limits"
                ),
                "min": 0.0,
                "max": 100.0,
                "required": False,
            },
            "max_dist": {
                "type": "float",
                "description": "Maximum distance for filtering analysis",
                "min": 10.0,
                "max": 10000.0,
                "required": False,
            },
            "bkg_fraction": {
                "type": "float",
                "description": "Background fraction for fitting",
                "min": 0.0,
                "max": 1.0,
                "required": False,
            },
            "fit_bkg": {
                "type": "bool",
                "description": "Whether to fit background",
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for generated files",
            },
            "density": {
                "type": ["float", "list"],
                "description": "Fitted spatial density value(s) in units^(-d)",
            },
            "bkg_fraction": {
                "type": "list",
                "description": "Background fraction values",
            },
            "fp_fig": {
                "type": ["str", "list"],
                "description": (
                    "Filepath(s) to CSR fit visualization figure(s)"
                ),
            },
            "wasserstein_distances_per_k": {
                "type": "list",
                "description": (
                    "Wasserstein distances for each k-th nearest neighbor"
                    + " order"
                ),
            },
            "mean_wasserstein_distance": {
                "type": ["float", "list"],
                "description": "Mean Wasserstein distance across all k orders",
            },
            "ks_pvalues_per_k": {
                "type": "list",
                "description": (
                    "Kolmogorov-Smirnov p-values for each k-th NN order"
                ),
            },
        }

        return parameters_spec, results_spec

    def save_single_dataset(self):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward.

        Parameters
        ----------
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
        parameters_spec = {
            "filename": {
                "type": "str",
                "description": "Custom filename for saved data",
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "filepath": {
                "type": "str",
                "description": "Path to saved dataset file",
            },
        }

        return parameters_spec, results_spec

    # Aggregation workflow modules
    def load_datasets_to_aggregate(self):
        """Loads the results of single-dataset workflows

        Parameters
        ----------
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
        parameters_spec = {
            "filepaths": {
                "type": "list",
                "description": "List of dataset files to load",
                "element_type": "file",
                "extensions": [".hdf5", ".h5"],
                "required": True,
            },
            "tags": {
                "type": "list",
                "description": "Custom tags for each dataset",
                "element_type": "str",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "n_datasets": {
                "type": "int",
                "description": "Number of datasets loaded",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def align_channels(self):
        """Aligns multiple channels to each other (part of an aggregation
        workflow)

        Parameters
        ----------
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
        parameters_spec = {
            # "reference_channel": {
            #     "type": "int",
            #     "description": "Reference channel index for alignment",
            #     "min": 0,
            #     "max": 10,
            #     "default": 0,
            #     "required": False,
            # },
            "filepaths": {
                "type": "list",
                "description": "Filepaths to localization datasets to be \
                    aligned, in case the data from previous modules is not\
                    to be used.",
                "required": False,
            },
            "fp_fiducials": {
                "type": "list",
                "description": "Filepaths to localization datasets to be \
                    used as fiducial markers for alignment.",
                "required": False,
            },
            "align_pars": {
                "type": "dict",
                "description": "Parameters for alignment.",
                "properties": {
                    "max_iterations": {
                        "type": "int",
                        "description": "Maximum iterations of alogrithm.",
                        "default": 5,
                    },
                    "convergence": {
                        "type": "float",
                        "description": "Convergence criterion.",
                        "default": 0.001,
                    },
                    "force_method": {
                        "type": "str",
                        "description": "Whether to force a method or"
                        " let the alogritm decide on the best method."
                        "options: 'RCC', 'picked', 'RSSO'",
                        "default": None,
                    },
                    "max_shift": {
                        "type": "float",
                        "description": "Maximum allowed shift in alignment.",
                        "default": None,
                    },
                    "plot_histogram": {
                        "type": "bool",
                        "description": "Whether to plot the histogram.",
                        "default": False,
                    },
                    "plot_dir": {
                        "type": "str",
                        "description": "Which directory to plot into.",
                        "default": None,
                    },
                },
            },
            "fig_filename": {
                "type": "str",
                "description": "Filename of the figure to be generated.",
                "required": False,
            },
            "crop_boundaries": {
                "type": "bool",
                "description": "Whether to crop the data to the region of \
                    overlap of all channels.",
                "required": False,
            },
            "fp_co_shift_channel_locs": {
                "type": "list",
                "description": "Filepaths to locs that should be shifted along\
                    with the channel data, but not used for assessing \
                    alignment. This could e.g. be clustered locs, when the \
                    alignment is done based on cluster centers.",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "fp_fiducials": {
                "type": "list",
                "description": "Filepaths to the fiducials for all channels",
            },
            "shifts": {
                "type": "array",
                "description": "The shifts of channels relative to first.",
            },
            "alignment_matrix": {
                "type": "numpy.ndarray",
                "description": "Transformation matrix for alignment",
            },
        }

        return parameters_spec, results_spec

    def combine_channels(self):
        """Combines multiple channels into one dataset. This is relevant
        e.g. for RESI.

        Parameters
        ----------
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
        parameters_spec = {
            "channel_weights": {
                "type": "list",
                "description": "Relative weights for each channel",
                "element_type": "float",
                "min": 0.0,
                "max": 10.0,
                "required": False,
            },
            "tag": {
                "type": "str",
                "description": "The tag/name to assign to the combined \
                    dataset.",
                "required": False,
            },
            "combine_col": {
                "type": "str",
                "description": "The column name for the IDs to the different \
                    datasets. Allows back-tracking locs to their origin.",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "combined_nlocs": {
                "type": "int",
                "description": (
                    "Total number of localizations in combined dataset"
                ),
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def save_datasets_aggregated(self):
        """Save data of multiple single-dataset workflows from one
        aggregation workflow.

        Saves all channel localization data and metadata from the aggregated
        workflow to individual files in the results folder.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters (unchanged)
        results : dict
            Updated results dictionary with saved file paths
        """
        parameters_spec = {}

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "filepaths": {
                "type": "list",
                "description": (
                    "List of all saved file paths from the aggregated"
                    + " datasets"
                ),
                "element_type": "str",
            },
        }

        return parameters_spec, results_spec

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
    #     parameters_spec = {
    #         "proposed_labeling_efficiency": {
    #             "type": "float",
    #             "description": "Labeling efficiency percentage, default for \
    #                 all targets used proposed value in spinna_config.csv and \
    #                 can be altered manually after the first run of this \
    #                 module",
    #             "min": 0.0,
    #             "max": 100.0,
    #             "required": True,
    #         },
    #         "proposed_labeling_uncertainty": {
    #             "type": "float",
    #             "description": "Labeling uncertainty [nm]; good value is e.g.\
    #                     5 used proposed value in spinna_config.csv and can be\
    #                     altered manually after the first run of this module",
    #             "default": 5,
    #             "required": True,
    #         },
    #         "proposed_n_simulate": {
    #             "type": "int",
    #             "description": "Number of Monte Carlo simulations",
    #             "min": 10,
    #             "max": 10000,
    #             "default": 1000,
    #             "required": True,
    #         },
    #         "proposed_density": {
    #             "type": "float",
    #             "description": "Density to simulate; area density if 2D; \
    #                 volume density if 3D used proposed value in \
    #                 spinna_config.csv and can be altered manually after the \
    #                 first run of this module",
    #             "required": True,
    #         },
    #         "proposed_nn_plotted": {
    #             "type": "int",
    #             "description": "Number of nearest neighbors to plot used \
    #                 proposed value in spinna_config.csv and can be \
    #                 alteredmanually after the first run of this module",
    #             "required": True,
    #         },
    #         "structures": {
    #             "type": "list",
    #             "element_type": "dict",
    #             "description": 'SPINNA structures. Each structure dict has \
    #                     "Molecular targets": list of str, \
    #                     "Structure title": str, \
    #                     "TARGET_x": list of float, \
    #                     "TARGET_y": list of float, \
    #                     "TARGET_z": list of float, \
    #                 where TARGET is one each of the target names in \
    #                 "Molecular targets"',
    #             "required": False,
    #         },
    #         "structures_d": {
    #             "type": "float",
    #             "description": "Distance between molecules within \
    #                 auto-generated structures, in nm. Only necessary if \
    #                 'structures' is not given.",
    #             "required": False,
    #         },
    #     }

    #     results_spec = {
    #         "start time": {
    #             "type": "str",
    #             "description": "Module execution start timestamp",
    #         },
    #         "end time": {
    #             "type": "str",
    #             "description": "Module execution end timestamp",
    #         },
    #         "duration": {
    #             "type": "float",
    #             "description": "Module execution duration in seconds",
    #             "min": 0.0,
    #         },
    #         "folder": {
    #             "type": "str",
    #             "description": "Output folder for module results",
    #         },
    #         "spinna_results": {
    #             "type": "dict",
    #             "description": "SPINNA analysis results",
    #         },
    #     }

    #     return parameters_spec, results_spec

    def spinna(self):
        """Direct implementation of spinna batch analysis.
        The current locs file(s) are saved into the results folder, and
        a template csv file is created. This csv needs to be filled out by the
        user in a manual step before the spinna analysis is carried out.

        Parameters
        ----------
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
                granularity : int
                the spinna granularity
            optional keys:
                density_app : list of float
                    apparent density in 1/nm^2;
                    this is the product of 'real' density & lbl efficiency
        """
        parameters_spec = {
            "max_radius": {
                "type": "float",
                "description": "Maximum analysis radius in nm",
                "min": 10.0,
                "max": 1000.0,
                "default": 200.0,
                "required": True,
            },
            "optimization_method": {
                "type": "str",
                "description": "Method for parameter optimization",
                "options": ["mle", "leastsq", "bayesian"],
                "default": "mle",
                "required": False,
            },
            "labeling_efficiency": {
                "type": "dict",
                "element_type": "float",
                "description": "Labeling efficiency (0-1), for all targets",
                "min": 0.0,
                "max": 1.0,
                "required": True,
            },
            "labeling_uncertainty": {
                "type": "float",
                "description": "labeling uncertainty [nm]; good value is e.g. \
                    5 assumed the same value for all targets (can be dict for \
                    targets)",
                "required": True,
            },
            "n_simulate": {
                "type": "int",
                "description": "Number of target molecules to simulated.",
                "min": 10,
                "max": 10000,
                "default": 5000,
                "required": True,
            },
            "structures": {
                "type": "str",
                "description": "Filepath to yaml file with structure \
                    definition. Can also be a list of dict of the \
                    definitions.",
                "required": True,
            },
            "fp_mask_dict": {
                "type": "str",
                "description": "The filepath to the mask_dict file.",
                "required": True,
            },
            "density": {
                "type": "list",
                "element_type": "float",
                "description": "Densities to simulate in 1/nm^d; area density \
                    if 2D; volume density if 3D.",
                "required": True,
            },
            "sim_repeats": {
                "type": "int",
                "description": "Number of simulation repeats.",
                "required": True,
            },
            "fit_NND_bin": {
                "type": "float",
                "description": "bin size of fits",
                "required": True,
            },
            "fit_NND_maxdist": {
                "type": "float",
                "description": "Maximum of histogram shown",
                "required": True,
            },
            "n_nearest_neighbors": {
                "type": "int",
                "description": "Number of nearest neighbors to evaluate.",
                "required": True,
            },
            "granularity": {
                "type": "int",
                "description": "The spinna granularity",
                "required": True,
            },
            "density_app": {
                "type": "float",
                "description": "Tpparent density in 1/nm^2; this is the \
                    product of 'real' density & lbl efficiency",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "optimal_parameters": {
                "type": "dict",
                "description": "Optimized SPINNA parameters",
            },
        }

        return parameters_spec, results_spec

    def spinna_batch(self):
        """Run a SPINNA batch analysis from a pre-existing config file.

        The current locs file(s) are saved as .hdf5 into the module's
        results folder. Their filepaths are written into the SPINNA
        batch config csv (given via ``fp_spinna_batch_config``) as one
        ``exp_data_<tag>`` column per channel, so the batch analysis
        runs on the locs produced by this workflow.

        File-path columns of the config csv (``structures_filename``,
        ``exp_data_*`` and ``mask_filename_*``) are converted to the
        current machine using the Drivepaths config. The modified
        config is written to a copy inside the module's results folder
        -- the user's original csv is not changed -- and that copy is
        passed on to picasso's batch analysis.

        The config csv must already be prepared by the user; only the
        ``exp_data_*`` columns are filled in here. See
        ``picasso.__main__._spinna_batch_analysis`` for the columns
        expected in the config file.

        Parameters
        ----------
        i : int
            the index of the module
        parameters : dict
            with required keys:
                fp_spinna_batch_config : str
                    path to the user-prepared spinna batch
                    analysis config csv file.
            with optional keys:
                use_workflow_locs : bool
                    whether to use the locs previously processed in
                    this workflow, otherwise those specified in the
                    csv. Default: False
        results : dict
            the results this function generates. This is created
            in the decorator wrapper
        """
        parameters_spec = {
            "fp_spinna_batch_config": {
                "type": "path",
                "description": "Path to the spinna batch analysis config file.",
                "required": True,
            },
            "use_workflow_locs": {
                "type": "bool",
                "description": "Use locs from workflow, otherwise as specified in csv.",
                "required": False,
                "default": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "fp_spinna_batch_config": {
                "type": "str",
                "description": "Path to the config csv copy actually used",
            },
            "result_dir": {
                "type": "str",
                "description": "Folder containing the SPINNA batch results",
            },
            "fp_summary": {
                "type": "str",
                "description": "Filepath of the SPINNA summary csv file",
            },
            "fp_figs": {
                "type": "list",
                "element_type": "str",
                "description": "Filepaths of the NND figures",
            },
        }

        return parameters_spec, results_spec

    def ripleysk(self):
        """Perforn Ripley's K analysis between the channels using
        Magdalena's code.

        Parameters
        ----------
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
        parameters_spec = {
            "radii": {
                "type": "list",
                "description": "List of analysis radii in nm",
                "element_type": "float",
                "min": 1.0,
                "max": 1000.0,
                "required": True,
            },
            "edge_correction": {
                "type": "str",
                "description": "Edge correction method",
                "options": ["none", "translation", "isotropic"],
                "default": "translation",
                "required": False,
            },
            "ripleys_n_random_controls": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of random controls, default: 100
                # Hint: type appears to be int
            },
            "ripleys_rmax": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the maximum radius, default 200
                # Hint: type appears to be int
            },
            "ripleys_dr": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the radius interval, default 5
                # Hint: type appears to be float
            },
            "ripleys_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the threshold of ripleys integrals above which the interaction is deemed significant.
                # Hint: type appears to be float
            },
            "fp_combined_locs": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: filepath to the combined locs of all channel_locs
                # Hint: type appears to be str
            },
            "atype": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the type of analysis: 'Ripleys' for the standard Ripley's K analysis, or 'RDF' for calculation of the radial distribution function instead of K, an...
                # Hint: type appears to be str
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "ripley_k_values": {
                "type": "numpy.ndarray",
                "description": "Calculated Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def ripleysk2(self):
        """Perforn Ripley's K analysis between the channels using
        Rafal's code.

        Parameters
        ----------
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
        parameters_spec = {
            "radii": {
                "type": "list",
                "description": "List of analysis radii in nm",
                "element_type": "float",
                "min": 1.0,
                "max": 1000.0,
                "required": True,
            },
            "algorithm": {
                "type": "str",
                "description": "Algorithm variant to use",
                "options": ["fast", "accurate", "parallel"],
                "default": "fast",
                "required": False,
            },
            "ripleys_n_random_controls": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of random controls, default: 100
                # Hint: type appears to be int
            },
            "ripleys_rmax": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the maximum radius, default 200
                # Hint: type appears to be int
            },
            "ripleys_dr": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the radius interval, default 5
                # Hint: type appears to be float
            },
            "ripleys_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the threshold of ripleys integrals above which the interaction is deemed significant.
                # Hint: type appears to be float
            },
            "area": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the cell area in µm^2 optional. only used with controltype=CSR
                # Hint: type appears to be float
            },
            "fp_mask": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the filepath to the cell mask. optional, only used with CSR. can be binary or density mask
                # Hint: type appears to be str
            },
            "mask_pixel_size": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the pixel size of mask pixels (move to mask class which internally keeps this information) optional, only used with controltype=CSR
                # Hint: type appears to be float
            },
            "metric": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the type of analysis: 'RK' for the standard Ripley's K analysis, or 'RDF' for calculation of the radial distribution function instead of K, and ran...
                # Hint: type appears to be str
            },
            "controltype": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: CSR simulation within the density mask, or randomizing the real data
                # Hint: type appears to be str
            },
            "randomization_radius": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: for controltype \"RND\", the radius [nm] by which to randomize. optional.
                # Hint: type appears to be float
            },
            "shuffle_self": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: for metric \"FRC\", whether to shuffle only other types or also the self type
                # Hint: type appears to be bool
            },
            "relocate_self": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: for metric \"FRC\", whether to relocate centerpoints to fraction_exclude
                # Hint: type appears to be bool
            },
            "significance_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: threshold above which heatmap entries are colored
                # Hint: type appears to be float
            },
            "normalization": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be str
            },
            "edge_correction": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: if True, only locs further from mask edges than max radius are used for evaluation
                # Hint: type appears to be bool
            },
            "showControlEnvelope": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be bool
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "ripley_k2_values": {
                "type": "numpy.ndarray",
                "description": "Calculated Ripley's K values (variant 2)",
            },
        }

        return parameters_spec, results_spec

    def ripleysk_average(self):
        """Average the results of multiple Ripley's K Analyses, analyse
        the significant pairs after averaging, and save them into the
        separate workflow manual folders (for further analysis there)

        Parameters
        ----------
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
        parameters_spec = {
            "weight_method": {
                "type": "str",
                "description": "Method for weighting individual analyses",
                "options": ["equal", "by_nlocs", "by_area"],
                "default": "equal",
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
            },
            "ripleys_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the threshold of ripleys integrals above which the interaction is deemed significant.
                # Hint: type appears to be float
            },
            "atype": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: # output_folders : list of str #     folders to write the significant pairs into. This can #     e.g. be the 'manual' results folders of the #     ...
                # Hint: type appears to be str
            },
            "swkfl_ripleysk_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the ripleysk module. e.g. '05_ripleysk'
                # Hint: type appears to be str
            },
            "swkfl_manual_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the manual module to save the integrals to if those two are not given, saving is not performed
                # Hint: type appears to be str
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "average_ripley_k": {
                "type": "numpy.ndarray",
                "description": "Averaged Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def ripleysk_average2(self):
        """Average the results of multiple Ripley's K Analyses, analyse
        the significant pairs after averaging, and save them into the
        separate workflow manual folders (for further analysis there)

        Parameters
        ----------
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
        parameters_spec = {
            "statistical_method": {
                "type": "str",
                "description": "Statistical method for averaging",
                "options": ["mean", "median", "trimmed_mean"],
                "default": "mean",
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
            },
            "ripleys_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the threshold of ripleys integrals above which the interaction is deemed significant.
                # Hint: type appears to be float
            },
            "metric": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the type of analysis: 'RK' for the standard Ripley's K analysis, or 'RDF' for calculation of the radial distribution function instead of K, and ran...
                # Hint: type appears to be str
            },
            "controltype": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: CSR simulation within the density mask, or randomizing the real data
                # Hint: type appears to be str
            },
            "randomization_radius": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: for controltype \"RND\", the radius [nm] by which to randomize. # output_folders : list of str #     folders to write the significant pairs into. Thi...
                # Hint: type appears to be float
            },
            "swkfl_ripleysk_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the ripleysk module. e.g. '05_ripleysk'
                # Hint: type appears to be str
            },
            "swkfl_manual_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the manual module to save the integrals to if those two are not given, saving is not performed
                # Hint: type appears to be str
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "alternative_average_k": {
                "type": "numpy.ndarray",
                "description": "Alternative averaged Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def protein_interactions(self):
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

        Parameters
        ----------
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
        parameters_spec = {
            "interaction_radius": {
                "type": "float",
                "description": "Maximum interaction distance in nm",
                "min": 1.0,
                "max": 500.0,
                "default": 50.0,
                "required": True,
            },
            "confidence_level": {
                "type": "float",
                "description": "Statistical confidence level",
                "min": 0.5,
                "max": 0.99,
                "default": 0.95,
                "required": False,
            },
            "channel_map": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: maps between channels (protein names, tags before combining) and index in the combine_id column of combined locs labeling_efficiency : dict, channe...
                # Hint: type appears to be dict
            },
            "labeling_efficiency": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be dict
            },
            "labeling_uncertainty": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be dict
            },
            "n_simulate": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of target molecules to be simulated; good value is e.g. 50000 density : dict, channel tag to float density to simulate [nm^2 or nm^3]; area ...
                # Hint: type appears to be int
            },
            "density": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be dict
            },
            "nn_nth": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of nearest neighbors to analyse
                # Hint: type appears to be int
            },
            "structure_distance": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the protein distance between each other in nm
                # Hint: type appears to be float
            },
            "res_factor": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the spinna res_factor
                # Hint: type appears to be float
            },
            "sim_repeats": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of simulation repeats, for noise reduction interaction_pairs: list of list of two strings, or str pairs that are able to interact
                # Hint: type appears to be int
            },
            "interaction_pairs": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be list
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "interaction_map": {
                "type": "numpy.ndarray",
                "description": "Spatial interaction map",
            },
        }

        return parameters_spec, results_spec

    def protein_interactions_average(self):
        """Average the results of multiple "protein_interactions" analyses.
        Create a bar plot with mean and stddev of the different proportions
        of interaction partners.

        Parameters
        ----------
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
        parameters_spec = {
            "normalization_method": {
                "type": "str",
                "description": "Method for normalizing interactions",
                "options": ["by_area", "by_density", "none"],
                "default": "by_area",
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
            },
            "swkfl_protint_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the protein interactions module. e.g. '05_protein_interactions' optional:
                # Hint: type appears to be str
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "average_interactions": {
                "type": "dict",
                "description": "Averaged interaction statistics",
            },
        }

        return parameters_spec, results_spec

    def create_mask(self):
        """
        This is Susanne's implementation of calculating a cell mask,
        written (ni part?) for the initial version of the DC-Atlas.
        May be obsolete with create_mask2, but kept for backwards
        compatibility. To be deprecated on the long run.

        Parameters
        ----------
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
        parameters_spec = {
            "threshold": {
                "type": "float",
                "description": "Density threshold for mask creation",
                "min": 0.0,
                "max": 1000.0,
                "required": True,
            },
            "smoothing_radius": {
                "type": "float",
                "description": "Radius for density smoothing",
                "min": 1.0,
                "max": 200.0,
                "default": 20.0,
                "required": False,
            },
            "fp_channel_map": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: filepath to the map from 'combine_channels' module, which is a dict from channel name to ID int in the locs['combine_id']
                # Hint: type appears to be str
                # Hint: required = True
            },
            "fp_combined_locs": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: filepath to the locs combined in 'combine_channels' module
                # Hint: type appears to be str
                # Hint: required = True
            },
            "margin": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Size of the added empty margin to the FOV, in nm
                # Hint: type appears to be float
                # Hint: required = True
            },
            "binsize": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Size o fthe 2D histogram bins of the first step, in nm
                # Hint: type appears to be float
                # Hint: required = True
            },
            "sigma_mask_blur": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: parameter of the gaussian blur in binsize units
                # Hint: type appears to be int
                # Hint: required = True
            },
            "mask_resolution": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Controls the digital resolution of the mask, in nm
                # Hint: type appears to be float
                # Hint: required = True
            },
            "combine_col": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the name of the combine column, e.g. 'combine_id' or 'protein'. Same as used in 'combine_channels' module
                # Hint: type appears to be str
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "mask": {
                "type": "numpy.ndarray",
                "description": "Generated density mask",
            },
        }

        return parameters_spec, results_spec

    def create_mask2(self):
        """
        This is Rafal's implementation of cell masking, written for the
        3rd version of the DC Atlas. It is (mostly?) identical with an
        implementation of it in spinna, which will be integrated into
        picasso soon. Evaluate deprecation (or moving source from
        outpost_modules/ripleys to picasso/spinna) at that time.

        the locs must be protein positions at this stage.

        Parameters
        ----------
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
        parameters_spec = {
            "algorithm": {
                "type": "str",
                "description": "Mask creation algorithm",
                "options": ["threshold", "otsu", "adaptive"],
                "required": True,
            },
            "parameters_dict": {
                "type": "dict",
                "description": "Algorithm-specific parameters",
                "required": False,
            },
            "binsize": {
                "type": "float",
                "description": "The bin size in nanometers.",
                "default": 20,
                "min": 0,
                "required": True,
            },
            "blursize": {
                "type": "float",
                "description": "The gaussian blur to apply in nanometers.",
                "min": 0,
                "default": 400,
                "required": True,
            },
            "mask_pixel_size": {
                "type": "float",
                "description": "The pixelsize of the final mask, in \
                    nanometers.",
                "min": 0,
                "required": True,
            },
            "threshold": {
                "type": "float",
                "description": "The threshold value below which the mask \
                    is set to zero.",
                "min": 0,
                "default": 0.333,
                "required": True,
            },
            "binary": {
                "type": "bool",
                "description": "Whether to create a binary or density mask",
                "required": True,
            },
            "select_cell": {
                "type": "bool",
                "description": "Whether to select the nth largest connected \
                    component, assumed to be the cell of interest.",
                "required": True,
            },
            "nth_largest_cell": {
                "type": "int",
                "description": "If select_cell is True: Select the nth \
                    largest cell by area (1 = largest, 2 = second \
                    largest, ...).",
                "default": 1,
                "min": 1,
                "required": True,
            },
            "fill_holes": {
                "type": "bool",
                "description": "Whether to fill holes in the cell mask",
                "required": True,
            },
            "dilate_nm": {
                "type": "float",
                "description": "The nanometers to dilate the mask (useful \
                    if a large threshold has been used)",
                "required": True,
            },
            "apply_to_locs": {
                "type": "bool",
                "description": "Whether to drop all localizations outside \
                    the masked area.",
                "required": True,
            },
            "fp_combined_locs": {
                "type": "path",
                "description": "Filepath to the locs combined previously \
                    in 'combine_channels' module. If None or '', \
                    loaded channel_locs is used",
                "required": False,
            },
            "fp_channel_map": {
                "type": "path",
                "description": "Filepath to the map from 'combine_channels' \
                    module, which is a dict from channel name to ID int in \
                    the locs['combine_id']",
                "required": False,
            },
            "combine_col": {
                "type": "str",
                "description": "The name of the combine column, e.g. \
                    'combine_id' or 'protein'. Same as used in \
                    'combine_channels' module",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "fp_mask": {
                "type": "Path",
                "description": "Path to the saved mask",
            },
            "area": {
                "type": "float",
                "description": "area in µm^2",
            },
        }

        return parameters_spec, results_spec

    def refine_mask_by_density(self):
        """
        This module analyses and refines a previously created mask for
        even signal.
        Particularly, the density histogram of the mask bins are plotted,
        and an area of homogeneous density can be selected

        The locs must be protein positions at this stage.

        Parameters
        ----------
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
                    1-based: set 1 for largest.
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
        parameters_spec = {
            # "density_range": {
            #     "type": "tuple",
            #     "description": "Min and max density values",
            #     "length": 2,
            #     "element_type": "float",
            #     "required": True,
            # },
            # "refinement_method": {
            #     "type": "str",
            #     "description": "Method for mask refinement",
            #     "options": ["erosion", "dilation", "opening", "closing"],
            #     "default": "opening",
            #     "required": False,
            # },
            "fp_mask": {
                "type": "path",
                "description": "The file path to the mask to refine.",
                "required": True,
            },
            "density_std_cutoff": {
                "type": "float",
                "description": "Density range in units of std/median, "
                "symmetric around median. Alternative to max/min",
                "min": 0,
                "default": 0,
                "required": False,
            },
            "min_density": {
                "type": "float",
                "description": "Lower density cutoff in µm^(-2). "
                "Alternative to density_std_cutoff",
                "default": 0,
                "min": 0,
                "required": False,
            },
            "max_density": {
                "type": "float",
                "description": "Higher density cutoff in µm^(-2). "
                "Alternative to density_std_cutoff",
                "default": 0,
                "min": 0,
                "required": False,
            },
            "nbins": {
                "type": "int",
                "description": "The number of bins for plotting",
                "required": False,
            },
            "nth_largest": {
                "type": "int",
                "description": "Select the nth largest contiguous area in density "
                "range (1 = largest, 2 = second largest, ...).",
                "required": False,
                "min": 1,
                "default": 1,
            },
            "apply_to_locs": {
                "type": "bool",
                "description": "Whether to apply the created mask to the \
                    locs.",
                "required": False,
            },
            "smoothe_nm": {
                "type": "float",
                "description": "The distance in nanometers to dilate and erode\
                    the mask. This can be useful to remove excessive holes and\
                    ragging in the mask due to the density thre...",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "refined_mask": {
                "type": "numpy.ndarray",
                "description": "Refined density mask",
            },
        }

        return parameters_spec, results_spec

    def dbscan_molint(self):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow

        Parameters
        ----------
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
        parameters_spec = {
            "radius": {
                "type": "float",
                "description": "DBSCAN radius for interaction clustering",
                "min": 1.0,
                "max": 500.0,
                "default": 30.0,
                "required": True,
            },
            "min_samples": {
                "type": "int",
                "description": "Minimum samples for interaction clusters",
                "min": 1,
                "max": 50,
                "default": 3,
                "required": True,
            },
            "interaction_threshold": {
                "type": "float",
                "description": "Distance threshold for interactions",
                "min": 1.0,
                "max": 200.0,
                "default": 50.0,
                "required": False,
            },
            "fp_channel_map": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: filepath to the map from 'combine_channels' module, which is a dict from channel name to ID int in the locs['combine_id']
                # Hint: type appears to be str
                # Hint: required = True
            },
            "epsilon_nm": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: dbscan epsilon in nm
                # Hint: type appears to be float
                # Hint: required = True
            },
            "minpts": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: minimum number of points
                # Hint: type appears to be int
                # Hint: required = True
            },
            "sigma_linker": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: ... in nm
                # Hint: type appears to be float
                # Hint: required = True
            },
            "fp_merge_mask": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: filepath to the merge mask (generated in module
                # Hint: type appears to be str
                # Hint: required = True
            },
            "thresh_type": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: ...
                # Hint: type appears to be str
                # Hint: required = True
            },
            "cell_name": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the name of the cell currently analyzed
                # Hint: type appears to be str
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "interaction_clusters": {
                "type": "numpy.ndarray",
                "description": "Clustered interaction data",
            },
        }

        return parameters_spec, results_spec

    def CSR_sim_in_mask(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask, and perform dbscan as well

        Parameters
        ----------
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
        parameters_spec = {
            "n_simulations": {
                "type": "int",
                "description": "Number of CSR simulations",
                "min": 10,
                "max": 10000,
                "default": 1000,
                "required": True,
            },
            "seed": {
                "type": "int",
                "description": "Random seed for reproducibility",
                "min": 0,
                "max": 2**32 - 1,
                "required": False,
            },
            "fp_channel_map": {
                "type": "path",
                "description": "Filepath to the map from \
                    'combine_channels' module, which is a dict from \
                    channel name to ID int in the locs['combine_id']",
                "required": True,
            },
            "fp_mask_dict": {
                "type": "path",
                "description": "Filepath to the mask_dict.pkl file \
                    generated in the 'create_mask' module",
                "required": True,
            },
            "N_repeats": {
                "type": "int",
                "description": "Number of simulation repeats",
                "required": True,
            },
            "epsilon_nm": {
                "type": "float",
                "description": "dbscan epsilon in nm",
                "required": True,
            },
            "minpts": {
                "type": "int",
                "description": "minimum number of points",
                "required": True,
            },
            "sigma_linker": {
                "type": "float",
                "description": "... in nm",
                "required": True,
            },
            "fp_merge_mask": {
                "type": "str",
                "description": "Filepath to the merge mask \
                    (generated in module)",
                "required": True,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "csr_simulations": {
                "type": "list",
                "description": "Generated CSR simulation data",
                "element_type": "numpy.ndarray",
            },
        }

        return parameters_spec, results_spec

    def find_cluster_motifs(self):
        """Analyses the binary barcode results of _do_dbscan_molint.
        Compares experimental to CSR data.
        Merged for multiple cells

        Parameters
        ----------
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
        parameters_spec = {
            "motif_size": {
                "type": "int",
                "description": "Size of motifs to search for",
                "min": 2,
                "max": 20,
                "default": 5,
                "required": True,
            },
            "similarity_threshold": {
                "type": "float",
                "description": "Threshold for motif similarity",
                "min": 0.0,
                "max": 1.0,
                "default": 0.8,
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
                # Hint: required = True
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
                # Hint: required = True
            },
            "swkfl_dbscan_molint_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the dbscan module. e.g. '09_dbscan_molint'
                # Hint: type appears to be str
                # Hint: required = True
            },
            "swkfl_CSR_sim_in_mask_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the CSR dbscan module. e.g. '10_CSR_sim_in_mask' population_threshold : float, 0 - 1 only select barcodes with a relative popula...
                # Hint: type appears to be str
                # Hint: required = True
            },
            "population_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
                # Hint: required = True
            },
            "ttest_pvalue_max": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
                # Hint: required = True
            },
            "channel_colors": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: colors to describe the receptors with
                # Hint: type appears to be list
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "identified_motifs": {
                "type": "list",
                "description": "List of identified cluster motifs",
                "element_type": "dict",
            },
        }

        return parameters_spec, results_spec

    def interaction_graph(self):
        """Plot the interaction graph, displaying the different targets
        and their interactions in a graph. The node sizes denote the
        density, and the ripley interaction matrix is represented in the
        edges.

        Parameters
        ----------
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
        parameters_spec = {
            "graph_type": {
                "type": "str",
                "description": "Type of interaction graph",
                "options": ["adjacency", "distance", "correlation"],
                "required": True,
            },
            "edge_threshold": {
                "type": "float",
                "description": "Threshold for graph edges",
                "min": 0.0,
                "max": 1000.0,
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
                # Hint: required = True
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
                # Hint: required = True
            },
            "swkfl_protint_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the protein_interactions module. e.g. '09_protein_interactions'
                # Hint: type appears to be str
                # Hint: required = True
            },
            "fp_density": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: fp to the denfsities of the channels.
                # Hint: type appears to be str
                # Hint: required = True
            },
            "fp_ripleys_meanvals": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the filepath to the interaction matrix
                # Hint: type appears to be str
                # Hint: required = True
            },
            "edge_factor": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: factor to display useful sizes
                # Hint: type appears to be float
                # Hint: required = True
            },
            "node_factor": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: factor to display useful sizes
                # Hint: type appears to be float
                # Hint: required = True
            },
            "channel_colors": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: colors to describe the receptors with
                # Hint: type appears to be list
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "interaction_graph": {
                "type": "dict",
                "description": "Generated interaction graph data",
            },
        }

        return parameters_spec, results_spec

    def plot_densities(self):
        """Aggregate densities and cell areas of multiple datasets and
        plot them

        Parameters
        ----------
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
        parameters_spec = {
            "plot_type": {
                "type": "str",
                "description": "Type of density plot",
                "options": ["heatmap", "contour", "3d"],
                "default": "heatmap",
                "required": False,
            },
            "color_map": {
                "type": "str",
                "description": "Color map for visualization",
                "options": ["viridis", "plasma", "hot", "jet"],
                "default": "viridis",
                "required": False,
            },
            "fp_workflows": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the paths to the folders of separate workflows where the separate ripleys analyses have been done
                # Hint: type appears to be list
                # Hint: required = True
            },
            "report_names": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the report names of those worklfows
                # Hint: type appears to be list
                # Hint: required = True
            },
            "swkfl_create_mask_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the dbscan module. e.g. '11_create_mask'
                # Hint: type appears to be str
                # Hint: required = True
            },
            "swkfl_protint_key": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the results key of the protein_interactions module. e.g. '09_protein_interactions'
                # Hint: type appears to be str
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "density_plots": {
                "type": "list",
                "description": "Generated density plot file paths",
                "element_type": "str",
            },
        }

        return parameters_spec, results_spec

    def find_gold(self):
        """Find localizations stemming from gold beads based on blinking
        kinetics.
        The metrics used are number of locs and rms deviation from mean
        frame

        Parameters
        ----------
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
        parameters_spec = {
            "nlocs_threshold": {
                "type": "int",
                "description": (
                    "Minimum number of localizations for gold detection"
                ),
                "min": 10,
                "max": 10000,
                "default": 100,
                "required": True,
            },
            "rms_threshold": {
                "type": "float",
                "description": "Maximum RMS deviation threshold",
                "min": 0.1,
                "max": 10.0,
                "default": 2.0,
                "required": True,
            },
            "frame_window": {
                "type": "int",
                "description": "Frame window for kinetics analysis",
                "min": 10,
                "max": 1000,
                "default": 100,
                "required": False,
            },
            "remove_gold": {
                "type": "bool",
                "description": "If present and set to True, the gold \
                    locs are discarded and self.locs is set to the \
                    nongold-locs",
                "required": False,
            },
            "diameter": {
                "type": "float",
                "description": "The pick similar diameter for \
                    identifying gold std_range, mean_rmsd : float \
                    the pick similar parameters identifying gold",
                "min": 0.0,
                "default": 2.0,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "gold_locs": {
                "type": "numpy.ndarray",
                "description": "Identified gold bead localizations",
            },
            "fp_gold": {
                "type": "str",
                "description": "filepath to the gold locs found",
            },
            "fp_nogold": {
                "type": "str",
                "description": "Filepath to the non-gold locs",
            },
        }

        return parameters_spec, results_spec

    def find_similar(self):
        """pick similar in nlocs/rmsd space (with specified limits in
        that space).

        Parameters
        ----------
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
        parameters_spec = {
            "reference_structure": {
                "type": "int",
                "description": "Index of reference structure",
                "min": 0,
                "required": True,
            },
            "similarity_metric": {
                "type": "str",
                "description": "Metric for similarity calculation",
                "options": ["euclidean", "cosine", "correlation"],
                "required": True,
            },
            "tolerance": {
                "type": "float",
                "description": "Tolerance for similarity matching",
                "min": 0.0,
                "max": 1.0,
                "default": 0.1,
                "required": False,
            },
            "diameter": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the pick similar diameter for identifying gold
                # Hint: type appears to be float
                # Hint: required = True
            },
            "min_n_locs_per_frame": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "max_n_locs_per_frame": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "min_rmsd": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the minimum root mean square distance from pick center to pick
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "max_rmsd": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the maximum root mean square distance from pick center to pick
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "n_plot_structures": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the number of structures to plot
                # Hint: type appears to be int
                # Hint: required = False (optional)
            },
            "display_pixelsize": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the pixelsize for display in nm, default: 1
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "similar_structures": {
                "type": "list",
                "description": "List of similar structure indices",
                "element_type": "int",
            },
        }

        return parameters_spec, results_spec

    def find_structures(self):
        """pick similar on clusters in nlocs/rmsd space.
        This may be useful for automated picking of origamis, and may
        help for defining parameters for finding gold

        Parameters
        ----------
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
        parameters_spec = {
            "cluster_method": {
                "type": "str",
                "description": "Method for structure clustering",
                "options": ["kmeans", "hierarchical", "spectral"],
                "required": True,
            },
            "n_structures": {
                "type": "int",
                "description": "Number of structure types to identify",
                "min": 1,
                "max": 20,
                "default": 5,
                "required": False,
            },
            "diameter": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the pick similar diameter for identifying gold
                # Hint: type appears to be float
                # Hint: required = True
            },
            "min_n_locs_per_frame": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the percentage of frames with events in the pick region below which there is noise. default: 0.01
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "n_plot_structures": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the number of structures to plot
                # Hint: type appears to be int
                # Hint: required = False (optional)
            },
            "display_pixelsize": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the pixelsize for display in nm, default: 1
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "xi": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the xi parameter for clustering. default 0.05
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "min_cluster_size": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the minimun cluster size (fract). default .05
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "structure_types": {
                "type": "list",
                "description": "Identified structure type classifications",
                "element_type": "int",
            },
        }

        return parameters_spec, results_spec

    def undrift_from_picked(self):
        """Performs undrift from piced locs.

        Parameters
        ----------
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
        parameters_spec = {
            "fp_picked_locs": {
                "type": ["numpy.ndarray", "str"],
                "description": "Picked localization coordinates or file path",
                "extensions": [".hdf5", ".txt"],
                "required": True,
            },
            "interpolation_method": {
                "type": "str",
                "description": "Method for drift interpolation",
                "options": ["linear", "spline", "polynomial"],
                "default": "spline",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "drift_corrected": {
                "type": "numpy.ndarray",
                "description": "Drift correction vectors",
            },
        }

        return parameters_spec, results_spec

    def filter_locs(self):
        """Filter localizations to lie within a min-max range of a metric.

        Parameters
        ----------
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
        parameters_spec = {
            "field": {
                "type": ["str", "list"],
                "description": "Field to filter by. One or list of columns in locs"
                " (e.g. 'photons', 'x', 'y', 'sx', 'sy')",
                "required": True,
            },
            "minval": {
                "type": ["float", "list"],
                "description": "Minimum filter cutoff value",
                "required": True,
            },
            "maxval": {
                "type": ["float", "list"],
                "description": "Maximum filter cutoff value",
                "required": True,
            },
            # "invert_filter": {
            #     "type": "bool",
            #     "description": "Whether to invert the filter logic",
            #     "default": False,
            #     "required": False,
            # },
            "mode": {
                "type": "str",
                "description": "Metric to apply filter values to",
                "options": [
                    "absolute",
                    "zscore",
                    "quantile",
                ],
                "default": "absolute",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "n_filtered": {
                "type": "int",
                "description": "Number of localizations after filtering",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def filter_transient_binding(self):
        """Filter molecule positions (after clustering or Gaussian Mixture)
        for those who show transient binding. Specifically, the mean frame
        should not be at extreme positions
        (default, 0.1 > mean frame / nframes > 0.9), and std of frames
        (default: 0.3 > std frame).

        Parameters
        ----------
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
        parameters_spec = {
            # "frame_percentiles": {
            #     "type": "tuple",
            #     "description": "Min and max frame percentiles for filtering",
            #     "length": 2,
            #     "element_type": "float",
            #     "min": 0.0,
            #     "max": 100.0,
            #     "default": [10.0, 90.0],
            #     "required": True,
            # },
            # "binding_duration_range": {
            #     "type": "tuple",
            #     "description": "Min and max binding duration range",
            #     "length": 2,
            #     "element_type": "float",
            #     "required": False,
            # },
            "meanframe_cutoff": {
                "type": "float",
                "description": "filter out positions at more extreme temporal positions",
                "min": 0,
                "max": 1,
                "default": 0.1,
                "required": False,
            },
            "stdframe_cutoff": {
                "type": "float",
                "description": "filter out positions with lower standard deviation",
                "min": 0,
                # "max": 1,
                "default": 0.16,
                "required": False,
            },
            "fp_locs": {
                "type": "path",
                "description": "The filepath to localizations (self.locs should be the"
                " centers). If given, locs are filtered and saved as well.",
                "required": False,
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the filepath to the underlying localizations (self.locs are centers). If given, these are filtered as well and saved with the same filename in the ...
                # Hint: type appears to be str
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "transient_events": {
                "type": "numpy.ndarray",
                "description": "Filtered transient binding events",
            },
        }

        return parameters_spec, results_spec

    def link_locs(self):
        """Link localizations.

        Parameters
        ----------
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
        parameters_spec = {
            "max_distance": {
                "type": "float",
                "description": "Maximum linking distance in nm",
                "min": 1.0,
                "max": 1000.0,
                "default": 100.0,
                "required": True,
            },
            "max_frame_gap": {
                "type": "int",
                "description": "Maximum frame gap for linking",
                "min": 1,
                "max": 100,
                "default": 5,
                "required": True,
            },
            "linking_algorithm": {
                "type": "str",
                "description": "Algorithm for localization linking",
                "options": ["nearest_neighbor", "hungarian", "lap"],
                "default": "hungarian",
                "required": False,
            },
            "d_max": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: maximum distance to link [px]
                # Hint: type appears to be int
                # Hint: required = True
            },
            "tolerance": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: maximum transient dark time [frames]
                # Hint: type appears to be int
                # Hint: required = True
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "linked_trajectories": {
                "type": "list",
                "description": "Linked localization trajectories",
                "element_type": "numpy.ndarray",
            },
        }

        return parameters_spec, results_spec

    def pairwise_module_executor(self):
        """Calls another module (as a sub-module) for all pairs in the
        channel_locs

        Parameters
        ----------
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
        parameters_spec = {
            "sub_module": {
                "type": "str",
                "description": "Name of the sub-module to execute",
                "required": True,
            },
            "sub_module_params": {
                "type": "dict",
                "description": "Parameters for the sub-module",
                "required": False,
            },
            "module_name": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the module to call
                # Hint: type appears to be str
                # Hint: required = True
            },
            "param_target1": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: parameter name of the first target to set for the module
                # Hint: type appears to be str
                # Hint: required = True
            },
            "param_target2": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: parameter name of the second target to set for the module
                # Hint: type appears to be str
                # Hint: required = True
            },
            "module_kwargs": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the other arguments to the module
                # Hint: type appears to be dict
                # Hint: required = True
            },
            "result_scalar": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the key to display in a heatmap as main result
                # Hint: type appears to be str
                # Hint: required = False (optional)
            },
            "scalar_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the saturation value in the heatmap
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "scalar_minval": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the minimum value for color in the heatmap
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "result_fpfig": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: the key to the filepath of one or more figures generated to display for documentation
                # Hint: type appears to be str or list
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "pairwise_results": {
                "type": "dict",
                "description": "Results from pairwise module execution",
            },
        }

        return parameters_spec, results_spec

    def random_val(self):
        """Generate random values and plot for debugging and testing the
        pairwise module.

        Creates a random value and generates a test plot with random data
        for debugging purposes in pairwise module workflows.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            Input parameters (unchanged)
        results : dict
            Updated results dictionary with random value and figure path
        """
        parameters_spec = {
            "xlabel": {
                "type": "str",
                "description": "Label for the x-axis of the test plot",
                "required": True,
            },
            "ylabel": {
                "type": "str",
                "description": "Label for the y-axis of the test plot",
                "required": True,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "random_val": {
                "type": "float",
                "description": "A random value between 0 and 1",
            },
            "fp_fig": {
                "type": "str",
                "description": "Filepath to the generated test figure",
            },
        }

        return parameters_spec, results_spec

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

        Parameters
        ----------
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
                granularity : int
                    the spinna granularity
                sim_repeats : int
                    number of simulation repeats, for noise reduction
            and optional keys:
                nn_nth : int
                    number of nearest neighbors to analyse
                    default: 1
                NND_bin : int
                    bin size (nm)
                    auto-calculated if None or 0
                NND_maxdist : int
                    maximum distance in histogram (nm)
                    auto-calculated if None or 0
        results : dict
            the results this function generates. This is created
            in the decorator wrapper
        """
        parameters_spec = {
            "target_name": {
                "type": "str",
                "description": "Name of target molecular species",
                "required": True,
            },
            "reference_name": {
                "type": "str",
                "description": "Name of reference species",
                "required": True,
            },
            # "efficiency_model": {
            #     "type": "str",
            #     "description": "Model for efficiency calculation",
            #     "options": ["poisson", "binomial", "maximum_likelihood"],
            #     "default": "maximum_likelihood",
            #     "required": False,
            # },
            "pair_distance": {
                "type": "int",
                "description": "Target-Reference pair distance [nm]",
                "default": 10,
                "required": True,
            },
            "labeling_uncertainty": {
                "type": "dict",
                "description": "Dictionary mapping from target/reference (tag) "
                "to labeling uncertainty [nm]",
                # "default": 5,
                "required": True,
            },
            "n_simulate": {
                "type": "int",
                "description": "Number of structures to simulate",
                "default": 500000,
                "required": True,
            },
            "density": {
                "type": "dict",
                "description": "Dictionary mapping from target/reference (tag) "
                "to molecular density [nm^-2 or nm^-3]",
                # "default": 5,
                "required": True,
            },
            "granularity": {
                "type": "int",
                "description": "The spinna granularity",
                "default": 100,
                "required": True,
            },
            "sim_repeats": {
                "type": "int",
                "description": "Number of simulation iterations",
                "default": 10,
                "required": True,
            },
            "nn_nth": {
                "type": "int",
                "description": "Neighbor up to which to evaluate",
                "default": 2,
                "required": False,
            },
            "NND_bin": {
                "type": "int",
                "description": "Bin size (nm). Auto-calculated if None or 0",
                "default": 0,
                "required": False,
            },
            "NND_maxdist": {
                "type": "int",
                "description": "Maximum distance in histogram (nm). Auto-calculated if None or 0.",
                "default": 0,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "labeling_efficiency": {
                "type": "float",
                "description": "Calculated labeling efficiency",
                "min": 0.0,
                "max": 1.0,
            },
        }

        return parameters_spec, results_spec

    def conditional_branch(self):
        """Execute different sub-module sequences based on a condition.

        Parameters
        ----------
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

        Returns
        -------
        parameters : dict
            as input, potentially changed values, for consistency
        results : dict
            the analysis results including:
                - condition_result : bool
                - branch_taken : str ("if_true" or "if_false")
                - if_branch : dict of sub-module results
                - branch_modules : dict of flat-indexed results
        """
        parameters_spec = {
            "xlabel": {
                "type": "str",
                "description": "Label for the x-axis of the test plot",
                "required": True,
            },
            "ylabel": {
                "type": "str",
                "description": "Label for the y-axis of the test plot",
                "required": True,
            },
            "condition": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: condition dictionary with keys: or logical condition with \"and\"/\"or\" keys
                # Hint: type appears to be dict
                # Hint: required = True
            },
            "if_true": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: list of (module_name, module_parameters) tuples to execute if condition is True
                # Hint: type appears to be list
                # Hint: required = True
            },
            "if_false": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: list of (module_name, module_parameters) tuples to execute if condition is False
                # Hint: type appears to be list
                # Hint: required = True
            },
            "parameter_command_executor": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: if provided, will be used for resolving parameter commands in condition values
                # Hint: type appears to be ParameterCommandExecutor
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "random_val": {
                "type": "float",
                "description": "A random value between 0 and 1",
            },
            "fp_fig": {
                "type": "str",
                "description": "Filepath to the generated test figure",
            },
        }

        return parameters_spec, results_spec

    def resolution_analysis(self):
        """Perform resolution analysis using point pattern autocorrelation

        This method calculates the spatial resolution of localizations
        by computing a 2D autocorrelation function and fitting a Gaussian to
        extract resolution metrics. The analysis includes 2D Gaussian fitting,
        radial profile computation, and 1D Gaussian fitting to the radial profile.

        Parameters
        ----------
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
        parameters_spec = {
            "xlabel": {
                "type": "str",
                "description": "Label for the x-axis of the test plot",
                "required": True,
            },
            "ylabel": {
                "type": "str",
                "description": "Label for the y-axis of the test plot",
                "required": True,
            },
            "delta_r": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: grid spacing for autocorrelation (default: 5 nm)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "r_max": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: maximum radius for autocorrelation (default: 100 nm)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "batch_size": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of data points per batch for chunking (auto-calculated if None)
                # Hint: type appears to be int or None
                # Hint: required = False (optional)
            },
            "n_processes": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of parallel processes (auto-detected if None, capped at 4)
                # Hint: type appears to be int or None
                # Hint: required = False (optional)
            },
            "use_chunking": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: enable memory-efficient chunking for large datasets (default: True)
                # Hint: type appears to be bool
                # Hint: required = False (optional)
            },
            "use_sparse": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: use sparse matrices for very large grids (default: False)
                # Hint: type appears to be bool
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "random_val": {
                "type": "float",
                "description": "A random value between 0 and 1",
            },
            "fp_fig": {
                "type": "str",
                "description": "Filepath to the generated test figure",
            },
            "resolution": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "fit_quality": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "autocorr_map": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "radial_profile": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "radial_distances": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "resolution_radial": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "resolution_dblradial": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "fig_resolution": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be str
            },
            "fig_radial": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be str
            },
        }

        return parameters_spec, results_spec

    def resolution_frc_spatial(self):
        """Calculate resolution using spatial FRC approach

        This method divides the FOV into spatial regions, computes FRC for each
        region independently, and averages the results. Benefits:
        - Lower memory usage (smaller images per region)
        - Better statistics through spatial averaging
        - Efficient multiprocessing (fully independent regions)
        - Preserves high spatial frequencies

        Parameters
        ----------
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
        parameters_spec = {
            "xlabel": {
                "type": "str",
                "description": "Label for the x-axis of the test plot",
                "required": True,
            },
            "ylabel": {
                "type": "str",
                "description": "Label for the y-axis of the test plot",
                "required": True,
            },
            "pixelsize_render": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: pixel size for rendered images in nm (default: 5 nm)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "smoothing_sigma": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Gaussian smoothing sigma in pixels (default: None)
                # Hint: type appears to be float or None
                # Hint: required = False (optional)
            },
            "threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: FRC threshold for resolution cutoff (default: 1/7 ≈ 0.143)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "region_size": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: size of each spatial region in micrometers (default: 10.0 µm)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
            "min_locs_per_region": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: minimum localizations per region to process (default: 500)
                # Hint: type appears to be int
                # Hint: required = False (optional)
            },
            "max_frc_range_nm": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: maximum FRC range in nm (default: None = full range)
                # Hint: type appears to be float or None
                # Hint: required = False (optional)
            },
            "n_processes": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: number of parallel processes (default: 4)
                # Hint: type appears to be int
                # Hint: required = False (optional)
            },
            "smoothing_window": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: moving average window size for FRC smoothing in 1/nm (default: 0.005)
                # Hint: type appears to be float
                # Hint: required = False (optional)
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "random_val": {
                "type": "float",
                "description": "A random value between 0 and 1",
            },
            "fp_fig": {
                "type": "str",
                "description": "Filepath to the generated test figure",
            },
            "resolution_frc_spatial": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "resolution_std": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "n_regions": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be int
            },
            "cutoff_frequency": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "frc_curve_mean": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "frc_curve_std": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "spatial_frequencies": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: FRC threshold for resolution cutoff (default: 1/7 ≈ 0.143)
                # Hint: type appears to be float
            },
            "fig_frc": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be str
            },
        }

        return parameters_spec, results_spec

    def undrift_rsso(self):
        """Undrift localized data using iterative RSSO-based drift correction

        This method applies an iterative RSSO (Redundant Spot Shift
        Overrepresentation) approach where each frame is compared against
        the whole dataset to compute total drift for that frame. The process
        is repeated iteratively with the undrifted dataset to improve accuracy.
        Includes uncertainty analysis, confidence evaluation, windowing and
        outlier detection.

        Parameters
        ----------
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

        Returns
        -------
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
        parameters_spec = {
            "xlabel": {
                "type": "str",
                "description": "Label for the x-axis of the test plot",
                "required": True,
            },
            "ylabel": {
                "type": "str",
                "description": "Label for the y-axis of the test plot",
                "required": True,
            },
            "ton": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Half-life of localization in frames (how long a spot stays visible)
                # Hint: type appears to be float
            },
            "toff": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Time in frames for a spot to reappear after disappearing
                # Hint: type appears to be float
            },
            "max_shift": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Maximum expected drift per frame in pixels optional items:
                # Hint: type appears to be float
            },
            "min_locs_per_frame": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Minimum localizations per frame for reliable drift estimation (default: 10)
                # Hint: type appears to be int
            },
            "max_iterations": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Maximum number of iterative refinement rounds (default: 5)
                # Hint: type appears to be int
            },
            "convergence_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: RMS drift change threshold for convergence in nm (default: 0.1)
                # Hint: type appears to be float
            },
            "plot_drift": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Whether to save drift plots (default: True)
                # Hint: type appears to be bool
            },
            # "save_locs": {
            #     # TODO: Add type, description, min, max, default, required, step, extensions, properties
            #     # Hint: description: Whether to save undrifted localizations (default: True)
            #     # Hint: type appears to be bool
            # },
            "n_processes": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Number of processes for parallel computation (default: auto)
                # Hint: type appears to be int or None
            },
            "confidence_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Confidence threshold for windowing analysis (default: 0.8)
                # Hint: type appears to be float
            },
            "outlier_detection_enabled": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Enable RSSO failure and outlier detection (default: True)
                # Hint: type appears to be bool
            },
            "outlier_z_threshold": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Z-score threshold for temporal outlier detection (default: 3.5)
                # Hint: type appears to be float
            },
            "min_signal_to_noise": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Minimum signal-to-noise ratio for drift measurements (default: 0.5)
                # Hint: type appears to be float
            },
            "windowing_enabled": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Enable adaptive windowing for low-confidence frames (default: True)
                # Hint: type appears to be bool
            },
            "window_size_range": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: description: Min and max window sizes for adaptive windowing (default: (3, 20))
                # Hint: type appears to be tuple
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "end time": {
                "type": "str",
                "description": "Module execution end timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "folder": {
                "type": "str",
                "description": "Output folder for module results",
            },
            "random_val": {
                "type": "float",
                "description": "A random value between 0 and 1",
            },
            "fp_fig": {
                "type": "str",
                "description": "Filepath to the generated test figure",
            },
            "success": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be bool
            },
            "drift_quality": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be ndarray
            },
            "n_iterations": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be int
            },
            "convergence_rms": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be float
            },
            "drift_plots": {
                # TODO: Add type, description, min, max, default, required, step, extensions, properties
                # Hint: type appears to be str
            },
        }

        return parameters_spec, results_spec


class SlurmCommunicator:
    """Communication interface for SLURM job scheduling and SSH command
    execution.

    This class provides methods to interact with SLURM job schedulers through
    SSH, create SLURM job scripts, submit jobs, and monitor their status. It's
    designed to work with remote compute clusters running SLURM workload
    manager.

    Attributes
    ----------
    hostname : str
        SSH hostname or IP address
    username : str
        SSH username for authentication
    port : int
        SSH port number (default: 22)
    ssh_key_path : str
        Path to SSH private key file
    timeout : int
        SSH connection timeout in seconds
    """

    def __init__(
        self, hostname, username, port=22, ssh_key_path=None, timeout=30
    ):
        """Initialize SLURM communicator with SSH connection parameters.

        Parameters
        ----------
        hostname : str
            SSH hostname or IP address
        username : str
            SSH username for authentication
        port : int, optional
            SSH port number. Defaults to 22.
        ssh_key_path : str, optional
            Path to SSH private key file.
            If None, uses default SSH authentication.
        timeout : int, optional
            SSH connection timeout in seconds. Defaults to 30.
        """
        self.hostname = hostname
        self.username = username
        self.port = port
        self.ssh_key_path = ssh_key_path
        self.timeout = timeout

        logger.info(
            f"Initialized SlurmCommunicator for {username}@{hostname}:{port}"
        )

    def execute_ssh_command(self, command, working_directory=None):
        """Execute a command via SSH on the remote host.

        Parameters
        ----------
        command : str
            Shell command to execute
        working_directory : str, optional
            Directory to execute command in

        Returns
        -------
        dict: Dictionary containing:
            - stdout (str): Standard output
            - stderr (str): Standard error
            - return_code (int): Command exit code
            - success (bool): True if return_code == 0

        Raises
        ------
        subprocess.CalledProcessError: If SSH connection fails
        FileNotFoundError: If SSH executable not found
        """
        ssh_cmd = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "ConnectTimeout={}".format(self.timeout),
            "-o",
            "StrictHostKeyChecking=no",
        ]

        if self.ssh_key_path:
            ssh_cmd.extend(["-i", self.ssh_key_path])

        # Construct full command
        if working_directory:
            remote_command = f"cd {working_directory} && {command}"
        else:
            remote_command = command

        ssh_cmd.extend([f"{self.username}@{self.hostname}", remote_command])

        logger.debug(f"Executing SSH command: {' '.join(ssh_cmd)}")

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
                * 2,  # Allow extra time for command execution
            )

            response = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "success": result.returncode == 0,
            }

            if not response["success"]:
                logger.warning(
                    "SSH command failed with return code "
                    + f"{result.returncode}: {result.stderr}"
                )
            else:
                logger.debug("SSH command completed successfully")

            return response

        except subprocess.TimeoutExpired:
            logger.error(
                f"SSH command timed out after {self.timeout * 2} seconds"
            )
            return {
                "stdout": "",
                "stderr": (
                    f"Command timed out after {self.timeout * 2} seconds"
                ),
                "return_code": -1,
                "success": False,
            }
        except Exception as e:
            logger.error(f"SSH command execution failed: {str(e)}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
            }

    def test_connection(self):
        """Test SSH connection to the remote host.

        Returns
        -------
        bool: True if connection successful, False otherwise
        """
        result = self.execute_ssh_command('echo "Connection test successful"')
        return result["success"]

    def assemble_slurm_commands(
        self, host_cluster, scriptname="start_workflow.py"
    ):
        """Assembles picasso-workflow specific commands for running a batch
        job on a SLURM cluster.
        """
        cluster_env = CONFIG.get("ClusterEnvironment", {}).get(host_cluster)
        conda_env = cluster_env.get("conda_env", "picasso-workflow")

        commands = []

        commands.append("source ~/.bashrc")

        commands.append("source /etc/profile.d/modules.sh")
        # if use_pw_module:
        #     commands.append(f"module load {pw_module}")
        # else:
        #     commands.append(f"module load {anaconda_module}")

        # if not use_pw_module:
        #     commands.append(f"conda activate {conda_env}")

        # instead of using slurm modules, let's directly append paths.
        bin_path = cluster_env.get("BinPath", None)
        if bin_path:
            commands.append(f"export PATH={bin_path}:$PATH")
        lib_path = cluster_env.get("LibraryPath", None)
        if lib_path:
            commands.append(
                f"export LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH"
            )
        python_path = cluster_env.get("PythonPath", None)
        if python_path:
            commands.append(f"export PYTHONPATH={python_path}:$PYTHONPATH")
        conda_env = cluster_env.get("CondaEnv", None)
        if conda_env:
            commands.append(f"export CONDA_DEFAULT_ENV={conda_env}")
        conda_prefix = cluster_env.get("CondaPrefix", None)
        if conda_prefix:
            commands.append(f"export CONDA_PREFIX={conda_prefix}")

        commands.append(f"srun python {scriptname}")

        return commands

    def create_slurm_script(
        self,
        job_name,
        commands,
        slurm_options=None,
        output_file=None,
        error_file=None,
        working_directory=None,
    ):
        """Create a SLURM job script with specified parameters.

        Parameters
        ----------
        job_name : str
            Name for the SLURM job
        commands : list or str
            List of shell commands or single command string
        slurm_options : dict, optional
            SLURM directive options. Keys should be SLURM parameter names
            (without --), values should be parameter values.
            Common options:
                  - partition: Queue/partition name
                  - nodes: Number of nodes
                  - ntasks: Number of tasks
                  - cpus-per-task: CPUs per task
                  - mem: Memory per node
                  - time: Time limit (HH:MM:SS)
                  - mail-type: Email notification type
                  - mail-user: Email address
        output_file : str, optional
            Path for stdout redirection
        error_file : str, optional
            Path for stderr redirection
        working_directory : str, optional
            Working directory for the job

        Returns
        -------
        str: Complete SLURM script content
        """
        script_lines = ["#!/bin/bash"]

        # Add SLURM directives
        script_lines.append(f"#SBATCH --job-name={job_name}")

        if output_file:
            script_lines.append(f"#SBATCH --output={output_file}")
        else:
            script_lines.append(f"#SBATCH --output={job_name}_%j.out")

        if error_file:
            script_lines.append(f"#SBATCH --error={error_file}")
        else:
            script_lines.append(f"#SBATCH --error={job_name}_%j.err")

        # Add additional SLURM options
        if slurm_options:
            for option, value in slurm_options.items():
                if value is not None:
                    script_lines.append(f"#SBATCH --{option}={value}")

        script_lines.append("")  # Empty line after SLURM directives

        # Add working directory change if specified
        if working_directory:
            script_lines.append(f"cd {working_directory}")
            script_lines.append("")

        # Add timestamp and job info
        script_lines.extend(
            [
                'echo "Job started at: $(date)"',
                'echo "Running on node: $(hostname)"',
                'echo "Job ID: $SLURM_JOB_ID"',
                'echo "Current working directory: $(pwd)"',
                'echo ""',
            ]
        )

        # Add user commands
        if isinstance(commands, str):
            commands = [commands]

        for cmd in commands:
            script_lines.append(cmd)

        # Add job completion timestamp
        script_lines.extend(
            ["", 'echo ""', 'echo "Job completed at: $(date)"']
        )

        return "\n".join(script_lines)

    def write_slurm_script(self, script_content, folder, local=True):
        """Write SLURM script content to a file on the remote host.

        Parameters
        ----------
        script_content : str
            Complete SLURM script content
        folder : str
            Path where to save the script on remote host
        local : bool
            whether the folder is available on the local system

        Returns
        -------
        dict: Result of the write operation (see execute_ssh_command)
        """
        if local:
            filepath = os.path.join(folder, "run_workflow_slurm.sh")
            # write with UNIX style newlines ('\n') instead of DOS ('\r\n')
            with open(filepath, "w", newline="\n") as f:
                f.write(script_content)
            return filepath

        # otherwise, copy to remote host
        # Create a temporary local file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False
        ) as tmp_file:
            tmp_file.write(script_content)
            tmp_file_path = tmp_file.name

        try:
            # Copy file to remote host using scp
            scp_cmd = [
                "scp",
                "-P",
                str(self.port),
                "-o",
                "ConnectTimeout={}".format(self.timeout),
                "-o",
                "StrictHostKeyChecking=no",
            ]

            if self.ssh_key_path:
                scp_cmd.extend(["-i", self.ssh_key_path])

            scp_cmd.extend(
                [
                    tmp_file_path,
                    f"{self.username}@{self.hostname}:{folder}",
                ]
            )

            logger.debug(f"Copying SLURM script: {' '.join(scp_cmd)}")

            result = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=self.timeout
            )

            if result.returncode == 0:
                # Make script executable
                chmod_result = self.execute_ssh_command(f"chmod +x {folder}")
                if chmod_result["success"]:
                    logger.info(
                        f"SLURM script written successfully to {folder}"
                    )
                    return {
                        "stdout": f"Script written to {folder}",
                        "stderr": "",
                        "return_code": 0,
                        "success": True,
                    }
                else:
                    return chmod_result
            else:
                logger.error(f"Failed to copy SLURM script: {result.stderr}")
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "success": False,
                }

        except Exception as e:
            logger.error(f"Error writing SLURM script: {str(e)}")
            return {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "success": False,
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass

    def submit_job(self, script_path, dest_machine, additional_options=None):
        """Submit a SLURM job using sbatch.

        Parameters
        ----------
        script_path : str
            Path to the SLURM script on remote host
        additional_options : list, optional
            Additional sbatch options

        Returns
        -------
        dict: Result containing job submission information:
            - job_id (int or None): SLURM job ID if successful
            - stdout (str): sbatch output
            - stderr (str): sbatch errors
            - success (bool): True if job submitted successfully
        """
        # check script path. if it is local, convert to its location on
        # the remote host
        from picasso_workflow.metaworkflow import PathParser

        pp = PathParser()
        script_path = pp.convert_path(script_path, dest_machine)

        sbatch_cmd = "sbatch"

        if additional_options:
            sbatch_cmd += " " + " ".join(additional_options)

        sbatch_cmd += f" {script_path}"

        logger.info(f"Submitting job via SSH: {sbatch_cmd}")
        result = self.execute_ssh_command(sbatch_cmd)

        # Parse job ID from output
        job_id = None
        if result["success"] and result["stdout"]:
            # sbatch typically outputs: "Submitted batch job 12345"
            import re

            match = re.search(r"Submitted batch job (\d+)", result["stdout"])
            if match:
                job_id = int(match.group(1))
                logger.info(f"Job submitted successfully with ID: {job_id}")
            else:
                logger.warning("Job submitted but could not parse job ID")

        return {
            "job_id": job_id,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "success": result["success"],
        }

    def get_job_status(self, job_id):
        """Get the status of a SLURM job.

        Parameters
        ----------
        job_id : int
            SLURM job ID

        Returns
        -------
        dict: Job status information:
            - status (str):
                Job status (PENDING, RUNNING, COMPLETED, FAILED, etc.)
            - details (dict): Additional job information
            - success (bool): True if status query successful
        """
        squeue_cmd = f'squeue --job={job_id} --format="%T %R %S %E" --noheader'
        result = self.execute_ssh_command(squeue_cmd)

        if result["success"] and result["stdout"].strip():
            # Parse squeue output
            fields = result["stdout"].strip().split()
            if len(fields) >= 1:
                status = fields[0]
                reason = fields[1] if len(fields) > 1 else ""
                start_time = fields[2] if len(fields) > 2 else ""
                end_time = fields[3] if len(fields) > 3 else ""

                return {
                    "status": status,
                    "details": {
                        "reason": reason,
                        "start_time": start_time,
                        "end_time": end_time,
                    },
                    "success": True,
                }

        # If squeue doesn't show the job, check if it's completed using sacct
        sacct_cmd = (
            f"sacct --job={job_id} --format=State --noheader --parsable2"
        )
        sacct_result = self.execute_ssh_command(sacct_cmd)

        if sacct_result["success"] and sacct_result["stdout"].strip():
            status = sacct_result["stdout"].strip().split("\n")[0]
            return {"status": status, "details": {}, "success": True}

        return {
            "status": "UNKNOWN",
            "details": {"error": result["stderr"]},
            "success": False,
        }

    def cancel_job(self, job_id):
        """Cancel a SLURM job.

        Parameters
        ----------
        job_id : int
            SLURM job ID to cancel

        Returns
        -------
        dict: Result of the cancellation (see execute_ssh_command)
        """
        scancel_cmd = f"scancel {job_id}"
        result = self.execute_ssh_command(scancel_cmd)

        if result["success"]:
            logger.info(f"Job {job_id} cancelled successfully")
        else:
            logger.error(f"Failed to cancel job {job_id}: {result['stderr']}")

        return result

    def list_jobs(self, user=None):
        """List SLURM jobs for a user.

        Parameters
        ----------
        user : str, optional
            Username to list jobs for.
                            If None, uses the SSH username.

        Returns
        -------
        dict: Dictionary containing:
            - jobs (list): List of job dictionaries with job information
            - success (bool): True if listing successful
        """
        if user is None:
            user = self.username

        squeue_cmd = (
            f"squeue --user={user} "
            + '--format="%i %T %j %u %P %M %l %D %R" --noheader'
        )
        result = self.execute_ssh_command(squeue_cmd)

        jobs = []
        if result["success"] and result["stdout"].strip():
            for line in result["stdout"].strip().split("\n"):
                fields = line.split()
                if len(fields) >= 6:
                    job_info = {
                        "job_id": fields[0],
                        "status": fields[1],
                        "job_name": fields[2],
                        "user": fields[3],
                        "partition": fields[4],
                        "time_used": fields[5],
                        "time_limit": fields[6] if len(fields) > 6 else "",
                        "nodes": fields[7] if len(fields) > 7 else "",
                        "reason": (
                            " ".join(fields[8:]) if len(fields) > 8 else ""
                        ),
                    }
                    jobs.append(job_info)

        return {
            "jobs": jobs,
            "success": result["success"],
            "stderr": result["stderr"],
        }

    def get_queue_info(self):
        """Get information about SLURM partitions/queues.

        Returns
        -------
        dict: Dictionary containing:
            - partitions (list): List of partition information dictionaries
            - success (bool): True if query successful
        """
        sinfo_cmd = 'sinfo --format="%P %a %l %D %T %N" --noheader'
        result = self.execute_ssh_command(sinfo_cmd)

        partitions = []
        if result["success"] and result["stdout"].strip():
            for line in result["stdout"].strip().split("\n"):
                fields = line.split()
                if len(fields) >= 5:
                    partition_info = {
                        "name": fields[0].rstrip(
                            "*"
                        ),  # Remove default partition marker
                        "availability": fields[1],
                        "time_limit": fields[2],
                        "nodes": fields[3],
                        "state": fields[4],
                        "node_list": (
                            " ".join(fields[5:]) if len(fields) > 5 else ""
                        ),
                    }
                    partitions.append(partition_info)

        return {
            "partitions": partitions,
            "success": result["success"],
            "stderr": result["stderr"],
        }


class FilePathEditor(QtWidgets.QWidget):
    """Custom editor widget for file path selection with browse button."""

    editingFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up the layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Create line edit and button
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.button = QtWidgets.QPushButton("Browse", self)
        self.button.setFocusPolicy(
            QtCore.Qt.FocusPolicy.NoFocus
        )  # Prevent focus issues

        # Add widgets to layout
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button)

        # Connect signals
        self.button.clicked.connect(self.browse)
        self.lineEdit.editingFinished.connect(self.on_edit_finished)

        # Track if we're in the browse dialog
        self._browsing = False

    def browse(self):
        """Open file dialog to select a file path."""
        self._browsing = True
        try:
            # Store current value in case widget gets deleted
            current_text = self.lineEdit.text()

            # Use window() to get the top-level window as a stable parent
            # This prevents memory issues with temporary editor widgets
            parent_window = self.window()

            # Open file dialog - this is a blocking call
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent_window,
                "Select File",
                current_text if current_text else "",
            )

            # Check if widget is still valid (might have been deleted during dialog)
            if path and not self.isHidden():
                try:
                    self.lineEdit.setText(os.path.normpath(path))
                    # Use QTimer to safely emit signal after returning to event loop
                    QtCore.QTimer.singleShot(0, self.editingFinished.emit)
                except RuntimeError:
                    # Widget was deleted, nothing we can do
                    pass
        finally:
            self._browsing = False

    def on_edit_finished(self):
        """Handle line edit finished signal, but not during browse."""
        if not self._browsing:
            self.editingFinished.emit()

    def setText(self, text):
        """Set the text in the line edit."""
        if text:
            self.lineEdit.setText(str(text))

    def text(self):
        """Get the current text from the line edit."""
        return self.lineEdit.text()


class DroppableLineEdit(QtWidgets.QLineEdit):
    """QLineEdit with drag-and-drop support for files from Finder/Explorer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            # Take the first file path
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    self.setText(url.toLocalFile())
                    break
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class DroppableFolderLineEdit(QtWidgets.QLineEdit):
    """QLineEdit for folder paths with drag-and-drop support.

    Dropping a folder populates the path with that folder; dropping a file
    populates it with the file's containing directory.
    """

    folderDropped = QtCore.pyqtSignal(str)  # Emits the dropped folder path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            # Take the first dropped item
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if not os.path.isdir(path):
                        # A file was dropped: use its containing directory
                        path = os.path.dirname(path)
                    path = os.path.normpath(path)
                    self.setText(path)
                    self.folderDropped.emit(path)
                    break
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class DroppableTableWidget(QtWidgets.QTableWidget):
    """QTableWidget with drag-and-drop support for files from Finder/Explorer."""

    filesDropped = QtCore.pyqtSignal(list)  # Emits list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    paths.append(url.toLocalFile())
            if paths:
                self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class DroppableTreeWidget(QtWidgets.QTreeWidget):
    """QTreeWidget with drag-and-drop support for files and internal moves."""

    filesDropped = QtCore.pyqtSignal(list, object)  # (file_paths, target_item)
    fileMoved = QtCore.pyqtSignal(object, object)  # (source_item, target_item)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._drag_source_item = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            # External file drop
            event.acceptProposedAction()
        elif event.source() == self:
            # Internal drag
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.source() == self:
            # Get item under cursor for visual feedback
            item = self.itemAt(event.position().toPoint())
            if item:
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def startDrag(self, supportedActions):
        """Store the source item when starting an internal drag."""
        items = self.selectedItems()
        if items:
            # Only allow dragging channel items (items with a parent)
            item = items[0]
            if item.parent() is not None:
                self._drag_source_item = item
                super().startDrag(supportedActions)
            else:
                self._drag_source_item = None
        else:
            self._drag_source_item = None

    def dropEvent(self, event):
        target_item = self.itemAt(event.position().toPoint())

        if event.mimeData().hasUrls():
            # External file drop
            paths = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    paths.append(url.toLocalFile())
            if paths and target_item:
                self.filesDropped.emit(paths, target_item)
                event.acceptProposedAction()
            else:
                event.ignore()
        elif event.source() == self and self._drag_source_item:
            # Internal move - only to channel items
            if target_item and target_item.parent() is not None:
                # Target is a channel item
                self.fileMoved.emit(self._drag_source_item, target_item)
                event.acceptProposedAction()
            else:
                event.ignore()
            self._drag_source_item = None
        else:
            event.ignore()


class FilePathDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_editor = None

    def createEditor(self, parent, option, index):
        editor = FilePathEditor(parent)
        # Keep a reference to prevent premature deletion
        self._current_editor = editor
        # Install event filter to prevent premature closure
        editor.installEventFilter(self)
        # Use functools.partial for cleaner signal connection
        editor.editingFinished.connect(
            functools.partial(self.commitAndCloseEditor, editor)
        )
        return editor

    def destroyEditor(self, editor, index):
        """Override to clean up our reference."""
        if self._current_editor == editor:
            self._current_editor = None
        super().destroyEditor(editor, index)

    def setEditorData(self, editor, index):
        value = index.model().data(index, QtCore.Qt.ItemDataRole.EditRole)
        if value:
            editor.setText(str(value))

    def setModelData(self, editor, model, index):
        # Check if editor is still valid
        try:
            text = editor.text()
            model.setData(index, text, QtCore.Qt.ItemDataRole.EditRole)
        except RuntimeError:
            pass  # Editor was deleted

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def commitAndCloseEditor(self, editor):
        # Check if editor still exists and is not browsing
        try:
            if hasattr(editor, "_browsing") and not editor._browsing:
                # Commit and close the editor
                self.commitData.emit(editor)
                self.closeEditor.emit(editor)
        except RuntimeError:
            pass  # Editor was already deleted

    def eventFilter(self, editor, event):
        """Prevent editor from closing when file dialog opens."""
        # Don't let focus out events close the editor if we're browsing
        if event.type() == QtCore.QEvent.Type.FocusOut:
            if hasattr(editor, "_browsing") and editor._browsing:
                return True  # Ignore the event
        return super().eventFilter(editor, event)


def dict_to_table(d, table):
    table.setRowCount(len(d))
    for row, (key, value) in enumerate(d.items()):
        key_item = QtWidgets.QTableWidgetItem(key)
        value_item = QtWidgets.QTableWidgetItem(value)
        table.setItem(row, 0, key_item)
        table.setItem(row, 1, value_item)


class ToolTipDelegate(QtWidgets.QStyledItemDelegate):
    def helpEvent(self, event, view, option, index):
        if event.type() == QEvent.Type.ToolTip:
            tooltip = index.data(Qt.ItemDataRole.ToolTipRole)
            if tooltip:
                QtWidgets.QToolTip.showText(event.globalPos(), tooltip)
                return True
        return super().helpEvent(event, view, option, index)


class ParameterWidgetInfo:
    """Container for parameter widget information."""

    def __init__(
        self,
        widget,
        cmd_button,
        row_widget,
        metadata,
        original_type,
        sub_parameters=None,
        toggle_function=None,
    ):
        """Initialize parameter widget info.

        Parameters
        ----------
        widget: The Qt widget for parameter input
        cmd_button: QPushButton for opening command dialog
        row_widget: Container QWidget for the row
        metadata: Parameter metadata dictionary
        original_type: Original type string ('int', 'float', 'bool', 'str', 'dict')
        sub_parameters: Dict of nested ParameterWidgetInfo for dict types (optional)
        toggle_function: Function to show/hide nested parameters (for dict types, optional)
        """
        self.widget = widget
        self.cmd_button = cmd_button
        self.row_widget = row_widget
        self.metadata = metadata
        self.original_type = original_type
        self.sub_parameters = (
            sub_parameters or {}
        )  # For nested dict parameters
        self.toggle_function = (
            toggle_function  # For dict parameters with checkboxes
        )


class ParameterCmdDialog(QtWidgets.QDialog):
    """Dialog for selecting a command as parameter value."""

    def __init__(
        self,
        workflow_modules,
        module_descriptor,
        current_module_index=0,
        parent=None,
    ):
        """Initialize the prior result dialog.

        Parameters
        ----------
        workflow_modules: List of tuples (module_name, param_dict) from workflow
        module_descriptor: ModuleDescriptor instance to get result specs
        current_module_index: the index of the currently selected module
        parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Select Command for parameter value")
        self.setModal(True)
        self.workflow_modules = workflow_modules
        self.module_descriptor = module_descriptor
        self.current_module_index = current_module_index
        self.parent = parent

        layout = QtWidgets.QVBoxLayout(self)

        # Timing selection
        layout.addWidget(QtWidgets.QLabel("Value collection timing:"))
        self.timing_group = QtWidgets.QButtonGroup(self)

        self.timing_before_radio = QtWidgets.QRadioButton(
            "Collect directly before module execution ($)"
        )
        self.timing_start_radio = QtWidgets.QRadioButton(
            "Collect at start of workflow stage ($$)"
        )
        self.timing_before_radio.setChecked(True)  # Default

        self.timing_group.addButton(self.timing_before_radio, 0)
        self.timing_group.addButton(self.timing_start_radio, 1)
        self.timing_group.buttonToggled.connect(self._on_timing_toggled)

        layout.addWidget(self.timing_before_radio)
        layout.addWidget(self.timing_start_radio)
        layout.addSpacing(10)

        # Command type selection
        layout.addWidget(QtWidgets.QLabel("Command type:"))
        self.command_combo = QtWidgets.QComboBox()
        self.command_combo.addItems(
            ["map", "index", "Previous Module Result", "Prior Result"]
        )  # , "sum", "max", "min"])
        self.command_combo.setItemDelegate(ToolTipDelegate(self.command_combo))
        self.command_combo.currentIndexChanged.connect(
            self._on_command_changed
        )
        # self.command_combo.model().setData(
        #     0, "Map different values onto workers (e.g. files to load)", Qt.ItemDataRole.ToolTipRole
        # )  # Tooltip
        # self.command_combo.model().setData(
        #     1, "Load value from a result of a previous module in this or a previous workflow stage", Qt.ItemDataRole.ToolTipRole
        # )  # Tooltip
        layout.addWidget(self.command_combo)
        layout.addSpacing(10)

        # Container for dynamic widgets
        self.dynamic_widget = QtWidgets.QWidget()
        self.dynamic_layout = QtWidgets.QVBoxLayout(self.dynamic_widget)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dynamic_widget)

        # Create widgets for "Prior Result" mode
        self.prior_mode = QtWidgets.QButtonGroup(self)
        self.prior_thisstage = QtWidgets.QRadioButton(
            "Result from current stage"
        )
        self.prior_singlestage_list = QtWidgets.QRadioButton(
            "Create list from the single stage entries"
        )
        self.prior_thisstage.setChecked(True)  # Default

        self.prior_mode.addButton(self.prior_thisstage, 0)
        self.prior_mode.addButton(self.prior_singlestage_list, 1)
        self.prior_mode.buttonToggled.connect(self._on_prior_mode_selected)

        self.module_label = QtWidgets.QLabel("Select module:")
        self.module_combo = QtWidgets.QComboBox()
        for i, (module_name, params) in enumerate(workflow_modules):
            self.module_combo.addItem(f"{i:02d}: {module_name}")
        self.module_combo.currentIndexChanged.connect(self._on_module_selected)

        self.result_label = QtWidgets.QLabel("Select result:")
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.currentIndexChanged.connect(self._on_result_selected)
        self.result_combo.setPlaceholderText("Select a result")

        # modify_vlayout = QtWidgets.QVBoxLayout()
        # self.modify_widget = QtWidgets.QWidget()
        # self.modify_widget.setLayout(modify_vlayout)
        self.modify_label = QtWidgets.QLabel("Modify result:")
        # modify_vlayout.addWidget(self.modify_label)
        # modify_hlayout = QtWidgets.QHBoxLayout()
        # modify_hwidget = QtWidgets.QWidget()
        # modify_hwidget.setLayout(modify_hlayout)
        self.modify_combo = QtWidgets.QComboBox()
        self.modify_combo.addItems(
            ["", "multiply", "divide", "add", "subtract"]
        )
        self.modify_combo.currentIndexChanged.connect(
            self._on_modify_operator_selected
        )
        # modify_hlayout.addWidget(self.modify_combo)
        self.modify_value = QtWidgets.QDoubleSpinBox()
        self.modify_value.valueChanged.connect(self._on_modify_value_changed)
        # modify_hlayout.addWidget(self.modify_value)

        # Create widgets for "map" mode
        self.map_label = QtWidgets.QLabel("Map option:")
        self.map_combo = QtWidgets.QComboBox()
        self.map_combo.addItems(["filepath", "#tags"])
        self.map_combo.currentTextChanged.connect(self._on_map_option)

        # Create widgets for "index" mode
        self.index_label = QtWidgets.QLabel("Index option:")
        self.index_combo = QtWidgets.QComboBox()
        self.index_combo.addItems(["filepath", "#tags"])
        self.index_combo.currentTextChanged.connect(self._on_index_option)
        self.index_spin = QtWidgets.QSpinBox()
        self.index_spin.setMinimum(0)
        self.index_spin.setValue(0)
        self.index_spin.valueChanged.connect(self._on_index_spin)

        layout.addWidget(QtWidgets.QLabel("Assembled Command:"))
        self.command_result = QtWidgets.QLineEdit()
        layout.addSpacing(10)
        layout.addWidget(self.command_result)

        # Initialize with first command type
        self._on_command_changed(0)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_timing_toggled(self, button, checked):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_map_option(self, option):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_index_option(self, option):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_index_spin(self, value):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_command_changed(self, index):
        """Handle command type change and update dynamic layout.

        Parameters
        ----------
        index: Index of selected command type
        """
        command_type = self.command_combo.currentText()

        # Clear dynamic layout
        while self.dynamic_layout.count():
            item = self.dynamic_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        if command_type == "map":
            # Show map option widgets
            self.dynamic_layout.addWidget(self.map_label)
            self.dynamic_layout.addWidget(self.map_combo)
        elif command_type == "index":
            self.dynamic_layout.addWidget(self.index_label)
            self.dynamic_layout.addWidget(self.index_combo)
            self.dynamic_layout.addWidget(self.index_spin)
        elif command_type == "Prior Result":
            # Show module and result selection widgets
            self.dynamic_layout.addWidget(self.prior_thisstage)
            self.dynamic_layout.addWidget(self.prior_singlestage_list)
            self.dynamic_layout.addWidget(self.module_label)
            self.dynamic_layout.addWidget(self.module_combo)
            self.dynamic_layout.addWidget(self.result_label)
            self.dynamic_layout.addWidget(self.result_combo)
            # self.dynamic_layout.addWidget(self.modify_widget)
            self.dynamic_layout.addWidget(self.modify_label)
            self.dynamic_layout.addWidget(self.modify_combo)
            self.dynamic_layout.addWidget(self.modify_value)

            # Populate results for initially selected module
            self._on_module_selected(self.module_combo.currentIndex())
        elif command_type == "Previous Module Result":
            self.dynamic_layout.addWidget(self.result_label)
            self.dynamic_layout.addWidget(self.result_combo)
            # self.dynamic_layout.addWidget(self.modify_widget)
            self.dynamic_layout.addWidget(self.modify_label)
            self.dynamic_layout.addWidget(self.modify_combo)
            self.dynamic_layout.addWidget(self.modify_value)
            # Populate results for previous module
            if self.current_module_index > 0:
                self._on_module_selected(self.current_module_index - 1)
        # For sum, max, min commands, also show module/result selection
        elif command_type in ["sum", "max", "min"]:
            self.dynamic_layout.addWidget(self.module_label)
            self.dynamic_layout.addWidget(self.module_combo)
            self.dynamic_layout.addWidget(self.result_label)
            self.dynamic_layout.addWidget(self.result_combo)

            # Populate results for initially selected module
            self._on_module_selected(self.module_combo.currentIndex())

        command = self.get_command()
        self.command_result.setText(command)

    def _on_prior_mode_selected(self, button, checked):
        self.module_combo.blockSignals(True)
        self.module_combo.clear()
        if self.prior_singlestage_list.isChecked():
            workflow_modules = self.parent.single_workflow_modules
        else:
            workflow_modules = self.workflow_modules

        for i, (module_name, params) in enumerate(workflow_modules):
            self.module_combo.addItem(f"{i:02d}: {module_name}")
        self.module_combo.blockSignals(False)

        # Reset to first module when switching lists to prevent index mismatch
        if self.module_combo.count() > 0:
            self.module_combo.setCurrentIndex(0)
            self._on_module_selected(0)

        command = self.get_command()
        self.command_result.setText(command)

    def _on_module_selected(self, index):
        """Populate result combo box when module is selected.

        Parameters
        ----------
        index: Index of selected module in workflow
        """
        if self.prior_singlestage_list.isChecked():
            workflow_modules = self.parent.single_workflow_modules
        else:
            workflow_modules = self.workflow_modules

        # Bounds check and auto-correct invalid index
        if index < 0 or index >= len(workflow_modules):
            if len(workflow_modules) > 0:
                index = 0
                self.module_combo.setCurrentIndex(0)
            else:
                self.result_combo.clear()
                self.result_combo.addItem("(no modules available)")
                return

        module_name = workflow_modules[index][0]

        # Clear current results
        self.result_combo.clear()

        # Get results spec from module descriptor
        try:
            desc_fun = getattr(self.module_descriptor, module_name, None)
            if desc_fun and callable(desc_fun):
                _, results_spec = desc_fun()

                # Populate combo box with result names
                if results_spec:
                    for result_name in results_spec.keys():
                        self.result_combo.addItem(result_name)
                else:
                    self.result_combo.addItem("(no results defined)")
            else:
                self.result_combo.addItem("(module not found)")
        except Exception as e:
            self.result_combo.addItem(f"(error: {str(e)})")

        command = self.get_command()
        self.command_result.setText(command)

    def _on_result_selected(self, index):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_modify_value_changed(self, value):
        command = self.get_command()
        self.command_result.setText(command)

    def _on_modify_operator_selected(self, index):
        command = self.get_command()
        self.command_result.setText(command)

    # def get_selection(self):
    #     """Get the selected module index, result name, command type, and timing.

    #     Returns:
    #         tuple: For "map" command: (None, map_option: str, "map", timing: str)
    #                For other commands: (module_index: int, result_name: str, command_type: str, timing: str)
    #                command_type is "map", "Prior Result", "sum", "max", or "min"
    #                timing is either "before" or "start"
    #                map_option is "filepath" or "#tags"
    #     """
    #     command_type = self.command_combo.currentText()
    #     timing = "before" if self.timing_before_radio.isChecked() else "start"

    #     if command_type == "map":
    #         # For map command, return map option instead of module/result
    #         map_option = self.map_combo.currentText()
    #         return None, map_option, command_type, timing
    #     else:
    #         # For other commands, return module index and result name
    #         module_index = self.module_combo.currentIndex()
    #         result_name = self.result_combo.currentText()
    #         return module_index, result_name, command_type, timing

    def get_command(self):
        command_type = self.command_combo.currentText()
        timing = "before" if self.timing_before_radio.isChecked() else "start"
        if timing == "before":
            timing_cmd = "$"
        else:
            timing_cmd = "$$"

        if command_type == "map":
            # For map command, return map option instead of module/result
            map_option = self.map_combo.currentText()
            command_string = f"('{timing_cmd}map', '{map_option}')"
        elif command_type == "index":
            map_option = self.index_combo.currentText()
            index = self.index_spin.value()
            command_string = f"('{timing_cmd}index {index}', '{map_option}')"
        elif command_type == "Prior Result":
            if self.prior_thisstage.isChecked():
                cmd_prefix = "results"
                workflow_modules = self.workflow_modules
            elif self.prior_singlestage_list.isChecked():
                cmd_prefix = "all_results, single_dataset, $$all"
                workflow_modules = self.parent.single_workflow_modules

            module_index = self.module_combo.currentIndex()
            # Validate index against the active workflow list
            if module_index < 0 or module_index >= len(workflow_modules):
                return "(invalid module selection)"
            module_name = self.module_combo.currentText()
            if ":" not in module_name:
                return "(invalid module selection)"
            module_name = module_name[module_name.index(":") + 2 :]
            result_name = self.result_combo.currentText()

            if self.modify_combo.currentText() == "multiply":
                modify_operator = "*"
            elif self.modify_combo.currentText() == "divide":
                modify_operator = "/"
            elif self.modify_combo.currentText() == "add":
                modify_operator = "+"
            elif self.modify_combo.currentText() == "subtract":
                modify_operator = "-"
            else:
                modify_operator = None
            modify_value = self.modify_value.value()

            if modify_operator is not None:
                mod_str = f" {modify_operator}{modify_value}"
            else:
                mod_str = ""

            command_string = f"('{timing_cmd}get_prior_result{mod_str}', '{cmd_prefix}, {module_index:02d}_{module_name}, {result_name}')"
        elif command_type == "Previous Module Result":
            result_name = self.result_combo.currentText()
            if self.modify_combo.currentText() == "multiply":
                modify_operator = "*"
            elif self.modify_combo.currentText() == "divide":
                modify_operator = "/"
            elif self.modify_combo.currentText() == "add":
                modify_operator = "+"
            elif self.modify_combo.currentText() == "subtract":
                modify_operator = "-"
            else:
                modify_operator = None
            modify_value = self.modify_value.value()
            if modify_operator is not None:
                mod_str = f" {modify_operator}{modify_value}"
            else:
                mod_str = ""

            command_string = f"('{timing_cmd}get_previous_module_result{mod_str}', '{result_name}')"
        return command_string


class Window(QtWidgets.QMainWindow):
    """Main window for the picasso-workflow GUI application."""

    def __init__(self):
        from picasso_workflow.metaworkflow import PathParser

        super().__init__()
        self.pathparser = PathParser()
        self.module_descriptor = ModuleDescriptor()

        self.setWindowTitle(f"picasso-workflow {__GUIVERSION__}")
        self.resize(1024, 600)

        self.single_workflow = []
        self.aggregation_workflow = []
        self.single_workflow_modules = []  # List of module name strings
        self.aggregation_workflow_modules = []  # List of module name strings
        self.parameter_widgets = (
            {}
        )  # Dict[param_name, (QLineEdit, param_metadata)]

        # Track currently editing workflow item for auto-save
        self.editing_workflow_index = (
            -1
        )  # -1 means not editing an existing item
        self.editing_workflow_tab = -1  # 0 = single, 1 = aggregation

        layout = QtWidgets.QGridLayout()
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        workflow_template_combo = QtWidgets.QComboBox()
        workflow_template_combo.addItem("")
        for template in CONFIG["Templates"].keys():
            workflow_template_combo.addItem(template)
        workflow_template_combo.setEditable(False)
        workflow_template_combo.currentTextChanged.connect(
            self.on_template_changed
        )
        workflow_template_combo.setToolTip("Load a template workflow")
        layout.addWidget(workflow_template_combo, 0, 0)
        # Results folder selection
        results_folder_button = QtWidgets.QPushButton("Results Folder")
        results_folder_button.setToolTip(
            "The folder must be accessible both from this and the compute machine (cluster)."
        )
        # self.files_box.addWidget(results_folder_button, 2, 0)
        layout.addWidget(results_folder_button, 0, 1)
        results_folder_button.clicked.connect(self.select_results_folder)
        self.results_folder_display = DroppableFolderLineEdit()
        self.results_folder_display.setReadOnly(False)
        self.results_folder_display.setPlaceholderText(
            "No folder selected (drag & drop a folder here)"
        )
        self.results_folder_display.textChanged.connect(
            self.set_results_folder_display
        )
        self.results_folder_display.folderDropped.connect(
            self.on_results_folder_dropped
        )
        # self.files_box.addWidget(self.results_folder_display, 2, 1, 1, 2)
        layout.addWidget(self.results_folder_display, 0, 2, 1, 2)
        # Investigation type
        self.workflow_type = QtWidgets.QComboBox()
        self.workflow_type.addItem("Single Workflow")
        self.workflow_type.addItem("Aggregation Workflow")
        self.workflow_type.addItem("Investigation Workflow")
        # disable Investigation workflow, which is in Development
        index = self.workflow_type.model().index(2, 0)
        self.workflow_type.model().setData(
            index, 0, Qt.ItemDataRole.UserRole - 1
        )  # 0 disables the item
        self.workflow_type.setItemDelegate(ToolTipDelegate(self.workflow_type))
        self.workflow_type.model().setData(
            index, "Not Implemented yet", Qt.ItemDataRole.ToolTipRole
        )  # Tooltip

        self.workflow_type.currentIndexChanged.connect(
            self._on_workflow_type_changed
        )
        layout.addWidget(self.workflow_type, 0, 4)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1, 0, 1, 5)

        # Workflow Config tab
        config_tab = QtWidgets.QWidget()
        config_layout = QtWidgets.QGridLayout(config_tab)
        self.tabs.addTab(config_tab, "Workflow Config")

        # Files and modules boxes in config tab
        self._files_box = QtWidgets.QGroupBox("Files")
        self.files_box = QtWidgets.QGridLayout(self._files_box)
        config_layout.addWidget(self._files_box, 0, 0, 1, 2)
        self._modules_box = QtWidgets.QGroupBox("Modules")
        self.modules_box = QtWidgets.QGridLayout(self._modules_box)
        config_layout.addWidget(self._modules_box, 0, 2, 1, 2)

        # Documentation Config tab
        docconfig_tab = QtWidgets.QWidget()
        docconfig_layout = QtWidgets.QGridLayout(docconfig_tab)
        self.tabs.addTab(docconfig_tab, "Documentation Config")

        # Documentation backend toggles
        self.document_confluence_checkbox = QtWidgets.QCheckBox(
            "Document to Confluence"
        )
        self.document_confluence_checkbox.setChecked(True)
        self.document_confluence_checkbox.setToolTip(
            "Upload the analysis report to Confluence. Uncheck to skip "
            "Confluence entirely (no credentials or connection needed)."
        )
        docconfig_layout.addWidget(
            self.document_confluence_checkbox, 0, 0, 1, 2
        )

        self.document_html_checkbox = QtWidgets.QCheckBox(
            "Generate local HTML report"
        )
        self.document_html_checkbox.setChecked(False)
        self.document_html_checkbox.setToolTip(
            "Write a navigable report.html (plus assets) into each run's "
            "result folder. Enabled automatically when Confluence is off."
        )
        docconfig_layout.addWidget(self.document_html_checkbox, 1, 0, 1, 2)

        # Confluence configuration group
        confluence_group = QtWidgets.QGroupBox("Confluence Settings")
        confluence_layout = QtWidgets.QGridLayout(confluence_group)
        docconfig_layout.addWidget(confluence_group, 2, 0, 1, 2)
        # grey out the Confluence fields when Confluence documentation is off
        self.document_confluence_checkbox.toggled.connect(
            confluence_group.setEnabled
        )

        # Get Confluence config with safe defaults
        confluence_config = CONFIG.get("Confluence", {})

        # Confluence URL
        confluence_layout.addWidget(QtWidgets.QLabel("URL:"), 0, 0)
        self.confluence_url_edit = QtWidgets.QLineEdit()
        self.confluence_url_edit.setPlaceholderText(
            "e.g., https://confluence.example.com"
        )
        self.confluence_url_edit.setToolTip(
            "If empty, host will load from environment variable"
        )
        self.confluence_url_edit.setText(confluence_config.get("URL", ""))
        confluence_layout.addWidget(self.confluence_url_edit, 0, 1)

        # Confluence Username
        confluence_layout.addWidget(QtWidgets.QLabel("Username:"), 1, 0)
        self.confluence_username_edit = QtWidgets.QLineEdit()
        self.confluence_username_edit.setPlaceholderText(
            "The email of your confluence acccount"
        )
        self.confluence_username_edit.setToolTip(
            "If empty, host will load from environment variable"
        )
        self.confluence_username_edit.setText(
            confluence_config.get("Username", "")
        )
        confluence_layout.addWidget(self.confluence_username_edit, 1, 1)

        # Confluence Space
        confluence_layout.addWidget(QtWidgets.QLabel("Space:"), 2, 0)
        self.confluence_space_edit = QtWidgets.QLineEdit()
        self.confluence_space_edit.setPlaceholderText(
            "e.g., ~username or TEAM"
        )
        self.confluence_space_edit.setToolTip(
            "If empty, host will load from environment variable"
        )
        self.confluence_space_edit.setText(confluence_config.get("Space", ""))
        confluence_layout.addWidget(self.confluence_space_edit, 2, 1)

        # Confluence Token: deliberately NOT an input field. The token is a
        # secret and is read only from the CONFLUENCE_TOKEN environment
        # variable at run time, so it is never typed into the GUI nor written
        # into the generated start_workflow.py.
        token_note = QtWidgets.QLabel(
            "Token: read from the CONFLUENCE_TOKEN environment variable "
            "(never stored in config or scripts)."
        )
        token_note.setWordWrap(True)
        token_note.setStyleSheet("color: #666;")
        confluence_layout.addWidget(token_note, 3, 0, 1, 2)

        # Parent Page
        confluence_layout.addWidget(QtWidgets.QLabel("Parent Page:"), 4, 0)
        self.confluence_parent_page_edit = QtWidgets.QLineEdit()
        self.confluence_parent_page_edit.setToolTip(
            "If empty, host will load from environment variable"
        )
        self.confluence_parent_page_edit.setPlaceholderText(
            "Page title to create reports under"
        )
        self.confluence_parent_page_edit.setText(
            confluence_config.get("DefaultPage", "")
        )
        confluence_layout.addWidget(self.confluence_parent_page_edit, 4, 1)

        # Add stretch to push widgets to the top
        docconfig_layout.setRowStretch(3, 1)

        # Run tab
        run_tab = QtWidgets.QWidget()
        run_layout = QtWidgets.QGridLayout(run_tab)
        self.tabs.addTab(run_tab, "Run")

        # run configuration
        self.run_tabs = QtWidgets.QTabWidget()
        run_layout.addWidget(self.run_tabs, 2, 0, 1, 4)

        run_on_cluster_tab = QtWidgets.QWidget()
        run_on_cluster_layout = QtWidgets.QVBoxLayout(run_on_cluster_tab)
        self.run_tabs.addTab(run_on_cluster_tab, "Run on SLURM Cluster")

        # Cluster configuration widgets
        cluster_config_layout = QtWidgets.QHBoxLayout()
        cluster_config_widget = QtWidgets.QWidget()
        cluster_config_widget.setLayout(cluster_config_layout)
        run_on_cluster_layout.addWidget(cluster_config_widget)

        # Cluster Host dropdown
        cluster_config_layout.addWidget(QtWidgets.QLabel("Cluster Host:"))
        self.cluster_host_combo = QtWidgets.QComboBox()
        for host in CONFIG["SlurmLoginNodes"].keys():
            self.cluster_host_combo.addItem(host)
        self.cluster_host_combo.setEditable(True)
        # self.cluster_host_combo.addItems(["localhost", "cluster.example.com"])
        cluster_config_layout.addWidget(self.cluster_host_combo)

        # Number of nodes
        cluster_config_layout.addWidget(QtWidgets.QLabel("#nodes:"))
        self.cluster_nodes_spin = QtWidgets.QSpinBox()
        self.cluster_nodes_spin.setMinimum(1)
        self.cluster_nodes_spin.setMaximum(1000)
        self.cluster_nodes_spin.setValue(
            CONFIG["SlurmDefault"].get("nodes", 1)
        )
        cluster_config_layout.addWidget(self.cluster_nodes_spin)

        # Number of cores per node
        cluster_config_layout.addWidget(QtWidgets.QLabel("#cores/node:"))
        self.cluster_cores_spin = QtWidgets.QSpinBox()
        self.cluster_cores_spin.setMinimum(1)
        self.cluster_cores_spin.setMaximum(256)
        self.cluster_cores_spin.setValue(
            CONFIG["SlurmDefault"].get("cores", 1)
        )
        cluster_config_layout.addWidget(self.cluster_cores_spin)

        # Number of GPUs per node (0 = no GPU requested)
        cluster_config_layout.addWidget(QtWidgets.QLabel("#GPUs/node:"))
        self.cluster_gpus_spin = QtWidgets.QSpinBox()
        self.cluster_gpus_spin.setMinimum(0)
        self.cluster_gpus_spin.setMaximum(64)
        self.cluster_gpus_spin.setValue(CONFIG["SlurmDefault"].get("gpus", 0))
        self.cluster_gpus_spin.setToolTip(
            "Number of GPUs to request per node via SLURM "
            "(adds '#SBATCH --gres=gpu:N'). 0 = no GPU."
        )
        cluster_config_layout.addWidget(self.cluster_gpus_spin)

        # Memory
        cluster_config_layout.addWidget(QtWidgets.QLabel("Memory:"))
        self.cluster_memory_edit = QtWidgets.QLineEdit()
        self.cluster_memory_edit.setPlaceholderText("e.g., 4GB")
        self.cluster_memory_edit.setText(
            CONFIG["SlurmDefault"].get("memory", "4GB")
        )
        self.cluster_memory_edit.setMaximumWidth(100)
        cluster_config_layout.addWidget(self.cluster_memory_edit)

        # Timeout
        cluster_config_layout.addWidget(QtWidgets.QLabel("Timeout:"))
        self.cluster_timeout_edit = QtWidgets.QLineEdit()
        self.cluster_timeout_edit.setPlaceholderText("e.g., 24:00:00")
        self.cluster_timeout_edit.setText(
            CONFIG["SlurmDefault"].get("timeout", "24:00:00")
        )
        self.cluster_timeout_edit.setMaximumWidth(100)
        cluster_config_layout.addWidget(self.cluster_timeout_edit)

        # self.cluster_use_module = QtWidgets.QCheckBox("Use p-w module")
        # self.cluster_use_module.setMaximumWidth(200)
        # self.cluster_use_module.setChecked(True)
        # self.cluster_use_module.setToolTip(
        #     "Use the miblab SLURM module for picasso-workflow (recommended). Otherwise, use Heinrich's repository."
        # )
        # # self.cluster_use_module.connect(self.on_cluster_use_module_state_change)
        # cluster_config_layout.addWidget(self.cluster_use_module)

        # Cluster configuration widgets
        cluster_settings_layout = QtWidgets.QHBoxLayout()
        cluster_settings_widget = QtWidgets.QWidget()
        cluster_settings_widget.setLayout(cluster_settings_layout)
        run_on_cluster_layout.addWidget(cluster_settings_widget)

        # user name on cluster login node
        cluster_settings_layout.addWidget(
            QtWidgets.QLabel(
                "User name on cluster login node (default '$USER' as provided locally):"
            )
        )
        self.cluster_username_edit = QtWidgets.QLineEdit()
        self.cluster_username_edit.setPlaceholderText("e.g., $USER")
        defaulttext = CONFIG.get("LoginNodeUserNames", "$USER")
        if isinstance(defaulttext, dict):
            defaulttext = defaulttext.get(
                list(CONFIG["SlurmLoginNodes"].keys())[0], "$USER"
            )
        self.cluster_username_edit.setText(defaulttext)
        # self.cluster_username_edit.setMaximumWidth(100)
        cluster_settings_layout.addWidget(self.cluster_username_edit)

        self.slurm_email_edit = QtWidgets.QLineEdit()
        self.slurm_email_edit.setPlaceholderText("e.g., you@institute.edu")
        defaulttext = CONFIG.get("SlurmDefault", {}).get("email", "")
        self.slurm_email_edit.setText(defaulttext)
        # self.cluster_username_edit.setMaximumWidth(100)
        cluster_settings_layout.addWidget(self.slurm_email_edit)

        slurm_buttons = QtWidgets.QHBoxLayout()
        self.slurm_buttons_widget = QtWidgets.QWidget()
        self.slurm_buttons_widget.setLayout(slurm_buttons)
        run_on_cluster_layout.addWidget(self.slurm_buttons_widget)

        estimate_start_button = QtWidgets.QPushButton(
            "Estimate Start on Cluster"
        )
        slurm_buttons.addWidget(estimate_start_button)
        estimate_start_button.clicked.connect(self.estimate_start)

        start_slurm_button = QtWidgets.QPushButton("Start Workflow on Cluster")
        slurm_buttons.addWidget(start_slurm_button)
        start_slurm_button.clicked.connect(self.start_slurm)

        queue_info_button = QtWidgets.QPushButton("Show Queue Info")
        queue_info_button.clicked.connect(self.on_show_queue_info)
        slurm_buttons.addWidget(queue_info_button)

        # Job management buttons
        job_management_buttons = QtWidgets.QHBoxLayout()
        job_management_widget = QtWidgets.QWidget()
        job_management_widget.setLayout(job_management_buttons)
        run_on_cluster_layout.addWidget(job_management_widget)

        cancel_job_button = QtWidgets.QPushButton("Cancel Job")
        cancel_job_button.clicked.connect(self.on_cancel_job)
        job_management_buttons.addWidget(cancel_job_button)

        job_status_button = QtWidgets.QPushButton("Show Job Status")
        job_status_button.clicked.connect(self.on_show_job_status)
        job_management_buttons.addWidget(job_status_button)

        list_jobs_button = QtWidgets.QPushButton("List All Jobs")
        list_jobs_button.clicked.connect(self.on_list_jobs)
        job_management_buttons.addWidget(list_jobs_button)

        # Job ID input field
        job_id_layout = QtWidgets.QHBoxLayout()
        job_id_widget = QtWidgets.QWidget()
        job_id_widget.setLayout(job_id_layout)
        run_on_cluster_layout.addWidget(job_id_widget)

        job_id_label = QtWidgets.QLabel("Current Job ID:")
        job_id_layout.addWidget(job_id_label)

        self.job_id_input = QtWidgets.QLineEdit()
        self.job_id_input.setPlaceholderText(
            "Enter job ID or auto-filled from submission"
        )
        job_id_layout.addWidget(self.job_id_input, stretch=1)

        # Display area for job information
        job_display_label = QtWidgets.QLabel("Job Information:")
        run_on_cluster_layout.addWidget(job_display_label)

        self.job_info_display = QtWidgets.QTextEdit()
        self.job_info_display.setReadOnly(True)
        self.job_info_display.setMaximumHeight(200)
        run_on_cluster_layout.addWidget(self.job_info_display)

        run_locally_tab = QtWidgets.QWidget()
        run_locally_layout = QtWidgets.QVBoxLayout(run_locally_tab)
        self.run_tabs.addTab(run_locally_tab, "Run locally")
        self.run_tabs.setTabEnabled(1, False)  # in development
        self.run_tabs.setTabToolTip(1, "Not Implemented yet")
        local_buttons = QtWidgets.QHBoxLayout()
        self.local_buttons_widget = QtWidgets.QWidget()
        self.local_buttons_widget.setLayout(local_buttons)
        run_locally_layout.addWidget(self.local_buttons_widget)
        start_locally_button = QtWidgets.QPushButton("Start Workflow locally")
        local_buttons.addWidget(start_locally_button)
        start_locally_button.clicked.connect(self.start_locally)

        # Results tab: browse a run folder, (re)generate its HTML report
        # from the saved state, and view/open it -- no Confluence needed.
        results_tab = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(results_tab)
        self.tabs.addTab(results_tab, "Results")
        self._build_results_tab(results_tab)

        # select files to process
        # Single Workflow only: choose how input data is provided - an
        # explicit file list, auto-detection from the results folder, or
        # no input files at all (the workflow runs once and its modules
        # load any data themselves).
        self.files_mode_label = QtWidgets.QLabel("Input files:")
        self.files_mode_combo = QtWidgets.QComboBox()
        self.files_mode_combo.addItems(
            [
                "Specify input files",
                "Auto-detect input files",
                "No input files (run once)",
            ]
        )
        self.files_mode_combo.setToolTip(
            "Specify input files: use the explicit file list below.\n"
            "Auto-detect input files: the generated script scans the "
            "results folder for DNA-PAINT raw files at run time.\n"
            "No input files: run the workflow exactly once with no "
            "dataset, for workflows whose modules load data themselves."
        )
        self.files_mode_combo.currentIndexChanged.connect(
            self._on_files_mode_changed
        )
        self.files_box.addWidget(self.files_mode_label, 0, 0)
        self.files_box.addWidget(self.files_mode_combo, 0, 1, 1, 2)

        # Create button container for dynamic button layout
        self.file_buttons_widget = QtWidgets.QWidget()
        self.file_buttons_layout = QtWidgets.QHBoxLayout(
            self.file_buttons_widget
        )
        self.file_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.files_box.addWidget(self.file_buttons_widget, 1, 0, 1, 3)

        # Initial buttons for Single Workflow (will be managed by _update_file_buttons)
        self.add_files_button = QtWidgets.QPushButton("Add files")
        self.add_files_button.clicked.connect(self.add_files)
        self.remove_files_button = QtWidgets.QPushButton("Remove selected")
        self.remove_files_button.clicked.connect(self.remove_selected_files)
        self.clear_files_button = QtWidgets.QPushButton("Clear list")
        self.clear_files_button.clicked.connect(self.clear_file_list)

        self.file_buttons_layout.addWidget(self.add_files_button)
        self.file_buttons_layout.addWidget(self.remove_files_button)
        self.file_buttons_layout.addWidget(self.clear_files_button)

        d = {"Name1": "/path/to/file1.txt", "Name2": "/path/to/file2.txt"}
        self.files_table = DroppableTableWidget()
        self.files_table.filesDropped.connect(self._on_files_dropped_table)
        self.files_table.setColumnCount(2)
        self.files_table.setHorizontalHeaderLabels(["Name", "File Path"])
        # Configure column stretching - Name column resizes to contents, File Path column stretches
        header = self.files_table.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        dict_to_table(d, self.files_table)
        # Store delegate as instance variable to prevent garbage collection
        self.file_path_delegate = FilePathDelegate(self)
        self.files_table.setItemDelegateForColumn(1, self.file_path_delegate)

        # Create QStackedWidget to hold all file selection widgets
        self.files_stack = QtWidgets.QStackedWidget()
        self.files_box.addWidget(self.files_stack, 2, 0, 1, 3)

        # Add existing table to stack (index 0 - Single Workflow)
        self.files_stack.addWidget(self.files_table)

        # Create tree for Aggregation Workflow (index 1)
        self.files_tree_agg = DroppableTreeWidget()
        self.files_tree_agg.filesDropped.connect(self._on_files_dropped_tree)
        self.files_tree_agg.fileMoved.connect(self._on_file_moved_tree)
        self.files_tree_agg.setColumnCount(3)
        self.files_tree_agg.setHeaderLabels(
            ["Dataset", "Channel", "File Path"]
        )
        header_agg = self.files_tree_agg.header()
        header_agg.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header_agg.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header_agg.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.file_path_delegate_tree_agg = FilePathDelegate(self)
        self.files_tree_agg.setItemDelegateForColumn(
            2, self.file_path_delegate_tree_agg
        )
        self.files_tree_agg.setAlternatingRowColors(True)
        self.files_tree_agg.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.files_stack.addWidget(self.files_tree_agg)

        # Create tree for Investigation Workflow (index 2)
        self.files_tree_inv = DroppableTreeWidget()
        self.files_tree_inv.filesDropped.connect(self._on_files_dropped_tree)
        self.files_tree_inv.fileMoved.connect(self._on_file_moved_tree)
        self.files_tree_inv.setColumnCount(4)
        self.files_tree_inv.setHeaderLabels(
            ["Dataset", "Channel", "File Path", "Condition"]
        )
        header_inv = self.files_tree_inv.header()
        header_inv.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header_inv.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        header_inv.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        header_inv.setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.file_path_delegate_tree_inv = FilePathDelegate(self)
        self.files_tree_inv.setItemDelegateForColumn(
            2, self.file_path_delegate_tree_inv
        )
        self.files_tree_inv.setAlternatingRowColors(True)
        self.files_tree_inv.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.files_stack.addWidget(self.files_tree_inv)

        # Initialize tree data structure
        self.tree_data = {
            "datasets": [],
            "channels": [],
            "file_paths": {},
            "conditions": {},
        }

        # Connect item changed signal for real-time updates
        self.files_tree_agg.itemChanged.connect(self._on_tree_item_changed)
        self.files_tree_inv.itemChanged.connect(self._on_tree_item_changed)

        # adding modules
        self.current_module = QtWidgets.QGroupBox("Current Module")
        current_layout = QtWidgets.QVBoxLayout(self.current_module)
        self.modules_box.addWidget(self.current_module, 0, 0)

        self.module_combobox = QtWidgets.QComboBox()
        self.module_combobox.addItem("Select module")
        self.module_combobox.addItems(
            self.module_descriptor.get_module_names()
        )
        self.module_combobox.currentTextChanged.connect(self.on_module_changed)
        current_layout.addWidget(self.module_combobox)
        # label describing the module selected
        self.current_module_desc = QtWidgets.QLabel("No module selected")
        current_layout.addWidget(self.current_module_desc)
        # parameters section (scrollable)
        module_parameters = QtWidgets.QWidget()
        self.module_parameters_layout = QtWidgets.QVBoxLayout(
            module_parameters
        )
        self.module_parameters_layout.addStretch()  # Push widgets to top

        # Wrap in scroll area
        parameters_scroll = QtWidgets.QScrollArea()
        parameters_scroll.setWidget(module_parameters)
        parameters_scroll.setWidgetResizable(True)
        parameters_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        parameters_scroll.setMinimumHeight(100)
        parameters_scroll.setMaximumHeight(300)
        current_layout.addWidget(parameters_scroll)
        # button to add the selected module
        self.add_module_button = QtWidgets.QPushButton("Add module")
        current_layout.addWidget(self.add_module_button)
        self.add_module_button.clicked.connect(self.add_module)
        self.add_module_button.setEnabled(False)

        # widget showing the workflow
        self.workflow_tabs = QtWidgets.QTabWidget()
        single_workflow_tab = QtWidgets.QWidget()
        single_workflow_layout = QtWidgets.QVBoxLayout(single_workflow_tab)
        self.workflow_tabs.addTab(
            single_workflow_tab, "Single Dataset Workflow"
        )
        self.single_workflow_list = QtWidgets.QListWidget()
        self.single_workflow_list.currentRowChanged.connect(
            self._on_workflow_selection_changed
        )
        single_workflow_layout.addWidget(self.single_workflow_list)

        aggregation_workflow_tab = QtWidgets.QWidget()
        aggregation_workflow_layout = QtWidgets.QVBoxLayout(
            aggregation_workflow_tab
        )
        self.workflow_tabs.addTab(
            aggregation_workflow_tab, "Aggregation Workflow"
        )
        self.aggregation_workflow_list = QtWidgets.QListWidget()
        self.aggregation_workflow_list.currentRowChanged.connect(
            self._on_workflow_selection_changed
        )
        aggregation_workflow_layout.addWidget(self.aggregation_workflow_list)
        # Add workflow tabs to modules box
        self.modules_box.addWidget(self.workflow_tabs, 1, 0)
        # Connect tab change signal
        self.workflow_tabs.currentChanged.connect(
            self._on_workflow_tab_changed
        )

        investigation_workflow_tab = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(investigation_workflow_tab)
        self.workflow_tabs.addTab(investigation_workflow_tab, "Investigation")

        # Set initial tab states based on default workflow type
        self._on_workflow_type_changed(self.workflow_type.currentIndex())

        # buttons for workflow manipulation
        workflow_buttons = QtWidgets.QHBoxLayout()
        self.workflow_buttons_widget = QtWidgets.QWidget()
        self.workflow_buttons_widget.setLayout(workflow_buttons)
        self.modules_box.addWidget(self.workflow_buttons_widget, 2, 0)
        remove_selected_button = QtWidgets.QPushButton("Remove selected")
        workflow_buttons.addWidget(remove_selected_button)
        remove_selected_button.clicked.connect(self.remove_selected)
        move_up_button = QtWidgets.QPushButton("Move up")
        workflow_buttons.addWidget(move_up_button)
        move_up_button.clicked.connect(self.move_up)
        move_down_button = QtWidgets.QPushButton("Move down")
        workflow_buttons.addWidget(move_down_button)
        move_down_button.clicked.connect(self.move_down)

        self.addl_options_widget = QtWidgets.QWidget()
        self.addl_options_layout = QtWidgets.QHBoxLayout(
            self.addl_options_widget
        )
        self.addl_options_layout.setContentsMargins(0, 0, 0, 0)
        self.modules_box.addWidget(self.addl_options_widget, 3, 0)

        # Add options
        self.always_save = QtWidgets.QCheckBox(
            "Save localizations after every module."
        )
        self.addl_options_layout.addWidget(self.always_save)

        # resize the widgets
        # Set fixed size for the group box
        self.current_module.setMinimumSize(500, 300)

        # Initially disable file and module widgets until results folder is selected
        self._set_widgets_enabled(False)

    def add_dataset(self):
        """Add new dataset with prompt for name."""
        dataset_name, ok = QtWidgets.QInputDialog.getText(
            self, "Add Dataset", "Enter dataset name:"
        )

        if not ok or not dataset_name.strip():
            return

        dataset_name = dataset_name.strip()

        # Validate no underscores
        if "_" in dataset_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Name",
                "Dataset names cannot contain underscores.\n\n"
                "Underscores are used to separate dataset and channel names "
                "in the {dataset}_{channel} format.",
            )
            return

        if dataset_name in self.tree_data["datasets"]:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate", f"Dataset '{dataset_name}' exists."
            )
            return

        # Add to data structure
        self.tree_data["datasets"].append(dataset_name)
        self.tree_data["file_paths"][dataset_name] = {}

        # Initialize all existing channels with empty paths
        for channel in self.tree_data["channels"]:
            self.tree_data["file_paths"][dataset_name][channel] = ""

        # Add to tree
        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.add_dataset", dataset=dataset_name
        )

    def add_channel(self):
        """Add new channel to ALL datasets."""
        channel_name, ok = QtWidgets.QInputDialog.getText(
            self, "Add Channel", "Enter channel name:"
        )

        if not ok or not channel_name.strip():
            return

        channel_name = channel_name.strip()

        # Validate no underscores
        if "_" in channel_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Name",
                "Channel names cannot contain underscores.\n\n"
                "Underscores are used to separate dataset and channel names "
                "in the {dataset}_{channel} format.",
            )
            return

        if channel_name in self.tree_data["channels"]:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate", f"Channel '{channel_name}' exists."
            )
            return

        # Add to channels list
        self.tree_data["channels"].append(channel_name)

        # Add to all datasets
        for dataset in self.tree_data["datasets"]:
            self.tree_data["file_paths"][dataset][channel_name] = ""

        # Refresh tree
        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.add_channel", channel=channel_name
        )

    def remove_channel(self):
        """Remove selected channel from ALL datasets."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        selected = current_tree.selectedItems()

        if not selected:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "Please select a channel to remove."
            )
            return

        # Get channel name from first selected item
        # Channel items are child items (have a parent)
        item = selected[0]
        if item.parent() is None:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid Selection",
                "Please select a channel (not a dataset) to remove.",
            )
            return

        channel_name = item.text(1)

        if not channel_name:
            return

        # Confirm removal
        msg = f"Remove channel '{channel_name}' from all datasets?"
        if (
            QtWidgets.QMessageBox.question(self, "Confirm", msg)
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return

        # Remove from channels list
        if channel_name in self.tree_data["channels"]:
            self.tree_data["channels"].remove(channel_name)

        # Remove from all datasets
        for dataset in self.tree_data["datasets"]:
            if channel_name in self.tree_data["file_paths"][dataset]:
                del self.tree_data["file_paths"][dataset][channel_name]

        # Refresh tree
        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.remove_channel", channel=channel_name
        )

    def rename_dataset(self):
        """Rename selected dataset with validation (no underscores)."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        selected = current_tree.selectedItems()

        if not selected:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "Please select a dataset to rename."
            )
            return

        # Get dataset item (must be top-level, no parent)
        item = selected[0]
        if item.parent() is not None:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid Selection",
                "Please select a dataset (not a channel) to rename.",
            )
            return

        old_name = item.text(0)

        # Prompt for new name
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Dataset",
            f"Enter new name for '{old_name}':",
            text=old_name,
        )

        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()

        # Validate no underscores
        if "_" in new_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Name",
                "Dataset names cannot contain underscores.\n\n"
                "Underscores are used to separate dataset and channel names "
                "in the {dataset}_{channel} format.",
            )
            return

        # Check if already exists
        if new_name in self.tree_data["datasets"] and new_name != old_name:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate", f"Dataset '{new_name}' already exists."
            )
            return

        # Update datasets list
        idx = self.tree_data["datasets"].index(old_name)
        self.tree_data["datasets"][idx] = new_name

        # Update file_paths (rename key)
        self.tree_data["file_paths"][new_name] = self.tree_data[
            "file_paths"
        ].pop(old_name)

        # Update conditions if exists (Investigation workflow)
        if old_name in self.tree_data["conditions"]:
            self.tree_data["conditions"][new_name] = self.tree_data[
                "conditions"
            ].pop(old_name)

        # Refresh tree
        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.rename_dataset", old=old_name, new=new_name
        )

    def _channel_name_from_item(self, item):
        """Return the channel name a tree item represents, by position.

        ``_populate_tree_from_data`` creates one child per entry in
        ``tree_data["channels"]`` in that exact order, so a channel item's
        index under its dataset parent is its index into the channels list.
        Using the position (rather than ``item.text(1)``) is authoritative:
        it is correct even when the cell was just edited inline and column 1
        already holds the new, not-yet-committed name.

        Returns None if the item is not a channel row or is out of range.
        """
        dataset_item = item.parent()
        if dataset_item is None:
            return None
        child_index = dataset_item.indexOfChild(item)
        channels = self.tree_data["channels"]
        if 0 <= child_index < len(channels):
            return channels[child_index]
        return None

    def _rename_channel(self, old_name, new_name):
        """Rename a channel across ALL datasets, with validation.

        Shared by the "Rename Channel" button and inline (double-click)
        editing so both go through identical logic. On any validation
        failure the tree is repopulated from ``tree_data`` (reverting an
        inline edit) and False is returned. Returns True on success.
        """
        new_name = (new_name or "").strip()

        # No-op / empty: just resync the view (reverts a blanked inline edit).
        if not new_name or new_name == old_name:
            self._populate_tree_from_data()
            return False

        # Validate no underscores
        if "_" in new_name:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Name",
                "Channel names cannot contain underscores.\n\n"
                "Underscores are used to separate dataset and channel names "
                "in the {dataset}_{channel} format.",
            )
            self._populate_tree_from_data()
            return False

        # Check if already exists
        if new_name in self.tree_data["channels"]:
            QtWidgets.QMessageBox.warning(
                self, "Duplicate", f"Channel '{new_name}' already exists."
            )
            self._populate_tree_from_data()
            return False

        if old_name not in self.tree_data["channels"]:
            logger.warning(
                f"Cannot rename channel: '{old_name}' not found in "
                f"tree_data channels {self.tree_data['channels']}"
            )
            self._populate_tree_from_data()
            return False

        # Update channels list
        idx = self.tree_data["channels"].index(old_name)
        self.tree_data["channels"][idx] = new_name

        # Update file_paths for all datasets (rename inner key)
        for dataset in self.tree_data["datasets"]:
            if old_name in self.tree_data["file_paths"][dataset]:
                self.tree_data["file_paths"][dataset][new_name] = (
                    self.tree_data["file_paths"][dataset].pop(old_name)
                )

        # Refresh tree
        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.rename_channel", old=old_name, new=new_name
        )
        return True

    def rename_channel(self):
        """Rename selected channel in ALL datasets with validation (no underscores)."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        # Use the focused item, not selectedItems()[0]: with
        # ExtendedSelection the latter returns items in tree order, so a
        # lingering selection on the first channel would always win.
        item = current_tree.currentItem()
        if item is None:
            selected = current_tree.selectedItems()
            item = selected[0] if selected else None

        if item is None:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "Please select a channel to rename."
            )
            return

        # Get channel item (must have parent)
        if item.parent() is None:
            QtWidgets.QMessageBox.information(
                self,
                "Invalid Selection",
                "Please select a channel (not a dataset) to rename.",
            )
            return

        old_name = self._channel_name_from_item(item)
        if old_name is None:
            return

        # Prompt for new name
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Channel",
            f"Enter new name for channel '{old_name}':",
            text=old_name,
        )

        if not ok or not new_name.strip():
            return

        self._rename_channel(old_name, new_name)

    def remove_tree_items(self):
        """Remove selected datasets or channels."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        selected = current_tree.selectedItems()

        if not selected:
            return

        datasets_to_remove = []

        for item in selected:
            if item.parent() is None:  # Top-level = dataset
                dataset_name = item.text(0)
                datasets_to_remove.append(dataset_name)

        if not datasets_to_remove:
            return

        msg = f"Remove {len(datasets_to_remove)} dataset(s)?"
        if (
            QtWidgets.QMessageBox.question(self, "Confirm", msg)
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return

        for dataset_name in datasets_to_remove:
            self.tree_data["datasets"].remove(dataset_name)
            del self.tree_data["file_paths"][dataset_name]
            if dataset_name in self.tree_data["conditions"]:
                del self.tree_data["conditions"][dataset_name]

        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.remove_datasets", datasets=datasets_to_remove
        )

    def clear_tree(self):
        """Clear all items from the tree."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        if current_tree.topLevelItemCount() == 0:
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Tree",
            "Clear all datasets and channels?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        current_tree.clear()
        self.tree_data = {
            "datasets": [],
            "channels": [],
            "file_paths": {},
            "conditions": {},
        }
        self._log_workflow_config_event("tree.clear")

    def _get_current_tree_widget(self):
        """Get currently visible tree widget based on workflow type."""
        workflow_type = self.workflow_type.currentIndex()
        if workflow_type == 1:
            return self.files_tree_agg
        elif workflow_type == 2:
            return self.files_tree_inv
        return None

    def _populate_tree_from_data(self):
        """Populate tree widget from self.tree_data."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        # Block signals during population to avoid triggering itemChanged
        current_tree.blockSignals(True)
        try:
            current_tree.clear()

            for dataset in self.tree_data["datasets"]:
                dataset_item = QtWidgets.QTreeWidgetItem(current_tree)
                dataset_item.setText(0, dataset)
                dataset_item.setFlags(
                    dataset_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                )

                for channel in self.tree_data["channels"]:
                    channel_item = QtWidgets.QTreeWidgetItem(dataset_item)
                    channel_item.setText(1, channel)
                    channel_item.setText(
                        2,
                        self.tree_data["file_paths"][dataset].get(channel, ""),
                    )

                    # Investigation workflow: add condition
                    if self.workflow_type.currentIndex() == 2:
                        channel_item.setText(
                            3, self.tree_data["conditions"].get(dataset, "")
                        )

                    # Make only File Path (and Condition) editable
                    flags = channel_item.flags() | Qt.ItemFlag.ItemIsEditable
                    channel_item.setFlags(flags)

                dataset_item.setExpanded(True)

        finally:
            current_tree.blockSignals(False)

        self._update_validation_display()

    def _on_tree_item_changed(self, item, column):
        """Handle tree item changes to update data structure."""
        if item.parent() is None:  # Dataset item
            return

        dataset_item = item.parent()
        dataset_name = dataset_item.text(0)
        channel_name = item.text(1)

        # Update data structure
        if column == 1:  # Channel renamed inline (double-click edit)
            # Resolve the pre-edit name by position (column 1 already holds
            # the new text), then funnel through the same core as the
            # "Rename Channel" button so behaviour is identical.
            old_name = self._channel_name_from_item(item)
            new_name = item.text(1)
            if old_name is not None:
                self._rename_channel(old_name, new_name)
            else:
                logger.debug(
                    "Could not resolve channel for inline rename to "
                    f"'{new_name}'"
                )
            return
        if column == 2:  # File Path
            file_path = item.text(2)
            self.tree_data["file_paths"][dataset_name][
                channel_name
            ] = file_path
            self._log_workflow_config_event(
                "tree.set_file_path",
                dataset=dataset_name,
                channel=channel_name,
                path=file_path,
            )
        elif column == 3:  # Condition (Investigation only)
            condition = item.text(3)
            self.tree_data["conditions"][dataset_name] = condition
            self._log_workflow_config_event(
                "tree.set_condition",
                dataset=dataset_name,
                condition=condition,
            )

        self._update_validation_display()

    def _on_files_dropped_table(self, file_paths):
        """Handle files dropped onto the Single Dataset table."""
        for path in file_paths:
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)
            # Use basename as default name
            name = os.path.basename(path)
            self.files_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.files_table.setItem(row, 1, QtWidgets.QTableWidgetItem(path))
        self._log_workflow_config_event(
            "files_table.drop", n_added=len(file_paths), paths=list(file_paths)
        )

    def _on_files_dropped_tree(self, file_paths, target_item):
        """Handle files dropped onto the Aggregation/Investigation tree."""
        if not target_item:
            return

        # Determine if target is a dataset or channel item
        if target_item.parent() is None:
            # Dropped on dataset item - distribute files across channels
            dataset_name = target_item.text(0)
            channels = self.tree_data["channels"]
            if not channels:
                return

            for i, path in enumerate(file_paths):
                if i < len(channels):
                    channel = channels[i]
                    self.tree_data["file_paths"][dataset_name][channel] = path
        else:
            # Dropped on channel item - assign files starting from this channel
            dataset_item = target_item.parent()
            dataset_name = dataset_item.text(0)
            target_channel = target_item.text(1)

            channels = self.tree_data["channels"]
            try:
                start_idx = channels.index(target_channel)
            except ValueError:
                return

            for i, path in enumerate(file_paths):
                channel_idx = start_idx + i
                if channel_idx < len(channels):
                    channel = channels[channel_idx]
                    self.tree_data["file_paths"][dataset_name][channel] = path

        self._populate_tree_from_data()
        self._log_workflow_config_event(
            "tree.drop", n_added=len(file_paths), paths=list(file_paths)
        )

    def _on_file_moved_tree(self, source_item, target_item):
        """Handle file moved between channels within the tree."""
        if not source_item or not target_item:
            return

        # Both should be channel items (have parents)
        if source_item.parent() is None or target_item.parent() is None:
            return

        source_dataset = source_item.parent().text(0)
        source_channel = source_item.text(1)
        source_path = source_item.text(2)

        target_dataset = target_item.parent().text(0)
        target_channel = target_item.text(1)

        # Move the file path
        if source_path:
            # Clear source
            self.tree_data["file_paths"][source_dataset][source_channel] = ""
            # Set target
            self.tree_data["file_paths"][target_dataset][
                target_channel
            ] = source_path
            self._populate_tree_from_data()
            self._log_workflow_config_event(
                "tree.move_file",
                source=(source_dataset, source_channel),
                target=(target_dataset, target_channel),
                path=source_path,
            )

    def _update_validation_display(self):
        """Update visual indicators for missing file paths."""
        current_tree = self._get_current_tree_widget()
        if not current_tree:
            return

        # setForeground emits itemChanged. Without blocking, these purely
        # cosmetic colour updates re-enter _on_tree_item_changed (e.g. the
        # column-1 path calls _rename_channel -> _populate_tree_from_data ->
        # clear()), deleting the items this loop is still iterating over.
        current_tree.blockSignals(True)
        try:
            root = current_tree.invisibleRootItem()

            for dataset_idx in range(root.childCount()):
                dataset_item = root.child(dataset_idx)

                for channel_idx in range(dataset_item.childCount()):
                    channel_item = dataset_item.child(channel_idx)
                    file_path = channel_item.text(2)

                    # Red text if empty, black otherwise
                    color = (
                        QtGui.QColor("red")
                        if not file_path.strip()
                        else QtGui.QColor("black")
                    )

                    for col in range(channel_item.columnCount()):
                        channel_item.setForeground(col, color)
        finally:
            current_tree.blockSignals(False)

    def _simple_to_tree(self):
        """Convert simple table to tree format using {dataset}_{channel} naming."""
        table_data = {}

        for row in range(self.files_table.rowCount()):
            name_item = self.files_table.item(row, 0)
            path_item = self.files_table.item(row, 1)

            if not name_item or not path_item:
                continue

            name = name_item.text()
            path = path_item.text()

            # Parse using rightmost underscore
            if "_" in name:
                dataset, channel = name.rsplit("_", 1)
            else:
                dataset, channel = name, "default"

            if dataset not in table_data:
                table_data[dataset] = {}
            table_data[dataset][channel] = path

        # Build tree data
        self.tree_data["datasets"] = list(table_data.keys())
        self.tree_data["channels"] = sorted(
            set(ch for ds in table_data.values() for ch in ds.keys())
        )
        self.tree_data["file_paths"] = table_data

        # Ensure all datasets have all channels
        for dataset in self.tree_data["datasets"]:
            for channel in self.tree_data["channels"]:
                if channel not in self.tree_data["file_paths"][dataset]:
                    self.tree_data["file_paths"][dataset][channel] = ""

        self._populate_tree_from_data()

    def _tree_to_simple(self):
        """Convert tree to simple table using {dataset}_{channel} naming."""
        self.files_table.setRowCount(0)

        row = 0
        for dataset in self.tree_data["datasets"]:
            for channel in self.tree_data["channels"]:
                file_path = self.tree_data["file_paths"][dataset].get(
                    channel, ""
                )
                name = f"{dataset}_{channel}"

                self.files_table.insertRow(row)
                self.files_table.setItem(
                    row, 0, QtWidgets.QTableWidgetItem(name)
                )
                self.files_table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(file_path)
                )
                row += 1

    def _validate_tree_data(self):
        """Validate tree data and return list of errors."""
        errors = []

        for dataset in self.tree_data["datasets"]:
            for channel in self.tree_data["channels"]:
                path = self.tree_data["file_paths"][dataset].get(channel, "")

                if not path.strip():
                    errors.append(f"Missing path: {dataset}_{channel}")
                # elif not os.path.exists(path):
                #     errors.append(f"File not found: {path} ({dataset}_{channel})")

        return errors

    # marker comment emitted into a generated "no input files" workflow
    # script, used to restore the combo box when the script is loaded.
    _NOFILES_MARKER = "# run workflow once without input files"

    def _on_files_mode_changed(self, _index):
        """Handle the input-files mode combo box being changed."""
        self._apply_files_mode_state()

    def _apply_files_mode_state(self):
        """Enable the explicit file table/buttons only in 'Specify input
        files' mode (combo index 0).

        Only relevant for Single Workflow - the combo box is hidden for
        the tree-based workflow types.
        """
        if self.workflow_type.currentIndex() != 0:
            return
        explicit = self.files_mode_combo.currentIndex() == 0
        self.files_table.setEnabled(explicit)
        for name in (
            "add_files_button",
            "remove_files_button",
            "clear_files_button",
        ):
            btn = getattr(self, name, None)
            if btn is not None:
                try:
                    btn.setEnabled(explicit)
                except RuntimeError:
                    pass

    def _apply_files_mode_from_source(self, source_text):
        """Restore the input-files mode combo box from a loaded workflow
        script.

        An auto-detect script calls find_dnapaint_raw at run time; a
        no-files script carries the _NOFILES_MARKER comment. Both modes
        are Single-Workflow only, so their presence also pins the
        workflow type to Single Workflow. Must be called after the
        workflow type has been set by the loader.
        """
        if "find_dnapaint_raw" in source_text:
            mode = 1  # Auto-detect input files
        elif self._NOFILES_MARKER in source_text:
            mode = 2  # No input files
        else:
            mode = 0  # Specify input files
        if mode != 0 and self.workflow_type.currentIndex() != 0:
            self.workflow_type.setCurrentIndex(0)
        self.files_mode_combo.setCurrentIndex(mode)
        self._apply_files_mode_state()

    def _update_file_buttons(self, workflow_type_index):
        """Update button layout based on workflow type."""
        # Clear existing buttons
        while self.file_buttons_layout.count():
            item = self.file_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.find_files_button = QtWidgets.QPushButton("Find in folder")
        self.find_files_button.clicked.connect(self.find_dataset_files)
        self.file_buttons_layout.addWidget(self.find_files_button)

        if workflow_type_index == 0:  # Single Workflow
            self.add_files_button = QtWidgets.QPushButton("Add files")
            self.add_files_button.clicked.connect(self.add_files)
            self.remove_files_button = QtWidgets.QPushButton("Remove selected")
            self.remove_files_button.clicked.connect(
                self.remove_selected_files
            )
            self.clear_files_button = QtWidgets.QPushButton("Clear list")
            self.clear_files_button.clicked.connect(self.clear_file_list)

            self.file_buttons_layout.addWidget(self.add_files_button)
            self.file_buttons_layout.addWidget(self.remove_files_button)
            self.file_buttons_layout.addWidget(self.clear_files_button)

            # Buttons were just recreated - re-apply the files-mode state
            self._apply_files_mode_state()

        else:  # Aggregation or Investigation
            self.add_dataset_button = QtWidgets.QPushButton("Add Dataset")
            self.add_dataset_button.clicked.connect(self.add_dataset)
            self.add_channel_button = QtWidgets.QPushButton("Add Channel")
            self.add_channel_button.clicked.connect(self.add_channel)
            self.rename_dataset_button = QtWidgets.QPushButton(
                "Rename Dataset"
            )
            self.rename_dataset_button.clicked.connect(self.rename_dataset)
            self.rename_channel_button = QtWidgets.QPushButton(
                "Rename Channel"
            )
            self.rename_channel_button.clicked.connect(self.rename_channel)
            self.remove_channel_button = QtWidgets.QPushButton(
                "Remove Channel"
            )
            self.remove_channel_button.clicked.connect(self.remove_channel)
            self.remove_dataset_button = QtWidgets.QPushButton(
                "Remove Dataset"
            )
            self.remove_dataset_button.clicked.connect(self.remove_tree_items)
            self.clear_tree_button = QtWidgets.QPushButton("Clear")
            self.clear_tree_button.clicked.connect(self.clear_tree)

            self.file_buttons_layout.addWidget(self.add_dataset_button)
            self.file_buttons_layout.addWidget(self.add_channel_button)
            self.file_buttons_layout.addWidget(self.rename_dataset_button)
            self.file_buttons_layout.addWidget(self.rename_channel_button)
            self.file_buttons_layout.addWidget(self.remove_channel_button)
            self.file_buttons_layout.addWidget(self.remove_dataset_button)
            self.file_buttons_layout.addWidget(self.clear_tree_button)

    def find_dataset_files(self):
        """Find dataset files (.tif, .ome.tif, .nd2) in a selected folder."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
        )
        if folder:
            from picasso_workflow.metaworkflow import find_dnapaint_raw

            datasets, _ = find_dnapaint_raw(folder)
            if datasets:
                workflow_type_index = self.workflow_type.currentIndex()
                if workflow_type_index == 0:
                    # Single workflow
                    for key, paths in datasets.items():
                        row = self.files_table.rowCount()
                        self.files_table.insertRow(row)
                        self.files_table.setItem(
                            row, 0, QtWidgets.QTableWidgetItem(key)
                        )
                        if isinstance(paths, list):
                            if len(paths) == 1:
                                self.files_table.setItem(
                                    row,
                                    1,
                                    QtWidgets.QTableWidgetItem(paths[0]),
                                )
                            else:
                                self.files_table.setItem(
                                    row,
                                    1,
                                    QtWidgets.QTableWidgetItem(repr(paths)),
                                )
                        else:
                            self.files_table.setItem(
                                row, 1, QtWidgets.QTableWidgetItem(paths)
                            )
                else:
                    # Aggregation / Investigation workflow
                    if not self.tree_data.get("channels"):
                        self.tree_data["channels"] = ["ch1"]

                    for key, paths in datasets.items():
                        # Ensure unique key
                        unique_key = key
                        counter = 1
                        while unique_key in self.tree_data.get("datasets", []):
                            unique_key = f"{key}_{counter}"
                            counter += 1

                        if "datasets" not in self.tree_data:
                            self.tree_data["datasets"] = []
                        if "conditions" not in self.tree_data:
                            self.tree_data["conditions"] = {}
                        if "file_paths" not in self.tree_data:
                            self.tree_data["file_paths"] = {}

                        self.tree_data["datasets"].append(unique_key)
                        self.tree_data["conditions"][unique_key] = ""
                        self.tree_data["file_paths"][unique_key] = {}

                        # Assign to the first channel
                        target_channel = self.tree_data["channels"][0]

                        if isinstance(paths, list):
                            if len(paths) == 1:
                                self.tree_data["file_paths"][unique_key][
                                    target_channel
                                ] = paths[0]
                            else:
                                self.tree_data["file_paths"][unique_key][
                                    target_channel
                                ] = repr(paths)
                        else:
                            self.tree_data["file_paths"][unique_key][
                                target_channel
                            ] = paths

                    self._populate_tree_from_data()
            else:
                QtWidgets.QMessageBox.information(
                    self,
                    "No files found",
                    f"Could not find any supported dataset files in {folder}",
                )

    def _set_widgets_enabled(self, enabled):
        """Enable or disable file and module widgets based on results folder selection."""
        self.workflow_type.setEnabled(enabled)
        # Files box widgets
        self.files_mode_combo.setEnabled(enabled)
        try:
            self.add_files_button.setEnabled(enabled)
            self.remove_files_button.setEnabled(enabled)
            self.clear_files_button.setEnabled(enabled)
            # self.files_table.setEnabled(enabled)
        except RuntimeError:
            # we're not in Single Workflow mode
            pass

        # Modules box widgets
        self.current_module.setEnabled(enabled)
        self.workflow_tabs.setEnabled(enabled)
        self.workflow_buttons_widget.setEnabled(enabled)

        # runing config
        self.run_tabs.setEnabled(enabled)

        # Keep the explicit file table disabled in non-explicit modes
        if enabled:
            self._apply_files_mode_state()

    # ------------------------------------------------------------------
    # Workflow-configuration logging helpers
    # ------------------------------------------------------------------
    # These produce a compact, structured snapshot of the parts of the GUI
    # state that get baked into the generated start_workflow.py script:
    # the input-files configuration (explicit table or tree, depending on
    # workflow type) and the module list with their parameter dicts.
    # Every user-driven mutation of either side calls
    # ``_log_workflow_config_event`` so the logfile holds an audit trail of
    # how the configuration evolved before submission.
    def _files_config_snapshot(self):
        """Return a dict describing the current input-files configuration.

        Schema:
            workflow_type: "Single" | "Aggregation" | "Investigation"
            files_mode:    "explicit" | "auto_detect" | "no_input_files"
                           (only meaningful for Single Workflow)
            files:         list[{name, path}]                (Single)
                           list[{dataset, channel, path, condition?}] (Tree)
        """
        try:
            workflow_type_index = self.workflow_type.currentIndex()
        except AttributeError:
            # Called before UI fully constructed; just return what we can.
            workflow_type_index = 0
        type_name = {0: "Single", 1: "Aggregation", 2: "Investigation"}.get(
            workflow_type_index, f"unknown({workflow_type_index})"
        )

        snapshot = {"workflow_type": type_name}

        if workflow_type_index == 0:
            try:
                mode_index = self.files_mode_combo.currentIndex()
            except AttributeError:
                mode_index = 0
            snapshot["files_mode"] = {
                0: "explicit",
                1: "auto_detect",
                2: "no_input_files",
            }.get(mode_index, f"unknown({mode_index})")
            files = []
            if hasattr(self, "files_table"):
                for row in range(self.files_table.rowCount()):
                    name_item = self.files_table.item(row, 0)
                    path_item = self.files_table.item(row, 1)
                    files.append(
                        {
                            "name": name_item.text() if name_item else "",
                            "path": path_item.text() if path_item else "",
                        }
                    )
            snapshot["files"] = files
        else:
            tree = getattr(self, "tree_data", None) or {}
            datasets = list(tree.get("datasets", []))
            channels = list(tree.get("channels", []))
            file_paths = tree.get("file_paths", {}) or {}
            conditions = tree.get("conditions", {}) or {}
            entries = []
            for ds in datasets:
                for ch in channels:
                    entry = {
                        "dataset": ds,
                        "channel": ch,
                        "path": file_paths.get(ds, {}).get(ch, ""),
                    }
                    if workflow_type_index == 2:
                        entry["condition"] = conditions.get(ds, "")
                    entries.append(entry)
            snapshot["datasets"] = datasets
            snapshot["channels"] = channels
            snapshot["files"] = entries
        return snapshot

    @staticmethod
    def _yaml_safe(value):
        """Recursively convert tuples to lists so the structure can be
        serialized with yaml.safe_dump (which has no Python-tuple
        representer). Module parameters often hold command tuples like
        ``('$map', 'filepath')`` that would otherwise raise."""
        if isinstance(value, dict):
            return {k: Window._yaml_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Window._yaml_safe(v) for v in value]
        return value

    def _modules_config_snapshot(self):
        """Return a dict with the current single and aggregation modules.

        Module entries are ``[name, params_dict]`` so the logged output
        stays a single line per module after yaml.safe_dump.
        """
        return {
            "single_workflow_modules": [
                [name, Window._yaml_safe(params)]
                for name, params in getattr(
                    self, "single_workflow_modules", []
                )
            ],
            "aggregation_workflow_modules": [
                [name, Window._yaml_safe(params)]
                for name, params in getattr(
                    self, "aggregation_workflow_modules", []
                )
            ],
        }

    def _log_workflow_config_event(self, event, **details):
        """Log a workflow-configuration mutation.

        The one-line INFO message names the event and includes a few
        compact stats; the DEBUG follow-ups dump the full files + modules
        snapshots so the logfile can be replayed to reconstruct any state.
        """
        try:
            files_snapshot = self._files_config_snapshot()
        except Exception as exc:  # pragma: no cover - defensive
            files_snapshot = {"error": repr(exc)}
        try:
            modules_snapshot = self._modules_config_snapshot()
        except Exception as exc:  # pragma: no cover - defensive
            modules_snapshot = {"error": repr(exc)}

        n_files = (
            len(files_snapshot.get("files", []))
            if isinstance(files_snapshot, dict)
            else "?"
        )
        n_sgl = len(modules_snapshot.get("single_workflow_modules", []))
        n_agg = len(modules_snapshot.get("aggregation_workflow_modules", []))
        detail_str = (
            " | " + ", ".join(f"{k}={v!r}" for k, v in details.items())
            if details
            else ""
        )
        logger.info(
            f"workflow-config: {event} "
            f"[type={files_snapshot.get('workflow_type', '?')}, "
            f"files={n_files}, sgl_modules={n_sgl}, "
            f"agg_modules={n_agg}]"
            f"{detail_str}"
        )
        logger.debug(
            "workflow-config files snapshot:\n"
            + yaml.safe_dump(files_snapshot, sort_keys=False)
        )
        logger.debug(
            "workflow-config modules snapshot:\n"
            + yaml.safe_dump(modules_snapshot, sort_keys=False)
        )

    def add_files(self):
        """Add a new row to the table below the selected row or at the end."""
        # Determine insertion position
        selected_indexes = self.files_table.selectionModel().selectedIndexes()
        if selected_indexes:
            # Get the maximum row number from selected cells and insert below it
            max_selected_row = max(index.row() for index in selected_indexes)
            insert_position = max_selected_row + 1
        else:
            # No selection, insert at the end
            insert_position = self.files_table.rowCount()

        # Insert new row
        self.files_table.insertRow(insert_position)

        # Add empty items to the new row
        self.files_table.setItem(
            insert_position, 0, QtWidgets.QTableWidgetItem("")
        )
        self.files_table.setItem(
            insert_position, 1, QtWidgets.QTableWidgetItem("")
        )
        self._log_workflow_config_event(
            "files_table.add_row", row=insert_position
        )

    def remove_selected_files(self):
        """Remove the selected row(s) from the table based on any selected cells."""
        selected_indexes = self.files_table.selectionModel().selectedIndexes()
        if not selected_indexes:
            return
        # Get unique row numbers from selected cells
        selected_row_numbers = sorted(
            set(index.row() for index in selected_indexes), reverse=True
        )
        # Remove rows in reverse order to avoid index shifting issues
        for row in selected_row_numbers:
            self.files_table.removeRow(row)
        self._log_workflow_config_event(
            "files_table.remove_rows", removed_rows=selected_row_numbers
        )

    def clear_file_list(self):
        """Clear all rows from the table."""
        if self.files_table.rowCount() == 0:
            return
        self.files_table.setRowCount(0)
        self._log_workflow_config_event("files_table.clear")

    # ------------------------------------------------------------------
    # Results tab: HTML report viewer
    # ------------------------------------------------------------------
    def _build_results_tab(self, results_tab):
        """Populate the Results tab with the HTML report viewer.

        Lets the user pick a run's result folder, (re)generate its local
        HTML report from the saved state and view it embedded (or open it in
        a browser), plus a per-module status overview.
        """
        layout = results_tab.layout()

        intro = QtWidgets.QLabel(
            "Pick a run found under the results folder to (re)generate and "
            "view its local HTML report."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run:"))
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.run_combo.setToolTip(
            "Runs (subfolders of the results folder) that contain a saved "
            "workflow state."
        )
        run_row.addWidget(self.run_combo, 1)
        rescan_button = QtWidgets.QPushButton("Rescan")
        rescan_button.setToolTip("Rescan the results folder for runs.")
        rescan_button.clicked.connect(self._results_scan_runs)
        run_row.addWidget(rescan_button)
        layout.addLayout(run_row)

        button_row = QtWidgets.QHBoxLayout()
        generate_button = QtWidgets.QPushButton(
            "Generate / Refresh HTML report"
        )
        generate_button.clicked.connect(self._results_refresh)
        button_row.addWidget(generate_button)
        open_button = QtWidgets.QPushButton("Open in browser")
        open_button.clicked.connect(self._results_open_in_browser)
        button_row.addWidget(open_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.report_status = QtWidgets.QTreeWidget()
        self.report_status.setHeaderLabels(
            ["Module", "Status", "Duration [s]"]
        )
        self.report_status.setRootIsDecorated(False)
        self.report_status.setMaximumHeight(180)
        layout.addWidget(self.report_status)

        self.report_view, self._report_view_is_web = self._make_report_view()
        layout.addWidget(self.report_view, 1)
        self._report_path = None

        # populate the run list from the currently configured results folder
        self._results_scan_runs()

    @staticmethod
    def _run_kind(folder):
        """Return 'aggregation', 'single' or None for a candidate folder."""
        if os.path.isfile(
            os.path.join(folder, "AggregationWorkflowRunner.yaml")
        ):
            return "aggregation"
        if os.path.isfile(os.path.join(folder, "WorkflowRunner.yaml")):
            return "single"
        return None

    def _find_runs(self, base, maxdepth=4):
        """Find run folders at or below ``base`` (bounded, depth-first).

        A run folder is one holding a saved workflow state. The search
        descends through intermediate folders (e.g. an ``AnalysisResults-*``
        wrapper) but does not descend into a run once found -- so an
        aggregation run is listed without flooding the list with its module
        subfolders or per-dataset children (those are reachable from the
        aggregation report's links).
        """
        skip = {"assets", "__pycache__", "logs", ".git", ".ipynb_checkpoints"}
        runs = []

        def walk(folder, depth):
            try:
                entries = sorted(os.listdir(folder))
            except OSError:
                return
            for name in entries:
                if name in skip:
                    continue
                sub = os.path.join(folder, name)
                if not os.path.isdir(sub):
                    continue
                if self._run_kind(sub):
                    runs.append(sub)  # a run; do not descend into it
                elif depth < maxdepth:
                    walk(sub, depth + 1)

        walk(base, 1)
        return sorted(runs)

    def _results_scan_runs(self):
        """Populate the run dropdown with runs found under the results folder.

        Searches subfolders (recursively, bounded) for saved workflow state,
        so runs nested inside an ``AnalysisResults-*`` wrapper are found too.
        """
        if not hasattr(self, "run_combo"):
            return
        previous = self.run_combo.currentData()
        self.run_combo.clear()
        base = self.results_folder_display.text().strip()
        if not base or not os.path.isdir(base):
            self.run_combo.setEnabled(False)
            return

        runs = self._find_runs(base)
        # Prefer short basenames as labels; fall back to the relative path
        # when basenames would collide.
        basenames = [os.path.basename(r) for r in runs]
        for run in runs:
            name = os.path.basename(run)
            label = (
                name
                if basenames.count(name) == 1
                else os.path.relpath(run, base)
            )
            self.run_combo.addItem(label, userData=run)
        self.run_combo.setEnabled(self.run_combo.count() > 0)
        # restore the previous selection if it is still present
        if previous is not None:
            idx = self.run_combo.findData(previous)
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)

    def _make_report_view(self):
        """Return ``(view_widget, is_web)`` for the report display.

        Uses an embedded ``QWebEngineView`` when QtWebEngine is available
        (full rendering), otherwise falls back to a ``QTextBrowser`` (basic
        HTML); either way the report can also be opened in a browser.
        """
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            return QWebEngineView(), True
        except Exception as e:
            logger.debug(f"QtWebEngine unavailable, using QTextBrowser: {e}")
            browser = QtWidgets.QTextBrowser()
            browser.setOpenExternalLinks(True)
            return browser, False

    def _results_refresh(self):
        """(Re)generate the HTML report for the selected run and show it."""
        from picasso_workflow.html_reporter import regenerate_html_report

        folder = self.run_combo.currentData()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(
                self,
                "No run selected",
                "Select a run from the dropdown (set the results folder on "
                "the Workflow Config tab, then Rescan).",
            )
            return
        try:
            path = regenerate_html_report(folder)
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(
                self,
                "No run state",
                "No WorkflowRunner.yaml or AggregationWorkflowRunner.yaml "
                "found in the selected folder.",
            )
            return
        except Exception as e:
            logger.error(f"Could not generate HTML report: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Report error", f"Could not generate report:\n{e}"
            )
            return

        self._report_path = path
        self._results_load_status(folder)
        self._load_report_into_view(path)

    def _load_report_into_view(self, path):
        """Display the generated report at ``path`` in the embedded view."""
        url = QtCore.QUrl.fromLocalFile(path)
        try:
            if self._report_view_is_web:
                self.report_view.load(url)
            else:
                self.report_view.setSource(url)
        except Exception as e:
            logger.error(f"Could not display report {path}: {e}")

    def _results_open_in_browser(self):
        """Open the last generated report in the system web browser."""
        if self._report_path and os.path.isfile(self._report_path):
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self._report_path)
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "No report", "Generate a report first."
            )

    def _results_load_status(self, folder):
        """Fill the status tree with per-module (or per-child) status."""
        from picasso_workflow.html_reporter import load_runner_state

        self.report_status.clear()
        agg_yaml = os.path.join(folder, "AggregationWorkflowRunner.yaml")
        sgl_yaml = os.path.join(folder, "WorkflowRunner.yaml")
        try:
            if os.path.isfile(agg_yaml):
                for name in sorted(os.listdir(folder)):
                    child = os.path.join(folder, name)
                    if not os.path.isfile(
                        os.path.join(child, "WorkflowRunner.yaml")
                    ):
                        continue
                    has_report = os.path.isfile(
                        os.path.join(child, "report.html")
                    )
                    QtWidgets.QTreeWidgetItem(
                        self.report_status,
                        [name, "report" if has_report else "-", ""],
                    )
            elif os.path.isfile(sgl_yaml):
                results = load_runner_state(sgl_yaml).get("results") or {}
                for key in sorted(results):
                    res = (
                        results[key] if isinstance(results[key], dict) else {}
                    )
                    ok = res.get("success")
                    status = "OK" if ok else ("FAILED" if ok is False else "?")
                    dur = res.get("duration")
                    dur_txt = (
                        f"{dur:.1f}" if isinstance(dur, (int, float)) else ""
                    )
                    QtWidgets.QTreeWidgetItem(
                        self.report_status, [key, status, dur_txt]
                    )
        except Exception as e:
            logger.debug(f"Could not load run status from {folder}: {e}")

    def select_results_folder(self):
        """Open a folder selection dialog and display the selected folder."""
        current_folder = self.results_folder_display.text()
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Results Folder",
            (
                current_folder
                if current_folder and current_folder != "No folder selected"
                else ""
            ),
        )
        if folder:
            self.results_folder_display.setText(os.path.normpath(folder))
            # Enable widgets when a folder is selected
            self._set_widgets_enabled(True)
            logger.info(f"workflow-config: loading results folder {folder}")

            # Search for YAML files and load file list
            self._load_yaml_file_list(folder)

            # Search for workflow definition and load it
            self._load_workflow_definition(folder)
            self._log_workflow_config_event(
                "results_folder.loaded", source=folder
            )
        else:
            # If dialog was cancelled and no folder is selected, disable widgets
            if not self.results_folder_display.text():
                self._set_widgets_enabled(False)

    def set_results_folder_display(self, folder):
        # Enable widgets when a folder is selected
        self._set_widgets_enabled(True)
        # keep the Results tab's run dropdown in sync with the folder
        self._results_scan_runs()

    def on_results_folder_dropped(self, folder):
        """Handle a folder dragged & dropped onto the results folder field.

        Mirrors select_results_folder so a dropped folder loads its YAML
        file list and workflow definition just like a folder picked via
        the dialog.
        """
        if not folder or not os.path.isdir(folder):
            return
        self._set_widgets_enabled(True)
        logger.info(f"workflow-config: results folder dropped: {folder}")
        self._load_yaml_file_list(folder)
        self._load_workflow_definition(folder)
        self._log_workflow_config_event(
            "results_folder.dropped", source=folder
        )

    def on_template_changed(self, template_name):
        """Load a template"""
        try:
            template_path = CONFIG["Templates"][template_name]
            if template_path[:2] == 'r"' or template_path[:2] == "r'":
                template_path = template_path[2:-1]

            # If path points to a file, use its directory
            if template_path.endswith(".py"):
                template_folder = os.path.dirname(template_path)
            else:
                template_folder = template_path
        except KeyError:
            return
        # # Enable widgets when a folder is selected
        # self._set_widgets_enabled(True)
        logger.info(
            f"workflow-config: loading template '{template_name}' "
            f"from {template_folder}"
        )

        # Search for YAML files and load file list
        self._load_yaml_file_list(template_folder)

        # Search for workflow definition and load it
        self._load_workflow_definition(template_folder)
        self._log_workflow_config_event(
            "template.loaded",
            template=template_name,
            source=template_folder,
        )

    def _load_yaml_file_list(self, folder):
        """Search for YAML files in folder and load file list if found.

        Parameters
        ----------
        folder: Path to the folder to search
        """
        # Search for specific YAML files
        yaml_files = ["src_loc.yaml", "raw_locs_list.yaml"]

        for yaml_file in yaml_files:
            yaml_path = os.path.join(folder, yaml_file)
            if os.path.exists(yaml_path):
                try:
                    # Load YAML file
                    with open(yaml_path, "r") as f:
                        file_dict = yaml.safe_load(f)

                    # Validate that it's a dictionary
                    if isinstance(file_dict, dict):
                        # it may still be a dict with keys #tags and
                        # filepaths and list values or tags to fileptahs dict
                        # as expected here
                        if (
                            "#tags" in file_dict.keys()
                            and "filepath" in file_dict.keys()
                            and isinstance(file_dict["filepath"], list)
                        ):
                            new_dict = {}
                            for k, v in zip(
                                file_dict["#tags"], file_dict["filepath"]
                            ):
                                new_dict[k] = v
                            file_dict = new_dict
                        # Clear existing file list
                        self.files_table.setRowCount(0)

                        # Populate table with YAML content
                        # Key -> Name, Value -> File Path
                        dict_to_table(file_dict, self.files_table)

                        logger.debug(f"Loaded file list from {yaml_file}")
                        self._log_workflow_config_event(
                            "files_table.load_yaml",
                            source=yaml_path,
                        )
                        return  # Stop after loading first matching file
                    else:
                        logger.warning(
                            f"{yaml_file} does not contain a dictionary"
                        )

                except Exception as e:
                    logger.error(f"Error loading {yaml_file}: {e}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "YAML Load Error",
                        f"Failed to load {yaml_file}:\n{str(e)}",
                    )

        # No YAML files found - this is normal, no action needed
        logger.debug("No src_loc.yaml or raw_locs_list.yaml found in folder")

    def _safe_eval_node(self, node, variables=None):
        """Evaluate AST node, handling literals, string concat, and f-strings.

        This is a custom evaluator that handles more complex expressions than
        ast.literal_eval, including:
        - Literals (strings, numbers, lists, tuples, dicts)
        - String concatenation (ast.BinOp with ast.Add)
        - F-strings (ast.JoinedStr) with variable substitution
        - Variable references (ast.Name) from a known variables dict

        Parameters
        ----------
        node: AST node to evaluate
        variables: Dict of known variable values
            (e.g., {'idx_last_sgl_module': 6})

        Returns
        -------
        Evaluated Python value

        Raises
        ------
        ValueError: If node cannot be safely evaluated
        """
        import ast

        if variables is None:
            variables = {}

        # Handle different node types
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.List):
            return [self._safe_eval_node(el, variables) for el in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(
                self._safe_eval_node(el, variables) for el in node.elts
            )

        elif isinstance(node, ast.Dict):
            keys = [
                self._safe_eval_node(k, variables) if k else None
                for k in node.keys
            ]
            vals = [self._safe_eval_node(v, variables) for v in node.values]
            return dict(zip(keys, vals))

        elif isinstance(node, ast.Set):
            return {self._safe_eval_node(el, variables) for el in node.elts}

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            # String/list concatenation
            left = self._safe_eval_node(node.left, variables)
            right = self._safe_eval_node(node.right, variables)
            return left + right

        elif isinstance(node, ast.JoinedStr):
            # F-string: f"text {var:fmt} more"
            parts = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    parts.append(str(part.value))
                elif isinstance(part, ast.FormattedValue):
                    val = self._safe_eval_node(part.value, variables)
                    if part.format_spec:
                        # Format spec is also a JoinedStr
                        fmt = self._safe_eval_node(part.format_spec, variables)
                        parts.append(format(val, fmt))
                    else:
                        parts.append(str(val))
            return "".join(parts)

        elif isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        # Python 3.7 compatibility (ast.Str, ast.Num deprecated but may exist)
        elif hasattr(ast, "Str") and isinstance(node, ast.Str):
            return node.s
        elif hasattr(ast, "Num") and isinstance(node, ast.Num):
            return node.n

        raise ValueError(f"Cannot evaluate node: {type(node).__name__}")

    def _load_workflow_definition_alt(self, folder):
        """Load workflow definition from functions (alternative format).

        This handles cases where workflow_modules_sgl and workflow_modules_agg
        are defined inside functions rather than at module level. Uses a custom
        AST evaluator to handle complex expressions like string concatenation,
        f-strings, and variable references.

        Parameters
        ----------
        folder: Path to the folder to search
        """
        import ast

        # print(f"DEBUG: _load_workflow_definition_alt called with folder={folder}")
        workflow_file = os.path.join(folder, "start_workflow.py")

        logger.debug("loading definition (alt)")

        if not os.path.exists(workflow_file):
            logger.debug("No start_workflow.py found in folder")
            print("No start_workflow.py found in folder")
            return

        try:
            # Read and parse the file
            with open(workflow_file, "r") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Find workflow module definitions
            workflow_modules_sgl = None
            workflow_modules_agg = None
            workflow_modules_multi = None

            # Variables dict for resolving variable references in expressions
            variables = {}

            # Helper to safely evaluate AST nodes using the custom evaluator
            def safe_eval_node(node, var_name, variables_dict):
                """Try to evaluate an AST node using custom evaluator.

                Returns None if evaluation fails.
                Note: Do NOT fall back to ast.literal_eval - it's more
                restrictive and cannot handle BinOp nodes (string concat).
                """
                try:
                    return self._safe_eval_node(node, variables_dict)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not evaluate {var_name}: {e}")
                    # print(f"DEBUG: Could not evaluate {var_name}: {e}")
                    return None

            def search_assignments(statements, variables_dict):
                """Search statements for workflow variable assignments.

                Returns tuple (sgl, agg, multi) of found values.
                Also updates variables_dict with any idx_last_sgl_module found.
                """
                sgl = None
                agg = None
                multi = None

                for stmt in statements:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                if target.id == "workflow_modules_sgl":
                                    sgl = safe_eval_node(
                                        stmt.value,
                                        "workflow_modules_sgl",
                                        variables_dict,
                                    )
                                    # Update idx_last_sgl_module for later use
                                    if sgl and isinstance(sgl, list):
                                        variables_dict[
                                            "idx_last_sgl_module"
                                        ] = (len(sgl) - 1)
                                        # print(
                                        #     f"DEBUG: Set idx_last_sgl_module="
                                        #     f"{variables_dict['idx_last_sgl_module']}"
                                        # )
                                elif target.id == "workflow_modules_agg":
                                    agg = safe_eval_node(
                                        stmt.value,
                                        "workflow_modules_agg",
                                        variables_dict,
                                    )
                                elif target.id == "workflow_modules_multi":
                                    multi = safe_eval_node(
                                        stmt.value,
                                        "workflow_modules_multi",
                                        variables_dict,
                                    )
                                elif target.id == "idx_last_sgl_module":
                                    # Try to evaluate the variable assignment
                                    try:
                                        val = self._safe_eval_node(
                                            stmt.value, variables_dict
                                        )
                                        variables_dict[
                                            "idx_last_sgl_module"
                                        ] = val
                                        # print(
                                        #     f"DEBUG: Found idx_last_sgl_module "
                                        #     f"assignment: {val}"
                                        # )
                                    except (ValueError, TypeError):
                                        pass
                return sgl, agg, multi

            # First search module-level assignments
            sgl, agg, multi = search_assignments(tree.body, variables)
            if sgl is not None:
                workflow_modules_sgl = sgl
            if agg is not None:
                workflow_modules_agg = agg
            if multi is not None:
                workflow_modules_multi = multi

            # print("DEBUG: After module-level search:")
            # print(f"  workflow_modules_sgl: {workflow_modules_sgl}")
            # print(f"  workflow_modules_agg: {workflow_modules_agg}")
            # print(f"  workflow_modules_multi: {workflow_modules_multi}")
            # print(f"  variables: {variables}")

            # Search inside function definitions if nothing meaningful found
            # Note: search functions if lists are None OR empty
            needs_function_search = (
                (workflow_modules_sgl is None or workflow_modules_sgl == [])
                and (
                    workflow_modules_agg is None or workflow_modules_agg == []
                )
                and (
                    workflow_modules_multi is None
                    or workflow_modules_multi == {}
                )
            )

            if needs_function_search:
                # print("DEBUG: Searching inside function definitions...")
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Reset variables for this function scope
                        func_variables = dict(variables)
                        sgl, agg, multi = search_assignments(
                            node.body, func_variables
                        )

                        # Update main variables dict with any found variables
                        variables.update(func_variables)

                        if sgl is not None and workflow_modules_sgl is None:
                            workflow_modules_sgl = sgl
                        if agg is not None and workflow_modules_agg is None:
                            workflow_modules_agg = agg
                        if (
                            multi is not None
                            and workflow_modules_multi is None
                        ):
                            workflow_modules_multi = multi

            # If workflow_modules_sgl was found but workflow_modules_agg was not,
            # try again with the updated variables dict (now has idx_last_sgl_module)
            if (
                workflow_modules_sgl is not None
                and isinstance(workflow_modules_sgl, list)
                and len(workflow_modules_sgl) > 0
                and (
                    workflow_modules_agg is None or workflow_modules_agg == []
                )
            ):
                # print(
                #     "DEBUG: Retrying workflow_modules_agg with updated "
                #     f"variables: {variables}"
                # )

                # Search module level again with updated variables
                for stmt in tree.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Name)
                                and target.id == "workflow_modules_agg"
                            ):
                                result = safe_eval_node(
                                    stmt.value,
                                    "workflow_modules_agg",
                                    variables,
                                )
                                if result is not None:
                                    workflow_modules_agg = result

                # Search inside functions again with updated variables
                if workflow_modules_agg is None or workflow_modules_agg == []:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            for stmt in node.body:
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if (
                                            isinstance(target, ast.Name)
                                            and target.id
                                            == "workflow_modules_agg"
                                        ):
                                            result = safe_eval_node(
                                                stmt.value,
                                                "workflow_modules_agg",
                                                variables,
                                            )
                                            if result is not None:
                                                workflow_modules_agg = result

                # if workflow_modules_agg is not None:
                #     print(
                #         f"DEBUG: After retry, found workflow_modules_agg with "
                #         f"{len(workflow_modules_agg)} modules"
                #     )
                # else:
                #     print("DEBUG: After retry, workflow_modules_agg still None")

            # Check for workflow_modules_multi dict format
            if workflow_modules_multi is not None and isinstance(
                workflow_modules_multi, dict
            ):
                logger.debug(
                    "Found workflow_modules_multi dict in function (alt)"
                )
                if workflow_modules_sgl is None:
                    workflow_modules_sgl = workflow_modules_multi.get(
                        "single_dataset_modules"
                    )
                if workflow_modules_agg is None:
                    workflow_modules_agg = workflow_modules_multi.get(
                        "aggregation_modules"
                    )

            # Debug logging
            logger.debug(
                f"Alt loader found: "
                f"workflow_modules_sgl={'list' if isinstance(workflow_modules_sgl, list) else type(workflow_modules_sgl).__name__}, "
                f"workflow_modules_agg={'list' if isinstance(workflow_modules_agg, list) else type(workflow_modules_agg).__name__}, "
                f"workflow_modules_multi={'dict' if isinstance(workflow_modules_multi, dict) else type(workflow_modules_multi).__name__}"
            )
            # print(
            #     f"Alt loader found: "
            #     f"workflow_modules_sgl={'list' if isinstance(workflow_modules_sgl, list) else type(workflow_modules_sgl).__name__}, "
            #     f"workflow_modules_agg={'list' if isinstance(workflow_modules_agg, list) else type(workflow_modules_agg).__name__}, "
            #     f"workflow_modules_multi={'dict' if isinstance(workflow_modules_multi, dict) else type(workflow_modules_multi).__name__}"
            # )

            # Load single dataset workflow if found
            if workflow_modules_sgl is not None and isinstance(
                workflow_modules_sgl, list
            ):
                self._populate_workflow_from_definition(
                    workflow_modules_sgl,
                    self.single_workflow_modules,
                    self.single_workflow_list,
                    "Single Dataset",
                )
                logger.info(
                    f"Loaded {len(workflow_modules_sgl)} modules to "
                    "Single Dataset workflow (alt)"
                )
                # print(
                #     f"Loaded {len(workflow_modules_sgl)} modules to "
                #     "Single Dataset workflow (alt)"
                # )
            else:
                logger.info(
                    "No single dataset modules found in workflow template "
                    "(alt)"
                )
                # print(
                #     "No single dataset modules found in workflow template "
                #     "(alt)"
                # )

            # Load aggregation workflow if found
            if (
                workflow_modules_agg is not None
                and isinstance(workflow_modules_agg, list)
                and len(workflow_modules_agg) > 0
            ):
                self._populate_workflow_from_definition(
                    workflow_modules_agg,
                    self.aggregation_workflow_modules,
                    self.aggregation_workflow_list,
                    "Aggregation",
                )
                logger.info(
                    f"Loaded {len(workflow_modules_agg)} modules to "
                    "Aggregation workflow (alt)"
                )
                # print(
                #     f"Loaded {len(workflow_modules_agg)} modules to "
                #     "Aggregation workflow (alt)"
                # )
                # Set workflow type to "Aggregation Workflow"
                self.workflow_type.setCurrentIndex(1)
            else:
                logger.info(
                    "No aggregation modules found in workflow template (alt)"
                )
                # print(
                #     "No aggregation modules found in workflow template (alt)"
                # )
                # Set workflow type to "Single Workflow"
                self.workflow_type.setCurrentIndex(0)

        except SyntaxError as e:
            logger.error(f"Syntax error in start_workflow.py: {e}")
            print(f"Syntax error in start_workflow.py: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                "Workflow Load Error",
                f"Syntax error in start_workflow.py:\n{str(e)}",
            )
        except Exception as e:
            logger.error(f"Error loading start_workflow.py (alt): {e}")
            print(f"Error loading start_workflow.py (alt): {e}")
            QtWidgets.QMessageBox.warning(
                self,
                "Workflow Load Error",
                f"Failed to load workflow from start_workflow.py:\n{str(e)}",
            )

    def _load_workflow_definition(self, folder):
        """Search for workflow definition file and load workflow modules.

        Parameters
        ----------
        folder: Path to the folder to search
        """
        workflow_file = os.path.join(folder, "start_workflow.py")

        logger.debug("loaing definition")

        if not os.path.exists(workflow_file):
            logger.debug("No start_workflow.py found in folder")
            return

        try:
            # Read raw source to detect run-time options that are not
            # exposed as module-level variables (e.g. find_dnapaint_raw).
            with open(workflow_file, "r") as f:
                source_text = f.read()

            # Dynamically load the Python file
            spec = importlib.util.spec_from_file_location(
                "start_workflow", workflow_file
            )
            if spec is None or spec.loader is None:
                logger.warning("Could not load start_workflow.py")
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract workflow definitions
            workflow_modules_sgl = getattr(
                module, "workflow_modules_sgl", None
            )
            workflow_modules_agg = getattr(
                module, "workflow_modules_agg", None
            )

            # Also check for workflow_modules_multi dict format
            # (used by standard_aggregation_workflows.py)
            workflow_modules_multi = getattr(
                module, "workflow_modules_multi", None
            )
            if workflow_modules_multi is not None and isinstance(
                workflow_modules_multi, dict
            ):
                logger.debug(
                    "Found workflow_modules_multi dict at module level"
                )
                if workflow_modules_sgl is None:
                    workflow_modules_sgl = workflow_modules_multi.get(
                        "single_dataset_modules"
                    )
                if workflow_modules_agg is None:
                    workflow_modules_agg = workflow_modules_multi.get(
                        "aggregation_modules"
                    )

            # Debug logging to track what was found
            logger.debug(
                f"Module-level variables found: "
                f"workflow_modules_sgl={'list' if isinstance(workflow_modules_sgl, list) else type(workflow_modules_sgl).__name__}, "
                f"workflow_modules_agg={'list' if isinstance(workflow_modules_agg, list) else type(workflow_modules_agg).__name__}, "
                f"workflow_modules_multi={'dict' if isinstance(workflow_modules_multi, dict) else type(workflow_modules_multi).__name__}"
            )

            # If no modules found at module level, try alternative loader
            if workflow_modules_sgl is None and workflow_modules_agg is None:
                logger.info(
                    "No workflow modules found at module level, trying alternative loader"
                )
                self._load_workflow_definition_alt(folder)
                self._apply_files_mode_from_source(source_text)
                return

            # Load single dataset workflow if present
            if workflow_modules_sgl is not None and isinstance(
                workflow_modules_sgl, list
            ):
                self._populate_workflow_from_definition(
                    workflow_modules_sgl,
                    self.single_workflow_modules,
                    self.single_workflow_list,
                    "Single Dataset",
                )
                logger.info(
                    f"Loaded {len(workflow_modules_sgl)} modules to "
                    "Single Dataset workflow"
                )
            else:
                logger.info(
                    "No single dataset modules found in workflow template"
                )

            # Load aggregation workflow if present
            if (
                workflow_modules_agg is not None
                and isinstance(workflow_modules_agg, list)
                and len(workflow_modules_agg) > 0
            ):
                self._populate_workflow_from_definition(
                    workflow_modules_agg,
                    self.aggregation_workflow_modules,
                    self.aggregation_workflow_list,
                    "Aggregation",
                )
                logger.info(
                    f"Loaded {len(workflow_modules_agg)} modules to "
                    "Aggregation workflow"
                )
                # Set workflow type to "Aggregation Workflow"
                self.workflow_type.setCurrentIndex(1)
            else:
                logger.info(
                    "No aggregation modules found in workflow template"
                )
                # Set workflow type to "Single Dataset Workflow"
                self.workflow_type.setCurrentIndex(0)

            # Restore the input-files mode (explicit / auto-detect / none)
            self._apply_files_mode_from_source(source_text)

        except Exception as e:
            logger.error(f"Error loading start_workflow.py: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                "Workflow Load Error",
                f"Failed to load workflow from start_workflow.py:\n{str(e)}",
            )

    def _populate_workflow_from_definition(
        self, workflow_def, workflow_list, list_widget, workflow_name
    ):
        """Populate workflow from loaded definition.

        Parameters
        ----------
        workflow_def: List of (module_name, params_dict) tuples
        workflow_list: Target workflow list (single_workflow_modules or aggregation_workflow_modules)
        list_widget: Target QListWidget for display
        workflow_name: Name of the workflow (for logging)
        """
        # Clear existing workflow
        workflow_list.clear()
        list_widget.clear()

        for module_name, params in workflow_def:
            # Store parameters in actionable format (no conversion)
            # Tuples remain tuples, dicts remain dicts, etc.
            workflow_list.append((module_name, params))
            index = len(workflow_list) - 1
            list_widget.addItem(f"{index:02d}: {module_name}")

        # Per-list audit event. Both _load_workflow_definition and the
        # alt loader funnel through here, so this also covers the early
        # return path where the alt loader is invoked.
        self._log_workflow_config_event(
            "modules.populate_from_definition",
            workflow=workflow_name,
            n_modules=len(workflow_def),
            modules=[
                [name, Window._yaml_safe(params)]
                for name, params in workflow_def
            ],
        )

    def _convert_param_to_gui_format(self, param_value):
        """Convert parameter value from workflow definition to GUI format.

        Parameters
        ----------
        param_value: Parameter value from workflow definition

        Returns
        -------
        tuple: (value_data, command_string)
               value_data can be a string, dict (for nested params), or other type
        """
        # # Handle tuples - check for special commands first
        # if isinstance(param_value, tuple) and len(param_value) >= 2:
        #     first_elem = param_value[0]

        #     # Check for special command markers
        #     if isinstance(first_elem, str):
        #         if first_elem.startswith("$$"):
        #             # Special map command: ("$$map", "key")
        #             command = first_elem[2:]  # Remove $$ prefix
        #             value = str(param_value[1])
        #             return value, command

        #         elif first_elem.startswith("$"):
        #             # Prior result reference: ("$get_previous_module_result", "module, result")
        #             # Convert to GUI format
        #             value = str(param_value[1]) if len(param_value) > 1 else ""
        #             return value, "prior result"

        #     # Not a special command - convert whole tuple to string
        #     return str(param_value), ""

        if isinstance(param_value, tuple):
            if len(param_value) == 1:
                return str(param_value[0]), ""
            elif (len(param_value) == 2) and param_value[1] == "":
                return str(param_value[0]), ""
            else:
                return str(param_value), ""

        # Handle nested dictionaries (for dict parameters)
        if isinstance(param_value, dict):
            # Recursively convert nested parameters
            nested_converted = {}
            for nested_param_name, nested_param_value in param_value.items():
                nested_value, nested_command = (
                    self._convert_param_to_gui_format(nested_param_value)
                )
                nested_converted[nested_param_name] = (
                    nested_value  # (nested_value, nested_command)
                )
            # Return dict as value (not as string)
            return nested_converted, ""

        # Handle lists - convert to string
        if isinstance(param_value, list):
            return str(param_value), ""

        # Default: plain value, no command
        return str(param_value), ""

    def add_module(self):
        """Add the currently selected module to the workflow."""
        module_name = self.module_combobox.currentText()

        # Capture current parameter values
        param_values = {}
        for param_name, widget_info in self.parameter_widgets.items():
            value = self._get_widget_value(
                widget_info.widget, widget_info.original_type, widget_info
            )
            # Skip None values (from unchecked optional dicts)
            if value is not None:
                param_values[param_name] = value

        # Add to the appropriate workflow list based on selected tab
        # Store as tuple: (module_name, {param: value})
        current_tab_index = self.workflow_tabs.currentIndex()
        if current_tab_index == 0:  # Single Dataset Workflow
            self.single_workflow_modules.append((module_name, param_values))
            index = len(self.single_workflow_modules) - 1
            self.single_workflow_list.addItem(f"{index:02d}: {module_name}")
            self._log_workflow_config_event(
                "modules.add",
                tab="single",
                index=index,
                module=module_name,
                params=param_values,
            )
        elif current_tab_index == 1:  # Aggregation Workflow
            self.aggregation_workflow_modules.append(
                (module_name, param_values)
            )
            index = len(self.aggregation_workflow_modules) - 1
            self.aggregation_workflow_list.addItem(
                f"{index:02d}: {module_name}"
            )
            self._log_workflow_config_event(
                "modules.add",
                tab="aggregation",
                index=index,
                module=module_name,
                params=param_values,
            )

    def _renumber_workflow_items(self, list_widget, modules):
        """Update QListWidget items with correct numbering after reordering."""
        for i in range(len(modules)):
            module_name = modules[i][
                0
            ]  # Extract name from (name, params) tuple
            list_widget.item(i).setText(f"{i:02d}: {module_name}")

    def _on_workflow_selection_changed(self, current_row):
        """Handle selection change in workflow list - display module in Current Module section."""
        if current_row < 0:
            # Clear editing state when nothing is selected
            self.editing_workflow_index = -1
            self.editing_workflow_tab = -1
            return

        # Save current parameters before loading new selection
        self._update_editing_workflow_item()

        # Get the current tab to determine which workflow list to use
        current_tab_index = self.workflow_tabs.currentIndex()

        # Track which workflow item is being edited
        self.editing_workflow_index = current_row
        self.editing_workflow_tab = current_tab_index

        if current_tab_index == 0:  # Single Dataset Workflow
            if current_row < len(self.single_workflow_modules):
                module_name, param_values = self.single_workflow_modules[
                    current_row
                ]
                # Update module combobox to show this module
                index = self.module_combobox.findText(module_name)
                if index >= 0:
                    # Block signals to prevent on_module_changed from firing
                    self.module_combobox.blockSignals(True)
                    try:
                        self.module_combobox.setCurrentIndex(index)
                        # Manually trigger widget update since signal is blocked
                        self.on_module_changed(module_name)
                        # Populate parameters with stored values
                        self._populate_stored_parameters(param_values)
                        self._validate_parameters()  # Clear red borders from filled fields
                    finally:
                        # Always unblock signals, even if exception occurs
                        self.module_combobox.blockSignals(False)
        elif current_tab_index == 1:  # Aggregation Workflow
            if current_row < len(self.aggregation_workflow_modules):
                module_name, param_values = self.aggregation_workflow_modules[
                    current_row
                ]
                # Update module combobox to show this module
                index = self.module_combobox.findText(module_name)
                if index >= 0:
                    # Block signals to prevent on_module_changed from firing
                    self.module_combobox.blockSignals(True)
                    try:
                        self.module_combobox.setCurrentIndex(index)
                        # Manually trigger widget update since signal is blocked
                        self.on_module_changed(module_name)
                        # Populate parameters with stored values
                        self._populate_stored_parameters(param_values)
                        self._validate_parameters()  # Clear red borders from filled fields
                    finally:
                        # Always unblock signals, even if exception occurs
                        self.module_combobox.blockSignals(False)

    def _populate_stored_parameters(self, param_values):
        """Populate parameter widgets with stored values from a workflow module.

        Parameters
        ----------
        param_values: Dict of {param_name: value}
        """
        for param_name, widget_info in self.parameter_widgets.items():
            if param_name in param_values:
                value_data = param_values[param_name]

                # Check if value is a command tuple (starts with $ or $$)
                if isinstance(value_data, tuple) and len(value_data) >= 1:
                    first_elem = str(value_data[0])
                    if first_elem.startswith("$"):
                        # This is a command - convert widget to textbox
                        self._convert_widget_to_textbox(
                            param_name, str(value_data)
                        )
                        continue

                # Set value in widget
                self._set_widget_value(
                    widget_info.widget,
                    value_data,
                    widget_info.original_type,
                    widget_info,
                )

    def _on_workflow_tab_changed(self, tab_index):
        """Handle workflow tab change - display selected module if any."""
        # Save current parameters before switching tabs
        self._update_editing_workflow_item()

        if tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row >= 0 and current_row < len(
                self.single_workflow_modules
            ):
                # Update editing state to new tab/row
                self.editing_workflow_tab = tab_index
                self.editing_workflow_index = current_row

                module_name, param_values = self.single_workflow_modules[
                    current_row
                ]
                index = self.module_combobox.findText(module_name)
                if index >= 0:
                    # Block signals to prevent on_module_changed from firing
                    self.module_combobox.blockSignals(True)
                    try:
                        self.module_combobox.setCurrentIndex(index)
                        # Manually trigger widget update since signal is blocked
                        self.on_module_changed(module_name)
                        # Populate parameters with stored values
                        self._populate_stored_parameters(param_values)
                        self._validate_parameters()  # Clear red borders from filled fields
                    finally:
                        # Always unblock signals, even if exception occurs
                        self.module_combobox.blockSignals(False)
        elif tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            if current_row >= 0 and current_row < len(
                self.aggregation_workflow_modules
            ):
                # Update editing state to new tab/row
                self.editing_workflow_tab = tab_index
                self.editing_workflow_index = current_row

                module_name, param_values = self.aggregation_workflow_modules[
                    current_row
                ]
                index = self.module_combobox.findText(module_name)
                if index >= 0:
                    # Block signals to prevent on_module_changed from firing
                    self.module_combobox.blockSignals(True)
                    try:
                        self.module_combobox.setCurrentIndex(index)
                        # Manually trigger widget update since signal is blocked
                        self.on_module_changed(module_name)
                        # Populate parameters with stored values
                        self._populate_stored_parameters(param_values)
                        self._validate_parameters()  # Clear red borders from filled fields
                    finally:
                        # Always unblock signals, even if exception occurs
                        self.module_combobox.blockSignals(False)

    def _on_workflow_type_changed(self, type_index):
        """Handle workflow type change - enable/disable workflow tabs accordingly.

        Parameters
        ----------
        type_index: Index of selected workflow type
                   0 = Single Workflow
                   1 = Aggregation Workflow
                   2 = Investigation Workflow
        """
        # Convert data format if needed
        if type_index == 0:  # Switching TO Single
            if self.tree_data["datasets"]:
                self._tree_to_simple()
        else:  # Switching TO Aggregation/Investigation
            if self.files_table.rowCount() > 0:
                self._simple_to_tree()

        # Switch visible widget
        self.files_stack.setCurrentIndex(type_index)

        # The input-files mode selector is only meaningful for Single
        # Workflow
        self.files_mode_label.setVisible(type_index == 0)
        self.files_mode_combo.setVisible(type_index == 0)

        # Update buttons
        self._update_file_buttons(type_index)

        # Re-apply the files-mode state for the (now current) workflow type
        self._apply_files_mode_state()

        # Tab indices:
        # 0 = Single Dataset Workflow
        # 1 = Aggregation Workflow
        # 2 = Investigation

        if type_index == 0:  # Single Workflow
            # Enable only Single Dataset Workflow tab
            self.workflow_tabs.setTabEnabled(
                0, True
            )  # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, False)  # Aggregation: disabled
            self.workflow_tabs.setTabEnabled(
                2, False
            )  # Investigation: disabled
            # Switch to Single Dataset tab if currently on a disabled tab
            if self.workflow_tabs.currentIndex() != 0:
                self.workflow_tabs.setCurrentIndex(0)

        elif type_index == 1:  # Aggregation Workflow
            # Enable Single Dataset and Aggregation, disable Investigation
            self.workflow_tabs.setTabEnabled(
                0, True
            )  # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, True)  # Aggregation: enabled
            self.workflow_tabs.setTabEnabled(
                2, False
            )  # Investigation: disabled
            # Switch to Aggregation tab if currently on Investigation
            if self.workflow_tabs.currentIndex() == 2:
                self.workflow_tabs.setCurrentIndex(1)

        elif type_index == 2:  # Investigation Workflow
            # Enable all tabs
            self.workflow_tabs.setTabEnabled(
                0, True
            )  # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, True)  # Aggregation: enabled
            self.workflow_tabs.setTabEnabled(2, True)  # Investigation: enabled

    # def on_cluster_use_module_state_change(self, state):
    #     if not self.cluster_use_module.isChecked():
    #         QtWidgets.QMessageBox.warning(
    #             self, "Warning",
    #             "Are you very sure?")

    def remove_selected(self):
        """Remove the selected module from the workflow."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row >= 0:
                removed = self.single_workflow_modules[current_row][0]
                self._remove_module(
                    self.single_workflow_list,
                    self.single_workflow_modules,
                    current_row,
                )
                self._log_workflow_config_event(
                    "modules.remove",
                    tab="single",
                    index=current_row,
                    module=removed,
                )
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            if current_row >= 0:
                removed = self.aggregation_workflow_modules[current_row][0]
                self._remove_module(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                    current_row,
                )
                self._log_workflow_config_event(
                    "modules.remove",
                    tab="aggregation",
                    index=current_row,
                    module=removed,
                )

    def _reorder_module(
        self, list_widget, modules, tab_index, from_row, to_row
    ):
        """Move a module from from_row to to_row, keeping its parameters intact.

        Reordering must not corrupt the parameters of the modules involved.
        Before swapping, any pending widget edits are flushed to the module
        currently being edited (at its pre-swap index). After swapping, the
        editing index is repointed to the moved module's new row and the
        selection is updated with the list-widget signal blocked, so the
        selection-changed handler does not re-save the editor widgets onto a
        now-stale index.
        """
        # Persist any pending edits to the currently edited item *before*
        # the swap, while indices still match the editor widgets.
        self._update_editing_workflow_item()

        # Swap the module tuples (name, params) in the data model.
        modules[from_row], modules[to_row] = (
            modules[to_row],
            modules[from_row],
        )

        # Refresh the displayed numbering/labels.
        self._renumber_workflow_items(list_widget, modules)

        # The editor widgets still show the moved module; point the editing
        # state at its new row and move the selection without triggering a
        # stale save/reload through currentRowChanged.
        self.editing_workflow_index = to_row
        self.editing_workflow_tab = tab_index
        list_widget.blockSignals(True)
        list_widget.setCurrentRow(to_row)
        list_widget.blockSignals(False)

    def _remove_module(self, list_widget, modules, row):
        """Remove the module at row, keeping remaining parameters intact.

        Clearing the editing state before touching the widget prevents the
        selection-changed handler (fired by takeItem/setCurrentRow) from
        flushing the editor widgets onto a now-stale index. The list-widget
        signal is blocked during the mutation, then a clean selection is
        restored so the editor reloads the correct module.
        """
        # We are deleting the item that may currently be in the editor; drop
        # the editing state so no stale save happens during the mutation.
        self.editing_workflow_index = -1
        self.editing_workflow_tab = -1

        list_widget.blockSignals(True)
        list_widget.takeItem(row)
        list_widget.blockSignals(False)
        del modules[row]
        self._renumber_workflow_items(list_widget, modules)

        # Restore a valid selection (and reload the editor) if any modules
        # remain, clamping to the last row when the tail item was removed.
        if modules:
            new_row = min(row, len(modules) - 1)
            list_widget.setCurrentRow(new_row)

    def move_up(self):
        """Move the selected module up in the workflow order."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row > 0:  # Can't move first item up
                self._reorder_module(
                    self.single_workflow_list,
                    self.single_workflow_modules,
                    current_tab_index,
                    current_row,
                    current_row - 1,
                )
                self._log_workflow_config_event(
                    "modules.move_up",
                    tab="single",
                    from_index=current_row,
                    to_index=current_row - 1,
                )
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            if current_row > 0:  # Can't move first item up
                self._reorder_module(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                    current_tab_index,
                    current_row,
                    current_row - 1,
                )
                self._log_workflow_config_event(
                    "modules.move_up",
                    tab="aggregation",
                    from_index=current_row,
                    to_index=current_row - 1,
                )

    def move_down(self):
        """Move the selected module down in the workflow order."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            max_row = len(self.single_workflow_modules) - 1
            if 0 <= current_row < max_row:  # Can't move last item down
                self._reorder_module(
                    self.single_workflow_list,
                    self.single_workflow_modules,
                    current_tab_index,
                    current_row,
                    current_row + 1,
                )
                self._log_workflow_config_event(
                    "modules.move_down",
                    tab="single",
                    from_index=current_row,
                    to_index=current_row + 1,
                )
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            max_row = len(self.aggregation_workflow_modules) - 1
            if 0 <= current_row < max_row:  # Can't move last item down
                self._reorder_module(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                    current_tab_index,
                    current_row,
                    current_row + 1,
                )
                self._log_workflow_config_event(
                    "modules.move_down",
                    tab="aggregation",
                    from_index=current_row,
                    to_index=current_row + 1,
                )

    def create_python_script(
        self, host_cluster, login_node, filename="start_workflow.py"
    ):
        """Generate a Python workflow script from current GUI settings.

        Parameters
        ----------
        filename: Name of the output script file
        """
        from datetime import datetime

        # Get workflow type
        workflow_type_index = self.workflow_type.currentIndex()
        workflow_type_names = [
            "Single Workflow",
            "Aggregation Workflow",
            "Investigation Workflow",
        ]
        workflow_type_name = (
            workflow_type_names[workflow_type_index]
            if workflow_type_index < len(workflow_type_names)
            else "Unknown"
        )

        # Single Workflow input-files mode (combo index): 0 = explicit
        # file list, 1 = auto-detect raw files in the results folder at
        # run time, 2 = no input files (run the workflow exactly once).
        # The mode is only meaningful for Single Workflow.
        files_mode = (
            self.files_mode_combo.currentIndex()
            if workflow_type_index == 0
            else 0
        )
        use_autodetect = files_mode == 1
        no_input_files = files_mode == 2

        # Validate tree data for Aggregation/Investigation workflows
        if workflow_type_index > 0:  # Tree-based workflows
            errors = self._validate_tree_data()
            if errors:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Validation Errors",
                    "Cannot generate script:\n\n" + "\n".join(errors[:10]),
                )
                return None

        # Build datasets dict based on workflow type
        datasets = {}

        if workflow_type_index == 0 and (use_autodetect or no_input_files):
            # Auto-detect: datasets are discovered at run time by
            # find_dnapaint_raw. No input files: the workflow runs once
            # with no datasets. Either way, no dict is baked in.
            pass

        elif workflow_type_index == 0:  # Single Workflow
            # Use simple table format
            for row in range(self.files_table.rowCount()):
                name_item = self.files_table.item(row, 0)
                path_item = self.files_table.item(row, 1)
                if name_item and path_item:
                    name = name_item.text()
                    path = path_item.text()
                    if (path[0] == "'" and path[-1] == "'") or (
                        path[0] == '"' and path[-1] == '"'
                    ):
                        path = path[1:-1]
                    path = self.pathparser.convert_path(path, host_cluster)

                    # Create lists for values
                    if name not in datasets:
                        datasets[name] = []
                    datasets[name].append(path)

        elif workflow_type_index == 1:  # Aggregation Workflow
            # Build datasets from tree structure
            tags = []
            filepaths = []

            for dataset in self.tree_data["datasets"]:
                for channel in self.tree_data["channels"]:
                    # Create tag in {dataset}_{channel} format
                    tag = f"{dataset}_{channel}"
                    tags.append(tag)

                    # Get file path
                    file_path = self.tree_data["file_paths"][dataset][channel]
                    if (file_path[0] == "'" and file_path[-1] == "'") or (
                        file_path[0] == '"' and file_path[-1] == '"'
                    ):
                        file_path = file_path[1:-1]
                    file_path = self.pathparser.convert_path(
                        file_path, host_cluster
                    )
                    filepaths.append(file_path)

            # Assign datasets dict for Aggregation workflow
            datasets = {"#tags": tags, "filepath": filepaths}

        else:  # Investigation Workflow
            # Build datasets from tree structure
            tags = []
            filepaths = []

            for dataset in self.tree_data["datasets"]:
                # Get condition for this dataset
                condition = self.tree_data["conditions"].get(dataset, "")

                for channel in self.tree_data["channels"]:
                    # Create tag with condition prefix if available
                    if condition:
                        tag = f"{condition}_{dataset}_{channel}"
                    else:
                        tag = f"{dataset}_{channel}"
                    tags.append(tag)

                    # Get file path
                    file_path = self.tree_data["file_paths"][dataset][channel]
                    if (file_path[0] == "'" and file_path[-1] == "'") or (
                        file_path[0] == '"' and file_path[-1] == '"'
                    ):
                        file_path = file_path[1:-1]
                    file_path = self.pathparser.convert_path(
                        file_path, host_cluster
                    )
                    filepaths.append(file_path)

            datasets = {"#tags": tags, "filepath": filepaths}

        # Helper function to format parameter values
        def format_value(value):
            """Format a parameter value for Python code."""
            if isinstance(value, str):
                # # Check if it's a command reference
                # if value.startswith(("before@", "start@", "sum(", "max(", "min(")):
                #     # Command references are stored as strings but should be tuples in code
                #     # Parse the format: "timing@index: module.result" or "cmd(timing@...)"
                #     if "(" in value:
                #         # Extract command and reference
                #         cmd = value.split("(")[0]
                #         ref = value[len(cmd)+1:-1]  # Remove cmd( and )
                #         return f'("{cmd}", "{ref}")'
                #     else:
                #         # Direct reference like "before@0: module.result"
                #         return f'("$get_previous_module_result", "{value}")'

                # Check if it looks like a path
                if (
                    "/" in value
                    or "\\" in value
                    or value.endswith(
                        (".yaml", ".hdf5", ".h5", ".tif", ".png", ".jpg")
                    )
                ):
                    # Convert path from local to host machine style
                    converted_path = self.pathparser.convert_path(
                        value, host_cluster
                    )
                    # Use os.path.join for path-like strings
                    parts = converted_path.replace("\\", "/").split("/")
                    if len(parts) > 1:
                        if parts[0] == "":
                            parts[0] = "/"
                        return f"os.path.join({', '.join(repr(p) for p in parts)})"

                return repr(value)
            elif isinstance(value, dict):
                # Format nested dicts recursively
                items = [
                    f"{repr(k)}: {format_value(v)}" for k, v in value.items()
                ]
                return "{" + ", ".join(items) + "}"
            elif isinstance(value, (list, tuple)):
                items = [format_value(v) for v in value]
                bracket = "[" if isinstance(value, list) else "("
                close = "]" if isinstance(value, list) else ")"
                return bracket + ", ".join(items) + close
            elif value is None:
                return "None"
            elif isinstance(value, bool):
                return "True" if value else "False"
            else:
                return str(value)

        # Format workflow modules
        def format_modules(modules):
            """Format list of (module_name, params) tuples as Python code."""
            if not modules:
                return "[]"

            lines = ["["]
            for module_name, params in modules:
                lines.append("    (")
                lines.append(f'        "{module_name}",')
                lines.append("        {")

                for param_name, param_value in params.items():
                    formatted_value = format_value(param_value)
                    lines.append(
                        f'            "{param_name}": {formatted_value},'
                    )

                lines.append("        },")
                lines.append("    ),")
            lines.append("]")
            return "\n".join(lines)

        # Generate script content
        script_lines = [
            "#!/usr/bin/env python",
            '"""',
            f"Script Name: {filename}",
            "Generated by: picasso-workflow GUI",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Workflow type: {workflow_type_name}",
            '"""',
            "import os",
        ]
        if not use_autodetect and not no_input_files:
            # io.save_info writes src_loc.yaml from the explicit datasets
            # dict. In auto-detect mode find_dnapaint_raw handles that
            # itself; in no-input-files mode there is no src_loc.yaml.
            script_lines.append("from picasso import io")

        # Add appropriate import based on workflow type
        if workflow_type_index == 0:  # Single Workflow
            if use_autodetect:
                script_lines.append(
                    "from picasso_workflow.metaworkflow import "
                    "find_dnapaint_raw, SingleWorkflowCoordinator"
                )
            else:
                script_lines.append(
                    "from picasso_workflow.metaworkflow import SingleWorkflowCoordinator"
                )
        elif workflow_type_index == 1:  # Aggregation Workflow
            script_lines.append(
                "from picasso_workflow.metaworkflow import AggregationWorkflowCoordinator"
            )
        elif workflow_type_index == 2:  # Investigation Workflow
            script_lines.append(
                "from picasso_workflow.metaworkflow import InvestigationWorkflowCoordinator"
            )

        cf_url = self.confluence_url_edit.text()
        if cf_url == "":
            cf_url = "os.getenv('CONFLUENCE_URL')"
        else:
            cf_url = f'"{cf_url}"'
        cf_username = self.confluence_username_edit.text()
        if cf_username == "":
            cf_username = "os.getenv('CONFLUENCE_USERNAME')"
        else:
            cf_username = f'"{cf_username}"'
        cf_space = self.confluence_space_edit.text()
        if cf_space == "":
            cf_space = "os.getenv('CONFLUENCE_SPACE')"
        else:
            cf_space = f'"{cf_space}"'
        # The token is a secret: never bake it into the generated script,
        # always read it from the environment at run time.
        cf_token = "os.getenv('CONFLUENCE_TOKEN')"
        cf_ppage = self.confluence_parent_page_edit.text()
        if cf_ppage == "":
            cf_ppage = "os.getenv('CONFLUENCE_BASE_PAGE')"
        else:
            cf_ppage = f'"{cf_ppage}"'

        script_lines.extend(
            [
                "",
                "",
                "# Confluence configuration (set via environment variables)",
                f"confluence_url = {cf_url}",
                f"confluence_token = {cf_token}",
                f"confluence_space = {cf_space}",
                f"confluence_username = {cf_username}",
                f"base_page = {cf_ppage}",
            ]
        )

        if not use_autodetect and not no_input_files:
            # Explicit dataset configuration. In auto-detect mode the
            # datasets dict is built at run time by find_dnapaint_raw; in
            # no-input-files mode there is no datasets dict at all.
            script_lines.extend(
                [
                    "",
                    "",
                    "# Dataset configuration",
                    "datasets = {",
                ]
            )
            # Add datasets
            for key, values in datasets.items():
                if isinstance(values, list) and len(values) == 1:
                    formatted = format_value(values[0])
                    script_lines.append(f"    {repr(key)}: {formatted},")
                else:
                    script_lines.append(f"    {repr(key)}: [")
                    for value in values:
                        formatted = format_value(value)
                        script_lines.append(f"        {formatted},")
                    script_lines.append("    ],")
            script_lines.append("}")

        script_lines.extend(
            [
                "",
                "",
                "# Single dataset workflow modules",
            ]
        )
        script_lines.append(
            "workflow_modules_sgl = "
            + format_modules(self.single_workflow_modules)
        )

        script_lines.extend(
            [
                "",
                "",
                "# Aggregation workflow modules",
            ]
        )
        script_lines.append(
            "workflow_modules_agg = "
            + format_modules(self.aggregation_workflow_modules)
        )

        main_lines = [
            "",
            "",
            'if __name__ == "__main__":',
            "    # Get working directory",
            "    # working_folder = os.path.dirname(os.path.abspath(__file__))",
            "    working_folder = os.environ.get('PWD', os.getcwd())",
        ]
        if use_autodetect:
            main_lines.extend(
                [
                    "    # parse the working folder for DNA-PAINT raw"
                    " datasets to analyse",
                    "    datasets, src_loc_file = "
                    "find_dnapaint_raw(working_folder)",
                ]
            )
        elif not no_input_files:
            main_lines.extend(
                [
                    "    src_loc_file = os.path.join(working_folder, 'src_loc.yaml')",
                    "    io.save_info(src_loc_file, [datasets])",
                ]
            )
        if not no_input_files:
            main_lines.extend(
                [
                    "",
                    "    print('datasets', datasets)",
                    "    print('src_loc', src_loc_file)",
                ]
            )
        main_lines.extend(
            [
                "    analysis_name = os.path.split(working_folder)[-1]",
                "",
            ]
        )
        script_lines.extend(main_lines)

        # Add coordinator creation based on workflow type
        always_save = self.always_save.isChecked()
        document_confluence = self.document_confluence_checkbox.isChecked()
        document_html = self.document_html_checkbox.isChecked()
        if workflow_type_index == 0:  # Single Workflow
            # script_lines.extend([
            #     "    # Create single workflow runner",
            #     "    runner = WorkflowRunner(",
            #     "        working_folder=working_folder,",
            #     "        analysis_name=analysis_name,",
            #     "        confluence_url=confluence_url,",
            #     "        confluence_space=confluence_space,",
            #     "        confluence_token=confluence_token,",
            #     "        confluence_username=confluence_username,",
            #     "        base_page=base_page,",
            #     "    )",
            #     "",
            #     "    # Run workflow",
            #     "    runner.run_workflow(workflow_modules_sgl)",
            # ])
            if no_input_files:
                # run the workflow exactly once with no dataset; the
                # marker comment lets the GUI restore the mode on load.
                src_loc_arg_line = f"        None,  {self._NOFILES_MARKER}"
            else:
                src_loc_arg_line = "        src_loc_file,"
            script_lines.extend(
                [
                    "    # Create single workflow coordinator",
                    "    coordinator = SingleWorkflowCoordinator(",
                    src_loc_arg_line,
                    "        analysis_name, working_folder,",
                    "        confluence_url, confluence_space, confluence_token,",
                    "        confluence_username=confluence_username,",
                    "        base_page=base_page,",
                    f"        dest_machine='{login_node}',",
                    f"        always_save={always_save},",
                    f"        document_confluence={document_confluence},",
                    f"        document_html={document_html},",
                    "    )",
                    "",
                    "    # Run workflow",
                    "    coordinator.run_analysis(workflow_modules_sgl)",
                ]
            )
        elif workflow_type_index == 1:  # Aggregation Workflow
            script_lines.extend(
                [
                    "    # Create aggregation workflow coordinator",
                    "    coordinator = AggregationWorkflowCoordinator(",
                    "        src_loc_file, analysis_name, working_folder,",
                    "        confluence_url, confluence_space, confluence_token,",
                    "        confluence_username=confluence_username,",
                    "        base_page=base_page,",
                    f"        dest_machine='{login_node}',",
                    f"        always_save={always_save},",
                    f"        document_confluence={document_confluence},",
                    f"        document_html={document_html},",
                    "    )",
                    "",
                    "    # Run analysis",
                    "    coordinator.run_analysis(workflow_modules_sgl, workflow_modules_agg)",
                ]
            )
        elif workflow_type_index == 2:  # Investigation Workflow
            script_lines.extend(
                [
                    "    # Create investigation workflow coordinator",
                    "    coordinator = InvestigationWorkflowCoordinator(",
                    "        src_loc_file, analysis_name, working_folder,",
                    "        confluence_url, confluence_space, confluence_token,",
                    "        base_page=base_page,",
                    "        confluence_username=confluence_username,",
                    f"        dest_machine='{login_node}',",
                    f"        always_save={always_save},",
                    f"        document_confluence={document_confluence},",
                    f"        document_html={document_html},",
                    "    )",
                    "",
                    "    # Run investigation",
                    "    coordinator.run_investigation(workflow_modules_sgl, workflow_modules_agg)",
                ]
            )

        script_lines.append("")  # Final newline

        # Write script to file
        script_content = "\n".join(script_lines)

        # Get output path from results folder
        results_folder = self.results_folder_display.text()
        if results_folder:
            output_path = os.path.join(results_folder, filename)
        else:
            output_path = filename

        # write with UNIX style newlines ('\n') instead of DOS ('\r\n')
        with open(output_path, "w", newline="\n") as f:
            f.write(script_content)

        # Make script executable on Unix systems
        import stat

        try:
            st = os.stat(output_path)
            os.chmod(
                output_path,
                st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
            )
        except Exception:
            pass  # Windows doesn't support chmod

        # print(f"Created workflow script: {output_path}")
        return output_path

    def assemble_slurm_scripts(self):
        import getpass

        host_cluster = str(self.cluster_host_combo.currentText())  # "hpcl8XXX"
        login_node = CONFIG["SlurmLoginNodes"][host_cluster]  # "hpcl8001"
        # print(f"'{host_cluster}', '{login_node}'")

        results_folder_local = self.results_folder_display.text()
        if (
            results_folder_local[0] == "'" and results_folder_local[-1] == "'"
        ) or (
            results_folder_local[0] == '"' and results_folder_local[-1] == '"'
        ):
            results_folder_local = results_folder_local[1:-1]
        if not os.path.exists(results_folder_local):
            os.makedirs(results_folder_local)
        results_folder_host = self.pathparser.convert_path(
            results_folder_local, host_cluster
        )

        if self.cluster_username_edit.text().strip() == "$USER":
            username = getpass.getuser()
        else:
            username = self.cluster_username_edit.text().strip()

        ssh_key_path_options = [
            os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
            os.path.join(os.path.expanduser("~"), ".ssh", "id_ed25519"),
        ]
        for sshpath in ssh_key_path_options:
            ssh_key_path = sshpath
            if os.path.exists(ssh_key_path):
                break
        # ssh_key_path = "~/.ssh/id_rsa"
        self.slurm_communicator = SlurmCommunicator(
            login_node, username, port=22, ssh_key_path=ssh_key_path
        )

        self.slurm_communicator.test_connection()

        scriptname = "start_workflow.py"
        python_script_path = self.create_python_script(
            host_cluster, login_node, scriptname
        )

        # Use the results folder's last directory as a meaningful job name
        # (falls back to "mypwjob" if it cannot be determined). Sanitize to
        # SLURM-safe characters: keep [A-Za-z0-9._-], collapse any run of
        # other characters (e.g. whitespace) into a single underscore.
        job_name = os.path.basename(os.path.normpath(results_folder_local))
        job_name = re.sub(r"[^A-Za-z0-9._-]+", "_", job_name).strip("_")
        if job_name in ("", "."):
            job_name = "mypwjob"
        slurm_options = {
            "nodes": self.cluster_nodes_spin.value(),
            # One task (rank) per node, each with cpus-per-task cores. This
            # makes `srun` launch exactly #nodes ranks, so SLURM_NTASKS equals
            # the node count and picasso-workflow distributes the single
            # workflows across the nodes (one rank per node).
            "ntasks-per-node": 1,
            "cpus-per-task": self.cluster_cores_spin.value(),
            "mem": self.cluster_memory_edit.text(),
            "time": self.cluster_timeout_edit.text(),
            # "mail-type": "ALL",
            # "mail-user": f"{username}@biochem.mpg.de",
        }
        if (email := self.slurm_email_edit.text().strip()) != "":
            slurm_options["mail-user"] = email
            slurm_options["mail-type"] = "ALL"

        # Request GPUs only when asked for (--gres=gpu:N)
        n_gpus = self.cluster_gpus_spin.value()
        if n_gpus > 0:
            slurm_options["gres"] = f"gpu:{n_gpus}"

        # use_pw_mod = self.cluster_use_module.isChecked()

        commands = self.slurm_communicator.assemble_slurm_commands(
            host_cluster, scriptname=scriptname
        )
        # scriptname=scriptname, use_pw_module=True)
        # scriptname=scriptname, use_pw_module=False)
        script_content = self.slurm_communicator.create_slurm_script(
            job_name,
            commands,
            slurm_options=slurm_options,
            output_file=f"{results_folder_host}/logs/%A.log",
            error_file=f"{results_folder_host}/logs/%A_err.log",
            working_directory=results_folder_host,
        )
        script_path = self.slurm_communicator.write_slurm_script(
            script_content, results_folder_local
        )

        # Persist the exact configuration that is about to be submitted to
        # the cluster. We log it twice: once via the logger (the same file
        # all the workflow-config edits land in) and once as a sibling
        # YAML next to the generated scripts so the run folder is
        # self-describing if the logfile is later rotated/lost.
        submission_summary = {
            "host_cluster": host_cluster,
            "login_node": login_node,
            "username": username,
            "ssh_key_path": ssh_key_path,
            "results_folder_local": results_folder_local,
            "results_folder_host": results_folder_host,
            "job_name": job_name,
            "slurm_options": slurm_options,
            "scriptname": scriptname,
            "python_script_path": python_script_path,
            "slurm_script_path": (
                script_path
                if isinstance(script_path, str)
                else str(script_path)
            ),
            "workflow_config": {
                "files": self._files_config_snapshot(),
                "modules": self._modules_config_snapshot(),
            },
        }
        logger.info(
            f"SLURM submission prepared: host={host_cluster} "
            f"login={login_node} user={username} "
            f"job={job_name} "
            f"results_folder={results_folder_local} "
            f"slurm_options={slurm_options}"
        )
        logger.info(
            "SLURM submission configuration:\n"
            + yaml.safe_dump(submission_summary, sort_keys=False)
        )
        logger.debug(
            "Generated start_workflow.py content:\n"
            + _read_text_safe(python_script_path)
        )
        logger.debug("Generated SLURM script content:\n" + script_content)
        # Sidecar file inside the run folder. Best-effort: if the disk
        # write fails we have already logged the same info above.
        try:
            sidecar = os.path.join(
                results_folder_local, "slurm_submission_config.yaml"
            )
            with open(sidecar, "w", newline="\n") as f:
                yaml.safe_dump(submission_summary, f, sort_keys=False)
            logger.info(f"Wrote SLURM submission config to {sidecar}")
        except OSError as exc:
            logger.warning(
                f"Could not write slurm_submission_config.yaml: {exc}"
            )

        return host_cluster, script_path

    def start_slurm(self):
        """"""
        logger.info("start_slurm: assembling SLURM scripts")
        host_cluster, script_path = self.assemble_slurm_scripts()
        logger.info(
            f"start_slurm: submitting job to {host_cluster} "
            f"using script {script_path}"
        )
        result = self.slurm_communicator.submit_job(
            script_path, host_cluster, additional_options=None
        )
        logger.info(
            f"start_slurm: submission result success={result['success']} "
            f"job_id={result.get('job_id')} "
            f"stdout={result.get('stdout', '').strip()!r} "
            f"stderr={result.get('stderr', '').strip()!r}"
        )

        # Store and display job ID
        if result["success"] and result["job_id"]:
            self.job_id_input.setText(str(result["job_id"]))
            self.job_info_display.append(
                f"Job submitted successfully!\nJob ID: {result['job_id']}"
            )
            # print(f"Starting SLURM on Cluster - Job ID: {result['job_id']}")
        else:
            self.job_info_display.append(
                f"Job submission failed!\n{result['stderr']}"
            )
            print("Failed to start SLURM on Cluster")

    def estimate_start(self):
        host_cluster, script_path = self.assemble_slurm_scripts()
        result = self.slurm_communicator.submit_job(
            script_path, host_cluster, additional_options=["--test-only"]
        )

        self.job_info_display.append(
            f"Job start estimation:\n{str(result['stderr'])}"
        )
        # # Store and display job ID
        # if result["success"] and result["job_id"]:
        #     self.job_id_input.setText(str(result["job_id"]))
        #     self.job_info_display.append(
        #         f"Job submitted successfully!\nJob ID: {result['job_id']}"
        #     )
        #     # print(f"Starting SLURM on Cluster - Job ID: {result['job_id']}")
        # else:
        #     self.job_info_display.append(
        #         f"Job submission failed!\n{result['stderr']}"
        #     )
        #     print("Failed to start SLURM on Cluster")

    def start_locally(self):
        """"""
        # TODO: load workflow
        print("starting workflow locally")

    def on_cancel_job(self):
        """Cancel the current SLURM job."""
        if (
            not hasattr(self, "slurm_communicator")
            or self.slurm_communicator is None
        ):
            self.job_info_display.append(
                "Error: Not connected to SLURM cluster.\nPlease submit a job first."
            )
            return

        job_id = self.job_id_input.text().strip()
        if not job_id:
            self.job_info_display.append(
                "Error: No job ID specified.\nPlease enter a job ID."
            )
            return

        try:
            job_id_int = int(job_id)
            result = self.slurm_communicator.cancel_job(job_id_int)

            if result["success"]:
                self.job_info_display.append(
                    f"Job {job_id} cancelled successfully."
                )
            else:
                self.job_info_display.append(
                    f"Failed to cancel job {job_id}:\n{result['stderr']}"
                )
        except ValueError:
            self.job_info_display.append(
                f"Error: Invalid job ID '{job_id}'. Must be a number."
            )

    def on_show_job_status(self):
        """Display the status of the current SLURM job."""
        if (
            not hasattr(self, "slurm_communicator")
            or self.slurm_communicator is None
        ):
            self.job_info_display.append(
                "Error: Not connected to SLURM cluster.\nPlease submit a job first."
            )
            return

        job_id = self.job_id_input.text().strip()
        if not job_id:
            self.job_info_display.append(
                "Error: No job ID specified.\nPlease enter a job ID."
            )
            return

        try:
            job_id_int = int(job_id)
            result = self.slurm_communicator.get_job_status(job_id_int)

            if result["success"]:
                self.job_info_display.append(f"\n=== Job {job_id} Status ===")
                self.job_info_display.append(f"Status: {result['status']}")
                if result["details"]:
                    self.job_info_display.append("Details:")
                    for key, value in result["details"].items():
                        if value:  # Only show non-empty values
                            self.job_info_display.append(f"  {key}: {value}")
            else:
                self.job_info_display.append(
                    f"Failed to get status for job {job_id}:\n{result.get('error', 'Unknown error')}"
                )
        except ValueError:
            self.job_info_display.append(
                f"Error: Invalid job ID '{job_id}'. Must be a number."
            )

    def on_list_jobs(self):
        """List all SLURM jobs for the current user."""
        if (
            not hasattr(self, "slurm_communicator")
            or self.slurm_communicator is None
        ):
            self.job_info_display.append(
                "Error: Not connected to SLURM cluster.\nPlease submit a job first."
            )
            return

        result = self.slurm_communicator.list_jobs()

        if result["success"]:
            jobs = result.get("jobs", [])
            if jobs:
                self.job_info_display.append("\n=== Your SLURM Jobs ===")
                self.job_info_display.append(
                    f"{'Job ID':<10} {'Status':<12} {'Name':<20} {'Time':<10}"
                )
                self.job_info_display.append("-" * 52)
                for job in jobs:
                    job_id = job.get("job_id", "N/A")
                    status = job.get("status", "N/A")
                    name = job.get("job_name", "N/A")
                    time = job.get("time", "N/A")
                    self.job_info_display.append(
                        f"{job_id:<10} {status:<12} {name:<20} {time:<10}"
                    )
            else:
                self.job_info_display.append("No jobs found for current user.")
        else:
            self.job_info_display.append(
                f"Failed to list jobs:\n{result.get('error', 'Unknown error')}"
            )

    def on_show_queue_info(self):
        """Display SLURM queue/partition information."""
        if (
            not hasattr(self, "slurm_communicator")
            or self.slurm_communicator is None
        ):
            self.job_info_display.append(
                "Error: Not connected to SLURM cluster.\nPlease submit a job first."
            )
            return

        result = self.slurm_communicator.get_queue_info()

        if result["success"]:
            partitions = result.get("partitions", [])
            if partitions:
                self.job_info_display.append(
                    "\n=== SLURM Queue Information ==="
                )
                self.job_info_display.append(
                    f"{'Partition':<15} {'Avail':<8} {'Nodes':<8} {'State':<12}"
                )
                self.job_info_display.append("-" * 43)
                for partition in partitions:
                    name = partition.get("name", "N/A")
                    avail = partition.get("availability", "N/A")
                    nodes = partition.get("nodes", "N/A")
                    state = partition.get("state", "N/A")
                    self.job_info_display.append(
                        f"{name:<15} {avail:<8} {nodes:<8} {state:<12}"
                    )
            else:
                self.job_info_display.append(
                    "No partition information available."
                )
        else:
            self.job_info_display.append(
                f"Failed to get queue info:\n{result.get('error', 'Unknown error')}"
            )

    def _clear_parameter_layout(self):
        """Clear all widgets from the module parameters layout."""
        while self.module_parameters_layout.count():
            item = self.module_parameters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_parameter_widget(self, param_name, param_metadata):
        """Factory method to create appropriate widget based on parameter type.

        Parameters
        ----------
        param_name: Name of the parameter
        param_metadata: Metadata dict with 'type', 'default', 'min', 'max', etc.

        Returns
        -------
        tuple: (widget, original_type_string)
        """
        # Check for options first - if present, use combo box regardless of type
        if "options" in param_metadata:
            widget = QtWidgets.QComboBox()
            options = param_metadata["options"]
            widget.addItems([str(opt) for opt in options])
            default = param_metadata.get("default")
            if default is not None:
                index = widget.findText(str(default))
                if index >= 0:
                    widget.setCurrentIndex(index)
            return widget, "options"

        param_type = param_metadata.get("type", "str")

        if param_type == "int":
            widget = QtWidgets.QSpinBox()
            widget.setMinimum(param_metadata.get("min", -2147483648))
            widget.setMaximum(param_metadata.get("max", 2147483647))
            widget.setSingleStep(param_metadata.get("step", 1))
            default = param_metadata.get("default")
            if default is not None:
                widget.setValue(int(default))
            return widget, "int"

        elif param_type == "float":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setMinimum(param_metadata.get("min", -1e308))
            widget.setMaximum(param_metadata.get("max", 1e308))
            widget.setSingleStep(param_metadata.get("step", 0.1))
            widget.setDecimals(6)
            default = param_metadata.get("default")
            if default is not None:
                widget.setValue(float(default))
            return widget, "float"

        elif param_type == "bool":
            widget = QtWidgets.QCheckBox()
            default = param_metadata.get("default", False)
            widget.setChecked(bool(default))
            return widget, "bool"

        elif param_type == "path":
            widget = DroppableLineEdit()
            default = param_metadata.get("default", "")
            if default is not None:
                widget.setText(str(default))
            if param_metadata.get("required", False):
                widget.setPlaceholderText(
                    "Required - drag file here or type path"
                )
            else:
                widget.setPlaceholderText("Drag file here or type path")
            return widget, "path"

        else:  # str or fallback
            widget = QtWidgets.QLineEdit()
            default = param_metadata.get("default", "")
            if default is not None:
                widget.setText(str(default))
            if param_metadata.get("required", False):
                widget.setPlaceholderText("Required")
            return widget, "str"

    def _parse_literal_string(self, text):
        """Try to parse a string representation of a Python literal back into its actual type.

        Handles tuples, lists, dicts, numbers, bools, None, etc.

        Parameters
        ----------
        text: String that might be a Python literal, e.g.,
              "('$$map', 'filepath')", "[1, 2, 3]", "{'key': 'value'}"

        Returns
        -------
        Parsed Python object if successful, original string otherwise
        """
        if not isinstance(text, str):
            return text

        stripped = text.strip()

        # Check if it looks like a Python literal
        # (starts with bracket/paren/brace or looks like a number/bool/None)
        if not stripped:
            return text

        first_char = stripped[0]
        # Check for likely Python literals
        if (
            first_char not in ("(", "[", "{", "-", "+")
            and not stripped[0].isdigit()
        ):
            # Could still be None, True, False
            if stripped not in ("None", "True", "False"):
                return text

        try:
            # Try to safely evaluate the string as a Python literal
            import ast

            result = ast.literal_eval(stripped)
            # Return the parsed result (tuple, list, dict, number, bool, None, etc.)
            return result
        except (ValueError, SyntaxError):
            # If parsing fails, return original string
            pass

        return text

    def _get_widget_value(self, widget, original_type, widget_info=None):
        """Get value from widget based on its original type.

        Parameters
        ----------
        widget: The Qt widget
        original_type: Original type string ('int', 'float', 'bool', 'str', 'path', 'options', 'dict')
        widget_info: ParameterWidgetInfo (needed for dict types)

        Returns
        -------
        str or dict or tuple: String representation of the value, dict for nested parameters,
                              or tuple for command references
        """
        if (
            original_type == "dict"
            and widget_info
            and widget_info.sub_parameters
        ):
            # For dict types with nested parameters
            if isinstance(widget, QtWidgets.QCheckBox):
                # Optional dict: only include if checkbox is checked
                if not widget.isChecked():
                    return None

            # Recursively get values from sub-parameters
            nested_values = {}
            for (
                sub_param_name,
                sub_widget_info,
            ) in widget_info.sub_parameters.items():
                sub_value = self._get_widget_value(
                    sub_widget_info.widget,
                    sub_widget_info.original_type,
                    sub_widget_info,
                )
                if sub_value is not None:  # Only include non-None values
                    nested_values[sub_param_name] = sub_value
            return nested_values

        elif isinstance(widget, QtWidgets.QLineEdit):
            text = widget.text()
            # Try to parse as Python literal (tuple, list, dict, etc.)
            return self._parse_literal_string(text)
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        elif isinstance(widget, QtWidgets.QSpinBox):
            return widget.value()  # Return native int
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            return widget.value()  # Return native float
        elif isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()  # Return native bool
        else:
            raise TypeError(f"Unknown widget type: {type(widget)}")

    def _set_widget_value(
        self, widget, value_data, original_type, widget_info=None
    ):
        """Set widget value from actionable format (native Python types).

        Parameters
        ----------
        widget: The Qt widget
        value_data: Value in actionable format (int, float, bool, str, tuple, list, dict, etc.)
        original_type: Original type string ('int', 'float', 'bool', 'str', 'path', 'options', 'dict')
        widget_info: ParameterWidgetInfo (needed for dict types)
        """
        if (
            original_type == "dict"
            and widget_info
            and widget_info.sub_parameters
        ):
            # For dict types with nested parameters
            if isinstance(value_data, dict):
                # value_data is a dict of nested values
                if isinstance(widget, QtWidgets.QCheckBox):
                    # Optional dict: check the checkbox to show nested params
                    widget.setChecked(True)

                    # Manually call the toggle function to ensure rows are shown
                    # (signal may not fire properly when loading)
                    if widget_info.toggle_function:
                        widget_info.toggle_function(2)  # 2 = Qt.Checked

                # Explicitly show nested rows using both methods as fallback
                # Method 1: Through sub_parameters
                for sub_widget_info in widget_info.sub_parameters.values():
                    if sub_widget_info.row_widget:
                        sub_widget_info.row_widget.setVisible(True)

                # Method 2: Through nested_rows attribute (if it exists)
                if hasattr(widget_info.row_widget, "nested_rows"):
                    for nested_row in widget_info.row_widget.nested_rows:
                        nested_row.setVisible(True)

                # Recursively set values in sub-parameters
                for (
                    sub_param_name,
                    sub_widget_info,
                ) in widget_info.sub_parameters.items():
                    if sub_param_name in value_data:
                        sub_value_data = value_data[sub_param_name]

                        # Recursively set nested parameter value
                        self._set_widget_value(
                            sub_widget_info.widget,
                            sub_value_data,
                            sub_widget_info.original_type,
                            sub_widget_info,
                        )
            # If value_data is None or not a dict, leave dict parameter unchecked/empty

        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value_data))
        elif isinstance(widget, QtWidgets.QComboBox):
            index = widget.findText(str(value_data))
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, QtWidgets.QSpinBox):
            try:
                widget.setValue(int(value_data))
            except (ValueError, TypeError):
                widget.setValue(widget.minimum())
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            try:
                widget.setValue(float(value_data))
            except (ValueError, TypeError):
                widget.setValue(widget.minimum())
        elif isinstance(widget, QtWidgets.QCheckBox):
            # Handle native bools and string representations
            if isinstance(value_data, bool):
                widget.setChecked(value_data)
            else:
                # Handle string representations
                is_checked = str(value_data).lower() in (
                    "true",
                    "1",
                    "yes",
                    "on",
                )
                widget.setChecked(is_checked)

    def _on_cmd_button_clicked(self, param_name):
        """Handle cmd button click - opens prior result dialog.

        Parameters
        ----------
        param_name: Name of the parameter to populate
        """
        # Determine which workflow list to use
        current_tab_index = self.workflow_tabs.currentIndex()
        if current_tab_index == 0:
            workflow_modules = self.single_workflow_modules
            curr_module_index = self.single_workflow_list.currentRow()
        elif current_tab_index == 1:
            workflow_modules = self.aggregation_workflow_modules
            curr_module_index = self.aggregation_workflow_list.currentRow()
        else:
            return

        # Create dialog
        dialog = ParameterCmdDialog(
            workflow_modules, self.module_descriptor, curr_module_index, self
        )
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            command_str = dialog.command_result.text()
            # Convert widget to QLineEdit and populate
            self._convert_widget_to_textbox(param_name, command_str)

    # def _handle_prior_result_command(self, param_name):
    #     """Open dialog for selecting prior result reference.

    #     Args:
    #         param_name: Name of the parameter to populate
    #     """

    #     if not workflow_modules:
    #         QtWidgets.QMessageBox.warning(
    #             self,
    #             "No Prior Modules",
    #             "There are no prior modules in the workflow to reference."
    #         )
    #         # Reset command combo
    #         widget_info = self.parameter_widgets[param_name]
    #         widget_info.command_combo.setCurrentIndex(0)
    #         return

    def _convert_widget_to_textbox(self, param_name, initial_value=""):
        """Convert parameter widget to QLineEdit for command values.

        Parameters
        ----------
        param_name: Name of the parameter
        initial_value: Initial text to populate
        """
        widget_info = self.parameter_widgets[param_name]
        row_widget = widget_info.row_widget
        row_layout = row_widget.layout()

        # Remove old widget (at index 1, between label and command combo)
        old_widget = widget_info.widget
        row_layout.removeWidget(old_widget)
        old_widget.deleteLater()

        # Create new QLineEdit
        new_widget = QtWidgets.QLineEdit()
        new_widget.setText(initial_value)
        new_widget.setToolTip(widget_info.metadata.get("description", ""))
        new_widget.editingFinished.connect(self._on_parameter_changed)

        # Insert at position 1 (after label, before command combo)
        row_layout.insertWidget(1, new_widget, stretch=2)

        # Update stored reference
        widget_info.widget = new_widget

        # Trigger validation update
        self._on_parameter_changed()

    def _create_parameter_row(
        self, param_name, param_metadata, indent_level=0
    ):
        """Create a parameter row with widgets, supporting nested dicts.

        Parameters
        ----------
        param_name: Name of the parameter
        param_metadata: Metadata dict with type, description, default, etc.
        indent_level: Indentation level for nested parameters (0 = top level)

        Returns
        -------
        ParameterWidgetInfo: Widget info for this parameter
        """
        # Create row container
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(indent_level * 20, 2, 0, 2)

        # Create label
        label = QtWidgets.QLabel(param_name)
        if param_metadata.get("required", False):
            font = label.font()
            font.setBold(True)
            label.setFont(font)
        row_layout.addWidget(label, stretch=0)

        # Check if this is a dict with nested properties
        param_type = param_metadata.get("type", "str")
        properties = param_metadata.get("properties", {})

        if param_type == "dict" and properties:
            # Handle nested dict parameter
            is_required = param_metadata.get("required", False)
            sub_parameters = {}
            sub_rows = []

            if is_required:
                # # Required dict: show label, create nested rows immediately
                # placeholder_widget = QtWidgets.QLabel("(nested parameters below)")
                # placeholder_widget.setStyleSheet("color: gray; font-style: italic;")
                # row_layout.addWidget(placeholder_widget, stretch=2)
                checkbox = QtWidgets.QCheckBox("Enable")
                checkbox.setChecked(True)
                row_layout.addWidget(checkbox, stretch=2)

                # Create cmd button (disabled for dict types)
                cmd_button = QtWidgets.QPushButton("cmd")
                cmd_button.setFixedWidth(50)
                cmd_button.setEnabled(False)  # Disable for dict types
                cmd_button.setToolTip("Create command to assign this value.")
                row_layout.addWidget(cmd_button, stretch=0)

                # Create nested parameter rows
                for sub_param_name, sub_param_metadata in properties.items():
                    sub_widget_info = self._create_parameter_row(
                        sub_param_name, sub_param_metadata, indent_level + 1
                    )
                    sub_parameters[sub_param_name] = sub_widget_info
                    sub_rows.append(sub_widget_info.row_widget)

                # Connect checkbox to trigger auto-save
                checkbox.stateChanged.connect(self._on_parameter_changed)

            else:
                # Optional dict: show checkbox, create nested rows (initially hidden)
                checkbox = QtWidgets.QCheckBox("Enable")
                checkbox.setChecked(False)
                row_layout.addWidget(checkbox, stretch=2)

                # Create cmd button (disabled for dict types)
                cmd_button = QtWidgets.QPushButton("cmd")
                cmd_button.setFixedWidth(50)
                cmd_button.setEnabled(False)
                row_layout.addWidget(cmd_button, stretch=0)

                # Create nested parameter rows (initially hidden)
                for sub_param_name, sub_param_metadata in properties.items():
                    sub_widget_info = self._create_parameter_row(
                        sub_param_name, sub_param_metadata, indent_level + 1
                    )
                    sub_parameters[sub_param_name] = sub_widget_info
                    sub_rows.append(sub_widget_info.row_widget)
                    sub_widget_info.row_widget.setVisible(
                        False
                    )  # Hide initially

                # Connect checkbox to toggle visibility
                def toggle_nested_params(state):
                    # stateChanged passes int (0=unchecked, 2=checked), convert to bool
                    is_checked = bool(state)
                    for sub_row in sub_rows:
                        sub_row.setVisible(is_checked)
                    # Also trigger parameter update for auto-save
                    self._on_parameter_changed()

                checkbox.stateChanged.connect(toggle_nested_params)

            # Store nested rows in row_widget for later access
            row_widget.nested_rows = sub_rows

            widget_info = ParameterWidgetInfo(
                widget=checkbox,
                cmd_button=cmd_button,
                row_widget=row_widget,
                metadata=param_metadata,
                original_type="dict",
                sub_parameters=sub_parameters,
                toggle_function=(
                    toggle_nested_params if not is_required else None
                ),
            )
            return widget_info

        else:
            # Regular parameter (not a dict)
            widget, original_type = self._create_parameter_widget(
                param_name, param_metadata
            )

            # Set tooltip
            description = param_metadata.get("description", "")
            description = textwrap.fill(description)
            if description:
                widget.setToolTip(description)
                label.setToolTip(description)

            # Connect type-specific validation signal
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(self._on_parameter_changed)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_parameter_changed)
            elif isinstance(
                widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
            ):
                widget.valueChanged.connect(self._on_parameter_changed)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(self._on_parameter_changed)

            row_layout.addWidget(widget, stretch=2)

            # Create cmd button
            cmd_button = QtWidgets.QPushButton("cmd")
            cmd_button.setFixedWidth(50)
            cmd_button.clicked.connect(
                lambda checked, pn=param_name: self._on_cmd_button_clicked(pn)
            )
            row_layout.addWidget(cmd_button, stretch=0)

            widget_info = ParameterWidgetInfo(
                widget=widget,
                cmd_button=cmd_button,
                row_widget=row_widget,
                metadata=param_metadata,
                original_type=original_type,
            )
            return widget_info

    def _populate_parameter_widgets(self, module_params):
        """Create and populate parameter entry widgets from module parameters.

        Parameters
        ----------
        module_params: Dict of parameter definitions with keys as parameter names
                       and values as dicts containing 'type', 'description', 'default',
                       'required', etc.
        """
        if not module_params:
            # Show "No parameters" message if module has no parameters
            no_params_label = QtWidgets.QLabel("No parameters")
            no_params_label.setStyleSheet("color: gray; font-style: italic;")
            self.module_parameters_layout.addWidget(no_params_label)
            self.module_parameters_layout.addStretch()
            return

        for param_name, param_metadata in module_params.items():
            # Create parameter row (handles nested dicts recursively)
            widget_info = self._create_parameter_row(
                param_name, param_metadata, indent_level=0
            )
            self.parameter_widgets[param_name] = widget_info

            # Add main row to layout
            self.module_parameters_layout.addWidget(widget_info.row_widget)

            # Add nested rows if this is a dict parameter
            if hasattr(widget_info.row_widget, "nested_rows"):
                for sub_row in widget_info.row_widget.nested_rows:
                    self.module_parameters_layout.addWidget(sub_row)

        # Add stretch at the end to push widgets to the top
        self.module_parameters_layout.addStretch()

    def _validate_parameters(self):
        """Validate that all required parameters are filled.

        Returns
        -------
        bool: True if all required parameters have values, False otherwise
        """
        all_valid = True
        for param_name, widget_info in self.parameter_widgets.items():
            is_required = widget_info.metadata.get("required", False)
            widget = widget_info.widget

            # Check if empty based on widget type
            is_empty = False
            if isinstance(widget, QtWidgets.QLineEdit):
                is_empty = not widget.text().strip()
            elif isinstance(widget, QtWidgets.QComboBox):
                # Combo boxes always have a selection, never considered "empty"
                is_empty = False
            elif isinstance(
                widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)
            ):
                # Spinboxes always have a value, never considered "empty"
                is_empty = False
            elif isinstance(widget, QtWidgets.QCheckBox):
                # Checkboxes always have a state, never considered "empty"
                is_empty = False

            if is_required and is_empty:
                # Visual feedback: red border for empty required fields (only QLineEdit)
                widget.setStyleSheet("border: 1px solid red;")
                all_valid = False
            else:
                # Clear styling if valid
                widget.setStyleSheet("")

        # Enable/disable "Add module" button based on validation
        self.add_module_button.setEnabled(all_valid)
        return all_valid

    def _update_editing_workflow_item(self):
        """Update parameters in workflow list if currently editing an existing item."""
        # Only update if we're editing an existing workflow item
        if self.editing_workflow_index < 0 or self.editing_workflow_tab < 0:
            return

        # Capture current parameter values from widgets
        param_values = {}
        for param_name, widget_info in self.parameter_widgets.items():
            value = self._get_widget_value(
                widget_info.widget, widget_info.original_type, widget_info
            )
            # Skip None values (from unchecked optional dicts)
            if value is not None:
                param_values[param_name] = value

        # Update the appropriate workflow list
        if self.editing_workflow_tab == 0:  # Single Dataset Workflow
            if self.editing_workflow_index < len(self.single_workflow_modules):
                module_name, old_params = self.single_workflow_modules[
                    self.editing_workflow_index
                ]
                if old_params != param_values:
                    # Update parameters while keeping module name
                    self.single_workflow_modules[
                        self.editing_workflow_index
                    ] = (
                        module_name,
                        param_values,
                    )
                    self._log_workflow_config_event(
                        "modules.params_changed",
                        tab="single",
                        index=self.editing_workflow_index,
                        module=module_name,
                        params=param_values,
                    )
        elif self.editing_workflow_tab == 1:  # Aggregation Workflow
            if self.editing_workflow_index < len(
                self.aggregation_workflow_modules
            ):
                module_name, old_params = self.aggregation_workflow_modules[
                    self.editing_workflow_index
                ]
                if old_params != param_values:
                    # Update parameters while keeping module name
                    self.aggregation_workflow_modules[
                        self.editing_workflow_index
                    ] = (module_name, param_values)
                    self._log_workflow_config_event(
                        "modules.params_changed",
                        tab="aggregation",
                        index=self.editing_workflow_index,
                        module=module_name,
                        params=param_values,
                    )

    def _on_parameter_changed(self):
        """Called when a parameter textbox loses focus (editingFinished signal)."""
        self._validate_parameters()
        # Auto-save parameters if editing an existing workflow item
        self._update_editing_workflow_item()

    def on_module_changed(self, text):
        import textwrap

        """Update the module description when a new module is selected."""
        # Clear editing state when user manually changes module
        # (This is only called via signal when not blocked, i.e., manual user action)
        if not self.module_combobox.signalsBlocked():
            self.editing_workflow_index = -1
            self.editing_workflow_tab = -1

        if text == "Select module":
            self.current_module_desc.setText("No module selected")
            self.add_module_button.setEnabled(False)
        else:
            try:
                desc = self.module_descriptor.get_docstring(text)

                # Use inspect.cleandoc to properly handle docstrings with unindented first line
                desc = inspect.cleandoc(desc)

                # Extract only the summary part before Args/Returns/etc sections
                section_headers = [
                    "Args:",
                    "Arguments:",
                    "Parameters:",
                    "Params:",
                    "Returns:",
                    "Return:",
                    "Yields:",
                    "Yield:",
                    "Raises:",
                    "Raise:",
                    "Note:",
                    "Notes:",
                    "Example:",
                    "Examples:",
                    "See Also:",
                    "Attributes:",
                    "References:",
                    "Warnings:",
                    "Warning:",
                ]

                # Find the first occurrence of any section header
                min_index = len(desc)
                for header in section_headers:
                    index = desc.find(header)
                    if index != -1 and index < min_index:
                        min_index = index

                # Crop to summary only
                if min_index < len(desc):
                    desc = desc[:min_index].strip()

                # Join lines unless they end with a full stop
                lines = desc.split("\n")
                processed_lines = []
                current_para = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    current_para.append(line)
                    if line.endswith("."):
                        processed_lines.append(" ".join(current_para))
                        current_para = []

                # Add any remaining paragraph
                if current_para:
                    processed_lines.append(" ".join(current_para))

                desc = "\n".join(processed_lines)

                # Calculate text width based on widget width
                font_metrics = QtGui.QFontMetrics(
                    self.current_module_desc.font()
                )
                widget_width = self.current_module_desc.width()
                avg_char_width = font_metrics.averageCharWidth()
                char_width = max(
                    40, int(1.33 * widget_width / avg_char_width - 2)
                )  # -2 for margin

                # Apply text wrapping with calculated width, preserving paragraph breaks
                wrapped_paragraphs = [
                    textwrap.fill(para, width=char_width)
                    for para in processed_lines
                ]
                desc = "\n".join(wrapped_paragraphs)

                self.current_module_desc.setText(desc)
            except Exception as e:
                raise e
                self.current_module_desc.setText(
                    f"There was an error loading the description of the module {text}."
                )

            # Set up parameter entry widgets
            desc_fun = getattr(self.module_descriptor, text)
            module_params, _ = desc_fun()

            # Clear existing parameter widgets
            self._clear_parameter_layout()
            self.parameter_widgets.clear()

            # Populate new parameter widgets
            self._populate_parameter_widgets(module_params)

            # Validate parameters and update button state
            self._validate_parameters()


def _app_icon():
    """Return a QIcon for the application, or an empty QIcon on failure."""
    try:
        import importlib.resources
        import pathlib

        # Strategy 1: importlib.resources (wheel installs)
        try:
            p = importlib.resources.files("picasso_workflow").joinpath(
                "picasso-workflow.ico"
            )
            ico = pathlib.Path(str(p)).resolve()
            if ico.exists():
                return QtGui.QIcon(str(ico))
        except Exception:
            pass

        # Strategy 2: relative to this file (editable installs)
        ico = pathlib.Path(__file__).parent / "picasso-workflow.ico"
        if ico.exists():
            return QtGui.QIcon(str(ico))
    except Exception:
        pass
    return QtGui.QIcon()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(_app_icon())

    # Keep a reference so the excepthook closure can use it even before
    # Window() returns.  It is None only if construction itself raises.
    window = None

    def excepthook(exc_type, exc_value, exc_tb):
        message = "".join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        )
        try:
            lib.cancel_dialogs()
            QtCore.QCoreApplication.instance().processEvents()
        except Exception:
            pass
        # QMessageBox.critical already shows the dialog and returns the
        # clicked button (an int) — do not call .exec_() on the return value.
        QtWidgets.QMessageBox.critical(window, "An error occurred", message)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    # Install before creating Window so construction errors are caught.
    sys.excepthook = excepthook

    try:
        window = Window()
    except Exception:
        message = traceback.format_exc()
        QtWidgets.QMessageBox.critical(None, "Startup error", message)
        sys.exit(1)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

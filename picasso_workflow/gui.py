#!/usr/bin/env python
"""
Module Name: gui.py
Author: Heinrich Grabmayr
Initial Date: August 4, 2024
Description: GUI descriptor module for picasso-workflow
"""

from picasso_workflow import util
import logging
import subprocess
import os
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
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QEvent


logger = logging.getLogger(__name__)
__GUIVERSION__ = "0.1.0"


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

        Attributes:
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

        Returns:
            list of str: List of module names that can be used in workflows.
                These correspond to all the abstract methods implemented from
                AbstractModuleCollection.

        Example:
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

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
        }

        return parameters_spec, results_spec

    def analysis_documentation(self):
        """Document the parameters of the analysis machine and software.

        Creates documentation of the analysis environment including system
        information, software versions, and configuration parameters.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys: (none)
                Optional keys: (none)
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Optional keys:
                    system_info : dict
                        System and software version information

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with added system documentation
        """
        parameters_spec = {}

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "system_info": {
                "type": "dict",
                "description": "System and software version information",
                "required": False,
            },
        }

        return parameters_spec, results_spec

    def convert_zeiss_movie(self):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso.

        Converts Zeiss .czi movie files to picasso-compatible .raw format
        for subsequent analysis steps in the workflow.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filepath : str
                        Path to the input Zeiss .czi file
                Optional keys:
                    output_filepath : str
                        Custom output path for the .raw file
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    filepath_raw : str
                        Path to the output .raw file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with added file path information
        """
        parameters_spec = {
            "filepath": {
                "type": "str",
                "description": "Path to the input Zeiss .czi file",
                "extensions": [".czi"],
                "required": True,
            },
            "output_filepath": {
                "type": "str",
                "description": "Custom output path for the .raw file",
                "extensions": [".raw"],
                "required": False,
                "mode": "save",
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "filepath_raw": {
                "type": "str",
                "description": "Path to the output .raw file",
            },
        }

        return parameters_spec, results_spec

    def load_dataset_movie(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads DNA-PAINT movie data and metadata into memory for subsequent
        analysis. Optionally creates sample movies and loads camera
        configuration.

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
                        Whether to load camera configuration
                        from picasso.CONFIG
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
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
                Input parameters, potentially modified
                (sample_movie paths updated)
            results : dict
                Input results with added movie information and metadata
        """
        parameters_spec = {
            "filename": {
                "type": "str",
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
                    },
                    "frames": {
                        "type": "int",
                        "description": "Number of frames to sample",
                        "min": 1,
                    },
                    "step": {
                        "type": "int",
                        "description": "Frame step size",
                        "min": 1,
                        "default": 1,
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
        }

        return parameters_spec, results_spec

    def load_dataset_localizations(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso.

        Loads pre-computed localization data from file for analysis workflows
        that skip the identification and localization steps.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filename : str
                        Path to the localization file to load
                Optional keys:
                    additional_info : dict
                        Additional metadata to include
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    picasso version : str
                        Version of picasso library used
                    nlocs : int
                        Number of localizations loaded

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with added localization information
        """
        parameters_spec = {
            "filename": {
                "type": "str",
                "description": "Path to the localization file to load",
                "extensions": [".hdf5", ".h5"],
                "required": True,
            },
            "additional_info": {
                "type": "dict",
                "description": "Additional metadata to include",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "picasso version": {
                "type": "str",
                "description": "Version of picasso library used",
            },
            "nlocs": {
                "type": "int",
                "description": "Number of localizations loaded",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def identify(self):
        """Identifies localizations in a loaded dataset.

        Identifies potential localization sites in the loaded movie using
        net gradient thresholding. Optionally performs automatic net gradient
        detection and creates identification vs frame plots.

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
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                    folder : str
                        Output folder for generated files
                Results updated with:
                    num_identifications : int
                        Total number of identifications found
                    auto_netgrad : dict
                        Results from automatic net gradient detection
                        (if requested)
                    ids_vs_frame : dict
                        Results from identifications vs frame analysis
                        (if requested)

        Returns:
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
                    },
                    "filename": {
                        "type": "str",
                        "description": (
                            "Output filename for auto-detection plot"
                        ),
                    },
                    "start_ng": {
                        "type": "float",
                        "description": "Starting net gradient value",
                        "min": 0.0,
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
                    "Parameters for plotting identifications vs time"
                ),
                "required": False,
                "properties": {
                    "filename": {
                        "type": "str",
                        "description": "Output filename for plot",
                    }
                },
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
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

        Performs sub-pixel localization of identified spots using various
        fitting algorithms (MLE, LQ, etc.).

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    box_size : int
                        Size of the localization box in pixels
                Optional keys:
                    method : str
                        Localization method ('mle', 'lq', 'avg')
                    convergence_criterion : float
                        Convergence criterion for iterative methods
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    nlocs : int
                        Number of localizations found

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with localization statistics
        """
        parameters_spec = {
            "box_size": {
                "type": "int",
                "description": "Size of the localization box in pixels",
                "min": 3,
                "max": 21,
                "step": 2,
                "default": 7,
                "required": True,
            },
            "method": {
                "type": "str",
                "description": "Localization method",
                "options": ["mle", "lq", "avg"],
                "default": "mle",
                "required": False,
            },
            "convergence_criterion": {
                "type": "float",
                "description": "Convergence criterion for iterative methods",
                "min": 1e-6,
                "max": 1e-2,
                "default": 1e-3,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "nlocs": {
                "type": "int",
                "description": "Number of localizations found",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def export_brightfield(self):
        """Opens a single-plane tiff image and saves it to png with
        contrast adjustment.

        Exports brightfield microscopy images with automatic contrast
        adjustment for documentation and visualization purposes.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    filepath : str
                        Path to the input brightfield image
                Optional keys:
                    output_filepath : str
                        Custom output path for the PNG file
                    contrast_percentiles : tuple
                        Percentiles for contrast adjustment
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    output_filepath : str
                        Path to the exported PNG file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with export information
        """
        parameters_spec = {
            "filepath": {
                "type": "str",
                "description": "Path to the input brightfield image",
                "extensions": [".tif", ".tiff", ".png", ".jpg"],
                "required": True,
            },
            "output_filepath": {
                "type": "str",
                "description": "Custom output path for the PNG file",
                "extensions": [".png"],
                "required": False,
                "mode": "save",
            },
            "contrast_percentiles": {
                "type": "tuple",
                "description": "Percentiles for contrast adjustment",
                "length": 2,
                "element_type": "float",
                "min": 0.0,
                "max": 100.0,
                "default": [1.0, 99.0],
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "output_filepath": {
                "type": "str",
                "description": "Path to the exported PNG file",
            },
        }

        return parameters_spec, results_spec

    def render(self):
        """Renders localizations.

        Creates rendered images of localization data using various
        visualization methods and color schemes.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    oversampling : int
                        Oversampling factor for rendering
                Optional keys:
                    method : str
                        Rendering method
                    blur_method : str
                        Blur method for rendering
                    colors : list
                        Color scheme for multi-channel rendering
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    rendered_image : numpy.ndarray
                        Rendered image array
                    render_filepath : str
                        Path to saved rendered image

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with rendering outputs
        """
        parameters_spec = {
            "oversampling": {
                "type": "int",
                "description": "Oversampling factor for rendering",
                "min": 1,
                "max": 50,
                "default": 10,
                "required": True,
            },
            "method": {
                "type": "str",
                "description": "Rendering method",
                "options": ["gaussian", "hist", "smooth"],
                "default": "gaussian",
                "required": False,
            },
            "blur_method": {
                "type": "str",
                "description": "Blur method for rendering",
                "options": ["gaussian", "uniform", "none"],
                "default": "gaussian",
                "required": False,
            },
            "colors": {
                "type": "list",
                "description": "Color scheme for multi-channel rendering",
                "element_type": "str",
                "required": False,
                "options": [
                    "red",
                    "green",
                    "blue",
                    "cyan",
                    "magenta",
                    "yellow",
                ],
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "rendered_image": {
                "type": "numpy.ndarray",
                "description": "Rendered image array",
            },
            "render_filepath": {
                "type": "str",
                "description": "Path to saved rendered image",
            },
        }

        return parameters_spec, results_spec

    def undrift_rcc(self):
        """Undrifts localized data using redundant cross correlation.

        Corrects sample drift during acquisition using redundant
        cross-correlation analysis of localization data.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    segmentation : int
                        Number of segments for drift analysis
                Optional keys:
                    display : bool
                        Whether to display drift correction plots
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    drift : numpy.ndarray
                        Calculated drift values
                    drift_plot_filepath : str
                        Path to drift visualization plot

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with drift correction data
        """
        parameters_spec = {
            "segmentation": {
                "type": "int",
                "description": "Number of segments for drift analysis",
                "min": 2,
                "max": 1000,
                "default": 50,
                "required": True,
            },
            "display": {
                "type": "bool",
                "description": "Whether to display drift correction plots",
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "drift": {
                "type": "numpy.ndarray",
                "description": "Calculated drift values",
            },
            "drift_plot_filepath": {
                "type": "str",
                "description": "Path to drift visualization plot",
            },
        }

        return parameters_spec, results_spec

    def undrift_aim(self):
        """Unrift localized data using the AIM algorithm.

        Corrects sample drift using the AIM (Accelerated Iterative Method)
        algorithm for drift correction.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    segmentation : int
                        Number of segments for drift analysis
                Optional keys:
                    iterations : int
                        Number of AIM iterations
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    drift : numpy.ndarray
                        Calculated drift values

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with drift correction data
        """
        parameters_spec = {
            "segmentation": {
                "type": "int",
                "description": "Number of segments for drift analysis",
                "min": 2,
                "max": 1000,
                "default": 50,
                "required": True,
            },
            "iterations": {
                "type": "int",
                "description": "Number of AIM iterations",
                "min": 1,
                "max": 100,
                "default": 10,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "drift": {
                "type": "numpy.ndarray",
                "description": "Calculated drift values",
            },
        }

        return parameters_spec, results_spec

    def manual(self):
        """Describes a manual step, for which the workflow is paused.

        Pauses the workflow for manual intervention or user input before
        continuing with subsequent analysis steps.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    message : str
                        Message to display to user
                    wait_for_input : bool
                        Whether to wait for user confirmation
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results (unchanged)
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
        }

        return parameters_spec, results_spec

    def summarize_dataset(self):
        """Summarizes the results of a dataset analysis.

        Creates a comprehensive summary of the analysis results including
        statistics, plots, and key findings.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    include_plots : bool
                        Whether to include visualization plots
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    summary_report : dict
                        Comprehensive analysis summary

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with summary information
        """
        parameters_spec = {
            "include_plots": {
                "type": "bool",
                "description": "Whether to include visualization plots",
                "default": True,
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "summary_report": {
                "type": "dict",
                "description": "Comprehensive analysis summary",
            },
        }

        return parameters_spec, results_spec

    def density(self):
        """Calculate local localization density.

        Computes local density of localizations using various methods
        for spatial analysis and clustering preparation.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        Radius for local density calculation in nm
                Optional keys:
                    save_locs : bool
                        Whether to save density-annotated localizations
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                    nlocs : int
                        Number of localizations processed
                Optional keys:
                    density_stats : dict
                        Statistical summary of density calculations

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with density information
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
            "save_locs": {
                "type": "bool",
                "description": (
                    "Whether to save density-annotated localizations"
                ),
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
                        Whether to save clustered localization data to
                        results folder
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
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
                    "Minimum number of samples required for a cluster"
                ),
                "min": 1,
                "max": 100,
                "default": 3,
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
            "save_locs": {
                "type": "bool",
                "description": (
                    "Whether to save clustered localization data to"
                    + " results folder"
                ),
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
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
        """Perform clustering using hdbscan.

        Applies HDBSCAN (Hierarchical DBSCAN) clustering algorithm to
        localizations for density-based clustering with hierarchical structure.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    min_cluster_size : int
                        Minimum cluster size for HDBSCAN
                    min_samples : int
                        Minimum samples parameter for HDBSCAN
                Optional keys:
                    continue_with_centers : bool
                        Whether to use cluster centers for subsequent analysis
                    save_locs : bool
                        Whether to save clustered localization data
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Optional keys:
                    n_clusters : int
                        Number of clusters identified
                    cluster_centers : numpy.ndarray
                        Coordinates of cluster centers

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with clustering information
        """
        parameters_spec = {
            "min_cluster_size": {
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
            "save_locs": {
                "type": "bool",
                "description": "Whether to save clustered localization data",
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        """Perform clustering using the smlm clusterer.

        Analyzes binding events in single-molecule localization data
        using specialized clustering algorithms.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        Clustering radius in nm
                Optional keys:
                    min_binding_time : float
                        Minimum binding time threshold
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    binding_events : int
                        Number of binding events detected

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with binding event analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "binding_events": {
                "type": "int",
                "description": "Number of binding events detected",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def smlm_clusterer(self):
        """Perform clustering using the smlm clusterer.

        Applies specialized SMLM clustering algorithms for single-molecule
        localization microscopy data analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    method : str
                        Clustering method to use
                Optional keys:
                    radius : float
                        Clustering radius parameter
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    n_clusters : int
                        Number of clusters found

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with clustering statistics
        """
        parameters_spec = {
            "method": {
                "type": "str",
                "description": "Clustering method to use",
                "options": ["voronoi", "dbscan_like", "hierarchical"],
                "required": True,
            },
            "radius": {
                "type": "float",
                "description": "Clustering radius parameter",
                "min": 1.0,
                "max": 1000.0,
                "default": 50.0,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "n_clusters": {
                "type": "int",
                "description": "Number of clusters found",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def gaussian_mixture_cluster(self):
        """Perform clustering using gaussian mixture models.

        Applies Gaussian Mixture Model clustering to localization data
        for probabilistic cluster assignment.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    n_components : int
                        Number of Gaussian components
                Optional keys:
                    covariance_type : str
                        Type of covariance matrix
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    cluster_assignments : numpy.ndarray
                        Cluster assignment probabilities

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with GMM clustering results
        """
        parameters_spec = {
            "n_components": {
                "type": "int",
                "description": "Number of Gaussian components",
                "min": 1,
                "max": 50,
                "default": 3,
                "required": True,
            },
            "covariance_type": {
                "type": "str",
                "description": "Type of covariance matrix",
                "options": ["full", "tied", "diag", "spherical"],
                "default": "full",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "cluster_assignments": {
                "type": "numpy.ndarray",
                "description": "Cluster assignment probabilities",
            },
        }

        return parameters_spec, results_spec

    def nneighbor(self):
        """Calculate Nearest Neighbor distances.

        Computes k-nearest neighbor distances for spatial randomness
        analysis and clustering validation.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    k : int
                        Number of nearest neighbors to calculate
                Optional keys:
                    save_data : bool
                        Whether to save nearest neighbor data
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    nneighbor_distances : numpy.ndarray
                        Nearest neighbor distance matrix

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with nearest neighbor data
        """
        parameters_spec = {
            "k": {
                "type": "int",
                "description": "Number of nearest neighbors to calculate",
                "min": 1,
                "max": 50,
                "default": 5,
                "required": True,
            },
            "save_data": {
                "type": "bool",
                "description": "Whether to save nearest neighbor data",
                "default": True,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "nneighbor_distances": {
                "type": "numpy.ndarray",
                "description": "Nearest neighbor distance matrix",
            },
        }

        return parameters_spec, results_spec

    def fit_csr(self):
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
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
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

        Saves localization data and metadata from single-dataset analysis
        for subsequent use in aggregation workflows.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    filename : str
                        Custom filename for saved data
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    saved_filepath : str
                        Path to saved dataset file

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with save information
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "saved_filepath": {
                "type": "str",
                "description": "Path to saved dataset file",
            },
        }

        return parameters_spec, results_spec

    # Aggregation workflow modules
    def load_datasets_to_aggregate(self):
        """Loads data of multiple single-dataset workflows into one
        aggregation workflow.

        Loads and combines data from multiple single-dataset analyses
        for aggregated statistical analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    file_list : list
                        List of dataset files to load
                Optional keys:
                    tags : list
                        Custom tags for each dataset
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    n_datasets : int
                        Number of datasets loaded

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with aggregation information
        """
        parameters_spec = {
            "file_list": {
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "n_datasets": {
                "type": "int",
                "description": "Number of datasets loaded",
                "min": 0,
            },
        }

        return parameters_spec, results_spec

    def align_channels(self):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward.

        Aligns multiple imaging channels using fiducial markers or
        cross-correlation methods for multi-color analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    method : str
                        Alignment method to use
                Optional keys:
                    reference_channel : int
                        Reference channel index for alignment
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    alignment_matrix : numpy.ndarray
                        Transformation matrix for alignment

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with alignment information
        """
        parameters_spec = {
            "method": {
                "type": "str",
                "description": "Alignment method to use",
                "options": ["fiducial", "cross_correlation", "manual"],
                "required": True,
            },
            "reference_channel": {
                "type": "int",
                "description": "Reference channel index for alignment",
                "min": 0,
                "max": 10,
                "default": 0,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

        Merges data from multiple imaging channels into a single dataset
        for combined analysis (e.g., RESI microscopy).

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    channel_weights : list
                        Relative weights for each channel
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    combined_nlocs : int
                        Total number of localizations in combined dataset

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with combination statistics
        """
        parameters_spec = {
            "channel_weights": {
                "type": "list",
                "description": "Relative weights for each channel",
                "element_type": "float",
                "min": 0.0,
                "max": 10.0,
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        """save data of multiple single-dataset workflows from one
        aggregation workflow.

        Saves aggregated analysis results from multiple datasets
        to individual files for downstream processing.

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
        parameters_spec = {}

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

    def spinna_manual(self):
        """Direct implementation of spinna batch analysis.

        Performs SPINNA (Spatial Point Pattern Analysis) using manual
        parameter specification for batch processing.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radii : list
                        List of analysis radii in nm
                Optional keys:
                    n_simulations : int
                        Number of Monte Carlo simulations
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    spinna_results : dict
                        SPINNA analysis results

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with SPINNA analysis
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
            "n_simulations": {
                "type": "int",
                "description": "Number of Monte Carlo simulations",
                "min": 10,
                "max": 10000,
                "default": 1000,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "spinna_results": {
                "type": "dict",
                "description": "SPINNA analysis results",
            },
        }

        return parameters_spec, results_spec

    def spinna(self):
        """implementation of a single spinna run.

        Performs a single SPINNA (Spatial Point Pattern Analysis) run
        with automated parameter optimization.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    max_radius : float
                        Maximum analysis radius in nm
                Optional keys:
                    optimization_method : str
                        Method for parameter optimization
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    optimal_parameters : dict
                        Optimized SPINNA parameters

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with optimization results
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "optimal_parameters": {
                "type": "dict",
                "description": "Optimized SPINNA parameters",
            },
        }

        return parameters_spec, results_spec

    def ripleysk(self):
        """Ripley's K analysis implementation.

        Performs Ripley's K-function analysis for spatial point pattern
        characterization and clustering assessment.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radii : list
                        List of analysis radii in nm
                Optional keys:
                    edge_correction : str
                        Edge correction method
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    ripley_k_values : numpy.ndarray
                        Calculated Ripley's K values

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with Ripley's K analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "ripley_k_values": {
                "type": "numpy.ndarray",
                "description": "Calculated Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def ripleysk2(self):
        """Alternative Ripley's K analysis implementation.

        Performs Ripley's K-function analysis using alternative algorithms
        or parameters for comparison and validation.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radii : list
                        List of analysis radii in nm
                Optional keys:
                    algorithm : str
                        Algorithm variant to use
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    ripley_k2_values : numpy.ndarray
                        Calculated Ripley's K values (variant 2)

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with alternative Ripley's K analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "ripley_k2_values": {
                "type": "numpy.ndarray",
                "description": "Calculated Ripley's K values (variant 2)",
            },
        }

        return parameters_spec, results_spec

    def ripleysk_average(self):
        """Averages multiple Ripley's K analyses.

        Computes average Ripley's K values across multiple datasets
        or analysis runs for statistical robustness.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    weight_method : str
                        Method for weighting individual analyses
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    average_ripley_k : numpy.ndarray
                        Averaged Ripley's K values

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with averaged Ripley's K analysis
        """
        parameters_spec = {
            "weight_method": {
                "type": "str",
                "description": "Method for weighting individual analyses",
                "options": ["equal", "by_nlocs", "by_area"],
                "default": "equal",
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "average_ripley_k": {
                "type": "numpy.ndarray",
                "description": "Averaged Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def ripleysk_average2(self):
        """Alternative averaging of multiple Ripley's K analyses.

        Computes alternative average Ripley's K values using different
        statistical methods for comparison and validation.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    statistical_method : str
                        Statistical method for averaging
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    alternative_average_k : numpy.ndarray
                        Alternative averaged Ripley's K values

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with alternative averaged analysis
        """
        parameters_spec = {
            "statistical_method": {
                "type": "str",
                "description": "Statistical method for averaging",
                "options": ["mean", "median", "trimmed_mean"],
                "default": "mean",
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "alternative_average_k": {
                "type": "numpy.ndarray",
                "description": "Alternative averaged Ripley's K values",
            },
        }

        return parameters_spec, results_spec

    def protein_interactions(self):
        """Protein interaction analysis.

        Analyzes protein-protein interactions in multi-color SMLM data
        using spatial correlation and clustering methods.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    interaction_radius : float
                        Maximum interaction distance in nm
                Optional keys:
                    confidence_level : float
                        Statistical confidence level
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    interaction_map : numpy.ndarray
                        Spatial interaction map

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with interaction analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "interaction_map": {
                "type": "numpy.ndarray",
                "description": "Spatial interaction map",
            },
        }

        return parameters_spec, results_spec

    def protein_interactions_average(self):
        """Average protein interaction analysis across datasets.

        Computes averaged protein interaction statistics across multiple
        datasets for population-level analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    normalization_method : str
                        Method for normalizing interactions
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    average_interactions : dict
                        Averaged interaction statistics

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with averaged interaction analysis
        """
        parameters_spec = {
            "normalization_method": {
                "type": "str",
                "description": "Method for normalizing interactions",
                "options": ["by_area", "by_density", "none"],
                "default": "by_area",
                "required": False,
            }
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "average_interactions": {
                "type": "dict",
                "description": "Averaged interaction statistics",
            },
        }

        return parameters_spec, results_spec

    def create_mask(self):
        """Create a density mask.

        Creates spatial masks based on localization density for
        region-specific analysis and filtering.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    threshold : float
                        Density threshold for mask creation
                Optional keys:
                    smoothing_radius : float
                        Radius for density smoothing
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    mask : numpy.ndarray
                        Generated density mask

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with mask information
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "mask": {
                "type": "numpy.ndarray",
                "description": "Generated density mask",
            },
        }

        return parameters_spec, results_spec

    def create_mask2(self):
        """Create a density mask.

        Creates spatial masks using alternative algorithms for
        comparison and validation of mask-based analysis.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    algorithm : str
                        Mask creation algorithm
                Optional keys:
                    parameters_dict : dict
                        Algorithm-specific parameters
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    mask2 : numpy.ndarray
                        Alternative generated mask

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with alternative mask
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "mask2": {
                "type": "numpy.ndarray",
                "description": "Alternative generated mask",
            },
        }

        return parameters_spec, results_spec

    def refine_mask_by_density(self):
        """refine a mask by a given density range.

        Refines existing spatial masks using density criteria
        for more precise region definition.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    density_range : tuple
                        Min and max density values
                Optional keys:
                    refinement_method : str
                        Method for mask refinement
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    refined_mask : numpy.ndarray
                        Refined density mask

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with refined mask
        """
        parameters_spec = {
            "density_range": {
                "type": "tuple",
                "description": "Min and max density values",
                "length": 2,
                "element_type": "float",
                "required": True,
            },
            "refinement_method": {
                "type": "str",
                "description": "Method for mask refinement",
                "options": ["erosion", "dilation", "opening", "closing"],
                "default": "opening",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "refined_mask": {
                "type": "numpy.ndarray",
                "description": "Refined density mask",
            },
        }

        return parameters_spec, results_spec

    def dbscan_molint(self):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow.

        Specialized DBSCAN clustering for molecular interaction analysis
        with additional interaction-specific parameters.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    radius : float
                        DBSCAN radius for interaction clustering
                    min_samples : int
                        Minimum samples for interaction clusters
                Optional keys:
                    interaction_threshold : float
                        Distance threshold for interactions
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    interaction_clusters : numpy.ndarray
                        Clustered interaction data

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with interaction clustering
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "interaction_clusters": {
                "type": "numpy.ndarray",
                "description": "Clustered interaction data",
            },
        }

        return parameters_spec, results_spec

    def CSR_sim_in_mask(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask.

        Simulates Complete Spatial Randomness within defined mask regions
        for statistical comparison and validation.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    n_simulations : int
                        Number of CSR simulations
                Optional keys:
                    seed : int
                        Random seed for reproducibility
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    csr_simulations : list
                        Generated CSR simulation data

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with CSR simulation data
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "csr_simulations": {
                "type": "list",
                "description": "Generated CSR simulation data",
                "element_type": "numpy.ndarray",
            },
        }

        return parameters_spec, results_spec

    def find_cluster_motifs(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask.

        Identifies recurring spatial patterns or motifs in cluster data
        for structural characterization.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    motif_size : int
                        Size of motifs to search for
                Optional keys:
                    similarity_threshold : float
                        Threshold for motif similarity
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    identified_motifs : list
                        List of identified cluster motifs

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with motif analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "identified_motifs": {
                "type": "list",
                "description": "List of identified cluster motifs",
                "element_type": "dict",
            },
        }

        return parameters_spec, results_spec

    def interaction_graph(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask.

        Creates interaction graphs showing spatial relationships
        between molecular species or clusters.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    graph_type : str
                        Type of interaction graph
                Optional keys:
                    edge_threshold : float
                        Threshold for graph edges
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    interaction_graph : dict
                        Generated interaction graph data

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with interaction graph
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "interaction_graph": {
                "type": "dict",
                "description": "Generated interaction graph data",
            },
        }

        return parameters_spec, results_spec

    def plot_densities(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask.

        Creates visualization plots of localization densities
        across datasets and conditions.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Optional keys:
                    plot_type : str
                        Type of density plot
                    color_map : str
                        Color map for visualization
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    density_plots : list
                        Generated density plot file paths

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with density visualizations
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        kinetics. The metrics used are number of locs and rms deviation
        from mean frame.

        Identifies gold nanoparticle localizations using blinking kinetics
        analysis for drift correction and alignment purposes.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    nlocs_threshold : int
                        Minimum number of localizations for gold detection
                    rms_threshold : float
                        Maximum RMS deviation threshold
                Optional keys:
                    frame_window : int
                        Frame window for kinetics analysis
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    gold_locs : numpy.ndarray
                        Identified gold bead localizations

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with gold bead identification
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "gold_locs": {
                "type": "numpy.ndarray",
                "description": "Identified gold bead localizations",
            },
        }

        return parameters_spec, results_spec

    def find_similar(self):
        """pick similar in nlocs/rmsd space.

        Identifies structures with similar characteristics in
        localization count and spatial distribution space.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    reference_structure : int
                        Index of reference structure
                    similarity_metric : str
                        Metric for similarity calculation
                Optional keys:
                    tolerance : float
                        Tolerance for similarity matching
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    similar_structures : list
                        List of similar structure indices

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with similarity analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

        Identifies structural patterns in clustered data using
        localization density and spatial distribution metrics.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    cluster_method : str
                        Method for structure clustering
                Optional keys:
                    n_structures : int
                        Number of structure types to identify
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    structure_types : list
                        Identified structure type classifications

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with structure identification
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

        Corrects sample drift using manually picked or automatically
        selected fiducial localizations.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    picked_locs : numpy.ndarray or str
                        Picked localization coordinates or file path
                Optional keys:
                    interpolation_method : str
                        Method for drift interpolation
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    drift_corrected : numpy.ndarray
                        Drift correction vectors

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with drift correction data
        """
        parameters_spec = {
            "picked_locs": {
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "drift_corrected": {
                "type": "numpy.ndarray",
                "description": "Drift correction vectors",
            },
        }

        return parameters_spec, results_spec

    def filter_locs(self):
        """Filter localizations to lie within a min-max range of a metric.

        Filters localization data based on specified metrics and
        value ranges for quality control and analysis refinement.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    metric : str
                        Metric to filter by
                    min_value : float
                        Minimum allowed value
                    max_value : float
                        Maximum allowed value
                Optional keys:
                    invert_filter : bool
                        Whether to invert the filter logic
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    n_filtered : int
                        Number of localizations after filtering

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with filtering statistics
        """
        parameters_spec = {
            "metric": {
                "type": "str",
                "description": "Metric to filter by",
                "options": [
                    "photons",
                    "lpx",
                    "lpy",
                    "lpx_err",
                    "lpy_err",
                    "bg",
                ],
                "required": True,
            },
            "min_value": {
                "type": "float",
                "description": "Minimum allowed value",
                "required": True,
            },
            "max_value": {
                "type": "float",
                "description": "Maximum allowed value",
                "required": True,
            },
            "invert_filter": {
                "type": "bool",
                "description": "Whether to invert the filter logic",
                "default": False,
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        should not be at extreme positions.

        Filters molecular binding events based on temporal characteristics
        to identify transient binding behaviors.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    frame_percentiles : tuple
                        Min and max frame percentiles for filtering
                Optional keys:
                    binding_duration_range : tuple
                        Min and max binding duration range
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    transient_events : numpy.ndarray
                        Filtered transient binding events

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with transient binding analysis
        """
        parameters_spec = {
            "frame_percentiles": {
                "type": "tuple",
                "description": "Min and max frame percentiles for filtering",
                "length": 2,
                "element_type": "float",
                "min": 0.0,
                "max": 100.0,
                "default": [10.0, 90.0],
                "required": True,
            },
            "binding_duration_range": {
                "type": "tuple",
                "description": "Min and max binding duration range",
                "length": 2,
                "element_type": "float",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "transient_events": {
                "type": "numpy.ndarray",
                "description": "Filtered transient binding events",
            },
        }

        return parameters_spec, results_spec

    def link_locs(self):
        """Link localizations.

        Links localizations across frames to track molecular trajectories
        and binding kinetics over time.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    max_distance : float
                        Maximum linking distance in nm
                    max_frame_gap : int
                        Maximum frame gap for linking
                Optional keys:
                    linking_algorithm : str
                        Algorithm for localization linking
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    linked_trajectories : list
                        Linked localization trajectories

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with linking analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        channel_locs.

        Executes a specified module for all pairwise combinations of
        channels in multi-channel datasets.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    sub_module : str
                        Name of the sub-module to execute
                Optional keys:
                    sub_module_params : dict
                        Parameters for the sub-module
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    pairwise_results : dict
                        Results from pairwise module execution

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with pairwise analysis
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
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
            },
            "pairwise_results": {
                "type": "dict",
                "description": "Results from pairwise module execution",
            },
        }

        return parameters_spec, results_spec

    def random_val(self):
        """For debugging and testing the pairwise module.

        Generate random values and plot for debugging and testing the pairwise
        module.
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

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

        Analyzes labeling efficiency in DNA-PAINT experiments using
        statistical models and binding event counting.

        Args:
            i : int
                The index of the module in the workflow
            parameters : dict
                Required keys:
                    target_species : str
                        Name of target molecular species
                    reference_species : str
                        Name of reference species
                Optional keys:
                    efficiency_model : str
                        Model for efficiency calculation
            results : dict
                Required keys:
                    start time : str
                        Module execution start timestamp
                    duration : float
                        Module execution duration in seconds
                Results updated with:
                    labeling_efficiency : float
                        Calculated labeling efficiency

        Returns:
            parameters : dict
                Input parameters (unchanged)
            results : dict
                Input results with labeling efficiency analysis
        """
        parameters_spec = {
            "target_species": {
                "type": "str",
                "description": "Name of target molecular species",
                "required": True,
            },
            "reference_species": {
                "type": "str",
                "description": "Name of reference species",
                "required": True,
            },
            "efficiency_model": {
                "type": "str",
                "description": "Model for efficiency calculation",
                "options": ["poisson", "binomial", "maximum_likelihood"],
                "default": "maximum_likelihood",
                "required": False,
            },
        }

        results_spec = {
            "start time": {
                "type": "str",
                "description": "Module execution start timestamp",
            },
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        """For debugging and testing the pairwise module.

        Generate random values and plot for debugging and testing the pairwise
        module.
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

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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
        """For debugging and testing the pairwise module.

        Generate random values and plot for debugging and testing the pairwise
        module.
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

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

    def resolution_frc_spatial(self):
        """For debugging and testing the pairwise module.

        Generate random values and plot for debugging and testing the pairwise
        module.
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

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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

    def undrift_rsso(self):
        """For debugging and testing the pairwise module.

        Generate random values and plot for debugging and testing the pairwise
        module.
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

        Returns:
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
            "duration": {
                "type": "float",
                "description": "Module execution duration in seconds",
                "min": 0.0,
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


class SlurmCommunicator:
    """Communication interface for SLURM job scheduling and SSH command
    execution.

    This class provides methods to interact with SLURM job schedulers through
    SSH, create SLURM job scripts, submit jobs, and monitor their status. It's
    designed to work with remote compute clusters running SLURM workload
    manager.

    Attributes:
        hostname (str): SSH hostname or IP address
        username (str): SSH username for authentication
        port (int): SSH port number (default: 22)
        ssh_key_path (str): Path to SSH private key file
        timeout (int): SSH connection timeout in seconds
    """

    def __init__(
        self, hostname, username, port=22, ssh_key_path=None, timeout=30
    ):
        """Initialize SLURM communicator with SSH connection parameters.

        Args:
            hostname (str): SSH hostname or IP address
            username (str): SSH username for authentication
            port (int, optional): SSH port number. Defaults to 22.
            ssh_key_path (str, optional):
                Path to SSH private key file.
                If None, uses default SSH authentication.
            timeout (int, optional):
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

        Args:
            command (str): Shell command to execute
            working_directory (str, optional): Directory to execute command in

        Returns:
            dict: Dictionary containing:
                - stdout (str): Standard output
                - stderr (str): Standard error
                - return_code (int): Command exit code
                - success (bool): True if return_code == 0

        Raises:
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

        Returns:
            bool: True if connection successful, False otherwise
        """
        result = self.execute_ssh_command('echo "Connection test successful"')
        return result["success"]

    def assemble_slurm_commands(
        self, scriptname="start_workflow.py", use_pw_module=True
    ):
        """Assembles picasso-workflow specific commands for running a batch
        job on a SLURM cluster.
        """
        commands = []
        if use_pw_module:
            commands.append("module load picasso-workflow2")
        else:
            commands.append("module load anaconda/3/2023.03")

        commands.append("source ~/.bashrc")

        if not use_pw_module:
            commands.append("conda activate picasso-workflow")

        commands.append(f"srun {scriptname}")

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

        Args:
            job_name (str): Name for the SLURM job
            commands (list or str):
                List of shell commands or single command string
            slurm_options (dict, optional):
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
            output_file (str, optional): Path for stdout redirection
            error_file (str, optional): Path for stderr redirection
            working_directory (str, optional): Working directory for the job

        Returns:
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

        Args:
            script_content (str): Complete SLURM script content
            folder (str): Path where to save the script on remote host
            local (bool): whether the folder is available on the local system

        Returns:
            dict: Result of the write operation (see execute_ssh_command)
        """
        if local:
            filepath = os.path.join(folder, "run_workflow_slurm.sh")
            with open(filepath, "w") as f:
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
                chmod_result = self.execute_ssh_command(
                    f"chmod +x {folder}"
                )
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

        Args:
            script_path (str): Path to the SLURM script on remote host
            additional_options (list, optional): Additional sbatch options

        Returns:
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

        Args:
            job_id (int): SLURM job ID

        Returns:
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

        Args:
            job_id (int): SLURM job ID to cancel

        Returns:
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

        Args:
            user (str, optional): Username to list jobs for.
                                If None, uses the SSH username.

        Returns:
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

        Returns:
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
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)  # Prevent focus issues

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
                    self.lineEdit.setText(path)
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
        value = index.model().data(index, QtCore.Qt.EditRole)
        if value:
            editor.setText(str(value))

    def setModelData(self, editor, model, index):
        # Check if editor is still valid
        try:
            text = editor.text()
            model.setData(index, text, QtCore.Qt.EditRole)
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
        if event.type() == QtCore.QEvent.FocusOut:
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
        if event.type() == QEvent.ToolTip:
            tooltip = index.data(Qt.ToolTipRole)
            if tooltip:
                QtWidgets.QToolTip.showText(event.globalPos(), tooltip)
                return True
        return super().helpEvent(event, view, option, index)


class ParameterWidgetInfo:
    """Container for parameter widget information."""

    def __init__(self, widget, cmd_button, row_widget, metadata, original_type, sub_parameters=None, toggle_function=None):
        """Initialize parameter widget info.

        Args:
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
        self.sub_parameters = sub_parameters or {}  # For nested dict parameters
        self.toggle_function = toggle_function  # For dict parameters with checkboxes


class ParameterCmdDialog(QtWidgets.QDialog):
    """Dialog for selecting a command as parameter value."""

    def __init__(self, workflow_modules, module_descriptor, parent=None):
        """Initialize the prior result dialog.

        Args:
            workflow_modules: List of tuples (module_name, param_dict) from workflow
            module_descriptor: ModuleDescriptor instance to get result specs
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Select Command")
        self.setModal(True)
        self.workflow_modules = workflow_modules
        self.module_descriptor = module_descriptor

        layout = QtWidgets.QVBoxLayout(self)

        # Timing selection
        layout.addWidget(QtWidgets.QLabel("Collection timing:"))
        self.timing_group = QtWidgets.QButtonGroup(self)

        self.timing_before_radio = QtWidgets.QRadioButton("Collect directly before module execution")
        self.timing_start_radio = QtWidgets.QRadioButton("Collect at start of workflow stage")
        self.timing_before_radio.setChecked(True)  # Default

        self.timing_group.addButton(self.timing_before_radio, 0)
        self.timing_group.addButton(self.timing_start_radio, 1)

        layout.addWidget(self.timing_before_radio)
        layout.addWidget(self.timing_start_radio)
        layout.addSpacing(10)

        # Command type selection
        layout.addWidget(QtWidgets.QLabel("Command type:"))
        self.command_combo = QtWidgets.QComboBox()
        self.command_combo.addItems(["Previous Result", "sum", "max", "min"])
        layout.addWidget(self.command_combo)
        layout.addSpacing(10)

        # Module selection
        layout.addWidget(QtWidgets.QLabel("Select module:"))
        self.module_combo = QtWidgets.QComboBox()
        for i, (module_name, params) in enumerate(workflow_modules):
            self.module_combo.addItem(f"{i}: {module_name}")
        self.module_combo.currentIndexChanged.connect(self._on_module_selected)
        layout.addWidget(self.module_combo)

        # Result selection (combo box instead of text input)
        layout.addWidget(QtWidgets.QLabel("Select result:"))
        self.result_combo = QtWidgets.QComboBox()
        self.result_combo.setPlaceholderText("Select a result")
        layout.addWidget(self.result_combo)

        # Populate results for initially selected module
        self._on_module_selected(0)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_module_selected(self, index):
        """Populate result combo box when module is selected.

        Args:
            index: Index of selected module in workflow
        """
        if index < 0 or index >= len(self.workflow_modules):
            return

        module_name = self.workflow_modules[index][0]

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

    def get_selection(self):
        """Get the selected module index, result name, command type, and timing.

        Returns:
            tuple: (module_index: int, result_name: str, command_type: str, timing: str)
                   command_type is "Previous Result", "sum", "max", or "min"
                   timing is either "before" or "start"
        """
        module_index = self.module_combo.currentIndex()
        result_name = self.result_combo.currentText()
        command_type = self.command_combo.currentText()
        timing = "before" if self.timing_before_radio.isChecked() else "start"
        return module_index, result_name, command_type, timing


class Window(QtWidgets.QMainWindow):
    """Main window for the picasso-workflow GUI application."""

    def __init__(self):
        super().__init__()
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
        self.editing_workflow_index = -1  # -1 means not editing an existing item
        self.editing_workflow_tab = -1  # 0 = single, 1 = aggregation

        layout = QtWidgets.QGridLayout()
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        # Results folder selection
        results_folder_button = QtWidgets.QPushButton("Results Folder")
        # self.files_box.addWidget(results_folder_button, 2, 0)
        layout.addWidget(results_folder_button, 0, 0)
        results_folder_button.clicked.connect(self.select_results_folder)
        self.results_folder_display = QtWidgets.QLineEdit()
        self.results_folder_display.setReadOnly(True)
        self.results_folder_display.setPlaceholderText("No folder selected")
        # self.files_box.addWidget(self.results_folder_display, 2, 1, 1, 2)
        layout.addWidget(self.results_folder_display, 0, 1, 1, 2)
        # Investigation type
        self.workflow_type = QtWidgets.QComboBox()
        self.workflow_type.addItem("Single Workflow")
        self.workflow_type.addItem("Aggregation Workflow")
        self.workflow_type.addItem("Investigation Workflow")
        # disable Investigation workflow, which is in Development
        index = self.workflow_type.model().index(2, 0)
        self.workflow_type.model().setData(index, 0, Qt.UserRole - 1)  # 0 disables the item
        self.workflow_type.setItemDelegate(ToolTipDelegate(self.workflow_type))
        self.workflow_type.model().setData(index, "Not Implemented yet", Qt.ToolTipRole)  # Tooltip

        self.workflow_type.currentIndexChanged.connect(self._on_workflow_type_changed)
        layout.addWidget(self.workflow_type, 0, 3)

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1, 0, 1, 4)

        # Config tab
        config_tab = QtWidgets.QWidget()
        config_layout = QtWidgets.QGridLayout(config_tab)
        self.tabs.addTab(config_tab, "Config")

        # Files and modules boxes in config tab
        self._files_box = QtWidgets.QGroupBox("Files")
        self.files_box = QtWidgets.QGridLayout(self._files_box)
        config_layout.addWidget(self._files_box, 0, 0, 1, 2)
        self._modules_box = QtWidgets.QGroupBox("Modules")
        self.modules_box = QtWidgets.QGridLayout(self._modules_box)
        config_layout.addWidget(self._modules_box, 0, 2, 1, 2)

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
        slurm_buttons = QtWidgets.QHBoxLayout()
        self.slurm_buttons_widget = QtWidgets.QWidget()
        self.slurm_buttons_widget.setLayout(slurm_buttons)
        run_on_cluster_layout.addWidget(self.slurm_buttons_widget)
        start_slurm_button = QtWidgets.QPushButton("Start Workflow on Cluster")
        slurm_buttons.addWidget(start_slurm_button)
        start_slurm_button.clicked.connect(self.start_slurm)

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

        queue_info_button = QtWidgets.QPushButton("Show Queue Info")
        queue_info_button.clicked.connect(self.on_show_queue_info)
        job_management_buttons.addWidget(queue_info_button)

        # Job ID input field
        job_id_layout = QtWidgets.QHBoxLayout()
        job_id_widget = QtWidgets.QWidget()
        job_id_widget.setLayout(job_id_layout)
        run_on_cluster_layout.addWidget(job_id_widget)

        job_id_label = QtWidgets.QLabel("Current Job ID:")
        job_id_layout.addWidget(job_id_label)

        self.job_id_input = QtWidgets.QLineEdit()
        self.job_id_input.setPlaceholderText("Enter job ID or auto-filled from submission")
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

        # Results tab
        results_tab = QtWidgets.QWidget()
        results_layout = QtWidgets.QVBoxLayout(results_tab)
        self.tabs.addTab(results_tab, "Results")
        self.tabs.setTabEnabled(2, False)  # in development
        self.tabs.setTabToolTip(2, "Not Implemented yet")

        # select files to process
        self.add_files_button = QtWidgets.QPushButton("Add files")
        self.files_box.addWidget(self.add_files_button, 0, 0)
        self.add_files_button.clicked.connect(self.add_files)
        self.remove_files_button = QtWidgets.QPushButton("Remove selected")
        self.files_box.addWidget(self.remove_files_button, 0, 1)
        self.remove_files_button.clicked.connect(self.remove_selected_files)
        self.clear_files_button = QtWidgets.QPushButton("Clear list")
        self.files_box.addWidget(self.clear_files_button, 0, 2)
        self.clear_files_button.clicked.connect(self.clear_file_list)

        d = {"Name1": "/path/to/file1.txt", "Name2": "/path/to/file2.txt"}
        self.files_table = QtWidgets.QTableWidget()
        self.files_table.setColumnCount(2)
        self.files_table.setHorizontalHeaderLabels(["Name", "File Path"])
        # Configure column stretching - Name column resizes to contents, File Path column stretches
        header = self.files_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        dict_to_table(d, self.files_table)
        # Store delegate as instance variable to prevent garbage collection
        self.file_path_delegate = FilePathDelegate(self)
        self.files_table.setItemDelegateForColumn(1, self.file_path_delegate)
        self.files_box.addWidget(self.files_table, 1, 0, 1, 3)

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
            QtCore.Qt.ScrollBarAlwaysOff
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
        investigation_workflow_layout = QtWidgets.QVBoxLayout(
            investigation_workflow_tab
        )
        self.workflow_tabs.addTab(
            investigation_workflow_tab, "Investigation"
        )

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


        # resize the widgets
        # Set fixed size for the group box
        self.current_module.setMinimumSize(500, 300)

        # Initially disable file and module widgets until results folder is selected
        self._set_widgets_enabled(False)

    def _set_widgets_enabled(self, enabled):
        """Enable or disable file and module widgets based on results folder selection."""
        self.workflow_type.setEnabled(enabled)
        # Files box widgets
        self.add_files_button.setEnabled(enabled)
        self.remove_files_button.setEnabled(enabled)
        self.clear_files_button.setEnabled(enabled)
        self.files_table.setEnabled(enabled)

        # Modules box widgets
        self.current_module.setEnabled(enabled)
        self.workflow_tabs.setEnabled(enabled)
        self.workflow_buttons_widget.setEnabled(enabled)

        # runing config
        self.run_tabs.setEnabled(enabled)

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

    def clear_file_list(self):
        """Clear all rows from the table."""
        if self.files_table.rowCount() == 0:
            return
        self.files_table.setRowCount(0)

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
            self.results_folder_display.setText(folder)
            # Enable widgets when a folder is selected
            self._set_widgets_enabled(True)

            # Search for YAML files and load file list
            self._load_yaml_file_list(folder)

            # Search for workflow definition and load it
            self._load_workflow_definition(folder)
        else:
            # If dialog was cancelled and no folder is selected, disable widgets
            if not self.results_folder_display.text():
                self._set_widgets_enabled(False)

    def _load_yaml_file_list(self, folder):
        """Search for YAML files in folder and load file list if found.

        Args:
            folder: Path to the folder to search
        """
        # Search for specific YAML files
        yaml_files = ["src_loc.yaml", "raw_locs_list.yaml"]

        for yaml_file in yaml_files:
            yaml_path = os.path.join(folder, yaml_file)
            if os.path.exists(yaml_path):
                try:
                    # Load YAML file
                    with open(yaml_path, 'r') as f:
                        file_dict = yaml.safe_load(f)

                    # Validate that it's a dictionary
                    if isinstance(file_dict, dict):
                        # Clear existing file list
                        self.files_table.setRowCount(0)

                        # Populate table with YAML content
                        # Key -> Name, Value -> File Path
                        dict_to_table(file_dict, self.files_table)

                        logger.info(f"Loaded file list from {yaml_file}")
                        return  # Stop after loading first matching file
                    else:
                        logger.warning(f"{yaml_file} does not contain a dictionary")

                except Exception as e:
                    logger.error(f"Error loading {yaml_file}: {e}")
                    QtWidgets.QMessageBox.warning(
                        self,
                        "YAML Load Error",
                        f"Failed to load {yaml_file}:\n{str(e)}"
                    )

        # No YAML files found - this is normal, no action needed
        logger.debug("No src_loc.yaml or raw_locs_list.yaml found in folder")

    def _load_workflow_definition(self, folder):
        """Search for workflow definition file and load workflow modules.

        Args:
            folder: Path to the folder to search
        """
        workflow_file = os.path.join(folder, "start_workflow.py")

        if not os.path.exists(workflow_file):
            logger.debug("No start_workflow.py found in folder")
            return

        try:
            # Dynamically load the Python file
            spec = importlib.util.spec_from_file_location("start_workflow", workflow_file)
            if spec is None or spec.loader is None:
                logger.warning("Could not load start_workflow.py")
                return

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract workflow definitions
            workflow_modules_sgl = getattr(module, 'workflow_modules_sgl', None)
            workflow_modules_agg = getattr(module, 'workflow_modules_agg', None)

            # Load single dataset workflow if present
            if workflow_modules_sgl is not None and isinstance(workflow_modules_sgl, list):
                self._populate_workflow_from_definition(
                    workflow_modules_sgl,
                    self.single_workflow_modules,
                    self.single_workflow_list,
                    "Single Dataset"
                )
                logger.info(f"Loaded {len(workflow_modules_sgl)} modules to Single Dataset workflow")

            # Load aggregation workflow if present
            if workflow_modules_agg is not None and isinstance(workflow_modules_agg, list):
                self._populate_workflow_from_definition(
                    workflow_modules_agg,
                    self.aggregation_workflow_modules,
                    self.aggregation_workflow_list,
                    "Aggregation"
                )
                logger.info(f"Loaded {len(workflow_modules_agg)} modules to Aggregation workflow")

        except Exception as e:
            logger.error(f"Error loading start_workflow.py: {e}")
            QtWidgets.QMessageBox.warning(
                self,
                "Workflow Load Error",
                f"Failed to load workflow from start_workflow.py:\n{str(e)}"
            )

    def _populate_workflow_from_definition(self, workflow_def, workflow_list, list_widget, workflow_name):
        """Populate workflow from loaded definition.

        Args:
            workflow_def: List of (module_name, params_dict) tuples
            workflow_list: Target workflow list (single_workflow_modules or aggregation_workflow_modules)
            list_widget: Target QListWidget for display
            workflow_name: Name of the workflow (for logging)
        """
        # Clear existing workflow
        workflow_list.clear()
        list_widget.clear()

        for module_name, params in workflow_def:
            # Convert parameters to GUI format: {param: (value, command)}
            converted_params = {}

            for param_name, param_value in params.items():
                # Convert parameter value to GUI format
                value_str, command_str = self._convert_param_to_gui_format(param_value)
                converted_params[param_name] = value_str  # (value_str, command_str)
            # print(converted_params)

            # Add to workflow
            workflow_list.append((module_name, converted_params))
            index = len(workflow_list) - 1
            list_widget.addItem(f"{index}: {module_name}")

    def _convert_param_to_gui_format(self, param_value):
        """Convert parameter value from workflow definition to GUI format.

        Args:
            param_value: Parameter value from workflow definition

        Returns:
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
                nested_value, nested_command = self._convert_param_to_gui_format(nested_param_value)
                nested_converted[nested_param_name] = nested_value  # (nested_value, nested_command)
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
            value = self._get_widget_value(widget_info.widget, widget_info.original_type, widget_info)
            # Skip None values (from unchecked optional dicts)
            if value is not None:
                param_values[param_name] = value

        # Add to the appropriate workflow list based on selected tab
        # Store as tuple: (module_name, {param: value})
        current_tab_index = self.workflow_tabs.currentIndex()
        if current_tab_index == 0:  # Single Dataset Workflow
            self.single_workflow_modules.append((module_name, param_values))
            index = len(self.single_workflow_modules) - 1
            self.single_workflow_list.addItem(f"{index}: {module_name}")
        elif current_tab_index == 1:  # Aggregation Workflow
            self.aggregation_workflow_modules.append(
                (module_name, param_values)
            )
            index = len(self.aggregation_workflow_modules) - 1
            self.aggregation_workflow_list.addItem(f"{index}: {module_name}")

    def _renumber_workflow_items(self, list_widget, modules):
        """Update QListWidget items with correct numbering after reordering."""
        for i in range(len(modules)):
            module_name = modules[i][
                0
            ]  # Extract name from (name, params) tuple
            list_widget.item(i).setText(f"{i}: {module_name}")

    def _on_workflow_selection_changed(self, current_row):
        """Handle selection change in workflow list - display module in Current Module section."""
        if current_row < 0:
            # Clear editing state when nothing is selected
            self.editing_workflow_index = -1
            self.editing_workflow_tab = -1
            return

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

        Args:
            param_values: Dict of {param_name: value}
        """
        for param_name, widget_info in self.parameter_widgets.items():
            if param_name in param_values:
                value_data = param_values[param_name]

                # Set value in widget
                self._set_widget_value(
                    widget_info.widget,
                    value_data,
                    widget_info.original_type,
                    widget_info
                )

    def _on_workflow_tab_changed(self, tab_index):
        """Handle workflow tab change - display selected module if any."""
        if tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row >= 0 and current_row < len(
                self.single_workflow_modules
            ):
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

        Args:
            type_index: Index of selected workflow type
                       0 = Single Workflow
                       1 = Aggregation Workflow
                       2 = Investigation Workflow
        """
        # Tab indices:
        # 0 = Single Dataset Workflow
        # 1 = Aggregation Workflow
        # 2 = Investigation

        if type_index == 0:  # Single Workflow
            # Enable only Single Dataset Workflow tab
            self.workflow_tabs.setTabEnabled(0, True)   # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, False)  # Aggregation: disabled
            self.workflow_tabs.setTabEnabled(2, False)  # Investigation: disabled
            # Switch to Single Dataset tab if currently on a disabled tab
            if self.workflow_tabs.currentIndex() != 0:
                self.workflow_tabs.setCurrentIndex(0)

        elif type_index == 1:  # Aggregation Workflow
            # Enable Single Dataset and Aggregation, disable Investigation
            self.workflow_tabs.setTabEnabled(0, True)   # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, True)   # Aggregation: enabled
            self.workflow_tabs.setTabEnabled(2, False)  # Investigation: disabled
            # Switch to Aggregation tab if currently on Investigation
            if self.workflow_tabs.currentIndex() == 2:
                self.workflow_tabs.setCurrentIndex(1)

        elif type_index == 2:  # Investigation Workflow
            # Enable all tabs
            self.workflow_tabs.setTabEnabled(0, True)   # Single Dataset: enabled
            self.workflow_tabs.setTabEnabled(1, True)   # Aggregation: enabled
            self.workflow_tabs.setTabEnabled(2, True)   # Investigation: enabled

    def remove_selected(self):
        """Remove the selected module from the workflow."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row >= 0:
                self.single_workflow_list.takeItem(current_row)
                del self.single_workflow_modules[current_row]
                self._renumber_workflow_items(
                    self.single_workflow_list, self.single_workflow_modules
                )
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            if current_row >= 0:
                self.aggregation_workflow_list.takeItem(current_row)
                del self.aggregation_workflow_modules[current_row]
                self._renumber_workflow_items(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                )

    def move_up(self):
        """Move the selected module up in the workflow order."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            if current_row > 0:  # Can't move first item up
                # Swap in list
                (
                    self.single_workflow_modules[current_row],
                    self.single_workflow_modules[current_row - 1],
                ) = (
                    self.single_workflow_modules[current_row - 1],
                    self.single_workflow_modules[current_row],
                )
                # Update display and maintain selection
                self._renumber_workflow_items(
                    self.single_workflow_list, self.single_workflow_modules
                )
                self.single_workflow_list.setCurrentRow(current_row - 1)
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            if current_row > 0:  # Can't move first item up
                # Swap in list
                (
                    self.aggregation_workflow_modules[current_row],
                    self.aggregation_workflow_modules[current_row - 1],
                ) = (
                    self.aggregation_workflow_modules[current_row - 1],
                    self.aggregation_workflow_modules[current_row],
                )
                # Update display and maintain selection
                self._renumber_workflow_items(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                )
                self.aggregation_workflow_list.setCurrentRow(current_row - 1)

    def move_down(self):
        """Move the selected module down in the workflow order."""
        current_tab_index = self.workflow_tabs.currentIndex()

        if current_tab_index == 0:  # Single Dataset Workflow
            current_row = self.single_workflow_list.currentRow()
            max_row = len(self.single_workflow_modules) - 1
            if 0 <= current_row < max_row:  # Can't move last item down
                # Swap in list
                (
                    self.single_workflow_modules[current_row],
                    self.single_workflow_modules[current_row + 1],
                ) = (
                    self.single_workflow_modules[current_row + 1],
                    self.single_workflow_modules[current_row],
                )
                # Update display and maintain selection
                self._renumber_workflow_items(
                    self.single_workflow_list, self.single_workflow_modules
                )
                self.single_workflow_list.setCurrentRow(current_row + 1)
        elif current_tab_index == 1:  # Aggregation Workflow
            current_row = self.aggregation_workflow_list.currentRow()
            max_row = len(self.aggregation_workflow_modules) - 1
            if 0 <= current_row < max_row:  # Can't move last item down
                # Swap in list
                (
                    self.aggregation_workflow_modules[current_row],
                    self.aggregation_workflow_modules[current_row + 1],
                ) = (
                    self.aggregation_workflow_modules[current_row + 1],
                    self.aggregation_workflow_modules[current_row],
                )
                # Update display and maintain selection
                self._renumber_workflow_items(
                    self.aggregation_workflow_list,
                    self.aggregation_workflow_modules,
                )
                self.aggregation_workflow_list.setCurrentRow(current_row + 1)

    def create_python_script(self, filename="start_workflow.py"):
        """Generate a Python workflow script from current GUI settings.

        Args:
            filename: Name of the output script file
        """
        import json
        from datetime import datetime

        # Get workflow type
        workflow_type_index = self.workflow_type.currentIndex()
        workflow_type_names = [
            "Single Workflow",
            "Aggregation Workflow",
            "Investigation Workflow"]
        workflow_type_name = workflow_type_names[workflow_type_index] if workflow_type_index < len(workflow_type_names) else "Unknown"

        # Build datasets dict from files table
        datasets = {}
        for row in range(self.files_table.rowCount()):
            name_item = self.files_table.item(row, 0)
            path_item = self.files_table.item(row, 1)
            if name_item and path_item:
                name = name_item.text()
                path = path_item.text()

                # Create lists for values
                if name not in datasets:
                    datasets[name] = []
                datasets[name].append(path)

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
                if "/" in value or "\\" in value or value.endswith(('.yaml', '.hdf5', '.h5', '.tif', '.png', '.jpg')):
                    # Use os.path.join for path-like strings
                    parts = value.replace("\\", "/").split("/")
                    if len(parts) > 1:
                        return f"os.path.join({', '.join(repr(p) for p in parts)})"

                return repr(value)
            elif isinstance(value, dict):
                # Format nested dicts recursively
                items = [f"{repr(k)}: {format_value(v)}" for k, v in value.items()]
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
                    lines.append(f'            "{param_name}": {formatted_value},')

                lines.append("        },")
                lines.append("    ),")
            lines.append("]")
            return "\n".join(lines)

        # Generate script content
        script_lines = [
            "#!/usr/bin/env python",
            '"""',
            f"Script Name: {filename}",
            f"Generated by: picasso-workflow GUI",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Workflow type: {workflow_type_name}",
            '"""',
            "import os",
            "from picasso import io",
        ]

        # Add appropriate import based on workflow type
        if workflow_type_index == 0:  # Single Workflow
            script_lines.append("from picasso_workflow.workflow import WorkflowRunner")
        elif workflow_type_index == 1:  # Aggregation Workflow
            script_lines.append("from picasso_workflow.metaworkflow import AggregationWorkflowCoordinator")
        elif workflow_type_index == 2:  # Investigation Workflow
            script_lines.append("from picasso_workflow.metaworkflow import InvestigationWorkflowCoordinator")

        script_lines.extend([
            "",
            "",
            "# Confluence configuration (set via environment variables)",
            "confluence_url = os.getenv('CONFLUENCE_URL')",
            "confluence_token = os.getenv('CONFLUENCE_BEARER')",
            "confluence_space = os.getenv('CONFLUENCE_SPACE')",
            "base_page = os.getenv('CONFLUENCE_BASE_PAGE')",
            "",
            "",
            "# Dataset configuration",
            "datasets = {",
        ])

        # Add datasets
        for key, values in datasets.items():
            script_lines.append(f"    {repr(key)}: [")
            for value in values:
                formatted = format_value(value)
                script_lines.append(f"        {formatted},")
            script_lines.append("    ],")
        script_lines.append("}")

        script_lines.extend([
            "",
            "",
            "# Single dataset workflow modules",
        ])
        script_lines.append("workflow_modules_sgl = " + format_modules(self.single_workflow_modules))

        script_lines.extend([
            "",
            "",
            "# Aggregation workflow modules",
        ])
        script_lines.append("workflow_modules_agg = " + format_modules(self.aggregation_workflow_modules))

        script_lines.extend([
            "",
            "",
            'if __name__ == "__main__":',
            "    # Get working directory",
            "    working_folder = os.path.dirname(os.path.abspath(__file__))",
            "    src_loc_file = os.path.join(working_folder, 'src_loc.yaml')",
            "    io.save_info(src_loc_file, [datasets])",
            "",
            "    print('datasets', datasets)",
            "    print('src_loc', src_loc_file)",
            "    analysis_name = os.path.split(working_folder)[-1]",
            "",
        ])

        # Add coordinator creation based on workflow type
        if workflow_type_index == 0:  # Single Workflow
            script_lines.extend([
                "    # Create single workflow runner",
                "    runner = WorkflowRunner(",
                "        working_folder=working_folder,",
                "        analysis_name=analysis_name,",
                "        confluence_url=confluence_url,",
                "        confluence_space=confluence_space,",
                "        confluence_token=confluence_token,",
                "        base_page=base_page,",
                "    )",
                "",
                "    # Run workflow",
                "    runner.run_workflow(workflow_modules_sgl)",
            ])
        elif workflow_type_index == 1:  # Aggregation Workflow
            script_lines.extend([
                "    # Create aggregation workflow coordinator",
                "    coordinator = AggregationWorkflowCoordinator(",
                "        src_loc_file, analysis_name, working_folder,",
                "        confluence_url, confluence_space, confluence_token,",
                "        base_page,",
                "        always_save=False",
                "    )",
                "",
                "    # Run analysis",
                "    coordinator.run_analysis(workflow_modules_sgl, workflow_modules_agg)",
            ])
        elif workflow_type_index == 2:  # Investigation Workflow
            script_lines.extend([
                "    # Create investigation workflow coordinator",
                "    coordinator = InvestigationWorkflowCoordinator(",
                "        src_loc_file, analysis_name, working_folder,",
                "        confluence_url, confluence_space, confluence_token,",
                "        base_page,",
                "        always_save=False",
                "    )",
                "",
                "    # Run investigation",
                "    coordinator.run_investigation(workflow_modules_sgl, workflow_modules_agg)",
            ])

        script_lines.append("")  # Final newline

        # Write script to file
        script_content = "\n".join(script_lines)

        # Get output path from results folder
        results_folder = self.results_folder_display.text()
        if results_folder:
            output_path = os.path.join(results_folder, filename)
        else:
            output_path = filename

        with open(output_path, "w") as f:
            f.write(script_content)

        # Make script executable on Unix systems
        import stat
        try:
            st = os.stat(output_path)
            os.chmod(output_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except:
            pass  # Windows doesn't support chmod

        # print(f"Created workflow script: {output_path}")
        return output_path

    def start_slurm(self):
        """"""
        import getpass
        from picasso_workflow.metaworkflow import PathParser
        pp = PathParser()

        hostname = "hpcl8001"
        host_cluster = "hpcl8"
        username = getpass.getuser()
        ssh_key_path = '~/.ssh/id_rsa'
        self.slurm_communicator = SlurmCommunicator(
            hostname, username, port=22, ssh_key_path=ssh_key_path)

        assert self.slurm_communicator.test_connection()

        scriptname = "start_workflow.py"
        self.create_python_script(scriptname)

        job_name = "mypwjob"
        slurm_options = {
            "nodes": 2,
            # "ntasks": Number of tasks,
            "cpus-per-task": 12,
            "mem": "50G",
            "time": "24:00:00",
            # "mail-type": "ALL",
            # "mail-user": f"{username}@biochem.mpg.de",
        }

        results_folder_local = self.results_folder_display.text()
        results_folder_host = pp.convert_path(results_folder_local, host_cluster)

        commands = self.slurm_communicator.assemble_slurm_commands(
            scriptname=scriptname, use_pw_module=True)
        script_content = self.slurm_communicator.create_slurm_script(
            job_name, commands, slurm_options=slurm_options,
            output_file=f"{results_folder_host}/logs/%A.log",
            error_file=f"{results_folder_host}/logs/%A_err.log",
            working_directory=results_folder_host)
        script_path = self.slurm_communicator.write_slurm_script(
            script_content, results_folder_local)
        result = self.slurm_communicator.submit_job(
            script_path, host_cluster, additional_options=None)

        # Store and display job ID
        if result["success"] and result["job_id"]:
            self.job_id_input.setText(str(result["job_id"]))
            self.job_info_display.append(f"Job submitted successfully!\nJob ID: {result['job_id']}")
            # print(f"Starting SLURM on Cluster - Job ID: {result['job_id']}")
        else:
            self.job_info_display.append(f"Job submission failed!\n{result['stderr']}")
            print("Failed to start SLURM on Cluster")

    def start_locally(self):
        """"""
        # TODO: load workflow
        print("starting workflow locally")

    def on_cancel_job(self):
        """Cancel the current SLURM job."""
        if not hasattr(self, 'slurm_communicator') or self.slurm_communicator is None:
            self.job_info_display.append("Error: Not connected to SLURM cluster.\nPlease submit a job first.")
            return

        job_id = self.job_id_input.text().strip()
        if not job_id:
            self.job_info_display.append("Error: No job ID specified.\nPlease enter a job ID.")
            return

        try:
            job_id_int = int(job_id)
            result = self.slurm_communicator.cancel_job(job_id_int)

            if result["success"]:
                self.job_info_display.append(f"Job {job_id} cancelled successfully.")
            else:
                self.job_info_display.append(f"Failed to cancel job {job_id}:\n{result['stderr']}")
        except ValueError:
            self.job_info_display.append(f"Error: Invalid job ID '{job_id}'. Must be a number.")

    def on_show_job_status(self):
        """Display the status of the current SLURM job."""
        if not hasattr(self, 'slurm_communicator') or self.slurm_communicator is None:
            self.job_info_display.append("Error: Not connected to SLURM cluster.\nPlease submit a job first.")
            return

        job_id = self.job_id_input.text().strip()
        if not job_id:
            self.job_info_display.append("Error: No job ID specified.\nPlease enter a job ID.")
            return

        try:
            job_id_int = int(job_id)
            result = self.slurm_communicator.get_job_status(job_id_int)

            if result["success"]:
                self.job_info_display.append(f"\n=== Job {job_id} Status ===")
                self.job_info_display.append(f"Status: {result['status']}")
                if result['details']:
                    self.job_info_display.append("Details:")
                    for key, value in result['details'].items():
                        if value:  # Only show non-empty values
                            self.job_info_display.append(f"  {key}: {value}")
            else:
                self.job_info_display.append(f"Failed to get status for job {job_id}:\n{result.get('error', 'Unknown error')}")
        except ValueError:
            self.job_info_display.append(f"Error: Invalid job ID '{job_id}'. Must be a number.")

    def on_list_jobs(self):
        """List all SLURM jobs for the current user."""
        if not hasattr(self, 'slurm_communicator') or self.slurm_communicator is None:
            self.job_info_display.append("Error: Not connected to SLURM cluster.\nPlease submit a job first.")
            return

        result = self.slurm_communicator.list_jobs()

        if result["success"]:
            jobs = result.get("jobs", [])
            if jobs:
                self.job_info_display.append("\n=== Your SLURM Jobs ===")
                self.job_info_display.append(f"{'Job ID':<10} {'Status':<12} {'Name':<20} {'Time':<10}")
                self.job_info_display.append("-" * 52)
                for job in jobs:
                    job_id = job.get("job_id", "N/A")
                    status = job.get("status", "N/A")
                    name = job.get("job_name", "N/A")
                    time = job.get("time", "N/A")
                    self.job_info_display.append(f"{job_id:<10} {status:<12} {name:<20} {time:<10}")
            else:
                self.job_info_display.append("No jobs found for current user.")
        else:
            self.job_info_display.append(f"Failed to list jobs:\n{result.get('error', 'Unknown error')}")

    def on_show_queue_info(self):
        """Display SLURM queue/partition information."""
        if not hasattr(self, 'slurm_communicator') or self.slurm_communicator is None:
            self.job_info_display.append("Error: Not connected to SLURM cluster.\nPlease submit a job first.")
            return

        result = self.slurm_communicator.get_queue_info()

        if result["success"]:
            partitions = result.get("partitions", [])
            if partitions:
                self.job_info_display.append("\n=== SLURM Queue Information ===")
                self.job_info_display.append(f"{'Partition':<15} {'Avail':<8} {'Nodes':<8} {'State':<12}")
                self.job_info_display.append("-" * 43)
                for partition in partitions:
                    name = partition.get("name", "N/A")
                    avail = partition.get("availability", "N/A")
                    nodes = partition.get("nodes", "N/A")
                    state = partition.get("state", "N/A")
                    self.job_info_display.append(f"{name:<15} {avail:<8} {nodes:<8} {state:<12}")
            else:
                self.job_info_display.append("No partition information available.")
        else:
            self.job_info_display.append(f"Failed to get queue info:\n{result.get('error', 'Unknown error')}")

    def _clear_parameter_layout(self):
        """Clear all widgets from the module parameters layout."""
        while self.module_parameters_layout.count():
            item = self.module_parameters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _create_parameter_widget(self, param_name, param_metadata):
        """Factory method to create appropriate widget based on parameter type.

        Args:
            param_name: Name of the parameter
            param_metadata: Metadata dict with 'type', 'default', 'min', 'max', etc.

        Returns:
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

        else:  # str or fallback
            widget = QtWidgets.QLineEdit()
            default = param_metadata.get("default", "")
            if default is not None:
                widget.setText(str(default))
            if param_metadata.get("required", False):
                widget.setPlaceholderText("Required")
            return widget, "str"

    def _get_widget_value(self, widget, original_type, widget_info=None):
        """Get value from widget based on its original type.

        Args:
            widget: The Qt widget
            original_type: Original type string ('int', 'float', 'bool', 'str', 'options', 'dict')
            widget_info: ParameterWidgetInfo (needed for dict types)

        Returns:
            str or dict: String representation of the value, or dict for nested parameters
        """
        if original_type == "dict" and widget_info and widget_info.sub_parameters:
            # For dict types with nested parameters
            if isinstance(widget, QtWidgets.QCheckBox):
                # Optional dict: only include if checkbox is checked
                if not widget.isChecked():
                    return None

            # Recursively get values from sub-parameters
            nested_values = {}
            for sub_param_name, sub_widget_info in widget_info.sub_parameters.items():
                sub_value = self._get_widget_value(
                    sub_widget_info.widget,
                    sub_widget_info.original_type,
                    sub_widget_info
                )
                if sub_value is not None:  # Only include non-None values
                    nested_values[sub_param_name] = sub_value
            return nested_values

        elif isinstance(widget, QtWidgets.QLineEdit):
            return widget.text()
        elif isinstance(widget, QtWidgets.QComboBox):
            return widget.currentText()
        elif isinstance(widget, QtWidgets.QSpinBox):
            return str(widget.value())
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            return str(widget.value())
        elif isinstance(widget, QtWidgets.QCheckBox):
            return str(widget.isChecked())
        else:
            raise TypeError(f"Unknown widget type: {type(widget)}")

    def _set_widget_value(self, widget, value_data, original_type, widget_info=None):
        """Set widget value from string based on original type.

        Args:
            widget: The Qt widget
            value_data: String representation of value, or dict for nested parameters
            original_type: Original type string ('int', 'float', 'bool', 'str', 'options', 'dict')
            widget_info: ParameterWidgetInfo (needed for dict types)
        """
        if original_type == "dict" and widget_info and widget_info.sub_parameters:
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
                if hasattr(widget_info.row_widget, 'nested_rows'):
                    for nested_row in widget_info.row_widget.nested_rows:
                        nested_row.setVisible(True)

                # Recursively set values in sub-parameters
                for sub_param_name, sub_widget_info in widget_info.sub_parameters.items():
                    if sub_param_name in value_data:
                        sub_value_data = value_data[sub_param_name]

                        # Recursively set nested parameter value
                        self._set_widget_value(
                            sub_widget_info.widget,
                            sub_value_data,
                            sub_widget_info.original_type,
                            sub_widget_info
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
            # Handle both bool strings and Python bool repr
            is_checked = str(value_data).lower() in ('true', '1', 'yes', 'on')
            widget.setChecked(is_checked)

    def _on_cmd_button_clicked(self, param_name):
        """Handle cmd button click - opens prior result dialog.

        Args:
            param_name: Name of the parameter to populate
        """
        # Determine which workflow list to use
        current_tab_index = self.workflow_tabs.currentIndex()
        if current_tab_index == 0:
            workflow_modules = self.single_workflow_modules
        elif current_tab_index == 1:
            workflow_modules = self.aggregation_workflow_modules
        else:
            return

        # Create dialog
        dialog = ParameterCmdDialog(workflow_modules, self.module_descriptor, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            module_index, result_name, command_type, timing = dialog.get_selection()
            if module_index is not None and result_name:
                # Format base reference: "timing@index: module_name.result_name"
                # e.g., "before@1: localize.net_gradient" or "start@0: identify.locs"
                # e.g. "('$get_prior_result', 'all_results,01_testmodule,myresult')"
                module_name = workflow_modules[module_index][0]
                if timing == "before":
                    timing_cmd = "$"
                else:
                    timing_cmd = "$$"
                base_reference = f"{timing}@{module_index}: {module_name}.{result_name}"

                # Wrap in command function if not "Previous Result"
                if command_type == "Previous Result":
                    reference_string = f"('{timing_cmd}get_prior_result', 'all_results, {module_index}_{module_name}, {result_name}')"
                else:
                    # Format as command(reference)
                    # e.g., "sum(before@1: localize.net_gradient)"
                    reference_string = f"{command_type.lower()}({base_reference})"

                # Convert widget to QLineEdit and populate
                self._convert_widget_to_textbox(param_name, reference_string)

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

        Args:
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

    def _create_parameter_row(self, param_name, param_metadata, indent_level=0):
        """Create a parameter row with widgets, supporting nested dicts.

        Args:
            param_name: Name of the parameter
            param_metadata: Metadata dict with type, description, default, etc.
            indent_level: Indentation level for nested parameters (0 = top level)

        Returns:
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
                    sub_widget_info.row_widget.setVisible(False)  # Hide initially

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
                toggle_function=toggle_nested_params if not is_required else None
            )
            return widget_info

        else:
            # Regular parameter (not a dict)
            widget, original_type = self._create_parameter_widget(param_name, param_metadata)

            # Set tooltip
            description = param_metadata.get("description", "")
            if description:
                widget.setToolTip(description)
                label.setToolTip(description)

            # Connect type-specific validation signal
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(self._on_parameter_changed)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentTextChanged.connect(self._on_parameter_changed)
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
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
                original_type=original_type
            )
            return widget_info

    def _populate_parameter_widgets(self, module_params):
        """Create and populate parameter entry widgets from module parameters.

        Args:
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
            widget_info = self._create_parameter_row(param_name, param_metadata, indent_level=0)
            self.parameter_widgets[param_name] = widget_info

            # Add main row to layout
            self.module_parameters_layout.addWidget(widget_info.row_widget)

            # Add nested rows if this is a dict parameter
            if hasattr(widget_info.row_widget, 'nested_rows'):
                for sub_row in widget_info.row_widget.nested_rows:
                    self.module_parameters_layout.addWidget(sub_row)

        # Add stretch at the end to push widgets to the top
        self.module_parameters_layout.addStretch()

    def _validate_parameters(self):
        """Validate that all required parameters are filled.

        Returns:
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
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
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
            value = self._get_widget_value(widget_info.widget, widget_info.original_type, widget_info)
            # Skip None values (from unchecked optional dicts)
            if value is not None:
                param_values[param_name] = value

        # Update the appropriate workflow list
        if self.editing_workflow_tab == 0:  # Single Dataset Workflow
            if self.editing_workflow_index < len(self.single_workflow_modules):
                module_name = self.single_workflow_modules[self.editing_workflow_index][0]
                # Update parameters while keeping module name
                self.single_workflow_modules[self.editing_workflow_index] = (module_name, param_values)
        elif self.editing_workflow_tab == 1:  # Aggregation Workflow
            if self.editing_workflow_index < len(self.aggregation_workflow_modules):
                module_name = self.aggregation_workflow_modules[self.editing_workflow_index][0]
                # Update parameters while keeping module name
                self.aggregation_workflow_modules[self.editing_workflow_index] = (module_name, param_values)

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()

    def excepthook(type, value, tback):
        lib.cancel_dialogs()
        QtCore.QCoreApplication.instance().processEvents()
        message = "".join(traceback.format_exception(type, value, tback))
        errorbox = QtWidgets.QMessageBox.critical(
            window, "An error occured", message
        )
        errorbox.exec_()
        sys.__excepthook__(type, value, tback)

    sys.excepthook = excepthook
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

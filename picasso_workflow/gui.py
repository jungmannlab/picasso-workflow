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
import tempfile

logger = logging.getLogger(__name__)


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

    def analysis_documentation(self, i, parameters, results):
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

    def convert_zeiss_movie(self, i, parameters, results):
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
                "type": "file",
                "description": "Path to the input Zeiss .czi file",
                "extensions": [".czi"],
                "required": True,
            },
            "output_filepath": {
                "type": "file",
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

    def load_dataset_movie(self, i, parameters, results):
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
                "type": "file",
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

    def load_dataset_localizations(self, i, parameters, results):
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
                "type": "file",
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

    def identify(self, i, parameters, results):
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
                "max": 10000.0,
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

    def localize(self, i, parameters, results):
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

    def export_brightfield(self, i, parameters, results):
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
                "type": "file",
                "description": "Path to the input brightfield image",
                "extensions": [".tif", ".tiff", ".png", ".jpg"],
                "required": True,
            },
            "output_filepath": {
                "type": "file",
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

    def render(self, i, parameters, results):
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

    def undrift_rcc(self, i, parameters, results):
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

    def undrift_aim(self, i, parameters, results):
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

    def manual(self, i, parameters, results):
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

    def summarize_dataset(self, i, parameters, results):
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

    def density(self, i, parameters, results):
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

    def dbscan(self, i, parameters, results):
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

    def hdbscan(self, i, parameters, results):
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

    def binding_event_analysis(self, i, parameters, results):
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

    def smlm_clusterer(self, i, parameters, results):
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

    def gaussian_mixture_cluster(self, i, parameters, results):
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

    def nneighbor(self, i, parameters, results):
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
                "type": ["file", "numpy.ndarray", "list"],
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

    def save_single_dataset(self, i, parameters, results):
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
    def load_datasets_to_aggregate(self, i, parameters, results):
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

    def align_channels(self, i, parameters, results):
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

    def combine_channels(self, i, parameters, results):
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

    def save_datasets_aggregated(self, i, parameters, results):
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

    def spinna_manual(self, i, parameters, results):
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

    def spinna(self, i, parameters, results):
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

    def ripleysk(self, i, parameters, results):
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

    def ripleysk2(self, i, parameters, results):
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

    def ripleysk_average(self, i, parameters, results):
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

    def ripleysk_average2(self, i, parameters, results):
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

    def protein_interactions(self, i, parameters, results):
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

    def protein_interactions_average(self, i, parameters, results):
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

    def create_mask(self, i, parameters, results):
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

    def create_mask2(self, i, parameters, results):
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

    def refine_mask_by_density(self, i, parameters, results):
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

    def dbscan_molint(self, i, parameters, results):
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

    def CSR_sim_in_mask(self, i, parameters, results):
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

    def find_cluster_motifs(self, i, parameters, results):
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

    def interaction_graph(self, i, parameters, results):
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

    def plot_densities(self, i, parameters, results):
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

    def find_gold(self, i, parameters, results):
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

    def find_similar(self, i, parameters, results):
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

    def find_structures(self, i, parameters, results):
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

    def undrift_from_picked(self, i, parameters, results):
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
                "type": ["numpy.ndarray", "file"],
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

    def filter_locs(self, i, parameters, results):
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

    def filter_transient_binding(self, i, parameters, results):
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

    def link_locs(self, i, parameters, results):
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

    def pairwise_module_executor(self, i, parameters, results):
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

    def random_val(self, i, parameters, results):
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

    def labeling_efficiency_analysis(self, i, parameters, results):
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

    def write_slurm_script(self, script_content, remote_path):
        """Write SLURM script content to a file on the remote host.

        Args:
            script_content (str): Complete SLURM script content
            remote_path (str): Path where to save the script on remote host

        Returns:
            dict: Result of the write operation (see execute_ssh_command)
        """
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
                    f"{self.username}@{self.hostname}:{remote_path}",
                ]
            )

            logger.debug(f"Copying SLURM script: {' '.join(scp_cmd)}")

            result = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=self.timeout
            )

            if result.returncode == 0:
                # Make script executable
                chmod_result = self.execute_ssh_command(
                    f"chmod +x {remote_path}"
                )
                if chmod_result["success"]:
                    logger.info(
                        f"SLURM script written successfully to {remote_path}"
                    )
                    return {
                        "stdout": f"Script written to {remote_path}",
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

    def submit_job(self, script_path, additional_options=None):
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

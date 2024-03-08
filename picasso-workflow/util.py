#!/usr/bin/env python
"""
Module Name: util.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Utility functions for the package
"""
import abc
import logging


logger = logging.getLogger(__name__)


class AbstractModuleCollection(abc.ABC):
    """Describes the modules an analysis and reporting pipeline
    must support. This needs to be implemented
    in classes in workflow.py, analyse.py and confluence.py,
    such that the workflow class can call the other's methods
    """
    def __init__(self):
        pass

    @classmethod
    def load_dataset(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso.
        """
        pass

    @classmethod
    def identify(self):
        """Identifies localizations in a loaded dataset.
        """
        pass

    @classmethod
    def localize(self):
        """Localizes Spots previously identified.
        """
        pass

    @classmethod
    def undrift_rcc(self):
        """Undrifts localized data using redundant cross correlation.
        """
       pass

    @classmethod
    def undrift(self):
        pass


def correct_path_separators(file_path):
	"""Ensure correct path separators ('/' or '\') in a file path.
	Args:
		file_path : str
			input file path with any of the two separators
	Returns:
		file_path : str
			the file path with separators according to operating system
	"""
    path_components = re.split(r'[\\/]', file_path)
    file_path = os.path.join(*path_components)
    return file_path

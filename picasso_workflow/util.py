#!/usr/bin/env python
"""
Module Name: util.py
Author: Heinrich Grabmayr
Initial Date: March 7, 2024
Description: Utility functions for the package
"""
import abc
import logging
import inspect
import re
import os
import copy


logger = logging.getLogger(__name__)


class AbstractModuleCollection(abc.ABC):
    """Describes the modules an analysis and reporting pipeline
    must support. This needs to be implemented
    in classes in analyse.py and confluence.py,
    such that the workflow class can call the other's methods
    """

    def __init__(self):
        pass

    ##########################################################################
    # Single-dataset workflow modules
    ##########################################################################

    @abc.abstractmethod
    def load_dataset(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso."""
        pass

    @abc.abstractmethod
    def identify(self):
        """Identifies localizations in a loaded dataset."""
        pass

    @abc.abstractmethod
    def localize(self):
        """Localizes Spots previously identified."""
        pass

    @abc.abstractmethod
    def undrift_rcc(self):
        """Undrifts localized data using redundant cross correlation."""
        pass

    @abc.abstractmethod
    def manual(self):
        """Describes a manual step, for which the workflow is paused."""
        pass

    @abc.abstractmethod
    def summarize_dataset(self):
        """Summarizes the results of a dataset analysis."""
        pass

    @abc.abstractmethod
    def save_single_dataset(self):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward."""
        pass

    ##########################################################################
    # Aggregation workflow modules
    ##########################################################################

    @abc.abstractmethod
    def load_datasets_to_aggregate(self):
        """Loads data of multiple single-dataset workflows into one
        aggregation workflow."""
        pass

    @abc.abstractmethod
    def align_channels(self):
        """Saves the locs and info of a single dataset; makes loading
        for the aggregation workflow more straightforward."""
        pass


class ParameterTiler:
    """Multiplies a set of parameters according to a tile command.
    This has the usecase of e.g. doing multiple analogue analyses
    for different datasets, which are then aggregated.
    Uses the ParameterCommandExecutor, so the same commands will
    be used.
    """

    def __init__(self, parent_object, tile_entries, map_dict={}):
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
        """
        logger.debug("Initializeing ParameterTiler")
        self.tile_entries = tile_entries
        self.ntiles = len(list(tile_entries.values())[0])
        self.map_dict = map_dict
        self.parent_object = parent_object

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
            pce = ParameterCommandExecutor(self.parent_object, self.map_dict)
            logger.debug(f"running with parameters {parameters}")
            result_parameters.append(pce.run(copy.deepcopy(parameters)))
        if (tags := self.tile_entries.get("#tags")) is None:
            tags = [""] * len(result_parameters)

        return result_parameters, tags


class ParameterCommandExecutor:
    """Scans parameter sets for commands and executes them.
    This is useful e.g. in the picasso-workflow.workflow.WorkflowRunner
    where some parameters of later modules depend on results of previous
    modules. These can be retrieved with this ParameterCommandExecutor.
    """

    def __init__(self, parent_object, map_dict={}):
        """
        Args:
            parent_object : object
                the object to execute the command on.
                e.g. the WorkflowRunner itself
            map_dict : dict
                a dictionary to map values using the $map command
        """
        self.parent_object = parent_object
        self.map = map_dict

    def run(self, parameters):
        """Scan a parameter set for commands to execute prior to module
        execution.
        commands: '$get_prior_result'
        Args:
            parameters : dict
                the parameters for a module
        """
        logger.debug("Running ParameterCommandExecutor")
        logger.debug(parameters)
        return self.scan(parameters)

    def scan(self, itrbl):
        if isinstance(itrbl, dict):
            res = self.scan_dict(itrbl)
        elif isinstance(itrbl, list):
            res = self.scan_list(itrbl)
        elif isinstance(itrbl, tuple):
            res = self.scan_tuple(itrbl)
        else:
            res = itrbl
        return res

    def scan_dict(self, d):
        for k, v in d.items():
            d[k] = self.scan(v)
        return d

    def scan_list(self, li):
        for i, it in enumerate(li):
            li[i] = self.scan(it)
        return li

    def scan_tuple(self, t):
        if len(t) == 2 and isinstance(t[0], str) and t[0][0] == "$":
            # this is a parameter command
            if t[0] == "$get_prior_result":
                logger.debug(f"Getting prior result from {t[1]}.")
                res = self.get_prior_result(t[1])
                logger.debug(f"Prior result is {res}.")
            elif t[0] == "$map":
                res = self.map[t[1]]
                logger.debug(f"Mapping {t[1]}: {res}")
            # elif add more parameter commands
            return res
        else:
            # it's just a normal tuple
            tout = []
            for i, it in enumerate(t):
                # logger.debug(f"{i}: {it}")
                tout.append(self.scan(it))
            return tuple(tout)

    def get_prior_result(self, locator):
        """In some cases, input parameters for a module should be taken from
        prior results. This is performed here
        Args:
            locator : str
                the chain of attributes for finding the prior result, comma
                separated. They all need to be obtainable with getattr,
                starting from this class e.g. "results, load, sample_movie,
                sample_frame_idx" obtains
                self.results['load']['sample_movie']['sample_frame_idx']
        Returns:
            the last attribute in the chain.
        """
        root_att = self.parent_object
        for att_name in locator.split(","):
            if att_name == "$all":
                # root_att is a list, and all items should be equally processed
                # in the next rounds
                pass
            else:
                if isinstance(root_att, list):
                    root_att = [
                        self.get_attribute(list_att, att_name)
                        for list_att in root_att
                    ]
                else:
                    root_att = self.get_attribute(root_att, att_name)
        logger.debug(f"Prior Result of {locator} is {root_att}")
        return root_att

    def get_attribute(self, root_att, att_name):
        if isinstance(root_att, dict):
            att = root_att.get(att_name.strip())
        elif isinstance(root_att, object):
            try:
                att = getattr(root_att, att_name.strip())
            except AttributeError as e:
                logger.error(
                    f"Could not get attribute {att_name} "
                    + f"from {str(root_att)}."
                )
                raise e
        # logger.debug(f'From {root_att}, extracting "{att_name}": {att}')
        return att


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

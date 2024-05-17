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
import numpy as np


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
    def convert_zeiss_movie(self):
        """Converts a DNA-PAINT movie into .raw, as supported by picasso."""
        pass

    @abc.abstractmethod
    def load_dataset_movie(self):
        """Loads a DNA-PAINT dataset in a format supported by picasso."""
        pass

    @abc.abstractmethod
    def load_dataset_localizations(self):
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
    def export_brightfield(self):
        """Opens a single-plane tiff image and saves it to png with
        contrast adjustment."""
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

    # @abc.abstractmethod
    # def aggregate_cluster(self):
    #     """Aggregate along the cluster column.
    #     Uses picasso.postprocess.cluster_combine"""
    #     pass

    @abc.abstractmethod
    def density(self):
        """Calculate local localization density"""
        pass

    @abc.abstractmethod
    def dbscan(self):
        """Perform clustering using dbscan"""
        pass

    @abc.abstractmethod
    def hdbscan(self):
        """Perform clustering using hdbscan"""
        pass

    @abc.abstractmethod
    def smlm_clusterer(self):
        """Perform clustering using the smlm clusterer"""
        pass

    @abc.abstractmethod
    def nneighbor(self):
        """Calculate Nearest Neighbor distances"""
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

    @abc.abstractmethod
    def combine_channels(self):
        """Combines multiple channels into one dataset. This is relevant
        e.g. for RESI."""
        pass

    @abc.abstractmethod
    def save_datasets_aggregated(self):
        """save data of multiple single-dataset workflows from one
        aggregation workflow."""
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
        elif isinstance(itrbl, np.generic):
            if self.to_simple_type:
                res = float(itrbl)
        else:
            res = itrbl
        return res

    def scan_dict(self, d):
        for k, v in d.items():
            d[k] = self.scan(v)
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
        The commands can be combined with numeric operations:
            ('$get_previous_module_result *2', 'nena')
            The arithmetic expression must not contain any spaces.
        """
        if (
            len(t) == 2
            and isinstance(t[0], str)
            and t[0][: len(self.command_sign)] == self.command_sign
        ):
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

                def is_valid_expression(expression):
                    pattern = r"^[\d+\-*/\s()]+$"
                    return re.match(pattern, expression) is not None

                if not is_valid_expression(aritexp):
                    raise PriorResultError(
                        f"'{aritexp}' is not a valid numeric "
                        + "arithmetic expression."
                    )
                res = eval(str(res) + aritexp)
            return res
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
                logger.debug(f"Leaving {root_att}, to get all.")
                pass
            else:
                try:
                    if isinstance(root_att, list):
                        logger.debug(
                            f"Getting all {att_name}attributes of {root_att}"
                        )
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
        logger.debug(f"Prior Result of {locator} is {root_att}")
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

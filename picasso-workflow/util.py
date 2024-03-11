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


class ParameterTiler():
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
                for example:
                    tile_entries = {'file_name': ['a1.tiff', 'a2.tiff']}
                    parameters = {'load': {'filename': ('$map', 'file_name')}}
            map_dict : dict
                a dictionary to map values using the $map command
        """
        self.tile_entries = tile_entries
        self.ntiles = tile_entries[tile_entries.keys()[0]]
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
                if the map_dict contains the key '$tags', its value is
                returned (supposed to be tags to use for naming),
                otherwise list of empty strings
        """
        result_parameters = []
        for i in range(self.ntiles):
            # set the tile parameters according to the iteration
            for k, v in self.tile_entries:
                self.map_dict[k] = v[i]
            pce = ParameterCommandExecutor(self.parent_object, self.map_dict)
            result_parameters.append(pce.run(parameters))
        if not tags := self.map_dict.get('$tags'):
            tags = [''] * len(result_parameters)

        return result_parameters, tags


class ParameterCommandExecutor():
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
        return scan_dict(parameters)

    def scan_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = scan_dict(v)
            elif isinstance(v, list):
                d[k] = scan_list(v)
            elif isinstance(v, tuple):
                d[k] = scan_tuple(v)
        return d

    def scan_list(self, l):
        for i, it in enumerate(l):
            if isinstance(it, dict):
                l[i] = scan_dict(it)
            elif isinstance(it, list):
                l[i] = scan_list(it)
            elif isinstance(it, tuple):
                l[i] = scan_tuple(it)
        return l

    def scan_tuple(self, t):
        if len(t) == 2 and isinstance(t[0], str) and t[0][0]=='$':
            # this is a parameter command
            if t[0] == '$get_prior_result':
                logger.debug(f'Getting prior result from {t[1]}.')
                res = self.get_prior_result(t[1])
                logger.debug(f'Prior result is {res}.')
            elif t[0] == '$map':
                res = self.map[t[1]]
            # elif add more parameter commands
            return res
        else:
            # it's just a normal tuple
            tout = []
            for i, it in enumerate(t):
                if isinstance(it, dict):
                    tout[i] = scan_dict(it)
                elif isinstance(it, list):
                    tout[i] = scan_list(it)
                elif isinstance(it, tuple):
                    tout[i] = scan_tuple(it)
            return tuple(tout)


    def get_prior_result(self, locator):
        """In some cases, input parameters for a module should be taken from
        prior results. This is performed here
        Args:
            locator : str
                the chain of attributes for finding the prior result, comma separated.
                They all need to be obtainable with getattr, starting from this class
                e.g. "results_load, sample_movie, sample_frame_idx"
                obtains self.results_load['sample_movie']['sample_frame_idx']
        Returns:
            the last attribute in the chain.
        """
        root_att = self.parent_object
        for att_name in locator.split(','):
            if att_name == '$all':
                # root_att is a list, and all items should be equally processed
                # in the next rounds
                pass
            else
                if isinstance(root_att, list):
                    root_att = [self.get_attribute(list_att, att_name) for list_att in root_att]
                else:
                    root_att = self.get_attribute(root_att, att_name)
        return root_att

    def get_attribute(self, root_att, att_name):
        try:
            att = getattr(root_att, att_name.strip())
        except AttributeError as e:
            logger.error(f'Could not get attribute {att_name} from {str(root_att)}.')
            raise e
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
    path_components = re.split(r'[\\/]', file_path)
    file_path = os.path.join(*path_components)
    return file_path

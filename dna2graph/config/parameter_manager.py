import os
import copy
import json

from appdirs import user_config_dir
from dna2graph.constants import (
    APP_NAME, 
    DEFAULT_CONFIG_FILENAME, 
    USER_CONFIG_FILENAME
)
from dna2graph.utils import get_asset_path


def update_dict(base, updates):
    '''
    Overrides values in the base dictionary with those in 
    the updates dictionary, recursively.
    Only updates keys that exist in the base dictionary,
    and where the type of the value in updates matches
    the type of the value in the base dictionary.
    '''
    result = copy.deepcopy(base)
    for k, v in result.items():
        if k not in updates or type(updates[k]) != type(v):
            continue

        if isinstance(v, dict):
            result[k] = update_dict(v, updates[k])
        else:
            result[k] = updates[k]

    return result


def dict_diff(base, updated):
    '''
    Returns a dictionary containing the differences between
    a base dictionary and an updated version of the same dictionary.
    Updates are assumed not to add new keys nor change value types.
    '''
    diff = {}
    for k, v in base.items():
        if v == updated[k]:
            continue

        if isinstance(v, dict):
            diff[k] = dict_diff(v, updated[k])
        else:
            diff[k] = updated[k]

    return diff


class ParameterManager:
    ''' 
    Manages application parameters.
    '''
    def __init__(self):
        # Load the default config
        self.default_config_path = get_asset_path(DEFAULT_CONFIG_FILENAME)

        with open(self.default_config_path, 'r') as f:
            self.default_config = json.load(f)

        # Load the user configuration
        user_config_directory = user_config_dir(APP_NAME)
        os.makedirs(user_config_directory, exist_ok=True)
        self.user_config_path = os.path.join(user_config_directory, USER_CONFIG_FILENAME)

        if os.path.exists(self.user_config_path):
            with open(self.user_config_path, 'r') as f:
                self.user_config = json.load(f)
        else:
            self.user_config = {}

        # Initialize in-memory parameters with default_config
        self.in_memory_config = copy.deepcopy(self.default_config)

        # Override default_config with user configuration
        self.update_in_memory(self.user_config)

    def reset_to_default_config(self):
        '''
        Resets the in-memory parameters to the default_config
        and removes the user configuration file.
        '''
        self.in_memory_config = copy.deepcopy(self.default_config)
        self.user_config = {}
        if os.path.exists(self.user_config_path):
            os.remove(self.user_config_path)

    def update_in_memory(self, updates):
        ''' 
        Updates the in-memory parameters with a dictionary.
        '''
        self.in_memory_config = update_dict(
            self.in_memory_config, 
            updates
        )
        self.user_config = dict_diff(
            self.default_config, 
            self.in_memory_config
        )

    def save_user_config(self):
        '''
        Saves the current in-memory parameters
        to the user configuration file.
        '''
        with open(self.user_config_path, 'w') as f:
            json.dump(self.user_config, f, indent=4)
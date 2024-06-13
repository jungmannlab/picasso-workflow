# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:12:22 2022

@author: reinhardt
"""
import yaml
import os
import h5py


# adapted from Picasso io.py
def load_info(path):
    path_base, path_extension = os.path.splitext(path)
    filename = path_base + ".yaml"
    try:
        with open(filename, "r") as info_file:
            info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    except FileNotFoundError:
        print(
            "\nAn error occured. Could not find metadata file:\n{}".format(
                filename
            )
        )
    return info


# adapted from Picasso io.py
def save_info(path, info, default_flow_style=False):
    with open(path, "w") as file:
        yaml.dump_all(info, file, default_flow_style=default_flow_style)


# adapted from Picasso io.py
def save_locs(path, locs, info):
    # locs = _lib.ensure_sanity(locs, info)
    with h5py.File(path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs)
    base, ext = os.path.splitext(path)
    info_path = base + ".yaml"
    save_info(info_path, info)

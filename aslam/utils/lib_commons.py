"""Commom Helpers."""

import glob

import simplejson
import yaml


def read_yaml_file(file_path):
    """Read yaml file."""
    with open(file_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data


def read_json_file(file_path):
    """Read json file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = simplejson.load(f)
    return data


def get_filenames(folder, is_base_name=False):
    """Get all filenames under the specific folder.

    e.g.:
        full name: data/rgb/000001.png
        base name: 000001.png
    """
    full_names = sorted(glob.glob(folder + "/*"))
    if is_base_name:
        base_names = [name.split("/")[-1] for name in full_names]
        return base_names

    return full_names


def get_intrinsic_mat(k_mat):
    """Get intrinsic for K."""
    return [
        k_mat[0],
        k_mat[3],
        k_mat[6],
        k_mat[1],
        k_mat[4],
        k_mat[7],
        k_mat[2],
        k_mat[5],
        k_mat[8],
    ]

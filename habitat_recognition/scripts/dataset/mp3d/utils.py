import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def get_glb_path(scan_dir, scan_name):
    return os.path.join(scan_dir, scan_name, f"{scan_name}.glb")


def get_ply_path(scan_dir, scan_name):
    return os.path.join(scan_dir, scan_name, f"{scan_name}_semantic.ply")


# O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0
def getOBB(object_dict):
    center = np.array(
        [
            float(object_dict["px"]),
            float(object_dict["py"]),
            float(object_dict["pz"]),
        ]
    )
    axis0 = np.array(
        [
            float(object_dict["a0x"]),
            float(object_dict["a0y"]),
            float(object_dict["a0z"]),
        ]
    )
    axis1 = np.array(
        [
            float(object_dict["a1x"]),
            float(object_dict["a1y"]),
            float(object_dict["a1z"]),
        ]
    )
    radii = np.array(
        [
            float(object_dict["r0"]),
            float(object_dict["r1"]),
            float(object_dict["r2"]),
        ]
    )

    # calculate the rotation matrix
    boxRotation = np.zeros((3, 3))
    boxRotation[:, 0] = axis0
    boxRotation[:, 1] = axis1
    boxRotation[:, 2] = np.cross(boxRotation[:, 0], boxRotation[:, 1])

    return center, boxRotation, radii


def read_house_file(house_file):
    house_format = """
        H name label num_images num_panoramas num_vertices num_surfaces num_segments num_objects num_categories num_regions num_portals num_levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        L level_index num_regions label  px py pz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
        P portal_index region0_index region1_index label  xlo ylo zlo xhi yhi zhi  0 0 0 0
        S surface_index region_index 0 label px py pz  nx ny nz  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        V vertex_index surface_index label  px py pz  nx ny nz  0 0 0
        P name  panorama_index region_index 0  px py pz  0 0 0 0 0
        I image_index panorama_index  name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
        C category_index category_mapping_index category_mapping_name mpcat40_index mpcat40_name 0 0 0 0 0
        O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0 
        E segment_index object_index id area px py pz xlo ylo zlo xhi yhi zhi  0 0 0 0 0
    """
    ############ parse house file format ########################3
    parse_dict = {}
    for line in house_format.splitlines():
        line = line.strip().split()
        if len(line) > 0:
            parse_dict[line[0]] = {}
            for shift in range(1, len(line)):
                if line[shift] != "0":  # drop zero column
                    parse_dict[line[0]][line[shift]] = shift

    ########### parse house file with format dictionary ##############
    house_dict = {
        "H": [],  # general house info
        "L": [],  # level
        "R": [],  # region
        "P": [],  # portal
        "S": [],  # triangle mesh surface
        "V": [],  # vertex
        "P": [],  # panorama image
        "I": [],  # image
        "C": [],  # category
        "O": [],  # object
        "E": [],  # segment
    }

    with open(house_file, "r") as f:
        next(f)  # skip file encoding header
        for line in f:
            line = line.split()
            if len(line) > 0:
                item_fmt = parse_dict[line[0]]
                item_dict = {
                    item_property: line[item_fmt[item_property]]
                    for item_property in item_fmt.keys()
                }
                house_dict[line[0]].append(item_dict)

    return house_dict, parse_dict


def read_label_mapping(filename, label_from="mpcat40", label_to="nyu40"):

    table_header = {"mpcat40": "mpcat40index", "nyu40": "nyu40id"}
    col_from = table_header[label_from]
    col_to = table_header[label_to]

    assert os.path.exists(filename)
    mapping = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[col_from]] = int(row[col_to])
    return mapping

# xyz_origin, rgb, label, instance_label

import os
import numpy as np
import argparse
from multiprocessing import Pool
import open3d as o3d
from parso import parse
from plyfile import PlyData, PlyElement
import time

# generate vertex instance segmentation label by voting of triangle faces containing this vertex
def generate_vertex_inst_label(scans_dir, scene_name):

    scene_dir = os.path.join(scans_dir, scene_name)
    scene_ply = os.path.join(scene_dir, f"{scene_name}_semantic.ply")
    plydata = PlyData.read(scene_ply)
    # scene_o3d = o3d.io.read_point_cloud(scene_ply)
    # NOTE: open3d does not support face color yet, 2021/11/15
    vertex = plydata["vertex"]
    face = plydata["face"]
    num_vertex = len(vertex.data)
    face_vote = [[] for _ in range(num_vertex)]
    vert_inst_labels = -np.ones(num_vertex, dtype=int)  # set default to -1

    for tri_face in face:
        face_verts = tri_face[0]
        face_inst_label = tri_face[1]
        # append inst label vote for all vertices of this triangle surface
        face_vote[face_verts[0]].append(face_inst_label)
        face_vote[face_verts[1]].append(face_inst_label)
        face_vote[face_verts[2]].append(face_inst_label)

    # select max face instance label vote as the instance label of the vertex
    for i in range(num_vertex):
        vert_face_vote = face_vote[i]
        if len(vert_face_vote) > 0:
            vert_inst_labels[i] = max(vert_face_vote, key=vert_face_vote.count)
        else:  # this vertex has no triangle mesh surface
            vert_inst_labels[i] = -1

    return vert_inst_labels


def wrap_save_vertex_inst_label(
    scans_dir, scene_name, save_suffix="_pclseg.txt"
):

    start = time.time()
    vert_inst_labels = generate_vertex_inst_label(scans_dir, scene_name)
    end = time.time()
    inst_label_file = os.path.join(
        scans_dir, scene_name, f"{scene_name}{save_suffix}"
    )
    np.savetxt(inst_label_file, vert_inst_labels, fmt="%d")
    print(f"Scene {scene_name} processed, {end-start} seconds elapsed")
    return


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_dir",
        type=str,
        default="/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans",
    )
    parser.add_argument("--dataset", type=str, default="matterport")
    parser.add_argument(
        "--scene_names", nargs="*", default=[]  # "17DRP5sb8fy", "1LXtFkjw3qL"
    )  # set to "17DRP5sb8fy" for testing
    # parser.add_argument("--segfile_appendix", type=str, default="pclseg.txt")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    if len(args.scene_names) > 0:  # test
        scene_names = args.scene_names
    else:  # all scenes in scan_dir
        scene_names = os.listdir(args.scan_dir)
    print(f"There are {len(scene_names)} scenes to process")

    if len(scene_names) == 1:
        wrap_save_vertex_inst_label(args.scan_dir, scene_names[0])
    else:
        zip_args = [(args.scan_dir, scene) for scene in scene_names]
        with Pool(processes=8) as p:
            p.starmap(wrap_save_vertex_inst_label, zip_args)


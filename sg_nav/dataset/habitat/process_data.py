# collect vertices of one object 
import json 
import numpy as np
import argparse
import os 
import pickle 

from tqdm import tqdm
from habitat_sim import Simulator
from habitat_sim.scene import SemanticScene
from dataset.habitat.simulator import init_sim, make_cfg
from plyfile import PlyData, PlyElement

# local import 
from dataset.habitat.utils import *

##################  dictionary format ####################
# def generate_vertex_seg(sim, settings, scene_ply, scene_glb, save_file=None):

#     if sim:
#         settings["scene"] = scene_glb
#         cfg = make_cfg(settings)
#         # load a new scene without closing and opening simulator 
#         sim.reconfigure(cfg)
#     else:
#         sim = init_sim(scene_glb)

#     plydata = PlyData.read(scene_ply)
#     # scene_o3d = o3d.io.read_point_cloud(scene_ply)
#     # NOTE: open3d does not support face color yet, 2021/11/15
#     vertex = plydata['vertex']
#     face = plydata['face']

#     object_ids = [int(obj.id.split("_")[-1]) for obj in sim.semantic_scene.objects]
#     pcl_seg_dict = {obj_id:[] for obj_id in object_ids}

#     for tri in face.data:
#         pcl_seg_dict[tri[1]].append(tri[0]) # append a (3,) numpy int32 array 

#     for obj_id in object_ids:
#         pcl_seg_dict[obj_id] =  np.unique(np.concatenate(pcl_seg_dict[obj_id])).astype(int).tolist()

#     if save_file:
#         with open(save_file, "w") as f:
#             json.dump(pcl_seg_dict, f)

################ list format ######################
# NOTE: SemanticScene object cannot be pickled 
#  
# def save_semantic_scene(scene_glb, save_file):
#     sim = init_sim(scene_glb)
#     with open(save_file, "wb") as f:
#         pickle.dump(sim.semantic_scene, f)
#     return 

# Update 2021/11/16: need to close simulator & restart it, 
#                   otherwise unreleased memory overflow   
def generate_vertex_annot(scene_ply, scene_glb, seg_file=None, normals_file=None):
    
    ############## segment point clouds with tri-mesh annotations ############ 
    sim,_,_ = init_sim(scene_glb)

    plydata = PlyData.read(scene_ply)
    # scene_o3d = o3d.io.read_point_cloud(scene_ply)
    # NOTE: open3d does not support face color yet, 2021/11/15
    vertex = plydata['vertex']
    face = plydata['face']
    num_vertex = len(vertex.data)

    object_ids = [int(obj.id.split("_")[-1]) for obj in sim.semantic_scene.objects]
    pcl_seg_arr = np.zeros(num_vertex, dtype=int) - 1
    pcl_seg_dict = {obj_id:[] for obj_id in object_ids}

    for tri in face.data:
        pcl_seg_dict[tri[1]].append(tri[0]) # append a (3,) numpy int32 array 

    for obj_id in object_ids:
        pcl_seg_dict[obj_id] =  np.unique(np.concatenate(pcl_seg_dict[obj_id])).astype(int)
        # convert to list of labels 
        pcl_seg_arr[pcl_seg_dict[obj_id]] = obj_id

    if seg_file:
        np.savetxt(seg_file, pcl_seg_arr, fmt="%d")
    
    # release resources 
    sim.close()

    ############## calculate vertex normals for feature extraction #########
    mesh_o3d = o3d.io.read_triangle_mesh(scene_ply)
    mesh_o3d.compute_vertex_normals()
    normals_arr = np.asarray(mesh_o3d.vertex_normals)
    if normals_file:
        with open(normals_file, "wb") as f:
            np.save(f, normals_arr)
    return 




def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, default="/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans")
    parser.add_argument("--dataset", type=str, default="matterport")
    parser.add_argument("--scene_names", nargs="*", default=[])  # set to "17DRP5sb8fy" for testing
    # parser.add_argument("--segfile_appendix", type=str, default="pclseg.txt")
    args = parser.parse_args()  

    return args 
    
if __name__ == "__main__":

    args = parse_args()
    if args.scene_names: # test
        scene_names = args.scene_names
    else:
        scene_names = os.listdir(args.scan_dir)

    # TODO:consider use multi-processing lib to accelerate data processing
    print("step2: save all point clouds segmentation from semantic mesh files")
    print(f"There are {len(scene_names)} scans to process...")
    for scan in tqdm(scene_names):
        
        scan_ply_path = get_ply_path(args.scan_dir, scan)
        scan_glb_path = get_glb_path(args.scan_dir, scan)
        pclseg_path = os.path.join(args.scan_dir, scan, f"{scan}_pclseg.txt")
        normals_path = os.path.join(args.scan_dir, scan, f"{scan}_normals.npy")
        generate_vertex_annot(scan_ply_path, scan_glb_path, seg_file=pclseg_path, normals_file=normals_path)

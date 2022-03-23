import argparse
import os 
from habitat_sim.agent.controls.controls import SceneNodeControl
import open3d as o3d
import numpy as np
from dataset.habitat.simulator import init_sim 
from plyfile import PlyData, PlyElement
import pickle 

# local import 
from scene_graph.scene_graph_cls import SceneGraphHabitat
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.utils import visualize_scene_graph
from scene_graph.scene_graph_pred import SceneGraphPredictor

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, default="/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans")
    parser.add_argument("--scene_name", type=str, default="fzynW3qQPVF") # fzynW3qQPVF 17DRP5sb8fy
    parser.add_argument("--dataset", type=str, default="matterport")
    parser.add_argument("--visualize", type=bool, default=True)
    # runtime options     
    parser.add_argument("--mode", type=str, default="runtime", choices=["runtime", "offline"])
    parser.add_argument("--rel_dist_thresh", type=float, default=2.0, help="threshold of max distance between two objects that could have a relationship")
    # offline options 
    parser.add_argument("--scene_graph_dump_path", type=str, default="/home/junting/project_cvl/SceneGraphNav/data/model_dump/mp3d_model_dump.pkl")

    args = parser.parse_args()  
    args.scene_ply_path = os.path.join(args.scan_dir, args.scene_name, f'{args.scene_name}_semantic.ply')
    args.scene_glb_path = os.path.join(args.scan_dir, args.scene_name, f'{args.scene_name}.glb')
    args.pclseg_path = os.path.join(args.scan_dir, args.scene_name, f'{args.scene_name}_pclseg.txt')
    args.pcl_normals_path = os.path.join(args.scan_dir, args.scene_name, f'{args.scene_name}_normals.npy')
    
    return args 

if __name__ == "__main__":
    
    args = parse_args()
    if args.mode == "runtime":
        # raise NotImplementedError
        ############ load ground truth pointclouds ####################
        # o3d_pcl = o3d.io.read_point_cloud(args.scene_ply_path) # only load vertices 
        # scene = o3d.io.read_triangle_mesh(args.scene_ply_path) # load mesh file \
        
        ############ initialize habitat simulator and ground truth scene graph ########
        sim, action_names, sim_settings = init_sim(args.scene_glb_path)

        # initialize ground truth scene graph 
        config = SceneGraphHabitatConfig()
        scene_graph = SceneGraphHabitat(config, scene_name=args.scene_name)
        scene_graph.load_gt_scene_graph(args.scene_ply_path, args.pclseg_path, args.pcl_normals_path, sim)

        ########### visualize loaded scene with bounding boxes ########################
        
        if args.visualize:
            visualize_scene_graph(scene_graph)

        ############ extract GCN features by pretrained 3DSSG model 
        feature_extractor = SceneGraphPredictor(args.rel_dist_thresh)
        object_nodes = [scene_graph.object_layer.obj_dict[obj_id]
            for obj_id in scene_graph.object_layer.obj_ids]

        ''' 
        extractor returns a dictionary:  
        results={
            "pred_obj_prob": pred_obj_prob, # (N, D=20) numpy array
            "pred_obj_confidence": pred_obj_confidence, # (N,) numpy array
            "pred_obj_label": pred_obj_label, # (N,) numpy array 
            "edges": edges, # (M,2) numpy array, represented by object index (not id!)
            "pred_rel_prob": pred_rel_prob, # (M, F=9)
            "pred_rel_confidence": pred_rel_confidence,
            "pred_rel_label": pred_rel_label
        }
        '''
        results = feature_extractor.predict(object_nodes)
        print(results)
    
    elif args.mode == "offline":
        with open(args.scene_graph_dump_path, "rb") as f:
            mp3d_data = pickle.load(f)
import argparse
import os 
from habitat_sim.agent.controls.controls import SceneNodeControl
import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from torch.multiprocessing import Pool, Process, set_start_method
import pickle 
import time 

# local import 
from dataset.habitat.simulator import init_sim 
from scene_graph.scene_graph_cls import SceneGraphHabitat
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.utils import visualize_scene_graph
from scene_graph.scene_graph_pred import SceneGraphPredictor

# workaround of path problem 
# import sys 
# import pathlib
# ROOT_PATH = pathlib.Path(__file__).parent.absolute()
# sys.path.append(ROOT_PATH)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str, default="/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans")
    parser.add_argument("--scene_name", type=str, default=None) # test scene : 17DRP5sb8fy
    parser.add_argument("--dataset", type=str, default="matterport")
    parser.add_argument("--dump_path", type=str, default="./mp3d_model_dump.pkl")
    parser.add_argument("--rel_dist_thresh", type=float, default=2.0, help="threshold of max distance between two objects that could have a relationship")
    parser.add_argument("--num_process", type=int, default=2) # 10GB GPU memory 
    args = parser.parse_args()  
    
    return args 


def process_run(feature_extractor, scan_dir, scene_name):
        
        scene_ply_path = os.path.join(scan_dir, scene_name, f'{scene_name}_semantic.ply')
        scene_glb_path = os.path.join(scan_dir, scene_name, f'{scene_name}.glb')
        pclseg_path = os.path.join(scan_dir, scene_name, f'{scene_name}_pclseg.txt')
        pcl_normals_path = os.path.join(scan_dir, scene_name, f'{scene_name}_normals.npy')
        
        ############ initialize habitat simulator and ground truth scene graph ########
        sim, action_names, sim_settings = init_sim(scene_glb_path)

        # intialize ground truth scene graph 
        config = SceneGraphHabitatConfig()
        scene_graph = SceneGraphHabitat(config, scene_name=scene_name)
        scene_graph.load_gt_scene_graph(scene_ply_path, pclseg_path, pcl_normals_path, sim)

        ############ extract GCN features by pretrained 3DSSG model 

        object_nodes = [scene_graph.object_layer.obj_dict[obj_id]
            for obj_id in scene_graph.object_layer.obj_ids]

        ''' 
        extractor returns a dictionary:  
        results={
            "pred_obj_prob": pred_obj_prob, # (N, D) numpy array
            "pred_obj_confidence": pred_obj_confidence, # (N,) numpy array
            "pred_obj_label": pred_obj_label, # (N,) numpy array 
            "edges": edges, # (M,2) numpy array, represented by object index (not id!)
            "pred_rel_prob": pred_rel_prob, # (M,2)
            "pred_rel_confidence": pred_rel_confidence,
            "pred_rel_label": pred_rel_label
        }
        '''
        results = feature_extractor.predict(object_nodes, return_input=True)
        results["scene_name"] = scene_name
        sim.close()
        return results

if __name__ == "__main__":
    
    args = parse_args()
    ############ parse all scene paths ####################  
    if args.scene_name == None:
        scene_names = os.listdir(args.scan_dir)
    else: # Set one scene name for debugging
        scene_names = [args.scene_name]
    
    ##################### FIXME: problem with multiprocessing ######################

    # feature_extractor = SceneGraphPredictor(args.rel_dist_thresh, multiprocess=True)
    # data_queue = [(feature_extractor, args.scan_dir, scene_name) for scene_name in scene_names]
    # start_tick = time.time()
    # set_start_method('spawn')


    # with Pool(args.num_process) as p:
    #     data = p.starmap(process_run, data_queue)

    # with open(args.dump_path, "wb") as f:
    #     pickle.dump(data, f)

    # stop_tick = time.time()
    # print(f"Total time: {stop_tick - start_tick} seconds, each scene costs {(stop_tick - start_tick)/len(scene_names)} seconds")
    
    ############### serial running ###########################
    feature_extractor = SceneGraphPredictor(args.rel_dist_thresh, multiprocess=False)
    start_tick = time.time()

    data = []
    for scene_name in scene_names:
        data.append(process_run(feature_extractor, args.scan_dir, scene_name))

    with open(args.dump_path, "wb") as f:
        pickle.dump(data, f)

    stop_tick = time.time()
    print(f"Total time: {stop_tick - start_tick} seconds, each scene costs {(stop_tick - start_tick)/len(scene_names)} seconds")
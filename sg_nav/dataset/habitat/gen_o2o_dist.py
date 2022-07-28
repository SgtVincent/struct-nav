# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import argparse
import os
import pickle
import time
import glob

import habitat_sim
import numpy as np
from numpy import ma
import open3d as o3d
from dataset.habitat.simulator import init_sim

# local import
from dataset.habitat.utils import display_map, display_path
from scene_graph.utils import getOBB, grid_xz_to_points, visualize_scene_graph
from tqdm import tqdm
from habitat.core.simulator import ShortestPathPoint

display = False
verbose = False

objnav_gibson_train25 = [
 'Allensville',
 'Beechwood',
 'Benevolence',
 'Coffeen',
 'Cosmos',
 'Forkland',
 'Hanson',
 'Hiteman',
 'Klickitat',
 'Lakeville',
 'Leonardo',
 'Lindenwood',
 'Marstons',
 'Merom',
 'Mifflinburg',
 'Newfields',
 'Onaga',
 'Pinesdale',
 'Pomaria',
 'Ranchester',
 'Shelbyville',
 'Stockman',
 'Tolstoy',
 'Wainscott',
 'Woodbine'
]

objnav_gibson_val5 = [
    "Collierville",
    "Corozal",
    "Darden",
    "Markleeville",
    "Wiconisco"
]

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
}

coco_label_mapping = {v:k for k, v in coco_categories.items()}

MAX_DIST = 1e8

def get_c2c_dist_from_o2o(categories, objects, o2o_dist):
    
    num_class = len(categories)
    c2c_dist = np.zeros((num_class, num_class))
    objects_cls_vec = np.array([obj['class_label'] for obj in objects])
    for i, source_cat in enumerate(categories):
        for j in range(i+1, num_class):
            target_cat = categories[j]
            source_label = source_cat['class_label']  
            target_label = target_cat['class_label']
            source_mask = objects_cls_vec == source_label
            target_mask = objects_cls_vec == target_label
            inst_dists = o2o_dist[source_mask, :][:, target_mask].reshape(-1)
            
            # filter out negative distances 
            # valid_mask = inst_dists >= 0
            # if np.any(valid_mask):
                # min_c2c_dist = inst_dists[valid_mask].min()
            # else: # no valid path for any two instances of the two categories
            #     min_c2c_dist = MAX_DIST
            min_c2c_dist = inst_dists.min()
            c2c_dist[i,j] = c2c_dist[j,i] = min_c2c_dist
        
    return c2c_dist


def gen_o2o_dist_for_scene(sim, scene_name, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # filter None objects in semantic_scene
    objects = [obj for obj in sim.semantic_scene.objects if obj is not None]
    num_obj = len(objects)
    o2o_dist = np.zeros((num_obj, num_obj))
    objects_list = []
    class_dict = {}
    
    for i, obj in enumerate(objects):
        
        # 1. clip start point(object position) to closest valid navigable point
        loc = obj.aabb.center
        nav_start = sim.pathfinder.snap_point(loc)
        class_name = obj.category.name()
        class_label = obj.category.index()
        object_dict = {
            'index': i,
            'class_name': class_name,
            'class_label': class_label,
            'location': loc,
            'size': obj.aabb.sizes,
            'nav_location': nav_start,
        }
        objects_list.append(object_dict)
        class_dict[class_name] = class_label

        for j, obj_t in enumerate(objects):
            # 2. clip end point to closest valid navigable point
            target_loc = obj_t.aabb.center
            nav_end = sim.pathfinder.snap_point(target_loc)

            # @markdown 2. Use ShortestPath module to compute path between samples.
            path = habitat_sim.ShortestPath()
            path.requested_start = nav_start
            path.requested_end = nav_end
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            if not found_path:
                geodesic_distance = MAX_DIST
                
            o2o_dist[i, j] = geodesic_distance
    
    categories = [{
        'class_name': k,
        'class_label': v,} 
        for k, v in class_dict.items()]

    c2c_dist = get_c2c_dist_from_o2o(categories, objects_list, o2o_dist)

    out_dict = {
        "scene_boundary": sim.pathfinder.get_bounds(),
        "objects": objects_list,
        "categories": categories,
        "o2o_dist": o2o_dist,
        "c2c_dist": c2c_dist,
    }
    save_path = os.path.join(
        save_dir, f"{scene_name}_o2o_dist.pickle"
    )
    with open(save_path, "wb") as f:
        pickle.dump(out_dict, f)

def get_scene_prior_mean_policy(o2o_dist_paths):
    
    num_class = len(coco_categories)
    # list of prior mats from all scenes
    prior_mats = []
    for path in o2o_dist_paths:
        # prior matrix is 15 x 15, using 15 coco categories 
        scene_prior_mat = ma.zeros((num_class, num_class)) 
        scene_prior_mat[:] = ma.masked
        
        with open(path, "rb") as f:
            o2o_dist_data = pickle.load(f) 
        
        cats = o2o_dist_data['categories']
        c2c_dist_matrix = o2o_dist_data['c2c_dist']
        c2c_dist_matrix = ma.masked_values(c2c_dist_matrix, MAX_DIST)
        
        prior_idx = []
        c2c_idx = []
        # collect c2c data of 15 coco categories from c2c_dist_matrix
        for i, cat in enumerate(cats):
            cat_name = cat['class_name']
            if cat_name in coco_categories:
                prior_idx.append(coco_categories[cat_name])
                c2c_idx.append(i)
        # gather data for intersection categories between coco 15 and categories in scene        
        prior_row_idx, prior_col_idx = np.meshgrid(prior_idx, prior_idx, indexing='ij')
        c2c_row_idx, c2c_col_idx = np.meshgrid(c2c_idx, c2c_idx, indexing='ij')
        scene_prior_mat[prior_row_idx.reshape(-1), prior_col_idx.reshape(-1)] = \
            c2c_dist_matrix[c2c_row_idx.reshape(-1), c2c_col_idx.reshape(-1)]
        
        prior_mats.append(scene_prior_mat)
        
    prior_mat = ma.concatenate(prior_mats, axis=0).reshape((-1, num_class, num_class)).mean(axis=0)
    prior_mat = prior_mat.filled(MAX_DIST)
    return prior_mat

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="/media/junting/SSD_data/habitat_data/scene_datasets/gibson",
    )
    parser.add_argument(
        "--scene_names",
        nargs="*",
        default=objnav_gibson_val5, 
        # default=objnav_gibson_train25,
        help="if not empty, only use scenes in the list"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/media/junting/SSD_data/struct_nav/gibson/val/o2o_dist"
        # default="/media/junting/SSD_data/struct_nav/gibson/train/o2o_dist"
    )
    # set to "17DRP5sb8fy" for testing
    parser.add_argument("--dataset", type=str, default="gibson")
    # parser.add_argument(
    #     "--rel_dist_thresh",
    #     type=float,
    #     default=2.0,
    #     help="threshold of max distance between two objects that could have a relationship",
    # )

    args = parser.parse_args()

    if len(args.scene_names) == 0:
        args.scene_names = [ scene.split("/")[-1].split(".")[0]
            for scene in glob.glob(f"{args.scene_dir}/*.glb") ]
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    return args


if __name__ == "__main__":

    args = parse_args()

    ######### extract scene meta info, o2o dist, min c2c dist from each scene ######
    # for scene_name in args.scene_names:
    #     # try:
    #     time_start = time.time()
    #     if args.dataset == "gibson":
    #         scene_glb_path = os.path.join(args.scene_dir, f"{scene_name}.glb")
    #     elif args.dataset == "mp3d":
    #         scene_glb_path = os.path.join(args.scene_dir, scene_name, f"{scene_name}.glb")
        
    #     sim, action_names, sim_settings = init_sim(scene_glb_path)
    #     # intialize ground truth scene graph
    #     # config = SceneGraphHabitatConfig()

    #     gen_o2o_dist_for_scene(sim, scene_name, save_dir=args.save_dir)

    #     sim.close()
    #     elapse = time.time() - time_start
    #     print(
    #         f"Generate object-to-object distance matrix in scene {scene_name}, elapse time:{elapse}"
    #     )
    ##### process the extracted scene layout info to construct scene prior matrix ####
    # TODO: decide which _o2o_dist.pickle files to process
    o2o_dist_paths = [
        "/media/junting/SSD_data/struct_nav/gibson/val/o2o_dist/Darden_o2o_dist.pickle",
    ]
    out_dir = "/home/junting/habitat_ws/src/struct-nav/habitat/prior_data"
    out_name = "Darden_mean.npy"
    prior_policy = "mean"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # mean: calculate the mean of c2c distance across all scenes
    prior_mats = []
    if prior_policy == "mean":
        scene_prior_matrix = get_scene_prior_mean_policy(o2o_dist_paths)
    else:
        raise NotImplementedError
    
    np.save(os.path.join(out_dir, out_name), scene_prior_matrix)
    
        
        
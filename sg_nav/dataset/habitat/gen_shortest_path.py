# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import argparse
import os
import pickle
import time

import habitat_sim
import numpy as np
import open3d as o3d
from dataset.habitat.simulator import init_sim

# local import
from dataset.habitat.utils import display_map, display_path
from habitat_sim.agent.controls.controls import SceneNodeControl
from plyfile import PlyData, PlyElement
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.scene_graph_cls import SceneGraphHabitat
from scene_graph.scene_graph_pred import SceneGraphPredictor
from scene_graph.utils import getOBB, grid_xz_to_points, visualize_scene_graph
from tqdm import tqdm

from habitat.core.simulator import ShortestPathPoint

display = False
verbose = False


def gen_shortest_path_for_scene(scene_graph, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    region_grid = scene_graph.region_layer.segment_grid
    free_space_grid = scene_graph.free_space_grid
    object_node_grid = scene_graph.object_layer.segment_grid
    # idx_arr: list of (row_idx, col_idx)
    idx_arr = np.argwhere(free_space_grid)
    # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map
    # need to swap the column in idx_arr
    free_space_points = grid_xz_to_points(
        scene_graph.scene_bounds,
        idx_arr[:, [1, 0]],
        scene_graph.height,
        scene_graph.meters_per_grid,
    )
    obj_ids = scene_graph.object_layer.obj_ids

    map_meta_dict = {
        "scene_boundary": scene_graph.scene_bounds,
        "meters_per_grid": scene_graph.meters_per_grid,
        "region_segmentation": region_grid,
        "free_space_grid": free_space_grid,
        "free_grid_idx": idx_arr,
        "free_space_points": free_space_points,
        "object_ids": obj_ids,
        "object_labels": scene_graph.object_layer.get_labels(obj_ids),
        "object_node_grid": object_node_grid,
    }
    meta_file_path = os.path.join(save_dir, f"geomap_meta.pickle")
    with open(meta_file_path, "wb") as f:
        pickle.dump(map_meta_dict, f)

    # NOTE: save large object with numpy array causes unknown memory error
    # save shortest paths by object ids instead!

    for object_id in tqdm(scene_graph.object_layer.obj_ids, desc="Objects:"):
        # 1. clip end point(target object position) to closest valid navigable point
        object_node = scene_graph.object_layer.obj_dict[object_id]
        end = object_node.center
        end_exact = sim.pathfinder.is_navigable(end)
        if not end_exact:
            end = sim.pathfinder.snap_point(end)
        if verbose:
            snap_info = (
                "" if end_exact else f", not navigable, snapped to {end}"
            )
            print(f"end point {object_node.center}", snap_info)

        # add geodesic distance map and shortest path dict for object
        obj_geomap_and_shpath_dict = {
            "object_id": object_id,
            "object_label": scene_graph.object_layer.get_labels([object_id])[
                0
            ],
            "end": end,
            "end_exact": end_exact,
            "geodesic_dist_map": np.full_like(
                free_space_grid, np.inf, dtype=float
            ),
            "shortest_path_list": [],
        }

        path_cnt = 0
        for i, pos in enumerate(free_space_points):
            # 2. clip start point to closest valid navigable point
            start = pos
            start_exact = sim.pathfinder.is_navigable(start)
            if not start_exact:
                start = sim.pathfinder.snap_point(start)
            if verbose:
                snap_info = (
                    ""
                    if start_exact
                    else f", not navigable, snapped to {start}"
                )
                print(f"start point {free_space_points[i,:]}", snap_info)
            shortest_path_dict = {
                "start": np.copy(start),
                "start_exact": start_exact,
            }

            # @markdown 2. Use ShortestPath module to compute path between samples.
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = end
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points = path.points

            # DEBUG info
            if verbose:
                print("found_path : " + str(found_path))
                print("geodesic_distance : " + str(geodesic_distance))
                print("path_points : " + str(path_points))

            if display and found_path:
                display_path(sim, path_points, plt_block=True)

            # add shortest path info of
            shortest_path_dict.update(
                {
                    "found_path": found_path,
                    "geodesic_distance": geodesic_distance,
                    "path_points": np.copy(path_points),
                }
            )
            map_idx = idx_arr[i]
            obj_geomap_and_shpath_dict["geodesic_dist_map"][
                map_idx[0], map_idx[1]
            ] = geodesic_distance
            obj_geomap_and_shpath_dict["shortest_path_list"].append(
                shortest_path_dict
            )

            if found_path:
                path_cnt += 1

        obj_save_path = os.path.join(
            save_dir, f"geomap_object_{object_id}.pickle"
        )
        with open(obj_save_path, "wb") as f:
            pickle.dump(obj_geomap_and_shpath_dict, f)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_dir",
        type=str,
        default="/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans",
    )
    # set to "17DRP5sb8fy" for testing
    parser.add_argument("--scene_names", nargs="*", default=[])
    parser.add_argument("--dataset", type=str, default="matterport")
    parser.add_argument(
        "--rel_dist_thresh",
        type=float,
        default=2.0,
        help="threshold of max distance between two objects that could have a relationship",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    ############ initialize habitat simulator and ground truth scene graph ########
    args = parse_args()
    if len(args.scene_names) > 0:
        scene_names = args.scene_names
    else:
        scene_names = os.listdir(args.scan_dir)

    # TODO: parallelize shortest path generation process
    error_scenes = []
    for scene in scene_names:
        try:
            time_start = time.time()
            scene_ply_path = os.path.join(
                args.scan_dir, scene, f"{scene}_semantic.ply"
            )
            scene_glb_path = os.path.join(args.scan_dir, scene, f"{scene}.glb")
            pclseg_path = os.path.join(
                args.scan_dir, scene, f"{scene}_pclseg.txt"
            )
            pcl_normals_path = os.path.join(
                args.scan_dir, scene, f"{scene}_normals.npy"
            )
            house_file_path = os.path.join(
                args.scan_dir, scene, f"{scene}.house"
            )
            shortest_path_dir = os.path.join(
                args.scan_dir, scene, f"shortest_paths"
            )

            sim, action_names, sim_settings = init_sim(scene_glb_path)
            # intialize ground truth scene graph
            config = SceneGraphHabitatConfig()
            sg = SceneGraphHabitat(config, scene_name=scene)
            sg.load_gt_scene_graph(
                scene_ply_path, pclseg_path, pcl_normals_path, sim
            )

            gen_shortest_path_for_scene(sg, save_dir=shortest_path_dir)

            sim.close()
            elapse = time.time() - time_start
            print(
                f"Generate shortest path for all free space in scene {scene}, elapse time:{elapse}"
            )
        except:
            error_scenes.append(scene)

    print("There are exceptions in these scenes:", error_scenes)


# @markdown 4. (optional) Place agent and render images at trajectory points (if found).
# display_path_agent_renders = True  # @param{type:"boolean"}
# if display_path_agent_renders:
#     print("Rendering observations at path points:")
#     tangent = path_points[1] - path_points[0]
#     agent_state = habitat_sim.AgentState()
#     for ix, point in enumerate(path_points):
#         if ix < len(path_points) - 1:
#             tangent = path_points[ix + 1] - point
#             agent_state.position = point
#             tangent_orientation_matrix = mn.Matrix4.look_at(
#                 point, point + tangent, np.array([0, 1.0, 0])
#             )
#             tangent_orientation_q = mn.Quaternion.from_matrix(
#                 tangent_orientation_matrix.rotation()
#             )
#             agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
#             agent.set_state(agent_state)

#             observations = sim.get_sensor_observations()
#             rgb = observations["color_sensor"]
#             semantic = observations["semantic_sensor"]
#             depth = observations["depth_sensor"]

#             if display:
#                 display_sample(rgb, semantic, depth)

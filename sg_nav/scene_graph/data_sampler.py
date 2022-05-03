import argparse
import logging
import os
import pickle
import random
import time
from collections import Counter

import numpy as np
import torch
from dataset.habitat.simulator import init_sim
from git import Object
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.scene_graph_cls import SceneGraphHabitat
from scene_graph.utils import display_image_arr, grid_xz_to_points, inNd
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from torch_geometric.utils.loop import add_self_loops, remove_self_loops

display = False
verbose = False


class PyGDatasetWrapper:
    def __init__(
        self, dataset, node_features, edges, map_ids2idx, add_loops=True
    ):
        self.node_features = node_features
        self.edges = edges
        self.map_ids2idx = map_ids2idx
        self.add_loops = add_loops
        self.data_list = dataset()
        self.data_list = list(map(self.process_data, self.data_list))

    def __call__(self):
        return self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def shuffle(self):
        random.shuffle(self.data_list)

    def process_data(self, data):
        (
            sample_obj_ids,
            objnav_labels,
            target_obj_inout_labels,
            fs_points,
            fs2obj_geo_dist,
        ) = data
        sample_idx = (
            sample_obj_ids.detach().clone().apply_(self.map_ids2idx.get)
        )
        inputs = self.node_features[sample_idx]
        if self.add_loops:
            edges, _ = remove_self_loops(self.edges)
            edges, _ = add_self_loops(self.edges)
        sg_edge_index = self.get_sg_edge_idx(
            self.node_features, sample_idx, edges
        )
        return Data(
            x=inputs,
            edge_index=sg_edge_index,
            y=target_obj_inout_labels,
            n_y=objnav_labels,
        )

    @staticmethod
    def get_sg_edge_idx(node_features, sample_idx, edges):
        map_gidx2sgidx = {
            gidx: sgidx for sgidx, gidx in enumerate(sample_idx.numpy())
        }
        node_mask = torch.zeros(
            node_features.size(0),
            dtype=torch.bool,
            device=node_features.device,
        )
        node_mask[sample_idx] = 1
        edge_mask = node_mask[edges[0]] & node_mask[edges[1]]
        sample_edge_index = edges[:, edge_mask]
        sg_edge_index = (
            sample_edge_index.detach()
            .clone()
            .cpu()
            .apply_(map_gidx2sgidx.get)
            .to(node_features.device)
        )
        return sg_edge_index


class DataSampler(Dataset):
    """Sample data from habitat scene graphs."""

    def __init__(
        self,
        config,
        scene_graph,
        save_dir,
        obj_nav_class_list,
        blk_size_min=8,
        sample_sizes_max=10000,
        source_sample_size=64,
    ) -> None:
        self.object_grid_scale = config.object_grid_scale
        self.scene_graph = scene_graph
        meta_file_path = os.path.join(save_dir, f"geomap_meta.pickle")
        with open(meta_file_path, "rb") as f:
            self.map_meta_dict = pickle.load(f)
        self.obj_ids = self.map_meta_dict["object_ids"]
        logging.info(f"load map with {len(self.obj_ids)} object")
        obj_names = self.scene_graph.object_layer.get_class_names(self.obj_ids)
        logging.info("object counts: {}".format(dict(Counter(obj_names))))

        obj_labels = []
        obj_geo_maps = []
        for obj_id in self.obj_ids:
            obj_save_path = os.path.join(
                save_dir, f"geomap_object_{obj_id}.pickle"
            )
            with open(obj_save_path, "rb") as f:
                obj_geomap_and_shpath_dict = pickle.load(f)
                obj_labels.append(obj_geomap_and_shpath_dict["object_label"])
                obj_geo_maps.append(
                    obj_geomap_and_shpath_dict["geodesic_dist_map"]
                )
        self.obj_labels = np.stack(obj_labels, axis=0)
        self.obj_geo_maps = np.stack(obj_geo_maps, axis=0)
        self.obj_nav_class_list = obj_nav_class_list
        self.num_objnav_class = len(obj_nav_class_list)
        self.class_geo_maps = self.get_class_geo_maps()

        self.blk_size_min = blk_size_min
        self.sample_sizes_max = sample_sizes_max
        self.source_sample_size = source_sample_size

        self.data_list = list(self.generate_sample())

    def __call__(self):
        return self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def shuffle(self):
        random.shuffle(self.data_list)

    def get_class_geo_maps(self):
        """Create geodesic maps for every class."""
        class_geo_maps = []
        for c in self.obj_nav_class_list:
            obj_mask = self.obj_labels == c
            geo_map = self.obj_geo_maps[obj_mask]
            if geo_map.size != 0:
                class_geo_map = np.amin(geo_map, axis=0)
                class_geo_maps.append(class_geo_map)
            else:
                class_geo_map = np.full_like(
                    self.map_meta_dict["free_space_grid"], np.inf, dtype=float
                )
                class_geo_maps.append(class_geo_map)
            if display:
                display_image_arr((1, 1), [class_geo_map])
        return np.stack(class_geo_maps, axis=-1)

    def generate_sample(self):
        """Generate data samples."""
        for data in self.block_sampling(
            self.blk_size_min, self.sample_sizes_max
        ):
            sample_obj_ids, sample_obj_labels, fs_block_arr = data
            if sample_obj_ids.size == 0:
                continue
            if fs_block_arr.size == 0:
                # Skip no free space samples
                continue
            target_obj_inout_labels = self.get_inout_labels(sample_obj_labels)
            objnav_labels = self.get_objnav_labels(sample_obj_labels)
            fs_idx_arr = self.source_sampling(
                fs_block_arr, self.source_sample_size
            )
            fs_points = self.get_free_space_points(fs_idx_arr)
            fs2obj_geo_dist = np.clip(
                self.get_geo_dist(fs_idx_arr), a_min=0.0, a_max=100.0
            )
            sample = [
                sample_obj_ids,
                objnav_labels,
                target_obj_inout_labels,
                fs_points,
                fs2obj_geo_dist,
            ]
            yield list(map(torch.from_numpy, sample))

    def block_sampling(self, blk_size_min=8, sample_sizes=1000):
        """Sampling a block of scene graph."""
        free_space_grid = self.map_meta_dict["free_space_grid"]
        free_space_points = self.map_meta_dict["free_space_points"]
        object_node_grid = self.map_meta_dict["object_node_grid"]
        map_h, map_w = free_space_grid.shape

        # generate random block sizes and start top left corners
        blk_size_hs = np.random.randint(
            blk_size_min, map_h + 1, size=sample_sizes
        )
        blk_size_ws = np.random.randint(
            blk_size_min, map_w + 1, size=sample_sizes
        )
        blk_tops = np.random.randint(
            0, map_h - blk_size_hs + 1, size=sample_sizes
        )
        blk_lefts = np.random.randint(
            0, map_w - blk_size_ws + 1, size=sample_sizes
        )
        for i in range(sample_sizes):
            blk_size_h = blk_size_hs[i]
            blk_size_w = blk_size_ws[i]
            blk_top = blk_tops[i]
            blk_left = blk_lefts[i]
            free_space_mask = np.ones_like(free_space_grid, dtype=bool)
            free_space_mask[
                blk_top : blk_top + blk_size_h,
                blk_left : blk_left + blk_size_w,
            ] = False
            free_space_block = free_space_grid.astype(int).copy()
            free_space_block[free_space_mask] = -1
            fs_block_arr = np.argwhere(free_space_block == 1)

            object_node_mask = np.ones_like(object_node_grid, dtype=bool)
            object_node_mask[
                blk_top
                * self.object_grid_scale : (blk_top + blk_size_h)
                * self.object_grid_scale,
                blk_left
                * self.object_grid_scale : (blk_left + blk_size_w)
                * self.object_grid_scale,
            ] = False
            object_node_block = object_node_grid.copy()
            object_node_block[object_node_mask] = -1
            obj_ids = np.unique(object_node_block[object_node_block != -1])
            obj_labels = self.scene_graph.object_layer.get_labels(obj_ids)
            obj_labels = np.asarray(obj_labels)

            if display:
                display_image_arr(
                    (2, 2),
                    [
                        free_space_grid,
                        free_space_block,
                        object_node_grid,
                        object_node_block,
                    ],
                )
            if verbose:
                print(obj_ids)
                print(self.scene_graph.object_layer.get_class_names(obj_ids))

            yield obj_ids, obj_labels, fs_block_arr

    def source_sampling(self, fs_block_arr, source_sample_size=64):
        "Sample the source nodes (free spaces)."
        # sample with replacement
        idx = np.random.choice(fs_block_arr.shape[0], source_sample_size)
        return fs_block_arr[idx, :]

    def get_inout_labels(self, obj_labels):
        "Sample the targe nodes (objects)."
        return np.asarray(
            [1 if c in obj_labels else 0 for c in self.obj_nav_class_list]
        )

    def get_objnav_labels(self, obj_labels):
        "Convert object labels to object navigation labels."
        return np.asarray(
            [
                self.obj_nav_class_list.index(c)
                if c in self.obj_nav_class_list
                else len(self.obj_nav_class_list)
                for c in obj_labels
            ]
        )

    def get_free_space_points(self, idx_arr):
        "Get free space points."
        free_space_points = grid_xz_to_points(
            self.scene_graph.scene_bounds,
            idx_arr[:, [1, 0]],
            self.scene_graph.height,
            self.scene_graph.meters_per_grid,
        )
        return free_space_points

    def get_geo_dist(self, fs_idx_arr):
        "Get geodesic distance to the targe object class."
        fs_idx_arr = fs_idx_arr.T
        return self.class_geo_maps[fs_idx_arr[0], fs_idx_arr[1], :]


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

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    from dataset.habitat.utils import mp3d_obj_nav_class_list

    ############ initialize habitat simulator and ground truth scene graph ########
    args = parse_args()
    if len(args.scene_names) > 0:
        scene_names = args.scene_names
    else:
        scene_list = os.listdir(args.scan_dir)

    # TODO: parallelize shortest path generation process
    for scene in scene_names:
        time_start = time.time()
        scene_ply_path = os.path.join(
            args.scan_dir, scene, f"{scene}_semantic.ply"
        )
        scene_glb_path = os.path.join(args.scan_dir, scene, f"{scene}.glb")
        pclseg_path = os.path.join(args.scan_dir, scene, f"{scene}_pclseg.txt")
        pcl_normals_path = os.path.join(
            args.scan_dir, scene, f"{scene}_normals.npy"
        )
        house_file_path = os.path.join(args.scan_dir, scene, f"{scene}.house")
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

        sampler = DataSampler(
            config, sg, shortest_path_dir, mp3d_obj_nav_class_list
        )

        for sample in sampler:
            print(sample)

        sim.close()
        elapse = time.time() - time_start
        print(
            f"Generate shortest path for all free space in scene {scene}, elapse time:{elapse}"
        )

import json
import os
from abc import ABC, abstractmethod, abstractproperty
from turtle import color

import numpy as np
import open3d as o3d
import regex
import scene_graph
import scipy
from scipy.spatial import KDTree
from habitat_sim import Simulator, scene
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.object_layer import ObjectLayer

# local import
from scene_graph.region_layer import RegionLayer
from scene_graph.utils import get_corners
from tqdm import tqdm

"""
########################## Update log ###################################
2021/11/06, Junting Chen: Only consider the scene graph of one level in one building 
"""


class SceneGraphBase():

    """Presume that the scene graph have two layers"""
    """ Set a layer to None if it does not exist in class instance """

    def __init__(self) -> None:
        """Initialize scene graph on different 3D datasets"""
        """Parsing of 3D datasets should be implemented in dataset module"""
        self.object_layer = ObjectLayer()
        self.region_layer = RegionLayer()
        return 

    def get_full_graph(self):
        """Return the full scene graph"""
        pass

    def sample_graph(self, method, *args, **kwargs):
        """Return the sub-sampled scene graph
        
        Assume all method-dependent variables passed through kwargs 
        
        """
        if method == "radius_sampling":
            sample_centers = kwargs.get("center")
            if len(sample_centers.shape) == 1:
                # add dummy dim 
                sample_centers = sample_centers[np.newaxis, :]
            
            sample_radius = kwargs.get("radius")
            # by default, calculate distance on x-y plane
            dist_dims = kwargs.get("dist_dims", [0,1])
            
            # build the kdtree to query objects inside the ball/circle
            obj_ids = self.object_layer.obj_ids
            if len(obj_ids) == 0: # empty scene graph
                return [[] for _ in range(sample_centers.shape[0])]
            
            obj_centers = self.object_layer.get_centers(obj_ids)
            kdtree = KDTree(obj_centers[:, dist_dims])
            sample_idx_list = kdtree.query_ball_point(
                sample_centers[:, dist_dims], sample_radius)
            sample_obj_ids_list = [[obj_ids[idx] for idx in sample_idx] 
                                   for sample_idx in sample_idx_list]
            return sample_obj_ids_list
        else:
            raise NotImplementedError
        return 


class SceneGraphHabitat(SceneGraphBase):

    def __init__(self, config, scene_name=None) -> None:
        super().__init__()
        self.config = config
        self.scene_name = scene_name

        # scene parameters
        assert os.path.exists(self.config.floor_heights_file)
        with open(self.config.floor_heights_file, "r") as f:
            floor_heights_dict = json.loads(f.read())
        self.floor_heights = floor_heights_dict[scene_name]
        self.meters_per_grid = (
            self.config.meters_per_grid
        )  # free space grid parameter
        self.object_grid_scale = self.config.object_grid_scale

        # intialize member objects
        self.region_layer = RegionLayer()
        self.object_layer = ObjectLayer()

    def load_gt_scene_graph(
        self, ply_file, pclseg_file, pcl_normals_file, sim: Simulator
    ):

        # 0. load groun truth 3D point clouds and class-agnostic instance segmentation
        o3d_pcl = o3d.io.read_point_cloud(
            ply_file
        )  # open scene mesh file for segmentation and feature extraction
        # points = np.concatenate(
        #     [np.asarray(o3d_pcl.points), np.asarray(o3d_pcl.colors)], axis=1
        # )
        xyz = np.asarray(o3d_pcl.points)
        colors = np.asarray(o3d_pcl.colors)
        # offline generated data
        # NOTE: need to first generate them with process_data.py
        pclseg = np.loadtxt(pclseg_file, dtype=int)
        pcl_normals = np.load(pcl_normals_file)

        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = sim.pathfinder.get_bounds()
        # NOTE: bottom of bounding box could NOT be the floor
        # self.height = self.scene_bounds[0][1] # y-axis points upwards
        self.height = self.floor_heights[0]  # assume one-layer scene
        self.free_space_grid = sim.pathfinder.get_topdown_view(
            self.meters_per_grid, self.height
        )  # binary matrix
        self.region_layer.init_map(
            self.scene_bounds, self.meters_per_grid, self.free_space_grid
        )

        self.dumy_space_grid = sim.pathfinder.get_topdown_view(
            self.meters_per_grid / self.object_grid_scale, self.height
        )  # binary matrix
        self.object_layer.init_map(
            self.scene_bounds,
            self.meters_per_grid / self.object_grid_scale,
            self.dumy_space_grid,
        )

        # 2. load region layer from habitat simulator
        semantic_scene = sim.semantic_scene
        for region in tqdm(semantic_scene.regions):
            # add region node to region layer
            gt_region_id = int(region.id.split("_")[-1])
            sg_region_id = gt_region_id  # counting from 0, -1 for background
            region_bbox = np.stack(
                [
                    region.aabb.center - region.aabb.sizes / 2,
                    region.aabb.center + region.aabb.sizes / 2,
                ],
                axis=0,
            )
            region_node = self.region_layer.add_region(
                region_bbox,
                id=sg_region_id,
                class_name=region.category.name(),
                label=region.category.index(),
            )

            # 3. load object layer from habitat simulator
            for obj in region.objects:
                # print(
                #     f"Object id:{obj.id}, category:{obj.category.name()},"
                #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                # )
                object_id = int(obj.id.split("_")[-1])  # counting from 0
                if self.config.aligned_bbox:
                    center = obj.aabb.center
                    rot_quat = np.array([0, 0, 0, 1])  # identity transform
                    size = obj.aabb.sizes
                else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                    center = obj.obb.center
                    rot_quat = obj.obb.rotation[1, 2, 3, 0]
                    size = obj.obb.sizes
                    size = obj.aabb.sizes

                # TODO: segment point using bounding box
                # Update: 2021/11/16 Using mesh face segmentation instead
                object_point_mask = pclseg == object_id
                object_vertices = xyz[object_point_mask]
                object_colors = colors[object_point_mask]
                object_normals = pcl_normals[object_point_mask]
                node_size = (
                    self.meters_per_grid / self.object_grid_scale
                )  # treat object as a point
                node_bbox = np.stack(
                    [center - node_size / 2, center + node_size / 2], axis=0
                )
                object_node = self.object_layer.add_object(
                    center,
                    rot_quat,
                    size,
                    id=object_id,
                    class_name=obj.category.name(),
                    label=obj.category.index(),
                    vertices=object_vertices,
                    colors=object_colors,
                    normals=object_normals,
                    bbox=node_bbox,
                )

                # connect object to region
                region_node.add_object(object_node)

        return

    def get_full_point_clouds(self):

        points = []
        points_object_ids = []

        for obj_id in self.object_layer.obj_ids:
            obj_node = self.object_layer.obj_dict[obj_id]
            obj_points = np.concatenate(
                [obj_node.vertices, obj_node.colors, obj_node.normals], axis=1
            )
            points.append(obj_points)

            points_object_ids.append(
                np.repeat(obj_id, obj_points.shape[0]).astype(int)
            )

        points = np.concatenate(points, axis=0)
        points_object_ids = np.concatenate(points_object_ids)

        return points, points_object_ids

    # def get_full_graph(self):
    #     pass

    # def sample_graph(self, method, *args, **kwargs):
    #     pass

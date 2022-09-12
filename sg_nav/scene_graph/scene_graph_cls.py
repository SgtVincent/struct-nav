import json
from math import dist
import os
from abc import ABC, abstractmethod, abstractproperty
from turtle import color
from typing import List
import numpy as np
import open3d as o3d
import regex
import scene_graph
import scipy
from scipy.spatial import KDTree
from habitat_sim import Simulator, scene
from scipy.ndimage.morphology import binary_dilation
from nav_msgs.msg import OccupancyGrid

# local import
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.object_layer import ObjectLayer, ObjectNode
from scene_graph.region_layer import RegionLayer, RegionNode
from scene_graph.utils import get_corners
from tqdm import tqdm

"""
########################## Update log ###################################
2021/11/06, Junting Chen: Only consider the scene graph of one level in one building 
"""


class SceneGraphBase:

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
            # get params
            sample_centers = kwargs.get("center")
            sample_radius = kwargs.get("radius")
            # by default, calculate distance on x-y plane
            dist_dims = kwargs.get("dist_dims", [0,1])
            if len(sample_centers.shape) == 1:
                # add dummy dim 
                sample_centers = sample_centers[np.newaxis, :]
                
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
        
        elif method == "soft_radius_sampling":
            sample_centers = kwargs.get("center")
            if len(sample_centers.shape) == 1:
                # add dummy dim 
                sample_centers = sample_centers[np.newaxis, :]
                
            obj_ids = self.object_layer.obj_ids
            if len(obj_ids) == 0: # empty scene graph
                return [[] for _ in range(sample_centers.shape[0])], \
                    [[] for _ in range(sample_centers.shape[0])]
            obj_centers = self.object_layer.get_centers(obj_ids)
            
            sample_obj_ids_list = []
            sample_obj_dists_list = []
            if self.object_layer.flag_grid_map and False: 
                # if there is grid map, dist represented by geodesic distance 
                # TODO: consider how to calculate obj distance to current pos
                pass
            else:
                # if the grid map is not initialized, use manhattan distance instead
                
                for center in sample_centers:  
                    sample_obj_ids_list.append(obj_ids)
                    dists = abs(center[0] - obj_centers[:, 0]) + \
                        abs(center[1] - obj_centers[:, 1])
                    sample_obj_dists_list.append(dists)
                return sample_obj_ids_list, sample_obj_dists_list
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
        xyz = None
        colors = None
        pclseg = None
        pcl_normals = None
        
        if os.path.exists(ply_file) and os.path.exists(pclseg_file):
            o3d_pcl = o3d.io.read_point_cloud(ply_file)  
            # points = np.concatenate(
            #     [np.asarray(o3d_pcl.points), np.asarray(o3d_pcl.colors)], axis=1
            # )
            xyz = np.asarray(o3d_pcl.points)
            colors = np.asarray(o3d_pcl.colors)
            # offline generated data
            # NOTE: need to first generate them with process_data.py
            pclseg = np.loadtxt(pclseg_file, dtype=int)
            
            if os.path.exists(pcl_normals_file):
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
                object_vertices = None
                object_colors = None
                object_normals = None
                if xyz is not None:
                    object_point_mask = pclseg == object_id
                    object_vertices = xyz[object_point_mask]
                    object_colors = colors[object_point_mask]
                    if pcl_normals is not None:
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

class GridMap:
    def __init__(self, time, resolution, width, height, pos, quat, grid):
        self.time = time
        self.resolution = resolution
        self.width = width
        self.height = height
        # the real-world pose of the cell (0,0) in the map.
        self.origin_pos = pos  # (x, y, z)
        self.origin_quat = quat  # quaternion

        self.grid = grid  # numpy array

    @classmethod
    def from_msg(cls, occ_msg: OccupancyGrid):

        grid = np.asarray(occ_msg.data, dtype=np.int8).reshape(
            occ_msg.info.height, occ_msg.info.width
        )
        grid[grid == 100] = 1
        ros_p = occ_msg.info.origin.position
        ros_q = occ_msg.info.origin.orientation
        return cls(
            occ_msg.header.stamp.to_sec(),
            occ_msg.info.resolution,
            occ_msg.info.width,
            occ_msg.info.height,
            np.array([ros_p.x, ros_p.y, ros_p.z]),
            np.quaternion(ros_q.w, ros_q.x, ros_q.y, ros_q.z),
            grid,
        )


class SceneGraphGibson(SceneGraphBase):

    def __init__(self, sim: Simulator, enable_region_layer=True) -> None:
        super().__init__()
        self.sim = sim
        # self.scene_name = scene_name

        # scene parameters
        # self.floor_heights = [0]
        self.height = sim.get_agent(0).state.position[1]
        self.meters_per_grid = 0.05
        self.object_grid_scale = 1
        self.aligned_bbox = True
        self.enable_region_layer = enable_region_layer
        
        # parse habitat.sim.SemanticScene
        self.load_gt_scene_graph()

    def load_gt_scene_graph(self):
        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = self.sim.pathfinder.get_bounds()
        # NOTE: bottom of bounding box could NOT be the floor
        # self.height = self.scene_bounds[0][1] # y-axis points upwards
        # self.height = self.floor_heights[0]  # assume one-layer scene
        self.free_space_grid = self.sim.pathfinder.get_topdown_view(
            self.meters_per_grid, self.height
        )  # binary matrix
        self.region_layer.init_map(
            self.scene_bounds, self.meters_per_grid, self.free_space_grid
        )

        self.dumy_space_grid = self.sim.pathfinder.get_topdown_view(
            self.meters_per_grid / self.object_grid_scale, self.height
        )  # binary matrix
        self.object_layer.init_map(
            self.scene_bounds,
            self.meters_per_grid / self.object_grid_scale,
            self.dumy_space_grid,
        )
        
        semantic_scene = self.sim.semantic_scene
        # 2. load region layer from habitat simulator
        if self.enable_region_layer: # matterport 3D has region annotations 
            for region in semantic_scene.regions:
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
                    if obj is not None:
                        object_id = int(obj.id)  
                        if self.aligned_bbox:
                            center = obj.aabb.center
                            rot_quat = np.array([0, 0, 0, 1])  # identity transform
                            size = obj.aabb.sizes
                        else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                            center = obj.obb.center
                            rot_quat = obj.obb.rotation[1, 2, 3, 0]
                            size = obj.obb.sizes
                            size = obj.aabb.sizes

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
                            bbox=node_bbox,
                        )

                        # connect object to region
                        region_node.add_object(object_node)
                    
        else: # Gibson (original ver.) does not have region annotations 
            # 3. load object layer from habitat simulator
            for object_id, obj in enumerate(semantic_scene.objects):
                if obj is not None:
                    if self.aligned_bbox:
                        center = obj.aabb.center
                        rot_quat = np.array([0, 0, 0, 1])  # identity transform
                        size = obj.aabb.sizes
                    else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                        center = obj.obb.center
                        rot_quat = obj.obb.rotation[1, 2, 3, 0]
                        size = obj.obb.sizes
                        size = obj.aabb.sizes

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
                        bbox=node_bbox,
                    )                
        return

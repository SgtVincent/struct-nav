# FIXME: This script temporarily import scene graph class from sg_nav package
# which is problematic and dangerous, and needs to be deleted before release

# TODO: Publish semantic_scene_graph from habitat to ROS, and reconstruction GT
# scene graph in sg_nav, after moving agent logics to sg_nav package

from functools import partial
import os
import sys
from typing import List
import pathlib
import numpy as np
import open3d as o3d
from scipy.ndimage.morphology import binary_dilation
import quaternion as qt

# import ros_numpy # weird class error
from habitat_sim import Simulator
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion
from utils.transformation import points_habitat2rtab, points_rtab2habitat

UTILS_DIR = pathlib.Path(__file__).resolve().parent
SG_DIR = os.path.join(UTILS_DIR.parent.parent.parent.parent, "sg_nav")
sys.path.append(SG_DIR)

from scene_graph.scene_graph_cls import SceneGraphBase, SceneGraphHabitat
from scene_graph.object_layer import ObjectLayer, ObjectNode
from scene_graph.region_layer import RegionLayer
from scene_graph.utils import project_points_to_grid_xz


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


class SceneGraphSimGT(SceneGraphBase):
    # layers #
    object_layer = ObjectLayer()
    region_layer = RegionLayer()

    def __init__(self, sim: Simulator) -> None:
        super().__init__()
        self.sim = sim
        # self.scene_name = scene_name

        # scene parameters
        # self.floor_heights = [0]
        self.height = sim.get_agent(0).state.position[1]
        self.meters_per_grid = 0.05
        self.object_grid_scale = 1
        self.aligned_bbox = True

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

        # 2. load region layer from habitat simulator
        semantic_scene = self.sim.semantic_scene
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
                # print(
                #     f"Object id:{obj.id}, category:{obj.category.name()},"
                #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                # )
                object_id = int(obj.id.split("_")[-1])  # counting from 0
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
        return

    def get_partial_scene_graph(
        self, grid_map: GridMap, grid_th=3, resolution=0.05
    ) -> List[ObjectNode]:

        # dilate rtabmap grid map
        traversed_grid = grid_map.grid > -1
        partial_mask = binary_dilation(traversed_grid, iterations=grid_th)

        # process object centers
        obj_nodes = self.object_layer.get_objects_by_ids(
            self.object_layer.obj_ids
        )
        obj_centers = np.stack(
            [obj_node.center for obj_node in obj_nodes], axis=0
        )
        obj_centers_rtab = points_habitat2rtab(obj_centers)

        # transform points from rtabmap to gridmap local frame
        r_mat = qt.as_rotation_matrix(grid_map.origin_quat)
        homo_tf = np.zeros((4, 4))
        homo_tf[:3, :3] = r_mat
        homo_tf[:3, 3] = grid_map.origin_pos
        homo_tf[3, 3] = 1.0
        homo_tf = np.linalg.inv(homo_tf)  # from rtabmap to grid_map frame
        homo_centers = np.concatenate(
            [obj_centers_rtab, np.ones((obj_centers_rtab.shape[0], 1))], axis=1
        )
        homo_centers_local = (homo_tf @ homo_centers.T).T  # (N,4)
        centers_local = homo_centers_local[:, :3] / homo_centers_local[:, 3:]

        # project points in map local frame to grids
        centers_grid_xy = (centers_local / resolution)[:, :2].astype(int)
        centers_row = centers_grid_xy[:, 1]
        centers_col = centers_grid_xy[:, 0]
        # obj_observable_mask = np.zeros_like(centers_row)

        obj_in_map_bound_mask = (
            (centers_row >= 0)
            & (centers_row < partial_mask.shape[0])
            & (centers_col >= 0)
            & (centers_col < partial_mask.shape[1])
        )

        obj_partial_mask = (
            partial_mask[
                centers_row[obj_in_map_bound_mask],
                centers_col[obj_in_map_bound_mask],
            ]
            == 1
        )
        obj_in_map_bound_idx = np.where(obj_in_map_bound_mask)[0]
        obj_observable_idx = obj_in_map_bound_idx[obj_partial_mask]
        return [obj_nodes[idx] for idx in obj_observable_idx]

    def get_full_graph(self):
        """Return the full scene graph"""
        return None

    def sample_graph(self, seed, sample_method, ratio, num_nodes):
        """Return the sub-sampled scene graph"""
        return None


if __name__ == "__main__":
    # import importlib.util
    # sys.path.append(UTILS_DIR.parent.parent)
    # NOTE: run this demo with python -m agents.utils.sg_utils
    from utils.simulator import init_sim
    import rosbag
    from habitat_sim.utils.common import quat_from_two_vectors
    import matplotlib.pyplot as plt

    test_scene = "/home/junting/Downloads/datasets/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
    sim = sim, action_names = init_sim(test_scene)
    gt_scenegraph = SceneGraphSimGT(sim)
    bag = rosbag.Bag(
        "/home/junting/habitat_ws/src/struct-nav/habitat/scripts/2022-06-05-11-53-35_0.bag",
        "r",
    )

    # read first message and construct grid map
    topic, grid_map_msg, t = next(
        bag.read_messages(topics=["/rtabmap/grid_map"])
    )
    grid_map = GridMap.from_msg(grid_map_msg)
    bag.close()

    # demo: return partial scene graph masked by grid_map
    partial_scenegraph = gt_scenegraph.get_partial_scene_graph(grid_map)

    # visualize rtabmap gridmap and habitat
    obj_centers = np.stack(
        [obj_node.center for obj_node in partial_scenegraph], axis=0
    )
    obj_centers_2d = project_points_to_grid_xz(
        gt_scenegraph.scene_bounds, obj_centers, gt_scenegraph.meters_per_grid
    )
    fig = plt.figure(figsize=(16, 8))
    fig.add_subplot(121)
    plt.imshow(gt_scenegraph.free_space_grid)
    plt.scatter(
        x=obj_centers_2d[:, 0], y=obj_centers_2d[:, 1], c="r", marker="o"
    )
    fig.add_subplot(122)
    plt.imshow(grid_map.grid, origin="lower")
    plt.show()

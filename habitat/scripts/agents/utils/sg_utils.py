# FIXME: This script temporarily import scene graph class from sg_nav package
# which is problematic and dangerous, and needs to be deleted before release

# TODO: Publish semantic_scene_graph from habitat to ROS, and reconstruction GT
# scene graph in sg_nav, after moving agent logics to sg_nav package

from functools import partial
import os
import sys
import time
from typing import List
import pathlib
import numpy as np
import pickle
import open3d as o3d
from scipy import stats
from scipy.ndimage.morphology import binary_dilation
import quaternion as qt
import ros_numpy 
from sklearn.cluster import DBSCAN
from matplotlib import cm 

from habitat_sim import Simulator
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion
from utils.transformation import points_habitat2world, points_world2habitat
from agents.utils.nms_utils import NMS
from envs.constants import coco_categories, coco_label_mapping

##### FIXME: re-implement scene graph agent to sg_nav package #####
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


class SceneGraphGTGibson(SceneGraphBase):
    """class to load ground truth scene graph from habitat simulator

    Gibson scenes: gibson scenes load layer of height 0.0 by default
    """
    def __init__(self, sim: Simulator, tf_habitat2rtabmap=None, height_filter=True) -> None:
        super().__init__()
        self.sim = sim
        if tf_habitat2rtabmap is None:
            self.rot = np.eye(3)
            self.trans = np.zeros([0,0,0], dtype=float)
        else:
            self.rot = tf_habitat2rtabmap[:3, :3]
            self.trans = tf_habitat2rtabmap[:3, 3]
        self.height_filter = height_filter
            
        # scene parameters
        # self.floor_heights = [0]
        self.height = sim.get_agent(0).state.position[1]
        self.max_height = self.height + 2.5
        self.min_height = self.height - 0.5
        self.meters_per_grid = 0.05
        self.object_grid_scale = 1
        self.aligned_bbox = True

        # parse habitat.sim.SemanticScene
        self.load_gt_scene_graph()

    def load_gt_scene_graph(self):
        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = np.stack(self.sim.pathfinder.get_bounds(), axis=0)
        self.scene_bounds = (self.rot @ self.scene_bounds.T).T + self.trans 
        
        # NOTE: bottom of bounding box could NOT be the floor
        # self.height = self.scene_bounds[0][1] # y-axis points upwards
        # self.height = self.floor_heights[0]  # assume one-layer scene
        self.free_space_grid = self.sim.pathfinder.get_topdown_view(
            self.meters_per_grid, self.height
        )  # binary matrix
        self.region_layer.init_map(
            self.scene_bounds, self.meters_per_grid, self.free_space_grid
        )
        
        # Object sementation on grid map is not needed, disable object layer grid map 
        # self.dumy_space_grid = self.sim.pathfinder.get_topdown_view(
        #     self.meters_per_grid / self.object_grid_scale, self.height
        # )  # binary matrix
        # self.object_layer.init_map(
        #     self.scene_bounds,
        #     self.meters_per_grid / self.object_grid_scale,
        #     self.dumy_space_grid,
        # )


        # 2. load object layer from habitat simulator
        semantic_scene = self.sim.semantic_scene
        for obj_id, obj in enumerate(semantic_scene.objects):
            if obj is not None:
                
                obj_height = obj.obb.center[1] # y-axis for height in habitat coords
                if obj_height > self.max_height or obj_height < self.min_height:
                    continue
                
                if self.aligned_bbox:
                    center = self.rot @ obj.aabb.center + self.trans
                    rot_quat = np.array([0, 0, 0, 1])  # identity transform
                    size = obj.aabb.sizes
                    size = size[[0,2,1]] # flip y-z sizes
                else:  # Use obb, NOTE: quaternion is [w,x,y,z] from habitat, need to convert to [x,y,z,w]
                    center = self.rot @ obj.obb.center + self.trans
                    rot_quat = obj.obb.rotation[1, 2, 3, 0]
                    size = obj.obb.sizes # flip y-z sizes

                class_name = obj.category.name()
                label = -1 
                if class_name in coco_categories:
                    label = coco_categories[class_name]
                object_node = self.object_layer.add_object(
                    center,
                    rot_quat,
                    size,
                    id=obj_id,
                    class_name=class_name,
                    label=label,
                )

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
        # obj_centers_rtab = points_habitat2world(obj_centers)

        # transform points from rtabmap to gridmap local frame
        r_mat = qt.as_rotation_matrix(grid_map.origin_quat)
        homo_tf = np.zeros((4, 4))
        homo_tf[:3, :3] = r_mat
        homo_tf[:3, 3] = grid_map.origin_pos
        homo_tf[3, 3] = 1.0
        homo_tf = np.linalg.inv(homo_tf)  # from rtabmap to grid_map frame
        homo_centers = np.concatenate(
            [obj_centers, np.ones((obj_centers.shape[0], 1))], axis=1
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


# NOTE: category index in habitat gibson is meaningless 
# def load_scene_priors(scene_prior_file):
#     with open(scene_prior_file, "rb") as f:
#         data = pickle.load(f) 
#     # TODO: load scene_prior_matrix from pickle file 
#     return scene_prior_matrix

class SceneGraphRtabmap(SceneGraphBase):
    # layers #
    object_layer = None
    region_layer = None
    
    # TODO: finetune the DBSCAN parameters
    def __init__(self, rtabmap_pcl, point_features=False, label_mapping=None, 
            scene_bounds=None, grid_map=None, map_resolution=0.05, dbscan_eps=1.0, 
            dbscan_min_samples=5, dbscan_num_processes=4, min_points_filter=5,
            dbscan_verbose=False, dbscan_vis=False, label_scale=2, 
            nms=True, nms_th=0.4):

        # 1. get boundary of the scene (one-layer) and initialize map
        self.scene_bounds = scene_bounds
        self.grid_map = grid_map
        self.map_resolution = map_resolution
        self.point_features = point_features
        self.object_layer = ObjectLayer()
        self.region_layer = RegionLayer()
        
        if self.grid_map is not None:
            self.region_layer.init_map(
                self.scene_bounds, self.map_resolution, self.grid_map
            )

        # 2. use DBSCAN with label as fourth dimension to cluster instance points
        
        points = ros_numpy.point_cloud2.pointcloud2_to_array(rtabmap_pcl)
        points = ros_numpy.point_cloud2.split_rgb_field(points)
        xyz = np.vstack((points["x"], points["y"], points["z"])).T
        # rgb = np.vstack((points["r"], points["g"], points["b"])).T
        # use g channel to store label 
        g = points["g"].T
        num_class = len(coco_categories)
        # cvrt from 0 for background to -1 for background
        class_label = np.round(g * float(num_class + 1) / 255.0).astype(int) - 1
        # filter out background points 
        objects_mask = (class_label >= 0)
        if not np.any(objects_mask): # no object points in semantic mesh 
            # stop initialization with empty scene graph
            return 
        objects_xyz = xyz[objects_mask]
        objects_label = class_label[objects_mask]
        sem_points = np.concatenate(
            (objects_xyz, label_scale * objects_label.reshape(-1, 1)), axis=1)

        # cluster semantic point clouds to object clusters 
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, 
                    n_jobs=dbscan_num_processes).fit(sem_points)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        inst_labels = db.labels_
        object_ids = set(inst_labels)
        if dbscan_verbose:
            num_clusters = len(object_ids) - (1 if -1 in inst_labels else 0)
            num_noise = (inst_labels == -1).sum()
            print(f"DBSCAN on semantic point clouds, num_clusters ({num_clusters}), num_noise ({num_noise})")
        
        if dbscan_vis:
            pass
        
        # 3. non-maximum suppression: filter out noisy detection result 

        if nms:
            valid_object_ids = []
            valid_object_score_bboxes = [] # [p, x, y, z, l,w,h]
            for obj_id in object_ids:
                if obj_id == -1: # outliers
                    continue
                obj_xyz = objects_xyz[inst_labels == obj_id]
                if obj_xyz.shape[0] > min_points_filter:
                    label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                    # select mode label as object label
                    obj_label = label_modes[0]
                    if obj_label < len(label_mapping):
                        obj_cls_name = label_mapping[obj_label]
                        center = np.mean(obj_xyz, axis=0)
                        size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                        score_bbox = np.array([obj_xyz.shape[0], # num of points  
                                            center[0], center[1], center[2],
                                            size[0], size[1], size[2],
                                            ])
                        valid_object_ids.append(obj_id)
                        valid_object_score_bboxes.append(score_bbox)
            
            object_ids = valid_object_ids
            # there could be no valid objects founded 
            if len(valid_object_ids) > 0:
                valid_object_score_bboxes = np.stack(valid_object_score_bboxes, axis=0)
                selected_indices, _ = NMS(valid_object_score_bboxes, nms_th)
                object_ids = [valid_object_ids[idx] for idx in selected_indices]
                
        # 4. create object nodes in scene graph 
        
        for obj_id in object_ids:
            
            if obj_id == -1: # outliers
                continue
            
            obj_xyz = objects_xyz[inst_labels == obj_id]
            if obj_xyz.shape[0] > min_points_filter:
                label_modes, _ = stats.mode(objects_label[inst_labels == obj_id], nan_policy="omit")
                # select mode label as object label
                obj_label = label_modes[0]
                obj_cls_name = ""
                if obj_label >= 0:
                    obj_cls_name = label_mapping[obj_label]
                # else:
                #     obj_cls_name = "background"
                    # use axis-aligned bounding box for now 
                    center = np.mean(obj_xyz, axis=0)
                    rot_quat = np.array([0, 0, 0, 1])  # identity transform
                    size = np.max(obj_xyz, axis=0) - np.min(obj_xyz, axis=0)
                    if not self.point_features:
                        object_vertices = None
                    else:
                        object_vertices = obj_xyz
                        
                    object_node = self.object_layer.add_object(
                        center,
                        rot_quat,
                        size,
                        id=obj_id,
                        class_name=obj_cls_name,
                        label=obj_label,
                        vertices=object_vertices   
                    )

                # no region prediction module implemented 
                # connect object to region
                # region_node.add_object(object_node)

        return

    # def get_full_graph(self):
    #     """Return the full scene graph"""
    #     return None

    # def sample_graph(self, method, *args, **kwargs):
    #     """Return the sub-sampled scene graph"""
        
    #     return None



if __name__ == "__main__":
    # import importlib.util
    # sys.path.append(UTILS_DIR.parent.parent)
    # NOTE: run this demo with python -m agents.utils.sg_utils
    from utils.simulator import init_sim
    import rosbag
    from habitat_sim.utils.common import quat_from_two_vectors
    import matplotlib.pyplot as plt
    import open3d as o3d 
    
    TEST_SCENEGRAPH_SIMGT = False
    TEST_SCENEGRAPH_RTABMAP = True
    
    if TEST_SCENEGRAPH_SIMGT:
        test_scene = "/home/junting/Downloads/datasets/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
        sim = sim, action_names = init_sim(test_scene)
        gt_scenegraph = SceneGraphGTGibson(sim)
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
    
    if TEST_SCENEGRAPH_RTABMAP:
        
        cm_viridis = cm.get_cmap('viridis')
        obj_color_max = 50
        bag = rosbag.Bag("/home/junting/Downloads/dataset/rtabmap_sem_pcl/2022-07-26-10-46-44.bag","r")
        cloud_msgs = [cloud_msg for topic, cloud_msg, t in
            bag.read_messages(topics=["/rtabsem/cloud_map"])]
        bag.close()
        # initialize open3d visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        ctr = vis.get_view_control()
        ctr.set_zoom(4)
        o3d_pcl = o3d.geometry.PointCloud()
        vis.add_geometry(o3d_pcl)
        
        bboxes = []
        for cloud_msg in cloud_msgs:
        # for cloud_msg in [cloud_msgs[-3]]:
            
            sg = SceneGraphRtabmap(cloud_msg, point_features=True, 
                                   label_mapping=coco_label_mapping)

            print({obj_id:(sg.object_layer.obj_dict[obj_id].class_name, 
                           len(sg.object_layer.obj_dict[obj_id].vertices))
                   for obj_id in sg.object_layer.obj_ids 
                   if sg.object_layer.obj_dict[obj_id].class_name != "background"
                   and len(sg.object_layer.obj_dict[obj_id].vertices) > 10})

            for bbox in bboxes:
                vis.remove_geometry(bbox)
            
            bboxes = []
            points = []
            colors = []
            for obj_id in sg.object_layer.obj_ids:
                
                obj_node = sg.object_layer.obj_dict[obj_id]
                if (obj_node.class_name != "background" and 
                    len(sg.object_layer.obj_dict[obj_id].vertices)):
                    obj_color = np.array(cm_viridis((obj_id % obj_color_max)/obj_color_max))
                    
                    points.append(obj_node.vertices)
                    colors.append(np.repeat(obj_color[:3].reshape(1,3), 
                                            repeats=obj_node.vertices.shape[0], axis=0))
                    min_bound = obj_node.center - obj_node.size / 2.0
                    max_bound = obj_node.center + obj_node.size / 2.0
                    o3d_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    # o3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                    #     o3d.utility.Vector3dVector(obj_node.vertices)
                    # )
                    o3d_bbox.color = (0,1,0)
                    bboxes.append(o3d_bbox)
            
            if len(points) > 0:
                points = np.concatenate(points, axis=0)     
                colors = np.concatenate(colors, axis=0)
                o3d_pcl.points = o3d.utility.Vector3dVector(points)
                o3d_pcl.colors = o3d.utility.Vector3dVector(colors)
            else: 
                o3d_pcl.points = o3d.utility.Vector3dVector([])
                o3d_pcl.colors = o3d.utility.Vector3dVector([])
            
            for bbox in bboxes:
                vis.add_geometry(bbox)
            
            vis.update_geometry(o3d_pcl)
            # ctr.set_zoom(1)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.3)
        vis.destroy_window()
            
            
            
        
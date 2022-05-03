#!/usr/bin/env python
# python 3

import json
import os
import struct
from os.path import join

# common modules
import numpy as np
import open3d as o3d
import pandas as pd
import quaternion as qt

# ros modules
import rospy
import scipy.ndimage as ndimage
import sensor_msgs.point_cloud2 as pcl2
from dataset.habitat.simulator import init_sim
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from habitat_sim.agent.controls.controls import SceneNodeControl
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from matplotlib import cm
from numpy.core.defchararray import center
from scene_graph.config import SceneGraphHabitatConfig

# local import
from scene_graph.scene_graph_cls import SceneGraphHabitat
from scene_graph.scene_graph_pred import SceneGraphPredictor
from scene_graph.utils import visualize_scene_graph
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

# local modules


SCANNET20_Label_Names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


class VisSceneGraphNode:
    def __init__(self, node_name="vis_scene_graph_node"):
        rospy.init_node(node_name)
        self.node_name = node_name
        self.class_file = "/home/junting/project_cvl/SceneGraphNav/SSG/data/all_es/classes160.txt"
        self.relation_file = "/home/junting/project_cvl/SceneGraphNav/SSG/data/all_es//all_es/relationships.txt"
        self.annot_json = "/home/junting/project_cvl/SceneGraphNav/SSG/data/all_es//all_es/relationships_train.json"
        self.scan_dir = (
            "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans"
        )
        self.scene_name = "17DRP5sb8fy"
        self.frame_id = "map"
        self.text_z_shift = 0.2
        self.rel_dist_thresh = 2.0

        # visualize config
        self.object_elavate_node = 3.0
        self.object_elavate_text = 0.5
        self.room_elavate_node = 8.0
        self.room_elavate_text = 1.0

        self.scene_ply_path = os.path.join(
            self.scan_dir, self.scene_name, f"{self.scene_name}_semantic.ply"
        )
        self.scene_glb_path = os.path.join(
            self.scan_dir, self.scene_name, f"{self.scene_name}.glb"
        )
        self.pclseg_path = os.path.join(
            self.scan_dir, self.scene_name, f"{self.scene_name}_pclseg.txt"
        )
        self.pcl_normals_path = os.path.join(
            self.scan_dir, self.scene_name, f"{self.scene_name}_normals.npy"
        )

        self.topic_pcl2 = "/scene_graph/vis_scene_graph_node/pointclouds"
        self.topic_objects = "/scene_graph/vis_scene_graph_node/objects"
        self.topic_obj_names = "/scene_graph/vis_scene_graph_node/obj_names"
        self.topic_relations = "/scene_graph/vis_scene_graph_node/relations"
        self.topic_rel_names = "/scene_graph/vis_scene_graph_node/rel_names"

        with open(
            "/home/junting/panoptic_ws/src/scene_graph/src/SSG/data/all_es/relationships.txt",
            "r",
        ) as f:
            self.relation_names = f.read().splitlines()

        # publisher of pointclouds2
        self.pub_pcl2 = rospy.Publisher(
            self.topic_pcl2, PointCloud2, queue_size=5
        )
        self.pub_object_nodes = rospy.Publisher(
            self.topic_objects, MarkerArray, queue_size=1
        )
        self.pub_object_names = rospy.Publisher(
            self.topic_obj_names, MarkerArray, queue_size=1
        )
        self.pub_relations = rospy.Publisher(
            self.topic_relations, MarkerArray, queue_size=1
        )
        self.pub_rel_names = rospy.Publisher(
            self.topic_rel_names, MarkerArray, queue_size=1
        )

        # publisher flags
        self.flag_publish_pcls = True
        self.flag_publish_objects = True
        self.flag_publish_obj_names = True
        self.flag_publish_relations = True
        self.flag_publish_rel_names = True

        # advanced options
        self.rel_same_part = 7  # "same part" relation label

        # transformations
        quat_alignGravity = quat_from_two_vectors(
            np.array([0, 0, -1]), np.array([0, -1, 0])
        )
        self.r_m2h = qt.as_rotation_matrix(quat_alignGravity)
        self.r_h2m = np.linalg.inv(self.r_m2h)

    def load_data(self):

        ############ initialize habitat simulator and ground truth scene graph ########
        sim, action_names, sim_settings = init_sim(self.scene_glb_path)

        # intialize ground truth scene graph
        config = SceneGraphHabitatConfig()
        self.scene_graph = SceneGraphHabitat(
            config, scene_name=self.scene_name
        )
        self.scene_graph.load_gt_scene_graph(
            self.scene_ply_path, self.pclseg_path, self.pcl_normals_path, sim
        )

        ############ extract GCN features by pretrained 3DSSG model
        feature_extractor = SceneGraphPredictor(self.rel_dist_thresh)
        object_nodes = [
            self.scene_graph.object_layer.obj_dict[obj_id]
            for obj_id in self.scene_graph.object_layer.obj_ids
        ]

        """ 
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
        """
        self.results = feature_extractor.predict(object_nodes)

        points, pcl_seg = self.scene_graph.get_full_point_clouds()

        # create pointclouds2 message
        if self.flag_publish_pcls:
            self.pcl2_msg = self.create_pcl2_msg(points[:, :3], points[:, 3:6])

        return

    def create_pcl2_msg(self, pcls, colors, a=255):

        if not np.issubdtype(
            type(colors[0]), int
        ):  # convert to integer if colors in [0,1]
            colors = (colors * 255).astype(int)

        points = []
        for i in range(pcls.shape[0]):
            pcl = pcls[i]
            c = colors[i]
            rgb = struct.unpack("I", struct.pack("BBBB", c[2], c[1], c[0], a))[
                0
            ]

            pt = [pcl[0], pcl[1], pcl[2], rgb]
            points.append(pt)

            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("rgba", 12, PointField.UINT32, 1),
            ]

        header = Header()
        header.frame_id = self.frame_id
        pcl2_msg = pcl2.create_cloud(header, fields, points)

        return pcl2_msg

    def publish_objects(self, vis_mode="inplace", flag_vis_name=True):
        assert vis_mode in ["elavated", "inplace"]
        # with self.mesh_layer.mutex_lock:
        marker_arr = MarkerArray()
        if flag_vis_name:
            name_marker_arr = MarkerArray()
        for id, node in self.scene_graph.object_layer.obj_dict.items():

            center = self.r_h2m @ node.center
            # center = (np.max(pcls, axis=0) + np.max(pcls, axis=0))/2.0 # select center of bbox
            if vis_mode == "elavated":
                center = center + np.array(
                    [0.0, 0.0, self.object_elavate_node]
                )

            # if self.map_rawlabel2panoptic_id[obj.label] == 0:
            #     ns = "structure"
            #     scale=Vector3(0.5, 0.5, 0.5)
            #     color = ColorRGBA(1.0, 0.5, 0.0, 0.5)
            # else: # self.map_rawlabel2panoptic_id[obj.label] == 1
            ns = "object"
            scale = Vector3(0.2, 0.2, 0.2)
            color = ColorRGBA(0.0, 1.0, 0.5, 0.5)

            marker = Marker(
                type=Marker.CUBE,
                id=id,
                ns=ns,
                # lifetime=rospy.Duration(2),
                pose=Pose(Point(*center), Quaternion(0, 0, 0, 0)),
                scale=scale,
                header=Header(frame_id=self.frame_id),
                color=color,
                # text=submap.class_name,
            )

            marker_arr.markers.append(marker)

            if flag_vis_name:
                label = self.results["pred_obj_label"][id]
                text_pos = center + np.array(
                    [0.0, 0.0, self.object_elavate_text]
                )
                name_marker = Marker(
                    type=Marker.TEXT_VIEW_FACING,
                    id=id,
                    ns=ns,
                    # lifetime=rospy.Duration(2),
                    pose=Pose(Point(*text_pos), Quaternion(0, 0, 0, 0)),
                    scale=Vector3(0.2, 0.2, 0.2),
                    header=Header(frame_id=self.frame_id),
                    color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                    text=SCANNET20_Label_Names[label],
                )

                name_marker_arr.markers.append(name_marker)

        self.pub_object_nodes.publish(marker_arr)
        if flag_vis_name:
            self.pub_object_names.publish(name_marker_arr)

    def publish_relations(
        self, vis_mode="inplace", color_map="viridis", vis_name=True
    ):
        assert vis_mode in ["elavated", "inplace"]

        """ 
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
        """
        # with self.mesh_layer.mutex_lock:
        edges = self.results["edges"]
        relations = self.results["pred_rel_label"]

        marker_arr = MarkerArray()
        if vis_name:
            name_marker_arr = MarkerArray()
        color_map = cm.get_cmap(color_map)

        # colors = color_map(labels.astype(float) / max(self.room_layer.room_ids))
        object_nodes = [
            self.scene_graph.object_layer.obj_dict[obj_id]
            for obj_id in self.scene_graph.object_layer.obj_ids
        ]

        centers = [
            self.r_h2m @ object_node.center for object_node in object_nodes
        ]

        marker_arr = MarkerArray()

        for i, edge in enumerate(edges):
            # point_elavated = np.concatenate((room.pos[:2], [elavate_z]))
            rel = relations[i]
            # relation=8: none
            if rel >= len(self.relation_names):
                continue
            color = color_map(float(rel) / len(self.relation_names))
            if vis_mode == "elavated":
                start_center = centers[edge[0]] + np.array(
                    (0.0, 0.0, self.object_elavate_node)
                )
                end_center = centers[edge[1]] + np.array(
                    (0.0, 0.0, self.object_elavate_node)
                )
            else:  # "inplace":
                start_center = centers[edge[0]]
                end_center = centers[edge[1]]

            start_point = Point(*start_center)
            end_point = Point(*end_center)

            marker = Marker(
                type=Marker.ARROW,
                id=i,
                ns=self.relation_names[rel],
                lifetime=rospy.Duration(3),
                # pose=Pose(Point(), Quaternion(0,0,0,1)),
                scale=Vector3(0.05, 0.1, 0.1),
                header=Header(frame_id=self.frame_id),
                color=ColorRGBA(*color),
                points=[start_point, end_point],
            )

            if vis_name:
                text_pos = (start_center + end_center) / 2.0 + np.array(
                    (0, 0, self.object_elavate_text)
                )
                name_marker = Marker(
                    type=Marker.TEXT_VIEW_FACING,
                    id=i,
                    ns=self.relation_names[rel],
                    lifetime=rospy.Duration(3),
                    pose=Pose(Point(*text_pos), Quaternion(*(0, 0, 0, 1))),
                    scale=Vector3(0.3, 0.3, 0.3),
                    header=Header(frame_id=self.frame_id),
                    color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                    text=self.relation_names[rel],
                )

                name_marker_arr.markers.append(name_marker)

            marker_arr.markers.append(marker)

        self.pub_relations.publish(marker_arr)
        if vis_name:
            self.pub_rel_names.publish(name_marker_arr)
        return

    # def publish_free_space(self):

    def run(self):
        self.load_data()

        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown():
            # for msg in self.pcl2_msgs:
            #     self.pub_pcl2.publish(msg)
            if self.flag_publish_pcls:
                self.pub_pcl2.publish(self.pcl2_msg)
            self.publish_objects(vis_mode="inplace")
            self.publish_relations(vis_mode="inplace")

            rate.sleep()


if __name__ == "__main__":

    scene_graph_vis_node = VisSceneGraphNode()
    scene_graph_vis_node.run()

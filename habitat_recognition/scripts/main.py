#! /usr/bin/env python

import random
from habitat_recognition.scripts.models.HAIS.model.hais.hais import HAIS
import rospy
import numpy as np
import open3d as o3d
from std_msgs.msg import Int32, Header, ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3

from utils.simulator import init_sim
from utils.subscribers import PointCloudSubscriber
from utils.publishers import PointCloudPublisher
from utils.transformation import coo_rtab2mp3d

# local import
import os
import sys
from habitat_recognition.scripts.models.detector_votenet import DetectorVoteNet
from utils.vis_utils import semantic_scene_to_markerarray

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(ROOT_DIR, "models")
# sys.path.append(MODEL_DIR)
# from models.config_votenet import ConfigVoteNet
# from models.detector_votenet import DetectorVoteNet
from models.config_hais import ConfigHAIS
from models.segmenter_hais import SegmenterHAIS

DEFAULT_RATE = 1.0
DEFAULT_AGENT_TYPE = "random_walk"
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1
VISUALIZE = False
DEFAULT_RUN_MODE = "segment"  # "detect"


def main():
    # Initialize ROS node and take arguments
    rospy.init_node("habitat_recognition_node")
    # parameters
    node_start_time = rospy.Time.now().to_sec()
    rate_value = rospy.get_param("~rate", DEFAULT_RATE)
    agent_type = rospy.get_param("~agent_type", DEFAULT_AGENT_TYPE)
    goal_radius = rospy.get_param("~goal_radius", DEFAULT_GOAL_RADIUS)
    max_d_angle = rospy.get_param("~max_d_angle", DEFAULT_MAX_ANGLE)
    run_mode = rospy.get_param("~run_mode", DEFAULT_RUN_MODE)

    # topics
    rgb_topic = rospy.get_param("~rgb_topic", None)
    depth_topic = rospy.get_param("~depth_topic", None)
    camera_info_topic = rospy.get_param("~camera_info_topic", None)
    true_pose_topic = rospy.get_param("~true_pose_topic", None)
    true_pose_topic = None
    cloud_topic = rospy.get_param("~cloud_topic", "/rtabmap/cloud_map")
    camera_info_file = rospy.get_param("~camera_calib", None)

    # ros pub and sub
    rate = rospy.Rate(rate_value)
    sub_cloud = PointCloudSubscriber(cloud_topic)

    # Initialize hais
    # TODO: add model wrapper and use config to control model initialization
    config = ConfigHAIS()
    if run_mode == "segment":
        model = SegmenterHAIS(config)
    elif run_mode == "detect":
        # model = DetectorVoteNet
        raise NotImplementedError
    else:
        raise NotImplementedError

    # initialize visualization publishers
    pub_inst_pcd = PointCloudPublisher("~inst_pointcloud")
    pub_gt_bbox = rospy.Publisher("~gt_bbox", MarkerArray)
    # pub_pred_bbox = rospy.Publisher("~pred_bbox", MarkerArray)
    gt_bbox_markerarray = read_gt_bbox(house_file)

    cnt_sub = 0

    while not rospy.is_shutdown():

        if sub_cloud.has_cloud():
            o3d_rtab = sub_cloud.get_cloud()
            # transform point clouds to z-axis upright coords
            o3d_mp3d = coo_rtab2mp3d(o3d_rtab)

            # process point cloud
            result = model.inference(o3d_mp3d)
            model.evaluate(result)

            # color the point cloud by inst segmentation
            o3d_inst = create_inst_pcl(o3d_mp3d, result)
            # visualization
            pub_inst_pcd.publish_cloud(o3d_inst)
            pub_gt_bbox.publish(gt_bbox_markerarray)
            # pub_pred_bbox.publish()

            cnt_sub += 1
            print(f"publish {cnt_sub} messages")

        rate.sleep()


if __name__ == "__main__":
    main()

#! /usr/bin/env python
import os
import sys

import numpy as np
import open3d as o3d
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from publishers import PointCloudPublisher

# local import
from simulator import init_sim
from subscribers import PointCloudSubscriber
from utils.transformation import o3d_rtab2mp3d

# from models.detector_votenet import DetectorVoteNet
from utils.vis_utils import create_inst_pcl, semantic_scene_to_markerarray

# from std_msgs.msg import Int32, Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(ROOT_DIR, "models")
# sys.path.append(MODEL_DIR)
# from models.config_votenet import ConfigVoteNet
# from models.detector_votenet import DetectorVoteNet
from config.config_hais import ConfigHAIS
from models.segmenter_hais import SegmenterHAIS

DEFAULT_RATE = 10.0
DEFAULT_AGENT_TYPE = "random_walk"
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1
VISUALIZE = False
DEFAULT_RUN_MODE = "segment"  # "detect"

# DEBUG utilities
DEBUG_SAVE_PCL = False
DUMP_DIR = "/home/junting/Downloads/dataset/rtabmap_dump_0_015"


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
    test_scene = rospy.get_param("~test_scene", None)
    scene_name = "Unknown"
    if test_scene:
        scene_name = os.path.split(os.path.split(test_scene)[0])[1]

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
    # gt_bbox_markerarray = read_gt_bbox(house_file)

    cnt_sub = 0

    while not rospy.is_shutdown():

        if sub_cloud.has_cloud():
            o3d_rtab = sub_cloud.get_cloud()
            # transform point clouds to z-axis upright coords
            o3d_mp3d = o3d_rtab2mp3d(o3d_rtab)
            if DEBUG_SAVE_PCL:
                if not os.path.exists(DUMP_DIR):
                    os.makedirs(DUMP_DIR)
                num_points = np.asarray(o3d_mp3d.points).shape[0]
                o3d.io.write_point_cloud(
                    os.path.join(
                        DUMP_DIR, f"{scene_name}_dump_{num_points}.ply"
                    ),
                    o3d_mp3d,
                )
                print(
                    "save point cloud at",
                    os.path.join(
                        DUMP_DIR, f"{scene_name}_dump_{num_points}.ply"
                    ),
                )
            # process point cloud
            # there might be down sampling procedure
            try:
                o3d_mp3d, result = model.predict(o3d_mp3d)

                # color the point cloud by inst segmentation
                o3d_inst = create_inst_pcl(o3d_mp3d, result)

                # visualization
                pub_inst_pcd.publish_cloud(o3d_inst)
                # pub_gt_bbox.publish(gt_bbox_markerarray)
                # pub_pred_bbox.publish()

                cnt_sub += 1
                print(f"publish {cnt_sub} messages")

            except:
                print("Unreliable prediction on partial observations!")

        rate.sleep()


if __name__ == "__main__":
    main()

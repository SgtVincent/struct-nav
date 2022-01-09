#! /usr/bin/env python

import random
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
from detector_votenet import DetectorVoteNet
from utils.vis_utils import semantic_scene_to_markerarray

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
sys.path.append(MODEL_DIR)
from config_votenet import ConfigVoteNet 

DEFAULT_RATE = 1.0
DEFAULT_AGENT_TYPE = 'random_walk'
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1
VISUALIZE = False


def main():
    # Initialize ROS node and take arguments
    rospy.init_node('habitat_detect_segment_node')
    node_start_time = rospy.Time.now().to_sec()
    rate_value = rospy.get_param('~rate', DEFAULT_RATE)
    agent_type = rospy.get_param('~agent_type', DEFAULT_AGENT_TYPE)
    goal_radius = rospy.get_param('~goal_radius', DEFAULT_GOAL_RADIUS)
    max_d_angle = rospy.get_param('~max_d_angle', DEFAULT_MAX_ANGLE)
    rgb_topic = rospy.get_param('~rgb_topic', None)
    depth_topic = rospy.get_param('~depth_topic', None)
    camera_info_topic = rospy.get_param('~camera_info_topic', None)
    true_pose_topic = rospy.get_param('~true_pose_topic', None)
    true_pose_topic = None
    cloud_topic = rospy.get_param('~cloud_topic', None)
    camera_info_file = rospy.get_param('~camera_calib', None)

    # ros pub and sub
    rate = rospy.Rate(rate_value)
    sub_cloud = PointCloudSubscriber(cloud_topic)
    # TODO: add sub to rgbd

    # Initial Sim
    test_scene= rospy.get_param('~test_scene', None)
    sim, action_names = init_sim(test_scene)
    # Run the simulator with agent
    observations = sim.reset()
    semantic_scene = sim.semantic_scene
    # observations = sim.step('stay')

    # Initialize detector
    config = ConfigVoteNet()
    config.semantic_scene = semantic_scene
    detector = DetectorVoteNet(config, device="cuda")
    
    # initialize visualization publishers 
    pub_rtab_pcd = PointCloudPublisher('~rtab_pointcloud')
    pub_gt_bbox = rospy.Publisher('~gt_bbox', MarkerArray)
    pub_pred_bbox = rospy.Publisher('~pred_bbox', MarkerArray)
    gt_bbox_markerarray = semantic_scene_to_markerarray(semantic_scene)
    
    # close simulator after gt data fetched
    sim.close()
    cnt_sub = 0

    while not rospy.is_shutdown():

        if sub_cloud.has_cloud():
            o3d_rtab = sub_cloud.get_cloud()
            # transform point clouds to z-axis upright coords
            o3d_mp3d = coo_rtab2mp3d(o3d_rtab)
            
            # process point cloud
            detect_result = detector.detect(o3d_mp3d)
            detector.evaluate(detect_result)
            
            # visualization
            pub_rtab_pcd.publish_cloud(o3d_mp3d)
            pub_gt_bbox.publish(gt_bbox_markerarray)
            # pub_pred_bbox.publish()
            
            cnt_sub += 1
            print(f"publish {cnt_sub} messages")

        rate.sleep()



if __name__ == '__main__':
    main()

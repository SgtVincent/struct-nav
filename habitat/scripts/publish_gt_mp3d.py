#! /usr/bin/env python
import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sys

sys.path.append(".")
sys.path.append("..")
from utils.lib_cloud_conversion_between_open3d_and_ros import cloud_o3d2ros


if __name__ == "__main__":
    ply_path = "/home/junting/Downloads/datasets/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy_semantic.ply"
    topic_name = "/gt/mp3d_pointcloud"
    o3d_pcl = o3d.io.read_point_cloud(ply_path)

    # init ros
    rospy.init_node("publish_gt_mp3d")

    # pts, rgb = np.asarray(o3d_pcl.points), np.asarray(o3d_pcl.colors)
    ros_pcl = cloud_o3d2ros(o3d_pcl, frame_id="map")
    pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)

    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        pub.publish(ros_pcl)
        rate.sleep()

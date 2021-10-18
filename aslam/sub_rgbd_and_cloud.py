#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script gives exampe code for subscribing.

(1) Color image.
(2) Depth image.
(3) Camera info.
(4) Point cloud (subscribed as open3d format).
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rospy

from .utils.lib_rgbd import MyCameraInfo
from .utils.lib_ros_point_cloud_pub_and_sub import PointCloudSubscriber
from .utils.lib_ros_rgbd_pub_and_sub import (
    CameraInfoSubscriber,
    ColorImageSubscriber,
    DepthImageSubscriber,
)

# -- Set ROS topic names for subscribing.
NS = "aslam/"  # ROS topic namespace.
COLOR_TOPIC_NAME = NS + "color"
DEPTH_TOPIC_NAME = NS + "depth"
CAMERA_INFO_TOPIC_NAME = NS + "camera_info"
CLOUD_TOPIC_NAME = NS + "point_cloud"

# -- Subscribe data and print.
def main():
    """Main func for subscribers."""
    # -- Set subscribers.
    sub_color = ColorImageSubscriber(COLOR_TOPIC_NAME, img_format="rgb8")
    sub_depth = DepthImageSubscriber(DEPTH_TOPIC_NAME)
    sub_camera_info = CameraInfoSubscriber(CAMERA_INFO_TOPIC_NAME)
    sub_cloud = PointCloudSubscriber(CLOUD_TOPIC_NAME)

    # -- RGB-D image visualization.
    plt.ion()
    fig = plt.figure()

    # -- cloud map visualization.
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window()
    last_geometry = None

    # -- Loop and subscribe.
    cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0  # color, depth, camera_info, cloud
    while not rospy.is_shutdown():

        # Color.
        if sub_color.has_image():
            color = sub_color.get_image()
            cnt_1 += 1
            rospy.loginfo(
                "Subscribe {}: color image, "
                "shape={}".format(cnt_1, color.shape)
            )

            # -- Visualize RGB images
            rgb_plt = fig.add_subplot(1, 2, 1)
            rgb_plt.set_title("Color image")
            plt.imshow(color)

        # Depth.
        if sub_depth.has_image():
            depth = sub_depth.get_image()
            cnt_2 += 1
            rospy.loginfo(
                "Subscribe {}: depth image, "
                "shape={}".format(cnt_2, depth.shape)
            )

            # -- Visualize depth images
            depth_plt = fig.add_subplot(1, 2, 2)
            depth_plt.set_title("Depth image")
            plt.imshow(depth)
            plt.pause(0.001)

        # Camera_info.
        if sub_camera_info.has_camera_info():
            ros_camera_info = sub_camera_info.get_camera_info()
            cnt_3 += 1
            rospy.loginfo(
                "Subscribe {}: camera_info, "
                "fx={}, fy={}.".format(
                    cnt_3,
                    ros_camera_info.K[0],
                    ros_camera_info.K[4],
                )
            )
            my_camera_info = MyCameraInfo(ros_camera_info=ros_camera_info)
            print(my_camera_info)

        # Point_cloud.
        if sub_cloud.has_cloud():
            open3d_cloud = sub_cloud.get_cloud()

            # -- Visualize o3d point clouds.
            if last_geometry is not None:
                o3d_vis.remove_geometry(last_geometry)
            open3d_cloud.transform(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )
            o3d_vis.add_geometry(open3d_cloud)
            last_geometry = open3d_cloud
            o3d_vis.poll_events()
            o3d_vis.update_renderer()

            cnt_4 += 1
            num_points = np.asarray(open3d_cloud.points).shape[0]
            rospy.loginfo(
                "Subscribe {}: point cloud, "
                "{} points.".format(cnt_4, num_points)
            )

        rospy.sleep(0.1)

    o3d_vis.destroy_window()


if __name__ == "__main__":
    NODE_NAME = "sub_rgbd_and_cloud"
    rospy.init_node(NODE_NAME)
    main()
    rospy.logwarn("Node `{}` stops.".format(NODE_NAME))

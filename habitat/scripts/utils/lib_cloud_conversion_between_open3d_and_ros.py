#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains 2 functions for converting cloud format \
    between Open3D and ROS.

* cloud_o3d2ros
* cloud_ros2o3d
where the ROS format refers to "sensor_msgs/PointCloud2.msg" type.

This script also contains a test case, which does such a thing:
(1) Read a open3d_cloud from .pcd file by Open3D.
(2) Convert it to ros_cloud.
(3) Publish ros_cloud to topic.
(4) Subscribe the ros_cloud from the same topic.
(5) Convert ros_cloud back to open3d_cloud.
(6) Display it.
You can test this script's function by rosrun this script.

"""

from ctypes import (  # convert float to uint32
    POINTER,
    c_float,
    c_uint32,
    cast,
    pointer,
)

import numpy as np
import open3d
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

# The data structure of each point in ros PointCloud2:
#    16 bits = x + y + z + rgb

FIELDS_XYZ = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + [
    PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00FF0000) >> 16,
    (rgb_uint32 & 0x0000FF00) >> 8,
    (rgb_uint32 & 0x000000FF),
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

# NOTE: unknown error: required argument is not an integer
# def cloud_o3d2ros(open3d_cloud, frame_id="base", stamp=None):
#     """Open3D point cloud to ROS PointCloud2 (XYZRGB only)."""
#     # Set "header"
#     header = Header()
#     if stamp is None:
#         header.stamp = rospy.Time.now()
#     else:
#         header.stamp = stamp
#     header.frame_id = frame_id

#     # Set "fields" and "cloud_data"
#     points = np.asarray(open3d_cloud.points)
#     if not open3d_cloud.colors:  # XYZ only
#         fields = FIELDS_XYZ
#         cloud_data = points
#     else:  # XYZ + RGB
#         fields = FIELDS_XYZRGB
#         # -- Change rgb color from "three float" to "one 24-byte int"
#         # 0x00FFFFFF is white, 0x00000000 is black.
#         colors = np.floor(np.asarray(open3d_cloud.colors) * 255)  # nx3 matrix
#         colors = (
#             colors[:, 0] * BIT_MOVE_16
#             + colors[:, 1] * BIT_MOVE_8
#             + colors[:, 2]
#         )
#         cloud_data = np.c_[points, colors]

#     # create ros_cloud
#     return pc2.create_cloud(header, fields, cloud_data)


def cloud_o3d2ros(open3d_cloud, frame_id="map"):

    header = Header(frame_id=frame_id)

    # Set "fields" and "cloud_data"
    points = np.asarray(open3d_cloud.points, dtype=np.float32)
    if not open3d_cloud.colors:  # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else:  # XYZ + RGB
        fields = FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors) * 255).astype(
            np.uint32
        )  # nx3 matrix
        cloud_data = []
        for i in range(points.shape[0]):
            xyz = points[i]
            c = colors[i]
            rgba = struct.unpack(
                "I", struct.pack("BBBB", c[2], c[1], c[0], 255)
            )[0]

            pt = [xyz[0], xyz[1], xyz[2], rgba]
            cloud_data.append(pt)

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)


def cloud_ros2o3d(ros_cloud):
    """Get cloud data from ros_cloud."""
    field_names = [field.name for field in ros_cloud.fields]
    cloud_data = list(
        pc2.read_points(ros_cloud, skip_nans=True, field_names=field_names)
    )

    # Check empty
    open3d_cloud = open3d.geometry.PointCloud()
    if len(cloud_data) == 0:
        print("Converting an empty cloud")
        return None

    # Set open3d_cloud
    if "rgb" in field_names:
        idx_rgb_in_field = 3  # x, y, z, rgb

        # Get xyz
        xyz = [(x, y, z) for x, y, z, rgb in cloud_data]

        # Get rgb
        # Check whether int or float
        if isinstance(
            cloud_data[0][idx_rgb_in_field], float
        ):  # if float (from pcl::toROSMsg)
            rgb = [
                convert_rgbFloat_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]
        else:
            rgb = [
                convert_rgbUint32_to_tuple(rgb) for x, y, z, rgb in cloud_data
            ]

        # combine
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
        open3d_cloud.colors = open3d.utility.Vector3dVector(
            np.array(rgb) / 255.0
        )
    else:
        xyz = list(cloud_data)  # get xyz
        open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

    # return
    return open3d_cloud


# -- Example of usage
if __name__ == "__main__":
    rospy.init_node(
        "test_pc_conversion_between_Open3D_and_ROS", anonymous=True
    )

    # -- Read point cloud from file
    import os

    PYTHON_FILE_PATH = os.path.join(os.path.dirname(__file__)) + "/"
    # test XYZ point cloud format
    # filename = PYTHON_FILE_PATH + "test_cloud_XYZ_noRGB.pcd"
    # test XYZRGB point cloud format
    filename = PYTHON_FILE_PATH + "test_cloud_XYZRGB.pcd"

    o3d_cloud = open3d.read_point_cloud(filename)
    rospy.loginfo("Loading cloud from file by open3d.read_point_cloud: ")
    print(o3d_cloud)
    print("")

    # -- Set publisher
    TOPIC_NAME = "camera/depth_registered/points"
    pub = rospy.Publisher(TOPIC_NAME, PointCloud2, queue_size=1)

    # -- Set subscriber
    # RECEIVED_ROS_CLOUD = None

    # def callback(ros_cloud):
    # """Callback for ros cloud."""
    # global RECEIVED_ROS_CLOUD
    # RECEIVED_ROS_CLOUD = ros_cloud
    # rospy.loginfo("-- Received ROS PointCloud2 message.")

    # rospy.Subscriber(TOPIC_NAME, PointCloud2, callback)

    class PointCloudSubscriber:
        """Simle point cloud scriber."""

        def __init__(self, topic_name):
            """Set point cloud scriber."""
            self._sub = rospy.Subscriber(
                topic_name, PointCloud2, self.callback
            )
            self.received_ros_cloud = None

        def callback(self, ros_cloud):
            """Callback for sub."""
            self.received_ros_cloud = ros_cloud
            rospy.loginfo("-- Received ROS PointCloud2 message.")

    my_pc_sub = PointCloudSubscriber(TOPIC_NAME)
    received_ros_cloud = my_pc_sub.received_ros_cloud

    # -- Convert open3d_cloud to ros_cloud, and publish.
    # -- Until the subscribe receives it.
    while received_ros_cloud is None and not rospy.is_shutdown():
        rospy.loginfo("-- Not receiving ROS PointCloud2 message yet ...")
        CLOUD_FROM_FILE = True

        if CLOUD_FROM_FILE:  # Use the cloud from file
            rospy.loginfo(
                "Converting cloud from Open3d to ROS PointCloud2 ..."
            )
            my_ros_cloud = cloud_o3d2ros(o3d_cloud)

        else:  # Use the cloud with 3 points generated below
            rospy.loginfo(
                "Converting a 3-point cloud into ROS PointCloud2 ..."
            )
            TEST_CLOUD_POINTS = [
                [1.0, 0.0, 0.0, 0xFF0000],
                [0.0, 1.0, 0.0, 0x00FF00],
                [0.0, 0.0, 1.0, 0x0000FF],
            ]
            my_ros_cloud = pc2.create_cloud(
                Header(frame_id="base"), FIELDS_XYZ, TEST_CLOUD_POINTS
            )

        # publish cloud
        pub.publish(my_ros_cloud)
        rospy.loginfo("Conversion and publish success ...\n")
        rospy.sleep(1)

    # -- After subscribing the ros cloud, convert it back to open3d, and draw
    received_open3d_cloud = cloud_ros2o3d(received_ros_cloud)
    print(received_open3d_cloud)

    # write to file
    output_filename = PYTHON_FILE_PATH + "conversion_result.pcd"
    open3d.write_point_cloud(output_filename, received_open3d_cloud)
    rospy.loginfo("-- Write result point cloud to: " + output_filename)

    # draw
    open3d.visualization.draw_geometries([received_open3d_cloud])
    rospy.loginfo("-- Finish display. The program is terminating ...\n")

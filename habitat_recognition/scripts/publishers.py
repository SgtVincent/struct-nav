"""Publisher helpers."""

import numpy as np
import rospy
import yaml
import ros_numpy
import open3d as o3d
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import struct

DEPTH_SCALE = 1

FIELDS_XYZ = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + [
    PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1)
]

# Bit operations
BIT_MOVE_16 = 2 ** 16
BIT_MOVE_8 = 2 ** 8


class PointCloudPublisher:
    """Publisher for point cloud."""

    def __init__(self, topic_name, queue_size=2):
        """Set point cloud publisher"""
        self.pub = rospy.Publisher(
            topic_name, PointCloud2, queue_size=queue_size
        )

    def publish_cloud(self, open3d_cloud):
        ros_cloud = self.cloud_o3d2ros(open3d_cloud)
        self.pub.publish(ros_cloud)
        return

    def cloud_o3d2ros(self, open3d_cloud, frame_id="map"):

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

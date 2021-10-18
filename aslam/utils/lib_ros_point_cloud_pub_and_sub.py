"""Publisher/Subscriber for point cloud."""

import queue

import rospy
from sensor_msgs.msg import PointCloud2

from .lib_cloud_conversion_between_open3d_and_ros import (
    cloud_o3d2ros,
    cloud_ros2o3d,
)


class PointCloudPublisher:
    """Publisher for point cloud."""

    def __init__(self, topic_name):
        """Set point cloud publisher."""
        self._pub = rospy.Publisher(topic_name, PointCloud2, queue_size=5)

    def publish(self, cloud, cloud_format="open3d", frame_id="head_camera"):
        """Publish point cloud."""
        if cloud_format == "open3d":
            cloud = cloud_o3d2ros(cloud, frame_id)
        else:  # ROS cloud: Do nothing.
            pass
        self._pub.publish(cloud)


class PointCloudSubscriber:
    """Subscriber for point cloud."""

    def __init__(self, topic_name, queue_size=2):
        """Set point cloud subscriber."""
        self._sub = rospy.Subscriber(
            topic_name, PointCloud2, self._callback_of_pcd_subscriber
        )
        self._clouds_queue = queue.Queue(maxsize=queue_size)

    def get_cloud(self):
        """Get the next cloud subscribed from ROS topic. \
        Convert it to open3d format and then return."""
        if not self.has_cloud():
            return None
        ros_cloud = self._clouds_queue.get(timeout=0.05)
        open3d_cloud = cloud_ros2o3d(ros_cloud)
        return open3d_cloud

    def has_cloud(self):
        """Has cloud in queue or not."""
        return self._clouds_queue.qsize() > 0

    def _callback_of_pcd_subscriber(self, ros_cloud):
        """Save the received point cloud into queue."""
        if self._clouds_queue.full():  # If queue is full, pop one.
            self._clouds_queue.get(timeout=0.001)
        self._clouds_queue.put(ros_cloud, timeout=0.001)  # Push cloud to queue

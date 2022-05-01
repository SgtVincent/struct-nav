import queue

import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import open3d as o3d

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
        open3d_cloud = self.cloud_ros2o3d(ros_cloud)
        return open3d_cloud

    def has_cloud(self):
        """Has cloud in queue or not."""
        return self._clouds_queue.qsize() > 0

    def _callback_of_pcd_subscriber(self, ros_cloud):
        """Save the received point cloud into queue."""
        if self._clouds_queue.full():  # If queue is full, pop one.
            self._clouds_queue.get(timeout=0.001)
        self._clouds_queue.put(ros_cloud, timeout=0.001)  # Push cloud to queue
    
    def cloud_ros2o3d(self, ros_cloud):
        field_names = [field.name for field in ros_cloud.fields]
        open3d_cloud = o3d.geometry.PointCloud()
        if "rgb" in field_names:
            points = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)
            points = ros_numpy.point_cloud2.split_rgb_field(points)
            xyz = np.vstack((points['x'], points['y'], points['z'])).T
            rgb = np.vstack((points['r'], points['g'], points['b'])).T
             # combine
            open3d_cloud.points = o3d.utility.Vector3dVector(xyz)
            open3d_cloud.colors = o3d.utility.Vector3dVector(
                rgb / 255.0
            )
        else:
            xyz = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(ros_cloud)
            open3d_cloud.points = o3d.utility.Vector3dVector(xyz)
        
        return open3d_cloud



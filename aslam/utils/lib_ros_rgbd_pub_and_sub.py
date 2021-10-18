#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RGBD image publisher and subscriber.

Publisher:
    class ColorImagePublisher, publish()
    class DepthImagePublisher, publish()
    class CameraInfoPublisher, publish()

Subscriber:
    class ColorImageSubscriber, has_image() & get_image()
    class DepthImageSubscriber, has_image() & get_image()

"""

import queue
from abc import ABCMeta, abstractmethod  # Abstract class.

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image  # This is ROS camera info.
from std_msgs.msg import Header

from .lib_commons import get_intrinsic_mat

# See: https: // stackoverflow.com/questions/13646245/
# is-it-possible-to-make-abstract-classes-in-python

# ========= Helper Functions ==========


def create_header(frame_id):
    """Helper for header."""
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    return header


class AbstractImagePublisher:
    """Base class for image publisher."""

    __metaclass__ = ABCMeta

    def __init__(self, image_topic, queue_size=10):
        """Initialize publisher and cv bridge."""
        self._pub = rospy.Publisher(image_topic, Image, queue_size=queue_size)
        self._cv_bridge = CvBridge()

    def publish(self, image, frame_id="head_camera"):
        """Publish ros images."""
        ros_image = self._to_ros_image(image)
        ros_image.header = create_header(frame_id)
        self._pub.publish(ros_image)

    @abstractmethod
    def _to_ros_image(self, image, **kwargs):
        """To ros image."""


class ColorImagePublisher(AbstractImagePublisher):
    """Publisher for color images."""

    def _to_ros_image(self, image, **kwargs):
        # -- Check input.
        img_format = kwargs.get("img_format", "bgr")
        cv2_uint8_image = image
        shape = cv2_uint8_image.shape  # (row, col, depth=3)
        assert len(shape) == 3 and shape[2] == 3

        # -- Convert image to bgr format.
        if img_format == "rgb":  # If rgb, convert to bgr.
            bgr_image = cv2.cvtColor(cv2_uint8_image, cv2.COLOR_RGB2BGR)
        elif img_format == "bgr":
            bgr_image = cv2_uint8_image
        else:
            raise RuntimeError("Wrong image format: " + img_format)

        # -- Convert to ROS format.
        ros_image = self._cv_bridge.cv2_to_imgmsg(bgr_image, "bgr8")
        return ros_image


class DepthImagePublisher(AbstractImagePublisher):
    """Publisher for depth images."""

    def _to_ros_image(self, image, **kwargs):
        # -- Check input.
        cv2_uint16_image = image
        shape = cv2_uint16_image.shape  # (row, col)
        assert len(shape) == 2
        assert isinstance(cv2_uint16_image[0, 0], np.uint16)

        # -- Convert to ROS format.
        ros_image = self._cv_bridge.cv2_to_imgmsg(cv2_uint16_image, "16UC1")
        return ros_image


class CameraInfoPublisher:
    """Publish image size and camera instrinsics to ROS CameraInfo topic.

    The distortion is not considered.
    """

    def __init__(self, topic_name):
        """Set publisher."""
        self._pub = rospy.Publisher(topic_name, CameraInfo, queue_size=5)

        # Create default camera info:
        camera_info = CameraInfo()  # This is ROS camera info. Not mine.
        camera_info.distortion_model = "plumb_bob"
        camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._default_camera_info = camera_info

    def publish(self, width, height, intrinsic_matrix, frame_id="head_camera"):
        """Publish camera_info constructed by: \
                (1) width \
                (2) height \
                (3) intrinsic_matrix.

        Args:
            width (int): Width.
            height (int): Height.
            intrinsic_matrix (List): If 1D list, \
                the data order is: column1, column2, column3. \
                (But ROS is row majored).
            frame_id (str): ID of frame.
        """
        # -- Set camera info.
        camera_info = self._default_camera_info
        camera_info.header = create_header(frame_id)
        self._set_size_and_intrinsics(
            camera_info, width, height, intrinsic_matrix
        )

        # -- Publish.
        self._pub.publish(camera_info)

    def publish_open3d_format_intrinsics(
        self, open3d_camera_intrinsic, frame_id="head_camera"
    ):
        """Publish camera intrinsic in o3d format."""
        width = open3d_camera_intrinsic.width
        height = open3d_camera_intrinsic.height
        k_mat = open3d_camera_intrinsic.intrinsic_matrix
        self.publish(width, height, k_mat, frame_id)

    def publish_ros_format_camera_info(self, camera_info):
        """Argument: camera_info {sensor_msgs.msg.CameraInfo}."""
        self._pub.publish(camera_info)

    @staticmethod
    def _2d_array_to_list(intrinsic_matrix):
        """Convert 2d arr to list."""
        res = []
        for i in range(3):
            for j in range(3):
                res.append(intrinsic_matrix[i][j])
        return res

    def _set_size_and_intrinsics(
        self, camera_info, width, height, intrinsic_matrix
    ):
        """Set size and camera intrinsics."""
        camera_info.height = height
        camera_info.width = width
        if isinstance(intrinsic_matrix, list):
            k_mat = intrinsic_matrix  # column majored --> row majored.
            k_mat = get_intrinsic_mat(k_mat)
        else:  # row majored.
            k_mat = self._2d_array_to_list(intrinsic_matrix)
        camera_info.K = k_mat
        camera_info.P = [
            k_mat[0],
            k_mat[1],
            k_mat[2],
            0.0,
            k_mat[3],
            k_mat[4],
            k_mat[5],
            0.0,
            k_mat[6],
            k_mat[7],
            k_mat[8],
            0.0,
        ]


# ========== Subscriber ==========


class AbstractImageSubscriber:
    """Base class for image subscriber."""

    __metaclass__ = ABCMeta

    def __init__(self, topic_name, queue_size=2):
        """Set subcriber, cv bridge and image queue."""
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber(
            topic_name, Image, self._callback_of_image_subscriber
        )
        self._imgs_queue = queue.Queue(maxsize=queue_size)

    @abstractmethod
    def _convert_ros_image_to_desired_image_format(self, ros_image):
        """To implement."""

    def get_image(self):
        """Get the next image subscribed from ROS topic, \
        convert to desired opencv format, and then return."""
        if not self.has_image():
            raise RuntimeError("Failed to get_image().")
        ros_image = self._imgs_queue.get(timeout=0.05)
        dst_image = self._convert_ros_image_to_desired_image_format(ros_image)
        return dst_image

    def has_image(self):
        """Has image or not."""
        return self._imgs_queue.qsize() > 0

    def _callback_of_image_subscriber(self, ros_image):
        """Save the received image into queue."""
        if self._imgs_queue.full():  # If queue is full, pop one.
            self._imgs_queue.get(timeout=0.001)
        self._imgs_queue.put(ros_image, timeout=0.001)  # Push image to queue


class ColorImageSubscriber(AbstractImageSubscriber):
    """RGB image subscriber."""

    def __init__(self, topic_name, queue_size=2, img_format="bgr8"):
        """Set image subscriber."""
        super().__init__(topic_name, queue_size)
        self._img_format = img_format

    def _convert_ros_image_to_desired_image_format(self, ros_image):
        """To np.ndarray np.uint8 BGR format."""
        return self._cv_bridge.imgmsg_to_cv2(ros_image, self._img_format)


class DepthImageSubscriber(AbstractImageSubscriber):
    """Depth image subscriber."""

    def _convert_ros_image_to_desired_image_format(self, ros_image):
        """To np.ndarray np.uint16 format."""
        return self._cv_bridge.imgmsg_to_cv2(ros_image, "16UC1")  # not 32FC1


class CameraInfoSubscriber:
    """Subscriber of camera info."""

    def __init__(self, topic_name):
        """Set camera info subscriber."""
        self._camera_info = None
        self._sub = rospy.Subscriber(topic_name, CameraInfo, self._callback)

    def get_camera_info(self):
        """Get camera info."""
        if self._camera_info is None:
            raise RuntimeError("Failed to get_camera_info().")
        camera_info = self._camera_info
        self._camera_info = None  # Reset data.
        return camera_info

    def has_camera_info(self):
        """Has camer info or not."""
        return self._camera_info is not None

    def _callback(self, camera_info):
        """Call back for camera info."""
        self._camera_info = camera_info

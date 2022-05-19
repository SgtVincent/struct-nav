"""Publisher helpers."""

import numpy as np
import rospy
import transformations as tf
import yaml
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image

DEPTH_SCALE = 1


def get_camera_info(filepath):
    """Get camera information from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    width = yaml_data["image_width"]
    height = yaml_data["image_height"]
    d = yaml_data["distortion_coefficients"]["data"]
    k = yaml_data["camera_matrix"]["data"]
    r = yaml_data["rectification_matrix"]["data"]
    p = yaml_data["projection_matrix"]["data"]
    return CameraInfo(width=width, height=height, D=d, K=k, R=r, P=p)


class HabitatObservationPublisher:
    """Publisher for observation of habitat."""

    def __init__(
        self,
        rgb_topic="",
        depth_topic="",
        semantic_topic="",
        camera_info_topic="",
        true_pose_topic="",
        camera_info_file="",
    ):
        """Initialize publisher with topic handles."""
        self.cvbridge = CvBridge()

        # Initialize camera info publisher.
        if len(camera_info_topic) > 0:
            self.publish_camera_info = True
            self.camera_info_publisher = rospy.Publisher(
                camera_info_topic, CameraInfo, latch=True, queue_size=100
            )
            self.camera_info = get_camera_info(camera_info_file)
        else:
            self.publish_camera_info = False

        # Initialize RGB image publisher.
        if len(rgb_topic) > 0:
            self.publish_rgb = True
            self.image_publisher = rospy.Publisher(
                rgb_topic, Image, latch=True, queue_size=100
            )
        else:
            self.publish_rgb = False

        # Initialize depth image publisher.
        if len(depth_topic) > 0:
            self.publish_depth = True
            self.depth_publisher = rospy.Publisher(
                depth_topic, Image, latch=True, queue_size=100
            )
        else:
            self.publish_depth = False

        # Initialize semantic image publisher
        if len(semantic_topic) > 0:
            self.publish_semantic = True
            self.semantic_publisher = rospy.Publisher(
                semantic_topic, Image, latch=True, queue_size=100
            )
        else:
            self.publish_semantic = False

        # Initialize position publisher.
        # if len(true_pose_topic) > 0:
        if False:
            self.publish_true_pose = True
            self.pose_publisher = rospy.Publisher(
                true_pose_topic, PoseStamped, latch=True, queue_size=100
            )
        else:
            self.publish_true_pose = False

    def publish(self, observations):
        """Publish messages."""
        cur_time = rospy.Time.now()

        # Publish RGB image.
        if self.publish_rgb:
            # self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb'])
            rgb_msg = self.cvbridge.cv2_to_imgmsg(
                observations["rgb"][:, :, 0:3]
            )
            rgb_msg.encoding = "rgb8"
            rgb_msg.header.stamp = cur_time
            rgb_msg.header.frame_id = "camera_link"
            self.image_publisher.publish(rgb_msg)

        # Publish depth image.
        if self.publish_depth:
            depth_msg = self.cvbridge.cv2_to_imgmsg(
                observations["depth"] * DEPTH_SCALE
            )
            depth_msg.header.stamp = cur_time
            depth_msg.header.frame_id = "base_scan"
            self.depth_publisher.publish(depth_msg)

        if self.publish_semantic:
            semantic = observations["semantic"]
            assert np.max(semantic) < 256  # use uint8 to encode image

            # convert semantic image to rgb color image to publish
            semantic_color = cv2.applyColorMap(
                semantic.astype(np.uint8), cv2.COLORMAP_JET
            )
            semantic_msg = self.cvbridge.cv2_to_imgmsg(semantic_color)
            semantic_msg.encoding = "bgr8"  # "rgb8"
            semantic_msg.header.stamp = cur_time
            semantic_msg.header.frame_id = "camera_link"
            self.semantic_publisher.publish(semantic_msg)

            # NOTE: following code could be use as reverse mapping from rgb color iamges
            # back to semantic images
            # create an inverse from the colormap to semantic values
            semantic_values = np.arange(256, dtype=np.uint8)
            color_values = map(
                tuple,
                cv2.applyColorMap(semantic_values, cv2.COLORMAP_JET).reshape(
                    256, 3
                ),
            )
            color_to_semantic_map = dict(zip(color_values, semantic_values))
            semantic_decoded = np.apply_along_axis(
                lambda bgr: color_to_semantic_map[tuple(bgr)],
                2,
                semantic_color,
            )

        # Publish camera info.
        if self.publish_camera_info:
            self.camera_info.header.stamp = cur_time
            self.camera_info_publisher.publish(self.camera_info)

        # Publish true pose
        if self.publish_true_pose:
            position, rotation = observations["agent_position"]
            y, z, x = position
            cur_orientation = rotation
            cur_euler_angles = tf.euler_from_quaternion(
                [
                    cur_orientation.w,
                    cur_orientation.x,
                    cur_orientation.z,
                    cur_orientation.y,
                ]
            )
            _, _, cur_z_angle = cur_euler_angles
            cur_z_angle += np.pi
            cur_pose = PoseStamped()
            cur_pose.header.stamp = cur_time
            cur_pose.header.frame_id = "map"
            cur_pose.pose.position.x = x
            cur_pose.pose.position.y = y
            cur_pose.pose.position.z = z
            (
                cur_pose.pose.orientation.w,
                cur_pose.pose.orientation.x,
                cur_pose.pose.orientation.y,
                cur_pose.pose.orientation.z,
            ) = tf.quaternion_from_euler(0, 0, cur_z_angle)
            self.pose_publisher.publish(cur_pose)

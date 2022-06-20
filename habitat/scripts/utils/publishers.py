"""Publisher helpers."""
import quaternion as qt 
import numpy as np
import rospy
import tf.transformations as tf
import yaml
import habitat_sim
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
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


def get_camera_info_config(config: habitat_sim.Configuration):
    """get camera info from habitat simulator config

    Args:
        sim (haibtat_sim.Simulator): simulator config class
    """
    assert NotImplementedError  # still errors remained, do not use
    sensor_spec: habitat_sim.sensor.CameraSensorSpec = config.agents[
        0
    ].sensor_specifications[1]
    height = sensor_spec.resolution[0].item()  # numpy.int32 to native int
    width = sensor_spec.resolution[1].item()
    hfov = float(sensor_spec.hfov) / 180.0 * np.pi  # degree to radius
    fx = cx = width / 2.0 / np.tan(hfov / 2.0)
    fy = cy = height / 2.0 / np.tan(hfov / 2.0)
    K_mat = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )
    P_mat = np.concatenate([K_mat, np.zeros((3, 1))], axis=1)
    rectification_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    distortion_coeff = [0.0, 0.0, 0.0, 0.0, 0.0]
    K = K_mat.reshape(-1).tolist()  # numpy.array to flat listtolist()
    P = P_mat.reshape(-1).tolist()  # numpy.array to flat listtolist()
    return CameraInfo(
        height=height,
        width=width,
        D=distortion_coeff,
        K=K,
        R=rectification_matrix,
        P=P,
    )


class HabitatObservationPublisher:
    """Publisher for observation of habitat."""

    def __init__(
        self,
        rgb_topic="",
        depth_topic="",
        semantic_topic="",
        camera_info_topic="",
        wheel_odom_topic="",
        true_pose_topic="",
        camera_info_file="",
        sim_config=None,
    ):
        """Initialize publisher with topic handles."""
        self.cvbridge = CvBridge()
        self.camera_info_topic = camera_info_topic
        self.publish_camera_info = False
        self.rgb_topic = rgb_topic
        self.publish_rgb = False
        self.depth_topic = depth_topic
        self.publish_depth = False
        self.semantic_topic = semantic_topic
        self.publish_semantic = False
        self.wheel_odom_topic = wheel_odom_topic
        self.publish_wheel_odom = False
        self.true_pose_topic = true_pose_topic
        self.publish_true_pose = False


        # Initialize camera info publisher.
        if len(camera_info_topic) > 0:
            self.publish_camera_info = True
            self.camera_info_publisher = rospy.Publisher(
                camera_info_topic, CameraInfo, latch=True, queue_size=100
            )
            if sim_config is not None:
                self.camera_info = get_camera_info_config(sim_config)
            else:
                self.camera_info = get_camera_info(camera_info_file)
            
        # Initialize RGB image publisher.
        if len(rgb_topic) > 0:
            self.publish_rgb = True
            self.image_publisher = rospy.Publisher(
                rgb_topic, Image, latch=True, queue_size=100
            )

        # Initialize depth image publisher.
        if len(depth_topic) > 0:
            self.publish_depth = True
            self.depth_publisher = rospy.Publisher(
                depth_topic, Image, latch=True, queue_size=100
            )
        
        # Initialize semantic image publisher
        if len(semantic_topic) > 0:
            self.publish_semantic = True
            self.semantic_publisher = rospy.Publisher(
                semantic_topic, Image, latch=True, queue_size=100
            )

        # Initialize wheel odometry publisher 
        if len(wheel_odom_topic) > 0:
            self.publish_wheel_odom = True
            self.wheel_odom_publisher = rospy.Publisher(
                wheel_odom_topic, Odometry, queue_size=10
            )

        # Initialize position publisher.
        # if len(true_pose_topic) > 0:
        if False:
            self.publish_true_pose = True
            self.pose_publisher = rospy.Publisher(
                true_pose_topic, PoseStamped, latch=True, queue_size=100
            )

    def publish(self, observations):
        """Publish messages."""
        cur_time = rospy.Time.now()


        # Publish camera info.
        if self.publish_camera_info:
            self.camera_info.header.stamp = cur_time
            self.camera_info_publisher.publish(self.camera_info)

        # Publish RGB image.
        if self.publish_rgb:
            # self.image = self.cvbridge.cv2_to_imgmsg(observations['rgb'])
            image = self.cvbridge.cv2_to_imgmsg(observations["rgb"][:, :, 0:3])
            image.encoding = "rgb8"
            image.header.stamp = cur_time
            image.header.frame_id = "camera_link"
            self.image_publisher.publish(image)

        # Publish depth image.
        if self.publish_depth:
            depth = self.cvbridge.cv2_to_imgmsg(
                observations["depth"] * DEPTH_SCALE
            )
            depth.header.stamp = cur_time
            depth.header.frame_id = "base_scan"
            self.depth_publisher.publish(depth)

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
            # semantic_values = np.arange(256, dtype=np.uint8)
            # color_values = map(
            #     tuple,
            #     cv2.applyColorMap(semantic_values, cv2.COLORMAP_JET).reshape(
            #         256, 3
            #     ),
            # )
            # color_to_semantic_map = dict(zip(color_values, semantic_values))
            # semantic_decoded = np.apply_along_axis(
            #     lambda bgr: color_to_semantic_map[tuple(bgr)],
            #     2,
            #     semantic_color,
            # )

        if self.publish_wheel_odom:
            
            pose_mat = observations["odom_pose_mat"]
            odom_msg = Odometry()

            odom_msg.header.stamp = cur_time
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'
            
            # construct position and rotation
            x, y, z = pose_mat[:3,3]
            quat = qt.from_rotation_matrix(pose_mat[:3, :3])
            odom_msg.pose.pose.position = Point(x, y, z)
            odom_msg.pose.pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)

            # create pseudo diagnal covariance matrix
            # robot on x-y plane
            p_cov = np.array([
                5e-2, 0., 0., 0., 0., 0.,
                0., 5e-2, 0., 0., 0., 0.,
                0., 0., 1e3, 0., 0., 0.,
                0., 0., 0., 1e3, 0., 0.,
                0., 0., 0., 0., 1e3, 0.,
                0., 0., 0., 0., 0., 1e-2,
            ])
            odom_msg.pose.covariance = p_cov.tolist()

            # NOTE: if to implement continous control, add psedu twist message
            
            # Publish odometry message
            self.wheel_odom_publisher.publish(odom_msg)


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

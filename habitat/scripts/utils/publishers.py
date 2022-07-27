"""Publisher helpers."""
import struct
from pyrsistent import s
import quaternion as qt 
import numpy as np
import rospy
import tf2_ros
from tf2_msgs.msg import TFMessage
from tf import transformations
import yaml
import habitat_sim
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Header
from geometry_msgs.msg import (
    PoseStamped, Point, Quaternion, TransformStamped, Vector3
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from envs.constants import coco_categories

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


def get_camera_info_file(filepath):
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
    # assert NotImplementedError  # still errors remained, do not use
    sensor_spec: habitat_sim.sensor.CameraSensorSpec = config.agents[
        0
    ].sensor_specifications[1]
    height = sensor_spec.resolution[0].item()  # numpy.int32 to native int
    width = sensor_spec.resolution[1].item()
    hfov = np.deg2rad(float(sensor_spec.hfov))  # degree to radius
    cx = width / 2.0 
    cy = height / 2.0 
    f = (width / 2.0) / np.tan(hfov / 2.0)
    K_mat = np.array(
        [
            [f, 0.0, cx],
            [0.0, f, cy],
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
        ground_truth_odom_topic="", # publish true odometry as Odometry message
        camera_info_file="",
        wheel_odom_frame_id="", # publish wheel_odom as tf: odom_wheel -> base_link
        true_pose_topic="",
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
        self.true_odom_topic = ground_truth_odom_topic
        self.publish_true_odom = False
        self.publish_true_odom_tf = False
        # publish wheel odometry as tf: odom_wheel -> base_link, 
        # see "guess_frame_id" in http://wiki.ros.org/rtabmap_ros
        self.wheel_odom_frame_id = wheel_odom_frame_id
        self.publish_wheel_odom_tf = False
        # true_pose_topic not used 
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
                self.camera_info = get_camera_info_file(camera_info_file)

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
        if len(ground_truth_odom_topic) > 0:
            self.publish_true_odom = True
            self.publish_true_odom_tf = True
            self.true_odom_publisher = rospy.Publisher(
                ground_truth_odom_topic, Odometry, queue_size=1
            )

        # Initialize tf publisher for wheel odometry
        if len(wheel_odom_frame_id) > 0:
            self.publish_wheel_odom_tf = True
            
        self.tf_publisher = rospy.Publisher("/tf", TFMessage, queue_size=1)

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
        
        ################# publish meta info ##########################
        # Publish camera info.
        if self.publish_camera_info:
            self.camera_info.header.stamp = cur_time
            self.camera_info_publisher.publish(self.camera_info)

        ############## publish transformations ########################
        
        # publish wheel odometry as tf: odom_wheel -> base_link
        # publish true odom tf: odom -> base_link
        # NOTE: publish guess frame before observations
        # Otherwise odometry update will be aborted 
        tf_msgs = []

        if self.publish_wheel_odom_tf:

            wo_msg = TransformStamped()
            wo_msg.header.stamp = cur_time
            wo_msg.header.frame_id = self.wheel_odom_frame_id
            wo_msg.child_frame_id = "base_link"
                
            # construct position and rotation
            pose_mat = observations["odom_pose_mat"]
            x, y, z = pose_mat[:3,3]
            quat = qt.from_rotation_matrix(pose_mat[:3, :3])
            wo_msg.transform.translation = Vector3(x, y, z)
            wo_msg.transform.rotation = Quaternion(quat.x, quat.y, quat.z, quat.w)

            tf_msgs.append(wo_msg)
            
        if self.publish_true_odom_tf:
            
            to_msg = TransformStamped()
            to_msg.header.stamp = cur_time
            to_msg.header.frame_id = "odom"
            to_msg.child_frame_id = "base_link"
                
            # construct position and rotation
            pose_mat = observations["true_odom_mat"]
            x, y, z = pose_mat[:3,3]
            quat = qt.from_rotation_matrix(pose_mat[:3, :3])
            to_msg.transform.translation = Vector3(x, y, z)
            to_msg.transform.rotation = Quaternion(quat.x, quat.y, quat.z, quat.w)

            tf_msgs.append(to_msg)
            
        if len(tf_msgs) > 0:
            tf_msgs = TFMessage(tf_msgs)
            self.tf_publisher.publish(tf_msgs)
            
        ################# publish odometry ########################
        if self.publish_true_odom:
            
            pose_mat = observations["true_odom_mat"]
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
                0., 0., 5e-2, 0., 0., 0.,
                0., 0., 0., 1e-2, 0., 0.,
                0., 0., 0., 0., 1e-2, 0.,
                0., 0., 0., 0., 0., 1e-2,
            ])
            odom_msg.pose.covariance = p_cov.tolist()

            # NOTE: if to implement continous control, add psedu twist message
            
            # Publish odometry message
            self.true_odom_publisher.publish(odom_msg)

        # NOTE: the observations should be published after camera info, tf, pose 
        # otherwise odometry and cloud registration will be invalidated by SLAM system 
        ############## publish observations ##########################
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
            ######### jet color map ###############
            # semantic_color = cv2.applyColorMap(
            #     semantic.astype(np.uint8), cv2.COLORMAP_JET
            # )
            # semantic_msg = self.cvbridge.cv2_to_imgmsg(semantic_color)
            # semantic_msg.encoding = "bgr8"  # "rgb8"
            
            ######### autumn color map ################
            num_class = len(coco_categories)
            semantic_color = np.zeros((semantic.shape[0], semantic.shape[1], 3)).astype(np.uint8)
            semantic_color[..., 0] = 255
            # 0 for background in semantic image 
            semantic_color[..., 1] = np.round(semantic / float(num_class + 1) * 255.0).astype(np.uint8)
            
            semantic_msg = self.cvbridge.cv2_to_imgmsg(semantic_color)
            semantic_msg.encoding = "rgb8"
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
 
        # Publish true pose
        if self.publish_true_pose:
            position, rotation = observations["agent_position"]
            y, z, x = position
            cur_orientation = rotation
            cur_euler_angles = transformations.euler_from_quaternion(
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
            ) = transformations.quaternion_from_euler(0, 0, cur_z_angle)
            self.pose_publisher.publish(cur_pose)


class PointCloudPublisher:
    """Publisher for point cloud."""

    def __init__(self, topic_name, frame_id="map", queue_size=1):
        """Set point cloud publisher"""
        self.pub = rospy.Publisher(
            topic_name, PointCloud2, queue_size=queue_size
        )
        self.frame_id = frame_id

    def publish_o3d_pcl(self, open3d_cloud):
        
        points = np.asarray(open3d_cloud.points, dtype=np.float32)
        colors = None
        if open3d_cloud.colors is not None:
            colors = np.floor(np.asarray(open3d_cloud.colors) * 255).astype(
                np.uint32
            ) 
        pcl_msg = self.create_msg(points, colors)
        self.pub.publish(pcl_msg)
        return

    def publish_np_pcl(self, points, colors=None):
        
        pcl_msg = self.create_msg(points, colors)
        self.pub.publish(pcl_msg)
        return 

    def create_msg(self, points, colors):

        header = Header(frame_id=self.frame_id)
        header.stamp = rospy.Time.now()

        # Set "fields" and "cloud_data"
        if colors is None:  # XYZ only
            fields = FIELDS_XYZ
            cloud_data = points
        else:  # XYZ + RGB
            fields = FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.

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

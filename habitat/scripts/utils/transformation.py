"""This is a collection of helper funtions for transformation."""

import copy
import sys

import numpy as np
import rospy
import tf2_ros
import quaternion as qt
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import PoseStamped, TransformStamped
from habitat_sim.agent import Agent
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs

# NOTE: tf2_geometry_msgs depends on cython library PyKDL, which is compiled when
# installing ros-xxx-full package. Need to add path to this package to sys.path
sys.path.append(
    "/usr/lib/python3/dist-packages"
)  # needed by tf2_geometry_msgs
import tf2_geometry_msgs

# NOTE: transformation between frames and transformation between points in frames are inverse matrix


def o3d_rtab2mp3d(o3d_cloud):
    """Convert rtab coordinates to mp3d coordinates."""
    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    o3d_quat = np.roll(quat_to_coeffs(quat), 1)
    r_mat = o3d_cloud.get_rotation_matrix_from_quaternion(o3d_quat)
    o3d_cloud_r = copy.deepcopy(o3d_cloud)
    o3d_cloud_r.rotate(r_mat, center=(0, 0, 0))
    return o3d_cloud_r


# TODO: fetch ground truth transformation from /habitat/matterport3d to /rtabmap/grid_map
def points_rtab2habitat(points):
    """Convert rtab coordinates to habitat coordinates."""
    quat = quat_from_two_vectors(np.array([0, 0, 1]), np.array([0, -1, 0]))
    r_mat = qt.as_rotation_matrix(quat)
    homo_r_mat = np.zeros((4, 4))
    homo_r_mat[:3, :3] = r_mat
    homo_r_mat[3, 3] = 1.0
    homo_points_rtab = np.concatenate(
        [points, np.ones((points.shape[0], 1))], axis=1
    )
    homo_points_habitat = (homo_r_mat @ homo_points_rtab.T).T  # (N,4)
    points_habitat = homo_points_habitat[:, :3] / homo_points_habitat[:, 3:]
    return points_habitat


def publish_static_base_to_cam(sensor_height):
    # 3. create mp3d_link to camera_link
    tf_mp3d2cam = TransformStamped()
    ros_time = rospy.Time.now()

    tf_mp3d2cam.header.stamp = ros_time
    tf_mp3d2cam.header.frame_id = "base_link"  # z-axis upright
    tf_mp3d2cam.child_frame_id = "camera_link"  # y-axis upright

    tf_mp3d2cam.transform.translation.x = 0.0
    tf_mp3d2cam.transform.translation.y = 0.0
    tf_mp3d2cam.transform.translation.z = sensor_height

    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    tf_mp3d2cam.transform.rotation.x = quat.x
    tf_mp3d2cam.transform.rotation.y = quat.y
    tf_mp3d2cam.transform.rotation.z = quat.z
    tf_mp3d2cam.transform.rotation.w = quat.w

    transforms = [tf_mp3d2cam]

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(transforms)

    return transforms


def publish_agent_init_tf(agent: Agent, sensor_id="rgb", instant_pub=True):

    """
    publish three TFs required by module:
    1. from world to odom: initial position of agent in matterport3D frame
    2. from base_link (agent base) to mp3d_link
    3. from mp3d_link to camera_link (habitat camera frame)

    NOTE: the default rotation in habitat is:
    /* rotation = quatf::FromTwoVectors(-vec3f::UnitZ(),vec3f::UnitY()) */
    From habitat->world: simply [x, y, z] -> [x, z, -y]
    From world->habitat: simply [x, y, z] -> [x, -z, y]

    For more details, please refer to readme or ./img/tf_tree.png
    """
    broadcaster = tf2_ros.StaticTransformBroadcaster()

    agent_state = agent.get_state()
    sensor_pos = agent_state.sensor_states[sensor_id].position
    # NOTE: habitat set y-axis in upright direction
    sensor_height = sensor_pos[1] - agent_state.position[1]
    base_pos = agent_state.position
    # quaternion, [w,x,y,z]
    base_quat = agent_state.rotation
    ros_time = rospy.Time.now()

    # 0. get initial pose in world (mp3d) frame
    temp_tranform_habitat2mp3d = TransformStamped()
    temp_tranform_habitat2mp3d.header.stamp = ros_time
    temp_tranform_habitat2mp3d.header.frame_id = "habitat_world"
    temp_tranform_habitat2mp3d.child_frame_id = "world"

    temp_tranform_habitat2mp3d.transform.translation.x = 0
    temp_tranform_habitat2mp3d.transform.translation.y = 0
    temp_tranform_habitat2mp3d.transform.translation.z = 0

    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    # quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 1, 0]))
    temp_tranform_habitat2mp3d.transform.rotation.x = quat.x
    temp_tranform_habitat2mp3d.transform.rotation.y = quat.y
    temp_tranform_habitat2mp3d.transform.rotation.z = quat.z
    temp_tranform_habitat2mp3d.transform.rotation.w = quat.w

    habitat_init_pose = PoseStamped()
    habitat_init_pose.pose.position.x = base_pos[0]
    habitat_init_pose.pose.position.y = base_pos[1]
    habitat_init_pose.pose.position.z = base_pos[2]

    habitat_init_pose.pose.orientation.x = base_quat.x
    habitat_init_pose.pose.orientation.y = base_quat.y
    habitat_init_pose.pose.orientation.z = base_quat.z
    habitat_init_pose.pose.orientation.w = base_quat.w

    habitat_init_pose.header.frame_id = "habitat_world"
    habitat_init_pose.header.stamp = ros_time

    world_init_pose = tf2_geometry_msgs.do_transform_pose(
        habitat_init_pose, temp_tranform_habitat2mp3d
    )
    world_init_pose.header.frame_id = "world"
    # 1. create transform message from world to map

    tf_world2map = TransformStamped()

    tf_world2map.header.stamp = ros_time
    tf_world2map.header.frame_id = "world"  # ground truth mp3d
    tf_world2map.child_frame_id = "map"

    tf_world2map.transform.translation.x = world_init_pose.pose.position.x
    tf_world2map.transform.translation.y = world_init_pose.pose.position.y
    tf_world2map.transform.translation.z = world_init_pose.pose.position.z

    tf_world2map.transform.rotation.x = world_init_pose.pose.orientation.x
    tf_world2map.transform.rotation.y = world_init_pose.pose.orientation.y
    tf_world2map.transform.rotation.z = world_init_pose.pose.orientation.z
    tf_world2map.transform.rotation.w = world_init_pose.pose.orientation.w

    # 2. create base_link to mp3d_link (relative to z-axis upright frame)
    # tf_base2mp3d = TransformStamped()

    # tf_base2mp3d.header.stamp = ros_time
    # tf_base2mp3d.header.frame_id = "base_link"
    # tf_base2mp3d.child_frame_id = "mp3d_link"

    # tf_base2mp3d.transform.translation.x = 0.0
    # tf_base2mp3d.transform.translation.y = 0.0
    # tf_base2mp3d.transform.translation.z = -sensor_height

    # tf_base2mp3d.transform.rotation.x = 0.0
    # tf_base2mp3d.transform.rotation.y = 0.0
    # tf_base2mp3d.transform.rotation.z = 0.0
    # tf_base2mp3d.transform.rotation.w = 1.0

    # 3. create base_link (rtabmap_coords) to camera_link (habitat_coords)
    tf_base2cam = TransformStamped()

    tf_base2cam.header.stamp = ros_time
    tf_base2cam.header.frame_id = "base_link"  # z-axis upright
    tf_base2cam.child_frame_id = "camera_link"  # y-axis upright

    tf_base2cam.transform.translation.x = 0.0
    tf_base2cam.transform.translation.y = 0.0
    tf_base2cam.transform.translation.z = sensor_height

    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    tf_base2cam.transform.rotation.x = quat.x
    tf_base2cam.transform.rotation.y = quat.y
    tf_base2cam.transform.rotation.z = quat.z
    tf_base2cam.transform.rotation.w = quat.w

    # transforms = [tf_world2map, tf_mp3d2cam]
    transforms = [tf_base2cam]
    if instant_pub:
        broadcaster.sendTransform(transforms)

    return transforms


def tf_stamped_2_se3(tf_stamped: TransformStamped):
    """get the 4x4 transformation matrix from ros tf

    NOTE: In ros tf, transform from frame A to B means the transform between frames,
    this function returns matrix which transform points in B to A, if you want to transform
    points from A to B, you would need the inverse of this matrix

    Args:
        tf_stamped (TransformStamped): ros transformation

    Returns:
        mat (numpy.ndarray):
    """
    tf_msg = tf_stamped.transform
    p = np.array(
        [tf_msg.translation.x, tf_msg.translation.y, tf_msg.translation.z]
    )
    q = np.array(
        [
            tf_msg.rotation.x,
            tf_msg.rotation.y,
            tf_msg.rotation.z,
            tf_msg.rotation.w,
        ]
    )
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)
            )
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    mat = quaternion_matrix(q)
    mat[0:3, 3] = p
    return mat


def demo_rtabmap_2_habitat(sensor_height=0.88):

    tf_base2cam = TransformStamped()
    ros_time = rospy.Time.now()
    tf_base2cam.header.stamp = ros_time
    tf_base2cam.header.frame_id = "base_link"  # z-axis upright
    tf_base2cam.child_frame_id = "camera_link"  # y-axis upright

    tf_base2cam.transform.translation.x = 0.0
    tf_base2cam.transform.translation.y = 0.0
    tf_base2cam.transform.translation.z = sensor_height

    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    tf_base2cam.transform.rotation.x = quat.x
    tf_base2cam.transform.rotation.y = quat.y
    tf_base2cam.transform.rotation.z = quat.z
    tf_base2cam.transform.rotation.w = quat.w

    # example 1: transform pose of camera principal axis from base_link frame to camera frame
    cam_axis_pose = PoseStamped()
    cam_axis_pose.pose.position.x = 0.0
    cam_axis_pose.pose.position.y = 0.0
    cam_axis_pose.pose.position.z = 0.0  # sensor_height
    quat = np.array([0, 0, 1, 0]) / np.linalg.norm(np.array([0, 0, 1, 0]))
    cam_axis_pose.pose.orientation.x = quat[0]
    cam_axis_pose.pose.orientation.y = quat[1]
    cam_axis_pose.pose.orientation.z = quat[2]
    cam_axis_pose.pose.orientation.w = quat[3]

    cam_axis_pose.header.frame_id = "camera_link"
    cam_axis_pose.header.stamp = ros_time

    cam_axis_pose_base = tf2_geometry_msgs.do_transform_pose(
        cam_axis_pose, tf_base2cam
    )
    print(cam_axis_pose_base.pose)

    # example 2: transform an array of points
    points_cam = np.random.rand(10, 3)  # random list of points
    tf_mat = tf_stamped_2_se3(tf_base2cam)
    homo_points_cam = np.concatenate([points_cam, np.ones((10, 1))], axis=1)
    homo_points_rtab = (tf_mat @ homo_points_cam.T).T  # (10,4)
    points_rtab = homo_points_cam[:, :3] / homo_points_cam[:, 3:]

    # print()


if __name__ == "__main__":

    ############## ROS test #######################################
    rospy.init_node("test_tf")
    # from geometry_msgs.msg import Transform
    # from simulator import init_sim

    # test_scene = "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
    # sim, sim_agent, action_names = init_sim(test_scene)
    # transforms = publish_agent_init_tf(sim_agent)

    # # DEBUG use: publish tf map->odom and odom->base_link to connect tf tree
    # broadcaster = tf2_ros.StaticTransformBroadcaster()
    # ros_time = transforms[0].header.stamp

    # tf_map2odom = TransformStamped()  # default identity tf
    # tf_map2odom.header.stamp = ros_time
    # tf_map2odom.header.frame_id = "map"  # ground truth mp3d
    # tf_map2odom.child_frame_id = "odom"
    # tf_map2odom.transform.rotation.w = 1.0

    # tf_odom2base = TransformStamped()
    # tf_odom2base.header.stamp = ros_time
    # tf_odom2base.header.frame_id = "odom"  # ground truth mp3d
    # tf_odom2base.child_frame_id = "base_link"
    # tf_odom2base.transform.rotation.w = 1.0

    # transforms.extend([tf_map2odom, tf_odom2base])
    # broadcaster.sendTransform(transforms)

    # print("TF published, spinning now...")
    # rospy.spin()

    ###################### transform example ########################
    # demo_rtabmap_2_habitat()

    # test if quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    # and quat_from_two_vectors(np.array([0, 0, 1]), np.array([0, -1, 0]))
    # are inverse transformations
    quat1 = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    mat1 = qt.as_rotation_matrix(quat1)
    quat2 = quat_from_two_vectors(np.array([0, 0, 1]), np.array([0, -1, 0]))
    mat2 = qt.as_rotation_matrix(quat2)

    print(mat1 @ mat2)
    """
    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]
    """

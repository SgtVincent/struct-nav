"""This is a collection of helper funtions for transformation."""

import copy
import sys

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from habitat_sim.agent import Agent
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs

# NOTE: tf2_geometry_msgs depends on cython library PyKDL, which is compiled when
# installing ros-xxx-full package. Need to add path to this package to sys.path
sys.path.append(
    "/usr/lib/python3/dist-packages"
)  # needed by tf2_geometry_msgs
import tf2_geometry_msgs


def coo_rtab2mp3d(o3d_cloud):
    """Convert rtab coordinates to mp3d coordinates."""
    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    o3d_quat = np.roll(quat_to_coeffs(quat), 1)
    r_mat = o3d_cloud.get_rotation_matrix_from_quaternion(o3d_quat)
    o3d_cloud_r = copy.deepcopy(o3d_cloud)
    o3d_cloud_r.rotate(r_mat, center=(0, 0, 0))
    return o3d_cloud_r


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

    # quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 1, 0]))
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

    # 3. create mp3d_link to camera_link
    tf_mp3d2cam = TransformStamped()

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

    # transforms = [tf_world2map, tf_mp3d2cam]
    transforms = [tf_mp3d2cam]
    if instant_pub:
        broadcaster.sendTransform(transforms)

    return transforms


if __name__ == "__main__":

    from geometry_msgs.msg import Transform
    from simulator import init_sim

    rospy.init_node("test_tf")
    test_scene = "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
    sim, sim_agent, action_names = init_sim(test_scene)
    transforms = publish_agent_init_tf(sim_agent)

    # DEBUG use: publish tf map->odom and odom->base_link to connect tf tree
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    ros_time = transforms[0].header.stamp

    tf_map2odom = TransformStamped()  # default identity tf
    tf_map2odom.header.stamp = ros_time
    tf_map2odom.header.frame_id = "map"  # ground truth mp3d
    tf_map2odom.child_frame_id = "odom"
    tf_map2odom.transform.rotation.w = 1.0

    tf_odom2base = TransformStamped()
    tf_odom2base.header.stamp = ros_time
    tf_odom2base.header.frame_id = "odom"  # ground truth mp3d
    tf_odom2base.child_frame_id = "base_link"
    tf_odom2base.transform.rotation.w = 1.0

    transforms.extend([tf_map2odom, tf_odom2base])
    broadcaster.sendTransform(transforms)

    print("TF published, spinning now...")
    rospy.spin()

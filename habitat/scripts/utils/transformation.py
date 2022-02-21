"""This is a collection of helper funtions for transformation."""

import copy

import numpy as np
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from habitat_sim.agent import Agent
from geometry_msgs.msg import TransformStamped
import rospy
import tf2_ros
import tf


def coo_rtab2mp3d(o3d_cloud):
    """Convert rtab coordinates to mp3d coordinates."""
    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    o3d_quat = np.roll(quat_to_coeffs(quat), 1)
    r_mat = o3d_cloud.get_rotation_matrix_from_quaternion(o3d_quat)
    o3d_cloud_r = copy.deepcopy(o3d_cloud)
    o3d_cloud_r.rotate(r_mat, center=(0, 0, 0))
    return o3d_cloud_r


# def publish_agent_init_tf(agent: Agent, sensor_id="rgb"):

#     """
#     # agent state attrs:
#     ['angular_velocity',
#     'force',
#     'position',
#     'rotation',
#     'sensor_states',
#     'torque',
#     'velocity']
#     """
#     agent_state = agent.get_state()
#     sensor_pos = agent_state.sensor_states[sensor_id].position
#     # NOTE: habitat set y-axis in upright direction
#     sensor_height = sensor_pos[1] - agent_state.position[1]
#     base_pos = agent_state.position
#     # quaternion, [w,x,y,z]
#     base_rot = agent_state.rotation

#     # create transform message from world to odom
#     static_transformStamped = TransformStamped()

#     static_transformStamped.header.stamp = rospy.Time.now()
#     static_transformStamped.header.frame_id = "world"
#     static_transformStamped.child_frame_id = "odom"

#     static_transformStamped.transform.translation.x = base_pos[0]
#     static_transformStamped.transform.translation.y = base_pos[1]
#     static_transformStamped.transform.translation.z = base

#     quat = tf.transformations.quaternion_from_euler(
#         float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
#     )
#     static_transformStamped.transform.rotation.x = quat[0]
#     static_transformStamped.transform.rotation.y = quat[1]
#     static_transformStamped.transform.rotation.z = quat[2]
#     static_transformStamped.transform.rotation.w = quat[3]

#     return


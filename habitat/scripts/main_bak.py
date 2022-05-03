#! /usr/bin/env python
"""Main function for the habitat node."""

import numpy as np
import open3d as o3d
import rospy
from agents.dummy_agents import *

# from agents.frontier_explore_agent import FrontierExploreAgent
from publishers import HabitatObservationPublisher
from simulator import init_sim

# from std_msgs.msg import Int32
from subscribers import PointCloudSubscriber
from utils import transformation

DEFAULT_RATE = 0.5
DEFAULT_AGENT_TYPE = "random_walk"
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1
VISUALIZE = False


def main():
    """Main function entry."""
    # Initialize ROS node and take arguments
    rospy.init_node("habitat_ros_node")
    node_start_time = rospy.Time.now().to_sec()
    # task_config = rospy.get_param("~task_config")
    rate_value = rospy.get_param("~rate", DEFAULT_RATE)
    agent_type = rospy.get_param("~agent_type", DEFAULT_AGENT_TYPE)
    # assert (
    #     agent_type in AGENT_CLASS_MAPPING.keys()
    # ), f"{agent_type} not in supported agent types: {AGENT_CLASS_MAPPING.keys()}"

    # goal_radius = rospy.get_param("~goal_radius", DEFAULT_GOAL_RADIUS)
    # max_d_angle = rospy.get_param("~max_d_angle", DEFAULT_MAX_ANGLE)
    rgb_topic = rospy.get_param("~rgb_topic", "")
    depth_topic = rospy.get_param("~depth_topic", "")
    camera_info_topic = rospy.get_param("~camera_info_topic", "")
    true_pose_topic = rospy.get_param("~true_pose_topic", "")
    cloud_topic = rospy.get_param("~cloud_topic", "")
    camera_info_file = rospy.get_param("~camera_calib", None)

    # topics for planning
    if agent_type not in ["random_walk", "spinning"]:
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        local_plan_topic = rospy.get_param(
            "~local_plan_topic", "/move_base/DWAPlannerROS/local_plan"
        )
        global_plan_topic = rospy.get_param(
            "~global_plan_topic", "/move_base/DWAPlannerROS/global_plan"
        )

    # ros pub and sub
    rate = rospy.Rate(rate_value)
    publisher = HabitatObservationPublisher(
        rgb_topic, depth_topic, camera_info_topic, true_pose_topic, camera_info_file,
    )
    # action_publisher = rospy.Publisher(
    #     "habitat_action", Int32, latch=True, queue_size=100
    # )

    sub_cloud = PointCloudSubscriber(cloud_topic)

    # Initial Sim
    test_scene = rospy.get_param("~test_scene", None)
    sim, action_names = init_sim(test_scene)

    # Initialize the agent and environment
    # env = habitat.Env(config=config)
    # env.reset()

    # Initialize TF tree with ground truth init pose (if any)
    sim_agent = sim.get_agent(0)
    transformation.publish_agent_init_tf(sim_agent)

    if agent_type == "random_walk":
        agent = RandomWalkAgent(action_names)
    elif agent_type == "spinning":
        agent = SpinningAgent("turn_right")
    # elif agent_type == "frontier_explore":
    #     agent = FrontierExploreAgent(odom_topic, local_plan_topic, global_plan_topic)
    else:
        print("AGENT TYPE {} IS NOT DEFINED!!!".format(agent_type))
        return

    # Run the simulator with agent
    # observations = sim.reset()
    observations = sim.step("stay")
    robot_start_time = rospy.Time.now().to_sec()
    print("TIME TO LAUNCH HABITAT:", robot_start_time - node_start_time)

    cnt_pub = 0
    cnt_sub = 0

    while not rospy.is_shutdown():
        publisher.publish(observations)
        cnt_pub += 1

        if sub_cloud.has_cloud():
            o3d_cloud = sub_cloud.get_cloud()
            cnt_sub += 1

            num_points = np.asarray(o3d_cloud.points).shape[0]
            rospy.loginfo(
                "Publish: {} th images, "
                "Subscribe: {} th point cloud, "
                "{} points.".format(cnt_pub, cnt_sub, num_points)
            )

            if VISUALIZE and cnt_sub == 11:
                coo = o3d.geometry.TriangleMesh.create_coordinate_frame()
                o3d_cloud = transformation.coo_rtab2mp3d(o3d_cloud)
                o3d.visualization.draw_geometries([coo, o3d_cloud])

        # action = agent.act(observations, sim)
        action = agent.act()
        # action_msg = Int32()
        # action_msg.data = action
        # action_publisher.publish(action_msg)
        observations = sim.step(action)
        rate.sleep()


if __name__ == "__main__":
    main()

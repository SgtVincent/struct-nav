#! /usr/bin/env python
"""Main function for the habitat node."""

from mimetypes import init
import quaternion as qt 
import numpy as np
import open3d as o3d
import rospy
from agents.dummy_agents import RandomWalkAgent, SpinningAgent
from agents.frontier_explore_agent import FrontierExploreAgent
from agents.utils.arguments import get_args
from agents.utils.ros_utils import safe_call_reset_service
from utils.publishers import HabitatObservationPublisher
from utils.simulator import init_sim
from utils.transformation import pose_habitat2rtabmap, homo_matrix_from_quat

# from std_msgs.msg import Int32
from subscribers import PointCloudSubscriber
from utils import transformation

# parameters used for debuging with python debugger
DEFAULT_RATE = 1.0
# DEFAULT_TEST_SCENE = "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
DEFAULT_TEST_SCENE = "/home/junting/Downloads/datasets/habitat_data/scene_datasets/mp3d/v1/scans/17DRP5sb8fy/17DRP5sb8fy.glb"
DEFAULT_CAMERA_CALIB = "./envs/habitat/configs/camera_info.yaml"
DEFAULT_AGENT_TYPE = "frontier_explore"  # "random_walk"
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
    camera_info_file = rospy.get_param("~camera_calib", DEFAULT_CAMERA_CALIB)
    # assert (
    #     agent_type in AGENT_CLASS_MAPPING.keys()
    # ), f"{agent_type} not in supported agent types: {AGENT_CLASS_MAPPING.keys()}"

    # goal_radius = rospy.get_param("~goal_radius", DEFAULT_GOAL_RADIUS)
    # max_d_angle = rospy.get_param("~max_d_angle", DEFAULT_MAX_ANGLE)
    rgb_topic = rospy.get_param("~rgb_topic", "/camera/rgb/image")
    depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image")
    semantic_topic = rospy.get_param(
        "~semantic_topic", ""
    )  # "/camera/semantic/image") # not to pulish gt by default
    camera_info_topic = rospy.get_param(
        "~camera_info_topic", "/camera/rgb/camera_info"
    )
    true_pose_topic = rospy.get_param("~true_pose_topic", "")
    cloud_topic = rospy.get_param("~cloud_topic", "/rtabmap/cloud_map")
    
    # topics for planning
    if agent_type in ["frontier_explore"]:
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        grid_map_topic = rospy.get_param(
            "~grid_map_topic", "/rtabmap/grid_map"
        )
        frontiers_topic = rospy.get_param("~frontiers_topic", "/frontiers")
        # goal_topic = rospy.get_param("~goal_topic", "/nav_goal")

    test_scene = rospy.get_param("~test_scene", DEFAULT_TEST_SCENE)
    # setup ground truth odometry 
    ground_truth_odom = rospy.get_param("~ground_truth_odom", False)
    ground_truth_odom_topic = ""
    if ground_truth_odom:
        ground_truth_odom_topic = rospy.get_param("~ground_truth_odom_topic", "")
        odom_topic = ground_truth_odom_topic
    init_pos = rospy.get_param("~init_pos", [-0.5, 0.0, -0.5]) # [-0.5, 0.0, -0.5]
    init_rot = rospy.get_param("~init_rot", [1.0, 0.0, 1.0, 0.0]) # [1.0, 0.0, -1.0, 0.0]

    # for debug use, relaunch agent without relaunch rtabmap 
    safe_call_reset_service("/rtabmap/reset")
    if not ground_truth_odom:
        safe_call_reset_service("/rtabmap/reset_odom")

    # ros pub and sub
    rate = rospy.Rate(rate_value)
    publisher = HabitatObservationPublisher(
        rgb_topic=rgb_topic,
        depth_topic=depth_topic,
        semantic_topic=semantic_topic,
        camera_info_topic=camera_info_topic,
        true_pose_topic=true_pose_topic,
        camera_info_file=camera_info_file,
    )
    # action_publisher = rospy.Publisher(
    #     "habitat_action", Int32, latch=True, queue_size=100
    # )

    sub_cloud = PointCloudSubscriber(cloud_topic)

    # Initial Sim
    init_rot = qt.quaternion(*(np.array(init_rot)/ np.linalg.norm(init_rot)))
    sim, action_names = init_sim(test_scene, init_pos, init_rot)
    observations = sim.step("stay")
    # Initialize TF tree with ground truth init pose (if any)
    sim_agent = sim.get_agent(0)
    transformation.publish_agent_init_tf(sim_agent)
    init_pos = sim_agent.state.position
    init_rot = sim_agent.state.rotation
    
    if agent_type == "random_walk":
        agent = RandomWalkAgent(action_names)

    elif agent_type == "spinning":
        agent = SpinningAgent("turn_right")

    elif agent_type == "frontier_explore":
        agent_args = get_args(silence_mode=True)
        agent_args.odom_topic = odom_topic
        agent_args.grid_map_topic = grid_map_topic
        agent_args.frontiers_topic = frontiers_topic
        agent_args.init_pos = init_pos
        agent_args.init_rot = init_rot
        agent_args.ground_truth_odom = ground_truth_odom
        # agent_args.goal_topic = goal_topic
        agent = FrontierExploreAgent(agent_args, sim)

    else:
        print("AGENT TYPE {} IS NOT DEFINED!!!".format(agent_type))
        return

    # ros pub and sub
    rate = rospy.Rate(rate_value)
    publisher = HabitatObservationPublisher(
        rgb_topic=rgb_topic,
        depth_topic=depth_topic,
        camera_info_topic=camera_info_topic,
        ground_truth_odom_topic=ground_truth_odom_topic,
        true_pose_topic=true_pose_topic,
        camera_info_file=camera_info_file,
        # sim_config=sim.config, # bugs remained!
    )

    # Run the simulator with agent
    # observations = sim.reset()
    # observations = sim.step("stay")
    robot_start_time = rospy.Time.now().to_sec()
    print("TIME TO LAUNCH HABITAT:", robot_start_time - node_start_time)

    cnt_pub = 0
    # cnt_sub = 0
    cnt_action = 0

    action = "stay"  # initial action to get first frame
    while not rospy.is_shutdown():
        print(f"Step {cnt_action}:simulator execute action {action}")
        if action != "stay" or cnt_action == 0:
            observations = sim.step(action)
            cnt_action += 1
            
            if ground_truth_odom:
                agent_state = sim.get_agent(0).get_state()
                habitat_rot = agent_state.rotation
                habitat_pos = agent_state.position
                pos_rtab, rot_rtab = pose_habitat2rtabmap(habitat_pos, habitat_rot, init_pos, init_rot)
                observations['true_odom_mat'] = homo_matrix_from_quat(rot_rtab, pos_rtab)
            
        publisher.publish(observations)
        cnt_pub += 1
        # FIXME: consider to move publishing point cloud to data process function
        # in agents
        # NOTE: test if rgbd data has been received by rtabmap
        # if sub_cloud.has_cloud():
        #     o3d_cloud = sub_cloud.get_cloud()
        #     cnt_sub += 1

        #     num_points = np.asarray(o3d_cloud.points).shape[0]
        #     rospy.loginfo(
        #         "Publish: {} th images, "
        #         "Subscribe: {} th point cloud, "
        #         "{} points.".format(cnt_pub, cnt_sub, num_points)
        #     )

        # if VISUALIZE and cnt_sub == 11:
        #     coo = o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     o3d_cloud = transformation.o3d_rtab2mp3d(o3d_cloud)
        #     o3d.visualization.draw_geometries([coo, o3d_cloud])

        # message processing and synchronization should be down in agent.act()
        action = agent.act()
        rate.sleep()


if __name__ == "__main__":
    main()

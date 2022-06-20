import math
import os
import time
from collections import deque
import agents.utils.visualization as vu
import cv2
import envs.utils.pose as pu
import numpy as np
# from habitat_sim import Simulator
# from matplotlib import cm
# from matplotlib.pyplot import grid
# from PIL import Image
from scipy.spatial.transform import Rotation as R
import skimage.morphology

# ros packages
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from utils.publishers import HabitatObservationPublisher
from utils.transformation import (
    publish_agent_init_tf,
    publish_static_base_to_cam,
)
# from utils.cam_utils import get_point_from_pixel
from arguments import get_args
from agents.utils.utils_frontier_explore import (
    frontier_goals, 
    dist_odom_to_goal,
    update_odom_by_action,
    target_goals,
)

from agents.utils.ros_utils import (
    safe_call_reset_service, 
    publish_frontiers,
    publish_object_goal,
    publish_pose
)
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from envs.utils.depth_utils import get_point_cloud_from_Y, get_camera_matrix
from envs.constants import color_palette, coco_categories
from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.utils.fmm_planner import FMMPlanner


# TODO: remove all these global default arguments
# parameters used for debuging with python debugger
DEFAULT_RATE = 1.0
DEFAULT_CAMERA_CALIB = "/home/junting/habitat_ws/src/struct-nav/habitat/scripts/envs/habitat/configs/camera_info.yaml"
# DEFAULT_CAMERA_CALIB = "/home/junting/habitat_ws/src/struct-nav/habitat/scripts/envs/habitat/configs/camera_info_gibson.yaml"
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1
VISUALIZE = False
DEBUG = False
DEBUG_VIS = False
DEBUG_WHEEL_ODOM = True

class Frontier2DDetectionAgent(ObjectGoal_Env):
    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        self.config_env = config_env
        self.rank = rank
        # agent depends on ROS and observations do not support vecpytroch
        # only intended for single process
        # assert rank == 0
        super().__init__(args, rank, config_env, dataset)
        
        # arguments
        self.map_resolution = args.map_resolution_cm / 100.0
        self.sem_model = args.sem_model
        self.success_dist = args.success_dist

        # args from config 
        self.obs_width = config_env.SIMULATOR.RGB_SENSOR.WIDTH
        self.obs_height = config_env.SIMULATOR.RGB_SENSOR.HEIGHT
        self.obs_fov = config_env.SIMULATOR.RGB_SENSOR.HFOV
        self.sensor_height = config_env.SIMULATOR.AGENT_0.HEIGHT
        self.forward_dist = config_env.SIMULATOR.FORWARD_STEP_SIZE # 0.25 by default
        self.turn_angle = config_env.SIMULATOR.TURN_ANGLE # 30 degrees by default 


        # TODO: should place 2D recognition models in a separate ROS pacakge
        # fix this before release! 
        # Initialize 2D recognition models 
        if self.sem_model == "detectron":
            self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        # TODO: fetch this from config file
        # self.action_names = {
        #     0: "stop",
        #     1: "move_forward",
        #     2: "turn_left",
        #     3: "turn_right",
        # }
        self.obs = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.goal_name = None
        self.spot_goal = False
        self.actions_queue = deque([], maxlen=4)
        
        # from ObjectGoalEnv
        self.info = {}
        self.info["distance_to_goal"] = None
        self.info["spl"] = None
        self.info["success"] = None

        ############## ROS setup #############################
        self.init_ros()

        if args.visualize or args.print_images:
            self.legend = cv2.imread("docs/legend.png")
            self.rgb_vis = None
            self.vis_image = vu.init_occ_image(
                self.goal_name
            )  # , self.legend)

    def init_ros(self):

        """initialize ros related publishers and subscribers for agent"""
        # task_config = rospy.get_param("~task_config")
        self.rate_value = rospy.get_param("~rate", DEFAULT_RATE)
        self.camera_info_file = rospy.get_param(
            "~camera_calib", DEFAULT_CAMERA_CALIB
        )
        self.wheel_odom = rospy.get_param("~wheel_odom", False)
        # assert (
        #     agent_type in AGENT_CLASS_MAPPING.keys()
        # ), f"{agent_type} not in supported agent types: {AGENT_CLASS_MAPPING.keys()}"

        # goal_radius = rospy.get_param("~goal_radius", DEFAULT_GOAL_RADIUS)
        # max_d_angle = rospy.get_param("~max_d_angle", DEFAULT_MAX_ANGLE)
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/rgb/image")
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/depth/image"
        )
        self.semantic_topic = rospy.get_param(
            "~semantic_topic", ""
        )  # "/camera/semantic/image") # not to pulish gt by default
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/rgb/camera_info"
        )
        self.true_pose_topic = rospy.get_param("~true_pose_topic", "")
        self.cloud_topic = rospy.get_param(
            "~cloud_topic", "/rtabmap/cloud_map"
        )
        # topics for planning
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.grid_map_topic = rospy.get_param(
            "~grid_map_topic", "/rtabmap/grid_map"
        )
        self.frontiers_topic = rospy.get_param(
            "~frontiers_topic", "/frontiers"
        )
        # goal_topic = rospy.get_param("~goal_topic", "/nav_goal")

        # ros pub and sub
        self.rate = rospy.Rate(self.rate_value)
        self.pub_obs = HabitatObservationPublisher(
            rgb_topic=self.rgb_topic,
            depth_topic=self.depth_topic,
            # semantic_topic=self.semantic_topic,
            camera_info_topic=self.camera_info_topic,
            true_pose_topic=self.true_pose_topic,
            camera_info_file=self.camera_info_file,
            # sim_config=self.habitat_env.sim.sim_config
        )

        self.cam_int_mat = np.array(self.pub_obs.camera_info.K).reshape((3,3))
        
        publish_static_base_to_cam(self.sensor_height)

        # action_publisher = rospy.Publisher(
        #     "habitat_action", Int32, latch=True, queue_size=100
        # )
        # publish messages to a topic using rospy.Publisher class
        # self.pub_action = rospy.Publisher("habitat_action", String, queue_size=1)

        self.sub_odom = rospy.Subscriber(
            self.odom_topic, Odometry, self.callback_odom
        )
        print(f"subscribing to {self.odom_topic}...")

        self.sub_grid_map = rospy.Subscriber(
            self.grid_map_topic, OccupancyGrid, self.callback_grid_map
        )
        print(f"subscribing to {self.grid_map_topic}...")

        self.pub_frontiers = rospy.Publisher(
            self.frontiers_topic, MarkerArray, queue_size=1
        )

        if DEBUG_WHEEL_ODOM:
            self.pub_wheel_odom_pose = rospy.Publisher(
                "~wheel_odom", PoseStamped, queue_size=1
            )

        self.rate = rospy.Rate(self.rate_value)

        # cached messages
        self.odom_msg = None
        self.grid_map_msg = None
        self.last_odom_msg_time = 0.0  # last odom_msg timestamp
        self.last_grid_map_msg_time = 0.0  # last grid_map_msg timestamp
        self.last_update_time = 0.0
        self.last_odom_mat = np.eye(4)

        # sub_cloud = PointCloudSubscriber(cloud_topic)

        self.cnt_pub = 0
        self.cnt_action = 0

    def reset(self, publish_obs=True):
        args = self.args

        # 1. call reset function of ObjectGoal_Env to reset scene in habitat-lab
        obs, info = super().reset()

        # 2. reset map and odom in rtabmap_ros
        safe_call_reset_service("/rtabmap/reset")  # reset map
        safe_call_reset_service("/rtabmap/reset_odom")  # reset odometry
        # Initialize TF tree with ground truth init pose (if any)
        # sim_agent = sim.get_agent(0)
        # transformation.publish_agent_init_tf(sim_agent)

        # 3. initialize episode variables
        # TODO: add map size to ros params ! 
        # map_shape = (
        #     args.map_size_cm // args.map_resolution_cm,
        #     args.map_size_cm // args.map_resolution_cm,
        # )
        map_shape = int(20 / 0.05)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.cnt_pub = 0
        self.cnt_action = 0
        self.spot_goal = False
        self.last_action = None
        self.last_odom_mat = np.eye(4)
        self.actions_queue = deque([], maxlen=4)
        # self.curr_loc = [
        #     args.map_size / 2.0,
        #     args.map_size / 2.0,
        #     0.0,
        # ]
        
        if args.visualize or args.print_images:
            self.vis_image = vu.init_occ_image(self.goal_name)

        # 4. publish observation if publish_obs set to True
        obs = self._preprocess_obs(obs)
        if publish_obs:
            self.pub_obs.publish(obs)
            self.cnt_pub += 1
            # if DEBUG:
            #     print(f"[DEBUG] Published {self.cnt_pub} observations.")
        
        if args.visualize or args.print_images:
            self.legend = cv2.imread("docs/legend.png")
            self.rgb_vis = None
            self.vis_image = vu.init_occ_image(
                self.goal_name
            )  # , self.legend)

        self.obs = obs 
        return obs, info

    def parse_ros_messages(self):
        odom_msg: Odometry = self.odom_msg
        grid_map_msg: OccupancyGrid = self.grid_map_msg

        # update timestamps for last processed messages
        self.last_grid_map_msg_time = grid_map_msg.header.stamp.to_sec()
        self.last_odom_msg_time = odom_msg.header.stamp.to_sec()
        self.last_update_time = rospy.Time().now().to_sec()

        grid_map = np.array(grid_map_msg.data, dtype=np.int8).reshape(
            grid_map_msg.info.height, grid_map_msg.info.width
        )
        # self.reset(grid_map.shape)

        # TODO: add map expansion logic for following maps
        # FIXME: the present strategy is simple:
        # - when grid_map is expanded, drop all previous memory and reallocate
        # empty maps to record collisions and visited places during running
        if (
            self.collision_map is None
            or self.collision_map.shape != grid_map.shape
        ):
            self.collision_map = np.zeros_like(grid_map)
            self.visited = np.zeros_like(grid_map)
            self.visited_vis = np.zeros_like(grid_map)
            # map_resolution = grid_map_msg.info.resolution

        map_origin_position = grid_map_msg.info.origin.position
        map_origin = np.array([map_origin_position.x, map_origin_position.y])

        # NOTE: there are zero quaternions when visual odometry is lost 
        odom_pose_mat = np.zeros((4,4))
        try: # try to parse visual odometry if valid 
            # parse odometry message
            odom_pose = odom_msg.pose
            odom_pos = np.array([
                odom_pose.pose.position.x,
                odom_pose.pose.position.y,
                odom_pose.pose.position.z,
            ])
            odom_rot = R.from_quat(
                [
                    odom_pose.pose.orientation.x,
                    odom_pose.pose.orientation.y,
                    odom_pose.pose.orientation.z,
                    odom_pose.pose.orientation.w,
                ]
            ).as_matrix()
            
            # odom_map_pose: (x, y, x-y rotation angle)
            odom_map_pose = np.array(
                [odom_pose.pose.position.x, odom_pose.pose.position.y, 0.0]
            )
            odom_angle = R.from_matrix(odom_rot).as_euler("zxy", degrees=True)[0]
            odom_map_pose[2] = odom_angle
            
            odom_pose_mat[:3, :3] = odom_rot
            odom_pose_mat[:3, 3] = odom_pos
            odom_pose_mat[3, 3] = 1.0
        
        except:
            # if visual odometry invalid (lost), use internal odometry 
            odom_pose_mat = update_odom_by_action(
                self.last_odom_mat,
                self.last_action,
                forward_dist=self.forward_dist, 
                turn_angle=self.turn_angle
            )
            odom_pos = odom_pose_mat[:3, 3]
            odom_rot = odom_pose_mat[:3, :3]

            # odom_map_pose: (x, y, x-y rotation angle)
            odom_map_pose = np.copy(odom_pos)
            odom_angle = R.from_matrix(odom_rot).as_euler("zxy", degrees=True)[0]
            odom_map_pose[2] = odom_angle

        # save odom_pose_mat in case visual odom lost 
        self.last_odom_mat = odom_pose_mat

        return grid_map, map_origin, odom_map_pose, odom_pose_mat

    # wrapper function to execute one step
    def plan_act(self):
        """This function generates one step for habitat simulator

        This function receives grid_map message, odometry message from ROS,
        and generate one step in habitat simulator.

        The process is:
        1. generate frontiers and goals from Frontiers-based exploration algorithm.
        Ref: A Frontier-Based Approach for Autonomous Exploration by Yamauchi, 1997
        2. calculate path of short-term goals by Fast Marching Method (global planner)
        3. select instant action within local window by gradient on distance map (local planner)


        Args:
            self

        Returns:
            action: string of an action accepted by habitat simulator

        """
        ############ parse ros messages ################

        # FIXME: cannot receive grid_map message
        while self.odom_msg == None or self.grid_map_msg == None:
            # waiting for data
            # return "stay"
            print("waiting for first odom_msg and grid_map_msg")
            time.sleep(0.5)

        while (
            self.last_odom_msg_time == self.odom_msg.header.stamp.to_sec()
            # or self.last_grid_map_msg_time == self.grid_map_msg.header.stamp.to_sec()
        ):
            # waiting for data
            cur_time = rospy.Time().now().to_sec()
            time_diff = cur_time - self.last_update_time
            # time_diff = cur_time - max(
            #     self.last_odom_msg_time, self.last_grid_map_msg_time
            # )
            if DEBUG:
                print(
                    f"[DEBUG] waiting for message update since {time_diff} seconds ago"
                )
            time.sleep(0.5)
            # return "stay"

        grid_map, map_origin, odom_map_pose, odom_pose_mat = self.parse_ros_messages()
        map_resolution = self.map_resolution

        ################## check if target object already known ############
        if (self.goal_idx + 1) in self.obs['semantic']: # zero for background
            self.spot_goal = True

        found_goal = 0
        if self.spot_goal == False:
            ##################  generate frontiers and goals ##############
            frontiers, goals = frontier_goals(
                grid_map,
                map_origin,
                map_resolution,
                odom_map_pose[:2],
                cluster_trashhole=self.args.cluster_trashhole,
                num_goals=1,
            )
            # TODO: think about goal selection strategy
            goal_position = goals[0, :]
            publish_frontiers(frontiers, goals, self.pub_frontiers)

        else: 
            ######### if target object already observed, go directly ########## 
            # TODO: Improve this by sliding window average ! 
            if (self.goal_idx + 1) in self.obs['semantic']: 
                camera_param=get_camera_matrix(
                    self.obs_width,
                    self.obs_height,
                    self.obs_fov
                )
                # targets in 3d, goals in 2d 
                targets, target_sizes, goals = target_goals(
                    sem_img=self.obs['semantic'],
                    depth_img=self.obs['depth'],
                    goal_idx=self.goal_idx + 1,  # zero for background
                    cam_param=camera_param,
                    odom_pose_mat=odom_pose_mat,
                    num_goals=1,
                )
                # set z-value to target size for visualization
                targets[:, 2] = target_sizes 
                goal_position = goals[0, :]
                publish_frontiers(targets, goals, self.pub_frontiers)
                self.last_goal_position = goal_position

            else: # object disappears from observation 
                goal_position = self.last_goal_position

            # calculate if goal is found within 
            dist2goal = dist_odom_to_goal(odom_pose_mat, goal_position, dist_2d=True)
            if dist2goal < self.success_dist * 0.8: # leave margin for noisy odometry
                found_goal = 1
            # only publish the only goal and empty frontiers 
            publish_object_goal([goal_position], self.pub_frontiers)
            # publish point cloud if needed here

        ################### generate action by planners ################
        # convert ros-format grid_map to object-goal-nav style maps
        occupancy_map = (grid_map == 100).astype(np.float32)
        explore_map = (grid_map >= 0).astype(np.float32)

        # convert (x,y) goal center to goal map
        goal_map = np.zeros_like(grid_map)
        pixel_position = (
            (goal_position - map_origin) / map_resolution
        ).astype(int)
        # NOTE: (x,y) is (col, row) in image
        goal_map[pixel_position[1], pixel_position[0]] = 1.0

        p_input = {}
        p_input["map_pred"] = occupancy_map
        p_input["exp_pred"] = explore_map
        p_input["pose_pred"] = np.zeros(7)
        # NOTE: in plan() function, the position is in map frame
        p_input["pose_pred"][:2] = odom_map_pose[:2] - map_origin
        # NOTE: in plan() function, xy-angle is calculated in pixel_x-pixel_y (y-x)
        # frame, instead of world xy-frame, need to plus 90 degree for correction
        p_input["pose_pred"][2] = odom_map_pose[2] + 90
        p_input["pose_pred"][3:] = np.array(
            [0, grid_map.shape[0], 0, grid_map.shape[1]]
        )
        p_input["goal"] = goal_map
        p_input["found_goal"] = found_goal
        p_input["wait"] = False

        action = self.plan(p_input)
        self.last_action = action  # needed for collision check
        self.actions_queue.append(action) # needed for stucking check 

        if self.args.visualize or self.args.print_images:
            self._visualize(p_input)

        return action
        # return self.action_names[action]

    def plan_act_and_preprocess(self):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # # plan
        # if planner_inputs["wait"]:
        #     self.last_action = None
        #     self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        #     return np.zeros(self.obs.shape), 0.0, False, self.info

        # # Reset reward if new long-term goal
        # if planner_inputs["new_goal"]:
        #     self.info["g_reward"] = 0

        action = self.plan_act()

        # if self.args.visualize or self.args.print_images:
        #     self._visualize(planner_inputs)
        if action >= 0:

            # act
            action = {"action": action}
            self.rate.sleep()
            obs, rew, done, info = super().step(action)
            self.last_action = action["action"]

            # preprocess obs and publish it 
            obs = self._preprocess_obs(obs)
            self.pub_obs.publish(obs)
            self.obs = obs
            self.info = info

            # info["g_reward"] += rew

            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            print("WARNING: None action executed!")
            return {}, 0.0, False, self.info
            # return np.zeros(self.obs_shape), 0.0, False, self.info

    def plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs["map_pred"])
        goal = planner_inputs["goal"]

        # Get pose prediction and global policy planning window
        # FIXME: neural slam has local planning window
        # now planning is running on global map, not scalable
        # fix this problem later
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs[
            "pose_pred"
        ]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        assert gx1 == 0 and gy1 == 0
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r / self.map_resolution - gx1),
            int(c / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if args.visualize or args.print_images:
            # Get last loc
            if self.last_loc != None:
                last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
                r, c = last_start_y, last_start_x
                last_start = [
                    int(r / self.map_resolution - gx1),
                    int(c / self.map_resolution - gy1),
                ]
                last_start = pu.threshold_poses(last_start, map_pred.shape)
                self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
                    last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
                )

        # FIXME: add collision check
        # Collision check
        # if self.last_action == 1:
        #     x1, y1, t1 = self.last_loc
        #     x2, y2, _ = self.curr_loc
        #     buf = 4
        #     length = 2

        #     if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
        #         self.col_width += 2
        #         if self.col_width == 7:
        #             length = 4
        #             buf = 3
        #         self.col_width = min(self.col_width, 5)
        #     else:
        #         self.col_width = 1

        #     dist = pu.get_l2_distance(x1, x2, y1, y2)
        #     if dist < args.collision_threshold:  # Collision
        #         width = self.col_width
        #         for i in range(length):
        #             for j in range(width):
        #                 wx = x1 + 0.05 * (
        #                     (i + buf) * np.cos(np.deg2rad(t1))
        #                     + (j - width // 2) * np.sin(np.deg2rad(t1))
        #                 )
        #                 wy = y1 + 0.05 * (
        #                     (i + buf) * np.sin(np.deg2rad(t1))
        #                     - (j - width // 2) * np.cos(np.deg2rad(t1))
        #                 )
        #                 r, c = wy, wx
        #                 r, c = (
        #                     int(r / self.map_resolution),
        #                     int(c / self.map_resolution),
        #                 )
        #                 [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
        #                 self.collision_map[r, c] = 1

        stg, stop = self._get_stg(
            map_pred, start, np.copy(goal), planning_window
        )

        # visualize long-term/short-term goal and agent pose
        if DEBUG_VIS:
            from matplotlib import cm
            from matplotlib import pyplot as plt
            from skimage.draw import disk, rectangle

            # import agents.utils.visualization as vu
            # import cv2
            # plt.rcParams["figure.figsize"] = [7.00, 3.50]
            plt.rcParams["figure.autolayout"] = True

            # visualize map
            exp_pred = np.rint(planner_inputs["exp_pred"])
            vis_map = np.copy(map_pred)
            vis_map[exp_pred == 0] = 0  # unknown
            vis_map[(map_pred == 0) & (exp_pred == 1)] = 1  # free space
            vis_map[map_pred > 0] = 4  # obstacle
            goal_vis = skimage.morphology.binary_dilation(
                goal, skimage.morphology.disk(3)
            )

            # visualize long/short-term goals
            vis_map[goal_vis == 1] = 2  # disk for long-term goal
            rr, cc = rectangle(
                np.array(stg) - 2, extent=7, shape=vis_map.shape
            )
            vis_map[
                rr.astype(int), cc.astype(int)
            ] = 3  # rect for short-term goal

            # visualize agent
            pos = [start[0], start[1], np.deg2rad(start_o)]
            agent_arrow = vu.get_contour_points(pos, origin=(0, 0), size=5)
            color = (255, 0, 0)
            plt.fill(agent_arrow[:, 1], agent_arrow[:, 0], facecolor="red")

            plt.imshow(vis_map, origin="lower")

        # Deterministic Local Policy
        if stop and planner_inputs["found_goal"] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(
                math.atan2(stg_x - start[0], stg_y - start[1])
            )
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            # if relative_angle > self.turn_angle / 2.0:
            #     action = 3  # Right
            # elif relative_angle < -self.turn_angle / 2.0:
            #     action = 2  # Left
            # else:
            #     action = 1  # Forward
            # NOTE: give local controller more tolerance in face of odometry noise
            # TODO: tune the factor for relative angle to turn 
            if relative_angle > 0.7 * self.turn_angle:
                action = 3  # Right
            elif relative_angle < -0.7 * self.turn_angle:
                action = 2  # Left
            else:
                action = 1  # Forward

            # overwrite local planner if agent is stuck locally
            if self.actions_queue.count(3) + self.actions_queue.count(2) >=4:
                action = 1 # Execute forward to get away from left-right swing cycle

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        # FIXME: collision_map and visited_map not functioning now
        traversible = (
            skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem)
            != True
        )
        traversible[
            self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1
        ] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(goal, selem) != True
        goal = 1 - goal * 1.0
        planner.set_multi_goal(goal)

        state = [start[0] - x1, start[1] - y1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)
        stg_x, stg_y = stg_x + x1, stg_y + y1
        # original code here: 
        # state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # stg_x, stg_y, _, stop = planner.get_short_term_goal(state)
        # stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _visualize(self, inputs):
        args = self.args
        dump_dir = f"{args.dump_location}/dump/{args.exp_name}/"
        ep_dir = f"{dump_dir}/episodes/"
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        occupancy_map = inputs["map_pred"]
        explore_map = inputs["exp_pred"]
        map_pred = np.zeros_like(occupancy_map)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs["pose_pred"]
        goal = inputs["goal"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        occpancy_mask = occupancy_map == 1
        # unknown_mask = explore_map == 0
        free_mask = (explore_map == 1) & (occupancy_map != 1)
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        # map_pred[unknown_mask] = 0 # default is 0
        map_pred[free_mask] = 1
        map_pred[occpancy_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True
        goal_mask = goal_mat == 1
        map_pred[goal_mask] = 2

        # convert heat map to rgb image
        map_pred_vis = (map_pred / 3.0 * 255).astype(
            np.uint8
        )  # convert to CV_8UC1 format
        map_pred_vis = np.flipud(map_pred_vis)  # flip y axis
        map_pred_vis = cv2.applyColorMap(map_pred_vis, cv2.COLORMAP_VIRIDIS)
        # color_pal = [int(x * 255.0) for x in color_palette]
        # sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        # sem_map_vis.putpalette(color_pal)
        # sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        # sem_map_vis = sem_map_vis.convert("RGB")
        # sem_map_vis = np.flipud(sem_map_vis)

        # map_pred_vis = map_pred_vis[:, [2, 1, 0]]
        map_pred_vis = cv2.resize(
            map_pred_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        # self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150, :] = map_pred_vis

        pos = (
            (start_x / self.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y / self.map_resolution + gx1)
            * 480
            / map_pred.shape[1],
            np.deg2rad(-start_o),
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (
            int(color_palette[11] * 255),
            int(color_palette[10] * 255),
            int(color_palette[9] * 255),
        )
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread 0", self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            fn = f"{ep_dir}/Vis-{timestamp}.png"
            cv2.imwrite(fn, self.vis_image)

    # TODO: add object detection / segmentation models here
    def _preprocess_obs(self, obs, use_seg=True):

        # args = self.args
        # not using VecPyTorch to wrap environemnts
        # obs = obs.transpose(1, 2, 0)
        # rgb = obs[:, :, :3]
        # depth = obs[:, :, 3:4]
        if self.sem_model == "ground_truth":
            # by default, semantic sensor is added to agent
            assert 'semantic' in obs
            
        if self.sem_model == "detectron":
            # overwrite semantic image with detectron prediction 
            rgb = obs['rgb']
            sem_seg_pred, _ = self.sem_pred.get_prediction(rgb.astype(np.uint8))
            semantic_image = np.zeros(rgb.shape[:2])
            for name, label in coco_categories.items():
                # NOTE: zero for background 
                semantic_image[sem_seg_pred[:,:,label] == 1] = label + 1 
            obs['semantic'] = semantic_image

        if self.wheel_odom or DEBUG_WHEEL_ODOM:
            if self.last_action:
                updated_odom_mat = update_odom_by_action(self.last_odom_mat, self.last_action)
                obs['odom_pose_mat'] = updated_odom_mat
            else:
                obs['odom_pose_mat'] = self.last_odom_mat
        
        if DEBUG_WHEEL_ODOM:
            publish_pose(obs['odom_pose_mat'], self.pub_wheel_odom_pose)

        # depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        # ds = args.env_frame_width // args.frame_width  # Downscaling factor
        # if ds != 1:
        #     rgb = np.asarray(self.res(rgb.astype(np.uint8)))
        #     depth = depth[ds // 2 :: ds, ds // 2 :: ds]
        #     sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

        # depth = np.expand_dims(depth, axis=2)
        # state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(
        #     2, 0, 1
        # )
        # passthrough
        return obs

    def callback_odom(self, odom_msg: Odometry):
        self.odom_msg = odom_msg

    def callback_grid_map(self, grid_map_msg: OccupancyGrid):
        self.grid_map_msg = grid_map_msg

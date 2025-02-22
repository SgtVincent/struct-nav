import math
import os
import time
from collections import deque
import quaternion as qt
from matplotlib.transforms import Transform
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
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker, MarkerArray

from utils.publishers import HabitatObservationPublisher, PointCloudPublisher
from utils.transformation import (
    publish_agent_init_tf,
    publish_static_base_to_cam,
    pose_habitat2rtabmap
)
# from utils.cam_utils import get_point_from_pixel
from arguments import get_args
from agents.utils.utils_frontier_explore import (
    frontier_goals, 
    dist_odom_to_goal,
    copy_map_overlap,
    update_odom_by_action,
    target_goals,
)

from agents.utils.ros_utils import (
    safe_call_reset_service, 
    safe_get_map_service,
    publish_frontiers,
    publish_targets,
    publish_pose
)
from envs.utils.depth_utils import get_point_cloud_from_Y, get_camera_matrix
from envs.constants import color_palette, coco_categories
from envs.habitat.objectgoal_env import ObjectGoal_Env
from envs.utils.fmm_planner import FMMPlanner

# DEBUG flags: 
DEBUG = False
DEBUG_VIS = False
DEBUG_WHEEL_ODOM = False
DEBUG_DETECT = False

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
        # self.goal_policy = args.goal_policy
        # only geo acceptable for this baseline 
        self.goal_policy = "geo"

        # args from config 
        self.obs_width = config_env.SIMULATOR.RGB_SENSOR.WIDTH
        self.obs_height = config_env.SIMULATOR.RGB_SENSOR.HEIGHT
        self.obs_fov = config_env.SIMULATOR.RGB_SENSOR.HFOV
        self.obs_min_depth = config_env.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.obs_max_depth = config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.sensor_height = config_env.SIMULATOR.AGENT_0.HEIGHT
        self.forward_dist = config_env.SIMULATOR.FORWARD_STEP_SIZE # 0.25 by default
        self.turn_angle = config_env.SIMULATOR.TURN_ANGLE # 30 degrees by default 
        
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
        self.goal_map = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.map_origin = None
        self.map_shape = None
        self.last_action = None
        self.count_forward_actions = None
        self.map_reset = False
        self.goal_name = None
        self.spot_goal = 0
        
        

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
        super().init_ros()
        
        # ros pub and sub
        self.rate = rospy.Rate(self.rate_value)
        
        self.pub_obs = HabitatObservationPublisher(
            rgb_topic=self.rgb_topic,
            depth_topic=self.depth_topic,
            semantic_topic=self.semantic_topic,
            camera_info_topic=self.camera_info_topic,
            ground_truth_odom_topic=self.ground_truth_odom_topic,
            true_pose_topic=self.true_pose_topic,
            camera_info_file=self.camera_info_file,
            wheel_odom_frame_id=self.whee_odom_frame_id,
            sim_config=self.habitat_env.sim.sim_config
        )

        self.cam_int_mat = np.array(self.pub_obs.camera_info.K).reshape((3,3))
        publish_static_base_to_cam(self.sensor_height)

        # action_publisher = rospy.Publisher(
        #     "habitat_action", Int32, latch=True, queue_size=100
        # )
        # publish messages to a topic using rospy.Publisher class
        # self.pub_action = rospy.Publisher("habitat_action", String, queue_size=1)
        
        self.sub_odom = None
        if not self.ground_truth_odom:
            self.sub_odom = rospy.Subscriber(
                self.odom_topic, Odometry, self.callback_odom
            )
            rospy.loginfo(f"subscribing to predicted odom {self.odom_topic}...")

        self.sub_grid_map = None
        if self.map_update_mode == "listen":
            rospy.loginfo(f"Agent running in listen mode to update map")

            self.sub_grid_map = rospy.Subscriber(
                self.grid_map_topic, OccupancyGrid, self.callback_grid_map
            )
            rospy.loginfo(f"subscribing to {self.grid_map_topic}...")
        elif self.map_update_mode == "request":
            rospy.loginfo(f"Agent running in request mode to update map")
            # skip subscriber initialization 
        else:
            raise NotImplementedError

        # tf listener 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_frontiers = rospy.Publisher(
            self.frontiers_topic, MarkerArray, queue_size=1
        )

        if DEBUG_WHEEL_ODOM:
            self.pub_wheel_odom_pose = rospy.Publisher(
                "~wheel_odom", PoseStamped, queue_size=1
            )
        
        self.pub_detected_goal_pts = None # do not publish 
        if DEBUG_DETECT:
            self.pub_detected_goal_pts = PointCloudPublisher(
                "~detected_goal_pts"
            )

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
        
        success = False
        while(not success):
            args = self.args
            # 1. call reset function of ObjectGoal_Env to reset scene in habitat-lab
            obs, info = super().reset()
            # NOTE: sometimes habitat returns empty observations from start for unknown error, skip the episode 
            if not np.any(obs['rgb']):
                rospy.logwarn("Habitat returns black observations from start, skip the episode")
                continue
            
            # 2. initialize episode variables
            
            # map-related info should be initialized by first ros message in an episode
            # map_shape = self.initial_map_size
            self.map_reset = True # flag to re-initialize map in process_ros_messages
            self.collision_map = None
            self.visited = None
            self.visited_vis = None
            self.goal_map = None
            self.curr_loc = None
            self.map_origin = None
            self.map_shape = None
            self.col_width = 1
            self.count_forward_actions = 0
            self.cnt_pub = 0
            self.cnt_action = 0
            self.spot_goal = 0
            self.last_action = None
            self.last_odom_mat = np.eye(4)
            
            if self.ground_truth_odom:
                # always publish current pose in initial pose frame 
                # this "normalization" avoids a lot of configurations in rtabmap
                self.init_pos = info['agent_pose'].position
                self.init_rot = info['agent_pose'].rotation

            if args.visualize or args.print_images:
                self.legend = cv2.imread("docs/legend.png")
                self.rgb_vis = None
                self.vis_image = vu.init_occ_image(
                    self.goal_name
                )  # , self.legend)

            ############ reset ROS-related class ###################
            
            # 3. reset map and odom in rtabmap_ros
            safe_call_reset_service("/rtabmap/reset")  # reset map
            # safe_call_reset_service("/rtabsem/reset") 
            if self.sub_odom:
                safe_call_reset_service("/rtabmap/reset_odom")  # reset odometry
            time.sleep(1.0)
            
            # 4. publish observation
            obs = self._preprocess_obs(obs, info)
            if publish_obs:
                self.pub_obs.publish(obs)
                self.cnt_pub += 1
                # self.rate.sleep()
                time.sleep(1/self.rate_value)
                # if DEBUG:
                #     print(f"[DEBUG] Published {self.cnt_pub} observations.")
            
            # NOTE: 1. after resetting the episode, if not to request new grid map, then
            # inconsistency between old grid map message from last episode and 
            # observations of this episode will lead to error 
            # NOTE: 2. there are some cases rtabmap does not have time to initialize 
            # new grid map, keep publishing observations until new map ready 
            timeout=5.
            map_ready = False
            max_fail = 3
            fail_cnt = 0
            
            while(not map_ready and fail_cnt < max_fail):
                try:
                    rospy.wait_for_service("/rtabmap/get_map", timeout=timeout)
                    response = rospy.ServiceProxy("/rtabmap/get_map", GetMap)()
                    map_ready = True
                    success = True
                    self.grid_map_msg = response.map
                    
                except rospy.ServiceException as e:
                    fail_cnt += 1
                    rospy.logwarn(f"Get map by calling /rtabmap/get_map failed: {e}")
                    # safe_call_reset_service("/rtabmap/reset")  # reset map
                    # safe_call_reset_service("/rtabmap/reset_odom")  # reset odometry
                    # time.sleep(1.0)
                    rospy.logwarn("Republishing observations...")
                    self.pub_obs.publish(obs)
                    # self.rate.sleep()
                    time.sleep(1/self.rate_value)
                    
        self.obs = obs 
        self.info = info
        return obs, info


    def process_ros_messages(self):
        
        # get odometry update  
        if not self.ground_truth_odom:
            while (
                self.odom_msg == None or 
                self.last_odom_msg_time == self.odom_msg.header.stamp.to_sec()
                # or self.last_grid_map_msg_time == self.grid_map_msg.header.stamp.to_sec()
            ):
                # waiting for data
                cur_time = rospy.Time().now().to_sec()
                time_diff = cur_time - self.last_update_time
                if DEBUG:
                    print(
                        f"[DEBUG] waiting for odometry message update since {time_diff} seconds ago"
                    )
                # self.rate.sleep()
                time.sleep(1/self.rate_value)

            odom_msg: Odometry = self.odom_msg
            self.last_odom_msg_time = odom_msg.header.stamp.to_sec()
        
        # get map update 
        if self.map_update_mode == "listen":
            while (
                self.grid_map_msg == None
                # or self.last_grid_map_msg_time == self.grid_map_msg.header.stamp.to_sec()
            ):
                # waiting for map data 
                cur_time = rospy.Time().now().to_sec()
                time_diff = cur_time - self.last_update_time
                # self.rate.sleep()
                time.sleep(1/self.rate_value)

            grid_map_msg: OccupancyGrid = self.grid_map_msg
            # update timestamps for last processed messages
            self.last_grid_map_msg_time = grid_map_msg.header.stamp.to_sec()

        elif self.map_update_mode == "request":
            grid_map_msg: OccupancyGrid = safe_get_map_service()
            # odom_msg: TransformStamped = self.tf_buffer.lookup_transform(
            #     "base_link", "map", rospy.Time(0))

        self.last_update_time = rospy.Time().now().to_sec()

        ################ process grid_map udpate ########################
        grid_map, map_origin = self.process_map_msg(grid_map_msg)

        ################ process odometry update ######################
        if self.ground_truth_odom:
            odom_pose_mat = self.obs['true_odom_mat']
            odom_map_pose = odom_pose_mat[:3, 3]
            odom_angle = R.from_matrix(odom_pose_mat[:3,:3]).as_euler("zxy", degrees=True)[0]
            odom_map_pose[2] = odom_angle
        else:
            odom_map_pose, odom_pose_mat = self.process_odom_msg(odom_msg)
        # save odom_pose_mat in case visual odom lost 
        self.last_odom_mat = odom_pose_mat

        return grid_map, map_origin, odom_map_pose, odom_pose_mat

    def process_map_msg(self, grid_map_msg: OccupancyGrid):
        
        grid_map = np.array(grid_map_msg.data, dtype=np.int8).reshape(
            grid_map_msg.info.height, grid_map_msg.info.width
        )
        map_origin_position = grid_map_msg.info.origin.position
        map_origin = np.array([map_origin_position.x, map_origin_position.y])

        # if reset() function is called, re-initialize map here 
        if self.map_reset:
            self.collision_map = np.zeros_like(grid_map)
            self.visited = np.zeros_like(grid_map)
            self.visited_vis = np.zeros_like(grid_map)
            self.goal_map = np.zeros_like(grid_map)
            self.curr_loc = None # used to detect collision
            self.map_origin = map_origin
            self.map_shape = grid_map.shape
            self.map_reset = False

        # if map has been expanded or shifted by rtabmap, re-initialize map 
        # and copy overlapping areas
        o_shift = np.round(
            (self.map_origin - map_origin) / self.map_resolution
        ).astype(int)
        
        if self.map_shape != grid_map.shape or np.any(o_shift != 0):
            
            old_collision_map = self.collision_map
            old_visited = self.visited
            old_visited_vis = self.visited_vis
            old_goal_map = self.goal_map
            old_curr_loc = self.curr_loc # [x, y, o]
            old_map_origin = self.map_origin
            # old_map_shape = self.map_shape
            
            # create expanded new maps
            self.collision_map = np.zeros_like(grid_map)
            self.visited = np.zeros_like(grid_map)
            self.visited_vis = np.zeros_like(grid_map)
            self.goal_map = np.zeros_like(grid_map)

            # # calculate mapping from old map to new map
            # # NOTE: new map_origin could be larger than old map_origin thus 
            # # making this origin shift negative 
            # self.curr_loc[:2] = old_curr_loc[:2] + old_map_origin - map_origin
            # r, c = o_shift[1], o_shift[0]
            
            # copy overlaping areas on old map to new map
            copy_map_overlap(old_map_origin, old_collision_map, map_origin, 
                self.collision_map, self.map_resolution)
            copy_map_overlap(old_map_origin, old_visited, map_origin, 
                self.visited, self.map_resolution)
            copy_map_overlap(old_map_origin, old_visited_vis, map_origin, 
                self.visited_vis, self.map_resolution)
            copy_map_overlap(old_map_origin, old_goal_map, map_origin, 
                self.goal_map, self.map_resolution)

            self.map_origin = map_origin
            self.map_shape = grid_map.shape
        
        return grid_map, map_origin

    def process_odom_msg(self, odom_msg):
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
        
        return odom_map_pose, odom_pose_mat            

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
        ############ process ros messages ################

        grid_map, map_origin, odom_map_pose, odom_pose_mat = self.process_ros_messages()
        # map_resolution = self.map_resolution

        ################## check if target object already known ############
        if (self.goal_idx + 1) in self.obs['semantic']: # zero for background
            self.spot_goal = 1

        if self.spot_goal == 0:
            ##################  generate frontiers and goals ##############
            # sometimes get empty map due to synchronization problem
            # then skip this frame  
            try:
                frontiers, goals, goal_map = frontier_goals(
                    grid_map,
                    map_origin,
                    self.map_resolution,
                    odom_map_pose[:2],
                    cluster_trashhole=self.args.cluster_trashhole,
                    num_goals=1,
                    goal_policy=self.goal_policy,
                    collision_map=self.collision_map,
                    visited_map=self.visited,
                )
            except:
                frontiers, goals, goal_map = [], [], np.zeros_like(grid_map)

            if len(frontiers) > 0:
                publish_frontiers(frontiers, goals, self.pub_frontiers)

        else: 
            ######### if target object already observed, go directly ########## 
            if (self.goal_idx + 1) in self.obs['semantic']: 
                camera_param=get_camera_matrix(
                    self.obs_width,
                    self.obs_height,
                    self.obs_fov
                )
                # targets in 3d, goals in 2d 
                targets, goal_map = target_goals(
                    grid_map,
                    map_origin,
                    self.map_resolution,
                    sem_img=self.obs['semantic'],
                    depth_img=self.obs['depth'],
                    goal_idx=self.goal_idx + 1,  # zero for background
                    cam_param=camera_param,
                    odom_pose_mat=odom_pose_mat,
                    max_depth=self.obs_max_depth,
                    min_depth=self.obs_min_depth,
                    # pub_goal_pts=self.pub_detected_goal_pts,
                    sample_targets=100,
                )
                # set z-value to unified size for visualization
                targets[:, 2] = 1. 
                publish_targets(targets, self.pub_frontiers)
                # merge new goal map with old 
                # goal_map should be updated in process_ros_messages if shape changed 
                assert grid_map.shape == self.goal_map.shape
                self.goal_map[goal_map==1.] = 1.
                goal_map = self.goal_map

            else: # object disappears from observation 
                goal_map = self.goal_map

            # calculate if goal is found within 
            # dist2goal = dist_odom_to_goal(odom_pose_mat, goal_position, dist_2d=True)
            # if dist2goal < self.success_dist * 0.8: # leave margin for noisy odometry
            #     found_goal = 1


        ################### generate action by planners ################
        # convert ros-format grid_map to object-goal-nav style maps
        occupancy_map = (grid_map == 100).astype(np.float32)
        explore_map = (grid_map >= 0).astype(np.float32)

        # convert (x,y) goal center to goal map
        # goal_map = np.zeros_like(grid_map)
        # pixel_position = (
        #     (goal_position - map_origin) / map_resolution
        # ).astype(int)
        # # NOTE: (x,y) is (col, row) in image
        # if occupancy_map[pixel_position[1], pixel_position[0]]: 
        #     # if goal not reachable, then find closest pixel as new goal 
        #     rs, cs = np.where(occupancy_map == 0.)
        #     free_locs = np.stack([cs, rs], axis=1)
        #     closest_idx = np.argmin(
        #         np.linalg.norm(free_locs - pixel_position, axis=1)
        #     )
        #     closest_loc = free_locs[closest_idx, :]
        #     goal_map[closest_loc[1], closest_loc[0]] = 1.0
            
        # else:
        #     goal_map[pixel_position[1], pixel_position[0]] = 1.0

        p_input = {}
        p_input["map_pred"] = occupancy_map
        p_input["exp_pred"] = explore_map
        p_input["pose_pred"] = np.zeros(7)
        # NOTE: in plan() function, the position is in map frame
        p_input["pose_pred"][:2] = odom_map_pose[:2] - map_origin
        # NOTE: in plan() function, xy-angle is calculated in pixel_x-pixel_y (y-x)
        # frame, instead of world xy-frame, need to plus 90 degree for correction
        # e.g., initial xy-angle from odom is 0 degree, while in map it is 90 degree (+y direction)
        p_input["pose_pred"][2] = odom_map_pose[2] + 90
        p_input["pose_pred"][3:] = np.array(
            [0, grid_map.shape[0], 0, grid_map.shape[1]]
        )
        p_input["goal"] = goal_map
        p_input["found_goal"] = self.spot_goal
        p_input["wait"] = False

        action = self.plan(p_input)
        self.last_action = action  # needed for collision check

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
            obs, rew, done, info = super().step(action)
            self.last_action = action["action"]

            # preprocess obs and publish it 
            obs = self._preprocess_obs(obs, info)
            self.pub_obs.publish(obs)
            self.obs = obs
            self.info = info
            # self.rate.sleep()
            time.sleep(1/self.rate_value)

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

        if np.all(goal == 0) and planner_inputs["found_goal"] == 0:
            action = 0  # Stop
            rospy.logwarn("Scene fully explored while no goal detected, stop episode")
            return action 

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

        # Collision check
        if self.last_action == 1 and self.last_loc is not None:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * (
                            (i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1))
                        )
                        wy = y1 + 0.05 * (
                            (i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1))
                        )
                        r, c = wy, wx
                        r, c = (
                            int(r / self.map_resolution),
                            int(c / self.map_resolution),
                        )
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

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
        action = self._get_action(start, start_o, stg, stop, planner_inputs)
        
        return action
    
    # def _preprocess_obs(self, obs, info=None):

    #     # preprocess broken meshes (0-depth) in depth image 
    #     self._preprocess_depth(obs)

    #     self._preprocess_sem(obs)

    #     # add pseudo wheel odometry pose to observations 
    #     if self.wheel_odom or DEBUG_WHEEL_ODOM:
    #         if self.last_action:
    #             updated_odom_mat = update_odom_by_action(self.last_odom_mat, self.last_action)
    #             obs['odom_pose_mat'] = updated_odom_mat
    #         else:
    #             obs['odom_pose_mat'] = self.last_odom_mat
        
    #     # add ground truth pose to observations
    #     if self.ground_truth_odom:
    #         true_state = info['agent_pose']
    #         true_pos_rtab, true_rot_rtab = pose_habitat2rtabmap(
    #             true_state.position, 
    #             true_state.rotation,
    #             self.init_pos,
    #             self.init_rot
    #         )
            
    #         true_odom_mat = np.zeros((4,4), dtype=float)
    #         true_odom_mat[:3, :3] = qt.as_rotation_matrix(true_rot_rtab)
    #         true_odom_mat[:3, 3] = true_pos_rtab
    #         true_odom_mat[3, 3] = 1.0
    #         obs['true_odom_mat'] = true_odom_mat
            
    #     # if DEBUG_WHEEL_ODOM:
    #     #     publish_pose(obs['odom_pose_mat'], self.pub_wheel_odom_pose)
        
    #     # ds = args.env_frame_width // args.frame_width  # Downscaling factor
    #     # if ds != 1:
    #     #     rgb = np.asarray(self.res(rgb.astype(np.uint8)))
    #     #     depth = depth[ds // 2 :: ds, ds // 2 :: ds]
    #     #     sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

    #     # depth = np.expand_dims(depth, axis=2)
    #     # state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(
    #     #     2, 0, 1
    #     # )
    #     # passthrough
    #     return obs

    def callback_odom(self, odom_msg: Odometry):
        self.odom_msg = odom_msg

    def callback_grid_map(self, grid_map_msg: OccupancyGrid):
        self.grid_map_msg = grid_map_msg

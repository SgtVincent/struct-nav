import enum
import math
import os
import time

import agents.utils.visualization as vu
import cv2
import envs.utils.pose as pu
import numpy as np
from scipy.spatial.transform import Rotation as R
import skimage.morphology
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
from habitat_sim import geo

# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from agents.utils.arguments import get_args
from agents.utils.utils_frontier_explore import frontier_goals
from envs.constants import color_palette
from envs.habitat.objectgoal_env import ObjectGoalEnv
from envs.utils.fmm_planner import FMMPlanner
from geometry_msgs.msg import PoseStamped
from habitat_sim import Simulator
from matplotlib import cm
from matplotlib.pyplot import grid
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from PIL import Image
from torchvision import transforms
from yaml import Mark

# ros packages
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, PoseStamped, Point, Quaternion
import tf2_geometry_msgs

"""
FrontierExploreAgent arguments:
------------------------------
args.map_resolution
args.visualize
args.print_images
args.map_size_cm
args.collision_threshold
args.turn_angle
args.min_depth
args.max_depth
args.env_frame_width 
args.frame_width
args.dump_location
args.exp_name
"""

DEBUG_VIS = False  # visualize frontiers, goals, odom on 2D grid map
DEBUG_TF = True  # print transformation from rtabmap to habitat w.r.t. time


class FrontierExploreAgent:
    def __init__(self, args, sim: Simulator):

        self.args = args
        # super().__init__(args, rank, config_env, dataset)
        self.sim = sim

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.action_names = {
            0: "stay",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
        }
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

        # from ObjectGoalEnv
        self.info = {}
        self.info["distance_to_goal"] = None
        self.info["spl"] = None
        self.info["success"] = None

        ############## ROS setup #############################
        self.odom_topic = self.args.odom_topic
        self.grid_map_topic = self.args.grid_map_topic
        self.frontiers_topic = self.args.frontiers_topic
        # self.goal_topic = self.args.goal_topic

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

        # cached messages
        self.odom_msg = None
        self.grid_map_msg = None

        # publish messages to a topic using rospy.Publisher class
        # self.pub_action = rospy.Publisher("habitat_action", String, queue_size=1)

        if args.visualize or args.print_images:
            self.legend = cv2.imread("docs/legend.png")
            self.rgb_vis = None
            self.vis_image = vu.init_occ_image(
                self.goal_name
            )  # , self.legend)

    def reset(self, map_shape):
        args = self.args

        # obs, info = super().reset()
        # obs = self._preprocess_obs(obs)

        # Episode initializations
        # map_shape = (
        #     args.map_size_cm // args.map_resolution,
        #     args.map_size_cm // args.map_resolution,
        # )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        # self.curr_loc = [
        #     args.map_size_cm / 100.0 / 2.0,
        #     args.map_size_cm / 100.0 / 2.0,
        #     0.0,
        # ]
        self.last_action = None
        if args.visualize or args.print_images:
            self.vis_image = vu.init_occ_image(
                self.goal_name
            )  # , self.legend)

    def parse_ros_messages(self):
        odom_msg: Odometry = self.odom_msg
        grid_map_msg: OccupancyGrid = self.grid_map_msg

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

        # parse odometry message
        odom_pose = odom_msg.pose
        # odom_map_pose: (x, y, x-y rotation angle)
        odom_map_pose = np.array(
            [odom_pose.pose.position.x, odom_pose.pose.position.y, 0.0]
        )
        # FIXME: there are zero quaternions for unknown reason
        try:
            odom_rot = R.from_quat(
                [
                    odom_pose.pose.orientation.x,
                    odom_pose.pose.orientation.y,
                    odom_pose.pose.orientation.z,
                    odom_pose.pose.orientation.w,
                ]
            )
            odom_angle = odom_rot.as_euler("zxy", degrees=True)[0]
        except:
            odom_angle = 0.0

        odom_map_pose[2] = odom_angle
        return grid_map, map_origin, odom_map_pose

    # wrapper function to execute one step
    def act(self):
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
            return "stay"

        grid_map, map_origin, odom_map_pose = self.parse_ros_messages()
        ##################  generate frontiers and goals ##############
        map_resolution = self.args.map_resolution

        frontiers, goals = frontier_goals(
            grid_map,
            map_origin,
            map_resolution,
            odom_map_pose[:2],
            cluster_trashhole=self.args.cluster_trashhole,
            num_goals=1,
            sim=self.sim,
        )
        # TODO: think about goal selection strategy
        goal_position = goals[0, :]

        self.publish_frontiers(frontiers, goals)

        if DEBUG_TF:
            self.debug_tf_func()

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
        p_input["found_goal"] = False
        p_input["wait"] = False

        action = self.plan(p_input)
        self.last_action = action  # needed for collision check

        if self.args.visualize or self.args.print_images:
            self._visualize(p_input)

        # TODO: add logic for STOP action

        return self.action_names[action]

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0.0, 0.0, 0.0]
            return np.zeros(self.obs.shape), 0.0, False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        action = self.plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)
        # NOTE: sem_exp uses habitat-lab api for simulation intereaction
        # if action >= 0:

        #     # act
        #     action = {"action": action}
        #     obs, rew, done, info = super().step(action)

        #     # preprocess obs
        #     obs = self._preprocess_obs(obs)
        #     self.last_action = action["action"]
        #     self.obs = obs
        #     self.info = info

        #     info["g_reward"] += rew

        #     return obs, rew, done, info

        # else:
        #     self.last_action = None
        #     self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        #     return np.zeros(self.obs_shape), 0.0, False, self.info

        # NOTE: this version uses habitat_sim for simulation interaction
        # since reward is not needed
        obs = self.sim.step(action)
        self.obs = obs
        return obs

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
            int(r / args.map_resolution - gx1),  # To comfirm: x and y
            int(c / args.map_resolution - gy1),
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
                    int(r / args.map_resolution - gx1),
                    int(c / args.map_resolution - gy1),
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
        #                     int(r / args.map_resolution),
        #                     int(c / args.map_resolution),
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

            # visualize long/short-term goals
            goal_vis = skimage.morphology.binary_dilation(
                goal, skimage.morphology.disk(3)
            )
            vis_map[goal_vis == 1] = 2  # disk for long-term goal
            rr, cc = rectangle(
                np.array(stg) - 2, extent=7, shape=vis_map.shape
            )
            vis_map[
                rr.astype(int), cc.astype(int)
            ] = 3  # rect for short-term goal

            # visualize agent
            vis_pos = [
                start[1],
                start[0],
                start_o / 180 * np.pi,
            ]  # To comfirm: x and y
            # pos = [start[0], start[1], start_o]
            agent_arrow = vu.get_contour_points(vis_pos, origin=(0, 0), size=5)
            color = (255, 0, 0)
            plt.fill(agent_arrow[:, 0], agent_arrow[:, 1], facecolor="red")
            # plt.fill(agent_arrow[:, 1], agent_arrow[:, 0], facecolor="red")
            # vis_map = vu.draw_polygen(vis_map, agent_arrow, value=5)
            # plt.arrow(pos[1], pos[0], dx=np.cos(start_o/180*np.pi)*10, dy=np.sin(start_o/180*np.pi)*10)

            plt.imshow(vis_map, origin="lower")
            plt.show()

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

            if relative_angle > self.args.turn_angle / 2.0:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.0:
                action = 2  # Left
            else:
                action = 1  # Forward

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

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

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
            (start_x / args.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y / args.map_resolution + gx1)
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

    def callback_odom(self, odom_msg: Odometry):
        self.odom_msg = odom_msg

    def callback_grid_map(self, grid_map_msg: OccupancyGrid):
        self.grid_map_msg = grid_map_msg

    def publish_frontiers(self, frontiers, goals):

        marker_arr = MarkerArray()
        # color_map = cm.get_cmap("plasma")
        # publish goals
        for i, g in enumerate(goals):

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.id = i  # avoid overwrite
            marker.type = marker.SPHERE
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = g[0]
            marker.pose.position.y = g[1]
            marker.pose.position.z = 0.5

            # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
            marker_arr.markers.append(marker)

        # publish frontiers
        num_goals = goals.shape[0]
        for i, f in enumerate(frontiers):

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.id = i + num_goals
            marker.type = marker.SPHERE
            marker.scale.x = f[2] / 1000.0
            marker.scale.y = f[2] / 1000.0
            marker.scale.z = f[2] / 1000.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = f[0]
            marker.pose.position.y = f[1]
            marker.pose.position.z = 0.5

            # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
            marker_arr.markers.append(marker)

        self.pub_frontiers.publish(marker_arr)

    # record /debug/tf_rtab2habitat by rosbag to do offline analysis
    def debug_tf_func(self):

        debug_tf_pub = rospy.Publisher(
            name="/debug/tf_rtab2habitat",
            data_class=PoseStamped,
            queue_size=10,
        )
        ############## calculate relative transformation from rtabmap to habitat #############
        odom_rot = self.odom_msg.pose.pose.orientation
        odom_pos = self.odom_msg.pose.pose.position
        odom_rot_mat = R.from_quat(
            [odom_rot.x, odom_rot.y, odom_rot.z, odom_rot.w]
        ).as_matrix()
        tf_rtab2agent = np.zeros((4, 4))
        tf_rtab2agent[:3, :3] = odom_rot_mat
        tf_rtab2agent[:3, 3] = np.array([odom_pos.x, odom_pos.y, odom_pos.z])
        tf_rtab2agent[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        # print("odom_mat norm:\n", np.linalg.det(odom_mat))
        habitat_rot = self.sim.get_agent(0).state.rotation
        habitat_pos = self.sim.get_agent(0).state.position
        habitat_rot_mat = R.from_quat(
            [habitat_rot.x, habitat_rot.y, habitat_rot.z, habitat_rot.w]
        ).as_matrix()
        tf_habitat2agent = np.zeros((4, 4))
        tf_habitat2agent[:3, :3] = habitat_rot_mat
        tf_habitat2agent[:3, 3] = habitat_pos
        tf_habitat2agent[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        # print("habitat_mat norm:\n", np.linalg.det(habitat_mat))

        tf_rtab2habitat = tf_rtab2agent @ np.linalg.inv(tf_habitat2agent)
        # tf_rtab2habitat[abs(tf_rtab2habitat) < 1e-3] = 0.0
        # print(tf_rtab2habitat)  # , np.linalg.det(tf_rtab2habitat))
        # print(R.from_matrix(tf_rtab2habitat[:3, :3]).as_quat())

        pose_h_in_r_relative = PoseStamped()
        pose_h_in_r_relative.header.frame_id = "map"
        pose_h_in_r_relative.pose.position = Point(*tf_rtab2habitat[:3, 3])
        tf_rtab2habitat_quat = R.from_matrix(tf_rtab2habitat[:3, :3]).as_quat()
        pose_h_in_r_relative.pose.orientation = Quaternion(
            *tf_rtab2habitat_quat
        )

        # debug_tf_pub.publish(pose_h_in_r_relative)

        ############### calculate agent's pose in rtabmap directly ###########

        pose_a_in_h = PoseStamped()
        pose_a_in_h.header.frame_id = "habitat"
        pose_a_in_h.pose.position = Point(*habitat_pos)
        pose_a_in_h.pose.orientation = Quaternion(
            habitat_rot.x, habitat_rot.y, habitat_rot.z, habitat_rot.w
        )

        tf_rtabmap2habitat = TransformStamped()
        tf_rtabmap2habitat.header.frame_id = "map"  # z-axis upright
        tf_rtabmap2habitat.child_frame_id = "habitat"  # y-axis upright

        tf_rtabmap2habitat.transform.translation.x = 0.0
        tf_rtabmap2habitat.transform.translation.y = 0.0
        tf_rtabmap2habitat.transform.translation.z = 0.0

        quat = quat_from_two_vectors(geo.GRAVITY, np.array([0, 0, -1]))
        tf_rtabmap2habitat.transform.rotation.x = quat.x
        tf_rtabmap2habitat.transform.rotation.y = quat.y
        tf_rtabmap2habitat.transform.rotation.z = quat.z
        tf_rtabmap2habitat.transform.rotation.w = quat.w

        pose_a_in_r = tf2_geometry_msgs.do_transform_pose(
            pose_a_in_h, tf_rtabmap2habitat
        )
        # NOTE: for the correct pose of agent in matterport3D,
        # you still need to flip y-z axes
        # according to this issue https://github.com/facebookresearch/habitat-sim/issues/1620

        debug_tf_pub.publish(pose_a_in_r)

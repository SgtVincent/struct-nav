import enum
import math
import os
import time
import cv2
from matplotlib.pyplot import grid
from matplotlib import cm
import numpy as np
from scipy.spatial.transform import Rotation as R
import skimage.morphology
from PIL import Image
from torchvision import transforms
from habitat_sim import Simulator
from scipy.spatial.transform import Rotation as R
from yaml import Mark

# ros packages
import rospy
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from visualization_msgs.msg import MarkerArray, Marker


from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.objectgoal_env import ObjectGoalEnv

# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from envs.constants import color_palette
import envs.utils.pose as pu
from agents.utils.utils_frontier_explore import frontier_goals
import agents.utils.visualization as vu
from agents.utils.arguments import get_args


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

        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.callback_odom)
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
            self.vis_image = None
            self.rgb_vis = None

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
            self.vis_image = vu.init_occ_image(self.goal_name, self.legend)

        return

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
        odom_msg: Odometry = self.odom_msg
        grid_map_msg: OccupancyGrid = self.grid_map_msg

        # FIXME: cannot receive grid_map message
        while odom_msg == None or grid_map_msg == None:
            # waiting for data
            return "stay"

        grid_map = np.array(grid_map_msg.data, dtype=np.int8).reshape(
            grid_map_msg.info.height, grid_map_msg.info.width
        )
        self.reset(grid_map.shape)
        # map_resolution = grid_map_msg.info.resolution
        # NOTE: args.map_resolution is in cm
        map_resolution = self.args.map_resolution / 100.0
        map_origin_position = grid_map_msg.info.origin.position
        map_origin = np.array([map_origin_position.x, map_origin_position.y])

        # parse odometry message
        odom_pose = odom_msg.pose
        odom_position_2d = np.array(
            [odom_pose.pose.position.x, odom_pose.pose.position.y,]
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
            odom_angle = 0

        ##################  generate frontiers and goals ##############
        frontiers, goals = frontier_goals(
            grid_map,
            map_origin,
            map_resolution,
            odom_position_2d,
            cluster_trashhole=self.args.cluster_trashhole,
            num_goals=1,
        )
        # TODO: think about goal selection strategy
        goal_position = goals[0, :]

        self.publish_frontiers(frontiers, goals)

        ################### generate action by planners ################

        # convert (x,y) goal center to goal map
        goal_map = np.zeros_like(grid_map)
        pixel_position = ((goal_position - map_origin) / map_resolution).astype(int)
        # NOTE: (x,y) is (col, row) in image
        goal_map[pixel_position[1], pixel_position[0]] = 1.0

        p_input = {}
        p_input["map_pred"] = grid_map.astype(np.float64)
        # p_input["exp_pred"] = local_map[e, 1, :, :]
        p_input["pose_pred"] = np.zeros(7)
        p_input["pose_pred"][:2] = odom_position_2d
        p_input["pose_pred"][2] = odom_angle
        p_input["pose_pred"][3:] = np.array(
            [0, grid_map.shape[0], 0, grid_map.shape[1]]
        )
        p_input["goal"] = goal_map
        p_input["found_goal"] = False
        p_input["wait"] = False

        action = self.plan(p_input)
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
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [
                int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1),
            ]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
                last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
            )

        # Collision check
        if self.last_action == 1:
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
                            int(r * 100 / args.map_resolution),
                            int(c * 100 / args.map_resolution),
                        )
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)

        # Deterministic Local Policy
        if stop and planner_inputs["found_goal"] == 1:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
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

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1 : h + 1, 1 : w + 1] = mat
            return new_mat

        traversible = (
            skimage.morphology.binary_dilation(grid[x1:x2, y1:y2], self.selem) != True
        )
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
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

        map_pred = inputs["map_pred"]
        exp_pred = inputs["exp_pred"]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs["pose_pred"]

        goal = inputs["goal"]
        sem_map = inputs["sem_map_pred"]

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.0) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1], sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(
            sem_map_vis, (480, 480), interpolation=cv2.INTER_NEAREST
        )
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis

        pos = (
            (start_x * 100.0 / args.map_resolution - gy1) * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100.0 / args.map_resolution + gx1)
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
        color_map = cm.get_cmap("plasma")
        # publish frontiers
        for i, f in enumerate(frontiers):

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.id = i
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
        # publish goals
        num_frontiers = frontiers.shape[0]
        for i, g in enumerate(goals):

            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.id = i + num_frontiers  # avoid overwrite
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

        self.pub_frontiers.publish(marker_arr)

import json
import os 
import bz2
import gzip
import time
import math 
from collections import deque
import _pickle as cPickle
import cv2
import numpy as np
import quaternion
import skimage.morphology
import habitat
import rospy 

from envs.utils.fmm_planner import FMMPlanner
from envs.constants import coco_categories
import envs.utils.pose as pu
import agents.utils.visualization as vu
from envs.constants import color_palette, coco_categories, coco_label_mapping

class ObjectGoal_Env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        super().__init__(config_env, dataset)

        # Loading dataset info file
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split
        )

        dataset_info_file = self.episodes_dir + "{split}_info.pbz2".format(
            split=self.split
        )
        with bz2.BZ2File(dataset_info_file, "rb") as f:
            self.dataset_info = cPickle.load(f)

        # Specifying action and observation space
        # not used, also raise error 
        # self.action_space = gym.spaces.Discrete(3)

        # self.observation_space = gym.spaces.Box(
        #     0, 255, (3, args.frame_height, args.frame_width), dtype="uint8"
        # )

        # Initializations
        self.episode_no = 0

        # Scene info
        self.last_scene_path = None
        self.scene_path = None
        self.scene_name = None

        # Episode Dataset info
        self.eps_data = None
        self.eps_data_idx = None
        self.gt_planner = None
        self.object_boundary = None
        self.goal_idx = None
        self.goal_name = None
        self.map_obj_origin = None
        self.starting_loc = None
        self.starting_distance = None

        # Episode tracking info
        self.curr_distance = None
        self.prev_distance = None
        self.timestep = None
        self.stopped = None
        self.path_length = None
        self.last_sim_location = None
        self.trajectory_states = []
        self.actions_queue = deque([], maxlen=4)
        self.info = {}
        self.info["distance_to_goal"] = None
        self.info["spl"] = None
        self.info["success"] = None

    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        args = self.args
        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        if self.scene_path != self.last_scene_path:
            episodes_file = (
                self.episodes_dir
                + "content/{}_episodes.json.gz".format(scene_name)
            )

            print("Loading episodes from: {}".format(episodes_file))
            with gzip.open(episodes_file, "r") as f:
                self.eps_data = json.loads(f.read().decode("utf-8"))[
                    "episodes"
                ]

            self.eps_data_idx = 0
            self.last_scene_path = self.scene_path

        # Load episode info
        episode = self.eps_data[self.eps_data_idx]
        self.eps_data_idx += 1
        self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        pos = episode["start_position"]
        rot = quaternion.from_float_array(episode["start_rotation"])

        goal_name = episode["object_category"]
        goal_idx = episode["object_id"]
        floor_idx = episode["floor_id"]

        # Load scene info
        scene_info = self.dataset_info[scene_name]
        sem_map = scene_info[floor_idx]["sem_map"]
        map_obj_origin = scene_info[floor_idx]["origin"]

        # Setup ground truth planner
        object_boundary = args.success_dist
        map_resolution = args.map_resolution_cm
        selem = skimage.morphology.disk(2)
        traversible = (
            skimage.morphology.binary_dilation(sem_map[0], selem) != True
        )
        traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(
            int(object_boundary * 100.0 / map_resolution)
        )
        goal_map = (
            skimage.morphology.binary_dilation(sem_map[goal_idx + 1], selem)
            != True
        )
        goal_map = 1 - goal_map
        planner.set_multi_goal(goal_map)

        # Get starting loc in GT map coordinates
        x = -pos[2]
        y = -pos[0]
        min_x, min_y = map_obj_origin / 100.0
        map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)

        self.gt_planner = planner
        self.starting_loc = map_loc
        self.object_boundary = object_boundary
        self.goal_idx = goal_idx
        self.goal_name = goal_name
        self.map_obj_origin = map_obj_origin

        self.starting_distance = (
            self.gt_planner.fmm_dist[self.starting_loc] / 20.0
            + self.object_boundary
        )
        self.prev_distance = self.starting_distance
        self._env.sim.set_agent_state(pos, rot)

        # The following two should match approximately
        # print(starting_loc)
        # print(self.sim_continuous_to_sim_map(self.get_sim_location()))

        obs = self._env.sim.get_observations_at(pos, rot)

        return obs

    def generate_new_episode(self):
        """The function generates a random valid episode. This function is used
        for training a model on the train split.
        """

        args = self.args

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        scene_name = self.scene_path.split("/")[-1].split(".")[0]

        scene_info = self.dataset_info[scene_name]
        map_resolution = args.map_resolution_cm

        floor_idx = np.random.randint(len(scene_info.keys()))
        floor_height = scene_info[floor_idx]["floor_height"]
        sem_map = scene_info[floor_idx]["sem_map"]
        map_obj_origin = scene_info[floor_idx]["origin"]

        cat_counts = sem_map.sum(2).sum(1)
        possible_cats = list(np.arange(6))

        for i in range(6):
            if cat_counts[i + 1] == 0:
                possible_cats.remove(i)

        object_boundary = args.success_dist

        loc_found = False
        while not loc_found:
            if len(possible_cats) == 0:
                print("No valid objects for {}".format(floor_height))
                eps = eps - 1
                continue

            goal_idx = np.random.choice(possible_cats)

            for key, value in coco_categories.items():
                if value == goal_idx:
                    goal_name = key

            selem = skimage.morphology.disk(2)
            traversible = (
                skimage.morphology.binary_dilation(sem_map[0], selem) != True
            )
            traversible = 1 - traversible

            planner = FMMPlanner(traversible)

            selem = skimage.morphology.disk(
                int(object_boundary * 100.0 / map_resolution)
            )
            goal_map = (
                skimage.morphology.binary_dilation(
                    sem_map[goal_idx + 1], selem
                )
                != True
            )
            goal_map = 1 - goal_map

            planner.set_multi_goal(goal_map)

            m1 = sem_map[0] > 0
            m2 = planner.fmm_dist > (args.min_d - object_boundary) * 20.0
            m3 = planner.fmm_dist < (args.max_d - object_boundary) * 20.0

            possible_starting_locs = np.logical_and(m1, m2)
            possible_starting_locs = (
                np.logical_and(possible_starting_locs, m3) * 1.0
            )
            if possible_starting_locs.sum() != 0:
                loc_found = True
            else:
                print(
                    "Invalid object: {} / {} / {}".format(
                        scene_name, floor_height, goal_name
                    )
                )
                possible_cats.remove(goal_idx)
                scene_info[floor_idx]["sem_map"][goal_idx + 1, :, :] = 0.0
                self.dataset_info[scene_name][floor_idx]["sem_map"][
                    goal_idx + 1, :, :
                ] = 0.0

        loc_found = False
        while not loc_found:
            pos = self._env.sim.sample_navigable_point()
            x = -pos[2]
            y = -pos[0]
            min_x, min_y = map_obj_origin / 100.0
            map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)
            if (
                abs(pos[1] - floor_height) < args.floor_thr / 100.0
                and possible_starting_locs[map_loc[0], map_loc[1]] == 1
            ):
                loc_found = True

        agent_state = self._env.sim.get_agent_state(0)
        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)
        rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        self.gt_planner = planner
        self.starting_loc = map_loc
        self.object_boundary = object_boundary
        self.goal_idx = goal_idx
        self.goal_name = goal_name
        self.map_obj_origin = map_obj_origin

        self.starting_distance = (
            self.gt_planner.fmm_dist[self.starting_loc] / 20.0
            + self.object_boundary
        )
        self.prev_distance = self.starting_distance

        self._env.sim.set_agent_state(pos, rot)

        # The following two should match approximately
        # print(starting_loc)
        # print(self.sim_continuous_to_sim_map(self.get_sim_location()))

        obs = self._env.sim.get_observations_at(pos, rot)

        return obs

    def sim_map_to_sim_continuous(self, coords):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        agent_state = self._env.sim.get_agent_state(0)
        y, x = coords
        min_x, min_y = self.map_obj_origin / 100.0

        cont_x = x / 20.0 + min_x
        cont_y = y / 20.0 + min_y
        agent_state.position[0] = cont_y
        agent_state.position[2] = cont_x

        rotation = agent_state.rotation
        rvec = quaternion.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = quaternion.from_rotation_vector(rvec)

        return agent_state.position, rot

    def sim_continuous_to_sim_map(self, sim_loc):
        """Converts absolute Habitat simulator pose to ground-truth 2D Map
        coordinates.
        """
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin / 100.0
        x, y = int((-x - min_x) * 20.0), int((-y - min_y) * 20.0)

        o = np.rad2deg(o) + 180.0
        return y, x, o

    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        args = self.args
        new_scene = self.episode_no % args.num_train_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.actions_queue = deque([], maxlen=4)
        
        if new_scene:
            obs = super().reset()
            self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
            print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        if self.split == "val":
            obs = self.load_new_episode()
        else:
            obs = self.generate_new_episode()

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        # state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info["time"] = self.timestep
        self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        self.info["goal_cat_id"] = self.goal_idx
        self.info["goal_name"] = self.goal_name
        self.info["agent_pose"] = super().habitat_env.sim.get_agent_state(0)
        
        return obs, self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True
            # Not sending stop to simulator, resetting manually
            action = 3

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info["sensor_pose"] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)
        self.info["agent_pose"] = super().habitat_env.sim.get_agent_state(0)

        spl, success, dist = 0.0, 0.0, 0.0
        if done:
            spl, success, dist = self.get_metrics()
            self.info["distance_to_goal"] = dist
            self.info["spl"] = spl
            self.info["success"] = success

        rgb = obs["rgb"].astype(np.uint8)
        depth = obs["depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.actions_queue.append(action)
        self.timestep += 1
        self.info["time"] = self.timestep

        # return state, rew, done, self.info
        return obs, rew, done, self.info

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0.0, 1.0)

    def get_reward(self, observations):
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        self.curr_distance = (
            self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        )

        reward = (
            self.prev_distance - self.curr_distance
        ) * self.args.reward_coeff

        self.prev_distance = self.curr_distance
        return reward

    def get_metrics(self):
        """This function computes evaluation metrics for the Object Goal task

        Returns:
            spl (float): Success weighted by Path Length
                        (See https://arxiv.org/pdf/1807.06757.pdf)
            success (int): 0: Failure, 1: Successful
            dist (float): Distance to Success (DTS),  distance of the agent
                        from the success threshold boundary in meters.
                        (See https://arxiv.org/pdf/2007.00643.pdf)
        """
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        dist = self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        if dist == 0.0:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance / self.path_length, 1)
        return spl, success, dist

    def get_done(self, observations):
        if self.info["time"] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done

    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    # not used, raise error 
    # def get_spaces(self):
    #     """Returns observation and action spaces for the ObjectGoal task."""
    #     return self.observation_space, self.action_space

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (
            axis % (2 * np.pi)
        ) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location
        )
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def _preprocess_depth(self, obs):
        depth = obs['depth'].squeeze()

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        obs['depth'] = depth
        
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
        # traversible[
        #     skimage.morphology.binary_dilation(
        #         self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.selem)
        #     == 1
        # ] = 0
        traversible[
                self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2]== 1
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
        
        try: 
            planner.set_multi_goal(goal)

            state = [start[0] - x1, start[1] - y1]
            stg_x, stg_y, stop = planner.get_short_term_goal(state)
            stg_x, stg_y = stg_x + x1, stg_y + y1
            # original code here: 
            # state = [start[0] - x1 + 1, start[1] - y1 + 1]
            # stg_x, stg_y, _, stop = planner.get_short_term_goal(state)
            # stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
            return (stg_x, stg_y), stop
        except Exception as e:
            rospy.logwarn(f"Planner failed, with error {e}, reset found_goal state...")
            self.spot_goal = 0
            return (-1, -1), 0

    def _get_action(self, start, start_o, stg, stop, planner_inputs):
        if stop and planner_inputs["found_goal"] == 1:
            # goal object found and local planner stopped
            action = 0  # Stop
        elif stop and planner_inputs["found_goal"] == 0:
            # goal object not found and local planner stopped (stg reached by frontier still exists)
            # NOTE: need to turn to face the goal center for more observation
            goal_map = planner_inputs["goal"]
            
            # Find the center of goal map 
            idx_x, idx_y = np.where(goal_map > 0)
            goal_x = np.average(idx_x)
            goal_y = np.average(idx_y)
            
            angle_st_goal = math.degrees(
                math.atan2(goal_x - start[0], goal_y - start[1])
            )
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > 0.7 * self.turn_angle:
                action = 3  # Right
            elif relative_angle < -0.7 * self.turn_angle:
                action = 2  # Left
            else:
                action = 1  # Forward
                
        else:
            (stg_x, stg_y) = stg
            
            if stg_x < 0: # planner failed due to synchronization problem of rtabmap
                # by default, move forward 
                action = 1

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
            # current policy: last 4 actions are lrlr or rlrl
            # TODO: investigate why it get stuck and solve the problem
            last_actions = list(self.actions_queue)
            if last_actions==[2,3,2,3] or last_actions==[3,2,3,2]:
                action = 1 # Execute forward to get away from left-right swing cycle
        
        return action 

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

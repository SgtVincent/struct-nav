import json
import os 
import gzip
import time
import math 
from collections import deque
import cv2
import numpy as np
import quaternion as qt 
import skimage.morphology
import habitat
import rospy 
import pickle
import numba
import numpy as np
import gym
from gym import spaces
from scipy import ndimage
import magnum as mn
import math
import random


from habitat.datasets.multi_object_nav.multi_object_nav_dataset import MultiObjectNavDatasetV1
from envs.utils.fmm_planner import FMMPlanner
import envs.utils.pose as pu
from envs.constants import coco_categories
from envs.constants import color_palette, coco_categories, coco_label_mapping
import agents.utils.visualization as vu
import habitat_sim
from agents.utils.utils_frontier_explore import update_odom_by_action
from utils.transformation import pose_habitat2rtabmap

#---------- MULTION CONSTANTS ---------------


COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON
#----------------------------------------------

DEBUG_VIS = False

class MultiON_Env(habitat.RLEnv):
# class MultiON_Env(habitat.Env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        self.rank = rank

        # arguments 
        self.map_resolution = args.map_resolution_cm / 100.0
        self.sem_model = args.sem_model
        self.success_dist = args.success_dist

        super().__init__(config_env, dataset)

        # Loading dataset info file
        # TODO: dataset info file no longer required in multion_env
        self.split = config_env.DATASET.SPLIT
        self.episodes_dir = config_env.DATASET.EPISODES_DIR.format(
            split=self.split
        )
        dataset_info_file = config_env.DATASET.DATA_PATH
        with gzip.open(dataset_info_file, 'rt', encoding='UTF-8') as f:   
            data = json.load(f)  
            self.map_cat2mp3d_label = data['category_to_mp3d_category_id']
            self.map_cat2task_label = data['category_to_task_category_id']
        
        # load dataset info from dataset 
        self.dataset: MultiObjectNavDatasetV1 = self._env._dataset
        self.category_to_task_category_id = self.dataset.category_to_task_category_id
            
        # # load label from instance id to class label 
        # if args.sem_model == "ground_truth":
        #     self.map_id2label = {}
        #     for obj in self._env.sim.semantic_scene.objects:
        #         if obj is not None:
        #             obj_id = int(obj.id.split('_')[-1])
        #             obj_cls_name = obj.category.name()
        #             if obj_cls_name in coco_categories:
        #                 self.map_id2label[obj_id] = coco_categories[obj_cls_name]
        #             else:
        #                 self.map_id2label[obj_id] = -1

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
        self.goal_cat_idx = None
        self.goal_name = None
        self.goal_cat_ids = None
        self.goal_names = None
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

    @property
    def episode_over(self) -> bool:
        return self._env.episode_over

    def init_ros(self, default_rate=4.0, default_map_size=10.0):

        """initialize ros related publishers and subscribers for agent"""
        # only read parameters from ROS, leave class initialization for each 
        # agent class 
        self.rate_value = rospy.get_param("~rate", default_rate)
        rospy.loginfo(f"agent update state from ros at rate {self.rate_value} hz")
        self.camera_info_file = rospy.get_param("~camera_calib", "")
        self.initial_map_size = rospy.get_param("~initial_map_size", default_map_size)
        
        # setup pseudo wheel odometry to fuse with rtabmap visual odometry 
        self.wheel_odom = rospy.get_param("~wheel_odom", False)
        self.whee_odom_frame_id = ""
        if self.wheel_odom:
            self.whee_odom_frame_id = rospy.get_param("~wheel_odom_frame_id", "")
        
        # setup ground truth odometry 
        self.ground_truth_odom = rospy.get_param("~ground_truth_odom", False)
        self.ground_truth_odom_topic = ""
        if self.ground_truth_odom:
            self.ground_truth_odom_topic = rospy.get_param("~ground_truth_odom_topic", "")
        
        self.map_update_mode = rospy.get_param("~map_update_mode", "listen")

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
        # self.wheel_odom_topic = rospy.get_param(
        #     "~wheel_odom_topic", ""
        # )
        self.true_pose_topic = rospy.get_param("~true_pose_topic", "")
        self.sem_cloud_topic = rospy.get_param(
            "~sem_cloud_topic", "/rtabsem/cloud_map"
        )
        self.cloud_topic = rospy.get_param(
            "~cloud_topic", "/rtabmap/cloud_map"
        )
        
        # topics for planning
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        if self.ground_truth_odom:
            self.odom_topic = self.ground_truth_odom_topic
            
        self.grid_map_topic = rospy.get_param(
            "~grid_map_topic", "/rtabmap/grid_map"
        )
        self.frontiers_topic = rospy.get_param(
            "~frontiers_topic", "/frontiers"
        )
        


    def load_new_episode(self):
        """The function loads a fixed episode from the episode dataset. This
        function is used for evaluating a trained model on the val split.
        """

        pass
        # args = self.args
        # self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id
        # scene_name = self.scene_path.split("/")[-1].split(".")[0]

        # if self.scene_path != self.last_scene_path:
        #     episodes_file = (
        #         self.episodes_dir
        #         + "content/{}_episodes.json.gz".format(scene_name)
        #     )

        #     print("Loading episodes from: {}".format(episodes_file))
        #     with gzip.open(episodes_file, "r") as f:
        #         self.eps_data = json.loads(f.read().decode("utf-8"))[
        #             "episodes"
        #         ]

        #     self.eps_data_idx = 0
        #     self.last_scene_path = self.scene_path

        # # Load episode info
        # episode = self.eps_data[self.eps_data_idx]
        # self.eps_data_idx += 1
        # self.eps_data_idx = self.eps_data_idx % len(self.eps_data)
        # pos = episode["start_position"]
        # rot = qt.from_float_array(episode["start_rotation"])

        # goal_name = episode["object_category"]
        # goal_cat_idx = episode["object_id"]
        # floor_idx = episode["floor_id"]

        # # Load scene info
        # # scene_info = self.dataset_info[scene_name]
        # # sem_map = scene_info[floor_idx]["sem_map"]
        # # map_obj_origin = scene_info[floor_idx]["origin"]

        # # Setup ground truth planner
        # object_boundary = args.success_dist
        # map_resolution = args.map_resolution_cm
        # selem = skimage.morphology.disk(2)
        # traversible = (
        #     skimage.morphology.binary_dilation(sem_map[0], selem) != True
        # )
        # traversible = 1 - traversible
        # planner = FMMPlanner(traversible)
        # selem = skimage.morphology.disk(
        #     int(object_boundary * 100.0 / map_resolution)
        # )
        # goal_map = (
        #     skimage.morphology.binary_dilation(sem_map[goal_cat_idx + 1], selem)
        #     != True
        # )
        # goal_map = 1 - goal_map
        # planner.set_multi_goal(goal_map)

        # # Get starting loc in GT map coordinates
        # x = -pos[2]
        # y = -pos[0]
        # min_x, min_y = map_obj_origin / 100.0
        # map_loc = int((-y - min_y) * 20.0), int((-x - min_x) * 20.0)

        # self.gt_planner = planner
        # self.starting_loc = map_loc
        # self.object_boundary = object_boundary
        # self.goal_cat_idx = goal_cat_idx
        # self.goal_name = goal_name
        # self.map_obj_origin = map_obj_origin

        # self.starting_distance = (
        #     self.gt_planner.fmm_dist[self.starting_loc] / 20.0
        #     + self.object_boundary
        # )
        # self.prev_distance = self.starting_distance
        # self._env.sim.set_agent_state(pos, rot)

        # The following two should match approximately
        # print(starting_loc)
        # print(self.sim_continuous_to_sim_map(self.get_sim_location()))

        # obs = self._env.sim.get_observations_at(pos, rot)

        # return obs

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
        rvec = qt.as_rotation_vector(rotation)

        if self.args.train_single_eps:
            rvec[1] = 0.0
        else:
            rvec[1] = np.random.rand() * 2 * np.pi
        rot = qt.from_rotation_vector(rvec)

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
        # new_scene = self.episode_no % args.num_train_episodes == 0

        self.episode_no += 1

        # Initializations
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.actions_queue = deque([], maxlen=4)
        
        # if new_scene:
        obs = super().reset()
        self.scene_name = self.habitat_env.sim.config.sim_cfg.scene_id
        print("Changing scene: {}/{}".format(self.rank, self.scene_name))

        self.scene_path = self.habitat_env.sim.config.sim_cfg.scene_id

        # NOTE: the object deletion & addition operations in Env.reset() is redundant
        # task.reset() will handle these operations 
        #----------- multion extra object operation in reset() -----------------
        # # Remove existing objects from last episode
        # for objid in self.habitat_env._sim.get_existing_object_ids():
        #     self.habitat_env._sim.remove_object(objid)

        # # Insert object here
        # obj_type = self.habitat_env._config["TASK"]["OBJECTS_TYPE"]
        # if obj_type == "CYL":
        #     object_to_datset_mapping = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2, 'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7}
        # else:
        #     object_to_datset_mapping = {'guitar':0, 'electric_piano':1, 'basket_ball':2,'toy_train':3, 'teddy_bear':4, 'rocking_horse':5, 'backpack': 6, 'trolley_bag':7}
            
            
        # for i in range(len(self.current_episode.goals)):
        #     current_goal = self.current_episode.goals[i].object_category
        #     dataset_index = object_to_datset_mapping[current_goal]
        #     ind = self.habitat_env._sim.add_object(dataset_index)
        #     self.habitat_env._sim.set_translation(np.array(self.current_episode.goals[i].position), ind)
            
        #     # random rotation only on the Y axis
        #     y_rotation = mn.Quaternion.rotation(
        #         mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
        #     )
        #     self.habitat_env._sim.set_rotation(y_rotation, ind)
        #     self.habitat_env._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)

        # if self.habitat_env._config["TASK"]["INCLUDE_DISTRACTORS"]:
        #     for i in range(len(self.current_episode.distractors)):
        #         current_distractor = self.current_episode.distractors[i].object_category
        #         dataset_index = object_to_datset_mapping[current_distractor]
        #         ind = self.habitat_env._sim.add_object(dataset_index)
        #         self.habitat_env._sim.set_translation(np.array(self.current_episode.distractors[i].position), ind)
                
        #         # random rotation only on the Y axis
        #         y_rotation = mn.Quaternion.rotation(
        #             mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
        #         )
        #         self.habitat_env._sim.set_rotation(y_rotation, ind)
        #         self.habitat_env._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)
        # #----------------------------------------------------------------------------

        # In multion benchmarking, the episode management is within env.reset()
        # if self.split == "val":
        #     obs = self.load_new_episode()
        # else:
        #     obs = self.generate_new_episode()
        self.goal_names = [goal.object_category 
                           for goal in self.current_episode.goals]
        self.goal_cat_ids = [self.category_to_task_category_id[i] 
                         for i in self.goal_names]        
        self.current_goal_idx = self._env.task.current_goal_index
        self.goal_cat_idx = self.goal_cat_ids[self.current_goal_idx]
        
        # rgb = obs["rgb"].astype(np.uint8)
        # depth = obs["depth"]
        # state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.last_sim_location = self.get_sim_location()

        # Set info
        self.info["time"] = self.timestep
        self.info["sensor_pose"] = [0.0, 0.0, 0.0]
        self.info["goal_cat_id"] = self.goal_cat_idx
        self.info["goal_name"] = self.goal_names[self.current_goal_idx]
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
        # if action == 0:
        #     self.stopped = True
        #     # Not sending stop to simulator, resetting manually
        #     action = 3

        #----------- multion episode termination condition in step() -----------
        #----------- this section should be executed before Env.step() ----------
        self.habitat_env.task.is_found_called = bool(action == 0)
            
        ##Terminates episode if wrong found is called
        if self.habitat_env.task.is_found_called == True and \
            self.habitat_env.task.measurements.measures[
            "sub_success"
        ].get_metric() == 0:
            self.habitat_env.task._is_episode_active = False
        
        ##Terminates episode if all goals are found
        if self.habitat_env.task.is_found_called == True and \
            self.habitat_env.task.current_goal_index == len(self.current_episode.goals):
            self.habitat_env.task._is_episode_active = False
        #-----------------------------------------------------------------------

        obs, rew, done, _ = super().step(action)

        # Get pose change
        dx, dy, do = self.get_pose_change()
        self.info["sensor_pose"] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)
        self.info["agent_pose"] = super().habitat_env.sim.get_agent_state(0)

        # spl, success, dist = 0.0, 0.0, 0.0
        # if done:
        #     spl, success, dist = self.get_metrics()
        #     self.info["distance_to_goal"] = dist
        #     self.info["spl"] = spl
        #     self.info["success"] = success

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
        pass
        # curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        # self.curr_distance = (
        #     self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
        # )

        # reward = (
        #     self.prev_distance - self.curr_distance
        # ) * self.args.reward_coeff

        # self.prev_distance = self.curr_distance
        # return reward

    # def get_metrics(self):
    #     """This function computes evaluation metrics for the Object Goal task

    #     Returns:
    #         spl (float): Success weighted by Path Length
    #                     (See https://arxiv.org/pdf/1807.06757.pdf)
    #         success (int): 0: Failure, 1: Successful
    #         dist (float): Distance to Success (DTS),  distance of the agent
    #                     from the success threshold boundary in meters.
    #                     (See https://arxiv.org/pdf/2007.00643.pdf)
    #     """
    #     curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
    #     dist = self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]] / 20.0
    #     if dist == 0.0:
    #         success = 1
    #     else:
    #         success = 0
    #     spl = min(success * self.starting_distance / self.path_length, 1)
    #     return spl, success, dist

    def get_done(self, observations):
        # TODO: consider to remove this time info, using self.episode_over instead
        if self.info["time"] >= self.args.max_episode_length - 1:
            done = True
        elif self.stopped:
            done = True
        elif self.episode_over:
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
        axis = qt.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (
            axis % (2 * np.pi)
        ) > 2 * np.pi - 0.1:
            o = qt.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - qt.as_euler_angles(agent_state.rotation)[1]
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

    # TODO: add object detection / segmentation models here
    def _preprocess_obs(self, obs, info=None):

        # preprocess broken meshes (0-depth) in depth image 
        self._preprocess_depth(obs)

        self._preprocess_sem(obs)

        # add pseudo wheel odometry pose to observations 
        if self.wheel_odom:
            if self.last_action:
                updated_odom_mat = update_odom_by_action(self.last_odom_mat, self.last_action)
                obs['odom_pose_mat'] = updated_odom_mat
            else:
                obs['odom_pose_mat'] = self.last_odom_mat
        
        # add ground truth pose to observations
        if self.ground_truth_odom:
            true_state = info['agent_pose']
            true_pos_rtab, true_rot_rtab = pose_habitat2rtabmap(
                true_state.position, 
                true_state.rotation,
                self.init_pos,
                self.init_rot
            )
            
            true_odom_mat = np.zeros((4,4), dtype=float)
            true_odom_mat[:3, :3] = qt.as_rotation_matrix(true_rot_rtab)
            true_odom_mat[:3, 3] = true_pos_rtab
            true_odom_mat[3, 3] = 1.0
            obs['true_odom_mat'] = true_odom_mat
            
        # if DEBUG_WHEEL_ODOM:
        #     publish_pose(obs['odom_pose_mat'], self.pub_wheel_odom_pose)
        return obs

    def _preprocess_sem(self, obs, gt_sem_range_clip=True, gt_depth_range=5.0):
        # preprocess semantic image
        if self.sem_model == "ground_truth":
            # NOTE: for Gibson dataset, semantic image is instance segmentation
            # need to convert that to semantic segmentation
            inst_img = obs['semantic']
            sem_img = np.zeros_like(inst_img)
            for inst_id in np.unique(inst_img):
                if inst_id > 0: # filter our background 
                    inst_label = self.map_id2label[inst_id] 
                    if inst_label >= 0: # only keep coco 15 categories 
                        sem_img[inst_img == inst_id] = inst_label + 1 # 0 for background 
                        
            # clip segmentation to depth range to make a reasonable GT model
            if gt_sem_range_clip:
                depth = obs['depth'].squeeze()
                invalid_sem_mask = depth > gt_depth_range
                sem_img[invalid_sem_mask] = 0
            
            obs['semantic'] = sem_img
            
        if self.sem_model == "detectron":
            # overwrite semantic image with detectron prediction 
            rgb = obs['rgb']
            sem_seg_pred, _ = self.sem_pred.get_prediction(rgb.astype(np.uint8))
            semantic_image = np.zeros(rgb.shape[:2])
            for name, label in coco_categories.items():
                # NOTE: zero for background 
                semantic_image[sem_seg_pred[:,:,label] == 1] = label + 1 
            obs['semantic'] = semantic_image

        if DEBUG_VIS:
            import matplotlib.pyplot as plt 
            # import numpy as np
            fig = plt.figure(figsize=(12,4))
            fig.add_subplot(131)   #top left
            plt.imshow(obs['rgb'])
            fig.add_subplot(132)   #top left
            plt.imshow(obs['depth'].squeeze())
            fig.add_subplot(133)   #top left
            plt.imshow(obs['semantic'])
            plt.show()


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

    #------------- multion_env util functions ------------
    def conv_grid(
        self,
        realworld_x,
        realworld_y,
        coordinate_min = COORDINATE_MIN,
        coordinate_max = COORDINATE_MAX,
        grid_resolution = (300, 300)
    ):
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        grid_x = int((coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - coordinate_min) / grid_size[1])
        return grid_x, grid_y

                


#! /usr/bin/env python
from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
from matplotlib import docstring
import torch.nn as nn
import torch
import numpy as np

# from model import RL_Policy, Semantic_Mapping
# from utils.storage import GlobalRolloutStorage
from envs import make_vec_envs
from envs.habitat import construct_single_env
from arguments import get_args

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import open3d as o3d
import rospy
from agents.frontier_2d_detect_agent import Frontier2DDetectionAgent
from arguments import get_args
from simulator import init_sim

# from std_msgs.msg import Int32
from subscribers import PointCloudSubscriber
from utils import transformation


def main():
    # Initialize ROS node and take arguments
    rospy.init_node("habitat_ros_node")
    node_start_time = rospy.Time.now().to_sec()

    # TODO: set all required arguments from rosparam server
    args = get_args("")  # use default arguments for now
    # overwrite default arguments
    DEFAULT_AGENT_TYPE = "frontier_2d_detection"
    args.agent = rospy.get_param("~agent_type", DEFAULT_AGENT_TYPE)
    args.config_dir = rospy.get_param("~config_dir", args.config_dir)
    args.task_config = rospy.get_param("~task_config", args.task_config)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # region: Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(filename=log_dir + "train.log", level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)
    # endregion

    # region: Logging and Initialize loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    best_g_reward = -np.inf

    episode_success = deque(maxlen=num_episodes)
    episode_spl = deque(maxlen=num_episodes)
    episode_dist = deque(maxlen=num_episodes)

    finished = 0
    wait_env = 0

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))
    # endregion

    # initialize agent and envrionment
    # TODO: Decouple ObjectGoal_Env class and Agent class
    # envs = make_vec_envs(args)
    env = construct_single_env(args)
    # also wait for rtabmap_ros initialization in env.reset()
    obs, infos = env.reset()

    torch.set_grad_enabled(False)
    
    # obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
    # endregion

    # region: Run steps
    start = time.time()
    g_reward = 0
    done = False
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.total_steps):
    # while(True)
        if finished:
            break

        # g_step = (step // args.num_local_steps) % args.num_global_steps
        # l_step = step % args.num_local_steps

        if done:
            spl = info["spl"]
            success = info["success"]
            dist = info["distance_to_goal"]
            spl_per_category[info["goal_name"]].append(spl)
            success_per_category[info["goal_name"]].append(success)
            if args.eval:
                episode_success.append(success)
                episode_spl.append(spl)
                episode_dist.append(dist)
                if len(episode_success) == num_episodes:
                    finished = 1
            else:
                episode_success.append(success)
                episode_spl.append(spl)
                episode_dist.append(dist)

            wait_env = 1
            # update_intrinsic_rew(e)
            # init_map_and_pose_for_env(e)

        # region: 5. Take action and get next observation

        obs, rew, done, info = env.plan_act_and_preprocess()
        # endregion
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # region: 7. Logging
        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join(
                [
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(step * num_scenes),
                    "FPS {},".format(int(step * num_scenes / (end - start))),
                ]
            )

            log += "\n\tRewards:"

            if len(g_episode_rewards) > 0:
                log += " ".join(
                    [
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards),
                        ),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards),
                        ),
                    ]
                )

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for acc in episode_success:
                    total_success.append(acc)
                for dist in episode_dist:
                    total_dist.append(dist)
                for spl in episode_spl:
                    total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl),
                    )
            else:
                if len(episode_success) > 100:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(episode_success),
                        np.mean(episode_spl),
                        np.mean(episode_dist),
                        len(episode_spl),
                    )

            log += "\n\tLosses:"
            if len(g_value_losses) > 0 and not args.eval:
                log += " ".join(
                    [
                        " Policy Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies),
                        ),
                    ]
                )

            print(log)
            logging.info(log)
        # endregion
        # ------------------------------------------------------------------

    # region: Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")

        total_success = []
        total_spl = []
        total_dist = []
        for acc in episode_success:
            total_success.append(acc)
        for dist in episode_dist:
            total_dist.append(dist)
        for spl in episode_spl:
            total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl),
            )

        print(log)
        logging.info(log)

        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(
                key,
                sum(success_per_category[key])
                / len(success_per_category[key]),
                sum(spl_per_category[key]) / len(spl_per_category[key]),
            )

        print(log)
        logging.info(log)

        with open(
            "{}/{}_spl_per_cat_pred_thr.json".format(dump_dir, args.split), "w"
        ) as f:
            json.dump(spl_per_category, f)

        with open(
            "{}/{}_success_per_cat_pred_thr.json".format(dump_dir, args.split),
            "w",
        ) as f:
            json.dump(success_per_category, f)
    # endregion


if __name__ == "__main__":
    main()

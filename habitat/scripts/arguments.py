import argparse
import torch
import sys


def get_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="ObjectGoalNav Evaluation arguments"
    )

    # General Arguments
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    parser.add_argument("--total_num_scenes", type=str, default="auto")
    parser.add_argument(
        "-n",
        "--num_processes",
        type=int,
        default=1,
        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=1,
        help="0: Train, 1: Evaluate (default: 0)",
    )
    parser.add_argument(
        "--num_train_episodes",
        type=int,
        default=sys.maxsize,  # 10000
        help="""number of train episodes per scene
                before loading the next scene by objectgoal_env.reset()""",
    )
    parser.add_argument('--total_steps', type=int, default=10000000)
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=200,
        help="number of test episodes per scene",
    )

    # parser.add_argument(
    #     "--no_cuda",
    #     action="store_true",
    #     default=True,
    #     help="disables CUDA training",
    # )

    # Logging, loading models, visualization
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="""log interval, one log per n updates
                                (default: 10) """,
    )
    parser.add_argument(
        "--save_interval", type=int, default=1, help="""save interval"""
    )
    parser.add_argument(
        "-d",
        "--dump_location",
        type=str,
        default="./tmp/",
        help="path to dump models and log (default: ./tmp/)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp1",
        help="experiment name (default: exp1)",
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=500000,
        help="Model save frequency in number of updates",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="0",
        help="""model path to load,
                                0 to not reload (default: 0)""",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        type=int,
        default=0,
        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""",
    )
    parser.add_argument(
        "--print_images",
        type=int,
        default=0,
        help="1: save visualization as images",
    )

    # TODO: parse task parameters also from task yaml config 
    parser.add_argument(
        "-el",
        "--max_episode_length",
        type=int,
        default=200, # 500
        help="""Maximum episode length""",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="habitat/scripts/envs/habitat/configs",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        # default="tasks/objectnav_gibson.yaml",
        default="tasks/objectnav_gibson_high_res.yaml",
        help="path to config yaml containing task information",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val", 
        help="dataset split (train | val | val_mini) ",
    )
    parser.add_argument(
        "--success_dist",
        type=float,
        default=1.0,
        help="success distance threshold in meters",
    )
    parser.add_argument(
        "--floor_thr", type=int, default=50, help="floor threshold in cm"
    )
    parser.add_argument(
        "--min_d",
        type=float,
        default=1.5,
        help="min distance to goal during training in meters",
    )
    parser.add_argument(
        "--max_d",
        type=float,
        default=100.0,
        help="max distance to goal during training in meters",
    )
    parser.add_argument(
        "--version", type=str, default="v1.1", help="dataset version"
    )

    # Model Hyperparameters
    # TODO: clean these parameters since RL model is not used 
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('--reward_coeff', type=float, default=0.1,
                        help="Object goal reward coefficient")
    parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.02,
                        help="intrinsic exploration reward coefficient")
    parser.add_argument('--num_sem_categories', type=float, default=16)

    # semantic model (ground truth / detectron2) 
    parser.add_argument('--sem_model', type=str, default="detectron", 
        choices=["none", "ground_truth", "detectron"])
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold")
    parser.add_argument('--sem_config_dir', type=str, default="habitat/scripts/agents/configs")
    parser.add_argument('--sem_device', type=str, default="gpu")
    parser.add_argument('--sem_dbscan_eps', type=float, default=1.0, 
        help="eps parameter in sklearn.cluster.DBSCAN to group detected targets")

    # Mapping
    parser.add_argument("--global_downscaling", type=int, default=2)
    parser.add_argument("--vision_range", type=int, default=100)
    parser.add_argument("--map_resolution_cm", type=int, default=5)
    parser.add_argument("--du_scale", type=int, default=1)
    parser.add_argument("--map_size_cm", type=int, default=2400)
    parser.add_argument("--cat_pred_threshold", type=float, default=5.0)
    parser.add_argument("--map_pred_threshold", type=float, default=1.0)
    parser.add_argument("--exp_pred_threshold", type=float, default=1.0)
    parser.add_argument("--collision_threshold", type=float, default=0.20)
    
    # Frontier Exploration
    parser.add_argument("--cluster_trashhole", type=float, default=0.2)
    # parser.add_argument("--frontier_mode", type=str, default="geo")
    # Utility function argumetns 
    parser.add_argument("--util_prior_combine_weight", type=float, default=1.0,
                        help="combine weight for semantic utility, utility=weight * scene_util + (1.0-weight)*lang_util") 
    parser.add_argument("--util_max_geo_weight", type=float, default=1.0, 
                        help="maximum weight for geometric utility")
    parser.add_argument("--util_min_geo_weight", type=float, default=0.2, 
                        help="maximum weight for geometric utility")
    parser.add_argument("--util_weight_dec_step", type=int, default=100, 
                        help="maximum steps to decrease geo weight linearly")
    # parse arguments
    args = parser.parse_args(args=args, namespace=namespace)

    args.cuda = True
    args.sem_gpu_id = 0

    return args

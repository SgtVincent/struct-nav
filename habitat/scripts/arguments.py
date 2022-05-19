import argparse
import torch


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
        default=5,
        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        help="0: Train, 1: Evaluate (default: 0)",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=200,
        help="number of test episodes per scene",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=True,
        help="disables CUDA training",
    )

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

    # Environment, dataset and episode specifications
    parser.add_argument(
        "-efw",
        "--env_frame_width",
        type=int,
        default=640,
        help="Frame width (default:640)",
    )
    parser.add_argument(
        "-efh",
        "--env_frame_height",
        type=int,
        default=480,
        help="Frame height (default:480)",
    )
    parser.add_argument(
        "-fw",
        "--frame_width",
        type=int,
        default=160,
        help="Frame width (default:160)",
    )
    parser.add_argument(
        "-fh",
        "--frame_height",
        type=int,
        default=120,
        help="Frame height (default:120)",
    )
    parser.add_argument(
        "-el",
        "--max_episode_length",
        type=int,
        default=500,
        help="""Maximum episode length""",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="tasks/objectnav_gibson.yaml",
        help="path to config yaml containing task information",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="dataset split (train | val | val_mini) ",
    )
    parser.add_argument(
        "--camera_height",
        type=float,
        default=0.88,
        help="agent camera height in metres",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=79.0,
        help="horizontal field of view in degrees",
    )
    parser.add_argument(
        "--turn_angle",
        type=float,
        default=30,
        help="Agent turn angle in degrees",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.5,
        help="Minimum depth for depth sensor in meters",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=5.0,
        help="Maximum depth for depth sensor in meters",
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

    # Mapping
    parser.add_argument("--global_downscaling", type=int, default=2)
    parser.add_argument("--vision_range", type=int, default=100)
    parser.add_argument("--map_resolution", type=int, default=5)
    parser.add_argument("--du_scale", type=int, default=1)
    parser.add_argument("--map_size_cm", type=int, default=2400)
    parser.add_argument("--cat_pred_threshold", type=float, default=5.0)
    parser.add_argument("--map_pred_threshold", type=float, default=1.0)
    parser.add_argument("--exp_pred_threshold", type=float, default=1.0)
    parser.add_argument("--collision_threshold", type=float, default=0.20)

    # parse arguments
    args = parser.parse_args(args=args, namespace=namespace)

    args.cuda = False
    args.sem_gpu_id = -2

    return args

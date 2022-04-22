import argparse


def get_args(silence_mode=False):
    parser = argparse.ArgumentParser(description="Goal-Oriented-Semantic-Exploration")
    # General Arguments
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

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
        "--exp_name", type=str, default="eval", help="experiment name (default: eval)"
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
        "--print_images", type=int, default=0, help="1: save visualization as images"
    )

    # Environment, dataset and episode specifications
    parser.add_argument(
        "-efw",
        "--env_frame_width",
        type=int,
        default=1280,
        help="Frame width (default:640)",
    )
    parser.add_argument(
        "-efh",
        "--env_frame_height",
        type=int,
        default=720,
        help="Frame height (default:480)",
    )
    parser.add_argument(
        "-fw", "--frame_width", type=int, default=160, help="Frame width (default:160)"
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
    # parser.add_argument(
    #     "--task_config",
    #     type=str,
    #     default="tasks/objectnav_gibson.yaml",
    #     help="path to config yaml containing task information",
    # )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="dataset split (train | val | val_mini) ",
    )
    parser.add_argument(
        "--camera_height",
        type=float,
        default=1.5,
        help="agent camera height in metres (default 0.88)",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90.0,
        help="horizontal field of view in degrees (default 79.0)",
    )
    parser.add_argument(
        "--turn_angle", type=float, default=30, help="Agent turn angle in degrees"
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

    # Frontier Exploration
    parser.add_argument("--cluster_trashhole", type=float, default=0.3)

    if silence_mode:
        # use default arguments
        args = parser.parse_args("")
    else:
        # read from sys.args
        args = parser.parse_args()

    return args

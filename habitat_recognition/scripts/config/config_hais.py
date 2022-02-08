import os
from os.path import dirname, join
import numpy as np

CONFIG_DIR = dirname(os.path.abspath(__file__))
ROOT_DIR = dirname(CONFIG_DIR)
METADATA_DIR = join(ROOT_DIR, "metadata")
HAIS_DIR = join(ROOT_DIR, "models", "HAIS")


class ConfigHAIS:
    def __init__(self) -> None:

        ################## dataset config #################
        self.dataset = "mp3d"
        self.dataset_label = "mpcat40"
        self.dataset_dir = (
            "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans"
        )
        self.label_mapping_file = join(METADATA_DIR, "mpcat40_to_nyu40.tsv")
        # TODO: move dataset details to ./dataset/mp3d/config_mp3d.py
        # once DatasetMP3D implemented

        # self.split = "val" # will be overwritten by yaml config file
        self.downsample_method = "random"  # ["random", "uniform", "voxel"]
        self.downsample_voxel_size = 0.02  # meters

        ################## model config ###################
        self.yaml_file = join(HAIS_DIR, "config", "hais_eval_scannet.yaml")
        self.run_mode = "test"
        self.pretrain = "/home/junting/project_cvl/HAIS/pretrain/hais_ckpt.pth"
        self.model_label = "nyu40"
        self.label2nyu40 = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            14,
            16,
            24,
            28,
            33,
            34,
            36,
            39,
        ]
        self.nyu40id2label = {
            nyu40id: i for i, nyu40id in enumerate(self.label2nyu40)
        }
        ##################### eval config ###############
        self.eval_logging = True
        self.eval_task = "detection"
        self.eval_dir = join(ROOT_DIR, "tb_logs")

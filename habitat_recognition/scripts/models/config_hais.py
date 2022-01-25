import os
from os.path import dirname, join
import numpy as np

MODELS_DIR = dirname(os.path.abspath(__file__))
HAIS_DIR = join(MODELS_DIR, "HAIS")
PROJECT_DIR = dirname(dirname(MODELS_DIR))


class ConfigHAIS:
    def __init__(self) -> None:

        ################## dataset config #################
        self.num_class = 18
        self.num_heading_bin = 1
        self.num_size_cluster = 18
        # self.split = "val" # will be overwritten by yaml config file

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}

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

        ################## model config ###################
        self.yaml_file = join(HAIS_DIR, "config", "hais_eval_scannet.yaml")
        self.run_mode = "test"
        self.pretrain = "/home/junting/project_cvl/HAIS/pretrain/hais_ckpt.pth"

        ##################### eval config ###############
        self.eval_task = "segmentation"
        self.eval_dir = join(PROJECT_DIR, "tb_logs")

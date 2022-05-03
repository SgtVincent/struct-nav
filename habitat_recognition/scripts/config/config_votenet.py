import os

import numpy as np


class ConfigVoteNet:
    def __init__(self) -> None:
        # dataset config
        self.num_class = 18
        self.num_heading_bin = 1
        self.num_size_cluster = 18

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
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
        self.mean_size_arr = np.load(
            "/home/junting/project_cvl/votenet/scannet/meta_data/scannet_means.npz"
        )["arr_0"]
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

        # model config
        self.num_target = 256
        self.use_height = True
        self.use_color = False
        self.num_input_channel = (
            int(self.use_color) * 3 + int(self.use_height) * 1
        )
        self.vote_factor = 1
        self.cluster_sampling = "seed_fps"  # vote_fps, seed_fps, random
        self.bbox_aligned = True  # bounding box aligned with xyz axes

        # eval config
        self.eval_realtime = True
        self.eval_dir = (
            "/home/junting/habitat_ws/src/habitat_detect_segment/eval_output"
        )
        self.label_mapping_file = "/home/junting/habitat_ws/src/habitat_detect_segment/scripts/data/mpcat40_to_nyu40.tsv"
        self.faster_eval = False
        self.use_3d_nms = True
        self.nms_iou = 0.25
        self.use_old_type_nms = False
        self.use_cls_nms = True
        self.per_class_proposal = True
        self.conf_thresh = 0.05
        self.semantic_scene = None
        self.ap_iou_thresholds = [0.25, 0.5]
        # visulization config

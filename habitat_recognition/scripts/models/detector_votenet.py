import os
import sys

import numpy as np
import rospy
import torch

# local import
from evaluation.eval_3d_detect import Eval3DDetection
from utils.votenet_utils import parse_groundtruths, parse_predictions

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
VOTENET_DIR = os.path.join(MODEL_DIR, "votenet")
sys.path.append(MODEL_DIR)
# sys.path.append(VOTENET_DIR)
# sys.path.append(os.path.join(VOTENET_DIR, 'utils'))
# sys.path.append(os.path.join(VOTENET_DIR, 'pointnet2'))
sys.path.append(os.path.join(VOTENET_DIR, "models"))
from ap_helper import APCalculator
from config_votenet import ConfigVoteNet
from votenet import VoteNet


class DetectorVoteNet:
    def __init__(self, config, device="cuda") -> None:

        # initialize model
        self.config = config
        self.net = VoteNet(
            num_class=config.num_class,
            num_heading_bin=config.num_heading_bin,
            num_size_cluster=config.num_size_cluster,
            mean_size_arr=config.mean_size_arr,
            num_proposal=config.num_target,
            input_feature_dim=config.num_input_channel,
            vote_factor=config.vote_factor,
            sampling=config.cluster_sampling,
        )
        self.device = device
        assert self.device != "cpu"  # pointnet2 only supports GPU
        self.net.to(device)
        self.parse_predict_config = {
            "num_class": config.num_class,
            "remove_empty_box": (not config.faster_eval),
            "use_3d_nms": config.use_3d_nms,
            "nms_iou": config.nms_iou,
            "use_old_type_nms": config.use_old_type_nms,
            "cls_nms": config.use_cls_nms,
            "per_class_proposal": config.per_class_proposal,
            "conf_thresh": config.conf_thresh,
            "mean_size_arr": config.mean_size_arr,
        }

        # initialize evaluator
        self.ap_calculators = {
            iou_thresh: APCalculator(iou_thresh)
            for iou_thresh in self.config.ap_iou_thresholds
        }
        if self.config.eval_realtime:
            self.evaluator = Eval3DDetection(self.config, self.ap_calculators)

        return

    def detect(self, o3d_pcd):

        input = self.generate_input(o3d_pcd)
        with torch.no_grad():
            end_points = self.net(input)

        pred_map_cls = parse_predictions(end_points, self.parse_predict_config)
        # pred_mat_cls: a list (len: batch_size) of list (len: num of predictions per sample) of tuples of
        # (pred_cls, pred_box, conf(0-1))
        return pred_map_cls

    def evaluate(self, pred_map_cls):

        # ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        #     for iou_thresh in AP_IOU_THRESHOLDS]
        if self.config.eval_realtime:
            self.evaluator.eval(pred_map_cls)

        return

    def generate_input(self, o3d_pcd):

        pcd_xyz = np.asarray(o3d_pcd.points)
        point_cloud = pcd_xyz

        if self.config.use_height:
            floor_height = np.percentile(pcd_xyz[:, 2], 0.99)
            height = pcd_xyz[:, 2] - floor_height
            point_cloud = np.concatenate(
                [pcd_xyz, np.expand_dims(height, 1)], 1
            )

        # if self.config.use_color:
        #     pcd_rgb = np.asarray(o3d_pcd.colors)
        #     point_cloud = np.concatenate([point_cloud, pcd_rgb],1)

        input = {}
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32)).to(
            self.device
        )
        input["point_clouds"] = torch.unsqueeze(
            point_cloud, 0
        )  # add batch dimension
        return input

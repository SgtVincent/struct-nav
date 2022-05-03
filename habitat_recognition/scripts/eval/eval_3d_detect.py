from logging import log
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from os.path import join, basename
import numpy as np
from scipy.spatial import KDTree
from timeit import default_timer as timer
import re

# local import
from utils.votenet_utils import parse_groundtruths
from dataset.mp3d.utils import read_label_mapping
from utils.box_util import get_3d_box
import pathlib

PACKAGE_DIR = pathlib.Path(
    __file__
).parent.parent.parent.absolute()  # package dir
DEFAULT_LOGDIR = join(PACKAGE_DIR, "tb_logs")
if not os.path.exists(DEFAULT_LOGDIR):
    os.makedirs(DEFAULT_LOGDIR)


class Eval3DDetection:
    def __init__(
        self,
        config,
        ap_calculators,
        logging=True,
        log_dir=DEFAULT_LOGDIR,
        bbox_aligned=True,
    ) -> None:

        ############################### parameters #########################3#
        self.semantic_scene = config.semantic_scene
        self.label_mapping_file = config.label_mapping_file
        self.type2class = config.type2class
        self.class2type = config.class2type
        self.nyu40ids = config.nyu40ids
        self.nyu40id2class = config.nyu40id2class

        self.ap_calculators = ap_calculators
        self.obj_threshold = 0.5
        self.eval_count = 0
        self.time_start = None  # initialized when first calling eval()
        self.rel_same_part = 7  # "same part" relation label
        self.use_time_stamp = (
            True  # use time stamp as x-axis of evaluation plots
        )

        ############## ground truth bounding boxes #####################
        # parse ground truth semantic scene to format:
        # list of tuples of (pred_cls, pred_box)
        self.gt_map_cls = []
        self.label_mapping_mp40_nyu40 = read_label_mapping(
            self.label_mapping_file, label_from="mpcat40", label_to="nyu40id"
        )

        for obj in self.semantic_scene.objects:

            # parse votenet label
            if obj.category.name() in self.label_mapping_mp40_nyu40:
                gt_nyu40_id = self.label_mapping_mp40_nyu40[
                    obj.category.name()
                ]
            else:  # category unknown, set to "misc", with nyu40id=40
                gt_nyu40_id = 40

            if gt_nyu40_id in self.nyu40ids:
                gt_cls = self.nyu40id2class[gt_nyu40_id]
            else:  # class not used to train votenet
                continue

            if bbox_aligned:
                gt_bbox = get_3d_box(obj.aabb.sizes, 0, obj.aabb.center)
            else:  # Use obj.obb, which has rotation
                # TODO: implement obb parsing
                raise NotImplementedError

            self.gt_map_cls.append((gt_cls, gt_bbox))
        self.gt_map_cls = [
            self.gt_map_cls
        ]  # aligned with shape of pred_map_cls

        ################# set logger ######################
        self.logging = logging
        if self.logging:
            self.log_dir = join(
                log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            self.logger = SummaryWriter(self.log_dir)

    def eval(self, pred_map_cls):

        # meta info
        self.eval_count += 1
        if self.time_start is None:
            self.time_start = timer()
        time_now = timer()
        time_elapse = time_now - self.time_start
        x_axis = self.eval_count
        if self.use_time_stamp:
            x_axis = time_elapse

        # Evaluate average precision
        for iou_thresh in self.ap_calculators.keys():
            ap_calculator = self.ap_calculators[iou_thresh]
            # print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
            ap_calculator.step(pred_map_cls, self.gt_map_cls)
            metrics_dict = ap_calculator.compute_metrics()
            ap_calculator.reset()

            if self.logging:

                self.logger.add_scalar(
                    f"mean AP @{iou_thresh}", metrics_dict["mAP"], x_axis
                )
                self.logger.add_scalar(
                    f"mean Recall @{iou_thresh}", metrics_dict["AR"], x_axis
                )

                # log class-specific analytics
                ap_re = re.compile("(\w+) Average Precision")
                rc_re = re.compile("(\w+) Recall")
                for key, val in metrics_dict.items():
                    if "Average Precision" in key:
                        class_name = ap_re.findall(key)[0]
                        self.logger.add_scalar(
                            f"Average Precision @{iou_thresh}/{class_name}",
                            val,
                            x_axis,
                        )
                    elif "Recall" in key:
                        class_name = rc_re.findall(key)[0]
                        self.logger.add_scalar(
                            f"Recall @{iou_thresh}/{class_name}", val, x_axis
                        )
                    else:  # skip
                        pass

        return

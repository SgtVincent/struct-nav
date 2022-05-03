import datetime
import os
import time
from re import L

import numpy as np
import open3d as o3d

# local import 
from config.config_hais import ConfigHAIS
from dataset.mp3d.utils import getOBB, read_house_file, read_label_mapping
from eval.ap_helper import APCalculator
from tensorboardX import SummaryWriter


class EvaluatorHAIS:
    def __init__(self, scene_name, config=ConfigHAIS()) -> None:
    
        ############################### parameters #########################3#
        self.scene_name = scene_name
        self.scene_dir = os.path.join(config.dataset_dir, scene_name)
        self.dataset_name = config.dataset
        self.dataset_label = config.dataset_label # "mp3d"
        self.label_mapping_file = config.label_mapping_file
        
        # TODO: now the model label fixed to nyu40, try to generalize this part 
        self.model_label = config.model_label # "nyu40"
        self.label2nyu40id = config.label2nyu40id
        self.nyu40id2label = config.nyu40id2label

        ############## dataset loading #####################
        # parse ground truth semantic scene to format:
        # list of tuples of (pred_cls, pred_box)
        self.gt_map_cls = []
        self.label_mapping_dataset2model = read_label_mapping(
            self.label_mapping_file, 
            label_from=self.dataset_label, 
            label_to=self.model_label
        )
        # TODO: 
        # 1. Use dataset_mp3d.DatasetMP3D instead of loading data in evaluator directly 
        # for better interface 
        # 2. Generalize dataset class for using other datasets 
        house_file = os.path.join(self.scene_dir, f"{self.scene_name}.house")
        ply_file = os.path.join(self.scene_dir, f"{self.scene_name}_semantic.ply")

        self.house_dict = read_house_file(house_file)
        self.gt_o3d_pcl= o3d.io.read_point_cloud(ply_file)
        self.gt_points = np.asarray(self.gt_o3d_pcl.points)

        # O object_index region_index category_index px py pz  a0x a0y a0z  a1x a1y a1z  r0 r1 r2 0 0 0 0 0 0 0 0 
        # for line in self.house_dict['O']:
        #     center, boxRotation, radii = getOBB(line)
        ################# set logger ######################
        self.eval_logging = config.eval_logging
        self.eval_dir = config.eval_dir
        if self.eval_logging:
            self.log_dir = os.path.join(
                self.eval_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            self.logger = SummaryWriter(self.log_dir)

    # TODO: call evaluator.eval() here
    def evaluate(self, o3d_pcl, pred_info):

        # meta info -- use eval count or time for x-axis in figure
        self.eval_count += 1
        if self.time_start is None:
            self.time_start = time.time()
        time_now = time.time()
        time_elapse = time_now - self.time_start

        if self.use_time_stamp:
            x_axis = time_elapse
        else:
            x_axis = self.eval_count

        # evaluation:
        # NOTE: since reconstructed mesh
        stat_dict = {}
        ap_calculator_list = [
            APCalculator(iou_thresh, DATASET_CONFIG.class2type)
            for iou_thresh in AP_IOU_THRESHOLDS
        ]
        net.eval()  # set model to eval mode (for bn and dp)
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            if batch_idx % 10 == 0:
                print("Eval batch: %d" % (batch_idx))
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)

            # Forward pass
            inputs = {"point_clouds": batch_data_label["point_clouds"]}
            with torch.no_grad():
                end_points = net(inputs)

            # Compute loss
            for key in batch_data_label:
                assert key not in end_points
                end_points[key] = batch_data_label[key]
            loss, end_points = criterion(end_points, DATASET_CONFIG)

            # Accumulate statistics and print out
            for key in end_points:
                if "loss" in key or "acc" in key or "ratio" in key:
                    if key not in stat_dict:
                        stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

            # Dump evaluation results for visualization
            if batch_idx == 0:
                MODEL.dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

        # Log statistics
        for key in sorted(stat_dict.keys()):
            log_string(
                "eval mean %s: %f"
                % (key, stat_dict[key] / (float(batch_idx + 1)))
            )

        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            print(
                "-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10
            )
            metrics_dict = ap_calculator.compute_metrics()
            for key in metrics_dict:
                log_string("eval %s: %f" % (key, metrics_dict[key]))

        mean_loss = stat_dict["loss"] / float(batch_idx + 1)

        # TODO: add logger after evaluation finished
        # if self.logging:
        if False:
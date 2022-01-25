import os
from ssl import HAS_ALPN
import sys
from unicodedata import name

import numpy as np
import scipy
import open3d as o3d
import torch
import yaml

# local import
##################################################################
# NOTE: There are relative imports in HAIS original code
# FIX: append those paths to sys.path
#################################################################
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
HAIS_DIR = os.path.join(FILE_DIR, "HAIS")
LIB_DIR = os.path.join(HAIS_DIR, "lib")
sys.path.append(HAIS_DIR)

from models.HAIS.model.hais.hais import HAIS, model_fn_decorator
from models.HAIS.util.utils import checkpoint_restore
from models.config_hais import ConfigHAIS
from dataset.mp3d.mp3d_hais import mp3dHAISDataset

# import wrapper functions of C++ ops
from models.HAIS.lib.hais_ops.functions import hais_ops


# TODO: implement base class for detector and segmenter if needed
class SegmenterHAIS:
    def __init__(self, config=ConfigHAIS()):

        ########## read parameters in config ###########
        # read model config from original yaml config file
        with open(config.yaml_file, "r") as f:
            hais_config = yaml.load(f, Loader=yaml.FullLoader)
        for key in hais_config:
            for k, v in config[key].items():
                setattr(config, k, v)

        self.config = config
        self.batch_size = config.batch_size
        # self.train_workers = config.train_workers
        # self.val_workers = config.train_workers
        self.full_scale = config.full_scale
        self.scale = config.scale
        self.max_npoint = config.max_npoint
        # voxelization mode 0=guaranteed unique 1=last item(overwrite) 2=first item(keep) 3=sum, 4=mean
        self.mode = config.mode
        self.model_epoch = config.test_epoch
        self.split = config.split
        self.label2nyu40 = config.label2nyu40

        ############# initialize dataset ###########
        # dataset class is the interface to convert scene pointclouds to tensors accepted by model
        # data augmentation and other utils not used yet
        # self.dataset = mp3dHAISDataset(self.config)
        self.batch_id = 0
        # TODO: load ground truth annotations from dataset here
        self.load_scene()

        ############## initialize model ####################

        use_cuda = torch.cuda.is_available()
        print("cuda available: {}".format(use_cuda))
        assert use_cuda

        self.model = HAIS(config)
        self.model = self.model.cuda()

        # print(
        #     "#classifier parameters (model): {}".format(
        #         sum([x.nelement() for x in self.model.parameters()])
        #     )
        # )
        self.model_fn = model_fn_decorator(test=True)

        # load model
        # resume from the latest epoch, or specify the epoch to restore
        checkpoint_restore(
            config,
            self.model,
            None,
            "",
            "",
            use_cuda,
            0,
            dist=False,
            f=config.pretrain,
        )

        # never train the model in this ros package
        self.model = self.model.eval()
        # TODO: add code here

        ############ initialize dataset annotations ######################

    def predict(self, o3d_pcl):

        batch = self._make_batch(o3d_pcl)
        preds = self.model_fn(batch, self.model, self.model_epoch)

        # decode results for
        N = batch["feats"].shape[0]
        # NOTE: since reconstructed point clouds has no GT segmentation annots,
        # instance segmentation cannot be evaluated

        semantic_scores = preds["semantic"]  # (N, nClass=20) float32, cuda
        semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
        pt_offsets = preds["pt_offsets"]  # (N, 3), float32, cuda

        # if self.model_epoch > self.config.prepare_epochs:
        scores = preds["score"]  # (nProposal, 1) float, cuda
        scores_pred = torch.sigmoid(scores.view(-1))

        # proposals_idx: (sumNPoint, 2), int, cpu, [:, 0] for cluster_id, [:, 1] for corresponding point idxs in N
        # proposals_offset: (nProposal + 1), int, cpu
        proposals_idx, proposals_offset, mask_scores = preds["proposals"]

        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, N),
            dtype=torch.int,
            device=scores_pred.device,
        )  # (nProposal, N), int, cuda

        # outlier filtering
        test_mask_score_thre = getattr(
            self.config, "test_mask_score_thre", -0.5
        )
        _mask = mask_scores.squeeze(1) > test_mask_score_thre
        proposals_pred[
            proposals_idx[_mask][:, 0].long(),
            proposals_idx[_mask][:, 1].long(),
        ] = 1
        semantic_id = torch.tensor(
            self.label2nyu40, device=scores_pred.device
        )[
            semantic_pred[
                proposals_idx[:, 1][proposals_offset[:-1].long()].long()
            ]
        ]  # (nProposal), long

        # score threshold
        # filter out inst proposals whose confidence score < TEST_SCORE_THRESH
        score_mask = scores_pred > self.config.TEST_SCORE_THRESH
        scores_pred = scores_pred[score_mask]
        proposals_pred = proposals_pred[score_mask]
        semantic_id = semantic_id[score_mask]
        # semantic_id_idx = semantic_id_idx[score_mask]

        # npoint threshold
        # filter out inst proposals which have < TEST_NPOINT_THRESH number of points
        proposals_pointnum = proposals_pred.sum(1)
        npoint_mask = proposals_pointnum >= self.config.TEST_NPOINT_THRESH
        scores_pred = scores_pred[npoint_mask]
        proposals_pred = proposals_pred[npoint_mask]
        semantic_id = semantic_id[npoint_mask]

        # NOTE: NMS not used in original HAIS model evaluation
        # nms (no need)
        # if getattr(self.config, "using_NMS", False):
        if False:
            if semantic_id.shape[0] == 0:
                pick_idxs = np.empty(0)
            else:
                proposals_pred_f = (
                    proposals_pred.float()
                )  # (nProposal, N), float, cuda
                intersection = torch.mm(
                    proposals_pred_f, proposals_pred_f.t()
                )  # (nProposal, nProposal), float, cuda
                proposals_pointnum = proposals_pred_f.sum(
                    1
                )  # (nProposal), float, cuda
                proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(
                    1, proposals_pointnum.shape[0]
                )
                proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(
                    proposals_pointnum.shape[0], 1
                )
                cross_ious = intersection / (
                    proposals_pn_h + proposals_pn_v - intersection
                )
                pick_idxs = non_max_suppression(
                    cross_ious.cpu().numpy(),
                    scores_pred.cpu().numpy(),
                    cfg.TEST_NMS_THRESH,
                )
                # int, (nCluster, N)
            clusters = proposals_pred[pick_idxs]
            cluster_scores = scores_pred[pick_idxs]
            cluster_semantic_id = semantic_id[pick_idxs]
        else:
            clusters = proposals_pred
            cluster_scores = scores_pred
            cluster_semantic_id = semantic_id

        nclusters = clusters.shape[0]

        pred_info = {}
        pred_info["inst_mask"] = clusters.cpu().numpy()  # (nCluster, Npoints)
        pred_info["inst_conf"] = cluster_scores.cpu().numpy()  # (nCluster)
        pred_info["label_id"] = cluster_semantic_id.cpu().numpy()  # (nCluster)

        return pred_info

    # TODO: call evaluator.eval() here
    def evaluate(self, pred_info):
        pass

    # TODO: load ground truth annotations
    def load_scene(self):
        pass

    # this function converts open3d point clouds and annotations to batch input accepted by model
    def _make_batch(self, o3d_pcl):

        # load from open3d point cloud
        xyz_origin = np.asarray(o3d_pcl.points)
        rgb = np.asarray(o3d_pcl.colors) * 2.0 - 1  # from [0,1] to [-1,1]

        locs = []
        locs_float = []
        feats = []

        batch_offsets = [0]
        # only inference once (one pointcloud)
        for i, idx in enumerate([self.batch_id]):

            # scale
            xyz = xyz_origin * self.scale
            # offset
            xyz -= xyz.min(0)
            # merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(
                torch.cat(
                    [
                        torch.LongTensor(xyz.shape[0], 1).fill_(i),
                        torch.from_numpy(xyz).long(),
                    ],
                    1,
                )
            )
            locs_float.append(torch.from_numpy(xyz_origin))
            feats.append(torch.from_numpy(rgb))

        # merge all the scenes in the batch
        batch_offsets = torch.tensor(
            batch_offsets, dtype=torch.int
        )  # int (B+1)
        locs = torch.cat(
            locs, 0
        )  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip(
            (locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None
        )  # long (3)

        # voxelize
        voxel_locs, p2v_map, v2p_map = hais_ops.voxelization_idx(
            locs, self.batch_size, self.mode
        )

        batch = {
            "locs": locs,  # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
            "voxel_locs": voxel_locs,  # (M, 1 + 3), long, cuda
            "p2v_map": p2v_map,  # (N), int, cuda
            "v2p_map": v2p_map,  # (M, 1 + maxActive), int, cuda
            "locs_float": locs_float,  # (N, 3), float32, cuda
            "feats": feats,  # (N, C), float32, cuda
            "id": self.batch_id,  # list of batch_idx in this dict
            "offsets": batch_offsets,  # (B + 1), int, cuda
            "spatial_shape": spatial_shape,  # (3), numpy array, voxel size in (x,y,z) directions
        }

        self.batch_id += 1

        return batch


# test this class with python -m models.segmenter_hais under ./scripts path
if __name__ == "__main__":

    config = ConfigHAIS()
    seg_hais = SegmenterHAIS(config)

    # change scans_dir to correct path
    scans_dir = (
        "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans"
    )
    scene_name = "17DRP5sb8fy"
    ply_path = os.path.join(scans_dir, scene_name)
    o3d_pcl = o3d.io.read_point_cloud(ply_path)

    result = seg_hais.predict(o3d_pcl)


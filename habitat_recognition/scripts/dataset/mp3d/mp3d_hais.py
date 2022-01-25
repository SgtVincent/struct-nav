import os
import sys

import numpy as np
import scipy
import open3d as o3d
import torch
import yaml

# local import
from models.config_hais import ConfigHAIS

# import wrapper functions of C++ ops
from models.HAIS.lib.hais_ops.functions import hais_ops


class mp3dHAISDataset:
    def __init__(self, config=ConfigHAIS()):
        self.data_root = config.data_root
        self.dataset = config.dataset
        self.filename_suffix = config.filename_suffix

        self.batch_size = config.batch_size
        self.train_workers = config.train_workers
        self.val_workers = config.train_workers

        self.full_scale = config.full_scale
        self.scale = config.scale
        self.max_npoint = config.max_npoint
        self.mode = config.mode

        # internal variables
        self.batch_id = 0

        # if test:
        #     self.test_split = config.split  # val or test
        #     self.test_workers = config.test_workers
        #     config.batch_size = 1

    # Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype("float32") / 3
        blur1 = np.ones((1, 3, 1)).astype("float32") / 3
        blur2 = np.ones((1, 1, 3)).astype("float32") / 3

        bb = np.abs(x).max(0).astype(np.int32) // gran + 3
        noise = [
            np.random.randn(bb[0], bb[1], bb[2]).astype("float32")
            for _ in range(3)
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur0, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur1, mode="constant", cval=0)
            for n in noise
        ]
        noise = [
            scipy.ndimage.filters.convolve(n, blur2, mode="constant", cval=0)
            for n in noise
        ]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [
            scipy.interpolate.RegularGridInterpolator(
                ax, n, bounds_error=0, fill_value=0
            )
            for n in noise
        ]

        def g(x_):
            return np.hstack([i(x_)[:, None] for i in interp])

        return x + g(x) * mag

    def getInstanceInfo(self, xyz, instance_label):
        """
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        """
        instance_info = (
            np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0
        )  # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []  # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            # instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            # instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return (
            instance_num,
            {
                "instance_info": instance_info,
                "instance_pointnum": instance_pointnum,
            },
        )

    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [
                    [math.cos(theta), math.sin(theta), 0],
                    [-math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ],
            )  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz):
        """
        :param xyz: (n, 3) >= 0
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.max_npoint:
            offset = np.clip(
                full_scale - room_range + 0.001, None, 0
            ) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * (
                (xyz_offset < full_scale).sum(1) == 3
            )
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label

    # convert open3d pointclouds to tensors for HAIS model
    def process_data(self, o3d_pcl):

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


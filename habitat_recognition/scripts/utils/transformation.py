"""This is a collection of helper funtions for transformation."""

import copy

import numpy as np
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs


def o3d_rtab2mp3d(o3d_cloud):
    """Convert rtab coordinates to mp3d coordinates."""
    quat = quat_from_two_vectors(np.array([0, 1, 0]), np.array([0, 0, -1]))
    o3d_quat = np.roll(quat_to_coeffs(quat), 1)
    r_mat = o3d_cloud.get_rotation_matrix_from_quaternion(o3d_quat)
    o3d_cloud_r = copy.deepcopy(o3d_cloud)
    o3d_cloud_r.rotate(r_mat, center=(0, 0, 0))
    return o3d_cloud_r

import numpy as np
import scipy 

# credit to https://github.com/Improbable-AI/airobot/blob/master/src/airobot/sensor/camera/rgbdcam.py
def get_point_from_pixel(depth_im, rs, cs, img_height, img_width, cam_int_mat, 
                        cam_ext_mat=None, depth_scale=1, in_world=True, 
                        filter_depth=False, k=1, ktype='median', 
                        depth_min=0.0, depth_max=5.0):
    """
    Calculate the 3D position of pixels in the RGB image.

    Args:
        rs (int or list or np.ndarray): rows of interest.
            It can be a list or 1D numpy array
            which contains the row indices. The default value is None,
            which means all rows.
        cs (int or list or np.ndarray): columns of interest.
            It can be a list or 1D numpy array
            which contains the column indices. The default value is None,
            which means all columns.
        in_world (bool): if True, return the 3D position in
            the world frame,
            Otherwise, return the 3D position in the camera frame.
        filter_depth (bool): if True, only pixels with depth values
            between [depth_min, depth_max]
            will remain.
        k (int): kernel size. A kernel (slicing window) will be used
            to get the neighboring depth values of the pixels specified
            by rs and cs. And depending on the ktype, a corresponding
            method will be applied to use some statistical value
            (such as minimum, maximum, median, mean) of all the depth
            values in the slicing window as a more robust estimate of
            the depth value of the specified pixels.
        ktype (str): what kind of statistical value of all the depth
            values in the sliced kernel
            to use as a proxy of the depth value at specified pixels.
            It can be `median`, `min`, `max`, `mean`.
        depth_min (float): minimum depth value. If None, it will use the
            default minimum depth value defined in the config file.
        depth_max (float): maximum depth value. If None, it will use the
            default maximum depth value defined in the config file.
        cam_ext_mat (np.ndarray): camera extrinsic matrix (shape: :math:`[4,4]`).
            If provided, it will be used to compute the points in the world frame.

    Returns:
        np.ndarray: 3D point coordinates of the pixels in
        camera frame (shape: :math:`[N, 3]`).
    """
    if not isinstance(rs, int) and not isinstance(rs, list) and \
            not isinstance(rs, np.ndarray):
        raise TypeError('rs should be an int, a list or a numpy array')
    if not isinstance(cs, int) and not isinstance(cs, list) and \
            not isinstance(cs, np.ndarray):
        raise TypeError('cs should be an int, a list or a numpy array')
    if isinstance(rs, int):
        rs = [rs]
    if isinstance(cs, int):
        cs = [cs]
    if isinstance(rs, np.ndarray):
        rs = rs.flatten()
    if isinstance(cs, np.ndarray):
        cs = cs.flatten()
    if not (isinstance(k, int) and (k % 2) == 1):
        raise TypeError('k should be a positive odd integer.')
    if k == 1:
        depth_im = depth_im[rs, cs]
    else:
        depth_im_list = []
        if ktype == 'min':
            ktype_func = np.min
        elif ktype == 'max':
            ktype_func = np.max
        elif ktype == 'median':
            ktype_func = np.median
        elif ktype == 'mean':
            ktype_func = np.mean
        else:
            raise TypeError('Unsupported ktype:[%s]' % ktype)
        for r, c in zip(rs, cs):
            s = k // 2
            rmin = max(0, r - s)
            rmax = min(img_height, r + s + 1)
            cmin = max(0, c - s)
            cmax = min(img_width, c + s + 1)
            depth_im_list.append(ktype_func(depth_im[rmin:rmax,
                                            cmin:cmax]))
        depth_im = np.array(depth_im_list)

    depth = depth_im.reshape(-1) * depth_scale
    img_pixs = np.stack((rs, cs)).reshape(2, -1)
    img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
    # depth_min = depth_min if depth_min else self.depth_min
    # depth_max = depth_max if depth_max else self.depth_max
    if filter_depth:
        valid = depth > depth_min
        valid = np.logical_and(valid,
                                depth < depth_max)
        depth = depth[:, valid]
        img_pixs = img_pixs[:, valid]
    
    uv_one = np.concatenate((img_pixs,
                                np.ones((1, img_pixs.shape[1]))))
    cam_int_mat_inv = np.linalg.inv(cam_int_mat)
    uv_one_in_cam = np.dot(cam_int_mat_inv, uv_one)
    pts_in_cam = np.multiply(uv_one_in_cam, depth)
    if in_world:
        pts_in_cam = np.concatenate((pts_in_cam,
                                        np.ones((1, pts_in_cam.shape[1]))),
                                    axis=0)
        pts_in_world = np.dot(cam_ext_mat, pts_in_cam)
        pts_in_world = pts_in_world[:3, :].T
        return pts_in_world
    else:
        return pts_in_cam.T
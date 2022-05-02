import numpy as np 

import os 

import open3d as o3d 
from matplotlib import pyplot as plt 
from scipy.spatial.transform import Rotation as R
from habitat_sim.scene import SemanticScene, SemanticObject
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_coeffs
import quaternion as qt

from dataset.habitat.utils import read_house_file, getOBB

###################### grid map ###############################

def project_points_to_grid_xz(bounds, points, meters_per_pixel):
    '''
    @bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    @points: numpy array with shape (N, 3)
    return: points_grid, numpy array (N,2), dtype=int
    '''

    # convert 3D x,z to topdown x,y
    pxs = (points[:,0] - bounds[0][0]) / meters_per_pixel
    pys = (points[:,2] - bounds[0][2]) / meters_per_pixel
    points_grid = np.stack([pxs, pys]).astype(int).T
    return points_grid

def grid_xz_to_points(bounds, grids, height, meters_per_pixel):
    '''
    @bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z))
    @grids: numpy array with shape (N, 2)
    @height: fixed height (y-axis) of points
    return: points, numpy array (N,3)
    '''

    points_x = grids[:,0] * meters_per_pixel + bounds[0][0]
    points_z = grids[:,1] * meters_per_pixel + bounds[0][2]
    points_y = np.full_like(points_x, height)
    return np.stack([points_x, points_y, points_z], axis=1)

################ bounding box ######################################

# vertices order of box for visualization 
'''
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
'''

BBOX_EDGES = np.asarray([[0,1],[0,3],[0,4],
                        [2,1],[2,3],[2,6],
                        [5,1],[5,4],[5,6],
                        [7,3],[7,4],[7,6]])


def rotate_in_xy(pt, theta):
# pt: (3,) numpy vector (x,y,z)
# theta: in radius
    r_mat = np.array([[np.cos(theta),  np.sin(theta),  0],
                    [-np.sin(theta),   np.cos(theta),  0],
                    [0,     0,      1]])
    return r_mat @ pt

def rotate(pt, r_mat):
# pt: (3,) numpy vector (x,y,z)
# mat: rotation mat
    return r_mat @ pt

# def get_corners(center, sizes, angle=0, quat=np.array([1,0,0,0])):
    
#     l, w, h = sizes
#     # TODO: implement rotation with quaternion

#     # calculate 8 bbox vertices in 3D space
#     vt_top1 = center + rotate_in_xy(np.array([l/2, w/2, h/2]), angle) # (+x,+y,+z)
#     vt_top2 = center + rotate_in_xy(np.array([-l/2, w/2, h/2]), angle) # (-x,+y,+z)
#     vt_top3 = center + rotate_in_xy(np.array([-l/2, -w/2, h/2]), angle) # (-x,-y,+z)
#     vt_top4 = center + rotate_in_xy(np.array([l/2, -w/2, h/2]), angle) # (+x,-y,+z)
#     vt_bot1 = center + rotate_in_xy(np.array([l/2, w/2, -h/2]), angle) # (+x,+y,-z)
#     vt_bot2 = center + rotate_in_xy(np.array([-l/2, w/2, -h/2]), angle) # (-x,+y,-z)
#     vt_bot3 = center + rotate_in_xy(np.array([-l/2, -w/2, -h/2]), angle) # (-x,-y,-z)
#     vt_bot4 = center + rotate_in_xy(np.array([l/2, -w/2, -h/2]), angle) # (+x,-y,-z)
    
#     corners = np.stack([vt_top1, vt_top2, vt_top3, vt_top4, 
#         vt_bot1, vt_bot2, vt_bot3, vt_bot4]) # shape 8x3

#     return corners

def get_corners(center, size, quat=np.array([0,0,0,1])):
    
    l, w, h = size

    # calculate 8 bbox vertices in 3D space
    local_corners = np.array([
        [l/2, w/2, h/2],    # (+x,+y,+z)
        [-l/2, w/2, h/2],   # (-x,+y,+z)
        [-l/2, -w/2, h/2],  # (-x,-y,+z)
        [l/2, -w/2, h/2],   # (+x,-y,+z)
        [l/2, w/2, -h/2],   # (+x,+y,-z)
        [-l/2, w/2, -h/2],  # (-x,+y,-z)
        [-l/2, -w/2, -h/2], # (-x,-y,-z)
        [l/2, -w/2, -h/2]   # (+x,-y,-z)
    ]) 
    r_mat = R.from_quat(quat).as_matrix()
    rotated_corners = (local_corners @ r_mat.T)
    world_corners = rotated_corners + center

    return world_corners
            
def visualize_scene_graph(scene_graph, 
    house_file_vis=False, house_file_path=None,
    pcl_mode="rgb", pcl_color_map="gist_rainbow", 
    bbox_vis=True, bbox_color=(0, 1, 0),
    free_space_vis=True, free_space_down_shift=0.5, free_space_color_map="viridis",
    save_file=None, use_o3d_bbox=False, mp3d_coord=True):
    
    o3d_vis_list = []
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    o3d_vis_list.append(axis_pcd)
    # NOTE: Option 1, render scene as a whole 
    points, pcl_seg = scene_graph.get_full_point_clouds()
    scene_o3d = o3d.geometry.PointCloud()
    scene_o3d.points = o3d.utility.Vector3dVector(points[:, :3])

    # transformation: from habitat (-y) to mp3d (-z)
    quat_alignGravity = quat_from_two_vectors(np.array([0, 0, -1]), np.array([0, -1, 0]))
    r_m2h = qt.as_rotation_matrix(quat_alignGravity)
    r_h2m = np.linalg.inv(r_m2h)

    ############### Visualize point clouds ####################
    if pcl_mode == "rgb":
        scene_o3d.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    elif pcl_mode == "segment":
        # color point clouds by its instance seg. id 
        labels = np.unique(pcl_seg)
        max_label = labels.max()
        labels[labels < 0] = max_label + 1 # -1 indicates non-labeled points 
        # print(f"scene {scene_ply.split('/')[-2]} has {max_label + 1} objects")
        colors = plt.get_cmap(pcl_color_map)(pcl_seg / (max_label+1))
        scene_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d_vis_list.append(scene_o3d)

    ############# Visualize .house file #########################
    if house_file_vis and house_file_path:
        house_dict, meta_dict = read_house_file(house_file_path)
        for region_dict in house_dict['R']:
            position = np.array([float(region_dict['px']), float(region_dict['py']), float(region_dict['pz'])])
            min_bound = np.array([float(region_dict['xlo']), float(region_dict['ylo']),float(region_dict['zlo'])])
            max_bound = np.array([float(region_dict['xhi']), float(region_dict['yhi']),float(region_dict['zhi'])])

            center = np.array([
                (float(region_dict['xhi']) + float(region_dict['xlo']))/2.,
                (float(region_dict['yhi']) + float(region_dict['ylo']))/2.,
                (float(region_dict['zhi']) + float(region_dict['zlo']))/2.,
            ])
            sizes = np.array([
                float(region_dict['xhi']) - float(region_dict['xlo']),
                float(region_dict['yhi']) - float(region_dict['ylo']),
                float(region_dict['zhi']) - float(region_dict['zlo']),
            ])
            region_idx = int(region_dict['region_index'])
            region_aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            region_aabb.color = (1,0,0)
            o3d_vis_list.append(region_aabb)
        
        for object_dict in house_dict['O']:
            center, sizes, r_mat = getOBB(object_dict)
            object_aabb = o3d.geometry.OrientedBoundingBox(center, r_mat, sizes)
            object_aabb.color = (1,0,1)
            o3d_vis_list.append(object_aabb)

    ############## visualize gt bounding boxes ##############
    if bbox_vis:
        if use_o3d_bbox:
            for obj_id in scene_graph.object_layer.obj_ids:
                obj_node = scene_graph.object_layer.obj_dict[obj_id]
                try:
                    o3d_pcl = o3d.geometry.PointCloud()
                    o3d_pcl.points = o3d.utility.Vector3dVector(obj_node.vertices)
                    obb = o3d_pcl.get_oriented_bounding_box()
                    if mp3d_coord:
                        obb.rotate(r_h2m)

                    obb.color = bbox_color
                    o3d_vis_list.append(obb)
                except:
                    print(f"WARNING: object with id{obj_id}, class name {obj_node.class_name} does not have point clouds")
        else:
            for obj_id in scene_graph.object_layer.obj_ids:
                obj_node = scene_graph.object_layer.obj_dict[obj_id]
                r_mat = R.from_quat(obj_node.rotation).as_matrix()
                obb = o3d.geometry.OrientedBoundingBox(obj_node.center, r_mat, obj_node.size)
                if mp3d_coord:
                    obb.rotate(r_h2m, (0,0,0))

                obb.color = bbox_color
                o3d_vis_list.append(obb)

    ############### visualize gt free space segmentation ###############
    if free_space_vis:
        
        segment_grid = scene_graph.region_layer.segment_grid
        # convert grid to index array and label array 
        idx_list = [[i,j] for i in range(segment_grid.shape[0]-1)
            for j in range(segment_grid.shape[1]-1)]
        seg_arr = np.array([segment_grid[idx[0], idx[1]] for idx in idx_list])
        idx_arr = np.array(idx_list)

        # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map 
        # need to swap the column in idx_arr
        vis_free_space_points = grid_xz_to_points(scene_graph.scene_bounds, idx_arr[:,[1,0]], 
            scene_graph.height - free_space_down_shift, scene_graph.meters_per_grid)
        
        free_space_o3d = o3d.geometry.PointCloud()
        free_space_o3d.points = o3d.utility.Vector3dVector(vis_free_space_points)
        max_label = seg_arr.max()
        colors = plt.get_cmap(pcl_color_map)( seg_arr/ (max_label+1))
        free_space_o3d.colors = o3d.utility.Vector3dVector(colors[:,:3])
        if mp3d_coord:
            free_space_o3d.rotate(r_h2m, (0,0,0))

        o3d_vis_list.append(free_space_o3d)

    o3d.visualization.draw_geometries(o3d_vis_list)    
    # o3d.visualization.draw_geometries(o3d_vis_list,
    #                                 zoom=0.455,
    #                                 front=[0.4999, 0.1659, 0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    
    # if save_file:
    #     o3d.io.write_point_cloud(save_file, scene_o3d, write_ascii=True)

def visualize_point_clouds(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] == 6:
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    o3d.visualization.draw_geometries([pcd])

def display_image_arr(sizes, img_arr, block=True):
    row, col = sizes
    f, axarr = plt.subplots(row, col)
    if row*col > 1:
        for i in range(row):
            for j in range(col):
                pos = axarr[i, j].imshow(img_arr[i*col+j])
                f.colorbar(pos, ax=axarr[i, j])
    else:
        pos = axarr.imshow(img_arr[0])
        f.colorbar(pos, ax=axarr)
    plt.show(block=block)

def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed on the entire row.
    """
    arr = np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


def inNd(a, b, assume_unique=False):
    a = asvoid(a)
    b = asvoid(b)
    return np.in1d(a, b, assume_unique)

if __name__ == "__main__":
    # test get_corners()
    center = np.array([0,0,0])
    size = np.array([1,1,1])
    quat_1 = np.array([0,0,0,1])
    print(get_corners(center, size, quat_1))
    quat_2 = np.array([1,0,0,1])
    print(get_corners(center, size, quat_2))
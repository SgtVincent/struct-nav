from math import dist
import numpy as np
import skimage
from matplotlib import docstring
from skimage.measure import find_contours
from numpy import ma 
import skfmm
from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# import plotly
# import plotly.express as px
from envs.utils.depth_utils import get_point_cloud_from_Y, get_camera_matrix

DEBUG_VIS = False
if DEBUG_VIS:
    from matplotlib import cm
    from matplotlib import pyplot as plt
        

class UnionFind:
    """Union-find data structure. Items must be hashable."""

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, obj):
        """X[item] will return the token object of the set which contains `item`"""

        # check for previously unknown object
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            return obj

        # find path of objects leading to the root
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def union(self, obj1, obj2):
        """Merges sets containing obj1 and obj2."""
        roots = [self[obj1], self[obj2]]
        heavier = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heavier:
                self.weights[heavier] += self.weights[r]
                self.parents[r] = heavier


# TODO: compare the performance & efficiency with scikit-DBSCAN
def group_frontier_grid(grids, distance=1):

    U = UnionFind()

    for (i, x) in enumerate(grids):
        for j in range(i + 1, len(grids)):
            y = grids[j]
            if max(abs(x[0] - y[0]), abs(x[1] - y[1])) <= distance:
                U.union(x, y)

    disjSets = {}
    for x in grids:
        s = disjSets.get(U[x], set())
        s.add(x)
        disjSets[U[x]] = s

    return [list(x) for x in disjSets.values()]


def get_frontiers(
    map_raw, map_origin, map_resolution, cluster_trashhole, dilate_size=3
):

    # save current occupancy grid for reliable computations
    saved_map = np.copy(map_raw)

    # compute contours
    contours_negative = find_contours(saved_map, -1.0, fully_connected="high")
    contours_positive = find_contours(saved_map, 1.0, fully_connected="high")
    contours_negative = np.concatenate(contours_negative, axis=0).astype(int)
    contours_positive = np.concatenate(contours_positive, axis=0).astype(int)

    ################## Improvement Trial 2 ##########################
    # Idea: dilate positive frontiers since frontiers should be away from obstacles
    # Result:

    contours_negative_map = np.zeros_like(saved_map)
    contours_positive_map = np.zeros_like(saved_map)

    contours_negative_map[
        contours_negative[:, 0], contours_negative[:, 1]
    ] = 1.0
    contours_positive_map[
        contours_positive[:, 0], contours_positive[:, 1]
    ] = 1.0

    # dilate contours_positive_map, which is boundary of obstacles
    selem = skimage.morphology.disk(dilate_size)
    contours_positive_dilated = skimage.morphology.binary_dilation(
        contours_positive_map, selem
    )
    contours_negative_map[contours_positive_dilated == 1] = 0
    candidates = np.argwhere(contours_negative_map == 1)
    # NOTE: Do not forget to convert (row, col) to (x, y) !!!!
    candidates = candidates[:, [1, 0]]
    # NOTE: change candidates to grid arrays instead of real world coords 
    # # translate contours to map frame
    # candidates = candidates * map_resolution + map_origin
    # convert set of frotiers into a list (hasable type data structre)
    candidates = candidates.tolist()
    candidates = [tuple(x) for x in candidates]

    if DEBUG_VIS:
        # from matplotlib import pyplot as plt
        # from matplotlib import cm

        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

        map_vis = np.copy(saved_map) + 1
        map_vis[saved_map >= 100] = 2.0  # obstacle
        im = plt.imshow(map_vis)
        # contours_candidates_vis = (
        #     np.array(candidates) - map_origin
        # ) / map_resolution
        contours_candidates_vis = np.array(candidates)

        plt.scatter(
            contours_candidates_vis[:, 0],
            contours_candidates_vis[:, 1],
            s=1,
            color="red",
        )
        # Invert y-axis
        ax = plt.gca()
        ax.invert_yaxis()

        plt.show()

    # group candidates points into clusters based on distance
    candidates = group_frontier_grid(candidates, cluster_trashhole / map_resolution)

    # make list of np arrays of clustered frontier points
    frontiers = []
    for i in range(len(candidates)):
        frontiers.append(np.array(candidates[i]))

    # return frontiers as list of np arays
    return frontiers


def compute_centroids(list_of_arrays, map_resolution):

    centroids = []
    for index, arr in enumerate(list_of_arrays):
        # compute num of elements in the array
        length = list_of_arrays[index].shape[0]
        # compute real world length of frontier in cm
        real_length = np.round(length * map_resolution * 100)
        # compute x coordanate of centroid
        sum_x = np.sum(list_of_arrays[index][:, 0])
        x = np.round(sum_x / length, 2)
        # compute y coodenate of centroid
        sum_y = np.sum(list_of_arrays[index][:, 1])
        y = np.round(sum_y / length, 2)
        # append coordanate in the form [x, y, real_length]
        centroids.append([x, y, real_length])

    # convert list of centroids into np array
    centroids = np.array(centroids)
    # return centroids as np array
    return centroids


def compute_goals(centroids, current_position, num_goals, dist_type="geo_dist", 
    map_raw=None, map_origin=None, map_resolution=None):

    # chosen utility function : length / distance
    # pre allocate utility_array
    utility_array = np.zeros((centroids.shape[0], centroids.shape[1]))

    # make a copy of centroids and use for loop to
    # substitute length atribute with utility of point
    utility_array = np.copy(centroids)

    if dist_type == "geo_dist":
        # compute traversible map  
        traversible = map_raw < 1.0
        traversible_ma = ma.masked_values(traversible * 1, 0)
        goal_map = np.zeros_like(traversible)
        goal_map[int(current_position[1]), int(current_position[0])] = 1
        selem = skimage.morphology.disk(3)
        goal_map = skimage.morphology.binary_dilation(goal_map, selem)
        traversible_ma[(goal_map == 1)&(traversible_ma.mask == False)] = 0
        fmm_dist = skfmm.distance(traversible_ma, dx=1)
        fmm_dist = ma.filled(fmm_dist, np.finfo('float').max)


    for index, c in enumerate(centroids):

        if dist_type == "man_dist": # compute manhattan distance
            
            dist = abs(current_position[0] - centroids[index][0]) + abs(
                current_position[1] - centroids[index][1]
            )
        elif dist_type == "geo_dist": # compute geodesic distance 
            
            dist = fmm_dist[int(centroids[index][1]), 
                int(centroids[index][0])]

        # compute length / distance
        # utility = centroids[index][2] ** 2 / dist
        # utility = centroids[index][2] / dist
        utility = np.sqrt(centroids[index][2]) / dist

        # substitute length attribute with utility of point
        utility_array[index][2] = utility

    # sort utility_array based on utility
    index = np.argsort(utility_array[:, 2])
    utility_array[:] = utility_array[index]

    # reverse utility_array to have greatest utility as index 0
    utility_array = utility_array[::-1]

    goals = []
    num_goals = min(num_goals, utility_array.shape[0])
    for i in range(num_goals):
        coordanate = []

        if i < len(utility_array):
            coordanate = [utility_array[i][0], utility_array[i][1]]
            goals.append(coordanate)

    # return goal as np array
    return np.array(goals)


def frontier_goals(
    map_raw,
    map_origin,
    map_resolution,
    current_position,
    cluster_trashhole=0.2,
    num_goals=3,
):
    """general function to calculate frontiers and goals


    Args:
        map_raw (_type_): (M,N) 2D int map, -1 for unknown, 0 for free space, 100 for obstacle
        map_origin (_type_): world frame coordinates [x,y] of map origin [0,0]
        map_resolution (_type_): real-world space size for one pixel
        current_position (_type_): current robot position [x,y] in world frame
        cluster_trashhole (float, optional):  param for frontier clustering
        num_goals (int): number of desired goals selected from frontiers

    Returns:
        centroids (numpy.Array): (C, 3), each row is (x, y, size)
        goals (numpy.Array): (num_goals, 2) positions of goals

    """
    ###################### Improvement Trial 1 #########################
    # NOTE: small free space grids near obstacle & unknown grids lead to undesired
    # frontier grids, which leads to bad frontier center, first filter out those
    # small area by dilation of obstacle map
    # Result: not working well
    # selem = skimage.morphology.disk(3)  # make sure be consistent with planner
    # occupancy_map = (map_raw == 100).astype(int)
    # map_dilated = np.copy(map_raw)
    # occupancy_dilated = skimage.morphology.binary_dilation(occupancy_map, selem)
    # map_dilated[occupancy_dilated == 1] = 100.0

    frontiers = get_frontiers(
        map_raw, map_origin, map_resolution, cluster_trashhole
    )
    centroids = compute_centroids(frontiers, map_resolution)
    current_position_grid = np.round(
        (current_position - map_origin) / map_resolution
    ).astype(int)
    goals_grid = compute_goals(
        centroids, 
        current_position_grid, 
        num_goals,
        dist_type="geo_dist",
        map_raw=map_raw, 
        map_origin=map_origin, 
        map_resolution=map_resolution
    )
    goals = goals_grid * map_resolution + map_origin

    # set flag to true in debug console dynamically for visualization
    if DEBUG_VIS:
        # from matplotlib import pyplot as plt
        # from matplotlib import cm

        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

        map_vis = np.copy(map_raw) + 1
        map_vis[map_vis > 100] = 2.0  # obstacle
        im = plt.imshow(map_vis)

        # frontiers_vis = [
        #     (np.array(f) - map_origin) / map_resolution for f in frontiers
        # ]
        frontiers_vis = frontiers
        colormap = cm.get_cmap("plasma")
        num_f = len(frontiers_vis)
        for i, f in enumerate(frontiers_vis):
            plt.scatter(f[:, 0], f[:, 1], color=colormap(float(i) / num_f))
        # visualize centroids
        # centroids_vis = (centroids[:, :2] - map_origin) / map_resolution
        centroids_vis = centroids[:, :2]
        sizes = centroids[:, 2] * 2
        N = centroids_vis.shape[0]
        colors = colormap(np.arange(0, N) / float(N))
        plt.scatter(
            x=centroids_vis[:, 0],
            y=centroids_vis[:, 1],
            s=sizes,
            c=colors,
            alpha=0.6,
        )
        # visualize goals
        goals_vis = (np.array(goals) - map_origin) / map_resolution
        n_goals = len(goals)
        alphas = np.linspace(0, 1, num=(n_goals + 2))
        for i, g in enumerate(goals_vis):
            plt.plot(
                g[0],
                g[1],
                color="red",
                marker="X",
                markersize=20,
                alpha=alphas[i + 1],
            )

        # Invert y-axis
        ax = plt.gca()
        ax.invert_yaxis()

        plt.show()
        plt.pause(0.001)
        input("press enter to continue")
        # plt.waitforbuttonpress(20)

    return centroids, goals

def dist_odom_to_goal(odom_mat, goal, dist_2d=True):
    
    if dist_2d: # ignore z-axis distance
        odom_pos = np.copy(odom_mat[:2,3])
    else: 
        odom_pos = np.copy(odom_mat[:3,3])

    return np.linalg.norm(odom_pos - goal)


def update_odom_by_action(odom_mat, action, forward_dist=0.25, turn_angle=30.):
    
    new_odom_mat = np.copy(odom_mat)
    
    if action == 0: # Stop
        pass
    
    elif action == 1: # Forward
        # NOTE: in rtabmap coords frame, front is +y direction 
        new_pos = odom_mat @ np.array([0., forward_dist, 0., 1.])
        new_odom_mat = np.copy(odom_mat)
        new_odom_mat[:3, 3] = new_pos[:3]

    elif action == 2: # Left, xy-plane counter-clockwise 
        rad = np.deg2rad(turn_angle)
        turn_mat = np.array([
            [np.cos(rad), -np.sin(rad), 0.],
            [np.sin(rad), np.cos(rad), 0.],
            [0.,        0.,     1.]
        ])
        new_rot = odom_mat[:3, :3] @ turn_mat
        new_odom_mat[:3, :3] = new_rot

    elif action == 3: # Right, xy-plane clockwise
        rad = np.deg2rad(turn_angle)
        turn_mat = np.array([
            [np.cos(rad), np.sin(rad), 0.],
            [-np.sin(rad), np.cos(rad), 0.],
            [0.,        0.,     1.]
        ])
        new_rot = odom_mat[:3, :3] @ turn_mat
        new_odom_mat[:3, :3] = new_rot

    return new_odom_mat

def target_goals(
    sem_img,
    depth_img,
    goal_idx,
    cam_param,
    odom_pose_mat,
    sensor_height=0.88,
    num_goals=1,
):
    mask = sem_img == goal_idx
    labels, num_targets = skimage.measure.label(mask, return_num=True, connectivity=2)
    # unproject segmentation mask to point clouds 
    pts_cam = get_point_cloud_from_Y(
        depth_img,
        camera_matrix=cam_param,
    ).squeeze()

    target_list = []
    target_sizes = []
    dists_to_odom = []
    for label in range(1, num_targets+1):
        rs, cs = np.where(labels == label)
        target_pts_cam = pts_cam[rs, cs]
        # sensor height does not affect xy-position
        target_pts_odom = target_pts_cam - np.array([0,0,sensor_height])
        target_pts_world = odom_pose_mat @ np.concatenate(
            (target_pts_odom, np.ones((target_pts_odom.shape[0],1))), axis=1
        ).T
        target_pts_world = (target_pts_world.T)[:, :3] 
        target = np.mean(target_pts_world, axis=0)
        target_list.append(target)
        target_sizes.append(target_pts_world.shape[0])
        dists_to_odom.append(dist_odom_to_goal(odom_pose_mat, target[:2]))

    targets = np.stack(target_list, axis=0)
    target_sizes = np.array(target_sizes)
    sort_idx = np.argsort(dists_to_odom)
    goals = targets[sort_idx[:num_goals], :2]
    return targets, target_sizes, goals

# def cluster_from_points(pts, method="dbscan", **kwargs):
#     if method == "dbscan":
#         # use DBSCAN to cluster detected target points 
#         dbscan_eps=kwargs.get('dbscan_eps', 0.1)
#         db = DBSCAN(eps=dbscan_eps, min_samples=10).fit(pts)
#         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#         core_samples_mask[db.core_sample_indices_] = True
#         labels = db.labels_
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#         n_noise = list(labels).count(-1)

#     else: 
#         raise NotImplementedError
    
#     return 

if __name__ == "__main__":

    # test update_odom_by_action function 
    import open3d as o3d 
    import copy 
    
    world_mat = np.eye(4)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=np.array([0., 0., 0.]))
    # rotate to left by 30 degrees
    left_rot_mat = update_odom_by_action(world_mat, 2)
    left_rot_frame = copy.deepcopy(world_frame).rotate(
        left_rot_mat[:3,:3], center=(0, 0, 0))
    # move forward by 2
    forward_mat = update_odom_by_action(left_rot_mat, 1, forward_dist=2.)
    forward_frame = copy.deepcopy(left_rot_frame).translate(
        forward_mat[:3, 3], relative=False)

    o3d.visualization.draw_geometries([world_frame, left_rot_frame, forward_frame])
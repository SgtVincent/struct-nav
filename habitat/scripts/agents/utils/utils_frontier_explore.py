from matplotlib import docstring
import numpy as np
from skimage.measure import find_contours

# import matplotlib.pyplot as plt
# import plotly
# import plotly.express as px


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


def get_frontiers(map_raw, map_origin, map_resolution, cluster_trashhole):

    # save current occupancy grid for reliable computations
    saved_map = np.copy(map_raw)

    # compute contours
    contours_negative = find_contours(saved_map, -1.0, fully_connected="high")
    contours_positive = find_contours(saved_map, 1.0, fully_connected="high")

    # wait until Ocuupancy grid output features
    # temporalUpdate = 30s (max wait time) / see launch file
    # if len(contours_negative) == 0 or len(contours_positive) == 0:
    #     rospy.sleep(10)
    #     controller()

    # translate contours to map frame
    contours_negative = np.concatenate(contours_negative, axis=0)
    for index in range(len(contours_negative)):
        contours_negative[index][0] = round(
            contours_negative[index][0] * map_resolution + map_origin[0], 2
        )
        contours_negative[index][1] = round(
            contours_negative[index][1] * map_resolution + map_origin[1], 2
        )

    # translate contours to map frame
    contours_positive = np.concatenate(contours_positive, axis=0)
    for index in range(len(contours_positive)):
        contours_positive[index][0] = round(
            contours_positive[index][0] * map_resolution + map_origin[0], 2
        )
        contours_positive[index][1] = round(
            contours_positive[index][1] * map_resolution + map_origin[1], 2
        )

    # convert contour np arrays into sets
    set_negative = set([tuple(x) for x in contours_negative])
    set_positive = set([tuple(x) for x in contours_positive])

    # perform set difference operation to find candidates
    candidates = set_negative.difference(set_positive)

    # convert set of frotiers into a list (hasable type data structre)
    candidates = [x for x in candidates]

    # group candidates points into clusters based on distance
    candidates = group_frontier_grid(candidates, cluster_trashhole)

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


def compute_goals(centroids, current_position, num_goals=3):

    # chosen utility function : length / distance

    # pre allocate utility_array
    utility_array = np.zeros((centroids.shape[0], centroids.shape[1]))

    # make a copy of centroids and use for loop to
    # substitute length atribute with utility of point
    utility_array = np.copy(centroids)

    # compute current position on the map
    # current_position, current_quaternion = get_current_pose('/map', '/odom')

    for index, c in enumerate(centroids):

        # compute manhattan distance
        man_dist = abs(current_position[0] - centroids[index][0]) + abs(
            current_position[1] - centroids[index][1]
        )

        # compute length / distance
        utility = centroids[index][2] / man_dist

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
        map_raw (_type_): (M,N) 2D int map, -1 for unknown, 0 for free space 
        map_origin (_type_): world frame coordinates [x,y] of map origin [0,0]
        map_resolution (_type_): real-world space size for one pixel 
        current_position (_type_): current robot position [x,y] in world frame 
        cluster_trashhole (float, optional):  param for frontier clustering 
        num_goals (int): number of desired goals selected from frontiers 
        
    Returns:
        centroids (numpy.Array): (C, 3), each row is (x, y, size) 
        goals (numpy.Array): (num_goals, 2) positions of goals 

    """

    frontiers = get_frontiers(map_raw, map_origin, map_resolution, cluster_trashhole)
    centroids = compute_centroids(frontiers, map_resolution)
    goals = compute_goals(centroids, current_position, num_goals)

    return centroids, goals


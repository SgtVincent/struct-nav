import matplotlib.pyplot as plt
from habitat.utils.visualizations import maps
from habitat_sim import PathFinder, Simulator
import magnum as mn
import math
import numpy as np
import skimage
from matplotlib import docstring
from skimage.measure import find_contours

# import matplotlib.pyplot as plt
# import plotly
# import plotly.express as px

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
    # translate contours to map frame
    candidates = candidates * map_resolution + map_origin
    # convert set of frotiers into a list (hasable type data structre)
    candidates = candidates.tolist()
    candidates = [tuple(x) for x in candidates]

    # convert contour np arrays into sets
    # set_negative = set([tuple(x) for x in contours_negative])
    # set_positive = set([tuple(x) for x in contours_positive])

    # perform set difference operation to find candidates
    # candidates = set_negative.difference(set_positive)

    # translate contours to map frame
    # for index in range(len(contours_negative)):
    #     contours_negative[index][0] = round(
    #         contours_negative[index][0] * map_resolution + map_origin[0], 2
    #     )
    #     contours_negative[index][1] = round(
    #         contours_negative[index][1] * map_resolution + map_origin[1], 2
    #     )

    # translate contours to map frame
    # for index in range(len(contours_positive)):
    #     contours_positive[index][0] = round(
    #         contours_positive[index][0] * map_resolution + map_origin[0], 2
    #     )
    #     contours_positive[index][1] = round(
    #         contours_positive[index][1] * map_resolution + map_origin[1], 2
    #     )

    # convert set of frotiers into a list (hasable type data structre)
    # candidates = [x for x in candidates]

    if DEBUG_VIS:
        # from matplotlib import pyplot as plt
        # from matplotlib import cm

        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

        map_vis = np.copy(saved_map) + 1
        map_vis[saved_map >= 100] = 2.0  # obstacle
        im = plt.imshow(map_vis)
        contours_candidates_vis = (
            np.array(candidates) - map_origin
        ) / map_resolution
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
        utility = centroids[index][2] ** 2 / man_dist  # why squared?

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


def compute_geo_utility(centroids, current_position):

    # chosen utility function : length / distance for geometry utility

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
        utility = centroids[index][2] ** 2 / man_dist  # why squared?

        # substitute length attribute with utility of point
        utility_array[index][2] = utility

    # sort utility_array based on utility
    index = np.argsort(utility_array[:, 2])
    utility_array[:] = utility_array[index]

    # reverse utility_array to have greatest utility as index 0
    utility_array = utility_array[::-1]

    return utility_array


def compute_sem_utility(centroids, scene_graph=None, goal_cat=None, sim=None):

    # NOTE: can use gt geodesic distance to object as oracle

    # utility function for semantic utility

    # pre allocate utility_array
    utility_array = np.zeros((centroids.shape[0], centroids.shape[1]))

    # make a copy of centroids and use for loop to
    # substitute length atribute with utility of point
    utility_array = np.copy(centroids)

    # TODO: utility propagation

    # NOTE: Oracle
    # TODO: Stop
    points = np.copy(centroids)
    # TODO: Rtabmap coordinate to habitat coordinate
    # points[:, 2] = 0.3  # TODO: height
    # points_hab = np.copy(points)
    # points_hab[:, 0] = points[:, 1]
    # points_hab[:, 1] = points[:, 0]
    utility = 1 / dist2obj_goal(sim, points_hab, goal_cat)
    utility_array[:, 2] = utility

    return utility_array


def dist2obj_goal(sim, points, goal_cat, verbose=True, display=False):
    # find distance to object goal with oracle
    import habitat_sim

    # find centers of goal category from habitat simulator
    ends = {}
    semantic_scene = sim.semantic_scene
    for region in semantic_scene.regions:
        # load object layer from habitat simulator
        for obj in region.objects:
            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()},"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            # )
            object_id = int(obj.id.split("_")[-1])  # counting from 0
            center = obj.obb.center
            # rot_quat = obj.obb.rotation[1, 2, 3, 0]
            cate = obj.category.name()
            if cate == goal_cat:
                end = center
                end_exact = sim.pathfinder.is_navigable(end)
                if not end_exact:
                    end = sim.pathfinder.snap_point(end)
                if verbose:
                    snap_info = (
                        "" if end_exact else f", not navigable, snapped to {end}"
                    )
                    print(f"end point {center}", snap_info)
                ends[object_id] = end
    print("found {} object in goal cate".format(len(ends.values())))

    point2goal_dists = []
    for p in points:
        start = p
        start_exact = sim.pathfinder.is_navigable(start)
        if not start_exact:
            start = sim.pathfinder.snap_point(start)
        if verbose:
            snap_info = (
                ""
                if start_exact
                else f", not navigable, snapped to {start}"
            )
            print(f"start point {p}", snap_info)

        # @markdown 2. Use ShortestPath module to compute path between samples.
        geod_dists = []
        for end in ends.values():
            path = habitat_sim.ShortestPath()
            path.requested_start = start
            path.requested_end = end
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points = path.points
            geod_dists.append(geodesic_distance)

            # DEBUG info
            if verbose:
                print("found_path : " + str(found_path))
                print("geodesic_distance : " + str(geodesic_distance))
                print("path_points : " + str(path_points))

            if display and found_path:
                display_path(sim, path_points, plt_block=True)

        shortest_goal_dist = sorted(geod_dists)[0]
        point2goal_dists.append(shortest_goal_dist)

    return np.array(point2goal_dists)

    # if display and found_path:
    #     display_path(sim, path_points, plt_block=True)


def combine_utilities(geo_utility_array, sem_utility_array):
    # TODO:
    # NOTE: Can be tuned manually or learned by RL
    utility_array = np.copy(geo_utility_array)
    print(utility_array[:, 2])
    # utility_array[:, 2] += sem_utility_array[:, 2]
    utility_array[:, 2] = sem_utility_array[:, 2]
    print(utility_array[:, 2])

    return utility_array


def get_goals(utility_array, num_goals=3):

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
    sim=None,
    mode='geo+sem'
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
    # NOTE: using learning to propose additional centroids?

    if mode == 'geo':
        goals = compute_goals(centroids, current_position, num_goals)
    elif mode == 'geo+sem':
        # NOTE: Combine pure geometry-based method with semantic method
        geo_utility_array = compute_geo_utility(centroids, current_position)
        # print("geo utility", geo_utility_array)
        sem_utility_array = compute_sem_utility(
            centroids, scene_graph=None, goal_cat="shower", sim=sim)
        # print("sem utility", sem_utility_array)
        utility_array = combine_utilities(geo_utility_array, sem_utility_array)
        # print("final utility", utility_array)
        goals = get_goals(utility_array, num_goals)

    # set flag to true in debug console dynamically for visualization
    if DEBUG_VIS:
        # from matplotlib import pyplot as plt
        # from matplotlib import cm

        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True

        map_vis = np.copy(map_raw) + 1
        map_vis[map_vis > 100] = 2.0  # obstacle
        im = plt.imshow(map_vis)

        frontiers_vis = [
            (np.array(f) - map_origin) / map_resolution for f in frontiers
        ]
        colormap = cm.get_cmap("plasma")
        num_f = len(frontiers_vis)
        for i, f in enumerate(frontiers_vis):
            plt.scatter(f[:, 0], f[:, 1], color=colormap(float(i) / num_f))
        # visualize centroids
        centroids_vis = (centroids[:, :2] - map_origin) / map_resolution
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

    return centroids, goals


# TODO: Reuse from sg_nav

# display a topdown map with matplotlib

def display_map(topdown_map, key_points=None, block=False):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=block)

# display the path on the 2D topdown map
# @path_points: list of (3,) positions in habitat coords frame


def display_path(
    sim: Simulator, path_points: list, meters_per_pixel=0.025, plt_block=False
):

    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    height = scene_bb.y().min
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
    # convert world trajectory points to maps module grid points
    trajectory = [
        maps.to_grid(
            path_point[2],
            path_point[0],
            grid_dimensions,
            pathfinder=sim.pathfinder,
        )
        for path_point in path_points
    ]
    grid_tangent = mn.Vector2(
        trajectory[1][1] - trajectory[0][1],
        trajectory[1][0] - trajectory[0][0],
    )
    path_initial_tangent = grid_tangent / grid_tangent.length()
    initial_angle = math.atan2(
        path_initial_tangent[0], path_initial_tangent[1]
    )
    # draw the agent and trajectory on the map
    maps.draw_path(top_down_map, trajectory)
    maps.draw_agent(
        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
    )
    print("\nDisplay the map with agent and path overlay:")
    display_map(top_down_map, block=plt_block)

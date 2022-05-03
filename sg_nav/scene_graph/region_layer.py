import copy

import numpy as np

# from utils.open3d_utils import
# local import
from scene_graph.object_layer import ObjectNode
from scene_graph.utils import project_points_to_grid_xz
from scipy.spatial import cKDTree


class RegionNode:
    def __init__(
        self,
        id,
        bbox,
        grid_map=None,
        grid_size=0.1,
        class_name=None,
        label=None,
    ):
        # required field: minimum attributes for a region: id, bounding box
        self.id = id
        self.bbox = bbox

        # optional field
        self.grid_map = grid_map
        self.grid_size = grid_size
        self.class_name = class_name
        self.label = label
        self.objects = []
        return

    def add_object(self, obj: ObjectNode):

        self.objects.append(obj)
        return


class RegionLayer:
    def __init__(self):

        self.flag_grid_map = False
        self.region_ids = []
        self.region_dict = {}
        return

    def __len__(self):
        return len(self.region_ids)

    # TODO: consider creating a standalone map class?
    def init_map(self, bounds, grid_size, free_space_grid, project_mode="xz"):

        self.bounds = (
            bounds  # the real map area corresponding to free space grid
        )
        self.grid_size = grid_size
        self.segment_grid = np.zeros_like(free_space_grid, dtype=int) - 1
        self.free_space_grid = np.array(free_space_grid).astype(bool)
        self.project_mode = (
            project_mode  # 'xz' means project xz axes in 3D to xy axes in map
        )

        # implement the data structure for closest grid point searching
        # self.grid_kd_tree = cKDTree(self.free_space_grid)
        self.flag_grid_map = True
        return

    def add_region(self, bbox, id=None, class_name=None, label=None):

        # add region node
        if id == None or id in self.region_ids:
            new_id = 0
            if len(self.region_ids) > 0:
                new_id = max(self.region_ids) + 1
            print(
                f"Warning: region id {id} already exists in region layer. Assign id {new_id} instead"
            )
            id = new_id
        self.region_ids.append(id)
        region_node = RegionNode(id, bbox)
        self.region_dict[id] = region_node

        # add segment on layer free space grid map
        if self.flag_grid_map:
            if self.project_mode == "xz":
                region_grid_map = self.segment_region_on_grid_map_xz(id, bbox)
            else:  # TODO: if there are other datasets ...
                raise NotImplementedError
            region_node.grid_size = self.grid_size
            region_node.grid_map = region_grid_map

        # add semantic info
        region_node.class_name = class_name
        region_node.label = label

        return region_node

    def segment_region_on_grid_map_xz(self, region_id, region_bbox):
        assert (
            self.flag_grid_map
        ), "called 'segment_region_on_grid_map_xz()' before grid map being initialized"
        # get region bbox on 2d grid map
        # region_2d_bbox: np.array: [[x_min, y_min],[x_max, y_max]]
        region_2d_bbox = project_points_to_grid_xz(
            self.bounds, region_bbox, self.grid_size
        )

        region_mask = np.zeros_like(self.free_space_grid).astype(bool)
        # NOTE: (row_idx, col_idx) corresponds to (y, x) in 2d grid map
        region_mask[
            int(np.ceil(region_2d_bbox[0][1])) : int(
                np.floor(region_2d_bbox[1][1])
            ),
            int(np.ceil(region_2d_bbox[0][0])) : int(
                np.floor(region_2d_bbox[1][0])
            ),
        ] = True
        # color the region on global grid map
        region_mask = np.logical_and(region_mask, self.free_space_grid)
        self.segment_grid[region_mask] = region_id

        return region_mask

    def create_topological_map(self):
        # fp = ndimage.generate_binary_structure(3,1)
        #     dilated_label_map = ndimage.morphology.grey_dilation(label_map, footprint=fp)
        pass

    def get_region_by_idx(self, idx):
        idx = np.argwhere(self.region_ids == idx)[0][0]
        return self.region_dict[idx]

    # def query_points_regions(self, points):

    #     dists, indices = self.grid_kd_tree.query(points)
    #     return self.grid_labels[indices]

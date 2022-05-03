import os

import geometry_msgs.msg
import numpy as np
import open3d as o3d
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from habitat_sim.scene import SemanticObject, SemanticScene
from matplotlib import cm
from std_msgs.msg import ColorRGBA, Header, Int32
from visualization_msgs.msg import Marker, MarkerArray

# local import
# from utils.box_util import get_3d_box

# vertices order of box for visualization
"""
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
"""

VIS_EDGES = np.asarray(
    [
        [0, 1],
        [0, 3],
        [0, 4],
        [2, 1],
        [2, 3],
        [2, 6],
        [5, 1],
        [5, 4],
        [5, 6],
        [7, 3],
        [7, 4],
        [7, 6],
    ]
)


def rotate_in_xy(pt, theta):
    # pt: (3,) numpy vector (x,y,z)
    # theta: in radius
    r_mat = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return r_mat @ pt


def rotate(pt, r_mat):
    # pt: (3,) numpy vector (x,y,z)
    # mat: rotation mat
    return r_mat @ pt


def get_corners(center, sizes, angle=0, quat=np.array([1, 0, 0, 0])):

    l, w, h = sizes
    # TODO: implement rotation with quaternion

    # calculate 8 bbox vertices in 3D space
    vt_top1 = center + rotate_in_xy(
        np.array([l / 2, w / 2, h / 2]), angle
    )  # (+x,+y,+z)
    vt_top2 = center + rotate_in_xy(
        np.array([-l / 2, w / 2, h / 2]), angle
    )  # (-x,+y,+z)
    vt_top3 = center + rotate_in_xy(
        np.array([-l / 2, -w / 2, h / 2]), angle
    )  # (-x,-y,+z)
    vt_top4 = center + rotate_in_xy(
        np.array([l / 2, -w / 2, h / 2]), angle
    )  # (+x,-y,+z)
    vt_bot1 = center + rotate_in_xy(
        np.array([l / 2, w / 2, -h / 2]), angle
    )  # (+x,+y,-z)
    vt_bot2 = center + rotate_in_xy(
        np.array([-l / 2, w / 2, -h / 2]), angle
    )  # (-x,+y,-z)
    vt_bot3 = center + rotate_in_xy(
        np.array([-l / 2, -w / 2, -h / 2]), angle
    )  # (-x,-y,-z)
    vt_bot4 = center + rotate_in_xy(
        np.array([l / 2, -w / 2, -h / 2]), angle
    )  # (+x,-y,-z)

    corners = np.stack(
        [
            vt_top1,
            vt_top2,
            vt_top3,
            vt_top4,
            vt_bot1,
            vt_bot2,
            vt_bot3,
            vt_bot4,
        ]
    )  # shape 8x3

    return corners


def semantic_obj_to_marker(
    obj: SemanticObject,
    id,
    color=[0.0, 1.0, 0.0, 0.5],
    scale=[0.1, 0, 0],
    frame_id="map",
    aligned_bbox=True,
):
    # create marker for one bounding box
    marker = Marker(
        type=Marker.LINE_LIST,
        id=id,
        ns=obj.category.name(),
        lifetime=rospy.Duration(100),
        # pose=Pose(Point(*point_elavated), Quaternion(0,0,0,1)),
        scale=Vector3(*scale),
        header=Header(frame_id=frame_id),
        color=ColorRGBA(*color),
        # text=text,
    )
    # create corners
    if aligned_bbox:
        corners = get_corners(obj.aabb.center, obj.aabb.sizes)
    else:
        # TODO: implement bbox with rotations
        raise NotImplementedError

    # LINE_LIST marker displays line between points 0-1, 1-2, 2-3, 3-4, 4-5
    for edge in VIS_EDGES:
        pt_1 = corners[edge[0]]
        pt_2 = corners[edge[1]]
        marker.points.append(Point(*pt_1))
        marker.points.append(Point(*pt_2))

    return marker


def semantic_scene_to_markerarray(
    scene: SemanticScene, frame_id="map", vis_name=True
):

    marker_arr = MarkerArray()
    # name_marker_arr = MarkerArray()
    id = 0
    for obj in scene.objects:
        marker = semantic_obj_to_marker(obj, id=id)
        marker_arr.markers.append(marker)
        # TODO: add name text topic
        # if vis_name:
        #     text_pos = centroid + np.array([0.0,0.0,self.object_elavate_text])
        #     name_marker = Marker(
        #         type=Marker.TEXT_VIEW_FACING,
        #         id=id,
        #         ns=ns,
        #         # lifetime=rospy.Duration(2),
        #         pose=Pose(Point(*text_pos), Quaternion(0,0,0,1)),
        #         scale=Vector3(0.2, 0.2, 0.2),
        #         header=Header(frame_id=submap_id),
        #         color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
        #         text=submap.class_name)

        # name_marker_arr.markers.append(name_marker)
        id += 1
    return marker_arr  # , name_marker_arr


def create_inst_pcl(o3d_pcl, result, color_map="plasma"):
    """
    o3d_pcl: open3d PointCloud object
    result["inst_mask": (nCluster, Npoints)
    result["inst_conf"]: (nCluster)
    result["label_id"]: (nCluster)
    """

    color_map = cm.get_cmap(color_map)
    num_inst = result["label_id"].shape[0]
    colors = np.zeros_like(o3d_pcl.colors)  # [0,1]

    for inst_id, sem_id in enumerate(result["label_id"]):
        color = color_map(float(inst_id) / num_inst)[:3]
        inst_mask = result["inst_mask"][inst_id, :]
        colors[inst_mask, :] = color

    o3d_pcl.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pcl

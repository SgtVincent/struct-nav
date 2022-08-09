import rospy
import time
import quaternion as qt 
import numpy as np 
from std_srvs.srv import Empty, EmptyRequest
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Point, Pose, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap


def safe_call_reset_service(service_name, timeout=5.):

    
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        rospy.ServiceProxy(service_name, Empty)()
        return True 
    except rospy.ServiceException as e:
        rospy.logwarn(f"Service call failed: {e}")
        return False

def safe_get_map_service(service_name="/rtabmap/get_map", timeout=5.):
    
    success = False
    while(not success):
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
            response = rospy.ServiceProxy(service_name, GetMap)()
            success = True
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service {service_name} call failed: {e}")
            time.sleep(0.5)
    return response.map 

def publish_frontiers(frontiers, goals, publisher, 
    f_c=(0., 1.0, 0., 0.5), g_c=(1.0, 0., 0., 0.5),
    lifetime=2.0):

    marker_arr = MarkerArray()
    # color_map = cm.get_cmap("plasma")
    # publish goals
    for i, g in enumerate(goals):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.lifetime = rospy.Duration(lifetime)
        marker.id = i  # avoid overwrite
        marker.type = marker.SPHERE
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        # color: (r,g,b,a) 4-tuple
        marker.color = ColorRGBA(*g_c)
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = g[0]
        marker.pose.position.y = g[1]
        marker.pose.position.z = 0.5

        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        marker_arr.markers.append(marker)

    # publish frontiers
    try:
        num_goals = goals.shape[0]
    except:
        num_goals = len(goals)
    
    frontiers_mean_scale = np.mean([f[2] for f in frontiers])
    
    for i, f in enumerate(frontiers):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.lifetime = rospy.Duration(lifetime)
        marker.id = i + num_goals
        marker.type = marker.SPHERE
        marker.scale.x = 0.5 * f[2] / frontiers_mean_scale
        marker.scale.y = 0.5 * f[2] / frontiers_mean_scale
        marker.scale.z = 0.5 * f[2] / frontiers_mean_scale
        marker.color = ColorRGBA(*f_c)
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = f[0]
        marker.pose.position.y = f[1]
        marker.pose.position.z = 0.5

        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        marker_arr.markers.append(marker)

    publisher.publish(marker_arr)
    return 



def publish_targets(targets, publisher, lifetime=2.0):

    marker_arr = MarkerArray()
    # color_map = cm.get_cmap("plasma")
    # publish goals
    for i, g in enumerate(targets):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.id = i  # avoid overwrite
        marker.lifetime = rospy.Duration(lifetime)
        marker.type = marker.SPHERE
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = g[0]
        marker.pose.position.y = g[1]
        marker.pose.position.z = 0.5

        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        marker_arr.markers.append(marker)

    publisher.publish(marker_arr)
    return 

def publish_odom(pose_mat, publisher, stamp=None, frame_id="odom", 
    child_frame_id="base_link"):
    """publish odometry message from pose matrix 

    Now assume each odometry message has unified covariance
    Could try covariance propagation later 

    Args:
        pose_mat (np.ndarray): (4x4) pose matrix 
        publisher (rospy.Publisher): odometry message publisher
    """
    msg = Odometry()
    if stamp is None:
        msg.header.stamp = rospy.Time.now()
    else: 
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id # i.e. '/odom'
    msg.child_frame_id = child_frame_id # i.e. '/base_footprint'
    
    # construct position and rotation
    x, y, z = pose_mat[:3,3]
    quat = qt.from_rotation_matrix(pose_mat[:3, :3])
    msg.pose.pose.position = Point(x, y, z)
    msg.pose.pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)

    # create pseudo diagnal covariance matrix
    # robot on x-y plane, more noise in x, y, z-axis rot
    p_cov = np.array([
        1e-3, 0., 0., 0., 0., 0.,
        0., 1e-3, 0., 0., 0., 0.,
        0., 0., 1e-6, 0., 0., 0.,
        0., 0., 0., 1e-6, 0., 0.,
        0., 0., 0., 0., 1e-6, 0.,
        0., 0., 0., 0., 0., 1e-3,
    ])
    msg.pose.covariance = p_cov.tolist()

    # NOTE: if to implement continous control, add psedu twist message
    
    # Publish odometry message
    publisher.publish(msg)

def publish_pose(pose_mat, publisher, stamp=None, frame_id="odom"):

    msg = PoseStamped()
    if stamp is None:
        msg.header.stamp = rospy.Time.now()
    else: 
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id # i.e. '/odom'
    
    # construct position and rotation
    x, y, z = pose_mat[:3,3]
    quat = qt.from_rotation_matrix(pose_mat[:3, :3])
    msg.pose.position = Point(x, y, z)
    msg.pose.orientation = Quaternion(quat.x, quat.y, quat.z, quat.w)

    # Publish odometry message
    publisher.publish(msg)
    
def publish_scene_graph(scene_graph, publisher, 
                        vis_mode="elavated", vis_name=True, name_publisher=None, 
                        rot=None, trans=None, frame_id="map", namespace="scene_graph",
                        lifetime=2.0):
    
    assert vis_mode in ["elavated", "inplace"]
    if rot is None:
        rot = np.eye(3)
    if trans is None:
        trans = np.array([0,0,0], dtype=float)
    box_elevate_dist = 1.0
    text_elevate_dist = 0.2
    
    marker_arr = MarkerArray()
    if vis_name:
        name_marker_arr = MarkerArray()
    for id, node in  scene_graph.object_layer.obj_dict.items():

        center =  rot @ node.center + trans
        if vis_mode == "elavated":
            center = center + np.array(
                [0.0, 0.0,  box_elevate_dist]
            )

        # if  map_rawlabel2panoptic_id[obj.label] == 0:
        #     ns = "structure"
        #     scale=Vector3(0.5, 0.5, 0.5)
        #     color = ColorRGBA(1.0, 0.5, 0.0, 0.5)
        # else: #  map_rawlabel2panoptic_id[obj.label] == 1
        scale = Vector3(0.2, 0.2, 0.2)
        color = ColorRGBA(0.0, 0.0, 1.0, 0.5)

        marker = Marker(
            type=Marker.CUBE,
            id=id,
            ns=namespace,
            lifetime=rospy.Duration(lifetime),
            pose=Pose(Point(*center), Quaternion(0, 0, 0, 0)),
            scale=scale,
            header=Header(frame_id=frame_id),
            color=color,
            # text=submap.class_name,
        )

        marker_arr.markers.append(marker)

        if vis_name:
            text_pos = center + np.array(
                [0.0, 0.0, text_elevate_dist]
            )
            name_marker = Marker(
                type=Marker.TEXT_VIEW_FACING,
                id=id,
                ns=namespace,
                lifetime=rospy.Duration(lifetime),
                pose=Pose(Point(*text_pos), Quaternion(0, 0, 0, 0)),
                scale=Vector3(0.2, 0.2, 0.2),
                header=Header(frame_id=frame_id),
                color=ColorRGBA(0.0, 0.0, 0.0, 0.8),
                text=node.class_name,
            )

            name_marker_arr.markers.append(name_marker)

    publisher.publish(marker_arr)
    
    if vis_name:
        name_publisher.publish(name_marker_arr)

def delete_all_markers(publisher, ns):
    marker_array_msg = MarkerArray()
    marker = Marker()
    marker.id = 0
    marker.ns = ns
    marker.action = Marker.DELETEALL
    marker_array_msg.markers.append(marker)
    publisher.publish(marker_array_msg)


def publish_target_name(publisher, target_name, agent_pos, 
                        text_up_dist=1.5, lifetime=2.0):
    text_pos = agent_pos + np.array(
        [0.0, 0.0, text_up_dist]
    )
    target_name_marker = Marker(
        type=Marker.TEXT_VIEW_FACING,
        id=0,
        ns="target",
        # lifetime=rospy.Duration(lifetime),
        pose=Pose(Point(*text_pos), Quaternion(0, 0, 0, 0)),
        scale=Vector3(0.4, 0.4, 0.4),
        header=Header(frame_id="map"),
        color=ColorRGBA(1.0, 0.5, 0.5, 0.8),
        text=target_name,
    )
    publisher.publish(target_name_marker)
    return 
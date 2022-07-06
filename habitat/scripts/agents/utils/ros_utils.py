import rospy
import time
import quaternion as qt 
import numpy as np 
from std_srvs.srv import Empty, EmptyRequest
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, PoseStamped
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
    f_c=(0., 1.0, 0., 0.5), g_c=(1.0, 0., 0., 0.5)):

    marker_arr = MarkerArray()
    # color_map = cm.get_cmap("plasma")
    # publish goals
    for i, g in enumerate(goals):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
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



def publish_targets(targets, publisher):

    marker_arr = MarkerArray()
    # color_map = cm.get_cmap("plasma")
    # publish goals
    for i, g in enumerate(targets):

        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "map"
        marker.id = i  # avoid overwrite
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
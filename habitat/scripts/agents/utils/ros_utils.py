import rospy
from std_srvs.srv import Empty, EmptyRequest
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np 


def safe_call_reset_service(service_name, timeout=5.):

    
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
        rospy.ServiceProxy(service_name, Empty)()
        return True 
    except rospy.ServiceException as e:
        rospy.logwarn(f"Service call failed: {e}")
        return False


def publish_frontiers(frontiers, goals, publisher):

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
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5
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
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = f[0]
        marker.pose.position.y = f[1]
        marker.pose.position.z = 0.5

        # We add the new marker to the MarkerArray, removing the oldest marker from it when necessary
        marker_arr.markers.append(marker)
        
    publisher.publish(marker_arr)
    return 

def publish_object_goal(goals, publisher):

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

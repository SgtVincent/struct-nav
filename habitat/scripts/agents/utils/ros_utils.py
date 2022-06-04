import rospy
from std_srvs.srv import Empty


def safe_call_reset_service(service_name, timeout=60):

    rospy.wait_for_service(service_name, timeout=timeout)
    try:
        _ = rospy.ServiceProxy("service_name", Empty)
        return True
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return False

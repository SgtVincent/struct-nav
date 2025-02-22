#! /usr/bin/env python
"""Dectect stuck and loss odom."""

import numpy as np
import rospy
import tf
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32

DEFAULT_STUCK_TIME = 1
DEFAULT_ODOM_THRESHOLD = 0.25

rospy.init_node("stuck_detector")

# tf_listener = tf.TransformListener()

commands = []
odometry = []

stuck_time = rospy.get_param("~stuck_time", DEFAULT_STUCK_TIME)
odom_threshold = rospy.get_param("~odometry_threshold", DEFAULT_ODOM_THRESHOLD)
alarm_publisher = rospy.Publisher(
    "stuck_alarm", Bool, latch=True, queue_size=100
)


def get_robot_pose(tf_listener, odom_pose):
    """Get the robot pose."""
    pos, quat = tf_listener.lookupTransform(
        "map", "odom", tf_listener.getLatestCommonTime("map", "odom")
    )

    _, _, tf_angle = tf.transformations.euler_from_quaternion(quat)
    _, _, odom_angle = tf.transformations.euler_from_quaternion(
        [
            odom_pose.orientation.x,
            odom_pose.orientation.y,
            odom_pose.orientation.z,
            odom_pose.orientation.w,
        ]
    )
    current_x, current_y = odom_pose.position.x, odom_pose.position.y
    current_x_new = current_x * np.cos(-tf_angle) + current_y * np.sin(
        -tf_angle
    )
    current_y_new = -current_x * np.sin(-tf_angle) + current_y * np.cos(
        -tf_angle
    )
    current_x_new += pos[0]
    current_y_new += pos[1]
    return current_x_new, current_y_new, odom_angle + tf_angle


class CMDCallback:
    """A cmd callback helper."""

    def __init__(self):
        """Initialize cmd call back."""
        self.fwd_start_time = 0

    def cmd_callback(self, msg):
        """CMD callback."""
        if len(odometry) == 0:
            print("NO ODOMETRY!!!")
            return
        cur_time = rospy.Time.now().to_sec()
        if self.fwd_start_time == 0:
            self.fwd_start_time = cur_time
        commands.append([msg.data, cur_time])
        if msg.data == 1:
            if cur_time - self.fwd_start_time > stuck_time:
                i = len(odometry) - 1
                while i > 0 and odometry[i][-1] > cur_time - stuck_time:
                    i -= 1
                dst = np.sqrt(
                    (odometry[-1][0] - odometry[i][0]) ** 2
                    + (odometry[-1][1] - odometry[i][1]) ** 2
                )
                if len(odometry) - i > 2 and dst < odom_threshold:
                    print("YOU STUCK!!! DRAW OBSTACLE AHEAD!!!")
                    alarm_publisher.publish(True)
                    return
        else:
            self.fwd_start_time = rospy.Time.now().to_sec()
        alarm_publisher.publish(False)


def odom_callback(msg):
    """Odom callback."""
    odom_pose = msg.pose.pose
    stamp = msg.header.stamp.to_sec()
    tf_listener = tf.TransformListener()
    x, y, angle = get_robot_pose(tf_listener, odom_pose)
    if x is not None:
        odometry.append([x, y, angle, stamp])


cmd_callback = CMDCallback()
cmd_subscriber = rospy.Subscriber("habitat_action", Int32, cmd_callback)
odom_subscriber = rospy.Subscriber("odom", Odometry, odom_callback)

rospy.spin()

"""Habitat agent."""

import random
import rospy
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class RandomWalkAgent:
    """A random walk agent."""

    def __init__(self, action_names):
        """Initialize an agent with a random action space."""
        self.action_names = action_names

    def get_action(self):
        """Get a random action."""
        action = random.choice(self.action_names)
        return action

    def act(self):
        """Perform an action."""
        action = self.get_action()
        return action


class SpinningAgent:
    """A spinning agent."""

    def __init__(self, action_name):
        """Initialize an agent with a spinning direction."""
        assert action_name in {"turn_right", "turn_left"}
        self.action_name = action_name

    def get_action(self):
        """Get as rotational action."""
        action = self.action_name
        return action

    def act(self):
        """Perform an action."""
        action = self.get_action()
        return action


class FrontierExploreAgent:
    def __init__(
        self, odom_topic, local_plan_topic, global_plan_topic,
    ):

        # state from ros messages
        self.odom = None
        self.local_plan = None
        self.global_plan = None
        self.actions = ["stay", "move_forward", "turn_left", "turn_right"]
        # subscribe to a topic using rospy.Subscriber class
        self.sub_odom = rospy.Subscriber(odom_topic, Odometry, self.callback_odom)
        print(f"subscribing to {odom_topic}...")
        self.sub_local_plan = rospy.Subscriber(
            local_plan_topic, Path, self.callback_local_plan
        )
        print(f"subscribing to {local_plan_topic}...")
        self.sub_global_plan = rospy.Subscriber(
            global_plan_topic, Path, self.callback_global_plan
        )
        print(f"subscribing to {global_plan_topic}...")

        # publish messages to a topic using rospy.Publisher class
        self.pub_action = rospy.Publisher("habitat_action", String, queue_size=1)

    def local_policy(self, start: PoseStamped, stg: PoseStamped, turn_angle=30.0):
        """function to generate action in habitat simulator  

        Args:
            start (geometry_msgs.msg): start pose 
            stg (geometry_msgs.msg): short term goal pose
            turn_angle (float, optional): rotate angle by one rotate action. Defaults to 30.0.

        Returns:
            action: action label to execute in habitat simulator 
        """
        start_pos = np.array(
            [start.pose.position.x, start.pose.position.y, start.pose.position.z]
        )

        start_rot = R.from_quat(
            [
                start.pose.orientation.x,
                start.pose.orientation.y,
                start.pose.orientation.z,
                start.pose.orientation.w,
            ]
        )

        start_angle = start_rot.as_euler("zxy", degrees=True)[0]

        stg_pos = np.array(
            [stg.pose.position.x, stg.pose.position.y, stg.pose.position.z]
        )

        angle_st_goal = math.degrees(
            math.atan2(stg_pos[0] - start_pos[0], stg_pos[1] - start_pos[1])
        )
        angle_agent = (start_angle) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > turn_angle / 2.0:
            action = 3  # turn_right
        elif relative_angle < -turn_angle / 2.0:
            action = 2  # turn_left
        else:
            action = 1  # move_forward

        return action

    # TODO: add check collision function

    def callback_local_plan(self, local_plan: Path):
        self.local_plan = local_plan

    def callback_global_plan(self, global_plan: Path):
        self.global_plan = global_plan

    def callback_odom(self, odom: Odometry):
        self.odom = odom

    def get_action(self):
        # select the last pose as the short-term goal
        # if self.local_plan and self.odom:
        #     stg: PoseStamped = self.local_plan.poses[-1]
        #     start: PoseStamped = self.odom.pose
        #     actions = {1: "Forward", 2:"Left", 3:"Right"}
        #     action = self.local_policy(start, stg)
        #     action_name = actions[action]
        #     self.pub_action.publish(action_name)
        #     return action_name
        if self.global_plan and self.odom:
            stg: PoseStamped = self.global_plan.poses[-1]
            start: PoseStamped = self.odom.pose
            actions = {1: "Forward", 2: "Left", 3: "Right"}
            action = self.local_policy(start, stg)
            action_name = actions[action]
            self.pub_action.publish(action_name)
            return action_name
        else:
            return "stay"

    def act(self):
        """Perform an action."""
        action = self.get_action()
        return action


AGENT_CLASS_MAPPING = {
    "random_walk": RandomWalkAgent,
    "spinning": SpinningAgent,
    "frontier_explore": FrontierExploreAgent,
}

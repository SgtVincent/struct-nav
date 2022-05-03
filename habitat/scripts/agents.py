"""Habitat agent."""

import random


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

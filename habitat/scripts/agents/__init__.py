from .dummy_agents import RandomWalkAgent, SpinningAgent
from .frontier_explore_agent import FrontierExploreAgent

AGENT_CLASS_MAPPING = {
    "random_walk": RandomWalkAgent,
    "spinning": SpinningAgent,
    "frontier_explore": FrontierExploreAgent,
}

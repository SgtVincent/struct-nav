from .dummy_agents import RandomWalkAgent, SpinningAgent
from .frontier_explore_agent import FrontierExploreAgent
from .frontier_2d_detect_agent import Frontier2DDetectionAgent

AGENT_CLASS_MAPPING = {
    "random_walk": RandomWalkAgent,
    "spinning": SpinningAgent,
    "frontier_explore": FrontierExploreAgent,
    "frontier_2d_detection": Frontier2DDetectionAgent,
}

import abc
from abc import ABC, abstractmethod, abstractproperty
import os 
import numpy as np

from envs.constants import coco_categories, coco_label_mapping

MAX_DIST = 1e8

def utility_reverse_euc_dist_sum(prior_mat, class_labels, goal_label, min_dist=0.5):
    
    prior_dists = prior_mat[class_labels, :][:, goal_label]
    prior_dists[prior_dists > min_dist] = min_dist
    return np.sum(1.0 / prior_dists)

class PriorBase(ABC):

    @abstractmethod
    def compute_sem_utility(self):
        """Compute semantic utility for frontiers
        """
        pass
        
    
class MatrixPrior(PriorBase):
    
    def __init__(self, scene_prior_path, language_prior_path):
        super().__init__()
        
        self.scene_prior_matrix = None
        self.language_prior_matrix = None
        if os.path.exists(scene_prior_path):
            self.scene_prior_matrix = np.load(scene_prior_path)
        if os.path.exists(language_prior_path):
            self.language_prior_matrix = np.load(language_prior_path)
        
    def compute_sem_utility(self, frontiers, current_position, goal_name,
                            scene_graph, method="radius_sum", 
                            **kwargs):
        """Compute semantic utility by calculating the average distance to the target 
        of all objects within a circle around the frontier.

        Args:
            frontiers (np.ndarray): centroids of frontiers 
            current_position (np.ndarray): agent's current position 
            scene_graph (SceneGraphBase): scene graph class
            method (str): method to compute semantic utility 
            grid_map (np.ndarray): grid map 
        """
        # TODO: finish semantic utility computation function 
        sem_utility = []
        goal_label = coco_categories[goal_name]
        
        if method == "radius_sum":
            radius = kwargs.get("radius", 2.0) 
            # for each frontier, get all objects in the ball centering the frontier 
            neighbor_objects_list = scene_graph.sample_graph(
                method="radius_sampling",
                center=frontiers,
                radius=radius,
                **kwargs
            )
            for i, frontier in enumerate(frontiers): 
                neighbor_obj_ids = neighbor_objects_list[i]
                if len(neighbor_obj_ids) == 0:
                    sem_utility.append(0) # no semantic utility value 
                else:
                    class_names = scene_graph.object_layer.get_class_names(neighbor_obj_ids)
                    class_labels = [coco_categories[name] for name in class_names
                                    if name in coco_categories]
                    scene_utility = utility_reverse_euc_dist_sum(
                        self.scene_prior_matrix, class_labels, goal_label)
                    # language_utility = utility_reverse_euc_dist_sum(
                    #     self.language_prior_matrix, class_labels, target_label
                    # )
                    sem_utility.append(scene_utility)
                    
        return np.array(sem_utility)
        
        
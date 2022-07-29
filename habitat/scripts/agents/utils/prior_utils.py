import abc
import os
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from envs.constants import coco_categories, coco_label_mapping
from sentence_transformers import SentenceTransformer, util

MAX_DIST = 1e8


def utility_reverse_euc_dist_sum(prior_mat,
                                 class_labels,
                                 goal_label,
                                 min_dist=0.5):

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

    def __init__(
        self,
        priors={'lang'},
        scene_prior_path='',
        scene_prior_matrix=None,
        lang_prior_path='',
        lang_prior_matrix=None,
        combine_weight=None,
    ):
        super().__init__()

        assert len(priors) > 0
        assert (priors | {'scene', 'lang'}) == {'scene', 'lang'}

        self.priors = priors
        self.scene_prior_matrix = scene_prior_matrix
        self.lang_prior_matrix = lang_prior_matrix
        if scene_prior_matrix is None and os.path.exists(scene_prior_path):
            self.scene_prior_matrix = np.load(scene_prior_path)
        if lang_prior_matrix is None and os.path.exists(lang_prior_path):
            self.lang_prior_matrix = np.load(lang_prior_path)
        self.combine_weight = combine_weight

        if 'lang' in self.priors:
            assert self.lang_prior_matrix is not None
        if 'scene' in self.priors:
            assert self.scene_prior_matrix is not None
        if self.priors == {'scene', 'lang'}:
            assert isinstance(self.combine_weight, float)
            assert self.combine_weight >= 0.0 and self.combine_weight <= 1.0

    def compute_sem_utility(self,
                            frontiers,
                            current_position,
                            goal_name,
                            scene_graph,
                            method="radius_sum",
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
                **kwargs)
            for i, frontier in enumerate(frontiers):
                neighbor_obj_ids = neighbor_objects_list[i]
                if len(neighbor_obj_ids) == 0:
                    sem_utility.append(0)  # no semantic utility value
                else:
                    class_names = scene_graph.object_layer.get_class_names(
                        neighbor_obj_ids)
                    class_labels = [
                        coco_categories[name] for name in class_names
                        if name in coco_categories
                    ]
                    if 'scene' in self.priors:
                        scene_utility = utility_reverse_euc_dist_sum(
                            self.scene_prior_matrix, class_labels, goal_label)
                    if 'lang' in self.priors:
                        lang_utility = utility_reverse_euc_dist_sum(
                            self.lang_prior_matrix, class_labels, goal_label)
                    utility = self.get_utility(scene_utility, lang_utility)
                    sem_utility.append(utility)

        return np.array(sem_utility)

    def get_utility(self, scene_utility, lang_utility):
        if self.priors == 'scene':
            return scene_utility
        if self.priors == 'lang':
            return lang_utility
        if self.priors == {'scene', 'lang'}:
            return self.combine_weight * scene_utility + (
                1 - self.combine_weight) * lang_utility


model_zoo = {
    'bert': 'sentence-transformers/bert-base-nli-mean-tokens',
    'clip': 'clip-ViT-L-14'
}


class LangPrior:

    def __init__(self, class_dict, mode='bert'):
        super().__init__()
        self.categories = [{
            'class_name': k,
            'class_label': v,
        } for k, v in class_dict.items()]

        self.mode = mode
        self.model = SentenceTransformer(model_zoo[mode])

        self.dist_matrix = self.create_matrix()

    def get_cos_dist(self, class_name1, class_name2):
        embedding_1 = self.model.encode(class_name1, convert_to_tensor=True)
        embedding_2 = self.model.encode(class_name2, convert_to_tensor=True)
        return (1 - util.pytorch_cos_sim(embedding_1, embedding_2))

    def create_matrix(self):
        num_class = len(self.categories)
        c2c_dist = np.zeros((num_class, num_class))
        for i in range(num_class):
            for j in range(i + 1, num_class):
                cat1, cat2 = self.categories[i], self.categories[j]
                dist = self.get_cos_dist(
                    cat1['class_name'],
                    cat2['class_name'],
                )
                c2c_dist[cat1['class_label'], cat2['class_label']] = dist
                c2c_dist[cat2['class_label'], cat1['class_label']] = dist

        return c2c_dist

    def load_matrix():
        pass


if __name__ == "__main__":
    sem_prior = MatrixPrior({'lang'},
                            lang_prior_matrix=LangPrior(
                                coco_categories, mode='bert').dist_matrix)
    print(sem_prior.lang_prior_matrix)
    print(sem_prior.lang_prior_matrix.shape)
    sem_prior = MatrixPrior({'lang'},
                            lang_prior_matrix=LangPrior(
                                coco_categories, mode='clip').dist_matrix)
    print(sem_prior.lang_prior_matrix)
    print(sem_prior.lang_prior_matrix.shape)

import abc
import os
from abc import ABC, abstractmethod, abstractproperty
from sys import flags

import numpy as np
from envs.constants import coco_categories, coco_label_mapping
from sentence_transformers import SentenceTransformer, util

MAX_DIST = 1e8


def utility_reverse_euc_dist(prior_mat, class_labels, goal_label, 
                             min_dist=0.5, mean=True):

    prior_dists = prior_mat[class_labels, :][:, goal_label]
    prior_dists[prior_dists > min_dist] = min_dist
    if mean:
        return np.average(1.0 / prior_dists)
    else: # sum
        return np.sum(1.0 / prior_dists)

def utility_reverse_euc_dist_discount_var(prior_mat, prior_var_mat, class_labels, 
                                          goal_label, min_dist=0.5, min_var=1.0, 
                                          order=-0.5, mean=True, weights=None):
    
    prior_dists = prior_mat[class_labels, :][:, goal_label]
    prior_vars = prior_var_mat[class_labels, :][:, goal_label]
    prior_dists[prior_dists < min_dist] = min_dist
    prior_vars[prior_vars < min_var] = min_var
    if weights is None:
        weights = np.ones(len(class_labels))
    weights = weights / np.float_power(prior_vars, order)
    if mean:
        return np.average((1.0 / prior_dists), weights=weights )
    else: # sum
        return np.sum((1.0 / prior_dists) * weights)

def utility_cos_sim(prior_mat, class_labels, goal_label, mean=True):
    
    prior_dists = prior_mat[class_labels, :][:, goal_label]
    if mean:
        return np.average(1 - prior_dists)
    else:
        return np.sum(1 - prior_dists)

def utility_cos_sim_discount_var(prior_mat, prior_var_mat, class_labels, 
                                goal_label, min_var=1.0, order=-0.5, 
                                mean=True, weights=None):
    
    prior_dists = prior_mat[class_labels, :][:, goal_label]
    prior_vars = prior_var_mat[class_labels, :][:, goal_label]
    prior_vars[prior_vars < min_var] = min_var
    if weights is None:
        weights = np.ones(len(class_labels))
    weights = weights / np.float_power(prior_vars, order)
    if mean:
        return np.average((1 - prior_dists), weights=weights)
    else:
        return np.sum((1 - prior_dists) * weights)

def hdist_std_weighted_diff_to_default(prior_mat, prior_var_mat, class_labels,
        goal_label, default_dist, min_var=1.0, order=-0.5, weights=None):
    
    prior_dists = prior_mat[class_labels, :][:, goal_label]
    prior_vars = prior_var_mat[class_labels, :][:, goal_label]
    prior_vars[prior_vars < min_var] = min_var
    diff = prior_dists - default_dist
    if weights is None:
        weights = np.ones(len(class_labels))
    weights = weights / np.float_power(prior_vars, order)
    hdist = np.average(diff, weights=weights) + default_dist
    return hdist

class PriorBase(ABC):

    @abstractmethod
    def compute_sem_utility(self):
        """Compute semantic utility for frontiers
        """
        pass

    @abstractmethod
    def compute_heuristic_dist(self):
        """Heuristic function to compute predicted distances from frontiers to target object
        """        
        pass
        
class MatrixPrior(PriorBase):

    def __init__(
        self,
        priors={'scene', 'lang'},
        scene_prior_path='',
        scene_prior_matrix=None,
        lang_prior_path='',
        lang_piror_type='bert_cos_dist',
        lang_prior_matrix=None,
        prior_var_matrix=None,
        combine_weight=None,
    ):
        super().__init__()

        assert len(priors) > 0
        assert (priors | {'scene', 'lang'}) == {'scene', 'lang'}

        self.priors = priors
        self.scene_prior_matrix = scene_prior_matrix
        self.lang_prior_matrix = lang_prior_matrix
        self.prior_var_matrix = prior_var_matrix
        
        if scene_prior_matrix is None and os.path.exists(scene_prior_path):
            scene_prior = dict(np.load(scene_prior_path))
            self.scene_prior_matrix = scene_prior['c2c_dist_mean']
            self.prior_var_matrix = scene_prior['c2c_dist_var']
            
        if lang_prior_matrix is None and os.path.exists(lang_prior_path):
            lang_prior = dict(np.load(lang_prior_path))
            self.lang_prior_matrix = lang_prior[lang_piror_type]
            
        self.combine_weight = combine_weight

        if 'lang' in self.priors:
            assert self.lang_prior_matrix is not None
        if 'scene' in self.priors:
            assert self.scene_prior_matrix is not None
        if self.priors == {'scene', 'lang'}:
            assert isinstance(self.combine_weight, float)
            assert self.combine_weight >= 0.0 and self.combine_weight <= 1.0

    def combine_utility(self, scene_utilities, lang_utilities):
    
        if self.priors == {'scene'}:
            scene_utilities = np.array(scene_utilities)
            if np.any(scene_utilities != 0):
                scene_utilities /= np.linalg.norm(scene_utilities)
            return scene_utilities
    
        elif self.priors == {'lang'}:
            lang_utilities = np.array(lang_utilities)
            if np.any(lang_utilities != 0):
                lang_utilities /= np.linalg.norm(lang_utilities)
            return lang_utilities
        
        elif self.priors == {'scene', 'lang'}:
            scene_utilities = np.array(scene_utilities)
            lang_utilities = np.array(lang_utilities)
            if np.sum(scene_utilities) > 0:
                scene_utilities /= np.sum(scene_utilities)
            if np.sum(lang_utilities) > 0:
                lang_utilities /= np.sum(lang_utilities)
            return self.combine_weight * scene_utilities + (
                1 - self.combine_weight) * lang_utilities
        else:
            raise NotImplementedError

    def compute_sem_utility(self,
                            frontiers,
                            current_position,
                            goal_name,
                            scene_graph,
                            method="radius_mean", # softr_mean
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
        
        scene_utilities = []
        lang_utilities = []
        goal_label = coco_categories[goal_name]
        graph_sample_method, obj_util_aggregate_method = method.split('_')
        
        if graph_sample_method == "radius":
            radius = kwargs.get("radius", 2.0)
            # for each frontier, get all objects in the ball centering the frontier
            neighbor_objects_list = scene_graph.sample_graph(
                method="radius_sampling",
                center=frontiers,
                radius=radius,
                **kwargs)
            neighbor_objects_weight_list = [
                np.ones(len(n_objs))
                for n_objs in neighbor_objects_list
            ]
        elif graph_sample_method == "softr":
            # soft radius sampling returns all objects along with their distance to frontiers
            neighbor_objects_list, neighbor_objects_dist_list = scene_graph.sample_graph(
                method="soft_radius_sampling",
                center=frontiers,
                **kwargs)
            neighbor_objects_weight_list = [
                1.0 / n_objs_dist
                for n_objs_dist in neighbor_objects_dist_list
            ]
        else: 
            raise NotImplementedError
        
        flag_mean = False
        if obj_util_aggregate_method == "mean":
            flag_mean = True
            
        for i, frontier in enumerate(frontiers):
            
            neighbor_obj_ids = neighbor_objects_list[i]
            obj_weights = neighbor_objects_weight_list[i]
            class_names = scene_graph.object_layer.get_class_names(
                    neighbor_obj_ids)
            class_labels = [
                coco_categories[name] for name in class_names
                if name in coco_categories
            ]
            
            if len(class_labels) == 0: # no valid object found in neighborhood
                scene_utilities.append(0)  
                lang_utilities.append(0)
            else:
                scene_utility = 0
                if 'scene' in self.priors:
                    scene_utility = utility_reverse_euc_dist_discount_var(
                        self.scene_prior_matrix, self.prior_var_matrix, 
                        class_labels, goal_label, mean=flag_mean, weights=obj_weights)
                    scene_utilities.append(scene_utility)
                    
                lang_utility = 0
                if 'lang' in self.priors:
                    lang_utility = utility_cos_sim_discount_var(
                        self.lang_prior_matrix, self.prior_var_matrix, 
                        class_labels, goal_label, mean=flag_mean, weights=obj_weights)
                    lang_utilities.append(lang_utility)
                    
        sem_utilities = self.combine_utility(scene_utilities, lang_utilities)

        return sem_utilities

    def compute_heuristic_dist(self,
                            frontiers,
                            goal_name,
                            scene_graph,
                            method="radius_mean",
                            default_dist=6.0, # mean of scene prior c2c matrix
                            **kwargs):
        
        hdists = []
        goal_label = coco_categories[goal_name]
        
        if method == "radius_mean":
            radius = kwargs.get("radius", 2.0)
            # for each frontier, get all objects in the ball centering the frontier
            neighbor_objects_list = scene_graph.sample_graph(
                method="radius_sampling",
                center=frontiers,
                radius=radius,
                **kwargs)
        else: 
            raise NotImplementedError
            
        for i, frontier in enumerate(frontiers):
            
            neighbor_obj_ids = neighbor_objects_list[i]
            class_names = scene_graph.object_layer.get_class_names(
                    neighbor_obj_ids)
            class_labels = [
                coco_categories[name] for name in class_names
                if name in coco_categories
            ]
            
            if len(class_labels) == 0: # no valid object found in neighborhood
                hdists.append(default_dist) 
            else:
                # objects in sampled subgraph should +/- estimated distance 
                # depending on the class labels and priors 
                if 'scene' in self.priors:
                    hdist = hdist_std_weighted_diff_to_default(
                        self.scene_prior_matrix, self.prior_var_matrix, 
                        class_labels, goal_label, default_dist)
                    hdists.append(hdist)
                    
                lang_utility = 0
                if 'lang' in self.priors:
                    # TODO: consider how semantic cos distance could help get
                    # heuristic distance to target object
                    raise NotImplementedError
                    
        
        return np.array(hdists)

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
    bert_sem_prior = MatrixPrior({'lang'},
                            lang_prior_matrix=LangPrior(
                                coco_categories, mode='bert').dist_matrix)
    # print(sem_prior.lang_prior_matrix)
    print("Generated bert language prior matrix of shape ", bert_sem_prior.lang_prior_matrix.shape)
    clip_sem_prior = MatrixPrior({'lang'},
                            lang_prior_matrix=LangPrior(
                                coco_categories, mode='clip').dist_matrix)
    # print(sem_prior.lang_prior_matrix)
    print("Generated clip language prior matrix of shape ", clip_sem_prior.lang_prior_matrix.shape)
    
    # save language prior matrices to npz file 
    np.savez("language_prior.npz",
             bert_cos_dist=bert_sem_prior.lang_prior_matrix,
             clip_cos_dist=clip_sem_prior.lang_prior_matrix)
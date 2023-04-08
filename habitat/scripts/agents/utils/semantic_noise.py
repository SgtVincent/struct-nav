"""class for noise model used on semantic segmentation images."""
from typing import List, Tuple
import numpy as np 


class SemanticNoiseModel:
    def __init__(self, args, object_labels: np.ndarray, seed=42):
        """Initializes the semantic noise model.
        
        @param args: arguments for the semantic noise model.
        @param object_labels: list of labels for the objects of interest. 0 reserved for background.
        
        """
        self.noise_models = args.sem_noise_model
        self.noise_model_rate = args.sem_noise_model_rate
        self.object_labels = object_labels
        
        self.rng = np.random.RandomState(seed=seed)
        
    def add_noise(self, semantic_image: np.ndarray) -> np.ndarray:
        """Applies semantic noise to the observations.
        """
        if "random_label_replace" in self.noise_models:
            semantic_image = self.random_label_replace(semantic_image) 
        
        if "random_label_drop" in self.noise_models:
            semantic_image = self.random_label_drop(semantic_image)

        return semantic_image
        
    def random_label_replace(self, image: np.ndarray)-> np.ndarray:
        """Randomly replaces labels of objects of interest with other object labels.
        """
        # generate random permutation of object labels for label replacing mapping 
        replace_mapping = self.rng.permutation(self.object_labels)
        # WARNING: make a copy of the semantic image to keep the original image for label indexing
        new_image = image.copy()
        
        # replace labels in image with another label with probability 
        for i in range(len(self.object_labels)):
            old_label = self.object_labels[i]
            new_label = replace_mapping[i]
            # if the label is in the image and the random number is less than the noise model rate
            if (old_label in image) and (self.rng.rand() < self.noise_model_rate):
                new_image[image == old_label] = new_label
        
        return new_image
            
    def random_label_drop(self, image: np.ndarray)-> np.ndarray:
        """Randomly drops labels of objects of interest, fill with background label 0.
        """
        # WARNING: make a copy of the semantic image to keep the original image for label indexing
        new_image = image.copy()
        
        for label in self.object_labels:
            if (label in image) and (self.rng.rand() < self.noise_model_rate):
                new_image[image == label] = 0
        
        return new_image
from os.path import join, basename
from scipy.integrate._ivp.radau import P
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch
import torch.nn.functional as F
import sys
import pickle
# local import 
from SSG.src.model_SGFN import SGFNModel
from SSG.src.op_utils import gen_descriptor
from SSG.src.config import Config
from scene_graph.object_layer import ObjectNode

# set config file path 
import pathlib
ROOT_PATH = pathlib.Path(__file__).parent.parent.absolute()
CONFIG_PATH = join(ROOT_PATH,"SSG", "config_CVPR21.json")

FLAG_DATA_DUMP = False

class SceneGraphPredictor:
    
    def __init__(self, rel_th=2.0, num_points=256, multiprocess=False):
        
        self.rel_th = rel_th
        self.num_points = num_points
        self.model_config = Config(CONFIG_PATH)
        self.multiprocess = multiprocess
        self.device = self.model_config.DEVICE
        self.model_type = "SGFN" # TODO: other models here! 
        self.init_model()
        
        # self.dump_num = 100
        # self.dump_file = "./scene_graph_dump.pkl"
        # self.dump_list = []
        # self.dumped = False


    def init_model(self):
        
        if self.model_type == "SGFN":
            self.model = SGFNModel(
                self.model_config,
                "CVPR21",
                20,  # pretrained model 
                9, # pretrained model 
            ).to(self.device)

            self.model.load()
            if self.multiprocess:
                self.model.share_memory()
        else:
            raise NotImplementedError
    
    # TODO: take into consideration the std, bbox, volume, etc. 
    def generate_edges(self, descriptors, mode='centroid'):
        
        if mode == 'centroid':
            centroids = descriptors[:, :3]
            n = centroids.shape[0]
            
            dist_mat = squareform(pdist(centroids))
            edges = np.argwhere((dist_mat > 0) & (dist_mat < self.rel_th))
        else:
            raise NotImplementedError

        return edges.astype(np.int64)

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0) # N, 3
        points -= centroid # n, 3, npts
        furthest_distance = points.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        points /= furthest_distance
        return points

    def data_process(self, submaps):
        
        num_submaps = len(submaps)
        sampled_points = []
        descriptors = torch.zeros(num_submaps, 11)
        for i, submap in enumerate(submaps):
            submap.updated = 0 # unset update flag

            pcls, colors, normals = submap.vertices, submap.colors, submap.normals
            #--------------- xyz ----------------# 
            # generate descriptors with full point clouds
            descriptors[i] = gen_descriptor(torch.from_numpy(pcls))
            # sample points uniformly 
            choice = np.random.choice(len(pcls), self.num_points, replace=len(pcls) < self.num_points)
            pcls = pcls[choice, :]
            pcls = self.norm_tensor(torch.from_numpy(pcls))

            #--------------- rgb ----------------#
            # preprocess of colors used for SGFN model 
            # convert [0-255] to (0,1.0)
            if issubclass(colors.dtype.type, np.integer):
                colors = colors/255.0
            colors = colors * 2 - 1.0
            colors = torch.from_numpy(colors[choice, :])

            #--------------- normals ----------------#
            normals = torch.from_numpy(normals[choice, :])

            # concatenate all info 
            points = torch.cat((pcls, colors, normals), dim=1)
            sampled_points.append(points)

        sampled_points = torch.stack(sampled_points, dim=0)
        edges = self.generate_edges(descriptors.numpy())
        edges = torch.from_numpy(edges)

        # if FLAG_DATA_DUMP:
        #     if len(self.dump_list) < self.dump_num and not self.dumped:
        #         saved_items = {
        #             "obj_points": sampled_points.numpy(),
        #             "edge_indices": edges.t().numpy(),
        #             "descriptor": descriptors.model())
        #         self.dumped = True
        #         print(f"Model input data dumped to file {self.dump_file}")


        # sample points 

        # sampled_points (num_obj, num_points, 3)
        # edges (num_edges, 2)
        # descriptors (num_obj, 11)
        return sampled_points.float().to(self.device), edges.to(self.device), descriptors.float().to(self.device)

    def predict(self, object_nodes, return_input=False, return_nparr=False):
        
        edges = [] 
        pred_rel_label = [] 
        pred_rel_prob = []

        if self.model_type == "SGFN":
            
            points, edges, descriptors = self.data_process(object_nodes)
            if len(edges) > 0:
                points = points.permute(0,2,1)
                pred_obj_cls, pred_rel_cls, = self.model(
                    points, edges.t().contiguous(), descriptors, return_meta_data=False)
                
                pred_obj_prob = torch.exp(pred_obj_cls)
                pred_obj_confidence, pred_obj_label = torch.max(pred_obj_prob, dim=1)

                pred_rel_prob = torch.exp(pred_rel_cls)
                pred_rel_confidence, pred_rel_label = torch.max(pred_rel_prob, dim=1)
                
                pred_obj_prob = pred_obj_prob.detach()
                pred_obj_confidence = pred_obj_confidence.detach()
                pred_obj_label = pred_obj_label.detach()
                pred_rel_prob = pred_rel_prob.detach()
                pred_rel_label = pred_rel_label.detach()
                pred_rel_confidence = pred_rel_confidence.detach()
                
                if return_nparr:
                    pred_obj_prob = pred_obj_prob.cpu().numpy()
                    pred_obj_confidence = pred_obj_confidence.cpu().numpy()
                    pred_obj_label = pred_obj_label.cpu().numpy()

                    pred_rel_prob = pred_rel_prob.cpu().numpy()
                    pred_rel_label = pred_rel_label.cpu().numpy()
                    pred_rel_confidence = pred_rel_confidence.cpu().numpy()
                    edges = edges.cpu().numpy()

        else:
            raise NotImplementedError

        if return_input:
            points = points.detach()
            descriptors = descriptors.detach()
            if return_nparr:
                points = points.cpu().numpy()
                descriptors = descriptors.cpu().numpy()
            results={
                "obj_points": points, 
                "descriptor": descriptors,
                "pred_obj_prob": pred_obj_prob,
                "pred_obj_confidence": pred_obj_confidence,
                "pred_obj_label": pred_obj_label,
                "edges": edges,
                "pred_rel_prob": pred_rel_prob,
                "pred_rel_confidence": pred_rel_confidence,
                "pred_rel_label": pred_rel_label
            }
        else: 
            results={
                "pred_obj_prob": pred_obj_prob,
                "pred_obj_confidence": pred_obj_confidence,
                "pred_obj_label": pred_obj_label,
                "edges": edges,
                "pred_rel_prob": pred_rel_prob,
                "pred_rel_confidence": pred_rel_confidence,
                "pred_rel_label": pred_rel_label
            }

        return results


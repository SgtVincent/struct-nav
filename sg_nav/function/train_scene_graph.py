import argparse
import os
import json
from tkinter import X
import numpy as np
from dataset.habitat.simulator import init_sim
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.loader import DataLoader
import logging

# local import
from scene_graph.scene_graph_cls import SceneGraphHabitat 
from scene_graph.config import SceneGraphHabitatConfig
from scene_graph.utils import visualize_scene_graph
from scene_graph.scene_graph_pred import SceneGraphPredictor
from scene_graph.data_sampler import DataSampler, PyGDatasetWrapper
from dataset.habitat.utils import mp3d_obj_nav_class_list as obj_nav_class_list
from models.deepset import Deepset
from models.deepgnn import DeeperGCN
from utils import cal_model_parms, set_seed, save_checkpoint, load_checkpoint
from utils.logger import setup_logger, generate_exp_directory, resume_exp_directory
from utils.training import build_optimizer, build_scheduler
from utils.wandb import Wandb
from utils.config import config
import wandb 


def parse_option():

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    # parser.add_argument("--local_rank", type=int, help='local rank is used for DistributedDataParallel')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    return args, config

def cfg_paths(config):
    config.data.train_scene_ply_path = os.path.join(
        config.data.data_root, config.data.train_scene_name, f'{config.data.train_scene_name}_semantic.ply')
    config.data.train_scene_glb_path = os.path.join(
        config.data.data_root, config.data.train_scene_name, f'{config.data.train_scene_name}.glb')
    config.data.train_pclseg_path = os.path.join(
        config.data.data_root, config.data.train_scene_name, f'{config.data.train_scene_name}_pclseg.txt')
    config.data.train_pcl_normals_path = os.path.join(
        config.data.data_root, config.data.train_scene_name, f'{config.data.train_scene_name}_normals.npy')
    config.data.train_shortest_path_dir = os.path.join(
        config.data.data_root, config.data.train_scene_name, f'shortest_paths')

    config.data.val_scene_ply_path = os.path.join(
        config.data.data_root, config.data.val_scene_name, f'{config.data.val_scene_name}_semantic.ply')
    config.data.val_scene_glb_path = os.path.join(
        config.data.data_root, config.data.val_scene_name, f'{config.data.val_scene_name}.glb')
    config.data.val_pclseg_path = os.path.join(
        config.data.data_root, config.data.val_scene_name, f'{config.data.val_scene_name}_pclseg.txt')
    config.data.val_pcl_normals_path = os.path.join(
        config.data.data_root, config.data.val_scene_name, f'{config.data.val_scene_name}_normals.npy')
    config.data.val_shortest_path_dir = os.path.join(
        config.data.data_root, config.data.val_scene_name, f'shortest_paths')

    return config

def extract_scene_graph_feature(config, sg_config, scene_settings):
    ############ initialize habitat simulator and ground truth scene graph ########
    scene_glb_path = scene_settings["scene_glb_path"]
    scene_name = scene_settings['scene_name']
    scene_ply_path = scene_settings['scene_ply_path']
    pclseg_path = scene_settings['pclseg_path']
    pcl_normals_path = scene_settings['pcl_normals_path']

    sim, action_names, sim_settings = init_sim(scene_glb_path)

    # initialize ground truth scene graph
    scene_graph = SceneGraphHabitat(sg_config, scene_name=scene_name)
    scene_graph.load_gt_scene_graph(
        scene_ply_path, pclseg_path, pcl_normals_path, sim)
    sim.close()


    ########### visualize loaded scene with bounding boxes ########################

    if config.visualize:
        visualize_scene_graph(scene_graph)

    # extract GCN features by pretrained 3DSSG model
    feature_extractor = SceneGraphPredictor(config.rel_dist_thresh)
    object_nodes = [scene_graph.object_layer.obj_dict[obj_id]
                    for obj_id in scene_graph.object_layer.obj_ids]
    map_ids2idx = {obj_id: idx for idx, obj_id in enumerate(scene_graph.object_layer.obj_ids)}
    # map_idx2ids = {idx: obj_id for idx, obj_id in enumerate(scene_graph.object_layer.obj_ids)}

    ''' 
    extractor returns a dictionary:  
    results={
        "pred_obj_prob": pred_obj_prob, # (N, D=20) numpy array
        "pred_obj_confidence": pred_obj_confidence, # (N,) numpy array
        "pred_obj_label": pred_obj_label, # (N,) numpy array 
        "edges": edges, # (M,2) numpy array, represented by object index (not id!)
        "pred_rel_prob": pred_rel_prob, # (M, F=9)
        "pred_rel_confidence": pred_rel_confidence,
        "pred_rel_label": pred_rel_label
    }
    '''
    with torch.no_grad():
        results = feature_extractor.predict(object_nodes)
        pred_obj_prob = results['pred_obj_prob']
        edges = results['edges']
        
    return scene_graph, pred_obj_prob, edges.t(), map_ids2idx


class SceneGraphTrainer(object):

    def __init__(self, model, num_epochs=1000, batch_size=16, val_freq=2, optimizer=None,
                scheduler=None, summary_writer=None, loss_w=1., num_objnav_class=None, device='cuda:0', config=None):
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.g_loss_fn = nn.BCELoss().to(device)
        self.n_loss_fn = F.nll_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.loss_w = loss_w
        self.num_objnav_class = num_objnav_class
        self.device = device
        self.config = config

    def train(self, train_sampler, val_sampler=None):
        for i in trange(self.num_epochs, desc="Epochs: "):
            step = 0
            correct = 0
            total = 0
            loss_cum = 0.
            val_iou = 0.
            best_val = 0.
            is_best = False
            Is = np.empty((len(train_sampler), self.num_objnav_class+1)) # last one for background class
            Us = np.empty((len(train_sampler), self.num_objnav_class+1))
            self.model.train()
            self.optimizer.zero_grad()
            with tqdm(train_sampler, desc='Train Steps: ') as tepoch:
                for sample in tepoch:
                    sample.to(self.device)
                    step += 1
                    inputs, edge_index, batch, y, n_y = sample.x, sample.edge_index, sample.batch, sample.y, sample.n_y
                    g_logits, n_logits = self.model(inputs, edge_index, batch=batch.to(device=inputs.device))
                    g_pred = torch.sigmoid(g_logits)
                    g_loss = self.g_loss_fn(g_pred,
                                            y.float().to(device=g_pred.device).view(g_pred.size(0), -1))
                    n_pred = F.log_softmax(n_logits, dim=1)
                    n_loss = self.n_loss_fn(n_pred, n_y.to(device=n_pred.device))
                    loss = g_loss + n_loss * self.loss_w
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    g_loss = g_loss.data.cpu().numpy()
                    n_loss = n_loss.data.cpu().numpy()
                    loss_cum += loss
                    pred_bin = np.round(g_pred.detach().cpu().numpy())
                    correct += (pred_bin ==
                                y.view(g_pred.size(0), -1).cpu().numpy()).sum()
                    total += y.size(0)
                    Is, Us = self.update_iu(Is, Us, n_pred, n_y, step-1)
                    tepoch.set_postfix(g_loss=g_loss, n_loss=n_loss)

            self.scheduler.step()
            loss_avg = loss_cum / step
            acc_avg = float(correct) / total
            iou = self.get_miou(Is, Us, phase='Train')
            logging.info('Train After epoch {0} Loss:{1:0.3f}, Train Accuracy: {2:0.3f}, Train IOU {3}'.format(i+1, loss_avg, acc_avg, iou))
            if i % self.val_freq == 0 and val_sampler:
                val_loss_avg, val_acc_avg, val_iou = self.val(val_sampler, i+1)
                if val_iou > best_val:
                    is_best = True
                    best_val = val_iou
            if is_best:
                save_checkpoint(self.config, i+1, self.model, self.optimizer, self.scheduler, is_best=is_best)
                is_best = False

            if self.summary_writer is not None:
                log_dict = {'train_loss': loss_avg,
                            'train_acc': acc_avg,
                            'val_loss': val_loss_avg,
                            'val_acc': val_acc_avg,
                            'val_iou': val_iou,
                            'learning_rate': optimizer.param_groups[0]['lr']}
                self.summary_writer.add_scalars('training', log_dict, i+1)
                wandb.log(log_dict)
        load_checkpoint(self.config, self.model,
                        load_path=os.path.join(self.config.ckpt_dir,
                        f'{self.config.logname}_ckpt_best.pth'),
                        printer=logging.info)
        set_seed(self.config.rng_seed)
        val_loss_avg, val_acc_avg, val_iou = self.val(val_sampler, -1)

    @torch.no_grad()
    def val(self, sampler, epoch=None):
        self.model.eval()
        step = 0
        correct = 0
        total = 0
        loss_cum = 0.
        Is = np.empty((len(sampler), self.num_objnav_class+1)) # last one for background class
        Us = np.empty((len(sampler), self.num_objnav_class+1))

        with tqdm(sampler, desc='Eval Steps: ') as tepoch:
            for sample in tepoch:
                sample.to(self.device)
                step += 1
                inputs, edge_index, batch, y, n_y = sample.x, sample.edge_index, sample.batch, sample.y, sample.n_y
                g_logits, n_logits = self.model(inputs, edge_index, batch=batch.to(device=inputs.device))
                g_pred = torch.sigmoid(g_logits)
                g_loss = self.g_loss_fn(g_pred,
                                        y.float().to(device=g_pred.device).view(g_pred.size(0), -1))
                n_pred = F.log_softmax(n_logits, dim=1)
                n_loss = self.n_loss_fn(n_pred, n_y.to(device=n_pred.device))
                loss = g_loss + n_loss * self.loss_w

                g_loss = g_loss.data.cpu().numpy()
                n_loss = n_loss.data.cpu().numpy()
                loss_cum += loss
                pred_bin = np.round(g_pred.detach().cpu().numpy())
                correct += (pred_bin ==
                                y.view(g_pred.size(0), -1).cpu().numpy()).sum()
                total += y.size(0)
                Is, Us = self.update_iu(Is, Us, n_pred, n_y, step-1)
                tepoch.set_postfix(g_loss=g_loss, n_loss=n_loss)

        loss_avg = loss_cum / step
        acc_avg = float(correct) / total
        iou = self.get_miou(Is, Us, phase='Eval')
        logging.info('Eval After epoch {0} Loss:{1:0.3f}, Val Accuracy: {2:0.3f}, Val IOU: {3}'.format(epoch, loss_avg, acc_avg, iou))
        return loss_avg, acc_avg, iou

    
    def update_iu(self, Is, Us, pred, target, i):
        for cl in range(self.num_objnav_class+1):
            cur_gt_mask = (target == cl).cpu().numpy()
            cur_pred_mask = (pred.max(dim=1)[1] == cl).cpu().numpy()
            I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
            Is[i, cl] = I
            Us[i, cl] = U
        return Is, Us
    
    def get_miou(self, Is, Us, phase='Train'):
        ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
        ious[np.isnan(ious)] = 1.
        iou = np.mean(ious)
        for cl in range(self.num_objnav_class+1):
            logging.info("===> {} mIOU for class {}: {}".format(phase, cl, ious[cl]))
        return iou

if __name__ == "__main__":

    opt, config = parse_option()
    cfg_paths(config)

    # random seed
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    set_seed(config.rng_seed)

    
    if config.load_path is None:
        tags = [config.data.dataset,
                config.mode,
                f'Arch{config.model.name}', f'L{config.model.num_layers}',
                f'D{config.model.hidden_dim}', f'Aggr{config.model.aggr}',
                f'B{config.training.batch_size}', f'LR{config.optimizer.lr}',
                f'Epoch{config.training.num_epochs}', f'Seed{config.rng_seed}',
                ]
        generate_exp_directory(config, tags)
        config.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
                resume_exp_directory(config, config.load_path)
                config.wandb.tags = ['resume']

    # wandb and tensorboard
    cfg_path = os.path.join(config.log_dir, "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system('cp %s %s' % (opt.cfg, config.log_dir))
    config.cfg_path = cfg_path

    # wandb config
    config.wandb.name = config.logname
    Wandb.launch(config, config.wandb.use_wandb, sync_tensorboard=False)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    if config.mode == "runtime":
        ############ load ground truth pointclouds ####################
        sg_config = SceneGraphHabitatConfig()

        ############ Training Scene ###########
        train_scene_settings = {"scene_glb_path": config.data.train_scene_glb_path,
                                "scene_name": config.data.train_scene_name,
                                "scene_ply_path": config.data.train_scene_ply_path,
                                "pclseg_path": config.data.train_pclseg_path,
                                "pcl_normals_path": config.data.train_pcl_normals_path}

        train_scene_graph, train_graph_feat, train_edges, train_map_ids2idx = extract_scene_graph_feature(
            config,
            sg_config,
            train_scene_settings)

        # Scene graph sampler
        train_sampler = DataSampler(
            sg_config, train_scene_graph, config.data.train_shortest_path_dir, obj_nav_class_list)

        train_dataset = PyGDatasetWrapper(train_sampler, train_graph_feat.cpu(), train_edges.cpu(), train_map_ids2idx)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        ############ Val Scene ###########
        val_scene_settings = {"scene_glb_path": config.data.val_scene_glb_path,
                                "scene_name": config.data.val_scene_name,
                                "scene_ply_path": config.data.val_scene_ply_path,
                                "pclseg_path": config.data.val_pclseg_path,
                                "pcl_normals_path": config.data.val_pcl_normals_path}

        val_scene_graph, val_graph_feat, val_edges, val_map_ids2idx = extract_scene_graph_feature(
            config,
            sg_config,
            val_scene_settings)

        # Scene graph sampler
        val_sampler = DataSampler(
            sg_config, val_scene_graph, config.data.val_shortest_path_dir, obj_nav_class_list)

        val_dataset = PyGDatasetWrapper(val_sampler, val_graph_feat.cpu(), val_edges.cpu(), val_map_ids2idx)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Scene graph executor
        print(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if config.model.name == "deepset":
            model = Deepset(config.model.hidden_dim, config.data.out_dim,
                            x_dim=config.data.in_dim, pool=config.model.aggr).to(device)
            train_args = (train_loader, train_graph_feat, train_map_ids2idx,
                          val_loader, val_graph_feat, val_map_ids2idx)
        elif config.model.name == 'deepgnn':
            model = DeeperGCN(config.data.in_dim, config.model.hidden_dim, config.data.out_dim,
                              num_layers=config.model.num_layers, aggr=config.model.aggr).to(device)
            train_args = (train_loader, val_loader)
        
        print(model)
        total_params, train_params = cal_model_parms(model)
        print(f'Total params: {total_params}, Trainable params: {train_params}')
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        trainer = SceneGraphTrainer(
            model,
            num_epochs=config.training.num_epochs,
            batch_size=config.training.batch_size,
            val_freq=config.training.val_freq,
            optimizer=optimizer,
            scheduler=scheduler,
            summary_writer=summary_writer,
            num_objnav_class=len(obj_nav_class_list),
            config=config
        )
        if config.load_path:
            load_checkpoint(config, model, optimizer, scheduler, printer=logging.info)
            if 'train' in config.mode:
                _, _, val_miou = trainer.val(val_loader, epoch=-1)
                logging.info(f'\nresume val mIoU is {val_miou}\n ')
            else:
                _, _, val_miou = trainer.val(val_loader, epoch=-1)
                logging.info(f'\nval mIoU is {val_miou}\n ')
                exit()
        trainer.train(*train_args)

    elif config.mode == "offline":
        with open(config.data.scene_graph_dump_path, "rb") as f:
            mp3d_data = pickle.load(f)

        # Todo

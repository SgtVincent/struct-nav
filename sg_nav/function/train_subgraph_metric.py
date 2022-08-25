import argparse
import json
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from models.deepgnn import DeeperGCN, DeeperMLP
from models.deepset import Deepset

# local import

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
from sg_nav_utils import (
    cal_model_parms,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from sg_nav_utils.config import config
from sg_nav_utils.logger import (
    generate_exp_directory,
    resume_exp_directory,
    setup_logger,
)
from sg_nav_utils.training import build_optimizer, build_scheduler
from sg_nav_utils.wandb import Wandb
from vis_ros.vis_obj_segmentation import SCANNET20_Label_Names


def parse_option():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="config file")
    # parser.add_argument("--local_rank", type=int, help='local rank is used for DistributedDataParallel')
    args, opts = parser.parse_known_args()
    config.load(args.cfg, recursive=True)
    config.update(opts)
    return args, config


def cfg_paths(config):
    config.data.train_scene_ply_paths = []
    config.data.train_scene_glb_paths = []
    config.data.train_pclseg_paths = []
    config.data.train_pcl_normals_paths = []
    config.data.train_shortest_path_dirs = []
    config.data.val_scene_ply_paths = []
    config.data.val_scene_glb_paths = []
    config.data.val_pclseg_paths = []
    config.data.val_pcl_normals_paths = []
    config.data.val_shortest_path_dirs = []

    for train_scene_name in config.data.train_scene_names:
        train_scene_ply_path = os.path.join(
            config.data.data_root,
            train_scene_name,
            f"{train_scene_name}_semantic.ply",
        )
        train_scene_glb_path = os.path.join(
            config.data.data_root, train_scene_name, f"{train_scene_name}.glb"
        )
        train_pclseg_path = os.path.join(
            config.data.data_root,
            train_scene_name,
            f"{train_scene_name}_pclseg.txt",
        )
        train_pcl_normals_path = os.path.join(
            config.data.data_root,
            train_scene_name,
            f"{train_scene_name}_normals.npy",
        )
        train_shortest_path_dir = os.path.join(
            config.data.data_root, train_scene_name, f"shortest_paths"
        )

        config.data.train_scene_ply_paths.append(train_scene_ply_path)
        config.data.train_scene_glb_paths.append(train_scene_glb_path)
        config.data.train_pclseg_paths.append(train_pclseg_path)
        config.data.train_pcl_normals_paths.append(train_pcl_normals_path)
        config.data.train_shortest_path_dirs.append(train_shortest_path_dir)

    for val_scene_name in config.data.val_scene_names:
        val_scene_ply_path = os.path.join(
            config.data.data_root,
            val_scene_name,
            f"{val_scene_name}_semantic.ply",
        )
        val_scene_glb_path = os.path.join(
            config.data.data_root, val_scene_name, f"{val_scene_name}.glb"
        )
        val_pclseg_path = os.path.join(
            config.data.data_root,
            val_scene_name,
            f"{val_scene_name}_pclseg.txt",
        )
        val_pcl_normals_path = os.path.join(
            config.data.data_root,
            val_scene_name,
            f"{val_scene_name}_normals.npy",
        )
        val_shortest_path_dir = os.path.join(
            config.data.data_root, val_scene_name, f"shortest_paths"
        )

        config.data.val_scene_ply_paths.append(val_scene_ply_path)
        config.data.val_scene_glb_paths.append(val_scene_glb_path)
        config.data.val_pclseg_paths.append(val_pclseg_path)
        config.data.val_pcl_normals_paths.append(val_pcl_normals_path)
        config.data.val_shortest_path_dirs.append(val_shortest_path_dir)

    return config


def extract_scene_graph_feature(config, sg_config, scene_settings):
    ############ initialize habitat simulator and ground truth scene graph ########
    scene_glb_path = scene_settings["scene_glb_path"]
    scene_name = scene_settings["scene_name"]
    scene_ply_path = scene_settings["scene_ply_path"]
    pclseg_path = scene_settings["pclseg_path"]
    pcl_normals_path = scene_settings["pcl_normals_path"]

    sim, action_names, sim_settings = init_sim(scene_glb_path)

    # initialize ground truth scene graph
    scene_graph = SceneGraphHabitat(sg_config, scene_name=scene_name)
    scene_graph.load_gt_scene_graph(
        scene_ply_path, pclseg_path, pcl_normals_path, sim
    )
    sim.close()

    ########### visualize loaded scene with bounding boxes ########################

    if config.visualize:
        visualize_scene_graph(scene_graph)

    # extract GCN features by pretrained 3DSSG model
    feature_extractor = SceneGraphPredictor(config.rel_dist_thresh)
    object_nodes = [
        scene_graph.object_layer.obj_dict[obj_id]
        for obj_id in scene_graph.object_layer.obj_ids
    ]
    map_ids2idx = {
        obj_id: idx
        for idx, obj_id in enumerate(scene_graph.object_layer.obj_ids)
    }
    # map_idx2ids = {idx: obj_id for idx, obj_id in enumerate(scene_graph.object_layer.obj_ids)}

    """ 
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
    """
    with torch.no_grad():
        results = feature_extractor.predict(object_nodes)
        pred_obj_prob = results["pred_obj_prob"]
        edges = results["edges"]

        ###############################
        pred_labels = results["pred_obj_label"]
        pred_names = [SCANNET20_Label_Names[i] for i in pred_labels]
        gt_names = scene_graph.object_layer.get_class_names(
            scene_graph.object_layer.obj_ids
        )
        print("pred_names", pred_names)
        print("gt_names", gt_names)

    return scene_graph, pred_obj_prob, edges.t(), map_ids2idx


def get_dataloader(
    config,
    sg_config,
    scene_settings,
    shortest_path_dir,
    obj_nav_class_list,
    batch_size=16,
    shuffle=True,
):
    scene_graph, graph_feat, edges, map_ids2idx = extract_scene_graph_feature(
        config, sg_config, scene_settings
    )

    # Scene graph sampler
    sampler = DataSampler(
        sg_config, scene_graph, shortest_path_dir, obj_nav_class_list
    )

    dataset = PyGDatasetWrapper(
        sampler, graph_feat.cpu(), edges.cpu(), map_ids2idx
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


class SceneGraphTrainer(object):
    def __init__(
        self,
        model,
        num_epochs=1000,
        batch_size=16,
        val_freq=2,
        optimizer=None,
        scheduler=None,
        summary_writer=None,
        loss_w=1.0,
        num_objnav_class=None,
        device="cuda:0",
        config=None,
    ):
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
            loss_cum = 0.0
            val_iou = 0.0
            best_val = 0.0
            is_best = False
            Is = np.empty(
                (len(train_sampler), self.num_objnav_class + 1)
            )  # last one for background class
            Us = np.empty((len(train_sampler), self.num_objnav_class + 1))
            self.model.train()
            self.optimizer.zero_grad()
            with tqdm(train_sampler, desc="Train Steps: ") as tepoch:
                for sample in tepoch:
                    sample.to(self.device)
                    step += 1
                    inputs, edge_index, batch, y, n_y = (
                        sample.x,
                        sample.edge_index,
                        sample.batch,
                        sample.y,
                        sample.n_y,
                    )
                    g_logits, n_logits = self.model(
                        inputs,
                        edge_index,
                        batch=batch.to(device=inputs.device),
                    )
                    g_pred = torch.sigmoid(g_logits)
                    g_loss = self.g_loss_fn(
                        g_pred,
                        y.float()
                        .to(device=g_pred.device)
                        .view(g_pred.size(0), -1),
                    )
                    n_pred = F.log_softmax(n_logits, dim=1)
                    n_loss = self.n_loss_fn(
                        n_pred, n_y.to(device=n_pred.device)
                    )
                    loss = 0 * g_loss + n_loss * self.loss_w  ########
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    g_loss = g_loss.data.cpu().numpy()
                    n_loss = n_loss.data.cpu().numpy()
                    loss_cum += loss
                    pred_bin = np.round(g_pred.detach().cpu().numpy())
                    correct += (
                        pred_bin == y.view(g_pred.size(0), -1).cpu().numpy()
                    ).sum()
                    total += y.size(0)
                    Is, Us = self.update_iu(Is, Us, n_pred, n_y, step - 1)
                    tepoch.set_postfix(g_loss=g_loss, n_loss=n_loss)

            self.scheduler.step()
            loss_avg = loss_cum / step
            acc_avg = float(correct) / total
            iou = self.get_miou(Is, Us, phase="Train")
            logging.info(
                "Train After epoch {0} Loss:{1:0.3f}, Train Accuracy: {2:0.3f}, Train IOU {3}".format(
                    i + 1, loss_avg, acc_avg, iou
                )
            )
            if i % self.val_freq == 0 and val_sampler:
                val_loss_avg, val_acc_avg, val_iou = self.val(
                    val_sampler, i + 1
                )
                if val_iou > best_val:
                    is_best = True
                    best_val = val_iou
            if is_best:
                save_checkpoint(
                    self.config,
                    i + 1,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    is_best=is_best,
                )
                is_best = False

            if self.summary_writer is not None:
                log_dict = {
                    "train_loss": loss_avg,
                    "train_acc": acc_avg,
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc_avg,
                    "val_iou": val_iou,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                self.summary_writer.add_scalars("training", log_dict, i + 1)
                if self.config.wandb.use_wandb:
                    wandb.log(log_dict)
        load_checkpoint(
            self.config,
            self.model,
            load_path=os.path.join(
                self.config.ckpt_dir, f"{self.config.logname}_ckpt_best.pth"
            ),
            printer=logging.info,
        )
        set_seed(self.config.rng_seed)
        val_loss_avg, val_acc_avg, val_iou = self.val(val_sampler, -1)

    @torch.no_grad()
    def val(self, sampler, epoch=None):
        self.model.eval()
        step = 0
        correct = 0
        total = 0
        loss_cum = 0.0
        Is = np.empty(
            (len(sampler), self.num_objnav_class + 1)
        )  # last one for background class
        Us = np.empty((len(sampler), self.num_objnav_class + 1))

        with tqdm(sampler, desc="Eval Steps: ") as tepoch:
            for sample in tepoch:
                sample.to(self.device)
                step += 1
                inputs, edge_index, batch, y, n_y = (
                    sample.x,
                    sample.edge_index,
                    sample.batch,
                    sample.y,
                    sample.n_y,
                )
                g_logits, n_logits = self.model(
                    inputs, edge_index, batch=batch.to(device=inputs.device)
                )
                g_pred = torch.sigmoid(g_logits)
                g_loss = self.g_loss_fn(
                    g_pred,
                    y.float()
                    .to(device=g_pred.device)
                    .view(g_pred.size(0), -1),
                )
                n_pred = F.log_softmax(n_logits, dim=1)
                n_loss = self.n_loss_fn(n_pred, n_y.to(device=n_pred.device))
                loss = 0 * g_loss + n_loss * self.loss_w  #########

                g_loss = g_loss.data.cpu().numpy()
                n_loss = n_loss.data.cpu().numpy()
                loss_cum += loss
                pred_bin = np.round(g_pred.detach().cpu().numpy())
                correct += (
                    pred_bin == y.view(g_pred.size(0), -1).cpu().numpy()
                ).sum()
                total += y.size(0)
                Is, Us = self.update_iu(Is, Us, n_pred, n_y, step - 1)
                tepoch.set_postfix(g_loss=g_loss, n_loss=n_loss)

        loss_avg = loss_cum / step
        acc_avg = float(correct) / total
        iou = self.get_miou(Is, Us, phase="Eval")
        logging.info(
            "Eval After epoch {0} Loss:{1:0.3f}, Val Accuracy: {2:0.3f}, Val IOU: {3}".format(
                epoch, loss_avg, acc_avg, iou
            )
        )
        return loss_avg, acc_avg, iou

    def update_iu(self, Is, Us, pred, target, i):
        for cl in range(self.num_objnav_class + 1):
            cur_gt_mask = (target == cl).cpu().numpy()
            cur_pred_mask = (pred.max(dim=1)[1] == cl).cpu().numpy()
            I = np.sum(
                np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32
            )
            U = np.sum(
                np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32
            )
            Is[i, cl] = I
            Us[i, cl] = U
        return Is, Us

    def get_miou(self, Is, Us, phase="Train"):
        ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
        ious[np.isnan(ious)] = 1.0
        iou = np.mean(ious)
        for cl in range(self.num_objnav_class + 1):
            logging.info(
                "===> {} mIOU for class {}: {}".format(phase, cl, ious[cl])
            )
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
        tags = [
            config.data.dataset,
            config.mode,
            f"Arch{config.model.name}",
            f"L{config.model.num_layers}",
            f"D{config.model.hidden_dim}",
            f"Aggr{config.model.aggr}",
            f"B{config.training.batch_size}",
            f"LR{config.optimizer.lr}",
            f"Epoch{config.training.num_epochs}",
            f"Seed{config.rng_seed}",
        ]
        generate_exp_directory(config, tags)
        config.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(config, config.load_path)
        config.wandb.tags = ["resume"]

    # wandb and tensorboard
    cfg_path = os.path.join(config.log_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system("cp %s %s" % (opt.cfg, config.log_dir))
    config.cfg_path = cfg_path

    # wandb config
    config.wandb.name = config.logname
    Wandb.launch(config, config.wandb.use_wandb, sync_tensorboard=False)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    if config.mode == "runtime":
        ############ load ground truth pointclouds ####################
        sg_config = SceneGraphHabitatConfig()

        ############ Training Scenes ###########
        train_loaders = []
        for i in range(len(config.data.train_scene_names)):
            train_scene_settings = {
                "scene_glb_path": config.data.train_scene_glb_paths[i],
                "scene_name": config.data.train_scene_names[i],
                "scene_ply_path": config.data.train_scene_ply_paths[i],
                "pclseg_path": config.data.train_pclseg_paths[i],
                "pcl_normals_path": config.data.train_pcl_normals_paths[i],
            }
            train_loader = get_dataloader(
                config,
                sg_config,
                train_scene_settings,
                config.data.train_shortest_path_dirs[i],
                obj_nav_class_list,
                batch_size=16,
                shuffle=True,
            )
            train_loaders.append(train_loader)

        val_loaders = []
        for i in range(len(config.data.val_scene_names)):
            val_scene_settings = {
                "scene_glb_path": config.data.val_scene_glb_paths[i],
                "scene_name": config.data.val_scene_names[i],
                "scene_ply_path": config.data.val_scene_ply_paths[i],
                "pclseg_path": config.data.val_pclseg_paths[i],
                "pcl_normals_path": config.data.val_pcl_normals_paths[i],
            }
            val_loader = get_dataloader(
                config,
                sg_config,
                val_scene_settings,
                config.data.val_shortest_path_dirs[i],
                obj_nav_class_list,
                batch_size=16,
                shuffle=False,
            )
            val_loaders.append(val_loader)

        # Scene graph executor
        print(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.model.name == "deepset":
            model = Deepset(
                config.model.hidden_dim,
                config.data.out_dim,
                x_dim=config.data.in_dim,
                pool=config.model.aggr,
            ).to(device)
        elif config.model.name == "deepgnn":
            # model = DeeperGCN(config.data.in_dim, config.model.hidden_dim, config.data.out_dim,
            #                  num_layers=config.model.num_layers, aggr=config.model.aggr).to(device)
            model = DeeperMLP(
                config.data.in_dim,
                config.model.hidden_dim,
                config.data.out_dim,
                num_layers=config.model.num_layers,
                aggr=config.model.aggr,
            ).to(device)

        print(model)
        total_params, train_params = cal_model_parms(model)
        print(
            f"Total params: {total_params}, Trainable params: {train_params}"
        )
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
            config=config,
        )
        if config.load_path:
            load_checkpoint(
                config, model, optimizer, scheduler, printer=logging.info
            )
            if "train" in config.mode:
                _, _, val_miou = trainer.val(val_loader, epoch=-1)
                logging.info(f"\nresume val mIoU is {val_miou}\n ")
            else:
                _, _, val_miou = trainer.val(val_loader, epoch=-1)
                logging.info(f"\nval mIoU is {val_miou}\n ")
                exit()

        num_loops = 100
        for _ in range(num_loops):
            for i in range(len(config.data.train_scene_names)):
                # train_args = (train_loaders[i], val_loaders[0])
                train_args = (train_loaders[i], train_loaders[i])
                trainer.train(*train_args)

    elif config.mode == "offline":
        with open(config.data.scene_graph_dump_path, "rb") as f:
            mp3d_data = pickle.load(f)

        # Todo

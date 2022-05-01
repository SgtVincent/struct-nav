#######################################################
# Reading inst masks and semantic segs for each object, 
# convert it to list of detect bounding boxes with labels
#######################################################

#%%
import numpy as np
import pandas as pd
import os, glob, argparse
import torch
from operator import itemgetter
import glob
import plotly
import plotly.io as pio
pio.renderers.default = 'vscode'
# local import 
from visualize_open3d import get_coords_color, COLOR_DETECTRON2, CLASS_COLOR, SEMANTIC_NAMES, SEMANTIC_IDX2NAME



#%% 
# initialization 
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./dataset/scannetv2', help='path to the dataset files')
parser.add_argument('--prediction_path', default='./exp/scannetv2/hais/hais_eval_scannet/result', help='path to the prediction results')
parser.add_argument('--data_split', help='train / val / test', default='val')
parser.add_argument('--room_name', help='room_name', default='scene0011_00')
arg = parser.parse_args([])


#%%[markdown]
## function to read instance segmentation 
#%%
# param definition 
# arg 
room_name = arg.room_name
#%%
# function content 
input_file = os.path.join(arg.data_path, arg.data_split, arg.room_name + '_inst_nostuff.pth')
assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
if arg.data_split == 'test':
    xyz, rgb = torch.load(input_file)
else:
    xyz, rgb, label, inst_label = torch.load(input_file)
rgb = (rgb + 1) * 127.5

instance_file = os.path.join(arg.prediction_path, arg.data_split, arg.room_name + '.txt')
assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
f = open(instance_file, 'r')
masks = f.readlines()
masks = [mask.rstrip().split() for mask in masks]
# inst_label_pred_rgb = np.zeros(rgb.shape)  # np.ones(rgb.shape) * 255 #

ins_num = len(masks)
ins_pointnum = np.zeros(ins_num)
inst_label = -np.ones(rgb.shape[0]).astype(np.int)

for i in range(len(masks)):
    mask_path = os.path.join(arg.prediction_path, arg.data_split, masks[i][0])
    assert os.path.isfile(mask_path), mask_path
    if (float(masks[i][2]) < 0.09):
        continue
    mask = np.loadtxt(mask_path).astype(np.int)
    print('{} {}: {} pointnum: {}'.format(i, masks[i], SEMANTIC_IDX2NAME[int(masks[i][1])], mask.sum()))      
    ins_pointnum[i] = mask.sum()
    inst_label[mask == 1] = i  



## check valid instance segmentation 
# sem_valid = (inst_label != -1)
# xyz = xyz[sem_valid]
# rgb = rgb[sem_valid]
# inst_label[sem_valid]
#%%
# inst_dir = "./exp/scannetv2/hais/hais_eval_scannet/result/val/instance"

# np.save(os.path.join(inst_dir, room_name), inst_label)
#%% 
# print(xyz)
# print(inst_label)
# import plotly.express as px
# df = pd.DataFrame({"instance label": inst_label})
# fig = px.histogram(df, x="instance label")
# fig.show()
# inst_label
#%%[markdown] 
## Bounding Box generation 
#%%
#---- convert instance segmentation to bounding box ------
# points = xyz
# # 
# inst_label = np.load(os.path.join(inst_dir, f"{room_name}.npy"))


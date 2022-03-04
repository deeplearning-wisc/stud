import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=1., type=float)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
args = parser.parse_args()



concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()



# ID data
ood_data = np.load('/nobackup/my_xfdu/video/vis/checkpoints/VIS/' + str(args.model) + '/ood.npy',allow_pickle=True)
id_data = np.load('/nobackup/my_xfdu/video/vis/checkpoints/VIS/' + str(args.model) + '/id.npy',allow_pickle=True)
# id_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# ood_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed' +'_'+str(args.seed)  +'/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# id_score = []
# energy score calculation.
# import ipdb; ipdb.set_trace()
index = 0
for data in id_data:
    if index == 0:
        id_data_all = data
        index += 1
    else:
        id_data_all = np.concatenate([id_data_all, data], 0)

id_data = torch.from_numpy(id_data_all)

index = 0
for data in ood_data:
    if index == 0:
        ood_data_all = data
        index += 1
    else:
        ood_data_all = np.concatenate([ood_data_all, data], 0)

ood_data = torch.from_numpy(ood_data_all)

# id_data = id_data / 1000 - torch.max(id_data, 1, keepdim=True)[0]
# ood_data = ood_data / 1000 - torch.max(ood_data, 1, keepdim=True)[0]
T = 1

# breakpoint()
# id_data = id_data[F.softmax(id_data[:,:-1], 1).max(1)[0] > 0.2]
# ood_data = ood_data[F.softmax(ood_data[:,:-1], 1).max(1)[0] > 0.2]

assert len(id_data[0]) == 41
if args.energy:
    id_score = -args.T * torch.logsumexp(id_data[:, :-1] / args.T, dim=1).cpu().data.numpy()
    ood_score = -args.T * torch.logsumexp(ood_data[:, :-1] / args.T, dim=1).cpu().data.numpy()
else:
    id_score = -np.max(F.softmax(id_data[:, :-1], dim=1).cpu().data.numpy(), axis=1)
    ood_score = -np.max(F.softmax(ood_data[:, :-1], dim=1).cpu().data.numpy(), axis=1)

###########
########
print(len(id_score))
print(len(ood_score))

# measures = get_measures(-id_score[:15866], -ood_score[:5258], plot=False)
measures = get_measures(-id_score, -ood_score, plot=False)
if args.energy:
    print_measures(measures[0], measures[1], measures[2], 'energy')
else:
    print_measures(measures[0], measures[1], measures[2], 'msp')



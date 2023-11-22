import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

import sys
sys.path.append('./models/') 

from cond_CFM import CNF, FlowMatchingLoss

import numpy as np
import normflows as nf
import os
from tqdm import tqdm
from typing import *
from zuko.utils import odeint
from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout

from jet_dataset import JetDataset

from typing import *

from sklearn.metrics import roc_curve, auc

#######################
### Define the data ###
#######################

batch_size = 131072

dont_use = ['jet_sdmass', 
            'jet_tau1',	'jet_tau2',	'jet_tau3',	'jet_tau4',
            'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

dataset = JetDataset("./jet_data",'train', del_context=dont_use)
dataset_val = JetDataset("./jet_data",'val', del_context=dont_use)

def alea_roc_curve(x_true, x_gen, treshholds = np.linspace(0,1,500)):
    '''
    x_true.shape: (N_jets)
    x_gen.shape: (N_jets, n_stat_alea, n_stat_epis)
    '''
    N_jets = len(x_true)
    assert N_jets == len(x_gen)
    n_stat_alea = x_gen.shape[1]
    n_stat_epis = x_gen.shape[2]

    tpr, fpr = np.zeros((len(treshholds), n_stat_epis)), np.zeros((len(treshholds), n_stat_epis))

    for i,t in tqdm(enumerate(treshholds)):
        label = x_gen >= t
        pos_ratio = np.sum(label, 1)/n_stat_alea
        neg_ratio = np.sum(label==0, 1)/n_stat_alea

        tp = np.sum(pos_ratio[x_true[:,0]==1], axis=0)
        fp = np.sum(pos_ratio[x_true[:,0]==0], axis=0)
        tn = np.sum(neg_ratio[x_true[:,0]==0], axis=0)
        fn = np.sum(neg_ratio[x_true[:,0]==1], axis=0)

        tpr[i] = tp/(tp+fn)
        fpr[i] = fp/(fp+tn)

    return fpr, tpr, treshholds

### plot the ROC curve including aleatoric uncertainty with epistemic envelopes ###

c_factor = 1
save_dir = f'./data/CFM_VIB_k{c_factor}_jets6/'

eval = False

n_stat_epis = 5
n_stat_alea = 20000
batchsize = 5
n_points = 10000

bins = np.linspace(0,1,21)
np.random.seed(0)
perm = np.random.permutation(len(dataset_val.features))
sort = np.argsort(dataset_val.features[:,1])

p_t_sorted = dataset_val.features[:,1][sort]

for start in [None, 200000, -100000, -n_points-1]:
    print('calculating:', start)
    save_str = 'x_gens'
    if start is not None:
        save_str += f'_{start}_{start+n_points}'
    
    with open(save_dir + save_str + '.npy', 'rb') as f:
        x_gens = np.load(f)
    
    x_true = dataset_val.truth[perm[:n_points]]

    fpr,tpr,t = roc_curve(dataset_val.truth[perm[:n_points]], dataset_val.raw_target[perm[:n_points]])
    gfpr,gtpr,_ = alea_roc_curve(x_true, x_gens, treshholds=t)

    with open(save_dir + save_str + '_fpr.npy', 'wb') as f:
        np.save(f, gfpr)
    with open(save_dir + save_str + '_tpr.npy', 'wb') as f:
        np.save(f, gtpr)

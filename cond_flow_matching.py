# Script for training a Conditonal Flow Matching Model either with or without Variational Inference Bayes

import torch
import torch.nn as nn

import sys
sys.path.append('./models/') 

from cond_CFM import CNF, FlowMatchingLoss

import numpy as np
import os
from tqdm import tqdm
from typing import *
from matplotlib import pyplot as plt

from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout
from models.cond_CFM import CNF, FlowMatchingLoss

from jet_dataset import JetDataset

def smooth(x, kernel_size=5):
    if kernel_size == 1:
        return x
    else:
        assert kernel_size % 2 != 0
        x_shape = x.shape
        x_tmp = np.array([x[i:x_shape[0]-kernel_size+i+1] for i in range(kernel_size)])
        edge1 = x[:int((kernel_size-1)/2)]
        edge2 = x[-int((kernel_size-1)/2):]
        x_out = np.concatenate((edge1, np.mean(x_tmp, 0),edge2),0)
        assert x_shape == x_out.shape
        return x_out #np.mean(np.array(x).reshape(-1, kernel_size),1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#######################
### Define the data ###
#######################

batch_size = 131072

dont_use = [#'jet_pt', 'jet_eta', 'jet_phi',	'jet_energy', 'jet_nparticles',	'jet_sdmass',
            #'jet_sdmass', 
            'jet_tau1',	
            'jet_tau2',	
            'jet_tau3',	
            'jet_tau4',
            'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

dataset = JetDataset("./jet_data",'train', del_context=dont_use)
dataset_val = JetDataset("./jet_data",'val', del_context=dont_use)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

###################################
### Define the Normalizing Flow ###
###################################

c_factor = 50
lr_decay = 1 #'cosine' #1 #0.999
approximate_gaussian_inference = True

if approximate_gaussian_inference:
    save_dir = f'./data/CFM_VIB_sampling_corrected_k{str(c_factor)}_jets{16-len(dont_use)}'
else:
    save_dir = f'./data/CFM_jets{16-len(dont_use)}'

if lr_decay != 1:
    save_dir += f'_lrdec{lr_decay}'
save_dir += '/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if approximate_gaussian_inference:
    model = CNF(1, conds = dataset.features.shape[1], n_nodes=[64] * 3, layer = LinearFlipout)

else:
    model = CNF(1, conds = dataset.features.shape[1], n_nodes=[64] * 3)
model.to(device)

for layer in model.modules():
    if isinstance(layer, LinearFlipout):
        layer._dnn_to_bnn_flag = True
        layer.auto_sample = False 

#####################
### Training Loop ###
#####################

lr = 1e-3
weight_decay = 0 

epochs = 4000

cfm_loss = FlowMatchingLoss(model)
optim = torch.optim.AdamW(model.parameters(), lr=lr)
if lr_decay == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
else:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
model.train()

loss_hist, prior_kld_hist = np.array([]), np.array([])

for ep in tqdm(range(epochs)):
    optim.zero_grad()
    
    for x, c in dataloader:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)

        for layer in model.modules():
            if isinstance(layer, LinearFlipout):
                layer.sample_weights()

        # Compute loss
        loss = cfm_loss(x, c)
        kl_loss = torch.nan_to_num(torch.sum(torch.tensor([layer.kl_loss() for layer in model.modules() 
                                                            if isinstance(layer, LinearFlipout)],  device=device))/len(x))
        
        loss_total = loss + c_factor*kl_loss

        # Do backprop and optimizer step
        if ~(torch.isnan(loss_total) | torch.isinf(loss_total)):
            loss.backward()
            optim.step()
            scheduler.step()
        
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    # Log loss
    prior_kld_hist = np.append(loss_hist, kl_loss.to('cpu').data.numpy())

    if ep%50 == 0:
        torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")

with open(save_dir + f'losses_{ep}.npy', 'wb') as f:
    np.save(f, loss_hist)

torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
print('saved model at ' + save_dir + f"model_{ep}.pth")

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(smooth(loss_hist,1), color = 'C1', alpha = 0.3)
plt.plot(smooth(loss_hist,11), label='loss', color = 'C1')
plt.grid()
plt.legend()
img_outfile = save_dir + f"image_ep{ep}_loss.png"
plt.savefig(img_outfile)
plt.show()

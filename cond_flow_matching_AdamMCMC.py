# Script for running AdamMCMC on a pretrained model

# %%
import torch
import torch.nn as nn

from models.cond_CFM import CNF
from models.MCMC_Adam import MCMC_by_bp

import numpy as np
import os
from tqdm import tqdm
from typing import *

from jet_dataset import JetDataset

from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

save_dir = './data/CFM_jets6/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

batch_size = 131072

dont_use = ['jet_sdmass', 
            'jet_tau1',	
            'jet_tau2',	
            'jet_tau3',	
            'jet_tau4',
            'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

dataset = JetDataset("./jet_data",'train', del_context=dont_use)
dataset_val = JetDataset("./jet_data",'val', del_context=dont_use)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, persistent_workers=True)

########################
### Define the Model ###
########################

model = CNF(1, conds = dataset.features.shape[1], n_nodes=[64] * 3).to(device)

#####################
### Training Loop ###
#####################

train = False
lr = 1e-2
weight_decay = 0 
lr_decay = 1 #0.999

epochs = 5000

ep = epochs-1

with open(save_dir + f'losses_{ep}.npy', 'rb') as f:
    loss_hist = np.load(f)

model.load_state_dict(torch.load(save_dir + f"model_{ep}.pth"))
print('loaded model from ' + save_dir + f"model_{ep}.pth")

model.eval()

# %%
MCMC_epochs = 400
MCMC_epochs_load = 596

name_add = '/AdamMCMC_models_lambda50/'

if not os.path.exists(save_dir + name_add):
    os.mkdir(save_dir + name_add)
    
if MCMC_epochs_load != 0:
    model.load_state_dict(torch.load(save_dir + name_add + f"AdamMCMC_model_{MCMC_epochs_load}.pth"))
    print('loaded model from ' + save_dir + f"AdamMCMC_model_{MCMC_epochs_load}.pth")

lr = 5e-6

temp = 50
sigma = 0.05

loop_kwargs = {
             'MH': True, #this is a little more than x2 runtime
             'verbose': False,
             'sigma_adam_dir': sigma, 
}

optim = torch.optim.Adam(model.parameters(), lr=lr)#, betas=(0.99,0.99999))
model.device = device

AdamMCMC = MCMC_by_bp(model, optim, temp, sigma)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
model.train()

loss_hist, acc_hist, b_hist = np.array([]), np.array([]), np.array([])
maxed_out_mbb_batches = 0

for ep in range(MCMC_epochs_load, MCMC_epochs_load+MCMC_epochs):
    optim.zero_grad()
    
    i = 0
    for x, c in tqdm(dataloader):
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)
        
        # Compute log Likl
        loss = lambda: -torch.mean(model.log_prob(x,c))

        # Do backprop and optimizer step
        #t1= time()
        maxed_out_mbb_batches += 1
        _,a,b,sigma,stop_dict = AdamMCMC.step(loss, **loop_kwargs)
        #print(f't_update: {time()-t1:4.4} s')
        
        if b: 
            maxed_out_mbb_batches  = 0
        if maxed_out_mbb_batches > 100:
            print('MBB sampling is not convergent, reinitializing the chain')
            AdamMCMC.start = True 

        scheduler.step()
        acc_hist = np.append(acc_hist, a.to('cpu').data.numpy())
        b_hist = np.append(b_hist, b)
        
    # Log loss
    loss_hist = np.append(loss_hist, loss().to('cpu').data.numpy())

    torch.save(model.state_dict(), save_dir + name_add + f"AdamMCMC_model_{ep}.pth")

with open(save_dir + name_add + f'AdamMCMC_losses_{ep}.npy', 'wb') as f:
    np.save(f, loss_hist)

with open(save_dir + name_add + f'AdamMCMC_acc_{ep}.npy', 'wb') as f:
    np.save(f, acc_hist)

with open(save_dir + name_add + f'AdamMCMC_b_{ep}.npy', 'wb') as f:
    np.save(f, b_hist)

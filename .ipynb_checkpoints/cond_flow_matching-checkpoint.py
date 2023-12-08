# # Testing the 'Flow Matching in 100 LOC'-code and augmenting it with Bayesian Methods

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
#from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout

from matplotlib import pyplot as plt

from jet_dataset import JetDataset

from typing import *
from zuko.utils import odeint

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
            #'jet_tau1',	
            #'jet_tau2',	
            #'jet_tau3',	
            #'jet_tau4',
            'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

dataset = JetDataset("./jet_data",'train', del_context=dont_use)
dataset_val = JetDataset("./jet_data",'val', del_context=dont_use)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

###################################
### Define the Normalizing Flow ###
###################################

c_factor = 1
lr_decay = 0.9999 #'cosine' #1 #0.999
approximate_gaussian_inference = True

if approximate_gaussian_inference:
    save_dir = f'./data/CFM_VIB_sampling_corrected_k{str(c_factor)}_jets{16-len(dont_use)}'
    #save_dir = f'./data/CFM_VIB_k{str(c_factor)}_jets{16-len(dont_use)}'
else:
    save_dir = f'./data/CFM_jets{16-len(dont_use)}'

if lr_decay != 1:
    save_dir += f'_lrdec{lr_decay}'
save_dir += '/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Construct flow model
#model = CNF(1, conds = dataset.features.shape[1], n_nodes=[64] * 3)

# Move model on GPU if available
#model = model.to(device)

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

train = True
lr = 1e-3
weight_decay = 0 

epochs = 7500

if train:
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

else:
    if c_factor == 100:
        load_epoch = 500
    elif c_factor == 1:
        load_epoch = 1150
    elif c_factor == 0.01:
        load_epoch = 2000

    model.load_state_dict(torch.load(save_dir + f"model_{load_epoch}.pth"))
    print('loaded model from ' + save_dir + f"model_{load_epoch}.pth")

model.eval()

# # Sampling
# with torch.no_grad():
# z = torch.randn(4096, 2)
# x = flow.decode(z).numpy()


# # Log-likelihood
# with torch.no_grad():
# log_p = flow.log_prob(data[:4])


### plot the ROC curve including aleatoric uncertainty with epistemic envelopes ###

eval = True

n_stat_epis = 5 if approximate_gaussian_inference else 1
n_stat_alea = 20000
batchsize = 5
n_points = 10000


bins = np.linspace(0,1,21)
np.random.seed(0)

if eval:
    perm = np.random.permutation(len(dataset_val.features))
    sort = np.argsort(dataset_val.features[:,1])

    for start in [None, 200000, -100000, -n_points-1]:
        if start is None:
            dat = dataset_val.features[perm[:n_points]]
        else:
            dat = dataset_val.features[sort[start:start+n_points]]

        x_gens = np.zeros((n_points, n_stat_alea, n_stat_epis))

        for k in tqdm(range(len(dat)//batchsize)):

            z = torch.randn(n_stat_alea*batchsize, 1).to(device)
            c = torch.Tensor(dat[k*batchsize:(k+1)*batchsize]).to(device)
            c = c.repeat_interleave(n_stat_alea, dim=0)
            
            for j in range(n_stat_epis):
                if approximate_gaussian_inference:
                    for layer in model.modules():
                        if isinstance(layer, LinearFlipout):
                            layer.auto_sample = False 
                            layer.sample_weights()

                x_gen = model.decode(z, cond=c)
                x_gens[k*batchsize:(k+1)*batchsize, :, j] = torch.sigmoid(x_gen*20).detach().cpu().numpy().reshape(batchsize, n_stat_alea)

        save_str = save_dir + 'x_gens'
        if start is not None:
            save_str += f'_{start}_{start+n_points}'
        save_str += '.npy'
        
        with open(save_str, 'wb') as f:
            np.save(f, x_gens)
# # Testing the normflows package and augmenting it with Bayesian Methods

import torch
import numpy as np
import normflows as nf
import os

#from matplotlib import pyplot as plt

from tqdm import tqdm

from jet_dataset import JetDataset

from models.normflows_wrapper.AutoregressiveRationalQuadraticSpline import AutoregressiveRationalQuadraticSpline
from models.custom_linear_flipout import custom_LinearFlipout as LinearFlipout

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

save_dir = './data/NF_VBI_jets6/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#######################
### Define the data ###
#######################

batch_size = 131072

dont_use = ['jet_sdmass', 
            'jet_tau1',	'jet_tau2',	'jet_tau3',	'jet_tau4',
            'aux_genpart_eta', 'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt', 'aux_truth_match']

dataset = JetDataset("./jet_data",'train', del_context=dont_use)
dataset_val = JetDataset("./jet_data",'val', del_context=dont_use)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

###################################
### Define the Normalizing Flow ###
###################################

# Define flows
K = 10

latent_size = 1
hidden_units = 128
hidden_layers = 2
context_size = dataset.features.shape[1]

flows = []
for i in range(K):
    flows += [AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model = model.to(device)

epoch_load = 9998
if epoch_load is not None:
    model.load_state_dict(torch.load(save_dir + f"model_{epoch_load}.pth"))
    print('loaded model from ' + save_dir + f"model_{epoch_load}.pth")

#####################
### Training Loop ###
#####################

train = True
lr = 1e-3
weight_decay = 0 
lr_decay = 1 #0.999

epochs = 5000

if train:
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, lr_decay)
    model.train()

    loss_hist, prior_kld_hist = np.array([]), np.array([])

    for ep in tqdm(range(epoch_load, epoch_load+epochs)):
        optim.zero_grad()
        
        for x, c in dataloader:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            
            # Compute loss
            loss = model.forward_kld(x, c)

            kl_loss = torch.nan_to_num(torch.sum(torch.tensor([layer.kl_loss() for layer in model.modules() 
                                                                if isinstance(layer, LinearFlipout)],  device=device))/len(x))
            
            loss_total = loss + kl_loss
            
            # Do backprop and optimizer step
            if ~(torch.isnan(loss_total) | torch.isinf(loss_total)):
                loss_total.backward()
                optim.step()
                scheduler.step()
            
        # Log loss
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                    
        # Log loss
        prior_kld_hist = np.append(prior_kld_hist, kl_loss.to('cpu').data.numpy())

        if ep%50 == 0:
            torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")

    model.eval()

    with open(save_dir + f'losses_{ep}.npy', 'wb') as f:
        np.save(f, loss_hist)

    with open(save_dir + f'prior_kld_{ep}.npy', 'wb') as f:
        np.save(f, prior_kld_hist)

    torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
    print('saved model at ' + save_dir + f"model_{ep}.pth")

else:
    ep = epochs-1

    with open(save_dir + f'losses_{ep}.npy', 'rb') as f:
        loss_hist = np.load(f)

    model.load_state_dict(torch.load(save_dir + f"model_{ep}.pth"))
    print('loaded model from ' + save_dir + f"model_{ep}.pth")


# Plot loss
# plt.figure(figsize=(10, 10))
# plt.plot(smooth(loss_hist,1), color = 'C1', alpha = 0.3)
# plt.plot(smooth(loss_hist,11), label='loss', color = 'C1')
# plt.plot(smooth(prior_kld_hist,1), color = 'C2', alpha = 0.3)
# plt.plot(smooth(prior_kld_hist,11), label='prior_kld', color = 'C2')

# plt.grid()
# plt.legend()
# img_outfile = save_dir + f"image_ep{ep}_loss.png"
# plt.savefig(img_outfile)
# plt.show()
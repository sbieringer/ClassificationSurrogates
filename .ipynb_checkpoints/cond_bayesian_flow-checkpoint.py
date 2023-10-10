import torch
import numpy as np
import normflows as nf
import os

#from matplotlib import pyplot as plt

from tqdm import tqdm
#from sklearn.metrics import roc_curve

import sys
sys.path.append('../JetSurrogate/')

from jet_dataset import JetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

save_dir = './data/NF_2_jets15/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

###################################
### Define the Normalizing Flow ###
###################################

# Define flows
K = 10

latent_size = 1
hidden_units = 64
hidden_layers = 2
context_size = 16

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
model = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model = model.to(device)

#######################
### Define the data ###
#######################

batch_size = 131072

dataset = JetDataset("./jet_data",'train')
dataset_val = JetDataset("./jet_data",'val')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#####################
### Training Loop ###
#####################

lr=1e-4 #1e-3 for first 2500
weight_decay= 0 #1e-5

load_epoch = 5499 #0

if load_epoch !=0:

    with open(save_dir + f'losses_{load_epoch}.npy', 'rb') as f:
        loss_hist = np.load(f)

    model.load_state_dict(torch.load(save_dir + f"model_{load_epoch}.pth"))
    print('loaded model from ' + save_dir + f"model_{load_epoch}.pth")
else:
    loss_hist = np.array([])


epochs = 2000+load_epoch+1
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.train()


for ep in range(load_epoch+1, epochs):
    optim.zero_grad()
    
    for x, c in dataloader:
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)
        
        # Compute loss
        loss = model.forward_kld(x, c)
        
        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optim.step()
        
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    if ep%50 == 0:
        torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
        print('saved model at ' + save_dir + f"model_{ep}.pth")

model.eval()

with open(save_dir + f'losses_{ep}.npy', 'wb') as f:
    np.save(f, loss_hist)

# Plot loss
# plt.figure(figsize=(10, 10))
# plt.plot(loss_hist, label='loss')
# plt.legend()
# plt.show()

torch.save(model.state_dict(), save_dir + f"model_{ep}.pth")
print('saved model at ' + save_dir + f"model_{ep}.pth")

################
### Plotting ###
################

# with torch.no_grad():
#     # get a batch to compare
#     conditions = torch.Tensor(dataset_val.features).to(device)
    
#     bs = 1000 
#     n_bs = len(conditions)//bs+1

#     x_gen, _ = model.sample(bs, context=conditions[:bs])
#     for i in range(n_bs-1):
#         c_data = conditions[(i+1)*bs:(i+2)*bs]
#         x_gen_tmp, _ = model.sample(len(c_data), context=c_data)
#         x_gen = torch.concat((x_gen, x_gen_tmp), 0)

#     x_gen = x_gen*20. #back to full scale
#     x_gen_raw = x_gen #HIER SOLLTE NOCH GEMITTELT WERDEN ÜBER OUTPUTS ZUR SELBEN CONDITION
#     x_gen = torch.sigmoid(x_gen)
#     print("plotting rocs")
#     fpr,tpr,_ = roc_curve(dataset_val.truth, dataset_val.raw_target)
#     gfpr,gtpr,_ = roc_curve(dataset_val.truth, x_gen.cpu())
#     plt.plot(tpr, fpr, label='"true" tagger')
#     plt.plot(gtpr, gfpr, label='surrogate')
#     plt.xlabel("Efficiency")
#     plt.ylabel("Fake rate")
#     plt.legend()
#     img_outfile = save_dir + f"image_ep{ep}.png"
#     plt.savefig(img_outfile)

#     plt.yscale('log')
#     img_outfile = save_dir + f"image_ep{ep}_log.png"
#     plt.savefig(img_outfile)
#     plt.show()

#     print("plotting raw", dataset_val.target.shape, x_gen_raw.shape)
#     _,b,_ = plt.hist(dataset_val.target[:,0]*20.,bins=100, label='"true" tagger',histtype='step')
#     print("plotting raw gen")
#     plt.hist(x_gen_raw.cpu().numpy()[:,0],bins=b, label='surrogate',histtype='step')
#     plt.legend()
#     img_outfile = save_dir + f"image_ep{ep}_raw.png"
#     plt.savefig(img_outfile)
#     plt.show()
#     print('saved images at ' + save_dir)
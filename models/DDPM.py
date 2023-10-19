''' 
This script does conditional image generation on MNIST, using a diffusion model
...and was then dumbed down for a jet diffusion surrogate

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

import torch
import torch.nn as nn
import numpy as np

## wrap jet data in dataset


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip = None) -> torch.Tensor:

        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            out = out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = x2
        
        if skip is not None:
            return out + skip
        return out


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualBlock(in_channels, n_feat, is_res=True)

        self.down1 = ResidualBlock(n_feat, n_feat//2)
        self.down2 = ResidualBlock(n_feat//2, n_feat//4)


        self.timeembed1 = EmbedFC(1, n_feat//2)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(16, n_feat//2)#16 features to embed
        self.contextembed2 = EmbedFC(16, n_feat)

        self.up0 = ResidualBlock(n_feat//4, n_feat//2) #can skip to down 1

        self.up1 = ResidualBlock(n_feat//2, n_feat) #can skip to init
        self.up2 = ResidualBlock(n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Linear(2*n_feat, n_feat),
            nn.ReLU(),
            nn.Linear(n_feat, self.in_channels),
        )

    def forward(self, x, c, t, context_mask=None):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = down2

        c = c.type(torch.float)
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat//2)
        temb1 = self.timeembed1(t).view(-1, self.n_feat//2)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat)
        temb2 = self.timeembed2(t).view(-1, self.n_feat)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        #print('>>>',hiddenvec.shape)
        up1 = self.up0(hiddenvec, down1)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        #print('>>>',c.shape,cemb1.shape, up1.shape, temb1.shape)
        up2 = self.up1(cemb1*up1+ temb1, x)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, x)
        out = self.out(torch.cat((up3, x), 1))
        return out

class SimpleCondFF(nn.Module):
    def __init__(self, n_classes, in_channels, n_feat = 256):
        super(SimpleCondFF, self).__init__()


        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        
        self.timeembed1 = EmbedFC(1, n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.contextembed1 = EmbedFC(16, n_feat)#16 features to embed
        self.contextembed2 = EmbedFC(16, n_feat)


        self.d0 = ResidualBlock(in_channels, n_feat)
        self.d1 = ResidualBlock(3 * n_feat, n_feat)
        self.d2 = ResidualBlock(3 * n_feat, n_feat)
        self.d3 = ResidualBlock(n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Linear(n_feat+1, n_feat),
            nn.GELU(),
            nn.Linear(n_feat, self.in_channels),
        )
    def forward(self, x, c, t, context_mask=None):

        cemb1 = self.contextembed1(c).view(-1, self.n_feat)
        temb1 = self.timeembed1(t).view(-1, self.n_feat)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat)
        temb2 = self.timeembed2(t).view(-1, self.n_feat)

        d0 = self.d0(x)
        d1 = self.d1(torch.cat([cemb1,temb1,d0], 1))
        d2 = self.d2(torch.cat([cemb2,temb2,d1], 1))
        d3 = self.d3(d2)
        return self.out(torch.cat([x,d3], 1))


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        #print('noise',noise.shape)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        #print('>> x_t',x_t.shape)
        # return MSE between added noise, and our predicted noise
        model_out = self.nn_model(x_t, c, _ts / self.n_T, context_mask)
        #print('>>>>>>',model_out.shape, noise.shape)
        return self.loss_mse(noise, model_out)

    def sample(self, conditions, device):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        if isinstance(conditions, np.ndarray):
            conditions = torch.from_numpy(conditions)
        n_sample = conditions.shape[0]

        x_i = torch.randn(n_sample, 1).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = conditions.to(device) # context for us just cycles throught the mnist labels
        

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1)

            z = torch.randn(n_sample, 1).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            #if i%20==0 or i==self.n_T or i<8:
            #    x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
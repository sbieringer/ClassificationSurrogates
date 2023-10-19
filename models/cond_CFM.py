### code from https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa slightly modded by Cedric Ewen###

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from zuko.utils import odeint
from typing import *

class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        activation: str = "ELU",
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), getattr(nn, activation)()])

        super().__init__(*layers[:-1])


class cond_MLP_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ELU",
        dim_t: int = 6,
        dim_cond: int = 1,
        n_nodes: list = [64,256,256,64]
    ):
        
        super().__init__()
        self.mlps1 = MLP(
            in_features + dim_t + dim_cond,
            out_features=n_nodes[0],
            hidden_features=[n_nodes[0], n_nodes[0]],
            activation=activation,
            )
        self.mlps = [self.mlps1]

        for i in range(len(n_nodes)-2):
            setattr(self, f'mlp{i+1}', MLP(
                n_nodes[i] + dim_t + dim_cond,
                out_features=n_nodes[i+1],
                hidden_features=[n_nodes[i+1], n_nodes[i+1]],
                activation=activation,
                ))
            self.mlps.append(getattr(self, f'mlp{i+1}'))
            
        setattr(self, f'mlp{i+2}', MLP(
            n_nodes[-2] + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=[n_nodes[-1], n_nodes[-1]],
            activation=activation,
            ))
        self.mlps.append(getattr(self, f'mlp{i+2}'))

    def forward(self, t, x, cond):
        for mlp in self.mlps:
            x = torch.cat([t, x, cond], dim=-1)
            x = mlp(x)
        return x

class CNF(nn.Module):
    def __init__(
        self,
        features: int,
        freqs: int = 3,
        conds: int = 0,
        **kwargs,
    ):
        super().__init__()

        #self.net = MLP(2 * freqs + features, features, **kwargs)
        self.net = cond_MLP_model(features, features, dim_t=2*freqs, dim_cond=conds, **kwargs)

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor, x: Tensor, cond: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(t, x, cond)

    def encode(self, x: Tensor, cond: Tensor) -> Tensor:
        return odeint(lambda t, x_tmp: self(t, x_tmp, cond), x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        return odeint(lambda t, x_tmp: self(t, x_tmp, cond), z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x: Tensor,  cond: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, cond: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x, cond)

            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, cond, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y, cond) - u).square().mean()


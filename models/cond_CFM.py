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

    def forward()


class small_cond_MLP_model(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "ELU",
        dim_t: int = 6,
        dim_cond: int = 1,
    ):
        super().__init__()
        self.mlp1 = MLP(
            in_features + dim_t + dim_cond,
            out_features=64,
            hidden_features=[64, 64],
            activation=activation,
        )
        self.mlp2 = MLP(
            64 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp3 = MLP(
            256 + dim_t + dim_cond,
            out_features=256,
            hidden_features=[256, 256],
            activation=activation,
        )
        self.mlp4 = MLP(
            256 + dim_t + dim_cond,
            out_features=out_features,
            hidden_features=[64, 64],
            activation=activation,
        )

    def forward(self, t, x, cond):
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp1(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp2(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp3(x)
        x = torch.cat([t, x, cond], dim=-1)
        x = self.mlp4(x)
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
        self.net = small_cond_MLP_model(2 * freqs + features, features, dim_t=freqs, dim_cond=conds)

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor, x: Tensor, cond: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(torch.cat((t, x, cond), dim=-1))

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, 0.0, 1.0, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        return odeint(self, z, 1.0, 0.0, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1]).to(x)
        I = I.expand(x.shape + x.shape[-1:]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, I, is_grads_batched=True, create_graph=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), 0.0, 1.0, phi=self.parameters())

        return Normal(0.0, z.new_tensor(1.0)).log_prob(z).sum(dim=-1) + ladj * 1e2


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0]).unsqueeze(-1)
        z = torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y) - u).square().mean()


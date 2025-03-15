import torch
import torch.nn as nn
import math
import numpy as np
import pdb
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.silu(input)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()

        self.dim = dim
        self.scale = scale

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        
        input = input*self.scale + 1
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb
    
    

class SDE(nn.Module):
    def __init__(self, shift_model, n_steps, time_dim):
        
        super().__init__()
        self.shift_model = shift_model
        self.n_steps = n_steps
        self.delta_t = 1/n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.time = nn.Sequential(
            TimeEmbedding(time_dim, scale=n_steps),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
        
    def forward(self, x0, gamma = 0.0, traj = False):
        x = x0
        t = (torch.zeros(x0.shape[0])).to(self.device)
        trajectory = [x0]
        
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                t_emb = self.time(t)[:, :, None, None]
                x = x + self.delta_t*self.shift_model(x, t_emb) + torch.randn_like(x)*np.sqrt(gamma*self.delta_t)
            else:
                x = x + self.delta_t*self.shift_model(x, t_emb)
            t += self.delta_t
            trajectory.append(x)
        if traj:
            return x, trajectory
        return x
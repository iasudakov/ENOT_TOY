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
    def __init__(self, shift_model, n_steps):
        
        super().__init__()
        self.shift_model = shift_model
        self.n_steps = n_steps
        self.delta_t = 1/n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x0, gamma = 0.0, traj = False):
        x = x0
        t = (torch.zeros(x0.shape[0])).to(self.device)
        trajectory = [x0]
        
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                x = x + self.delta_t*self.shift_model(x, t) + torch.randn_like(x)*np.sqrt(gamma*self.delta_t)
            else:
                x = x + self.delta_t*self.shift_model(x, t)
            t += self.delta_t
            trajectory.append(x)
        if traj:
            return x, trajectory
        return x
    
    
class SDE_denoiser(nn.Module):
    def __init__(self, denoiser, n_steps):
        
        super().__init__()
        self.denoiser = denoiser
        self.n_steps = n_steps
        self.delta_t = 1/n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x0, gamma = 0.0, traj = False):
        x = x0
        t = (torch.zeros(x0.shape[0])).to(self.device)
        trajectory = [x0]
        
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                x = x + self.delta_t*(self.denoiser(x, t) - x)/(1-torch.tensor(t)[:, None, None, None].cuda()) + torch.randn_like(x)*np.sqrt(gamma*self.delta_t)
            else:
                x = x + self.delta_t*(self.denoiser(x, t) - x)/(1-torch.tensor(t)[:, None, None, None].cuda())
            t += self.delta_t
            trajectory.append(x)
        if traj:
            return x, trajectory
        return x
    


class G_wrapper(nn.Module):
    def __init__(self, G, nz=100, z_std=0.1):
        super().__init__()
        self.G = G
        self.nz = nz
        self.z_std = z_std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x0):
        x = torch.cat([x0, torch.randn_like(x0[:, :1, :, :])], dim=1)
        xN = self.G(x)
        return xN
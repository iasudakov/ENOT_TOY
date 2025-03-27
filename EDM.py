import os, sys
sys.path.append("..")

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm
from IPython.display import clear_output

from MNIST_models.plotters import plot_trajectories, plot_images

from src.tools import load_dataset

import wandb
import gc

import os
SEED = 0xBADBEEF
torch.manual_seed(SEED); np.random.seed(SEED)

batch_size = 64
IMG_SIZE = 32
IMG_CHANNELS = 3
ZC = 1
Z_STD = 1.0
GAMMA = 0.5

TIME_DIM = 128
UNET_BASE_FACTOR = 48
N_STEPS = 10

lr = 1e-4

G_ITERS = 10
D_ITERS = 1
f_ITERS = 2
MAX_STEPS = 50000



dct = {
'batch_size' : batch_size,
'IMG_SIZE': IMG_SIZE,
'IMG_CHANNELS': IMG_CHANNELS,
'ZC': ZC,
'Z_STD': Z_STD,
'GAMMA': GAMMA,

'TIME_DIM': TIME_DIM,
'UNET_BASE_FACTOR': UNET_BASE_FACTOR,
'N_STEPS': N_STEPS,
'lr': lr,

'G_ITERS': G_ITERS,
'D_ITERS': D_ITERS,
'f_ITERS': f_ITERS,
'MAX_STEPS': MAX_STEPS
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampler3, test_sampler3 = load_dataset('MNIST-colored_3', './datasets/MNIST', img_size=IMG_SIZE, batch_size=batch_size, device=device)
sampler2, test_sampler2 = load_dataset('MNIST-colored_2', './datasets/MNIST', img_size=IMG_SIZE, batch_size=batch_size, device=device)
Y_sampler = sampler3
X_sampler = sampler2


from EDM_models.D import SongUNet_D
from EDM_models.G import SongUNet_G
from EDM_models.f import SongUNet_f

from EDM_models.enot import SDE_denoiser, G_wrapper


model_channels = 32

D = SongUNet_D(IMG_SIZE, IMG_CHANNELS, model_channels=model_channels).to(device)
G = SongUNet_G(IMG_SIZE, IMG_CHANNELS+1, IMG_CHANNELS, model_channels=model_channels).to(device)
G = G_wrapper(G, ZC, Z_STD)
f = SongUNet_f(IMG_SIZE, IMG_CHANNELS, IMG_CHANNELS, model_channels=model_channels).to(device)
sde = SDE_denoiser(denoiser=f, n_steps=N_STEPS).to(device)

sde_opt = Adam(sde.parameters(), lr=lr*10)
G_opt = Adam(G.parameters(), lr=lr)
D_opt = Adam(D.parameters(), lr=lr)
    
print('D params:', np.sum([np.prod(p.shape) for p in D.parameters()]))
print('G params:', np.sum([np.prod(p.shape) for p in G.parameters()]))
print('sde params:', np.sum([np.prod(p.shape) for p in sde.parameters()]))


def trainENOT(X_sampler, Y_sampler, G, G_opt, D, D_opt, sde, sde_opt):
    
    for step in tqdm(range(MAX_STEPS)):
            
        for G_iter in range(G_ITERS):

            for f_iter in range(f_ITERS):
                x0 = X_sampler.sample(batch_size)
                xN = G(x0)
                
                t = (torch.rand(batch_size)*0.989).to(device)
                xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*GAMMA)[:, None, None, None]
                
                f_loss = ((sde.denoiser(xt, t) - xN) ** 2).mean()
                sde_opt.zero_grad(); f_loss.backward(); sde_opt.step()

            x0 = X_sampler.sample(batch_size)
            xN = G(x0)

            t = (torch.rand(batch_size)*0.998).to(device)
            xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*GAMMA)[:, None, None, None]
            
            f_x_t = (sde.denoiser(xt, t) - xt)
            E = (xN - xt)

            loss1 = ((f_x_t*E).mean() - (f_x_t*f_x_t).mean()/2)*2
            loss2 = - D(xN).mean()

            G_loss = loss1 + loss2
            
            G_opt.zero_grad(); G_loss.backward(); G_opt.step()
        
        
        # plotter 1
        if step % 50 == 0:
            clear_output(wait=True)
            
            with torch.no_grad():
                X = X_sampler.sample(batch_size)

                T_XZ_np = []
                
                # Our method results G
                for i in range(100):
                    T_XZ_np.append(G(X).cpu().numpy())
            
                T_XZ_np = np.array(T_XZ_np)
                wandb.log({f'G var' : T_XZ_np.var(axis=0).mean(axis=0).mean().item()}, step=step)

                T_X_np = []
                for i in range(100):
                    T_X_np.append(sde(X, GAMMA).cpu().numpy())
            
                T_X_np = np.array(T_X_np)
                wandb.log({f'sde var' : T_X_np.var(axis=0).mean(axis=0).mean().item()}, step=step)
                
            
                X = X_sampler.sample(batch_size)
                
                G_dataset = G(X).detach()
                f_dataset = sde(X, GAMMA).detach()
                
                wandb.log({f'G mse' : F.mse_loss(X.detach(), G_dataset).item()}, step=step)
                wandb.log({f'sde mse' : F.mse_loss(X.detach(), f_dataset).item()}, step=step)

                torch.cuda.empty_cache(); gc.collect()
                
                fig1 = plot_trajectories(sde, GAMMA, X_sampler, 3)
                wandb.log({"trajectories": wandb.Image(fig1)}, step=step)
                plt.close(fig1)

                torch.cuda.empty_cache(); gc.collect()

                fig2 = plot_images(sde, X_sampler, 4, 2)
                wandb.log({"generated_images": wandb.Image(fig2)}, step=step)
                plt.close(fig2)

                torch.cuda.empty_cache(); gc.collect()
            
    
        for D_iter in range(D_ITERS):
            x0 = X_sampler.sample(batch_size)
            x1 = Y_sampler.sample(batch_size)
            xN = G(x0)
            D_loss = (- D(x1) + D(xN)).mean()
            D_opt.zero_grad(); D_loss.backward(); D_opt.step()

        wandb.log({f'f_loss' : f_loss.item()}, step=step)
        wandb.log({f'G_loss' : G_loss.item()}, step=step)
        wandb.log({f'D_loss' : D_loss.item()}, step=step)
        
        
wandb.init(project='MNIST_EDM')


stats = trainENOT(X_sampler, Y_sampler, G, G_opt, D, D_opt, sde, sde_opt)
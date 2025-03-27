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

from src.fid import save_model_samples

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

model_channels = 32



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
'MAX_STEPS': MAX_STEPS,
"model_channels": model_channels
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sampler3, test_sampler3, loader3, test_loader3 = load_dataset('MNIST-colored_3', './datasets/MNIST', img_size=IMG_SIZE, batch_size=batch_size, device=device)
sampler2, test_sampler2, loader2, test_loader2 = load_dataset('MNIST-colored_2', './datasets/MNIST', img_size=IMG_SIZE, batch_size=batch_size, device=device)
Y_sampler = sampler3
X_sampler = sampler2

Y_loader_test = loader3
X_loader_test = loader2



from EDM_models.D import SongUNet_D
from EDM_models.G import SongUNet_G
from EDM_models.f import SongUNet_f

from EDM_models.enot import SDE_denoiser, G_wrapper



D = SongUNet_D(IMG_SIZE, IMG_CHANNELS, model_channels=model_channels*2).to(device)
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


from src.fid import calculate_inception_stats, calculate_fid_from_inception_stats

def calc_fid(G_samples_path, Y_samples_path, num_expected, batch):
    mu_X, sigma_X = calculate_inception_stats(image_path=G_samples_path, num_expected=num_expected, max_batch_size=batch)
    mu_Y, sigma_Y = calculate_inception_stats(image_path=Y_samples_path, num_expected=num_expected, max_batch_size=batch)
    fid = calculate_fid_from_inception_stats(mu_X, sigma_X, mu_Y, sigma_Y)
    return fid



def trainENOT(X_sampler, Y_sampler, G, G_opt, D, D_opt, sde, sde_opt):
    
    for step in tqdm(range(MAX_STEPS)):
            
        for G_iter in range(G_ITERS):

            for f_iter in range(f_ITERS):
                x0 = X_sampler.sample(batch_size)
                xN = G(x0)
                
                t = torch.rand(batch_size).to(device)
                xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*GAMMA)[:, None, None, None]
                
                f_loss = ((sde.denoiser(xt, t) - xN) ** 2).mean()
                sde_opt.zero_grad(); f_loss.backward(); sde_opt.step()

            x0 = X_sampler.sample(batch_size)
            xN = G(x0)

            t = torch.rand(batch_size).to(device)
            xt = x0 + (xN - x0) * t[:, None, None, None] + torch.randn_like(x0)*torch.sqrt(t*(1-t)*GAMMA)[:, None, None, None]
            
            f_x_t = (sde.denoiser(xt, t) - xt)
            E = (xN - xt)

            G_loss = ((f_x_t*E).mean() - (f_x_t*f_x_t).mean()/2)*2 - D(xN).mean()
            G_opt.zero_grad(); G_loss.backward(); G_opt.step()
        
        
        if step % 50 == 0:
            clear_output(wait=True)
            
            with torch.no_grad():
                X = X_sampler.sample(batch_size)

                T_XZ_np = []
                for i in range(100):
                    T_XZ_np.append(G(X).cpu().numpy())
                T_XZ_np = np.array(T_XZ_np)
                wandb.log({f'G var' : T_XZ_np.var(axis=0).mean().item()}, step=step)

                T_X_np = []
                for i in range(100):
                    T_X_np.append(sde(X, GAMMA).cpu().numpy())
                T_X_np = np.array(T_X_np)
                wandb.log({f'sde var' : T_X_np.var(axis=0).mean().item()}, step=step)
            
                G_dataset = G(X).detach()
                f_dataset = sde(X, GAMMA).detach()
                
                wandb.log({f'G L2' : F.mse_loss(X.detach(), G_dataset).item()}, step=step)
                wandb.log({f'sde L2' : F.mse_loss(X.detach(), f_dataset).item()}, step=step)
                torch.cuda.empty_cache(); gc.collect()
                
                fig1 = plot_trajectories(sde, GAMMA, X_sampler, 3)
                wandb.log({"trajectories": wandb.Image(fig1)}, step=step)
                plt.close(fig1)
                torch.cuda.empty_cache(); gc.collect()

                fig2 = plot_images(G, X_sampler, 4, 2)
                wandb.log({"G_images": wandb.Image(fig2)}, step=step)
                plt.close(fig2)
                torch.cuda.empty_cache(); gc.collect()
                
                fig3 = plot_images(sde, X_sampler, 4, 2, GAMMA)
                wandb.log({"SDE_images": wandb.Image(fig3)}, step=step)
                plt.close(fig3)
                torch.cuda.empty_cache(); gc.collect()
                
                l2, lpips = save_model_samples('samplesG', G, X_loader_test, 32, 1000, device, 'samplesY', Y_loader_test)
                fid = calc_fid('samplesG', 'samplesY', 1000, 32)
                wandb.log({f'FID' : fid}, step=step)
                wandb.log({f'lpips' : lpips}, step=step)
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
        
wandb.init(project='MNIST_EDM', config = dct)



stats = trainENOT(X_sampler, Y_sampler, G, G_opt, D, D_opt, sde, sde_opt)


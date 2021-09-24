
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_kld(beta):
    ''' return MSE and KL divergence loss for VAE and Beta VAE'''
   
    def criterion(x_recon, x, mu, logvar):
        mse = F.mse_loss(x_recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = mse + beta * kld
        return mse, kld, total
    return criterion


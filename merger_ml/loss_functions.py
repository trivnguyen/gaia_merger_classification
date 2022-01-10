
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_kld(beta):
    ''' return MSE and KL divergence loss for VAE and Beta VAE'''

    def criterion(x_recon, x, mu, logvar):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kld
        return recon, kld, total
    return criterion

def bce_kld(beta):
    ''' return MSE and KL divergence loss for VAE and Beta VAE'''

    def criterion(x_recon, x, mu, logvar):
        recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kld
        return recon, kld, total
    return criterion


import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc31 = nn.Linear(128, z_dim)
        self.fc32 = nn.Linear(128, z_dim)

    def forward(self, x):
        ''' Calculate mu, logvar of latent distribution
        and sample the latent variable z '''
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc31(h)
        logvar = self.fc32(h)
        z = self.sample(mu, logvar)
        return z, mu, logvar

    def sample(self, mu, logvar):
        ''' Using parameterize trick to sample z from the normal distribution
        N(mu, logvar) '''
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        ''' Reconstruct the original input x from the latent variable z'''
        x_recon = F.relu(self.fc1(z))
        x_recon = F.relu(self.fc2(x_recon))
        x_recon = self.fc3(x_recon)
        return x_recon

class VAENetwork(nn.Module):
    ''' Simple Variational Autoencoder '''

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim)

    def forward(self, x):
        ''' Forward propagate x '''
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar

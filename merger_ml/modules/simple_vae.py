
import torch
import torch.nn as nn

# define default parameters tuning config
DEFAULT_CONFIG = {
    'lr': {
        'dist': 'LOG_UNIFORM',
        'min': 1e-6,
        'max': 1e-2,
    },
    'beta': {
        'dist': 'UNIFORM',
        'min': 1,
        'max': 10,
    },
    'lencode': {
        'dist': 'UNIFORM',
        'min': 32,
        'max': 512,
    },
    'ldecode': {
        'dist': 'UNIFORM',
        'min': 32,
        'max': 512,
    }
}

class FCBlock(nn.Module):
    ''' Convienient FC block '''
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dims, lencode=32):
        super().__init__()
        self.fc1 = FCBlock(input_dims, lencode)
        self.fc2 = nn.Linear(lencode, latent_dims)
        self.fc3 = nn.Linear(lencode, latent_dims)
    
    def forward(self, x):
        ''' Calculate mu, logvar of latent distribution 
        and sample the latent variable z '''
        x = self.fc1(x)
        
        # calculate mu, logvar of the normal latent distribution
        mu = self.fc2(x)
        logvar = self.fc3(x)
        
        # sample z using reparameterized trick
        z = mu + logvar.mul(0.5).exp_() * torch.randn_like(mu)
        return z, mu, logvar
        

class Decoder(nn.Module):
    def __init__(self, output_dims, latent_dims, ldecode=32):
        super().__init__()
        self.fc = nn.Sequential(
            FCBlock(latent_dims, ldecode),
            nn.Linear(ldecode, output_dims)
        )
    def forward(self, z):
        ''' Reconstruct the original input x from the latent variable z'''
        return self.fc(z)


class SimpleVAE(nn.Module):
    ''' Simple Variational Autoencoder '''
    
    def __init__(self, input_dims, latent_dims, lencode=32, ldecode=32):
        super().__init__()
        self.encoder = Encoder(input_dims, 2, lencode)
        self.decoder = Decoder(input_dims, 2, ldecode)
        
    def forward(self, x):
        ''' Forward propagate x '''
        z, mu, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar


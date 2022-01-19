
import json
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# define default parameters tuning config
DEFAULT_CONFIG = {
    'lr': {
        'dist': 'LOG_UNIFORM',
        'min': 1e-6,
        'max': 1e-2,
    },
    'num_hidden': {
        'dist': 'UNIFORM_INT',
        'min': 5,
        'max': 10,
    },
    'layer_sizes': {
        'dist': 'UNIFORM_INT',
        'min': 8,
        'max': 128,
    }
}

class FCBlock(nn.Module):
    ''' Convienient FC block '''
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        return self.fc(x)

class FCNetwork(nn.Module):
    ''' FC network that takes in kinematics of a star (e.g. ra, dec, parallax, pmra, pmdec)
    '''
    def __init__(self, input_dims=1, hidden_layers=[8, 16, 32], init_weights=False,
                 hyperparameters_fn=None):
        ''' initialize NN '''
        super().__init__()
        self.name = 'FC'

        if hyperparameters_fn is None:
            self.hyperparameters = {
                'input_dims': input_dims,
                'hidden_layers': hidden_layers,
                'init_weights': init_weights,
            }
        else:
            logger.info('Hyperparameters file is given. Ignore all other args')
            with open(hyperparameters_fn, 'r') as f:
                self.hyperparameters = json.load(f)
            input_dims = self.hyperparameters['input_dims']
            hidden_layers = self.hyperparameters['hidden_layers']
            init_weights = self.hyperparameters['init_weights']

        num_hidden = len(hidden_layers)
        layers = []
        layers.append(FCBlock(input_dims, hidden_layers[0]))   # input layers
        for i in range(num_hidden - 1):
            l_prev = hidden_layers[i]
            l_next = hidden_layers[i+1]
            layers.append(FCBlock(l_prev, l_next))
        layers.append(nn.Linear(hidden_layers[-1], 1))  # output layers
        self.fc = nn.Sequential(*layers)

        if init_weights:
            self.apply(self._init_weights)

    def forward(self, x):
        ''' Forward propagate x '''
        return self.fc(x)

    def _init_weights(self, m):
        ''' Initialize weight '''
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


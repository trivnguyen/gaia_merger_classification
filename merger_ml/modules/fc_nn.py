
import torch
import torch.nn as nn

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
    def __init__(self, input_dims, hidden_layers=[8, 16, 32], init_weights=True):
        ''' initialize NN '''
        super().__init__()
        self.name = 'FC'

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
            torch.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


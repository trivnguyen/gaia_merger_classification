
import torch
import torch.nn as nn

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

class SimpleFC(nn.Module):
    ''' Simple FC network that takes in kinematics of a star (e.g. ra, dec, parallax, pmra, pmdec)
        and classify whether a star is accreted or formed in situ with the MW
    '''
    def __init__(self, input_dims):
        ''' initialize NN '''
        super().__init__()
        self.name = 'Simple FC'

        self.fc = nn.Sequential(
            FCBlock(input_dims, 32),
            FCBlock(32, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        ''' Forward propagate x '''
        return self.fc(x)

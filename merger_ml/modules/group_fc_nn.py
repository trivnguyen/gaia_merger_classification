
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

class GroupFCNetwork(nn.Module):
    ''' FC network that takes in a group of points (each with features)
        and predicts the label of the first point '''
    def __init__(self, input_dim, perm_invar=True, n_points=None):
        super().__init__()

        self.input_dim = input_dim
        self.perm_invar = perm_invar
        if not perm_invar and n_points is None:
            raise ValueError('n_points must not be None if perm_invar is False')

        # FC_x0 for the primary star
        self.fc1_x0 = FCBlock(input_dim, 16)
        self.fc2_x0 = FCBlock(16, 32)

        # FC_gr for all stars
        if perm_invar:
            self.fc1_gr = FCBlock(input_dim, 16)
            self.fc1_gr_out_features = self.fc1_gr.fc[0].out_features
        else:
            self.fc1_gr = FCBlock(input_dim * n_points, 16)
        self.fc2_gr = FCBlock(16, 32)

        # FC layers for combined output of FC_x0 and FC_gr
        fc1_in_features = (
            self.fc2_x0.fc[0].out_features +
            self.fc2_gr.fc[0].out_features
        )
        self.fc1 = FCBlock(fc1_in_features, 64)
        self.fc2 = nn.Linear(64, 1)

    def _forward_group(self, x):
        ''' forward propagation for group of points '''
        if self.perm_invar:
            out_gr = self.fc1_gr(x.view(-1, self.input_dim))
            out_gr = out_gr.view(-1, x.size(1), self.fc1_gr_out_features)
            out_gr = out_gr.mean(1)
            out_gr = self.fc2_gr(out_gr)
        else:
            out_gr = self.fc1_gr(x.view(x.size(0), -1))
            out_gr = self.fc2_gr(out_gr)

        return out_gr

    def forward(self, x):
        ''' forward propagation function
        - shape of x: (N, S, D) where N is the batch size,
            S is the number of points, and D is the number of features
        - each star is put through some FC layers, the mean of the
            output features over all stars are computed to ensure
            permutation invariance
        - the first star is put through some additional FC layers
        - the features of the first star are concatenated with the
            mean of the features over all stars, and then put through
            some additional FC layers for classification
        '''

        # forward prop for primary star
        x0 = x[:, 0]
        out0 = self.fc1_x0(x0)
        out0 = self.fc2_x0(out0)

        # forward prop for all stars
        out_gr = self.fc1_gr(x.view(-1, self.input_dim))
        out_gr = out_gr.view(-1, x.size(1), self.fc1_gr_out_features)
        out_gr = out_gr.mean(1)
        out_gr = self.fc2_gr(out_gr)

        # combine output features of x0 FC and group FC
        out = torch.cat([out0, out_gr], 1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

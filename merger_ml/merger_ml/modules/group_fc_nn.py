
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
    def __init__(self, input_dim):
        super().__init__()
        
        # FC_x0 for the primary star
        self.fc1_x0 = FCBlock(input_dim, 16)
        self.fc2_x0 = FCBlock(16, 32)
        
        # FC_gr for all stars
        self.fc1_gr = FCBlock(input_dim, 16)
        self.fc2_gr = FCBlock(16, 32)
        self.fc1_gr_out_features = self.fc1_gr.fc[0].out_features
        
        # FC layers for combined output of FC_x0 and FC_gr 
        fc1_in_features = (
            self.fc2_x0.fc[0].out_features + 
            self.fc2_gr.fc[0].out_features
        )
        self.fc1 = FCBlock(fc1_in_features, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        
        # forward prop for primary star
        x0 = x[:, 0]
        out0 = self.fc1_x0(x0)
        out0 = self.fc2_x0(out0)

        # forward prop for all stars
        out_gr = self.fc1_gr(x.view(-1, input_dim))
        out_gr = out_gr.view(-1, x.size(1), self.fc1_gr_out_features)
        out_gr = out_gr.mean(1)
        out_gr = self.fc2_gr(out_gr)
        
        # combine output features of x0 FC and group FC
        out = torch.cat([out0, out_gr], 1)        
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out

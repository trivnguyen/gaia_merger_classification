

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl


class FCBlock(nn.Module):
    ''' Convienient FC block '''
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        return self.fc(x)

class FCClassifier(pl.LightningModule):
    ''' Fully-connected Classifier '''

    def __init__(self, in_dim=1, out_dim=1, num_layers=2, hidden_dim=128,
                 dropout=0., init_weights=False, lr_scheduler=False, extra_hparams={}):
        super().__init__()
        self.save_hyperparameters()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.init_weights = init_weights
        self.pos_weight = extra_hparams.get('pos_weight')
        if self.pos_weight is not None:
            self.pos_weight = torch.tensor(self.pos_weight, dtype=torch.float)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.lr_scheduler = lr_scheduler
        self.extra_hparams = extra_hparams

        # Create hidden layers
        layers = []
        layers.append(FCBlock(in_dim, hidden_dim, dropout=dropout))   # input layer
        for i in range(num_layers - 1):
            layers.append(FCBlock(hidden_dim, hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))  # output layer
        self.fc = nn.Sequential(*layers)

        # If enable, xavier initialize weight
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        ''' Initialize weight '''
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        ''' Forward propagate x '''
        return self.fc(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.extra_hparams.get('lr', '1e-3'))
        if not self.lr_scheduler:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
                    'monitor': 'train_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1, self.in_dim)
        yhat = self(x)
        loss = self.criterion(yhat, y)
        pred = (yhat > 0).float()
        acc = (pred == y).float().mean()
        self.log_dict({'train_loss': loss, 'train_acc': acc},
                      on_step=False, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, self.in_dim)
        yhat = self(x)
        loss = self.criterion(yhat, y)
        pred = (yhat > 0).float()
        acc = (pred == y).float().mean()
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_step=False, on_epoch=True, batch_size=len(x))
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, self.in_dim)
        yhat = self(x)
        loss = self.criterion(yhat, y)
        pred = (yhat > 0).float()
        acc = (pred == y).float().mean()
        self.log_dict({'test_loss': loss, 'test_acc': acc},
                      on_step=False, on_epoch=True, batch_size=len(x))

    def predict_step(self, predict_batch, batch_idx, dataloader_idx=0):
        if len(predict_batch) > 1:
            x, y = predict_batch
        else:
            x = predict_batch[0]
            y = None
        x = x.view(-1, self.in_dim)
        yhat = self(x)
        if y is not None:
            loss = self.criterion(yhat, y)
            return yhat, y, loss
        else:
            return yhat

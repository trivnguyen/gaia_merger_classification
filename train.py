#!/bin/bash python

import os
import sys
import h5py
import argparse
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, stream=sys.stdout)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# import customized library
from merger_ml import classifier, classifier_logger, data_utils

# define global variables
NUM_WORKERS = 4    # number of workers for PyTorch DataLoader
BATCH_SIZE = 64
MAX_EPOCHS = 5

# check if GPU is available, if not use CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
    PIN_MEMORY = True    # for DataLoader
else:
    DEVICE = 'cpu'
    PIN_MEMORY = False    # for DataLoader


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', required=True,
                        help='path to input directory')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='path to output directory')
    parser.add_argument('--store-val-output', action='store_true',
                        help='Enable to store output of validation set')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging mode')

    return parser.parse_args()


if __name__ == '__main__':

    # parse command line argument
    params = parse_cmd()
    device = torch.device(DEVICE)

    # read in training and validation dataset
    # because the dataset is too big to fit into memory, we customize our Dataset object
    # to read in only one file at once
    train_dataset = None
    val_dataset = None

    # DEBUG ONLY: fake dataset for debugging purposes
    if params.debug:
        logging.info('Debugging mode')
        x_train = np.random.rand(1024, 5)
        y_train = np.random.randint(2, size=(1024, 1))
        x_val = np.random.rand(128, 5)
        y_val = np.random.randint(2, size=(128, 1))
        train_dataset = list(zip(x_train, y_train))
        val_dataset = list(zip(x_val, y_val))

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    logging.info('Number of training samples: {:d}'.format(n_train))
    logging.info('Number of validation samples: {:d}'.format(n_val))

    # we'll use PyTorch Dataloader to manage dataset (e.g. shuffle, batching, etc.)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    n_batch_total = len(train_loader)

    # initialize a NN classifier
    net = classifier.SimpleClassifier(5)
    net.to(device)    # move NN to GPU if enabled

    # initialize gradient descent optimizer, loss function,
    optimizer = optim.Adam(net.parameters(), lr=1e-3)   # use ADAM optimizer with default LR
    criterion = nn.BCEWithLogitsLoss(reduction='sum')    # Sigmoid + BCE loss

    # Start training
    logging.info('Batch size: {:d}'.format(BATCH_SIZE))
    logging.info('Max epochs: {:d}'.format(MAX_EPOCHS))

    # logger object for bookkeeping
    train_log = classifier_logger.ClassifierLogger(
        out_dir=params.out_dir, metrics=['loss'])

    for epoch in range(MAX_EPOCHS):

        # Training loop
        net.train()    # switch NN to training mode
        train_loss = 0.
        for i, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()    # reset optimizer gradient

            # convert data into the correct tensor type
            xb = xb.float().to(device)
            yb = yb.float().to(device)

            yhatb = net(xb)    # forward pass
            loss = criterion(yhatb, yb)    # calculate loss
            loss.backward()    # backward pass
            optimizer.step()     # gradient descent
            train_loss += loss.item()    # update training loss
    #     scheduler.step()  # update LR scheduler

        # Evaluation loop
        net.eval()    # switch NN to evaluation mode
        val_loss = 0.
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                # convert data into the correct tensor type
                xb = xb.float().to(device)
                yb = yb.float().to(device)

                yhatb = net(xb)    # forward pass
                loss = criterion(yhatb, yb)    # calculate loss
                val_loss += loss.item()    # update training loss

                # store output of validation set if enabled
                if params.store_val_output:
                    train_log.update_predict(yhatb, yb)

        # logging at the end of each epoch
        # store average loss per sample
        train_loss /= n_train
        val_loss /= n_val
        train_log.update_metric(
            'loss', train_loss, epoch, test_metric=val_loss,
            n_batch_total=n_batch_total)

        # store metric and model
        train_log.log_metric()
        train_log.save_model(net, epoch)
        train_log.save_optimizer(optimizer, epoch)

        # store output of validation set if enabled
        if params.store_val_output:
            train_log.save_predict(epoch)

        # print out status
        train_log.display_status(
            'loss', train_loss, epoch, test_metric=val_loss,
            max_epochs=MAX_EPOCHS, n_batch_total=n_batch_total)

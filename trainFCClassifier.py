#!/bin/bash python

import os
import sys
import h5py
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from merger_ml import training_logger, utils
from merger_ml.modules import fc_nn

# define global constant
INPUT_KEY = {
    'ACTION': ('Jr', 'Jphi', 'Jz'),
    'ACTION_FEH': ('Jr', 'Jphi', 'Jz', 'feh'),
}

# check if GPU is available, if not use CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
    PIN_MEMORY = True    # for DataLoader
else:
    DEVICE = 'cpu'
    PIN_MEMORY = False    # for DataLoader

def preprocess(input_key, FLAGS):
    ''' Read in preprocess file and return preprocessing function based on input key '''

    preprocess_file = os.path.join(FLAGS.input_dir, 'preprocess.json')
    if not os.path.exists(preprocess_file):
        raise FileNotFoundError('preprocess file not found in input_dir')

    with open(preprocess_file, 'r') as f:
        preprocess_dict = json.load(f)

    # Get mean and standard deviation of each keys and write to output dir
    mean = [preprocess_dict[k]['mean'] for k in input_key]
    stdv = [preprocess_dict[k]['stdv'] for k in input_key]
    with open(os.path.join(FLAGS.out_dir, 'preprocess.json'), 'w') as f:
        json.dump({'mean': mean, 'stdv': stdv}, f, indent=4)

    # define transformation function
    # in this case transform is a standard scaler
    def transform(x):
        return (x - mean) / stdv
    return transform

def set_logger():
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add streaming handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def parse_cmd():
    parser = argparse.ArgumentParser()

    # io args
    parser.add_argument('-i', '--input-dir', required=True,
                        help='path to input directory.')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='path to output directory.')
    parser.add_argument(
        '-k', '--input-key', required=False, default=['5D', ], nargs='+',
        help='List of keys of input features. If either 5D or 6D, will use default key sets.')
    parser.add_argument('--store-val-output', action='store_true', required=False,
                        help='Enable to store output of validation set.')

    # nn args
    parser.add_argument(
        '-H', '--hidden-layers', required=False, default=[32, 64, 128], type=int, nargs='+',
        help='List of hidden layer sizes for FC network')

    # training args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '-e', '--max-epochs', required=False, type=int, default=20,
        help='Maximum number of epochs. Stop training automatically if exceeds. ' +
                'Default is 10.')
    parser.add_argument(
        '-lr', '--learning-rate', dest='lr', required=False, type=float, default=1e-3,
        help='Learning rate of ADAM. Default is 1e-3')
    parser.add_argument(
        '--use-lr-scheduler', action='store_true', required=False,
        help='Enable to use LR scheduler. LR scheduler is set to reduced on plateau.')
    parser.add_argument(
        '-w', '--use-weights', action='store_true', required=False,
        help='Enable to use loss weights to account for imbalance training set.')
    parser.add_argument(
        '--pos-weight-factor', required=False, type=float, default=5,
        help='If loss weights are enable, divide label-1 weight by this factor.')
    parser.add_argument(
        '--num-workers', required=False, type=int, default=1,
        help='Number of workers for Pytorch DataLoader.')

    return parser.parse_args()


if __name__ == '__main__':

    # parse command line argument
    FLAGS = parse_cmd()
    device = torch.device(DEVICE)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # set up logger
    logger = set_logger()

    # define the input features for the network
    if (len(FLAGS.input_key) == 1) and (FLAGS.input_key[0].upper() in INPUT_KEY.keys()):
        input_key = INPUT_KEY[FLAGS.input_key[0].upper()]
    else:
        input_key = FLAGS.input_key
    logger.info('Input key: {}'.format(input_key))

    # define preprocess transformation function
    transform  = preprocess(input_key, FLAGS)

    # read in training and validation dataset
    # because the dataset is too big to fit into memory,
    # we customize our Dataset object to read in only one file at once
    train_dataset = utils.dataset.Dataset(
        os.path.join(FLAGS.input_dir, 'train'),
        input_key=input_key, label_key='labels', transform=transform,)
    val_dataset = utils.dataset.Dataset(
        os.path.join(FLAGS.input_dir, 'val'),
        input_key=input_key, label_key='labels', transform=transform,)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    input_dims = train_dataset.input_dims
    logging.info('Number of training samples  : {:,d}'.format(n_train))
    logging.info('Number of validation samples: {:,d}'.format(n_val))

    # we'll use PyTorch Dataloader to manage dataset (e.g. shuffle, batching, etc.)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              pin_memory=PIN_MEMORY, num_workers=FLAGS.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size,
                            pin_memory=PIN_MEMORY, num_workers=FLAGS.num_workers)
    n_batch_total = len(train_loader)

    # initialize a NN classifier
    net = fc_nn.FCNetwork(input_dims, hidden_layers=FLAGS.hidden_layers)
    net.to(device)    # move NN to GPU if enabled

    # use ADAM optimizer with default LR
    optimizer = optim.Adam(net.parameters(), lr=FLAGS.lr)
    if FLAGS.use_lr_scheduler:
        logging.info('Use LR scheduler')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = None

    # use binary cross entropy loss function: Sigmoid + BCE loss
    # weights to account for imbalanced dataset
    if FLAGS.use_weights:
        with open(os.path.join(FLAGS.input_dir, 'properties.json'), 'r') as f:
            properties = json.load(f)
        w_0 = 1. / properties['train']['f_0'] # w_0 = N_total / N_1
        # w_1 = N_total / (N_1 * pos_weight_factor)
        w_1 = 1. / properties['train']['f_1']  / FLAGS.pos_weight_factor
        pos_weight = torch.as_tensor(w_1 / w_0).to(device)
        logging.info('Use imbalance weight: {:.4f}'.format(pos_weight.item()))
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Start training
    logging.info('Batch size: {:d}'.format(FLAGS.batch_size))
    logging.info('Max epochs: {:d}'.format(FLAGS.max_epochs))

    # logger object for bookkeeping
    train_log = training_logger.Logger(
        out_dir=FLAGS.out_dir, metrics=['loss'])

    for epoch in range(FLAGS.max_epochs):

        # Training loop
        net.train()    # switch NN to training mode
        train_loss = 0.
        for i, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()    # reset optimizer gradient

            # convert data into the correct tensor type
            xb = xb.float().view(-1, input_dims).to(device)
            yb = yb.float().to(device)

            yhatb = net(xb)    # forward pass
            loss = criterion(yhatb, yb)    # calculate loss
            loss.backward()    # backward pass
            optimizer.step()     # gradient descent
            train_loss += loss.item() * len(xb)    # update training loss
        train_loss /= n_train
        # update LR scheduler
        if scheduler is not None:
            scheduler.step(train_loss)

        # Evaluation loop
        net.eval()    # switch NN to evaluation mode
        val_loss = 0.
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                # convert data into the correct tensor type
                xb = xb.float().view(-1, input_dims).to(device)
                yb = yb.float().to(device)

                yhatb = net(xb)    # forward pass
                loss = criterion(yhatb, yb)    # calculate loss
                val_loss += loss.item() * len(xb)    # update training loss

                # store output of validation set if enabled
                if FLAGS.store_val_output:
                    train_log.update_predict(yhatb, yb)
        val_loss /= n_val

        # logging at the end of each epoch
        # store average loss per sample
        train_log.update_metric(
            'loss', train_loss, epoch, test_metric=val_loss,
            n_batch_total=n_batch_total)

        # store metric and model
        train_log.log_metric()
        train_log.save_model(net, epoch)
        train_log.save_optimizer(optimizer, epoch)

        # store output of validation set if enabled
        if FLAGS.store_val_output:
            train_log.save_predict(epoch)

        # print out status
        train_log.display_status(
            'loss', train_loss, epoch, test_metric=val_loss,
            max_epochs=FLAGS.max_epochs, n_batch_total=n_batch_total)

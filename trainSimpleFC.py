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

from merger_ml import classifier_logger, data_utils
from merger_ml.modules import simple_fc

# define global constant
NUM_WORKERS = 1    # number of workers for PyTorch DataLoader
INPUT_KEY = {
    '5D': ('l', 'b', 'parallax', 'pmra', 'pmdec'),
    '6D': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity')
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
    
    mean = [preprocess_dict[k]['mean'] for k in input_key]
    std = [preprocess_dict[k]['stdv'] for k in input_key]    

    # define transformation function
    # in this case transform is a standard scaler
    def transform(x):    
        return (x - mean) / std
    return transform
    
def parse_cmd():
    parser = argparse.ArgumentParser()

    # io args
    parser.add_argument('-i', '--input-dir', required=True,
                        help='path to input directory')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='path to output directory')
    parser.add_argument('-l', '--log-file', required=False,
                        help='if given, log output to file in output directory')
    parser.add_argument('-t','--input-type', choices=('5D', '6D'), type=str.upper, default='5D',
                        help='Input type for NN. Default to 5D. Choices: 5D, 6D')
    parser.add_argument('--store-val-output', action='store_true', required=False,
                        help='Enable to store output of validation set')
    # training args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '-e', '--max-epochs', required=False, type=int, default=20,
        help='Maximum number of epochs. Stop training automatically if exceeds. ' + 
                'Default to 10')
    parser.add_argument(
        '-w', '--use-weights', action='store_true', required=False,
        help='Enable to use loss weights to account for imbalance training set')
    
    # debug args
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging mode')
    parser.add_argument('--n-max-files', required=False, type=int,
                        help='Maximum number of files to read. For debugging purposes only')
    
    return parser.parse_args()


if __name__ == '__main__':

    # parse command line argument
    FLAGS = parse_cmd()
    device = torch.device(DEVICE)

    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if FLAGS.log_file is not None:
        if os.path.isabs(FLAGS.log_file):
            log_file = FLAGS.log_file
        else:
            log_file = os.path.join(FLAGS.out_dir, FLAGS.log_file)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # read in training and validation dataset
    # because the dataset is too big to fit into memory, 
    # we customize our Dataset object to read in only one file at once
    input_key = INPUT_KEY[FLAGS.input_type]
    
    # define preprocess transformation function
    transform = preprocess(input_key, FLAGS)
    
    if FLAGS.n_max_files is not None:
        if FLAGS.n_max_files <= 0:
            raise ValueError('Flag n_max_file must be greater than 0')
        logger.warn('Flag n_max_files={:d} is given.'.format(FLAGS.n_max_files))
    
    # DEBUG ONLY: fake dataset for debugging purposes
    if not FLAGS.debug:
        train_dataset = data_utils.Dataset(
            os.path.join(FLAGS.input_dir, 'train'), 
            input_key=input_key, label_key='labels', transform=transform, 
            n_max_file=FLAGS.n_max_files)
        val_dataset = data_utils.Dataset(
            os.path.join(FLAGS.input_dir, 'val'), 
            input_key=input_key, label_key='labels', transform=transform,
            n_max_file=FLAGS.n_max_files)
        input_dims = train_dataset.input_dims
    else:
        logging.info('Debugging mode')
        x_train = np.random.rand(1024, 5)
        y_train = np.random.randint(2, size=(1024, 1))
        x_val = np.random.rand(128, 5)
        y_val = np.random.randint(2, size=(128, 1))
        train_dataset = list(zip(x_train, y_train))
        val_dataset = list(zip(x_val, y_val))
        input_dims = 5
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    logging.info('Number of training samples: {:d}'.format(n_train))
    logging.info('Number of validation samples: {:d}'.format(n_val))

    # we'll use PyTorch Dataloader to manage dataset (e.g. shuffle, batching, etc.)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size,
                            pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    n_batch_total = len(train_loader)

    # initialize a NN classifier
    net = simple_fc.SimpleFC(input_dims)
    net.to(device)    # move NN to GPU if enabled

    # use ADAM optimizer with default LR
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # use binary cross entropy loss function: Sigmoid + BCE loss 
    # weights to account for imbalanced dataset
    if FLAGS.weights:
        with open(os.path.join(out_dir_base, 'properties.json'), 'r') as f:
            properties = json.load(f)
        w_insitu = 1. / properties['train']['f_insitu'] # w_insitu = N_total / N_insitu
        # w_accreted = N_total / (N_accreted * 5)
        w_accreted = 1. / properties['train']['f_accreted']  / 5
        pos_weight = w_accreted / w_insitu
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitLoss(reduction='sum', pos_weight=pos_weight)

    # Start training
    logging.info('Batch size: {:d}'.format(FLAGS.batch_size))
    logging.info('Max epochs: {:d}'.format(FLAGS.max_epochs))

    # logger object for bookkeeping
    train_log = classifier_logger.ClassifierLogger(
        out_dir=FLAGS.out_dir, metrics=['loss'])

    for epoch in range(FLAGS.max_epochs):

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
                if FLAGS.store_val_output:
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
        if FLAGS.store_val_output:
            train_log.save_predict(epoch)

        # print out status
        train_log.display_status(
            'loss', train_loss, epoch, test_metric=val_loss,
            max_epochs=FLAGS.max_epochs, n_batch_total=n_batch_total)

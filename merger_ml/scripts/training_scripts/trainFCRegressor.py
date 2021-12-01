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

from merger_ml import training_logger, data_utils, utils
from merger_ml.modules import simple_fc

# define global constant
INPUT_KEY = {
    '5D': ('l', 'b', 'parallax', 'pmra', 'pmdec'),
    '6D': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity'),
    '5D_FEH': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'feh'),
    '6D_FEH': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity', 'feh'),
    '5D_PHOT': ('l', 'b', 'parallax', 'pmra', 'pmdec',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'),
    '6D_PHOT': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'),
    '5D_FEH_PHOT': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'feh',
                    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'),
    '6D_FEH_PHOT': ('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity', 'feh',
                    'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'),
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

def set_logger(FLAGS):
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # add streaming handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # add output log file handler
    if FLAGS.log_file is not None:
        if os.path.isabs(FLAGS.log_file):
            log_file = FLAGS.log_file
        else:
            log_file = os.path.join(FLAGS.out_dir, FLAGS.log_file)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def parse_cmd():
    parser = argparse.ArgumentParser()

    # io args
    parser.add_argument('-i', '--input-dir', required=True,
                        help='path to input directory.')
    parser.add_argument('-o', '--out-dir', required=True,
                        help='path to output directory.')
    parser.add_argument('-l', '--log-file', required=False,
                        help='if given, log output to file in output directory.')
    parser.add_argument('-L', '--label-key', required=False, default='labels',
                        help='Label key from dataset.')
    parser.add_argument(
        '-k', '--input-key', required=False, default=['5D', ], nargs='+',
        help='List of keys of input features. If either 5D or 6D, will use default key sets.')
    parser.add_argument('--store-val-output', action='store_true', required=False,
                        help='Enable to store output of validation set.')
    parser.add_argument(
        '--resume', required=False, action='store_true',
        help='Enable to resume training from the last checkpoint. Ovewrite all hyperparameters.')

    # nn args
    parser.add_argument(
        '-l1', '--hidden-layer-1', dest='l1', required=False, type=int, default=32,
        help='Dim of hidden layer 1')
    parser.add_argument(
        '-l2', '--hidden-layer-2', dest='l2', required=False, type=int, default=64,
        help='Dim of hidden layer 2')

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
        '-N', '--num-workers', required=False, type=int, default=1,
        help='Number of workers for Pytorch DataLoader.')

    # tuning args
    parser.add_argument(
        '-t', '--tuning', action='store_true', required=False,
        help='Enable to start tuning mode. ' +
            'Overwrite lr, l1, l2 with random values drawn from a fix distribution.')
    parser.add_argument(
        '-c', '--tuning-config', required=False,
        help='Config file in JSON format if tuning mode is enabled. ' +
            'If not give, use DEFAULT_CONFIG of module.'
    )

    # debug args
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging mode.')
    parser.add_argument('--n-max-files', required=False, type=int,
                        help='Maximum number of files to read. For debugging purposes only')

    return parser.parse_args()


if __name__ == '__main__':

    # parse command line argument
    FLAGS = parse_cmd()
    device = torch.device(DEVICE)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # set up logger
    logger = set_logger(FLAGS)

    # define the input features for the network
    if (len(FLAGS.input_key) == 1) and (FLAGS.input_key[0].upper() in INPUT_KEY.keys()):
        input_key = INPUT_KEY[FLAGS.input_key[0].upper()]
    else:
        input_key = FLAGS.input_key
    logger.info('Input key: {}'.format(input_key))
    logger.info('Label key: {}'.format(FLAGS.label_key))

    # define preprocess transformation function
    transform  = preprocess(input_key, FLAGS)

    if FLAGS.n_max_files is not None:
        if FLAGS.n_max_files <= 0:
            raise ValueError('Flag n_max_file must be greater than 0')
        logger.warning('Flag n_max_files={:d} is given.'.format(FLAGS.n_max_files))

    # tuning options
    if FLAGS.tuning:
        logger.info('Tuning option is enabled. Randomized hyperparameters.')

        # read in tuning config file
        if FLAGS.tuning_config is not None:
            logger.info('Read tuning config from {}'.format(FLAGS.tuning_config))
            with open(FLAGS.tuning_config, 'r') as f:
                tuning_config = json.load(f)
        else:
            logger.info('Use default tuning config')
            tuning_config = simple_fc.DEFAULT_CONFIG
        logger.info(tuning_config)

        # Sample and replace FLAGS parameters
        rvars =  utils.sample_tuning(tuning_config)
        temp = vars(FLAGS)
        for key, rvar in rvars.items():
            temp[key] = rvar
    else:
        rvars = {}
        rvars['lr'] = FLAGS.lr
        rvars['l1'] = FLAGS.l1
        rvars['l2'] = FLAGS.l2

    # Print out and write hyperparameters to file
    logger.info('Network and training parameters: ')
    for key, rvar in rvars.items():
        logger.info('- {}: {}'.format(key, rvar))
    with open(os.path.join(FLAGS.out_dir, 'hyperparams.json'), 'w') as f:
        json.dump(rvars, f, indent=4)

    # read in training and validation dataset
    # because the dataset is too big to fit into memory,
    # we customize our Dataset object to read in only one file at once
    train_dataset = data_utils.Dataset(
        os.path.join(FLAGS.input_dir, 'train'),
        input_key=input_key, label_key=FLAGS.label_key, transform=transform,
        n_max_file=FLAGS.n_max_files)
    val_dataset = data_utils.Dataset(
        os.path.join(FLAGS.input_dir, 'val'),
        input_key=input_key, label_key=FLAGS.label_key, transform=transform,
        n_max_file=FLAGS.n_max_files)

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
    net = simple_fc.SimpleFC(input_dims, l1=FLAGS.l1, l2=FLAGS.l2)
    net.to(device)    # move NN to GPU if enabled

    # ADAM optimizer with default LR
    optimizer = optim.Adam(net.parameters(), lr=FLAGS.lr)
    if FLAGS.use_lr_scheduler:
        logging.info('Use LR scheduler')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        scheduler = None

    # Loss function is MSE loss
    # criterion = nn.MSELoss()
    def criterion(predict, target, weight=None):
        ''' MSE Loss with weight '''
        if weight is not None:
            loss = torch.mean((predict - target)**2 * weight)
        else:
            loss = torch.mean((predict - target)**2)
        return loss

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
            xb = xb.float().to(device)
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
                xb = xb.float().to(device)
                yb = yb.float().to(device)

                yhatb = net(xb)    # forward pass
                loss = criterion(yhatb, yb)    # calculate loss
                val_loss += loss.item() * len(xb)   # update training loss

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

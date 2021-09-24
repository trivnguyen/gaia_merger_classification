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

from merger_ml import data_utils
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
    parser.add_argument('-o', '--out-file', required=True,
                        help='path to write output file in HDF5 format')
    parser.add_argument('-s', '--state', required=True,
                        help='path to Pytorch state dict')
    parser.add_argument('-l', '--log-file', required=False,
                        help='if given, log output to file in output directory')
    parser.add_argument(
            '-d', '--dataset', required=False, default='test', choices=('train', 'val', 'test'),
            help='Choose which dataset to use. Must be "train", "val", or "test".')
    parser.add_argument(
        '-k', '--input-key', required=False, default='5D', nargs='+',
        help='List of keys of input features. If either 5D or 6D, will use default key sets.')   
    parser.add_argument('--store-val-output', action='store_true', required=False,
                        help='Enable to store output of validation set')
    
    # nn args
    parser.add_argument(
        '-l1', '--hidden-layer-1', dest='l1', required=False, type=int, default=32,
        help='Dim of hidden layer 1')
    parser.add_argument(
        '-l2', '--hidden-layer-2', dest='l2', required=False, type=int, default=64,
        help='Dim of hidden layer 2')
    parser.add_argument('-c', '--config', required=False,
        help='Hyperparameters config in JSON format. If given, will overwrite l1 and l2.')
            
    # validation args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '--compute-loss', action='store_true', required=False,
        help='Enable to compute loss function')
    parser.add_argument(
        '-w', '--use-weights', action='store_true', required=False,
        help='Enable to compute the loss function with weights. Works only if compute_loss is enabled'
    ) 
    parser.add_argument(
        '--pos-weight-factor', required=False, type=float, default=5,
        help='If loss weights are enable, divide label-1 weight by this factor.')
    parser.add_argument(
        '-N', '--num-workers', required=False, type=int, default=1,
        help='Number of workers for Pytorch DataLoader'
    )

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
            
    # define the input features for the network
    if (len(FLAGS.input_key) == 1) and (FLAGS.input_key[0].upper() in INPUT_KEY.keys()):
        input_key = INPUT_KEY[FLAGS.input_key[0].upper()]    
    else:
        input_key = FLAGS.input_key
    if FLAGS.compute_loss:
        label_key = 'labels'
    else:
        label_key = None
    logger.info('Input key: {}'.format(input_key))
    logger.info('Label key: {}'.format(label_key))
    
    # define preprocess transformation function
    transform = preprocess(input_key, FLAGS)
    
    if FLAGS.n_max_files is not None:
        if FLAGS.n_max_files <= 0:
            raise ValueError('Flag n_max_file must be greater than 0')
        logger.warning('Flag n_max_files={:d} is given.'.format(FLAGS.n_max_files))
        
    # read in training and validation dataset
    # because the dataset is too big to fit into memory, 
    # we customize our Dataset object to read in only one file at once
    logger.info('Inference on {} dataset'.format(FLAGS.dataset))
    test_dataset = data_utils.Dataset(
        os.path.join(FLAGS.input_dir, FLAGS.dataset), input_key=input_key, label_key=label_key,
        transform=transform, n_max_file=FLAGS.n_max_files,
    )
    input_dims = test_dataset.input_dims    
    n_test = len(test_dataset)

    logging.info('Number of testing samples  : {:,d}'.format(n_test))

    # we'll use PyTorch Dataloader to manage dataset (e.g. shuffle, batching, etc.)
    test_loader = DataLoader(
        test_dataset, batch_size=FLAGS.batch_size,
        pin_memory=PIN_MEMORY, num_workers=FLAGS.num_workers)
    n_batch_total = len(test_loader)

    # initialize a NN classifier
    # if given, read in hyperparameters config
    if FLAGS.config is not None:
        with open(FLAGS.config, 'r') as f:
            hyperparams = json.load(f)
        l1 = hyperparams['l1']
        l2 = hyperparams['l2']
    else:
        l1 = FLAGS.l1
        l2 = FLAGS.l2
    
    net = simple_fc.SimpleFC(input_dims, l1=l1, l2=l2)
    net.load_state_dict(torch.load(FLAGS.state, map_location='cpu'))
    net.to(device)    # move NN to GPU if enabled
    net.eval()    # switch to evaluation mode
    
    # use binary cross entropy loss function: Sigmoid + BCE loss 
    # weights to account for imbalanced dataset
    if FLAGS.compute_loss:
        if FLAGS.use_weights:
            with open(os.path.join(FLAGS.input_dir, 'properties.json'), 'r') as f:
                properties = json.load(f)
            w_0 = 1. / properties['train']['f_0'] # w_1 = N_total / N_0
            # w_1 = N_total / (N_1 * pos_weight_factor)
            w_1 = 1. / properties['train']['f_1']  / FLAGS.pos_weight_factor   
            pos_weight = torch.as_tensor(w_1 / w_0).to(device)
            logging.info('Use imbalance weight: {:.4f}'.format(pos_weight.item()))
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

    # Start testing
    logging.info('Batch size: {:d}'.format(FLAGS.batch_size))
    
    test_loss = 0
    target = []
    predict = []
    with torch.no_grad():
        for i, datab in enumerate(test_loader):
            
            # forward pass
            xb = datab[0].float().to(device)
            yhatb = net(xb)    # forward pass
            
            if FLAGS.compute_loss:
                yb = datab[1].float().to(device)
                loss = criterion(yhatb, yb)
                test_loss += loss.item()    # update test loss
                
                target.append(yb.cpu().numpy())
            predict.append(yhatb.cpu().numpy())
                
    predict = np.concatenate(predict)
    target = np.concatenate(target)
    test_loss /= len(test_dataset)
                
    # write output file
    logging.info('Writing output file to {}'.format(FLAGS.out_file))
    with h5py.File(FLAGS.out_file, 'w') as f:
        f.create_dataset('predict', data=predict)
        if FLAGS.compute_loss:
            f.attrs['loss'] = test_loss
            f.create_dataset('target', data=target)

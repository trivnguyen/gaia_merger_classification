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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from merger_ml import classifier_logger, data_utils
from merger_ml.modules import simple_vae

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

    # nn args
    parser.add_argument(
        '-z', '--latent-dim', required=False, type=int, default=2,
        help='Dim of latent dimension')
    parser.add_argument(
        '-lencode', '--hidden-layer-encode', dest='lencode', required=False, type=int, default=32,
        help='Dim of hidden layer of Encoder')
    parser.add_argument(
        '-ldecode', '--hidden-layer-decode', dest='ldecode', required=False, type=int, default=32,
        help='Dim of hidden layer of Decoder')
    parser.add_argument('-c', '--config', required=False,
        help='Hyperparameters config in JSON format. If given, will overwrite lencode and ldecode')

    # validation args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '--compute-loss', action='store_true', required=False,
        help='Enable to compute loss function')
    parser.add_argument(
        '-N', '--num-workers', required=False, type=int, default=1,
        help='Number of workers for Pytorch DataLoader.')
   
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
        transform=transform, n_max_file=FLAGS.n_max_files, shuffle=False,
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
        lencode = hyperparams['lencode']
        ldecode = hyperparams['ldecode']
    else:
        lencode = FLAGS.lencode
        ldecode = FLAGS.ldecode
    
    net = simple_vae.SimpleVAE(
        input_dims, FLAGS.latent_dim, lencode=lencode, ldecode=ldecode)
    net.load_state_dict(torch.load(FLAGS.state, map_location='cpu'))
    net.to(device)    # move NN to GPU if enabled
    net.eval()    # switch to evaluation mode
    
    # Start testing
    logging.info('Batch size: {:d}'.format(FLAGS.batch_size))
    
    latents, recons = [], []
    with torch.no_grad():
        for i, xb in enumerate(test_loader):
            
            # forward pass
            xb = xb.float().to(device)
            xb_recon, z, mu, logvar = net(xb)
            
            # add to list
            latents.append(z.cpu().numpy())
            recons.append(xb_recon.cpu().numpy())

    latents = np.concatenate(latents)
    recons = np.concatenate(recons)

    # write output file
    logging.info('Writing output file to {}'.format(FLAGS.out_file))
    with h5py.File(FLAGS.out_file, 'w') as f:
        f.create_dataset('latents', data=latents)
        f.create_dataset('recons', data=recons)


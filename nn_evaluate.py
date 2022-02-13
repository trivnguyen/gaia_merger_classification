#!/bin/bash python

import os
import sys
import h5py
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from merger_ml import utils
from merger_ml.modules import fc_nn

def preprocess(input_key, FLAGS):
    ''' Read in preprocess file and return preprocessing function based on input key '''

    # Open an input file and read in the meta data
    fn = os.path.join(FLAGS.input_dir, 'n00.hdf5')
    meta = {}
    with h5py.File(fn, 'r') as f:
        meta_group = f['meta']
        for prop in meta_group:
            meta[prop] = dict(meta_group[prop].attrs)

    # Get mean and standard deviation of each keys and write to output dir
    mean = np.array([meta[k]['mean'] for k in input_key], dtype=np.float32)
    stdv = np.array([meta[k]['stdv'] for k in input_key], dtype=np.float32)

    # define transformation function
    # in this case transform is a standard scaler
    def transform(x):
        return (x - mean) / stdv
    return transform, mean, stdv

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
    parser.add_argument('-o', '--output', required=True,
                        help='path to output evaluation file.')
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='path to checkpoint with pretrained model')

    # dataset and preprocessing args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '--use-data-preprocess', action='store_true',
        help='Enable to use preprocess features from evaluation data instead of training data')

    # gpu/cpu arguments
    parser.add_argument(
        '--accelerator', required=False, default='auto',
        help='Type of accelerators to use. Default will use GPUs if the machine has GPUs')
    parser.add_argument(
        '--devices', required=False, type=int, default=1,
        help='Number of cpu/gpu devices to use')
    parser.add_argument(
        '--num-workers', required=False, type=int, default=1,
        help='Number of workers in DataLoader')

    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line argument
    FLAGS = parse_cmd()
    logger = set_logger()

    # Load checkpoint
    model = fc_nn.FCClassifier.load_from_checkpoint(FLAGS.checkpoint)
    model.eval()
    extra_hparams = model.extra_hparams

    # Read in dataset
    if FLAGS.use_data_preprocess:
        logger.info('Using preprocessing features from evaluation data')
        transform, mean, stdv = preprocess(extra_hparams['key'], FLAGS)
    else:
        logger.info('Using preprocessing features from training data')
        def transform(x):
            mean = np.float32(extra_hparams['preprocess']['mean'])
            stdv = np.float32(extra_hparams['preprocess']['stdv'])
            return (x - mean) / stdv
    dataset = utils.dataset.Dataset(
        FLAGS.input_dir, key=extra_hparams['key'], target_key='labels',
        transform=transform)
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    dataloader = DataLoader(
        dataset, batch_size=FLAGS.batch_size,
        pin_memory=pin_memory, num_workers=FLAGS.num_workers,)

    logger.info('Reading dataset from {}'.format(FLAGS.input_dir))
    logger.info('Number of evaluation samples: {:,d}'.format(len(dataset)))

    # Start evaluating
    logger.info('Begin evaluating dataset')
    predict = []
    target = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.view(-1, model.in_dim)
            yhat = model(x)
            predict.append(yhat.cpu().numpy())
            target.append(y.cpu().numpy())
    predict = np.concatenate(predict)
    target = np.concatenate(target)

    # Write output to file
    logger.info('Write output to {}'.format(FLAGS.output))
    with h5py.File(FLAGS.output, 'w') as f:
        f.attrs.update({
            'checkpoint': os.path.abspath(FLAGS.checkpoint),
            'input_dir': os.path.abspath(FLAGS.input_dir),
            'use_data_preprocess': int(FLAGS.use_data_preprocess),
        })
        f.create_dataset('predict', data=predict)
        f.create_dataset('target', data=target)


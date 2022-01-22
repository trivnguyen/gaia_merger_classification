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

    # dataset args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')

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

    # weights to account for imbalanced dataset
    if FLAGS.use_weights:
        with open(os.path.join(FLAGS.input_dir, 'properties.json'), 'r') as f:
            properties = json.load(f)
        w_0 = 1. / properties['train']['f_0'] # w_0 = N_total / N_1
        # w_1 = N_total / (N_1 * pos_weight_factor)
        w_1 = 1. / properties['train']['f_1']  / FLAGS.pos_weight_factor
        pos_weight = w_1 / w_0
        logging.info('Use imbalance weight: {:.4f}'.format(pos_weight))
    else:
        pos_weight = None

    # Load checkpoint
    model = fc_nn.FCClassifier.load_from_checkpoint(FLAGS.checkpoint)
    model.eval()
    extra_hparams = model.extra_hparams

    # Read in dataset
    def transform(x):
        mean = extra_hparams['preprocess']['mean']
        stdv = extra_hparams['preprocess']['stdv']
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
            'input_dir': os.path.abspath(FLAGS.input_dir)
        })
        f.create_dataset('predict', data=predict)
        f.create_dataset('target', data=target)


#!/bin/bash python

import os
import sys
import json
import argparse
import logging
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from merger_ml import utils
from merger_ml.modules import fc_nn


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
    parser.add_argument('-o', '--out-dir', required=True,
                        help='path to output directory.')
    parser.add_argument('-k', '--input-key', required=True, nargs='+',
                        help='List of keys of input features.')
    parser.add_argument('--overwrite', action='store_true',
                        help='If enable, overwrite the previous driectory')
    parser.add_argument('--name', required=False,default='default',
                        help='Name of run')

    # nn args
    parser.add_argument(
        '-N', '--num-layers', required=False, default=3, type=int,
        help='Number of hidden layers')
    parser.add_argument(
        '-H', '--hidden-dim', required=False, default=128, type=int,
        help='Dimension of the hidden layers')
    parser.add_argument(
        '-D', '--dropout', required=False, default=0., type=float,
        help='Dropout rate')
    parser.add_argument(
        '-I', '--init-weights', action='store_true',
        help='enable to apply Xavier uniform init to weights')

    # training args
    parser.add_argument(
        '-b', '--batch-size', required=False, type=int, default=1000,
        help='Batch size. Default to 1000')
    parser.add_argument(
        '-e', '--max-epochs', required=False, type=int, default=20,
        help='Maximum number of epochs. Stop training automatically if exceeds')
    parser.add_argument(
        '-lr', '--learning-rate', dest='lr', required=False, type=float, default=1e-3,
        help='Learning rate of ADAM. Default is 1e-3')
    parser.add_argument(
        '--lr-scheduler', action='store_true', required=False,
        help='Enable to use LR scheduler. LR scheduler is set to reduced on plateau.')
    parser.add_argument(
        '-w', '--use-weights', action='store_true', required=False,
        help='Enable to use loss weights to account for imbalance training set.')
    parser.add_argument(
        '--pos-weight-factor', required=False, type=float, default=5,
        help='If loss weights are enable, divide label-1 weight by this factor.')

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

    # Read in training and validation set
    input_key = utils.io.get_input_key(FLAGS.input_key)
    transform, input_mean, input_stdv  = preprocess(input_key, FLAGS)
    train_dataset = utils.dataset.Dataset(
        os.path.join(FLAGS.input_dir, 'train'),
        key=input_key, target_key='labels', transform=transform,)
    val_dataset = utils.dataset.Dataset(
        os.path.join(FLAGS.input_dir, 'val'),
        key=input_key, target_key='labels', transform=transform,)
    logger.info('Number of training samples  : {:,d}'.format(len(train_dataset)))
    logger.info('Number of validation samples: {:,d}'.format(len(val_dataset)))

    # use PyTorch Dataloader to manage dataset (shuffle, batching, etc.)
    if torch.cuda.is_available() and FLAGS.accelerator != 'cpu':
        pin_memory = True
    else:
        pin_memory = False
    train_loader = DataLoader(
        train_dataset, batch_size=FLAGS.batch_size,
        pin_memory=pin_memory, num_workers=FLAGS.num_workers,)
    val_loader = DataLoader(
        val_dataset, batch_size=FLAGS.batch_size,
        pin_memory=pin_memory, num_workers=FLAGS.num_workers)

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

    # Create model
    model = fc_nn.FCClassifier(
        in_dim=len(input_key), out_dim=1, num_layers=FLAGS.num_layers,
        hidden_dim=FLAGS.hidden_dim, lr_scheduler=FLAGS.lr_scheduler,
        init_weights=FLAGS.init_weights, dropout=FLAGS.dropout,
        extra_hparams={
            'lr': FLAGS.lr,
            'pos_weight': pos_weight,
            'key': list(input_key),
            'preprocess': {
                'mean': input_mean,
                'stdv': input_stdv,
            }
        }
    )

    if FLAGS.overwrite and os.path.exists(FLAGS.out_dir):
        logger.info('Overwriting existing directory: {}'.format(FLAGS.out_dir))
        shutil.rmtree(FLAGS.out_dir)

    # Create trainer
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss", mode='min', filename="{epoch}-{val_loss:.4f}",
            save_top_k=3, save_weights_only=True),
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, mode='min', verbose=True)
    ]
    trainer_logger = CSVLogger(FLAGS.out_dir, name=FLAGS.name)
    trainer = pl.Trainer(
        default_root_dir=FLAGS.out_dir,
        accelerator=FLAGS.accelerator, devices=FLAGS.devices,
        max_epochs=FLAGS.max_epochs, callbacks=callbacks, logger=trainer_logger)

    # Start training
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


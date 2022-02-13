#!/bin/bash python

import os
import sys
import json
import h5py
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

    # Open an input file and read in the meta data
    fn = os.path.join(FLAGS.input_dir, 'train/n00.hdf5')
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

def get_weight(FLAGS):
    ''' Get imbalance weight '''
    weight, pos_weight = None, None
    if not FLAGS.use_weight:
        return weight, pos_weight

    fn = os.path.join(FLAGS.input_dir, 'train/n00.hdf5')
    with h5py.File(fn, 'r') as f:
        meta_group = f['meta']
        num_classes = f.attrs['n_classes']
        weight = [1./ meta_group['train'].attrs[f'f_{i}'] for i in range(num_classes)]
        if num_classes == 2:
            pos_weight = weight[1] / weight[0] / FLAGS.pos_weight_factor
            logging.info('Use imbalance weight: {:.4f}'.format(pos_weight))
    return weight, pos_weight

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
    parser.add_argument('-c', '--checkpoint', required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--overwrite', action='store_true',
                        help='If enable, overwrite the previous driectory')
    parser.add_argument('--name', required=False,default='default',
                        help='Name of the run')

    # transer learning args
    parser.add_argument('--train-first', action='store_true',
                        help='Enable to train the first layer. Not exclusive with --train-last')
    parser.add_argument('--train-last', action='store_true',
                        help='Enable to train the last layer. Not exclusive with --train-first')

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
        help='Enable to use LR scheduler. LR scheduler is set to reduced on plateau')
    parser.add_argument(
        '-w', '--use-weight', action='store_true', required=False,
        help='Enable to use loss weights to account for imbalance training set')
    parser.add_argument(
        '--pos-weight-factor', required=False, type=float, default=1,
        help='If loss weights are enable, divide label-1 weight by this factor')

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

    # throw some errors for invalid
    params = parser.parse_args()
    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line argument
    FLAGS = parse_cmd()
    logger = set_logger()

    if FLAGS.overwrite and os.path.exists(FLAGS.out_dir):
        logger.info('Overwriting existing directory: {}'.format(FLAGS.out_dir))
        shutil.rmtree(FLAGS.out_dir)

    # Load checkpoint
    model = fc_nn.FCClassifier.load_from_checkpoint(FLAGS.checkpoint)
    extra_hparams = model.extra_hparams

    # Read in training and validation dataset
    input_key = extra_hparams['key']
    transform, input_mean, input_stdv = preprocess(input_key, FLAGS)

    # Read in training and validation set
    input_key = extra_hparams['key']
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
    weight, pos_weight = get_weight(FLAGS)

    # Update model with new hypeperameters
    model = fc_nn.FCClassifier.load_from_checkpoint(
        FLAGS.checkpoint, strict=False, lr_scheduler=FLAGS.lr_scheduler,
        extra_hparams={
            'lr': FLAGS.lr,
            'pos_weight': pos_weight,
            'key': list(input_key),
            'preprocess': {
                'mean': input_mean,
                'stdv': input_stdv,
            },
            'transfer': {
                'train_first': FLAGS.train_first,
                'train_last': FLAGS.train_last,
            }
        }
    )
    #for parameter in model.parameters():
    #    if parameter.requires_grad:
    #        print(parameter)

    # Create trainer
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss", mode='min', filename="{epoch}-{val_loss:.4f}",
            save_top_k=3, save_last=True,save_weights_only=True),
        EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, mode='min', verbose=True)
    ]
    trainer_logger = CSVLogger(FLAGS.out_dir, name=FLAGS.name)
    trainer = pl.Trainer(
        default_root_dir=FLAGS.out_dir,
        accelerator=FLAGS.accelerator, devices=FLAGS.devices,
        max_epochs=FLAGS.max_epochs, min_epochs=min(10, FLAGS.max_epochs),
        callbacks=callbacks, logger=trainer_logger)

    # Start training
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluating on validation set and save
    results = trainer.predict(model, val_loader, ckpt_path=callbacks[0].best_model_path)
    predict = torch.cat([res[0] for res in results]).numpy()
    target = torch.cat([res[1] for res in results]).numpy()

    res_fn = os.path.join(trainer_logger.log_dir, 'best_val_results.hdf5')
    with h5py.File(res_fn, 'w') as f:
        f.create_dataset('predict', data=predict)
        f.create_dataset('target', data=target)


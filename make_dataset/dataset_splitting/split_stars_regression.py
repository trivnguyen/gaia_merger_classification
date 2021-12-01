#!/usr/bin/env python
# coding: utf-8

import os
import sys
import h5py
import json
import glob
import argparse
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt

PROPERTIES =('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity', 'feh', 'labels')

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='path to input directory')
    parser.add_argument('-o', '--output', required=True,
                        help='path to output directory')
    parser.add_argument('-rv', '--radial-velocity-only', action='store_true',
                        help='Enable to only include stars with radial velocity measurement')
    parser.add_argument('-p', '--partition', nargs='+', type=int, default=[8, 1, 1],
                        help='Train/val/test partition ratio')
    parser.add_argument('--which-label', required=False, type=int,
                        help='Which dimension of the label to pick. If not given, use all.')
    parser.add_argument('--label-bins', required=False, type=float, nargs='+',
                        help='Apply bins to label and calculate weight. If a list is given, assume (min, max, num_bins).')
    parser.add_argument('--n-max-per-file', required=False, default=1000000, type=int,
                        help='Maximum number of samples per file. Default to 1M samples.')
    parser.add_argument('--overwrite', action='store_true', help='If enable, overwrite old dataset')

    return parser.parse_args()

if __name__ == '__main__':

    # parse command-line args
    FLAGS = parse_cmd()

    # get input file
    in_files = sorted(glob.glob(os.path.join(FLAGS.input, '*.hdf5')))
    print('input file:')
    print('\n'.join(in_files))

    seed1, seed2 = np.random.randint(10000000, size=2)

    if FLAGS.overwrite and os.path.exists(FLAGS.output):
        shutil.rmtree(FLAGS.output)

    # if should include only stars with radial velocity measurement
    N_total = 0
    if FLAGS.radial_velocity_only:
        print('Only include stars with a radial velocity measurement')
        rv = []
        for in_file in in_files:
            with h5py.File(in_file, 'r') as input_f:
                rv.append(input_f['radial_velocity'][:])
        rv = np.concatenate(rv)
        selection = ~np.isnan(rv)
        N_total += np.sum(selection)
    else:
        for in_file in in_files:
            with h5py.File(in_file, 'r') as input_f:
                N_total += input_f['radial_velocity'].len()

    # compute partition fraction and index of each train/val/test set
    N_train = int(N_total * FLAGS.partition[0] / np.sum(FLAGS.partition))
    N_val = int(N_total * FLAGS.partition[1] / np.sum(FLAGS.partition))
    N_test = N_total - N_train - N_val

    print('Total number of samples: {}'.format(N_total))
    print('Number of train/val/test samples: {}, {}, {}'.format(N_train, N_val, N_test))
    print('Choose label dimension: {}'.format(FLAGS.which_label))

    index = np.random.permutation(N_total)
    train_idx = index[:N_train]
    val_idx = index[N_train: N_val + N_train]
    test_idx = index[N_val + N_train:]

    # split input files into train/test/val
    for p in PROPERTIES:

        full_data = []
        for in_file in in_files:
            with h5py.File(in_file, 'r') as input_f:
                full_data.append(input_f[p][:])
        full_data = np.concatenate(full_data)
        if FLAGS.radial_velocity_only:
            full_data = full_data[selection]

        if p == 'labels':
            if FLAGS.which_label is not None:
                full_data = full_data[:, FLAGS.which_label]

        # split dataset into training and validation
        train_data = full_data[train_idx]
        val_data = full_data[val_idx]
        test_data = full_data[val_idx]

        # write the properties into each dataset
        for data, flag in zip(
            (train_data, val_data, test_data), ('train', 'val', 'test')):
            n_file = data.shape[0] // FLAGS.n_max_per_file + 1

            out_dir_flag = os.path.join(FLAGS.output, flag)
            os.makedirs(out_dir_flag, exist_ok=True)

            for i in range(n_file):
                file = os.path.join(out_dir_flag, 'n{:02d}.hdf5'.format(i))
                with h5py.File(file, 'a') as output_f:
                    # write sliced dataset to file
                    start = i * FLAGS.n_max_per_file
                    end = (i + 1) * FLAGS.n_max_per_file
                    output_f.create_dataset(p, data=data[start: end])

    # Bin labels and calculate weight
    if FLAGS.label_bins is not None:
        out_dir_train = os.path.join(FLAGS.output, 'train')
        labels = []
        for i in range(10000000):
            file = os.path.join(out_dir_train, 'n{:02d}.hdf5'.format(i))
            if not os.path.exists(file):
                break
            with h5py.File(file, 'r') as f:
                # read in labels
                labels.append(f['labels'][:])
        # bin and calculate weight
        if len(FLAGS.label_bins) == 1:
            bins = int(FLAGS.label_bins[0])
        else:
            min_bins = min(FLAGS.label_bins[0], min(labels))
            max_bins = min(FLAGS.label_bins[1], min(labels))
            num_bins = int(FLAGS.label_bins[2])
            bins = np.linspace(min_bins, max_bins, num_bins)
        counts, bins = np.histogram(labels, bins)
        # weights = np.where(counts > 0, np.log10(np.sum(counts) /counts), 0)
        weights = np.where(counts > 0, np.sum(counts) / counts, 0)

        # write counts and bins
        with h5py.File(os.path.join(FLAGS.output, 'weight_properties.hdf5'), 'w') as f:
            f.create_dataset('counts', data=counts)
            f.create_dataset('bins', data=bins)

        # write weight back to file
        for flag in ('train', 'val', 'test'):
            out_dir_flag = os.path.join(FLAGS.output, flag)
            for i in range(10000000):
                file = os.path.join(out_dir_flag, 'n{:02d}.hdf5'.format(i))
                if not os.path.exists(file):
                    break
                with h5py.File(file, 'a') as f:
                    # read in labels
                    labels = f['labels'][:]
                    bins_idx = np.digitize(labels, bins) - 1
                    bins_idx[bins_idx < 0] = 0
                    bins_idx[bins_idx >= len(counts)] = len(counts) - 1
                    f.create_dataset('weight', data=weights[bins_idx])

    # Preprocessing dict
    # during training, we will standard-scale each feature
    train_files = sorted(glob.glob(os.path.join(FLAGS.output, 'train/*')))
    preprocess_dict = {}

    # loop over all keys and compute mean and stdv
    for p in PROPERTIES:
        if p == 'labels':
            continue
        data = []
        for file in train_files:
            with h5py.File(file, 'r') as f:
                data.append(f[p][:])
        data = np.concatenate(data)

        mean = np.nanmean(data)
        stdv = np.nanstd(data)

        print(f'Keys: {p}')
        print('Mean: {:.8e}'.format(mean))
        print('stdv: {:.8e}'.format(stdv))
        print('----------------')

        preprocess_dict[p] = {
            'mean': float(mean),
            'stdv': float(stdv),
        }

    with open(os.path.join(FLAGS.output, 'preprocess.json'), 'w') as f:
        json.dump(preprocess_dict, f, indent=4)

    print('Done!')


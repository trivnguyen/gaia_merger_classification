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

PROPERTIES =('l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity', 'feh', 'labels')

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='path to input directory')
    parser.add_argument('-o', '--output', required=True,
                        help='path to output directory')
    parser.add_argument('-p', '--partition', nargs='+', type=int, default=[8, 1, 1],
                        help='Train/val/test partition ratio')
    parser.add_argument('--which-label', required=False, type=int,
                        help='Which dimension of the label to pick. If not given, use all.')

    # weighting for regression
    parser.add_argument(
        '--label-bins', required=False, type=float, nargs='+',
        help='Apply bins to label and calculate weight. If a list is given, assume (min, max, num_bins).')
    parser.add_argument(
        '--weight-clip', required=False, type=float, default=1000,
        help='Maximum value of weight. Weights exceeded max will be set to max')
    parser.add_argument(
        '--log-weight', required=False, action='store_true',
        help='Enable to use log weight')

    parser.add_argument('--n-max-per-file', required=False, default=1000000, type=int,
                        help='Maximum number of samples per file. Default to 1M samples.')
    parser.add_argument('--overwrite', action='store_true', help='If enable, overwrite old dataset')

    return parser.parse_args()

if __name__ == '__main__':

    # parse command line arguments
    FLAGS = parse_cmd()

    # get input file
    in_files = sorted(glob.glob(os.path.join(FLAGS.input, '*.hdf5')))
    print('input file:')
    print('\n'.join(in_files))

    # overwrite directory if enable
    if FLAGS.overwrite and os.path.exists(FLAGS.output):
        shutil.rmtree(FLAGS.output)

    # from the simulation files, read in the parentid and label of each star
    # the parentid determines the stream/cluster the star originally belong too in FIRE
    parentid = []
    for in_file in in_files:
        with h5py.File(in_file, 'r') as input_f:
            parentid.append(input_f['parentid'][:])
    parentid = np.concatenate(parentid)

    # First, we partition the star particles into train/val/test
    # get unique parentid (i.e. id of star particles in the dataset) and shuffle
    unique, counts = np.unique(parentid, return_counts=True)
    shuffle = np.random.permutation(len(unique))
    unique = unique[shuffle]
    counts = counts[shuffle]
    cum_counts = np.cumsum(counts)

    # calculate the number of stars in the train/val/test set and split
    N_total = len(parentid)
    N_val = int(N_total * FLAGS.partition[1] / np.sum(FLAGS.partition))
    N_test = int(N_total * FLAGS.partition[2] / np.sum(FLAGS.partition))
    N_train = N_total - N_val - N_test

    print('Number of particles: {}'.format(N_total))
    print('Number of train/val/test particles: {}, {}, {}'.format(N_train, N_val, N_test))
    print('Choose label dimension {}'.format(FLAGS.which_label))

    # partition parentid into train, val, test dataset
    part = []
    part.append(np.where(cum_counts > N_train)[0][0])
    part.append(np.where(cum_counts > N_train + N_val)[0][0])
    part_parentid = {
        'train': unique[: part[0]],
        'val': unique[part[0]: part[1]],
        'test': unique[part[1]:]
    }

    # get the index of each star in each dataset
    train_idx = np.array([i for i in range(len(parentid))
                          if parentid[i] in part_parentid['train']])
    val_idx = np.array([i for i in range(len(parentid))
                        if parentid[i] in part_parentid['val']])
    test_idx = np.array([i for i in range(len(parentid))
                         if parentid[i] in part_parentid['test']])

    # Write the properties of stars into output train/val/test set based on their parentid
    for p in PROPERTIES:
        full_data = []
        for in_file in in_files:
            with h5py.File(in_file, 'r') as input_f:
                full_data.append(input_f[p][:])
        full_data = np.concatenate(full_data)

        if p == 'labels':
            if FLAGS.which_label is not None:
                full_data = full_data[:, FLAGS.which_label]

        # split the dataset into training, validation, and test set
        train_data = full_data[train_idx]
        val_data = full_data[val_idx]
        test_data = full_data[test_idx]

        # write each dataset
        for data, flag in zip(
            (train_data, val_data, test_data), ('train', 'val', 'test')):
            n_file = data.shape[0] // FLAGS.n_max_per_file + 1

            out_dir_flag = os.path.join(FLAGS.output, flag)
            os.makedirs(out_dir_flag, exist_ok=True)

            for i in range(n_file):
                fn = os.path.join(out_dir_flag, 'n{:02d}.hdf5'.format(i))
                with h5py.File(fn, 'a') as output_f:
                    # write sliced dataset to file
                    start = i * FLAGS.n_max_per_file
                    end = (i + 1) * FLAGS.n_max_per_file
                    output_f.create_dataset(p, data=data[start: end])

    # Bin labels and calculate weight
    if FLAGS.label_bins is not None:
        print('Bin label and calculate weight:')

        # read in labels
        labels = []
        for in_file in in_files:
            with h5py.File(in_file, 'r') as f:
                labels.append(f['labels'][:])
        labels = np.concatenate(labels)

        # bin and calculate weight
        if len(FLAGS.label_bins) == 1:
            bins = int(FLAGS.label_bins[0])
        else:
            min_bins = FLAGS.label_bins[0]
            max_bins = FLAGS.label_bins[1]
            num_bins = int(FLAGS.label_bins[2])
            bins = np.linspace(min_bins, max_bins, num_bins)
        counts, bins = np.histogram(labels, bins)

        # calculate weight for each bin
        if FLAGS.log_weight:
            print('Using log weight')
            weights = np.where(counts > 0, np.log10(np.sum(counts) /counts), 0)
        else:
            print('Using normal weight')
            weights = np.where(counts > 0, np.sum(counts) /counts, 0)

        # clipping weight
        N_large = np.sum(counts[weights > FLAGS.weight_clip])
        weights[weights > FLAGS.weight_clip] = FLAGS.weight_clip
        print('Clipping weight at: {}. Number of weights: {}'.format(FLAGS.weight_clip, N_large))

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

                    weight = weights[bins_idx]
                    f.create_dataset('weight', data=weight)

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


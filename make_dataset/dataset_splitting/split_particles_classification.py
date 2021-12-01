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
    parser.add_argument('--is-classification', required=False, action='store_true',
                        help='If the dataset is a classification dataset')
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
    print('Dataset type: {}'.format(dataset_type))

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

    # print out the counts and fraction of in situ and accreted stars for each flag
    # we want to make sure each flag is drawn from the same distribution
    properties = {}
    for flag in ('train', 'val', 'test'):
        files = sorted(glob.glob(os.path.join(FLAGS.output, f'{flag}/*')))

        labels = []
        for file in files:
            with h5py.File(file, 'r') as f:
                labels.append(f['labels'][:])
        labels = np.concatenate(labels)
        n_total = len(labels)

        print(flag)
        temp = {}
        for l in np.unique(labels):
            l = int(l)
            n = int(np.sum(labels==l))
            temp['n_{}'.format(l)] = n
            temp['f_{}'.format(l)] = n / n_total

            print('Number of class {:d} samples: {:d}'.format(l, n))
            print('Fraction of class {:d} samples: {:.4f}'.format(l, n / n_total))

        properties[flag] = temp

        print('---------------')

    with open(os.path.join(FLAGS.output, 'properties.json'), 'w') as f:
        json.dump(properties, f, indent=4)

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


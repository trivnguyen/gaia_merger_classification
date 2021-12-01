#!/bin/bash python


''' Subsample Anake synthetic Gaia survey data using a parallax cut parallax_over_error > 10 '''

import os
import sys
import h5py
import glob
import shutil
import argparse

import numpy as np

# properties to write to subsample output files
PROPS = ['source_id', 'parentid', 'ra', 'dec', 'l', 'b', 'parallax', 'pmra', 'pmdec',
         'radial_velocity', 'feh', 'px_true', 'py_true', 'pz_true', 'vx_true', 'vy_true', 'vz_true',
         'parallax_over_error']

def parse_cmd():
    ''' Parse command-line arguments '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Path to simulation directory')
    parser.add_argument('-o', '--output-dir', required=False, default='subsamples',
                        help='Path to output subsample directory')
    parser.add_argument('-l', '--labels-mapping', required=False,
                        help='Path to labels mapping file')
    parser.add_argument('-lp', '--labels-properties', required=False, nargs='+', default=['labels', ],
                        help='Properties in labels mapping file to map')
    parser.add_argument('-p', '--prefix', required=False, default='lsr-0')
    parser.add_argument('-b', '--batch-size', required=False, type=int, default=100000,
                        help='Batch size to load input data')
    parser.add_argument('-a', '--add-keys', nargs='+', required=False,
                        help='New keys to add to existing dataset')
    parser.add_argument('--overwrite', action='store_true', required=False,
                        help='Enable to overwrite existing dataset')
    return parser.parse_args()

if __name__ == '__main__':

    # parse command-line arguments
    FLAGS = parse_cmd()

    # get all simulation files
    sim_files = sorted(glob.glob(os.path.join(FLAGS.input_dir, '*.hdf5')))
    print('Simulation files: ')
    print("\n".join(sim_files))

    # if the file containing the index-to-label mapping array is given,
    # then also label each stars. Stars without any label will not be included
    # in the final subsample
    if FLAGS.labels_mapping is not None:
        PROPS.append('labels')

        print('Mapping {} from labels mapping file'.format(FLAGS.labels_properties))
        with h5py.File(FLAGS.labels_mapping, 'r') as f:
            id_stars = f['id_stars'][:]

            # get labels
            labels_mapping = []
            for prop in FLAGS.labels_properties:
                labels_mapping.append(f[prop][:])
            labels_mapping = np.stack(labels_mapping, axis=1)

            # sort by index
            sort = np.argsort(id_stars)
            id_stars = id_stars[sort]
            labels_mapping = labels_mapping[sort]

            print(dict(f.attrs))

    else:
        print(PROPS)

    # Overwrite existing dataset
    if FLAGS.overwrite and os.path.exists(FLAGS.output_dir):
        print('Overwriting existing dataset')
        shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    for i_file in range(len(sim_files)):
        output_fname = '{}-rslice-{:d}.m12i-res7100-subsamples.hdf5'.format(FLAGS.prefix, i_file)
        output_fname = os.path.join(FLAGS.output_dir, output_fname)

        print('No: [{:d}/ {:d}]'.format(i_file, len(sim_files)))
        print('Read in : {}'.format(sim_files[i_file]))
        print('Write to: {}'.format(output_fname))

        # skip file if output already exists
        if os.path.exists(output_fname) and (FLAGS.add_keys is None):
            print('Output file already exists. Skipping...')
            print('-------------')
            continue

        with h5py.File(sim_files[i_file], 'r') as input_f:
            N_max = input_f['parallax_over_error'].len()    # max number of accepted data

            # divide input data into slices, each with a size of batch_size
            slices = np.arange(N_max + FLAGS.batch_size, step=FLAGS.batch_size)

            # create an output HDF5 file
            if FLAGS.add_keys is None:
                mode = 'w'
            else:
                mode = 'a'
                PROPS = FLAGS.add_keys
                for p in PROPS:
                    if input_f.get(p) is None:
                        raise KeyError('key {} is not in input file',format(p))
                print('Adding keys {}'.format(PROPS))

            N_out_total = 0
            with h5py.File(output_fname, mode) as output_f:

                # start slicing input data
                for i in np.arange(len(slices) - 1):

                    sys.stdout.write('\rSubsampling progress: [{:d} / {:d}]'.format(
                        i, len(slices) - 1))
                    sys.stdout.flush()
                    start = slices[i]
                    stop = slices[i + 1]

                    # get parallax over error and apply a parallax cut
                    parallax_over_error = input_f['parallax_over_error'][start: stop]
                    cut = parallax_over_error > 10   # equivalent to dp / p  < 0.1

                    # if label mapping array is given, stars without labels are not considered
                    if FLAGS.labels_mapping is not None:
                        parentid = input_f['parentid'][start: stop]
                        cut = cut & np.isin(parentid, id_stars)

                    N_slices = np.sum(cut)
                    N_out_total += N_slices

                    # in case none of the data pass the cut
                    if N_slices == 0:
                        continue

                    # get other properties of the data and add to output dataset
                    for p in PROPS:
                        if p == 'parallax_over_error':
                            values = parallax_over_error[cut]
                        # map parentid to labels
                        elif p == 'labels':
                            values = labels_mapping[np.digitize(parentid[cut], id_stars) - 1]
                        else:
                            values = input_f[p][start: stop][cut]

                        # if dataset does not exist
                        if output_f.get(p) is None:
                            dset = output_f.create_dataset(p, data=values, maxshape=(N_max, *values.shape[1:]))
                        else:
                            # resize output dataset and add values
                            dset = output_f[p]
                            dset.resize(dset.shape[0] + N_slices, axis=0)
                            dset[-N_slices:] = values
        # delete empty files
        if N_out_total == 0:
            print('\nWARNING: file is empty. Removing...')
            os.remove(output_fname)
        print('-------------')
    print('Done!')


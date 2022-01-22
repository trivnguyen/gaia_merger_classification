
import os
import glob
import h5py

import numpy as np

# Define some global constants
INPUT_KEY_DICT = {
    'ACTION': ('Jr', 'Jphi', 'Jz'),
    'ACTION_FEH': ('Jr', 'Jphi', 'Jz', 'feh'),
    '6D_CYLC': ('r', 'phi', 'z', 'Jr', 'Jphi', 'Jz'),
    '6D_CYLC_FEH': ('r' , 'phi', 'z', 'Jr', 'Jphi', 'Jz', 'feh'),
    '6D_CART': ('x', 'y', 'z', 'Jr', 'Jphi', 'Jz'),
    '6D_CART_FEH': ('x' , 'y', 'z', 'Jr', 'Jphi', 'Jz', 'feh'),
    'ACTION_TIME': ('Jr', 'Jphi', 'Jz', 'a', 'T_uni'),
    'ACTION_FEH_TIME': ('Jr', 'Jphi', 'Jz', 'feh', 'a', 'T_uni'),
    '6D_CYLC_TIME': ('r', 'phi', 'z', 'Jr', 'Jphi', 'Jz', 'a', 'T_uni'),
    '6D_CYLC_FEH_TIME': ('r' , 'phi', 'z', 'Jr', 'Jphi', 'Jz', 'feh', 'a', 'T_uni'),
    '6D_CART_TIME': ('x', 'y', 'z', 'Jr', 'Jphi', 'Jz', 'a', 'T_uni'),
    '6D_CART_FEH_TIME': ('x' , 'y', 'z', 'Jr', 'Jphi', 'Jz', 'feh', 'a', 'T_uni'),
}

def get_input_key(key):
    ''' Convienient function to get input key from dict '''
    if (len(key) == 1) and (key[0].upper() in INPUT_KEY_DICT.keys()):
        input_key = INPUT_KEY_DICT[key[0].upper()]
    else:
        input_key = key
    return input_key

def choose_best_model(pretrained_model_dir, metrics='loss', mode='min'):
    ''' Choose best model from a pretrained model directory '''

    metrics_fn = os.path.join(pretrained_model_dir, 'metrics/{}.txt'.format(metrics))
    _, _, _, val_metric = np.genfromtxt(metrics_fn, unpack=True)
    if mode == 'min':
        best_ep = np.argmin(val_metric)
    elif mode == 'max':
        best_ep = np.argmax(val_metric)

    return os.path.join(pretrained_model_dir, 'models/epoch_{}'.format(best_ep))


def read_data_dir(data_dir, properties=None):
    ''' Convenient function to read multiple HDF5 files into dictionary '''
    # query all input files from data directory
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.hdf5')))

    # create dictionary and read in files
    data = {}
    for fn in data_files:
        with h5py.File(fn, 'r') as f:
            if properties is None:
                properties = f.keys()
            for k in properties:
                if data.get(k) is None:
                    data[k] = []
                data[k].append(f[k][:])
    for k, v in data.items():
        data[k] = np.concatenate(v)
    return data


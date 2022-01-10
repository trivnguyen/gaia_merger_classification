
import os
import glob
import h5py

def read_dataset(data_dir, properties=None):
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


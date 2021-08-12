
import os
import h5py
import glob

import numpy as np

import torch

def sort_key(string):
    base = os.path.basename(string)
    base = os.path.splitext(base)[0]
    return int(base.split('_')[-1][1:])

class Dataset(torch.utils.data.Dataset):
    ''' Customized Dataset object for large dataset with multiple HDF5 files.
        Iterate over the input directory and read in only one HDF5 file at once to save memory.
        Can be passed directly into torch.utils.data.DataLoader.
    '''
    def __init__(self, dataset_dir, keys=('data', 'label'), shuffle=False):
        self.dataset_dir = dataset_dir
        self.keys = keys
        self.shuffle = shuffle
        # get file list
        self.fn_list = glob.glob(os.path.join(dataset_dir, 'n*.h5'))
        self.fn_list = sorted(self.fn_list, key=sort_key)
        self.__check_keys(keys)
        self.sizes = self.__get_sizes()
        self.sizes_c = np.cumsum(self.sizes)
        # initiate cache
        self.data_cache = []
        self.curr_fn_id = None

    def __check_keys(self, keys):
        for fn in self.fn_list:
            with h5py.File(fn, 'r') as f:
                for k in keys:
                    if f.get(k) is None:
                        raise KeyError('Key {} does not exist in file {}'.format(k, fn))
        
    def __set_cache(self, fn_id):
        self.data_cache = []
        with h5py.File(self.fn_list[fn_id], 'r') as f:
            size = f.attrs['size']
            indices = np.random.permutation(size)
            for i, key in enumerate(self.keys):
                data = f[key][:]
                if self.shuffle:
                    data = data[indices]
                self.data_cache.append(data)
        
    def __get_sizes(self):
        sizes = []
        for fn in self.fn_list:
            with h5py.File(fn, 'r') as f:
                sizes.append(f.attrs['size'])
        return sizes
  
    def __len__(self):
        return np.sum(self.sizes)
    
    def __getitem__(self, idx):
        fn_id = np.digitize(idx, self.sizes_c)
        if fn_id != self.curr_fn_id:
            self.__set_cache(fn_id)
            self.curr_fn_id = fn_id
        if fn_id != 0:
            idx -= self.sizes_c[fn_id-1]
            
        data = []
        for i in range(len(self.keys)):
            data.append(self.data_cache[i][idx])
        return data

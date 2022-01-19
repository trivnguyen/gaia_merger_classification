
import os
from os.path import join
import h5py
import glob

import numpy as np
import torch

### Dataset class to take care of data loader
### ----------------------------------------
def sort_key(string):
    base = os.path.basename(string)
    base = os.path.splitext(base)[0]
    return int(base.split('_')[-1][1:])

class Dataset(torch.utils.data.Dataset):
    ''' Customized Dataset object for large dataset with multiple HDF5 files.
        Iterate over the input directory and read in only one HDF5 file at once to save memory.
        Can be passed directly into torch.utils.data.DataLoader.
    '''
    def __init__(self, in_dir, key, target_key=None, weight_key=None,
                 transform=None, target_transform=None, weight_transform=None,
                 shuffle=False, n_files=1000):

        super().__init__()

        self.in_dir = in_dir
        self.key = key
        self.target_key = target_key
        self.weight_key = weight_key
        self.transform = transform
        self.target_transform = target_transform
        self.weight_transform = weight_transform
        self.shuffle = shuffle

        # get a list of files in input directory
        self.in_files = []
        for i in range(n_files):
            in_file = join(in_dir, f'n{i:02d}.hdf5')
            if not os.path.exists(in_file):
                break
            self.in_files.append(in_file)
        self.__check_keys(key)
        self.__check_keys([target_key, ])

        self.sizes = self.__get_sizes()
        self.sizes_c = np.cumsum(self.sizes)

        # set data cache
        self.data = [None, None, None]
        self.current_file_idx = None

    def __check_keys(self, keys):
        ''' Iterate through self.input_files and check if keys exist '''
        for in_file in self.in_files:
            with h5py.File(in_file, 'r') as f:
                for k in keys:
                    if k is not None:
                        if f.get(k) is None:
                            raise KeyError('Key {} does not exist in file {}'.format(k, in_file))

    def __get_sizes(self):
        ''' Iterate through self.input_files and count the
        number of samples in each file '''
        sizes = []
        for file in self.in_files:
            with h5py.File(file, 'r') as f:
                sizes.append(f[self.key[0]].len())
        return sizes

    def __len__(self):
        ''' Return number of samples '''
        return np.sum(self.sizes)

    def __cache(self, file_idx):
        ''' Cache data '''
        with h5py.File(self.in_files[file_idx], 'r') as f:
            data = []
            for k in self.key:
                data.append(f[k][:])
            data = np.stack(data, -1)
            if self.shuffle:
                rand = np.random.permutation(data.shape[0])
                data = data[rand]

            if self.target_key is not None:
                target = f[self.target_key][:]
                target = target[rand] if self.shuffle else target
            else:
                target = None
            if self.weight_key is not None:
                weight = f[self.weight_key][:]
                weight = weight[rand] if self.shuffle else weight
            else:
                weight = None
        # reset cache data
        self.data = [data, target, weight]

    def __getitem__(self, idx):
        ''' Get item of a given index. Index continuous between files
        in the same order of self.input_files '''
        if idx >= self.__len__():
                raise IndexError('list index out of range')

        file_idx = np.digitize(idx, self.sizes_c)
        if file_idx != self.current_file_idx:
            self.__cache(file_idx)
            self.current_file_idx = file_idx
        if file_idx != 0:
            idx -= self.sizes_c[file_idx-1]

        # return data
        data, target, weight = self.data
        return_data = []
        if self.transform is not None:
            data = self.transform(data[idx])
            data = torch.FloatTensor(data)
        else:
            data = torch.FloatTensor(data[idx])
        return_data.append(data)

        if target is not None:
            if self.target_transform is not None:
                target = self.target_transform(target[idx])
                target = torch.FloatTensor(target[idx])
            else:
                target = torch.FloatTensor(target[idx])
            return_data.append(target)

        if weight is not None:
            if self.weight_transform is not None:
                weight = self.weight_transform(weight[idx])
                weight = torch.FloatTensor(weight[idx])
            else:
                weight = torch.FloatTensor(weight[idx])
            return_data.append(weight)

        return return_data


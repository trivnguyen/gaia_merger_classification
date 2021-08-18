
import os
import h5py
import glob

import numpy as np

import torch.utils.data

def sort_key(string):
    base = os.path.basename(string)
    base = os.path.splitext(base)[0]
    return int(base.split('_')[-1][1:])

class Dataset(torch.utils.data.Dataset):
    ''' Customized Dataset object for large dataset with multiple HDF5 files.
        Iterate over the input directory and read in only one HDF5 file at once to save memory.
        Can be passed directly into torch.utils.data.DataLoader.
    '''
    def __init__(self, input_dir, input_key, label_key=None, 
                 transform=None, shuffle=False, n_max_file=None):
        super().__init__()
        
        if isinstance(input_key, str):
            input_key = [input_key, ]

        # set attributes
        self.input_dir = input_dir
        self.input_key = input_key
        self.label_key = label_key
        self.shuffle = shuffle
        self.transform = transform

        self.input_dim = len(input_key)
    
        # get a list of file in directory
        self.input_files = sorted(
            glob.glob(os.path.join(input_dir, 'n*.hdf5')), key=sort_key)[:n_max_file]
        self.__check_keys(input_key)
        self.__check_keys(label_key)
        
        self.sizes = self.__get_sizes()
        self.sizes_c = np.cumsum(self.sizes)
        
        # set data cache
        self.cache = [None, None]
        self.current_file_index = None
    
    def __check_keys(self, key):
        ''' Iterate through self.input_files and check if keys exist '''
        if key is None:
            return
        
        for file in self.input_files:
            with h5py.File(file, 'r') as f:
                if isinstance(key, str):
                    if f.get(key) is None:
                        raise KeyError('Key {} does not exist in file {}'.format(key, file))
                else:
                    for k in key:
                        if f.get(k) is None:
                            raise KeyError('Key {} does not exist in file {}'.format(k, file))
      
    def __get_sizes(self):
        ''' Iterate through self.input_files and count the 
        number of samples in each file '''
        
        sizes = []
        for file in self.input_files:
            with h5py.File(file, 'r') as f:
                sizes.append(f[self.input_key[0]].len())
        return sizes
    
    def __len__(self):
        ''' Return number of samples '''
        return np.sum(self.sizes)
        
    def __cache(self, file_index):
        ''' Cache data '''
        # reset cache data
        self.cache = [None, None]
        
        with h5py.File(self.input_files[file_index], 'r') as f:
            # read in input data
            input_data = []
            for key in self.input_key:
                input_data.append(f[key][:])
            input_data = np.stack(input_data, 1)
                
            if self.shuffle:
                rand = np.random.permutation(input_data.shape[0])
                input_data = input_data[rand]
            self.cache[0] = input_data
            
            # read in label data
            if self.label_key is not None:
                label = f[self.label_key][:].reshape(-1, 1)
                if self.shuffle:
                    label = label[rand]
                self.cache[1] = label
    
    def __getitem__(self, index):
        ''' Get item of a given index. Index continuous between files 
        in the same order of self.input_files '''
        
        file_index = np.digitize(index, self.sizes_c)
        if file_index != self.current_file_index:
            self.__cache(file_index)
            self.current_file_index = file_index
        if file_index != 0:
            index -= self.sizes_c[file_index-1]
          
        if self.transform is not None:
            input_data = self.transform(self.cache[0][index])
        else:
            input_data = self.cache[0][index]
        
        if self.label_key is not None:
            labels = self.cache[1][index]
            return input_data, labels
        return input_data

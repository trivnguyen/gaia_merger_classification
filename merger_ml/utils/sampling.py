
import numpy as np


### Function to sample dataset
### -------------------------------
def sample_by_parts():
    pass

def sample_by_mergers(log_mass, train_frac):
    ''' Sample by mergers '''

    log_mass_mergers = np.unique(log_mass)

    train_ind, val_ind = [], []
    for m in log_mass_mergers:
        # get all indices of stars from the merger
        ind = np.where(log_mass == m)[0]
        ind = np.random.permutation(ind)

        # divide into training and validation set
        n_train = int(np.ceil(len(ind) * train_frac))
        train_ind.append(ind[:n_train])
        val_ind.append(ind[n_train:])
    train_ind = np.concatenate(train_ind)
    val_ind = np.concatenate(val_ind)

    # shuffle again
    train_ind = np.random.permutation(train_ind)
    val_ind = np.random.permutation(val_ind)

    return train_ind, val_ind

def sample_by_stars(n_samples, train_frac=0.9):
    ''' Sample dataset by stars. Return index '''
    ind = np.random.permutation(n_samples)
    n_train = int(np.ceil(n_samples * train_frac))
    train_ind = ind[:n_train]
    val_ind = ind[n_train: ]
    return train_ind, val_ind




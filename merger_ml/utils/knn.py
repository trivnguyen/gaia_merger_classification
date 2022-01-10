
import numpy as np
from sklearn.neighbors import KDTree

### KD Tree KNN function
### ---------------------------
def find_knn_catalog(catalog, k, leaf_size=40, batch_size=1024):
    ''' Return the distances and indices of k-nearest neightbors of each star '''

    # Construct KD tree
    kd_tree = KDTree(catalog, leaf_size=leaf_size)

    # Find the k-nearest neighbors for each star and get their distance
    n_catalog = len(catalog)
    n_batch = int(np.ceil(n_catalog / batch_size))

    # output array have shape (k+1) to include the targeted stars
    distances = np.zeros((n_catalog, k+1), dtype=float)
    indices = np.zeros((n_catalog, k+1), dtype=int)

    # Iterate over catalogs
    for i_batch in range(n_batch):
        if i_batch % (n_batch // 10) == 0:
            print('Progress: {:d}/{:d}'.format(i_batch, n_batch))

        istart = i_batch * batch_size
        istop = istart + batch_size

        # query k-nearest neighbors
        # note that we use k+1 because KDTree.query will return the targeted stars
        points = catalog[istart: istop]
        dist, ind = kd_tree.query(points, k+1, return_distance=True)

        # store to array
        distances[istart: istop] = dist
        indices[istart: istop] = ind

    return distances, indices



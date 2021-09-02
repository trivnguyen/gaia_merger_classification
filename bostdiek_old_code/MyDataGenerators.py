import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, normstatistics, labels,
                 batch_size=32, shuffle=True, class_weights={0: 1, 1: 1}):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.normstatistics = normstatistics
        self.list_IDs = list(list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.total_len = len(list_IDs)
        self.labels = labels
        self.class_weights = class_weights

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               min(self.total_len,
                                   (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = np.array(sorted([self.list_IDs[k] for k in indexes]))
#         print list_IDs_temp

        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp)

        return X, y, weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X :
        var1true, var2true, var3true, var4true, var5true = self.data
        var1_mean, var1_std = self.normstatistics[0]
        var2_mean, var2_std = self.normstatistics[1]
        var3_mean, var3_std = self.normstatistics[2]
        var4_mean, var4_std = self.normstatistics[3]
        var5_mean, var5_std = self.normstatistics[4]
        HaloLabels = self.labels

        # Get the galactic longitude 'var1_true'
        var1 = var1true[list_IDs_temp]
        var1 = (var1 - var1_mean) / var1_std

        # Get the galactic latitude 'var2_true'
        var2 = var2true[list_IDs_temp]
        var2 = (var2 - var2_mean) / var2_std

        # Get the var3 'var3'
        var3 = var3true[list_IDs_temp]
        var3 = (var3 - var3_mean) / var3_std

        # Get the proper motion in radial ascension 'var4_true'
        var4 = var4true[list_IDs_temp]
        var4 = (var4 - var4_mean) / var4_std

        # Get the proper motion in declination 'var5_true'
        var5 = var5true[list_IDs_temp]
        var5 = (var5 - var5_mean) / var5_std

        # Get the labels
        labels = HaloLabels[list_IDs_temp]
        labels = labels[:].tolist()
        # print labels
        # print labels.shape
        sample_weights = np.array([self.class_weights[x] for x in labels])
        # print sample_weights

        # Initialization
        X = np.empty((len(list_IDs_temp), 5))

        # Generate data
        X[:, 0] = var1.flatten()
        X[:, 1] = var2.flatten()
        X[:, 2] = var3.flatten()
        X[:, 3] = var4.flatten()
        X[:, 4] = var5.flatten()

        return X, labels, sample_weights

class DataGeneratorErrorSampling(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, normstatistics, labels,
                 batch_size=32, shuffle=True, class_weights={0: 1, 1: 1},
                 random_sample_size=100
                 ):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.normstatistics = normstatistics
        self.list_IDs = list(list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.total_len = len(list_IDs)
        self.labels = labels
        self.class_weights = class_weights
        self.random_sample_size = random_sample_size
        # self.IsParallax = IsParallax

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               min(self.total_len,
                                   (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = np.array(sorted([self.list_IDs[k] for k in indexes]))
#         print list_IDs_temp

        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp)

        return X, y, weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X :
        # print(len(self.data))
        var1true, var2true, var3true, var4true, var5true = self.data[:5]
        var1uncertainty, var2uncertainty, var3uncertainty, var4uncertainty,\
            var5uncertainty = self.data[-5:]
        var1_mean, var1_std = self.normstatistics[0]
        var2_mean, var2_std = self.normstatistics[1]
        var3_mean, var3_std = self.normstatistics[2]
        var4_mean, var4_std = self.normstatistics[3]
        var5_mean, var5_std = self.normstatistics[4]
        HaloLabels = self.labels

        # Get the galactic longitude 'var1_true'
        var1 = var1true[list_IDs_temp]
        var1uncertainty = var1uncertainty[list_IDs_temp]
        var1 = var1.reshape(var1.shape[0], 1)
        var1uncertainty = var1uncertainty.reshape(var1uncertainty.shape[0], 1)
        tmprvs = np.random.randn(var1.shape[0], self.random_sample_size)
        tmprvs[:, 0] = 0
        var1 = var1 + np.multiply(tmprvs, var1uncertainty)
        var1 = var1.reshape((1, var1.shape[0] * self.random_sample_size))
        var1 = (var1 - var1_mean) / var1_std
        # print(var1)

        # Get the galactic latitude 'var2_true'
        var2 = var2true[list_IDs_temp]
        var2uncertainty = var2uncertainty[list_IDs_temp]
        var2 = var2.reshape(var2.shape[0], 1)
        var2uncertainty = var2uncertainty.reshape(var2uncertainty.shape[0], 1)
        tmprvs = np.random.randn(var2.shape[0], self.random_sample_size)
        tmprvs[:, 0] = 0
        var2 = var2 + np.multiply(tmprvs, var2uncertainty)
        var2 = var2.reshape((1, var2.shape[0] * self.random_sample_size))
        var2 = (var2 - var2_mean) / var2_std
        # print(var2)

        # Get the var3 'var3'
        var3 = var3true[list_IDs_temp]
        var3uncertainty = var3uncertainty[list_IDs_temp]
        var3 = var3.reshape(var3.shape[0], 1)
        var3uncertainty = var3uncertainty.reshape(var3uncertainty.shape[0], 1)
        tmprvs = np.random.randn(var3.shape[0], self.random_sample_size)
        tmprvs[:, 0] = 0
        var3 = var3 + np.multiply(tmprvs, var3uncertainty)
        var3 = var3.reshape((1, var3.shape[0] * self.random_sample_size))
        var3 = (var3 - var3_mean) / var3_std
        # print(var3)

        # Get the proper motion in radial ascension 'var4_true'
        var4 = var4true[list_IDs_temp]
        var4uncertainty = var4uncertainty[list_IDs_temp]
        var4 = var4.reshape(var4.shape[0], 1)
        var4uncertainty = var4uncertainty.reshape(var4uncertainty.shape[0], 1)
        tmprvs = np.random.randn(var4.shape[0], self.random_sample_size)
        tmprvs[:, 0] = 0
        var4 = var4 + np.multiply(tmprvs, var4uncertainty)
        var4 = var4.reshape((1, var4.shape[0] * self.random_sample_size))
        var4 = (var4 - var4_mean) / var4_std
        # print(var4)

        # Get the proper motion in declination 'var5_true'
        var5 = var5true[list_IDs_temp]
        var5uncertainty = var5uncertainty[list_IDs_temp]
        var5 = var5.reshape(var5.shape[0], 1)
        var5uncertainty = var5uncertainty.reshape(var5uncertainty.shape[0], 1)
        tmprvs = np.random.randn(var5.shape[0], self.random_sample_size)
        tmprvs[:, 0] = 0
        var5 = var5 + np.multiply(tmprvs, var5uncertainty)
        var5 = var5.reshape((1, var5.shape[0] * self.random_sample_size))
        var5 = (var5 - var5_mean) / var5_std
        # print(var5)

        # Get the labels
        labels = HaloLabels[list_IDs_temp]
        labels = labels.reshape(labels.shape[0], 1)
        labels = np.multiply(labels, np.ones((1, self.random_sample_size)))
        labels = labels.flatten()
        sample_weights = np.array([self.class_weights[x] for x in labels])

        # Initialization
        X = np.empty((var1.shape[1], 5))

        # Generate data
        # print(X.shape)
        X[:, 0] = var1.flatten()
        X[:, 1] = var2.flatten()
        X[:, 2] = var3.flatten()
        X[:, 3] = var4.flatten()
        X[:, 4] = var5.flatten()

        # print(X[:20])
        # print(labels[:20])
        # print(sample_weights[:20])

        return X, labels, sample_weights


class DataGeneratorWithMetallicity(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, normstatistics, labels,
                 batch_size=32, shuffle=True, class_weights={0: 1, 1: 1}):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.normstatistics = normstatistics
        self.list_IDs = list(list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.total_len = len(list_IDs)
        self.labels = labels
        self.class_weights = class_weights

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               min(self.total_len,
                                   (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = np.array(sorted([self.list_IDs[k] for k in indexes]))
#         print list_IDs_temp

        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp)

        return X, y, weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X :
        var1true, var2true, var3true, var4true, var5true, var6true = self.data
        var1_mean, var1_std = self.normstatistics[0]
        var2_mean, var2_std = self.normstatistics[1]
        var3_mean, var3_std = self.normstatistics[2]
        var4_mean, var4_std = self.normstatistics[3]
        var5_mean, var5_std = self.normstatistics[4]
        var6_mean, var6_std = self.normstatistics[5]
        HaloLabels = self.labels

        # Get the galactic longitude 'var1_true'
        var1 = var1true[list_IDs_temp]
        var1 = (var1 - var1_mean) / var1_std

        # Get the galactic latitude 'var2_true'
        var2 = var2true[list_IDs_temp]
        var2 = (var2 - var2_mean) / var2_std

        # Get the var3 'var3'
        var3 = var3true[list_IDs_temp]
        var3 = (var3 - var3_mean) / var3_std

        # Get the proper motion in radial ascension 'var4_true'
        var4 = var4true[list_IDs_temp]
        var4 = (var4 - var4_mean) / var4_std

        # Get the proper motion in declination 'var5_true'
        var5 = var5true[list_IDs_temp]
        var5 = (var5 - var5_mean) / var5_std

        # Get the metallicity 'var6_true'
        var6 = var6true[list_IDs_temp]
        var6 = (var6 - var6_mean) / var6_std

        # Get the labels
        labels = HaloLabels[list_IDs_temp]
        labels = labels.reshape(labels.shape[0], 1)
        labels = labels.flatten()
        sample_weights = np.array([self.class_weights[x] for x in labels])
        # print sample_weights

        # Initialization
        X = np.empty((len(list_IDs_temp), 6))

        # Generate data
        X[:, 0] = var1.flatten()
        X[:, 1] = var2.flatten()
        X[:, 2] = var3.flatten()
        X[:, 3] = var4.flatten()
        X[:, 4] = var5.flatten()
        X[:, 5] = var6.flatten()

        return X, labels, sample_weights


class DataGeneratorVariableLength(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data, normstatistics, labels,
                 batch_size=32, shuffle=True, class_weights={0: 1, 1: 1},
                 random_sample_size=100):
        'Initialization'
        self.batch_size = batch_size
        self.data = data
        self.normstatistics = normstatistics
        self.list_IDs = list(list_IDs)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.total_len = len(list_IDs)
        self.labels = labels
        self.class_weights = class_weights
        self.random_sample_size = random_sample_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
                               min(self.total_len,
                                   (index + 1) * self.batch_size)]

        # Find list of IDs
        list_IDs_temp = np.array(sorted([self.list_IDs[k] for k in indexes]))
#         print list_IDs_temp

        # Generate data
        X, y, weights = self.__data_generation(list_IDs_temp)

        return X, y, weights

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X :

        # Initialization
        X = np.empty((len(list_IDs_temp) * self.random_sample_size, len(self.normstatistics)))
        HaloLabels = self.labels
        # print(len(self.normstatistics))
        vardata = self.data[:len(self.normstatistics)]
        uncertaintydata = self.data[len(self.normstatistics):]

        for i, vartrue in enumerate(vardata):
            var_mean, var_std = self.normstatistics[i]
            varuncertainty = uncertaintydata[i]
            # Get the galactic longitude 'var_true'
            var = vartrue[list_IDs_temp]
            varuncertainty = varuncertainty[list_IDs_temp]
            var = var.reshape(var.shape[0], 1)
            varuncertainty = varuncertainty.reshape(varuncertainty.shape[0], 1)
            tmprvs = np.random.randn(var.shape[0], self.random_sample_size)
            tmprvs[:, 0] = 0
            var = var + np.multiply(tmprvs, varuncertainty)
            var = var.reshape((1, var.shape[0] * self.random_sample_size))

            var = (var - var_mean) / var_std
            X[:, i] = var.flatten()

        # Get the labels
        labels = HaloLabels[list_IDs_temp]
        labels = labels.reshape(labels.shape[0], 1)
        labels = np.multiply(labels, np.ones((1, self.random_sample_size)))
        labels = labels.flatten()
        sample_weights = np.array([self.class_weights[x] for x in labels])

        return X, labels, sample_weights

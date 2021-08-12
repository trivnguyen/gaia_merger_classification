
import os
import h5py
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

# default plotting style
plt.style.use('seaborn-colorblind')
mpl.rc('font', size=15)
mpl.rc('figure', facecolor='w', figsize=(8, 5))

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class Logger:

    def __init__(self, out_dir, metrics):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.metrics = dict([(m, {'steps': [], 'epochs': [],
                                  'train': [], 'test': []}) for m in metrics])

        self.predict = []
        self.target = []

        self._print_train()

    def update_metric(self, metric_name, train_metric, epoch, test_metric=None,
                      n_batch_total=1, n_batch=None):
        # convert torch tensor to numpy array
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        if test_metric is None:
            test_metric = np.nan

        # update metric to dictionary
        if n_batch is None:
            n_batch = n_batch_total
        step = Logger._step(epoch, n_batch, n_batch_total)
        self.metrics[metric_name]['train'].append(train_metric)
        self.metrics[metric_name]['test'].append(test_metric)
        self.metrics[metric_name]['steps'].append(step)
        self.metrics[metric_name]['epochs'].append(step / n_batch_total)

    def log_metric(self, metric_name=None, max_epochs=None):

        # create a directory for metric
        out_dir = os.path.join(self.out_dir, 'metrics')
        os.makedirs(out_dir, exist_ok=True)

        # If name is not given, log all metrics
        if metric_name is not None:
            train = self.metrics[metric_name]['train']
            test = self.metrics[metric_name]['test']
            steps = self.metrics[metric_name]['steps']
            epochs = self.metrics[metric_name]['epochs']

            array = np.vstack((steps, epochs, train, test)).T
            header = 'Step     Epochs    Train     Test'
            np.savetxt(os.path.join(out_dir, f'{metric_name}.txt'),
                       array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
            self._plot_metric(metric_name)
        else:
            for metric_name in self.metrics.keys():
                train = self.metrics[metric_name]['train']
                test = self.metrics[metric_name]['test']
                steps = self.metrics[metric_name]['steps']
                epochs = self.metrics[metric_name]['epochs']

                array = np.vstack((steps, epochs, train, test)).T
                header = 'Step     Epochs    Train     Test'
                np.savetxt(os.path.join(out_dir, f'{metric_name}.txt'),
                           array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
                self._plot_metric(metric_name)

    def _plot_metric(self, metric_name):
            train = self.metrics[metric_name]['train']
            test = self.metrics[metric_name]['test']
            steps = self.metrics[metric_name]['steps']
            epochs = self.metrics[metric_name]['epochs']

            # plot
            fig, ax = plt.subplots()
            ax.plot(steps, train, label='Training')
            ax.plot(steps, test, label='Validation')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_name)
            ax.legend()

            axtwin = ax.twiny()
            axtwin.plot(epochs, train, alpha=0.)
            axtwin.grid(False)
            axtwin.set_xlabel('Epoch')

            # save plot
            out_dir = os.path.join(self.out_dir, 'metrics')
            fig.savefig(os.path.join(out_dir, f'{metric_name}.png'), dpi=300)
            plt.close()

    def display_status(self, metric_name, train_metric, epoch, test_metric=None,
                       max_epochs=None, n_batch_total=1, n_batch=None, show_epoch=True):
        # convert torch tensor to numpy array
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        if n_batch is None:
            n_batch = n_batch_total
        if max_epochs is None:
            max_epochs = np.nan

        # print out epoch number
        if show_epoch:
            logging.info('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch, max_epochs, n_batch, n_batch_total))
        logging.info('Train {0:}: {1:.4f}, Test {0:}: {2:.4f}'.format(
            metric_name, train_metric, test_metric))

    def update_predict(self, predict, target):
        # convert torch tensor to numpy array
        if isinstance(predict, torch.Tensor):
            predict = predict.data.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.data.cpu().numpy()

        self.predict.append(predict)
        self.target.append(target)

    def save_predict(self, epoch, n_batch=None, reset=True):
        out_dir = os.path.join(self.out_dir, 'results')
        os.makedirs(out_dir, exist_ok=True)
        if n_batch is not None:
            fn = os.path.join(out_dir, f'epoch_{epoch}_batch_{n_batch}.h5')
        else:
            fn = os.path.join(out_dir, f'epoch_{epoch}.h5')

        predict = np.concatenate(self.predict)
        target = np.concatenate(self.target)
        size = len(predict)

        # writing data in HDF5 format
        with h5py.File(fn, 'w') as f:
            f.attrs.update({
                'size': size,
                'epoch': epoch,
            })
            if n_batch is not None:
                f.attrs['n_batch'] = n_batch
            f.create_dataset('predict', data=predict, chunks=True)
            f.create_dataset('target', data=target, chunks=True)

        # reset array after storing data
        if reset:
            self.predict = []
            self.target = []

    def save_model(self, model, epoch, n_batch=None):
        out_dir = os.path.join(self.out_dir, 'models')
        os.makedirs(out_dir, exist_ok=True)
        if n_batch is not None:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'epoch_{epoch}_batch_{n_batch}'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(out_dir, f'epoch_{epoch}'))

    def save_optimizer(self, optimizer, epoch, n_batch=None):
        out_dir = os.path.join(self.out_dir, 'optimizers')
        os.makedirs(out_dir, exist_ok=True)
        if n_batch is not None:
            torch.save(optimizer.state_dict(),
                       os.path.join(out_dir, f'epoch_{epoch}_batch_{n_batch}'))
        else:
            torch.save(optimizer.state_dict(),
                       os.path.join(out_dir, f'epoch_{epoch}'))

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _print_train():
        ''' Most important function '''
        logger.info('Start training')
        logger.info('------------------------------------')
        logger.info('------------------------------------')
        logger.info('')
        logger.info('         ..oo0  ...ooOO00           ')
        logger.info('        ..     ...             !!!  ')
        logger.info('       ..     ...      o       \o/  ')
        logger.info('   Y  ..     /III\    /L ---    n   ')
        logger.info('  ||__II_____|\_/| ___/_\__ ___/_\__')
        logger.info('  [[____\_/__|/_\|-|______|-|______|')
        logger.info(' //0 ()() ()() 0   00    00 00    00')
        logger.info('')
        logger.info('------------------------------------')
        logger.info('------------------------------------')


class ClassifierLogger(Logger):

    def __init__(self, out_dir, metrics):
        super().__init__(out_dir, metrics)

    def save_binhist(self, binhist, epoch, n_batch):
        out_dir = os.path.join(self.out_dir, 'binhist')
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = binhist.plot()
        fig.savefig(os.path.join(out_dir, f'epoch_{epoch}_batch_{n_batch}.png'))
        plt.close()

    def save_cmatrix(self, cmatrix, epoch, n_batch, cmap=plt.cm.Blues):
        out_dir = os.path.join(self.data_subdir, 'cmatrices')
        os.makedirs(out_dir, exist_ok=True)

        # confusion matrix
        fig, ax = cmatrix.plot(norm=False, cmap=cmap)
        fig.savefig(os.path.join(out_dir, f'epoch_{epoch}_batch_{n_batch}.png'))
        plt.close()

        # normalized confusion matrix
        fig, ax = cmatrix.plot(norm=True, cmap=cmap)
        fig.savefig(os.path.join(out_dir, f'norm_epoch_{epoch}_batch_{n_batch}.png'))
        plt.close()



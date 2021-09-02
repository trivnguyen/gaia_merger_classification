import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'cmr10',
                     'font.size': 13})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 15

f0 = h5py.File('../data/lsr-0-rslice-0.m12i-res7100-md-sliced-gcat-dr2.hdf5', 'r')
f1 = h5py.File('../data/lsr-0-rslice-1.m12i-res7100-md-sliced-gcat-dr2.hdf5', 'r')

with open('../data/MeansAndVariance.txt', 'a') as f:
    # f.write('variable,mean,variance\n')
    for obs in ['radial_velocity', 'radial_velocity_error',
                'radial_velocity_true',
                'phot_g_mean_mag', 'phot_g_mean_mag_error',
                'phot_g_mean_mag_true',
                'phot_bp_mean_mag', 'phot_bp_mean_mag_error',
                'phot_bp_mean_mag_true',
                'phot_rp_mean_mag', 'phot_rp_mean_mag_error',
                'phot_rp_mean_mag_true']:
        tmplist = np.hstack([f0[obs], f1[obs]])
        tmp_mean = np.nanmean(tmplist)
        tmp_var = np.nanvar(tmplist)
        f.write('{0},{1},{2}\n'.format(obs, tmp_mean, tmp_var))

        plt.figure(figsize=(4, 4))
        plt.hist(tmplist[np.isfinite(tmplist)], histtype='step', bins=100)
        plt.ylabel('Stars per bin')
        plt.xlabel(obs)
        plt.yscale('log')
        plt.savefig('../plots/hist_{0}.pdf'.format(obs), bbox_inches='tight')
    # feh = np.hstack([f0['feh'], f1['feh']])
    # feh_mean = np.mean(feh)
    # feh_var = np.var(feh)
    # f.write('feh,{0},{1}\n'.format(feh_mean, feh_var))

    # Distance
    # px = np.hstack([f0['px_true'], f1['px_true']])
    # py = np.hstack([f0['py_true'], f1['py_true']])
    # pz = np.hstack([f0['pz_true'], f1['pz_true']])
    # distance = np.sqrt(np.square(px) + np.square(py) + np.square(pz))
    # distance_mean = np.mean(distance)
    # distance_var = np.var(distance)
    # f.write('distance,{0},{1}\n'.format(distance_mean, distance_var))

    # plt.figure(figsize=(4, 4))
    # plt.hist(distance, histtype='step', bins=100)
    # plt.ylabel('Stars per bin')
    # plt.xlabel('Distance')
    # plt.yscale('log')
    # plt.savefig('plots/hist_distance.pdf', bbox_inches='tight')

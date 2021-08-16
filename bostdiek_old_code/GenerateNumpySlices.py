import numpy as np
import h5py


f0 = h5py.File('../data/m12iLSR0/lsr-0-rslice-0.m12i-res7100-md-sliced-gcat-dr2.hdf5',
               'r')
f1 = h5py.File('../data/m12iLSR0/lsr-0-rslice-1.m12i-res7100-md-sliced-gcat-dr2.hdf5',
               'r')

for var in ['alpha', 'age', 'logg']:
    tmp = np.hstack([f0[var], f1[var]])
    np.save('../data/m12iLSR0/sliced_{0}.npy'.format(var), tmp)
    del(tmp)
# var = 'parentid'
# tmp = np.hstack([f0[var], f1[var]])
# np.save('../data/m12iLSR0/sliced_{0}.npy'.format(var), tmp)

# HighFormationDist = np.load('../data/accreted_high_formation_distance_ids_m12i.npy')
# HighFormationDistLabel0 = np.in1d(f0['parentid'], HighFormationDist)
# HighFormationDistLabel1 = np.in1d(f1['parentid'], HighFormationDist)
# #
# HighFormationDistLabel = np.hstack([HighFormationDistLabel0,
#                                     HighFormationDistLabel1]).astype(int)
# np.save('../data/m12iLSR0/StarLabels_HighFormationDistance.npy',
#         HighFormationDistLabel)
#
# HaloStarIDS_accretedvirial = np.load('../data/accreted_stellar_ids_virialized-2.npy')
# HaloLabels0_accretedvirial = np.in1d(f0['parentid'], HaloStarIDS_accretedvirial)
# HaloLabels1_accretedvirial = np.in1d(f1['parentid'], HaloStarIDS_accretedvirial)
# # #
# HL_accretedvirial = np.hstack([HaloLabels0_accretedvirial,
#                                HaloLabels1_accretedvirial]).astype(int)
# np.save('../data/m12iLSR0/StarLabels_accreted_virialized.npy',
#         HL_accretedvirial)
# # # #
# HaloStarIDS_accreted = np.load('../data/accreted_stellar_ids.npy')
# HaloLabels0_accreted = np.in1d(f0['parentid'], HaloStarIDS_accreted)
# HaloLabels1_accreted = np.in1d(f1['parentid'], HaloStarIDS_accreted)
# # #
# HL_accreted = np.hstack([HaloLabels0_accreted,
#                          HaloLabels1_accreted]).astype(int)
# # #
# # print(float(np.sum(HL_accretedvirial == HL_accreted)) / len(HL_accretedvirial))
# # #
# np.save('../data/m12iLSR0/StarLabels_accreted.npy',
#         HL_accreted)
#
# # StarIndices = np.load('../data/StarIndexList.npy', mmap_mode='r')
#
# tmplist = np.hstack([f0['parallax_over_error'], f1['parallax_over_error']])
# small_errors = np.argwhere(abs(1.0 / tmplist) < 0.1)
# small_errors.reshape(small_errors.shape[0], 1)
# np.save('../data/m12iLSR0/small_error_indicies_sorted.npy', small_errors)
#
# #
# #
# # HaloLabels = np.load('../data/StarLabels.npy')
# #
# # halo_small = np.sum(HaloLabels[small_errors])
# # non_halo_small = small_errors.shape[0] - halo_small
# #
# # print('Total number of stars: {0}'.format(tmplist.shape[0]))
# # print('Number of stars with small error: {0}'.format(small_errors.shape[0]))
# # print('{0} of those are Halo'.format(halo_small))
# # print('{0} of those are Non-Halo'.format(non_halo_small))
# # print('Weights:')
# # print(' 0: {0}'.format(float(small_errors.shape[0]) / non_halo_small))
# # print(' 1: {0}'.format(float(small_errors.shape[0]) / halo_small))
# #
# np.random.shuffle(small_errors)
# np.save('../data/m12iLSR0/small_error_indicies_random.npy', small_errors)
#
# for var in ['feh',
#             'radial_velocity', 'radial_velocity_error',
#             'l', 'b',
#             'pmra', 'pmra_error',
#            'pmdec', 'pmdec_error',
#            # 'parallax', 'parallax_error',
#             'radial_velocity', 'radial_velocity_error',
#             'phot_g_mean_mag', 'phot_g_mean_mag_error',
#             'phot_bp_mean_mag', 'phot_bp_mean_mag_error',
#             'phot_rp_mean_mag', 'phot_rp_mean_mag_error',
#             'parentid',
#             'px_true', 'py_true', 'pz_true',
#             'vx_true', 'vy_true', 'vz_true'
#            # ]:
#    # tmp = np.hstack([f0[var], f1[var]])
#    # np.save('../data/m12iLSR0/sliced_{0}.npy'.format(var), tmp)
#    # del(tmp)
#    # tmp = np.load('../data/m12iLSR0/sliced_{0}.npy'.format(var))
#    # print(tmp[:10])
#    # del(tmp)
#
# px = np.hstack([f0['px_true'], f1['px_true']])
# py = np.hstack([f0['py_true'], f1['py_true']])
# pz = np.hstack([f0['pz_true'], f1['pz_true']])
# #
# distance = np.sqrt(np.square(px) + np.square(py) + np.square(pz))
# # del(px)
# # del(py)
# # del(pz)
#
# # distance = distance[StarIndices]
# np.save('../data/m12iLSR0/sliced_distance.npy', distance)

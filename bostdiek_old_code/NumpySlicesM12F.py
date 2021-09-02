import numpy as np
import h5py


f0 = h5py.File('../data/m12flsr1/lsr-1-rslice-0.m12f-res7100-md-sliced-gcat-dr2.hdf5',
               'r')
f1 = h5py.File('../data/m12flsr1/lsr-1-rslice-1.m12f-res7100-md-sliced-gcat-dr2.hdf5',
               'r')

for var in ['alpha', 'age', 'logg']:
    tmp = np.hstack([f0[var], f1[var]])
    np.save('../data/m12flsr1/sliced_{0}.npy'.format(var), tmp)
    del(tmp)

# HighFormationDist25 = np.load('../data/ids_m12f_formation_cut25.npy')
# HighFormationDistLabel0 = np.in1d(f0['parentid'], HighFormationDist25)
# HighFormationDistLabel1 = np.in1d(f1['parentid'], HighFormationDist25)
# # #
# HighFormationDistLabel = np.hstack([HighFormationDistLabel0,
#                                     HighFormationDistLabel1]).astype(int)
# np.save('../data/m12flsr1/StarLabels_HighFormationDistance25.npy',
#         HighFormationDistLabel)
#
# HaloStarIDS_accretedvirial = np.load('../data/virialized_stars_m12f.npy')
# HaloLabels0_accretedvirial = np.in1d(f0['parentid'], HaloStarIDS_accretedvirial)
# HaloLabels1_accretedvirial = np.in1d(f1['parentid'], HaloStarIDS_accretedvirial)
# # # # #
# HL_accretedvirial = np.hstack([HaloLabels0_accretedvirial,
#                                HaloLabels1_accretedvirial]).astype(int)
# np.save('../data/m12flsr1/StarLabels_accreted_virialized.npy',
#         HL_accretedvirial)
# # # # #
# HaloStarIDS_accreted = np.load('../data/accreted_stars_m12f.npy')
# HaloLabels0_accreted = np.in1d(f0['parentid'], HaloStarIDS_accreted)
# HaloLabels1_accreted = np.in1d(f1['parentid'], HaloStarIDS_accreted)
# # # # #
# HL_accreted = np.hstack([HaloLabels0_accreted,
#                          HaloLabels1_accreted]).astype(int)
# # # #
# print(float(np.sum(HL_accretedvirial == HL_accreted)) / len(HL_accretedvirial))
# # # #
# np.save('../data/m12flsr1/StarLabels_accreted.npy',
#         HL_accreted)
# #
# StarIndicies = np.arange(HL_accretedvirial.shape[0])
# np.random.shuffle(StarIndicies)
# np.save('../data/m12flsr1/StarIndexList.npy', StarIndicies)
# #
# tmplist = np.hstack([f0['parallax_over_error'], f1['parallax_over_error']])
# small_errors = np.argwhere(abs(1.0 / tmplist) < 0.1)
# small_errors.reshape(small_errors.shape[0], 1)
# np.save('../data/m12flsr1/small_error_indicies_sorted.npy', small_errors)
#
# np.random.shuffle(small_errors)
# np.save('../data/m12flsr1/small_error_indicies_random.npy', small_errors)
# #
# for var in ['feh',
#             'radial_velocity', 'radial_velocity_error',
#             'l', 'b',
#             'pmra', 'pmra_error',
#             'pmdec', 'pmdec_error',
#             'parallax', 'parallax_error',
#             'phot_g_mean_mag', 'phot_g_mean_mag_error',
#             'phot_bp_mean_mag', 'phot_bp_mean_mag_error',
#             'phot_rp_mean_mag', 'phot_rp_mean_mag_error',
#             'parentid',
#             'px_true', 'py_true', 'pz_true',
#             'vx_true', 'vy_true', 'vz_true'
#             ]:
#     tmp = np.hstack([f0[var], f1[var]])
#     np.save('../data/m12flsr1/sliced_{0}.npy'.format(var), tmp)
#     del(tmp)

# px = np.hstack([f0['px_true'], f1['px_true']])
# py = np.hstack([f0['py_true'], f1['py_true']])
# pz = np.hstack([f0['pz_true'], f1['pz_true']])
#
# distance = np.sqrt(np.square(px) + np.square(py) + np.square(pz))
# del(px)
# del(py)
# del(pz)

# distance = distance
# np.save('../data/m12flsr1/sliced_distance.npy', distance)

for var in ['source_id']:
    tmp = np.hstack([f0[var], f1[var]])
    np.save('../data/m12flsr1/sliced_{0}.npy'.format(var), tmp)
    del(tmp)

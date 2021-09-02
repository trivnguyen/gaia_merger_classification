import numpy as np
import pandas as pd

StarIndices = np.load('../data/m12iLSR0/small_error_indicies_random.npy')
# StarIndices = np.load('../data/m12iLSR0/StarIndexList.npy')

ParentID = np.load('../data/m12iLSR0/sliced_parentid.npy')
AccretedVirializedLabels = np.load('../data/m12iLSR0/StarLabels_accreted_virialized.npy')
Accreted = np.load('../data/m12iLSR0/StarLabels_accreted.npy')
HighFormDist = np.load('../data/m12iLSR0/StarLabels_HighFormationDistance.npy')
l = np.load('../data/m12iLSR0/sliced_l_true.npy')
b = np.load('../data/m12iLSR0/sliced_b_true.npy')
disttrue = np.load('../data/m12iLSR0/sliced_distance.npy')
parallax = np.load('../data/m12iLSR0/sliced_parallax.npy')
# parallax_true = np.load('../data/m12iLSR0/sliced_parallax_true.npy')
pmra = np.load('../data/m12iLSR0/sliced_pmra.npy')
pmdec = np.load('../data/m12iLSR0/sliced_pmdec.npy')
# pmra_true = np.load('../data/m12iLSR0/sliced_pmra_true.npy')
# pmdec_true = np.load('../data/m12iLSR0/sliced_pmdec_true.npy')
pmra_error = np.load('../data/m12iLSR0/sliced_pmra_error.npy')
pmdec_error = np.load('../data/m12iLSR0/sliced_pmdec_error.npy')
parallax_error = np.load('../data/m12iLSR0/sliced_parallax_error.npy')
vx_true = np.load('../data/m12iLSR0/sliced_vx_true.npy')
vy_true = np.load('../data/m12iLSR0/sliced_vy_true.npy')
vz_true = np.load('../data/m12iLSR0/sliced_vz_true.npy')
pz_true = np.load('../data/m12iLSR0/sliced_pz_true.npy')
py_true = np.load('../data/m12iLSR0/sliced_py_true.npy')
px_true = np.load('../data/m12iLSR0/sliced_px_true.npy')
feh = np.load('../data/m12iLSR0/sliced_feh.npy')

alpha = np.load('../data/m12iLSR0/sliced_alpha.npy')
age = np.load('../data/m12iLSR0/sliced_age.npy')
logg = np.load('../data/m12iLSR0/sliced_logg.npy')

radial_velocity = np.load('../data/m12iLSR0/sliced_radial_velocity.npy')
radial_velocity_error = np.load('../data/m12iLSR0/sliced_radial_velocity_error.npy')
# radial_velocity_true = np.load('../data/m12iLSR0/sliced_radial_velocity_true.npy')

phot_g_mean_mag = np.load('../data/m12iLSR0/sliced_phot_g_mean_mag.npy')
# phot_g_mean_mag_true = np.load('../data/m12iLSR0/sliced_phot_g_mean_mag_true.npy')
phot_g_mean_mag_error = np.load('../data/m12iLSR0/sliced_phot_g_mean_mag_error.npy')

phot_bp_mean_mag = np.load('../data/m12iLSR0/sliced_phot_bp_mean_mag.npy')
# phot_bp_mean_mag_true = np.load('../data/m12iLSR0/sliced_phot_bp_mean_mag_true.npy')
phot_bp_mean_mag_error = np.load('../data/m12iLSR0/sliced_phot_bp_mean_mag_error.npy')

phot_rp_mean_mag = np.load('../data/m12iLSR0/sliced_phot_rp_mean_mag.npy')
# phot_rp_mean_mag_true = np.load('../data/m12iLSR0/sliced_phot_rp_mean_mag_true.npy')
phot_rp_mean_mag_error = np.load('../data/m12iLSR0/sliced_phot_rp_mean_mag_error.npy')

print(float(sum(AccretedVirializedLabels == Accreted)) /
      len(AccretedVirializedLabels))


# The test set is the last 1 million stars, but they have been randomized
# in the Star indices list. Take the last million of these and then sort
# StarIndices = StarIndices[np.isfinite(radial_velocity[StarIndices])]

teststars = StarIndices[-10000000:]
teststars.reshape(teststars.shape[0]).sort()
teststars = teststars.flatten()
print(teststars[:20])
print(len(teststars))
TestStarDict = {'StarIndex': teststars,
                'l': l[teststars],
                'b': b[teststars],
                'parallax': parallax[teststars],
                # 'parallax_true': parallax_true[teststars],
                'parallax_error': parallax_error[teststars],
                'dist_true': disttrue[teststars],
                'pmra': pmra[teststars],
                # 'pmra_true': pmra_true[teststars],
                'pmra_error': pmra_error[teststars],
                'pmdec': pmdec[teststars],
                # 'pmdec_true': pmdec_true[teststars],
                'pmdec_error': pmdec_error[teststars],
                'feh': feh[teststars],
                'alpha': alpha[teststars],
                'age': age[teststars],
                'logg': logg[teststars],
                'vx_true': vx_true[teststars],
                'vy_true': vy_true[teststars],
                'vz_true': vz_true[teststars],
                'pz_true': pz_true[teststars],
                'px_true': px_true[teststars],
                'py_true': py_true[teststars],
                # 'HaloLabel': HaloLabels[teststars],
                'AccretedVirializedLabel': AccretedVirializedLabels[teststars],
                'AccretedLabel': Accreted[teststars],
                'HighFormDist': HighFormDist[teststars],
                'radial_velocity': radial_velocity[teststars],
                'radial_velocity_error': radial_velocity_error[teststars],
                # 'radial_velocity_true': radial_velocity_true[teststars],
                'phot_g_mean_mag': phot_g_mean_mag[teststars],
                # 'phot_g_mean_mag_true': phot_g_mean_mag_true[teststars],
                'phot_g_mean_mag_error': phot_g_mean_mag_error[teststars],
                'phot_bp_mean_mag': phot_bp_mean_mag[teststars],
                # 'phot_bp_mean_mag_true': phot_bp_mean_mag_true[teststars],
                'phot_bp_mean_mag_error': phot_bp_mean_mag_error[teststars],
                'phot_rp_mean_mag': phot_rp_mean_mag[teststars],
                # 'phot_rp_mean_mag_true': phot_rp_mean_mag_true[teststars],
                'phot_rp_mean_mag_error': phot_rp_mean_mag_error[teststars]
                }

TestSet = pd.DataFrame.from_dict(TestStarDict)
TestSet.to_csv('../data/m12iLSR0/MediumSet_WithAgeandAlpha.csv'
               )

print(float(sum(AccretedVirializedLabels[teststars] == Accreted[teststars])) /
      len(AccretedVirializedLabels[teststars]))

print(float(sum(HighFormDist[teststars] == Accreted[teststars])) /
      len(HighFormDist[teststars]))

import numpy as np
import pandas as pd

for lsr in [0, 1, 2]:
    # StarIndices = np.load('../data/m12iLSR{0}/small_error_indicies_random.npy'.format(lsr))
    ParentID = np.load('../data/m12iLSR{0}/sliced_parentid.npy'.format(lsr))
    AccretedVirializedLabels = np.load('../data/m12iLSR{0}/StarLabels_accreted_virialized.npy'.format(lsr))
    Accreted = np.load('../data/m12iLSR{0}/StarLabels_accreted.npy'.format(lsr))
    if lsr == 0:
        l = np.load('../data/m12iLSR{0}/sliced_l_true.npy'.format(lsr))
        b = np.load('../data/m12iLSR{0}/sliced_b_true.npy'.format(lsr))
    else:
        l = np.load('../data/m12iLSR{0}/sliced_l.npy'.format(lsr))
        b = np.load('../data/m12iLSR{0}/sliced_b.npy'.format(lsr))
    disttrue = np.load('../data/m12iLSR{0}/sliced_distance.npy'.format(lsr))
    parallax = np.load('../data/m12iLSR{0}/sliced_parallax.npy'.format(lsr))
    pmra = np.load('../data/m12iLSR{0}/sliced_pmra.npy'.format(lsr))
    pmdec = np.load('../data/m12iLSR{0}/sliced_pmdec.npy'.format(lsr))
    pmra_error = np.load('../data/m12iLSR{0}/sliced_pmra_error.npy'.format(lsr))
    pmdec_error = np.load('../data/m12iLSR{0}/sliced_pmdec_error.npy'.format(lsr))
    parallax_error = np.load('../data/m12iLSR{0}/sliced_parallax_error.npy'.format(lsr))
    vx_true = np.load('../data/m12iLSR{0}/sliced_vx_true.npy'.format(lsr))
    vy_true = np.load('../data/m12iLSR{0}/sliced_vy_true.npy'.format(lsr))
    vz_true = np.load('../data/m12iLSR{0}/sliced_vz_true.npy'.format(lsr))
    pz_true = np.load('../data/m12iLSR{0}/sliced_pz_true.npy'.format(lsr))
    py_true = np.load('../data/m12iLSR{0}/sliced_py_true.npy'.format(lsr))
    px_true = np.load('../data/m12iLSR{0}/sliced_px_true.npy'.format(lsr))
    feh = np.load('../data/m12iLSR{0}/sliced_feh.npy'.format(lsr))

    radial_velocity = np.load('../data/m12iLSR{0}/sliced_radial_velocity.npy'.format(lsr))
    radial_velocity_error = np.load('../data/m12iLSR{0}/sliced_radial_velocity_error.npy'.format(lsr))

    phot_g_mean_mag = np.load('../data/m12iLSR{0}/sliced_phot_g_mean_mag.npy'.format(lsr))
    phot_g_mean_mag_error = np.load('../data/m12iLSR{0}/sliced_phot_g_mean_mag_error.npy'.format(lsr))

    phot_bp_mean_mag = np.load('../data/m12iLSR{0}/sliced_phot_bp_mean_mag.npy'.format(lsr))
    phot_bp_mean_mag_error = np.load('../data/m12iLSR{0}/sliced_phot_bp_mean_mag_error.npy'.format(lsr))

    phot_rp_mean_mag = np.load('../data/m12iLSR{0}/sliced_phot_rp_mean_mag.npy'.format(lsr))
    phot_rp_mean_mag_error = np.load('../data/m12iLSR{0}/sliced_phot_rp_mean_mag_error.npy'.format(lsr))

    print(float(sum(AccretedVirializedLabels == Accreted)) /
          len(AccretedVirializedLabels))


    # The test set is the last 1 million stars, but they have been randomized
    # in the Star indices list. Take the last million of these and then sort
    # StarIndices = StarIndices[np.isfinite(radial_velocity[StarIndices])]

    # teststars = StarIndices
    # teststars.reshape(teststars.shape[0]).sort()
    # teststars = teststars.flatten()
    # print(teststars[:20])
    # print(len(teststars))
    TestStarDict = {'l': l,  # 'l': l[teststars],
                    'b': b,
                    'parallax': parallax,
                    'parallax_error': parallax_error,
                    'dist_true': disttrue,
                    'pmra': pmra,
                    'pmra_error': pmra_error,
                    'pmdec': pmdec,
                    'pmdec_error': pmdec_error,
                    'feh': feh,
                    'vx_true': vx_true,
                    'vy_true': vy_true,
                    'vz_true': vz_true,
                    'pz_true': pz_true,
                    'px_true': px_true,
                    'py_true': py_true,
                    'AccretedVirializedLabel': AccretedVirializedLabels,
                    'AccretedLabel': Accreted,
                    'radial_velocity': radial_velocity,
                    'radial_velocity_error': radial_velocity_error,
                    'phot_g_mean_mag': phot_g_mean_mag,
                    'phot_g_mean_mag_error': phot_g_mean_mag_error,
                    'phot_bp_mean_mag': phot_bp_mean_mag,
                    'phot_bp_mean_mag_error': phot_bp_mean_mag_error,
                    'phot_rp_mean_mag': phot_rp_mean_mag,
                    'phot_rp_mean_mag_error': phot_rp_mean_mag_error
                    }
    cnames = ['l', 'b', 'pmra', 'pmdec', 'parallax', 'radial_velocity',
              'feh',
              'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
              'pmra_error', 'pmdec_error', 'parallax_error',
              'radial_velocity_error',
              'phot_g_mean_mag_error', 'phot_bp_mean_mag_error',
              'phot_rp_mean_mag_error',
              'dist_true', 'px_true', 'py_true', 'pz_true',
              'vx_true', 'vy_true', 'vz_true',
              'AccretedVirializedLabel', 'AccretedLabel'
              ]
    TestSet = pd.DataFrame.from_dict(TestStarDict)
    TestSet = TestSet[cnames]
    # TestSet.to_hdf('../MediumData_m12iLSR{0}.h5'.format(lsr),
    #                key='StarData'
    #                )
    np.save('../LargeData_m12iLSR{0}.npy'.format(lsr),
            TestSet.as_matrix())
    # if lsr == 0:
    #     with open('../ColumnNamesForNPFiles.txt', 'w') as fout:
    #         for i, col in enumerate(cnames):
    #             fout.write('{0},{1}\n'.format(i, col))

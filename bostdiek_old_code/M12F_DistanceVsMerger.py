import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'cmr10', 'font.size': 13})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 15
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True


def ShiftStarsAndCalculate(dataframe, positions, velocities):
    x_LSR, y_LSR, z_LSR = positions
    vx_LSR, vy_LSR, vz_LSR = velocities
    for var, val in zip(['px_true', 'py_true', 'pz_true'],
                        positions):
        dataframe['Shifted_{0}'.format(var)] = dataframe[var] + val

    for var, val in zip(['vx_true', 'vy_true', 'vz_true'],
                        velocities):
        dataframe['Shifted_{0}'.format(var)] = dataframe[var] + val

    dataframe['Shifted_r'] = np.sqrt(dataframe['Shifted_px_true']**2 +
                                     dataframe['Shifted_py_true']**2
                                     )
    dataframe['Shifted_phi'] = np.arctan2(dataframe['Shifted_py_true'],
                                          dataframe['Shifted_px_true'])
    phi_LSR = np.arctan2(y_LSR, x_LSR)
    vphi_LSR = -(- vx_LSR * np.sin(phi_LSR) + vy_LSR * np.cos(phi_LSR))
    vr_LSR = vx_LSR * np.cos(phi_LSR) + vy_LSR * np.sin(phi_LSR)
    print('LSR velocities: vr={0:.2f}, vphi={1:.2f}'.format(vr_LSR, vphi_LSR))
    print('Rotating LSR by {0:2f} pi radians'.format(phi_LSR / np.pi))

    dataframe['vr'] = ((np.cos(dataframe['Shifted_phi']) * dataframe['Shifted_vx_true'] +
                        np.sin(dataframe['Shifted_phi']) * dataframe['Shifted_vy_true'])
                       )
    dataframe['vphi'] = -((-np.sin(dataframe['Shifted_phi']) * dataframe['Shifted_vx_true'] +
                           np.cos(dataframe['Shifted_phi']) * dataframe['Shifted_vy_true'])
                          )
    dataframe['phi'] = dataframe['Shifted_phi'] - phi_LSR


m12f = pd.read_csv('../data/m12flsr1/MediumTestSet_WithHighFormDist.csv')
mf_px, mf_py, mf_pz = -7.1014, -4.1, 0
mf_vx, mf_vy, mf_vz = -114.0351, 208.7267, 5.0635
ShiftStarsAndCalculate(m12f, [mf_px, mf_py, mf_pz], [mf_vx, mf_vy, mf_vz])

print('*' * 40)
print('Total Stars: {0}'.format(len(m12f)))
print('...Truth accreted: {0}'.format(np.sum(m12f['AccretedLabel'] == 1)))
print('...Truth in HighFormDist: {0}'.format(np.sum(m12f['HighFormDist'] == 1)))
print('*' * 40)

# ************** Plotting ************************
myvars = ['Shifted_r', 'phi', 'Shifted_pz_true',
          'vr', 'vphi', 'Shifted_vz_true',
          'feh', 'alpha', 'age'
          ]
mylabels = [r'$r$', r'$\phi$', r'$z$',
            r'$v_r$', r'$v_{\phi}$', r'$v_z$',
            '[Fe/H]', '[Mg/Fe]', 'age'
            ]
binranges = [(4, 12), (-.5, .5), (-4, 4),
             (-750, 750), (-750, 750), (-500, 500),
             (-4.5, 2), (-1, 0.5), (5, 10.5)
             ]

plt.figure(figsize=(8, 3))
combined_bin = np.array([])
combined_bin_a = np.array([])
combined_bin_dist = np.array([])

for i, (label, binrange, var) in enumerate(zip(mylabels, binranges, myvars)):
    mfTrue, mfBins = np.histogram(m12f[var], bins=100, range=binrange)
    mfTrue_a, mfBins = np.histogram(m12f[m12f['AccretedLabel'] == 1][var],
                                    bins=100, range=binrange)
    mfTrue_dist, mfBins = np.histogram(m12f[m12f['HighFormDist'] == 1][var],
                                       bins=100, range=binrange)
    combined_bin = np.append(combined_bin, mfTrue)
    combined_bin_a = np.append(combined_bin_a, mfTrue_a)
    combined_bin_dist = np.append(combined_bin_dist, mfTrue_dist)
    plt.text(50 + i * 100, 3e6, label, ha='center', va='bottom')
    # plt.text(50 + i * 100, 8e1, '{0:0.3f}'.format(tmpchi[0]), ha='center', va='top')
    # plt.text(50 + i * 100, 2e1, '{0:0.1e}'.format(tmpchi[0] / tmpchi[1]),
    # ha='center', va='top')
plt.xlabel('Bin number')
plt.ylabel('Counts')
plt.plot(combined_bin, lw=0.5, color='k')
plt.plot(combined_bin_a, lw=1, color='C0')
plt.plot(combined_bin_dist, lw=1, color='C1', linestyle=':')
plt.plot([], [], color='k', lw=0.5, label='All stars')
plt.plot([], [], color='C0', label='Merger Tree')
plt.plot([], [], color='C1', linestyle=':', label=r'Formation Dist $> 25$')
plt.legend(loc='upper center', frameon=False, ncol=3)

plt.yscale('log')
plt.ylim(1e1, 1e9)
plt.xlim(-100, 950)
plt.title(r'm12f LSR 1 stars with $\delta\varpi / \varpi < 0.1$')
plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
           # ['', '', '', '', '', '', '', '', '', '', '']
           )

plt.savefig('../m12fDistVsMergerTree.pdf', bbox_inches='tight')

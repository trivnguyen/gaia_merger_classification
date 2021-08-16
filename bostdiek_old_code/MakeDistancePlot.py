import numpy as np

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

# HaloLabels = np.load('../data/m12iLSR0/StarLabels_accreted_virialized.npy')
AccretedLabels = np.load('../data/m12iLSR0/StarLabels_accreted.npy')
distance = np.load('../data/m12iLSR0/sliced_distance.npy')
StarIndices = np.load('../data/m12iLSR0/StarIndexList.npy')
MediumIndicies = np.load('../data/m12iLSR0/small_error_indicies_random.npy')

radial_velocity = np.load('../data/m12iLSR0/sliced_radial_velocity.npy')
MediumPlusVRIndicies = MediumIndicies[np.isfinite(radial_velocity[MediumIndicies])]
VRIndicies = StarIndices[np.isfinite(radial_velocity[StarIndices])]

plt.hist(distance, range=(0,4.5), bins=100, histtype='step',
         color='k')
# plt.hist(distance[HaloLabels==0], range=(0,4.5), bins=100, histtype='step',
#          color='k', ls='--')
plt.hist(distance[AccretedLabels==1], range=(0,4.5), bins=100, histtype='step',
         color='k', ls=':')
# plt.hist(distance[HaloLabels==1], range=(0,4.5), bins=100, histtype='step',
#          color='k', ls='--')

plt.hist(distance[MediumIndicies], range=(0,4.5), bins=100, histtype='step',
         color='C0')

# print(np.sum([HaloLabels == 1]))
# MediumHalo = HaloLabels[MediumIndicies]
MediumAccreted = AccretedLabels[MediumIndicies]

plt.hist(distance[MediumIndicies][MediumAccreted == 1], range=(0,4.5),
         bins=100, histtype='step',
         color='C0', ls=':')
# plt.hist(distance[MediumIndicies][MediumHalo == 1], range=(0,4.5),
#          bins=100, histtype='step',
#          color='C0', ls='--')

plt.hist(distance[MediumPlusVRIndicies], range=(0,4.5), bins=100, histtype='step',
         color='C1')
# SmallHalo = HaloLabels[MediumPlusVRIndicies]
SmallAccreted = AccretedLabels[MediumPlusVRIndicies]
plt.hist(distance[MediumPlusVRIndicies][SmallAccreted == 1], range=(0,4.5),
         bins=100, histtype='step',
         color='C1', ls=':')
# plt.hist(distance[MediumPlusVRIndicies][SmallHalo == 1], range=(0,4.5),
#          bins=100, histtype='step',
#          color='C1', ls='--')

pltall, = plt.plot([],[], color='k')
# plt.plot([],[], color='k', ls=':', label='Accreted and virialized')
pltmed, = plt.plot([],[], color='C0')
pltsmall, = plt.plot([],[], color='C1')

legend1 = plt.legend([pltall, pltmed, pltsmall],
                     ['All stars', 'Small parallax \nerror',
                      'Small parallax \nerror+' + r'$v_{r}$'],
                     loc=(0.07,0.00), frameon=False, fontsize=11
                     )

pltsolid, = plt.plot([],[], color='gray')
pltdotted, = plt.plot([],[], color='gray', ls=':')
# pltdashed, = plt.plot([],[], color='gray', ls='--')
legend2 = plt.legend([pltsolid, pltdotted],  # pltdashed],
                     ['Full set', 'Accreted'],   # 'Accreted and\nvirialized'],
                     ncol=2,
                     loc=(0.1, 0.85),
                     frameon=False, fontsize=11
                     )

plt.yscale('log')
plt.ylim(1e-2,2e9)
plt.xlabel('Distance [kpc]')
plt.ylabel('Stars per bin')
plt.title('FIRE - m12i - LSR0')
plt.minorticks_on()
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.savefig('../DistanceHistogramm12iLSR0.pdf', bbox_inches='tight')

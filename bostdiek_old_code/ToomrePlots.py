import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from sklearn.mixture import GaussianMixture
# Set the plotting styles

plt.rcParams.update({'font.family': 'cmr10', 'font.size': 13})
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.labelsize'] = 13
plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

vsundict = {
    'm12i0': [-17.8, 224.4, -3.9] * u.km / u.s,
    'm12i1': [-24.4, 210.9, -1.5] * u.km / u.s,
    'm12i2': [22.1, 206.5, 9.5] * u.km / u.s,
    'm12f0': [14.9, 227.9, 4.9] * u.km / u.s,
    'm12f1': [-3.4, 244.3, -5.1] * u.km / u.s,
}

galaxy = 'm12i'
LSR = '0'
ddir = '../data/m12iLSR'
for galaxy in ['m12f']:
    if galaxy == 'm12i':
        ddir = '../data/m12iLSR'
    else:
        ddir = '../data/m12flsr'
    for LSR in ['1']:
        # if galaxy == 'm12i' and LSR != '0':
        #     continue
        vsun = vsundict[galaxy + LSR]

        StarIndices = np.load(ddir + LSR + '/small_error_indicies_random.npy')
        radial_velocity = np.load(ddir + LSR + '/sliced_radial_velocity.npy')
        VRIndicies = StarIndices[np.isfinite(radial_velocity[StarIndices])]

        AccretedLabels = np.load(ddir + LSR + '/StarLabels_accreted.npy')[VRIndicies]
        if (galaxy == 'm12i') and (LSR == '0'):
            ltrue = np.load(ddir + LSR + '/sliced_l_true.npy')[VRIndicies]
            btrue = np.load(ddir + LSR + '/sliced_b_true.npy')[VRIndicies]
        else:
            ltrue = np.load(ddir + LSR + '/sliced_l.npy')[VRIndicies]
            btrue = np.load(ddir + LSR + '/sliced_b.npy')[VRIndicies]
        pmra = np.load(ddir + LSR + '/sliced_pmra.npy')[VRIndicies]
        pmdec = np.load(ddir + LSR + '/sliced_pmdec.npy')[VRIndicies]
        parallax = np.load(ddir + LSR + '/sliced_parallax.npy')[VRIndicies]
        feh = np.load(ddir + LSR + '/sliced_feh.npy')[VRIndicies]

        Stars = pd.DataFrame(
            {'l': ltrue,
             'b': btrue,
             'pmra': pmra,
             'pmdec': pmdec,
             'parallax': parallax,
             'AccretedLabel': AccretedLabels,
             'radial_velocity': radial_velocity[VRIndicies],
             'feh': feh
             }
        )

        gc = SkyCoord(l=Stars['l'].as_matrix() * u.degree,
                      b=Stars['b'].as_matrix() * u.degree,
                      frame='galactic'
                      )
        # Transfer this to the IRCS frame (ra, and dec)
        icrs1 = gc.transform_to('icrs')

        distances = (Stars['parallax'].values * u.mas).to(u.pc, u.parallax())
        icrs1 = coord.ICRS(ra=icrs1.ra,
                           dec=icrs1.dec,
                           distance=distances,
                           pm_dec=Stars['pmdec'].as_matrix() * u.mas / u.yr,
                           pm_ra_cosdec=Stars['pmra'].as_matrix() * u.mas / u.yr,
                           radial_velocity=Stars['radial_velocity'].as_matrix() * u.km / u.s
                           )
        gc2 = icrs1.transform_to(coord.Galactocentric(galcen_distance=8.2 * u.kpc,
                                                      z_sun=0 * u.kpc,
                                                      galcen_v_sun=coord.CartesianDifferential(
                                                          vsun[0], vsun[1], vsun[2])
                                                      ))
        Stars['v_x_inf'] = gc2.v_x.value
        Stars['v_y_inf'] = gc2.v_y.value
        Stars['v_z_inf'] = gc2.v_z.value

        Stars['p_x_inf'] = gc2.x.value / 1000
        Stars['p_y_inf'] = gc2.y.value / 1000
        Stars['p_z_inf'] = gc2.z.value / 1000

        Stars['r_inf'] = np.sqrt(Stars['p_x_inf']**2 +
                                 Stars['p_y_inf']**2)

        Stars['phi_inf'] = np.arctan2(Stars['p_y_inf'],
                                      Stars['p_x_inf'])

        Stars['vr_inf'] = (np.cos(Stars['phi_inf']) * Stars['v_x_inf'] +
                           np.sin(Stars['phi_inf']) * Stars['v_y_inf'])
        Stars['vphi_inf'] = (np.sin(Stars['phi_inf']) * Stars['v_x_inf'] -
                             np.cos(Stars['phi_inf']) * Stars['v_y_inf'])


        Stars['vxvz'] = np.sqrt(Stars['v_x_inf'] ** 2 +
                                Stars['v_z_inf'] ** 2
                                )
        Accreted = Stars[Stars['AccretedLabel'] == 1]
        InSitu = Stars[Stars['AccretedLabel'] == 0]

        Stars['ZMLabel'] = (np.abs(Stars['p_z_inf']) > 1.5) & (Stars['feh'] < -1.5)
        ZM = Stars[Stars['ZMLabel'] == 1]
        NotZM = Stars[Stars['ZMLabel'] == 0]

        # Kinematic Selection
        vLSR = vsun[1] * u.s / u.km

        KinematicLabel = ((Stars['v_y_inf'] - vLSR)**2 +
                          Stars['v_z_inf']**2 +
                          Stars['v_x_inf']**2) > vLSR**2
        Kin = Stars[KinematicLabel == 1]
        NotKin = Stars[KinematicLabel == 0]
        # Metallicity
        X = Stars[['v_x_inf', 'v_y_inf', 'v_z_inf']][:1000000]

        GaussMix = GaussianMixture(n_components=2,
                                   verbose_interval=100,
                                   init_params='random',
                                   random_state=2018,
                                   n_init=100
                                  )
        GaussMix.fit(X)

        Stars['GaussianMixtureElement'] = GaussMix.predict(Stars[['v_x_inf',
                                                                  'v_y_inf',
                                                                  'v_z_inf']])

        plt.figure(figsize=(4, 4))
        plt.hist(Stars.query('GaussianMixtureElement == 0')['vphi_inf'],
                 histtype='step', range=(-500, 500), bins=100, label='0')
        plt.hist(Stars.query('GaussianMixtureElement == 1')['vphi_inf'],
                 histtype='step', range=(-500, 500), bins=100, label='1')
        plt.legend()
        plt.yscale('log')
        plt.savefig('Test_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        MetallicitySelection = (Stars['GaussianMixtureElement'] == 0) & (Stars['feh'] < -1.0)
        if (galaxy == 'm12f') and (LSR == '0'):
            MetallicitySelection = (Stars['GaussianMixtureElement'] == 1) & (Stars['feh'] < -1.0)
        MS = Stars[MetallicitySelection == 1]
        NotMS = Stars[MetallicitySelection == 0]
        #  Tab 2
        plt.close()
        plt.clf()
        # plt.figure(figsize=(10, 4))
        # plt.subplot(2, 4, 1)
        # plt.hist2d(Accreted['v_y_inf'],
        #            Accreted['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} Accreted Stars'.format(len(Accreted)))
        #
        # plt.subplot(2, 4, 2)
        # plt.hist2d(Kin['v_y_inf'],
        #            Kin['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} Kinematic Selection'.format(len(Kin)))
        #
        # plt.subplot(2, 4, 3)
        # plt.hist2d(MS['v_y_inf'],
        #            MS['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} Metallicity Selection'.format(len(MS)))
        #
        # plt.subplot(2, 4, 4)
        # plt.hist2d(ZM['v_y_inf'],
        #            ZM['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # cb = plt.colorbar()
        # cb.set_label(r'Stars')
        # cb.ax.tick_params(size=0)
        # plt.title('{0} ZM Selection'.format(len(ZM)))
        #
        # plt.subplot(2, 4, 5)
        # plt.hist2d(InSitu['v_y_inf'],
        #            InSitu['vxvz'],
        #            bins=(np.linspace(-500, 500, 200),
        #                  np.linspace(0, 500, 100)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e5),
        #            vmin=1, vmax=1e5,
        #            cmap='magma'
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} In Situ Stars'.format(len(InSitu)))
        #
        # plt.subplot(2, 4, 6)
        # plt.hist2d(NotKin['v_y_inf'],
        #            NotKin['vxvz'],
        #            bins=(np.linspace(-500, 500, 200),
        #                  np.linspace(0, 500, 100)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e5),
        #            vmin=1, vmax=1e5,
        #            cmap='magma'
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} Not selected'.format(len(NotKin)))
        #
        # plt.subplot(2, 4, 7)
        # plt.hist2d(NotMS['v_y_inf'],
        #            NotMS['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e5),
        #            vmin=1, vmax=1e5,
        #            cmap='magma'
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # cb = plt.colorbar()
        # # cb.set_label(r'Stars')
        # # cb.ax.tick_params(size=0)
        # plt.title('{0} Not Selected'.format(len(NotMS)))
        #
        # plt.subplot(2, 4, 8)
        # plt.hist2d(NotZM['v_y_inf'],
        #            NotZM['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e5),
        #            vmin=1, vmax=1e5,
        #            cmap='magma'
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # cb = plt.colorbar()
        # cb.set_label(r'Stars')
        # cb.ax.tick_params(size=0)
        # plt.title('{0} Not Selected'.format(len(NotZM)))
        #
        # plt.tight_layout()
        # # plt.suptitle(galaxy + ' LSR' + LSR + ' Truth values',
        # #              fontsize=12,
        # #              y=1.05
        # #              )
        # plt.savefig('Toomre_{0}LSR{1}_all.pdf'.format(galaxy, LSR),
        #             bbox_inches='tight'
        #             )
        # plt.close()


        #  Only truth info
        plt.figure(figsize=(7, 3))
        gs0 = gs.GridSpec(1, 2, width_ratios=[4, 4], wspace=0.5)

        # plt.subplot(gs0[0])
        plt.subplot(1, 2, 1)
        cs = plt.hist2d(Accreted['v_y_inf'],
                        Accreted['vxvz'],
                        bins=(np.linspace(-500, 500, 150),
                              np.linspace(0, 600, 75)
                              ),
                        norm=LogNorm(vmin=1, vmax=1e3),
                        vmin=1, vmax=1e3
                        )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        plt.text(-450, 580, '{0} LSR{1}'.format(galaxy, LSR), ha='left', va='top')
        plt.ylim(0, 600)
        # cb = plt.colorbar()
        # cb.set_label(r'Counts')
        # cb.ax.tick_params(size=0)
        plt.title('Accreted')


        # plt.subplot(gs0[1])
        plt.subplot(1, 2, 2)
        plt.hist2d(InSitu['v_y_inf'],
                   InSitu['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # cb = plt.colorbar()
        # cb.set_label(r'Counts')
        # cb.ax.tick_params(size=0)
        plt.title(r'$\mathit{In}$ $\mathit{Situ}$')
        plt.ylim(0, 600)

        plt.tight_layout(w_pad=4)
        plt.savefig('ToomreTruthOnly_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()

        plt.clf()
        plt.figure(figsize=(4, 4))
        cb = plt.colorbar(cs[-1])
        cb.set_label(r'Stars / 50 (km/s)$^2$')
        cb.ax.tick_params(size=0)
        plt.axis('off')
        plt.savefig('ColorBarVert.pdf', bbox_inches='tight')
        plt.close()
        plt.clf()

        plt.clf()
        plt.figure(figsize=(4, 4))
        cb = plt.colorbar(cs[-1],
                          orientation='horizontal',
                          pad=-0.15
                          )
        cb.set_label(r'Stars / 50 (km/s)$^2$')
        cb.ax.tick_params(size=0)
        plt.axis('off')
        plt.savefig('ColorBarHorizontal.pdf', bbox_inches='tight')
        plt.close()
        plt.clf()

        ##################
        plt.figure(figsize=(3, 3))
        plt.hist2d(Accreted['v_y_inf'],
                   Accreted['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        plt.text(-450, 580, '{0} LSR{1}'.format(galaxy, LSR), ha='left', va='top')
        plt.title('Accreted')
        plt.savefig('ToomreTruthAccreted_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()
        #################
        plt.figure(figsize=(3, 3))
        plt.hist2d(InSitu['v_y_inf'],
                   InSitu['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        plt.title(r'$\mathit{In}$ $\mathit{Situ}$')
        plt.savefig('ToomreTruthInSitu_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()
        ##################
        plt.figure(figsize=(3, 3))
        plt.hist2d(Kin['v_y_inf'],
                   Kin['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        plt.text(-450, 580, '{0} LSR{1}'.format(galaxy, LSR), ha='left', va='top')
        plt.ylim(0, 600)
        # plt.title(r'$\mathbf{V}$ Selection')
        plt.savefig('ToomreKinematicSelection_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()
        ##################
        plt.figure(figsize=(3, 3))
        plt.hist2d(MS['v_y_inf'],
                   MS['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # plt.text(-450, 580, '{0} LSR{1}'.format(galaxy, LSR), ha='left', va='top')
        plt.ylim(0, 600)
        # plt.title(r'$\mathbf{VM}$ Selection')
        plt.savefig('ToomreVMSelection_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()
        ##################
        plt.figure(figsize=(3, 3))
        plt.hist2d(ZM['v_y_inf'],
                   ZM['vxvz'],
                   bins=(np.linspace(-500, 500, 150),
                         np.linspace(0, 600, 75)
                         ),
                   norm=LogNorm(vmin=1, vmax=1e3),
                   vmin=1, vmax=1e3
                   )
        myxspace = np.linspace(0, 500, 500)
        plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
                 ls='--',
                 color='k'
                 )
        plt.minorticks_on()
        plt.xlabel(r'$v_{y}$ [km/s]')
        plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # plt.text(-450, 580, '{0} LSR{1}'.format(galaxy, LSR), ha='left', va='top')
        plt.ylim(0, 600)
        # plt.title(r'$\mathbf{ZM}$ Selection')
        plt.savefig('ToomreZMSelection_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        plt.close()

        ##################
        # plt.figure(figsize=(9, 2.9))
        # gs0 = gs.GridSpec(1, 3, wspace=0.05)
        #
        # ax1 = plt.subplot(gs0[0])
        # plt.hist2d(Kin['v_y_inf'],
        #            Kin['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.xlim(-500, 500)
        # plt.xticks(np.linspace(-500, 500, 11), ['', '-400', '', '-200', '', '0',
        #                                         '', '200', '', 400, ''])
        # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # plt.text(-450, 480, 'm12i LSR0', ha='left', va='top')
        # # plt.title(r'$\mathbf{V}$ Selection')
        #
        # ax2 = plt.subplot(gs0[1], sharey=ax1)
        # plt.hist2d(MS['v_y_inf'],
        #            MS['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.xlim(-500, 500)
        # plt.xticks(np.linspace(-500, 500, 11), ['', '-400', '', '-200', '', '0',
        #                                         '', '200', '', 400, ''])
        # # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # # plt.text(-450, 480, 'm12i LSR0', ha='left', va='top')
        # # plt.title(r'$\mathbf{VM}$ Selection')
        # plt.setp(ax2.get_yticklabels(), visible=False)
        #
        # ax3 = plt.subplot(gs0[2], sharey=ax1)
        # plt.hist2d(ZM['v_y_inf'],
        #            ZM['vxvz'],
        #            bins=(np.linspace(-500, 500, 150),
        #                  np.linspace(0, 500, 75)
        #                  ),
        #            norm=LogNorm(vmin=1, vmax=1e3),
        #            vmin=1, vmax=1e3
        #            )
        # myxspace = np.linspace(0, 500, 500)
        # plt.plot(myxspace, np.sqrt(100**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.plot(myxspace, np.sqrt(vLSR**2 - (myxspace - vLSR)**2),
        #          ls='--',
        #          color='k'
        #          )
        # plt.minorticks_on()
        # plt.xlabel(r'$v_{y}$ [km/s]')
        # plt.xticks(np.linspace(-500, 500, 11), ['', '-400', '', '-200', '', '0',
        #                                         '', '200', '', 400, ''])
        # plt.xlim(-500, 500)
        # # plt.ylabel(r'$\sqrt{v_{x}^2 + v_{z}^2}$ [km/s]')
        # plt.text(450, 480, '{0} LSR{1}'.format(galaxy, LSR), ha='right', va='top')
        # plt.title(r'$\mathbf{ZM}$ Selection')
        # plt.setp(ax3.get_yticklabels(), visible=False)
        #
        # plt.savefig('ToomreTraditionalMethods_{0}.pdf'.format(galaxy + str(LSR)), bbox_inches='tight')
        # plt.close()

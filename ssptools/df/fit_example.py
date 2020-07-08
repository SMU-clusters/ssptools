#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from scipy.stats import multivariate_normal as mvn
from matplotlib import pyplot as plt
from sys import argv
import emcee
from corner import corner
from astropy.table import Table
from GMM import GMM2D

g = [2.114, -0.600]


def lnprob(p, D, lx, ly):
    M = GMM2D(lx=lx, ly=ly)
    x = D[0]
    y = D[1]
    dx = D[2]
    dy = D[3]
    xycorr = D[4]

    x0 = p[0]
    y0 = p[1]
    s0 = p[2]
    N = p[3]
    x1 = p[4]
    y1 = p[5]
    s1 = p[6]

    d0 = np.sqrt((x0 - g[0])**2 + (y0 - g[1])**2)
    d1 = np.sqrt((x1 - g[0])**2 + (y1 - g[1])**2)

    if s1 < 0 or s0 < 0 or N < 0 or N > len(x):
        return -np.Inf

    # The prior on s0 can be defined as the on-sky velocity dispersion give by
    # sig_PM ~ f * 10 km/s / DISTANCE[kpc] * 4.7
    # f is chosen to be 2, since some biases in the velocity dispersion
    # have been reported in Gaia DR2
    if d0 > 0.05 or s0 > s1 or s0 > 0.4:
        return -np.Inf

    L = (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0)
    L += N * M.pdf(x, y, dx, dy, xycorr, x1, y1, s1)
    lnL = np.ma.sum(np.ma.log(L))

    print(lnL, p)
    return lnL


f = Table.read('/home/eb0025/1851.hdf5')

ra = f['ra']
dec = f['dec']
pmra = f['pmra']
epmra = f['pmra_error']
pmdec = f['pmdec']
epmdec = f['pmdec_error']
epmrapmdec = f['pmra_pmdec_corr']
G = f['phot_g_mean_mag']

j = ~(np.isnan(pmra) | np.isnan(pmdec))
j *= (G < 16)

x = pmra[j]
y = pmdec[j]
dx = epmra[j]
dy = epmdec[j]
xycorr = epmrapmdec[j]

#plt.plot(x,y,'k.', ms=1)
#plt.show()
#exit()

D = np.array([x, y, dx, dy, xycorr])

if argv[1] == 'fit':

    p0 = [g[0], g[1], 0.0001, 1e3, 3, 3.0, 2]

    ndim, nwalkers = 7, 48
    p0 = [
        np.array(p0) + [
            0.01 * np.random.randn(), 0.01 * np.random.randn(),
            0.01 * np.abs(np.random.randn()), 100 * np.random.randn(), 0.1 *
            np.random.randn(), 0.1 * np.random.randn(), 1 * np.random.randn()
        ] for i in range(nwalkers)
    ]
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    lnprob,
                                    args=[D, 8, 8],
                                    threads=8)
    pos, prob, state = sampler.run_mcmc(p0, 800)
    sampler.reset()
    sampler.run_mcmc(pos, 500)

    chain = sampler.flatchain

    plt.figure()
    corner(chain)

    np.save('chain.npy', chain)

chain = np.load('chain.npy')
corner(chain,
       labels='pmra_cl pmdec_cl sig_cl N_bg pmra_bg pmdec_bg sig_bg'.split())

plt.figure()

M = GMM2D(8, 8)

x0 = np.mean(chain[:, 0])
y0 = np.mean(chain[:, 1])
s0 = np.mean(chain[:, 2])
N = np.mean(chain[:, 3])
x1 = np.mean(chain[:, 4])
y1 = np.mean(chain[:, 5])
s1 = np.mean(chain[:, 6])

# The membership probability is defined as P_GM/(P_GM + P_uniform)
L = (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0)
L /= (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0) + N * M.pdf(
    x, y, dx, dy, xycorr, x1, y1, s1)

prob = np.ma.array(L)
prob[np.isnan(prob)] = 0.0
prob[prob < 0] = 0.0

plt.figure()
print(np.min(prob), np.max(prob))
plt.hist(prob, bins=2000)

plt.figure()
#plt.scatter(x, y, c=prob, vmin=0.8, vmax=1.0)
plt.scatter(x, y, c=prob, vmin=0, vmax=1, s=10)
cb = plt.colorbar()
cb.set_label('Memb. Prob.')

from scipy.interpolate import griddata

xi = np.linspace(-2, 4, 100)
yi = np.linspace(-2, 4, 100)
zi = griddata((x, y), prob, (xi[None, :], yi[:, None]), method='linear')
CS = plt.contour(xi, yi, zi, [0.8, 0.9, 0.95, 0.99], colors='k')
plt.xlim(0, 3)
plt.ylim(-2, 2)

plt.xlabel('pmra')
plt.ylabel('pmdec')

plt.errorbar(x0, y0, xerr=np.std(chain[:, 0]), yerr=np.std(chain[:, 1]))

plt.gca().set_aspect('equal')

plt.show()

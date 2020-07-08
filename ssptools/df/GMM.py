#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


class GMM2D(object):
    """2D Gaussian Mixture Model (GMM)

    Note: scipy provides stats.multivaraite_normal, however it is not
    vectorized in an efficient way. This approach is faster, but less
    flexible.
    """

    def __init__(self, lx, ly, ndim=2):
        """ Initialize model

        :ndim: number of dimensions, currently can only be 2
        :lx, ly: size of the region of interest

        """
        if ndim > 2 or ndim <= 1:
            raise ValueError("Only works for ndim=2")

        self._ndim = ndim
        self._norm = np.sqrt((2 * np.pi) ** ndim)
        self._lx = lx
        self._ly = ly

    def _get_det(self, sig, corr, s0):
        """ Get determinant of the covariance matrix for N points

        :sig: sigma for each component (sig.shape == [ndim, N])
        :corr: correlations (len(corr) = N)
        """

        sx = sig[:, 0]
        sy = sig[:, 1]

        self._Det = np.sqrt(
            s0 ** 4
            + s0 ** 2 * sx ** 2
            + s0 ** 2 * sy ** 2
            + sx ** 2 * sy ** 2
            - corr ** 2 * sx ** 2 * sy ** 2
        )

    def pdf(self, x, y, dx, dy, xycorr, x0, y0, s0):

        ee = dy ** 2 * (x - x0) ** 2
        ee += s0 ** 2 * (x ** 2 - 2 * x * x0 + x0 ** 2 + (y - y0) ** 2)
        ee += -2 * xycorr * dx * dy * (x - x0) * (y - y0)
        ee += dx ** 2 * (y - y0) ** 2
        ee *= -0.5
        ee /= (
            s0 ** 4
            - (-1 + xycorr ** 2) * dx ** 2 * dy ** 2
            + s0 ** 2 * (dx ** 2 + dy ** 2)
        )

        self._get_det(np.array([dx, dy]).T, xycorr, s0)
        norm = 1.0 / (self._norm * self._Det)

        return norm * np.exp(ee)

    def uniformBG(self, N):
        """ Uniform background probability
        :lx: size of the regionn of interest in x
        :ly: same but for y
        """

        A = self._lx * self._ly
        return N / A


if __name__ == "__main__":

    from scipy.stats import multivariate_normal as mvn
    from matplotlib import pyplot as plt
    from scipy.optimize import minimize
    import emcee

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

        if s0 < 0 or N < 0 or N >= len(x):
            return -np.Inf

        L = (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0)
        L += N * M.uniformBG(1)
        lnL = np.ma.sum(np.ma.log(L))

        print(lnL, p)
        return lnL

    # Parameters for mock data
    x0 = 0
    y0 = 0
    s0 = 0.6  # Intrinsic spread
    Nbg = 500

    # Nbg stars in a 4x4 box
    xbg = -4 + 8 * np.random.rand(Nbg)
    ybg = -4 + 8 * np.random.rand(Nbg)

    # 2d gaussian assuming 0 intrinsic correlation
    ico = 0.0
    f = mvn.rvs(
        mean=(x0, y0),
        cov=[[s0 * s0, ico * s0 * s0], [ico * s0 * s0, s0 * s0]],
        size=5000,
    )

    x = np.r_[xbg, f[:, 0]]
    y = np.r_[ybg, f[:, 1]]

    x = x + np.random.randn(len(x)) * 0.1  # random noise
    y = y + np.random.randn(len(y)) * 0.1

    dx = dy = xycorr = np.zeros_like(x)
    dx += 0.05
    dy += 0.05
    xycorr += 0.0

    D = np.array([x, y, dx, dy, xycorr])
    p0 = [0, 0, 0.2, 500]

    ndim, nwalkers = 4, 32
    p0 = [
        np.array(p0)
        + [
            -0.2 + 0.1 * np.random.rand(),
            -0.2 + 0.1 * np.random.rand(),
            0.1 * np.random.rand(),
            0.5 * len(x) * np.random.rand(),
        ]
        for i in range(nwalkers)
    ]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[D, 8, 8], threads=16)
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 200)

    chain = sampler.flatchain
    from corner import corner

    plt.figure()
    corner(chain, truths=[x0, y0, s0, Nbg])

    plt.figure()

    M = GMM2D(8, 8)

    x0 = np.mean(chain[:, 0])
    y0 = np.mean(chain[:, 1])
    s0 = np.mean(chain[:, 2])
    N = np.mean(chain[:, 3])

    # The membership probability is defined as P_GM/(P_GM + P_uniform)
    L = (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0)
    L /= (len(x) - N) * M.pdf(x, y, dx, dy, xycorr, x0, y0, s0) + N * M.uniformBG(1)

    prob = L

    plt.figure()
    plt.scatter(x, y, c=prob, vmin=0)
    cb = plt.colorbar()
    cb.set_label("Memb. Prob.")

    plt.figure()
    plt.hist(prob, bins=20)

    plt.figure()
    j = prob > 0.5
    plt.plot(x[j], y[j], ".")

    plt.show()

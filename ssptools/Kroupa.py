# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
from numpy import power as pow
from scipy.integrate import quad

debug = False


def intinfo(a, mmin, mmax, XMIN, XMAX, xmin, xmax, II, II2):
    ostr = """a={:.2f}, mlim=[{:.2f}, {:.2f}],
Int range=[{:.2f}, {:.2f}],
Requested range=[{:.2f}, {:.2f}],
I0={:.6f}, I1={:.6f}\n""".format(
        a, mmin, mmax, XMIN, XMAX, xmin, xmax, II, II2
    )
    print(ostr.rstrip("\r"))


class Kroupa:
    """An piece-wise powerlaw mass function :math`f(x) \\sim x^{-\\alpha}`
    The PDF in normalized to 1.

    NOTE: only works for len(a) >= 2

    Args:
        a (list): the powerlaw exponents for each piece
        mlim (list): the mass ranges for each piece

    Returns:
        Object

    """

    def __init__(self, a=[1.3, 2.35], mlim=[0.08, 0.5, 120.0]):

        a = np.array(a)
        mlim = np.array(mlim)
        self._a = a
        self._mlim = mlim

        norm = np.zeros(len(a))
        area = np.zeros(len(a))
        C = np.zeros(len(a))

        # Assure piecewise continuity
        C[0] = pow(1.0 / mlim[1], -a[0])  # i=0
        C[1] = pow(1.0 / mlim[1], -a[1])  # i=1
        for i in range(2, len(a)):  # i>1
            C[i] = pow(1.0 / mlim[i], -a[i])
            for j in range(1, i):
                C[i] *= pow((mlim[j + 1] / mlim[j]), -a[j])

        # Loop through pieces to find normalization
        for i in range(len(a)):
            area[i] = self._mom0(mlim[i], mlim[i + 1], a[i])
        norm = area * C
        self._norm = 1.0 / np.sum(norm)
        self._area = norm * self._norm
        self._C = C

    def eval(self, X, N0=1):
        X = np.array([X])  # make it an array
        mlim = self._mlim
        a = self._a
        C = self._C
        norm = self._norm
        Y = np.array([])
        # Loop through pieces, maybe there is a more efficient way, but usually
        # npieces <= 3 for a Kroupa-like IMF so overhead is small
        for i in range(len(a)):
            j = (X >= mlim[i]) * (X < mlim[i + 1])
            y = N0 * norm * C[i] * pow(X[j], -a[i])
            Y = np.r_[Y, y]
        return Y

    def integral(self, xmin, xmax):
        """
        Compute the zeroth and first moment of the PDF
        """

        mlim = self._mlim
        a = self._a
        C = self._C
        norm = self._norm

        if xmin < mlim[0]:
            raise ValueError(
                "xmin is less than the lower domain boundary of {:.2f}".format(mlim[0])
            )
        if xmax > mlim[-1]:
            raise ValueError(
                "xmax is larger than the upper domain boundary of {:.2f}".format(
                    mlim[-1]
                )
            )

        imin = np.where(xmin / mlim >= 1)[0][-1]
        if xmax == mlim[-1]:
            imax = len(mlim) - 1
        else:
            imax = np.where(xmax / mlim < 1)[0][0]

        I = 0
        I2 = 0
        if imin == imax:
            I = norm * C[imin] * self._mom0(xmin, xmax, a[imin])
            I2 = norm * C[imin] * self._mom1(xmin, xmax, a[imin])
            i = imin
            if debug == True:
                print(
                    "part {i} of {j} {imin}, {imax}".format(
                        i=i + 1, j=imax - imin, imin=imin, imax=imax
                    )
                )
                intinfo(a[i], mlim[i], mlim[i + 1], XMIN, XMAX, xmin, xmax, II, II2)
        else:
            for h, i in enumerate(range(imin, imax)):
                XMIN = np.max((mlim[i], xmin))
                XMAX = np.min((mlim[i + 1], xmax))
                II = norm * C[i] * self._mom0(XMIN, XMAX, a[i])
                II2 = norm * C[i] * self._mom1(XMIN, XMAX, a[i])
                I += II
                I2 += II2
                if debug == True:
                    print(
                        "part {h} of {j} {imin}, {imax}".format(
                            h=h + 1, j=imax - imin, imin=imin, imax=imax
                        )
                    )
                    intinfo(a[i], mlim[i], mlim[i + 1], XMIN, XMAX, xmin, xmax, II, II2)
        return (I, I2)

    def sample(self, n):
        """Sample n stars from the mass function.

        Args:
            n (int): number of points to be sampled
        Return:
            mass (float or array): sampled mass values.
        """
        mlim = self._mlim
        a = self._a
        C = self._C
        norm = self._norm
        area = self._area

        X = np.random.rand(n)
        mass = np.array([])
        for i in range(len(a)):
            j = (X >= np.sum(area[0:i])) * (X < np.sum(area[0 : i + 1]))
            x = np.random.rand(len(X[j]))
            m = self._getmass(x, a[i], mlim[i], mlim[i + 1])
            mass = np.r_[mass, m]
        return mass

    def _mom0(self, xmin, xmax, a):
        """ First moment """
        if a == 1:
            return np.log(xmin) - np.log(xmax)
        else:
            return (pow(xmax, 1.0 - a) - pow(xmin, 1.0 - a)) / (1.0 - a)

    def _mom1(self, xmin, xmax, a):
        """ Second moment """
        if a == 0:
            return np.log(xmin) - np.log(xmax)
        else:
            return (pow(xmax, 2.0 - a) - pow(xmin, 2.0 - a)) / (2.0 - a)

    def _getmass(self, x, slope, xmin, xmax):
        """
        This is the sampler for a single power-law. In the form
        :math:`\\xi(m) = A\\,m^{\\alpha}`.

        Args:
            x (float or array): random variable
            slope (float): :math:`\\alpha` slope of the power law
            xmin, xmax (float, float): domain of the power-law PDF
        Return:
            mass (float or array): sampled mass values.
        """
        if slope == 1:
            A = np.log(xmax) - np.log(xmin)
        else:
            A = (1.0 / (1 - slope)) * (pow(xmax, 1.0 - slope) - pow(xmin, 1 - slope))
        mass = (1.0 - slope) * x * A + xmin ** (1.0 - slope)
        Z = 1.0 / (1.0 - slope)
        mass = mass ** Z
        return mass


if __name__ == "__main__":

    from scipy.integrate import quad, quadrature
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as p
    from sys import argv

    a = np.r_[np.zeros(5) + 1.3, np.zeros(10) + 2.3]
    mlim = np.r_[
        np.logspace(np.log10(0.08), np.log10(0.5), 6),
        np.logspace(np.log10(0.5), np.log10(120), 11)[1:],
    ]

    K = Kroupa(a=a, mlim=mlim)
    mlim = K._mlim

    eep, m, logT, logL, g, r, i = np.loadtxt(
        "/home/balbinot/ssptools/ssptools/data/models/MIST_iso_5b0533fab00b3.iso.cmd",
        usecols=(0, 2, 4, 6, 10, 11, 12),
        unpack=True,
    )

    ilogL = interp1d(m, logL, bounds_error=False)
    ilogT = interp1d(m, logT, bounds_error=False)
    ig = interp1d(m, g, bounds_error=False)
    ir = interp1d(m, r, bounds_error=False)
    ii = interp1d(m, i, bounds_error=False)

    def evalf(m):
        return K.eval(m) * (np.exp(ilogL(m)))

    # Test integration
    for mmin, mmax in zip(m[0:-1], m[1:]):
        I, I2 = K.integral(mmin, mmax)
        print(
            "{:.8f} {:.8f} {:.8f} {:.8f} {:.8f}".format(mmin, mmax, mmax - mmin, I, I2)
        )

    # Show synthetic CMD
    N = np.int(3.0e5)
    p.figure(figsize=(8, 8))
    mass = K.sample(N)  # sample some masses

    sg = ig(mass)
    sr = ir(mass)

    def err(x):
        ee = 0.002 + np.exp((x - 12) / 1.3)
        ee[ee > 0.5] = 0.5
        return ee

    sg += err(sg) * np.random.randn(len(sg))
    sr += err(sr) * np.random.randn(len(sr))

    p.plot(sg - sr, sg, "k.", ms=1, alpha=0.8)
    p.ylim(14, -2)
    p.xlim(-1.2, 3)

    # Show MF
    p.figure(figsize=(8, 8))
    X = np.linspace(mlim[0], mlim[-1] - 0.01, 10000)
    Y = K.eval(X, N0=N) / N
    print("average mass {:.4f}".format(np.mean(mass)))
    bins = np.linspace(np.log10(mlim[0]), np.log10(mlim[-1]), 100)
    p.hist(
        np.log10(mass),
        bins=bins,
        normed=True,
        log=True,
        histtype="stepfilled",
        color="red",
        alpha=0.5,
    )

    p.plot(np.log10(X), X * np.log(10) * Y, "k-")
    p.xlabel(r"log(m/M$_{\odot})$")
    p.ylabel(r"log($\xi$(m))")
    p.show()

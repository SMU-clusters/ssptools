#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np

SMALLNUMBER = 1e-9
dev = False

if dev:
    _ROOT = "/home/balbinot/ssptools/ssptools"
else:
    _ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """
    Get data from path relative to install dir.
    """
    return os.path.join(_ROOT, "data", path)


class IFMR:
    def __init__(self, FeHe):
        """
        Provides a class for the initial-final mass of black holes. Uses
        tabulated values for the polynomial approximations. These are based
        on SSE models at different metallicities.
        """

        bhgrid = np.loadtxt(get_data("sevtables/bhifmr.dat"))
        wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))
        self._bhgrid = bhgrid
        self._wdgrid = wdgrid
        self.FeHe = FeHe
        self.FeHe_WD = self.FeHe_BH = self.FeHe

        # Make sure chonse metallicity is withing the valid range or fall back
        # to best possible value
        self._check_feh_bounds()

        # Interpolate coefficients to chosen metallicity
        BHconstants = []
        for loop in range(1, len(bhgrid[0])):
            BHconstants.append(np.interp(FeHe, bhgrid[:, 0], bhgrid[:, loop]))
        BHconstants = np.array(BHconstants)

        WDconstants = []
        # dont interpolate, but get the closest model
        j = np.argmin(np.abs(self.FeHe_WD - wdgrid[:, 0]))
        WDconstants = wdgrid[j, :]

        self.m_min, self.B, self.C = BHconstants[:3]
        self.p1 = np.poly1d(BHconstants[3:5])
        self.p2 = np.poly1d(BHconstants[5:7])
        self.p3 = np.poly1d(BHconstants[7:])
        self.mBH_min = self.predict(self.m_min)

        self.wd_m_max = WDconstants[1]
        self.p4 = np.poly1d(WDconstants[2:])

    def _check_feh_bounds(self):
        feh = self.FeHe
        bhgrid = self._bhgrid
        wdgrid = self._wdgrid

        if feh < np.min(bhgrid[:, 0]):
            fback = np.min(bhgrid[:, 0])
            print("""{:.2f} is out of bounds for the BH metallicity grid,
                  falling back to minimum of {:.2f}""".format(feh, fback))
            self.FeHe_BH = fback

        elif feh > np.max(bhgrid[:, 0]):
            fback = np.max(bhgrid[:, 0])
            print("""{:.2f} is out of bounds for the BH metallicity grid,
                  falling back to maximum of {:.2f}""".format(feh, fback))
            self.FeHe_BH = fback

        if feh < np.min(wdgrid[:, 0]):
            fback = np.min(wdgrid[:, 0])
            print("""{:.2f} is out of bounds for the WD metallicity grid,
                  falling back to minimum of {:.2f}""".format(feh, fback))
            self.FeHe_WD = fback

        elif feh > np.max(wdgrid[:, 0]):
            fback = np.max(wdgrid[:, 0])
            print("""{:.2f} is out of bounds for the WD metallicity grid,
                  falling back to maximum of {:.2f}""".format(feh, fback))
            self.FeHe_WD = fback

    def predict(self, m_in):

        # BH pieces
        if m_in >= self.C:
            return self.p3(m_in)
        if m_in >= self.B:
            return self.p2(m_in)
        if m_in >= self.m_min:
            return self.p1(m_in)

        # NS piece (all NS have the same mass)
        if m_in <= self.m_min and m_in > self.wd_m_max:
            return 1.4

        # WD piece
        else:
            return self.p4(m_in)


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    X = np.arange(0.8, 100, 0.1)
    for feh in np.arange(-2.5, -0.5, 0.2):
        IFM = IFMR(feh)
        Y = []
        for x in X:
            Y.append(IFM.predict(x))

        plt.plot(X, Y, label="{:.2f}".format(feh))

    plt.loglog()
    plt.legend(loc="best")
    plt.xlabel("M_initial")
    plt.ylabel("M_final")

    plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import logging

import numpy as np
from scipy.interpolate import UnivariateSpline


_ROOT = pathlib.Path(__file__).parent


def get_data(path):
    """Get data from path relative to install dir."""
    return _ROOT / "data" / path


class IFMR:

    def __init__(self, FeH):
        """
        Provides a class for the initial-final mass of all stellar remnants.
        These are based on MIST and SSE models at different metallicities.
        """

        # ------------------------------------------------------------------
        # Check metallicity bounds
        # ------------------------------------------------------------------

        self.FeH = FeH

        self._check_feh_bounds()

        # ------------------------------------------------------------------
        # White Dwarfs
        # ------------------------------------------------------------------

        wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))

        # Get the closest model
        j = np.argmin(np.abs(self.FeH_WD - wdgrid[:, 0]))
        self.WD_m_max, WD_coeffs = wdgrid[j, 1], wdgrid[j, 2:]

        self.WD_poly = np.polynomial.Polynomial(WD_coeffs[::-1])

        # ------------------------------------------------------------------
        # Black Holes
        # ------------------------------------------------------------------

        bhifmr = np.loadtxt(get_data(f"sse/MP_FEH{self.FeH_BH:+.2f}.dat"))

        # Grab only stellar type 14
        bh_mi, bh_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

        self.BH_spline = UnivariateSpline(bh_mi, bh_mf, s=0)

        self.BH_m_min = bh_mi[0]
        self.mBH_min = self.predict(self.BH_m_min)

    def _check_feh_bounds(self):

        mssg = ("{0:.2f} is out of bounds for the {1} metallicity grid, "
                "falling back to {2} of {3:.2f}")

        wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))[:, 0]

        if self.FeH < (fback := np.min(wdgrid)):
            logging.debug(mssg.format(self.FeH, 'WD', 'minimum', fback))
            self.FeH_WD = fback

        elif self.FeH > (fback := np.max(wdgrid)):
            logging.debug(mssg.format(self.FeH, 'WD', 'maximum', fback))
            self.FeH_WD = fback
        else:
            self.FeH_WD = self.FeH

        bhgrid = np.array([float(fn.stem.split('FEH')[-1])
                           for fn in get_data('sse').glob('MP*dat')])

        if self.FeH < (fback := np.min(bhgrid)):
            logging.debug(mssg.format(self.FeH, 'BH', 'minimum', fback))
            self.FeH_BH = fback

        elif self.FeH > (fback := np.max(bhgrid)):
            logging.debug(mssg.format(self.FeH, 'BH', 'maximum', fback))
            self.FeH_BH = fback
        else:
            self.FeH_BH = self.FeH

    def predict(self, m_in):
        """Predict the final mass given the initial mass(es) `m_in`"""

        final = np.where(
            m_in >= self.BH_m_min, self.BH_spline(m_in),
            np.where(
                (self.WD_m_max < m_in) & (m_in <= self.BH_m_min), 1.4,
                self.WD_poly(np.array(m_in))
            )
        )

        # if m_in is a single float, reconvert to match
        if not final.shape:
            final = float(final)

        return final


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    X = np.arange(0.8, 100, 0.1)
    for feh in np.arange(-2.5, -0.5, 0.2):
        IFM = IFMR(feh)
        Y = IFM.predict(X)

        plt.plot(X, Y, label="{:.2f}".format(feh))

    plt.loglog()
    plt.legend(loc="best")
    plt.xlabel("M_initial")
    plt.ylabel("M_final")

    plt.show()

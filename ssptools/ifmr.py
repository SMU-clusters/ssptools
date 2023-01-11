#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import logging
import collections

import numpy as np
from scipy.interpolate import UnivariateSpline


bounds = collections.namedtuple('bounds', ('lower', 'upper'))

_ROOT = pathlib.Path(__file__).parent


def get_data(path):
    '''Get path of data from path relative to install dir.'''
    return _ROOT / "data" / path


class IFMR:
    '''
    Provides a class for the initial-final mass of all stellar remnants.
    These are based on MIST and SSE models at different metallicities.


    mBH_min : float
        Alias to BH_mf[0], for backwards compatibility

    mWD_max : float
        Alias to WD_mf[1], for backwards compatibility
    '''

    def __init__(self, FeH):

        # ------------------------------------------------------------------
        # Check metallicity bounds
        # ------------------------------------------------------------------

        self.FeH = FeH

        self._check_feh_bounds()

        # ------------------------------------------------------------------
        # Black Holes
        # ------------------------------------------------------------------

        bhifmr = np.loadtxt(get_data(f"sse/MP_FEH{self.FeH_BH:+.2f}.dat"))

        # Grab only stellar type 14
        BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

        # linear spline to avoid boundary effects near m_A, m_B, etc
        self._BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1)

        self.BH_mi = bounds(BH_mi[0], BH_mi[-1])

        # self.mBH_min, self.mBH_max = np.min(BH_mf), np.max(BH_mf)
        self.BH_mf = bounds(np.min(BH_mf), np.inf)
        self.mBH_min = self.BH_mf.lower

        # ------------------------------------------------------------------
        # White Dwarfs
        # ------------------------------------------------------------------

        wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))

        # Get the closest model
        j = np.argmin(np.abs(self.FeH_WD - wdgrid[:, 0]))
        WD_m_max, WD_coeffs = wdgrid[j, 1], wdgrid[j, 2:]

        self._WD_spline = np.polynomial.Polynomial(WD_coeffs[::-1])

        # TODO polynomial starts misbehaving far above 0, but don't know where
        self.WD_mi = bounds(0.0, WD_m_max)

        # TODO not technically correct due to possible bump in top of polynomial
        #   Should really stop using polynomials and use interpolated grid.
        self.WD_mf = bounds(0.0, self.predict(WD_m_max))
        self.mWD_max = self.WD_mf.upper

        # ------------------------------------------------------------------
        # Neutron Stars
        # ------------------------------------------------------------------

        self.NS_mi = bounds(self.WD_mi[1], self.BH_mi[0])
        self.NS_mf = bounds(1.4, 1.4)

    def _check_feh_bounds(self):
        '''Ensure FeH is within model grid and adjust otherwise'''

        mssg = ("{0:.2f} is out of bounds for the {1} metallicity grid, "
                "falling back to {2} of {3:.2f}")

        # ------------------------------------------------------------------
        # White Dwarfs
        # ------------------------------------------------------------------

        wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))[:, 0]

        if self.FeH < (fback := np.min(wdgrid)):
            logging.debug(mssg.format(self.FeH, 'WD', 'minimum', fback))
            self.FeH_WD = fback

        elif self.FeH > (fback := np.max(wdgrid)):
            logging.debug(mssg.format(self.FeH, 'WD', 'maximum', fback))
            self.FeH_WD = fback

        else:
            self.FeH_WD = self.FeH

        # ------------------------------------------------------------------
        # Black Holes
        # ------------------------------------------------------------------

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

    def predict_type(self, m_in):
        '''Predict the remnant type (WD, NS, BH) given the initial mass(es)'''

        rem_type = np.where(
            m_in >= self.BH_mi[0], 'BH',
            np.where(
                (self.WD_mi[1] < m_in) & (m_in <= self.BH_mi[0]), 'NS',
                'WD'
            )
        )

        return rem_type.tolist()

    def predict(self, m_in):
        '''Predict the final mass given the initial mass(es) `m_in`'''

        final = np.where(
            m_in >= self.BH_mi[0], self._BH_spline(m_in),
            np.where(
                (self.WD_mi[1] < m_in) & (m_in <= self.BH_mi[0]), 1.4,
                self._WD_spline(np.array(m_in))
            )
        )

        # If outside boundaries of the IFMR, warn user
        # TODO should splines extrapolate or cutoff outise bounds? (ext=0|3?)
        if np.any((m_in <= self.WD_mi[0]) | (m_in > self.BH_mi[1])):
            mssg = ("input mass exceeds IFMR grid, resulting mass is "
                    "extrapolated and may be incorrect")
            logging.warning(mssg)

        # if m_in is a single float, reconvert to match
        if not final.shape:
            final = float(final)

        return final

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import logging
import collections

import numpy as np
from scipy.interpolate import UnivariateSpline


__all__ = ["IFMR", "get_data"]


bounds = collections.namedtuple('bounds', ('lower', 'upper'))

_ROOT = pathlib.Path(__file__).parent


def get_data(path):
    '''Get path of data from path relative to install dir.'''
    return _ROOT / "data" / path


# --------------------------------------------------------------------------
# White Dwarf Initial-Final mass predictors
# --------------------------------------------------------------------------


def _MIST2018_WD_predictor(FeH):
    '''Return func(mi), WD_mi, WD_mf based on MIST 2018 interpolations
    '''

    wdgrid = np.loadtxt(get_data("sevtables/wdifmr.dat"))

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    mssg = ("{0:.2f} is out of bounds for the {1} metallicity grid, "
            "falling back to {2} of {3:.2f}")

    if FeH < (fback := np.min(wdgrid[:, 0])):
        logging.debug(mssg.format(FeH, 'WD', 'minimum', fback))
        FeH = fback

    elif FeH > (fback := np.max(wdgrid[:, 0])):
        logging.debug(mssg.format(FeH, 'WD', 'maximum', fback))
        FeH = fback

    # ----------------------------------------------------------------------
    # Compute the Polynomial fit based on the coeffs
    # ----------------------------------------------------------------------

    # Get the closest model
    j = np.argmin(np.abs(FeH - wdgrid[:, 0]))
    WD_m_max, WD_coeffs = wdgrid[j, 1], wdgrid[j, 2:]

    WD_spline = np.polynomial.Polynomial(WD_coeffs[::-1])

    # ----------------------------------------------------------------------
    # Compute the WD initial and final mass boundaries based on the polynomial
    # ----------------------------------------------------------------------

    # TODO polynomial starts misbehaving far above 0, but don't know where
    WD_mi = bounds(0.0, WD_m_max)

    # Compute the max/mins taking by derivative of polynomial
    WD_minmax = WD_spline.deriv().roots().real

    # Restrict x (initial mass) to between 1 and max_mi
    # TODO lower 1 is required to avoid polynomial effects, but is arbitrary
    #   really it won't matter, maximum will always be far above 1
    restr = (1. < WD_minmax) & (WD_minmax <= WD_m_max)

    # Determine the maximum WD final mass (including upper bound in case)
    WD_max = WD_spline(np.r_[WD_minmax[restr], WD_m_max]).max()

    WD_mf = bounds(0.0, WD_max)

    return WD_spline, WD_mi, WD_mf


# --------------------------------------------------------------------------
# Black Hole Initial-Final mass predictors
# --------------------------------------------------------------------------


def _F12_BH_predictor(FeH):

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    bhgrid = np.array([float(fn.stem.split('FEH')[-1])
                       for fn in get_data('sse').glob('MP*dat')])

    mssg = ("{0:.2f} is out of bounds for the {1} metallicity grid, "
            "falling back to {2} of {3:.2f}")

    if FeH < (fback := np.min(bhgrid)):
        logging.debug(mssg.format(FeH, 'BH', 'minimum', fback))
        FeH = fback

    elif FeH > (fback := np.max(bhgrid)):
        logging.debug(mssg.format(FeH, 'BH', 'maximum', fback))
        FeH = fback

    # ----------------------------------------------------------------------
    # Load BH IFMR values
    # ----------------------------------------------------------------------

    # TODO if 5e-3 < FeH < 0.0, this will put wrong sign on filename
    bhifmr = np.loadtxt(get_data(f"sse/MP_FEH{FeH:+.2f}.dat"))

    # Grab only stellar type 14 (BHs)
    BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

    # linear spline to avoid boundary effects near m_A, m_B, etc
    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


# --------------------------------------------------------------------------
# Combined IFMR class for all remnant types
# --------------------------------------------------------------------------


class IFMR:
    '''
    Provides a class for the initial-final mass of all stellar remnants.
    These are based on MIST and SSE models at different metallicities.


    mBH_min : float
        Alias to BH_mf[0], for backwards compatibility

    mWD_max : float
        Alias to WD_mf[1], for backwards compatibility
    '''

    def __repr__(self):
        return f"IFMR(FeH={self.FeH})"

    def __init__(self, FeH):

        # ------------------------------------------------------------------
        # Check metallicity bounds
        # ------------------------------------------------------------------

        self.FeH = FeH

        self._check_feh_bounds()

        # ------------------------------------------------------------------
        # Black Holes
        # ------------------------------------------------------------------

        self._BH_spline, self.BH_mi, self.BH_mf = _F12_BH_predictor(FeH)

        # self.mBH_min, self.mBH_max = np.min(BH_mf), np.max(BH_mf)
        self.mBH_min = self.BH_mf.lower

        # ------------------------------------------------------------------
        # White Dwarfs
        # ------------------------------------------------------------------

        self._WD_spline, self.WD_mi, self.WD_mf = _MIST2018_WD_predictor(FeH)

        self.mWD_max = self.WD_mf.upper

        # ------------------------------------------------------------------
        # Neutron Stars
        # ------------------------------------------------------------------

        self.NS_mi = bounds(self.WD_mi[1], self.BH_mi[0])
        self.NS_mf = bounds(1.4, 1.4)

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
        if np.any((m_in <= self.WD_mi[0]) | (m_in > self.BH_mi[1])):
            mssg = ("input mass exceeds IFMR grid, resulting mass is "
                    "extrapolated and may be very incorrect")
            logging.warning(mssg)

        # if m_in is a single float, reconvert to match
        if not final.shape:
            final = float(final)

        return final

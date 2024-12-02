#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import logging
import functools
import collections

import numpy as np
from scipy.interpolate import UnivariateSpline


__all__ = ["IFMR", "get_data"]


bounds = collections.namedtuple('bounds', ('lower', 'upper'))

_ROOT = pathlib.Path(__file__).parent


# SOME default COSMIC/BSE params which don't really affect single star IFMRs.
_DEFAULT_BSEDICT = dict(
    pts1=0.001, pts2=0.02, pts3=0.02,  # These 3 do matter but are just t-steps
    neta=0.5, bwind=0.0, hewind=0.5, beta=-1.0, xi=1.0,  acc2=1.5, alpha1=1.0,
    lambdaf=0., ceflag=1, cekickflag=2, cehestarflag=0, cemergeflag=0, qcflag=1,
    kickflag=0, bhflag=0, sigma=265.0, bhsigmafrac=1.0, sigmadiv=-20.0,
    ecsn=2.25, ecsn_mlow=1.4, polar_kick_angle=90, aic=1, ussn=1,
    mxns=2.5, rembar_massloss=0.5, wd_mass_lim=1, bhspinflag=0, bhspinmag=0.0,
    grflag=1, eddfac=1., gamma=-2., don_lim=-1, acc_lim=-1, tflag=1, ST_tide=1,
    ifflag=0, wdflag=1, epsnov=0.001, bdecayfac=1, bconst=3000, ck=1000,
    rejuv_fac=1.0, rejuvflag=0, bhms_coll_flag=0, htpmb=1, ST_cr=1, rtmsflag=0,
    qcrit_array=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    natal_kick_array=[[-100.0, -100.0, -100.0, -100.0, 0.0],
                      [-100.0, -100.0, -100.0, -100.0, 0.0]],
    fprimc_array=[2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0,
                  2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0,
                  2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0,
                  2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0, 2.0 / 21.0],
)


def get_data(path):
    '''Get path of data from path relative to install dir.'''
    return _ROOT / "data" / path


# --------------------------------------------------------------------------
# Predictor helpers
# --------------------------------------------------------------------------


def _line(mi, exponent, slope, scale):
    return (slope * mi**exponent) + scale


def _powerlaw_predictor(exponent, slope, scale, m_lower, m_upper):
    '''Simple power law function; `slope * m^exponent + scale`.'''

    if not ((0 < _line(m_lower, exponent, slope, scale) <= m_lower)
            and (0 < _line(m_upper, exponent, slope, scale) <= m_upper)):
        mssg = (f"Invalid line parameter (m={slope}, b={scale}, k={exponent}); "
                "Must be bounded within mf∈(0,mi] for m_lower < mi < m_upper.")
        raise ValueError(mssg)

    elif m_lower < 0.0:
        raise ValueError(f"Invalid line parameter; {m_lower=} cannot be < 0")

    elif m_lower > m_upper:
        raise ValueError(f"Invalid line parameter; "
                         f"{m_lower=} cannot be greater than {m_upper=}")

    return functools.partial(_line, exponent=exponent, slope=slope, scale=scale)


def _lines(mi, slopes, scales, exponents, m_breaks):
    bounds = [(lw_bnd <= mi) & (mi <= up_bnd)
              for lw_bnd, up_bnd in zip(m_breaks[:-1], m_breaks[1:])]

    vals = (slopes * mi[..., np.newaxis]**exponents + scales).T

    return np.select(bounds, vals, default=np.nan)  # nan?


def _broken_powerlaw_predictor(exponents, slopes, scales, m_breaks):
    '''Broken power law with N components, not guaranteed to be smooth.'''

    # Coerce all inputs to arrays, just in case
    exponents = np.asanyarray(exponents)
    slopes = np.asanyarray(slopes)
    scales = np.asanyarray(scales)
    m_breaks = np.asanyarray(m_breaks)

    # Just re-use the checks from _powerlaw_predictor, but ignore output
    for i in range(exponents.size):
        _powerlaw_predictor(
            exponent=exponents[i], slope=slopes[i], scale=scales[i],
            m_lower=m_breaks[i], m_upper=m_breaks[i + 1]
        )

    return functools.partial(_lines, slopes=slopes, scales=scales,
                             exponents=exponents, m_breaks=m_breaks)


# --------------------------------------------------------------------------
# White Dwarf Initial-Final mass predictors
# --------------------------------------------------------------------------


def _MIST18_WD_predictor(FeH):
    '''Return WD IFMR function, based on interpolated MIST 2018 models.'''

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


def _linear_WD_predictor(slope=0.15, scale=0.5, m_upper=5.5):
    '''Return simple linear WD IFMR function.'''

    WD_line = _powerlaw_predictor(1, slope, scale, m_lower=0.0, m_upper=m_upper)

    # Don't actually limit function, just suggest this limit
    WD_mi = bounds(0.0, m_upper)

    WD_mf = bounds(WD_line(0.0), WD_line(m_upper))

    return WD_line, WD_mi, WD_mf

# --------------------------------------------------------------------------
# Black Hole Initial-Final mass predictors
# --------------------------------------------------------------------------


def _check_IFMR_FeH_bounds(FeH, loc='sse'):
    # Sometimes this is needed elsewhere (e.g. kicks) so make separate func

    bhgrid = np.array([float(fn.stem.split('FEH')[-1])
                       for fn in get_data(loc).glob('*dat')])

    mssg = ("{0:.2f} is out of bounds for the {1} metallicity grid, "
            "falling back to {2} of {3:.2f}")

    if FeH < (fback := np.min(bhgrid)):
        logging.debug(mssg.format(FeH, 'BH', 'minimum', fback))
        return fback

    elif FeH > (fback := np.max(bhgrid)):
        logging.debug(mssg.format(FeH, 'BH', 'maximum', fback))
        return fback

    else:
        return FeH


def _Ba20_r_BH_predictor(FeH):
    '''Return BH IFMR function, based on Fryer+2012 rapid-SNe prescription.'''

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    FeH = _check_IFMR_FeH_bounds(FeH, loc='ifmr/uSSE_rapid')

    # ----------------------------------------------------------------------
    # Load BH IFMR values
    # ----------------------------------------------------------------------

    # TODO if 5e-3 < FeH < 0.0, this will put wrong sign on filename
    bhifmr = np.loadtxt(get_data(f"ifmr/uSSE_rapid/IFMR_FEH{FeH:+.2f}.dat"))

    # Grab only stellar type 14 (BHs)
    BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

    # linear spline to avoid boundary effects near m_A, m_B, etc
    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


def _Ba20_d_BH_predictor(FeH):
    '''Return BH IFMR function, based on Fryer+2012 delayed-SNe prescription.'''

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    FeH = _check_IFMR_FeH_bounds(FeH, loc='ifmr/uSSE_delayed')

    # ----------------------------------------------------------------------
    # Load BH IFMR values
    # ----------------------------------------------------------------------

    # TODO if 5e-3 < FeH < 0.0, this will put wrong sign on filename
    bhifmr = np.loadtxt(get_data(f"ifmr/uSSE_delayed/IFMR_FEH{FeH:+.2f}.dat"))

    # Grab only stellar type 14 (BHs)
    BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

    # linear spline to avoid boundary effects near m_A, m_B, etc
    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


def _COSMIC_full_BH_predictor(FeH, remnantflag=3, windflag=3, eddlimflag=0,
                              pisn=45., zsun=0.02, **bse_kwargs):
    from multiprocessing import cpu_count

    from cosmic.evolve import Evolve
    from cosmic.sample.initialbinarytable import InitialBinaryTable

    Z = zsun * 10**FeH

    mi_grid = np.arange(15, 150.1, 0.1)
    n = mi_grid.size
    t_final = 13700  # [Myr]

    init_table = InitialBinaryTable.InitialBinaries(
        m1=mi_grid, m2=np.zeros(n),
        porb=np.ones(n), ecc=np.ones(n),
        tphysf=np.ones(n) * t_final, kstar1=np.ones(n),
        kstar2=np.ones(n), metallicity=np.ones(n) * Z
    )

    # Package COSMIC/BSE args together
    important_kwargs = dict(remnantflag=remnantflag, windflag=windflag,
                            eddlimflag=eddlimflag, pisn=pisn, zsun=zsun)
    bsedict = _DEFAULT_BSEDICT | bse_kwargs | important_kwargs

    _, bcm, *_ = Evolve.evolve(initialbinarytable=init_table,
                               BSEDict=bsedict, nproc=cpu_count())

    bh_index = bcm[bcm['kstar_1'] == 14].index
    BH_mi = bcm[bcm.index.isin(bh_index)][::2]['mass_1'].values
    BH_mf = bcm[bcm.index.isin(bh_index)][1::2]['mass_1'].values

    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


def _COSMIC_r_BH_predictor(FeH):

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    FeH = _check_IFMR_FeH_bounds(FeH, loc='ifmr/COSMIC_rapid')

    # ----------------------------------------------------------------------
    # Load BH IFMR values
    # ----------------------------------------------------------------------

    # TODO if 5e-3 < FeH < 0.0, this will put wrong sign on filename
    bhifmr = np.loadtxt(get_data(f"ifmr/COSMIC_rapid/IFMR_FEH{FeH:+.2f}.dat"))

    # Grab only stellar type 14 (BHs)
    BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

    # linear spline to avoid boundary effects near m_A, m_B, etc
    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


def _COSMIC_d_BH_predictor(FeH):

    # ----------------------------------------------------------------------
    # Check [Fe/H] is within model grid and adjust otherwise
    # ----------------------------------------------------------------------

    FeH = _check_IFMR_FeH_bounds(FeH, loc='ifmr/COSMIC_delayed')

    # ----------------------------------------------------------------------
    # Load BH IFMR values
    # ----------------------------------------------------------------------

    # TODO if 5e-3 < FeH < 0.0, this will put wrong sign on filename
    bhifmr = np.loadtxt(get_data(f"ifmr/COSMIC_delayed/IFMR_FEH{FeH:+.2f}.dat"))

    # Grab only stellar type 14 (BHs)
    BH_mi, BH_mf = bhifmr[:, :2][bhifmr[:, 2] == 14].T

    # linear spline to avoid boundary effects near m_A, m_B, etc
    BH_spline = UnivariateSpline(BH_mi, BH_mf, s=0, k=1, ext=0)

    BH_mi = bounds(BH_mi[0], BH_mi[-1])

    BH_mf = bounds(np.min(BH_mf), np.inf)

    return BH_spline, BH_mi, BH_mf


def _linear_BH_predictor(slope=0.4, scale=0.7, m_lower=19):
    '''Return simple linear BH IFMR function.'''

    BH_line = _powerlaw_predictor(1, slope, scale,
                                  m_lower=m_lower, m_upper=np.inf)

    BH_mi = bounds(m_lower, np.inf)

    BH_mf = bounds(BH_line(m_lower), np.inf)

    return BH_line, BH_mi, BH_mf


def _powerlaw_BH_predictor(exponent=3, slope=3e-5, scale=14, m_lower=19):
    '''Return simple single power law BH IFMR function.'''

    BH_line = _powerlaw_predictor(exponent, slope, scale,
                                  m_lower=m_lower, m_upper=np.inf)

    BH_mi = bounds(m_lower, np.inf)

    BH_mf = bounds(BH_line(m_lower), np.inf)

    return BH_line, BH_mi, BH_mf


def _brokenpl_BH_predictor(exponents=[1, 3, 1], slopes=[1, 6e-4, 0.43],
                           scales=[0, 0, 0], m_breaks=[20, 22, 36, 100]):
    '''Return N-component power law BH IFMR function.'''

    # TODO wont accept lists, need to coerce in _lines
    BH_line = _broken_powerlaw_predictor(exponents, slopes, scales, m_breaks)

    BH_mi = bounds(m_breaks[0], m_breaks[-1])

    # Manually check the bound masses for each line, to find max/min values
    # This is necessary because the discontinuity at m_break ruins minimizers
    mfl, mfu = np.inf, 0.
    for i in range(len(exponents)):
        vl, vr = _powerlaw_predictor(
            exponent=exponents[i], slope=slopes[i], scale=scales[i],
            m_lower=m_breaks[i], m_upper=m_breaks[i + 1]
        )(np.asanyarray(m_breaks[i:i + 2]))

        mfl = min(mfl, vl, vr)
        mfu = max(mfu, vl, vr)

    BH_mf = bounds(mfl, mfu)

    return BH_line, BH_mi, BH_mf


# --------------------------------------------------------------------------
# Combined IFMR class for all remnant types
# --------------------------------------------------------------------------


class IFMR:
    '''Initial-final mass relations for stellar remnants.

    Provides methods for determining the final (individual) remnant mass and
    type for a given initial stellar mass, based on a number of available
    algorithms and prescriptions.

    Parameters
    ----------
    FeH : float
        Metallicity. Note that most methods which require a metallicity will be
        based on a grid which this metallicity will be interpolated onto.
        Values far outside the edge of these grids may behave unexpectedly.

    NS_mass : float, optional
        The (constant) final mass to be used for all neutron stars. Defaults
        to the typically used value of 1.4 Msun.

    WD_method : {"mist18", "linear"}, optional
        The White Dwarf IFMR algorithm to use. Defaults to the MIST 2018 method.

    WD_kwargs : dict, optional
        All arguments passed to the WD IFMR algorithm. See the specified
        functions for information on all required methods.
        This will fail if the required arguments are not passed here.

    BH_method : {"fryer12", "linear", "powerlaw", "brokenpowerlaw"}, optional
        The Black Hole IFMR algorithm to use. Defaults to the Rapid supernovae
        schema presented by Fryer+2012.

    BH_kwargs : dict, optional
        All arguments passed to the BH IFMR algorithm. See the specified
        functions for information on all required methods.
        This will fail if the required arguments are not passed here.

    Attributes
    ----------
    BH_mi : bounds
        The mass bounds defining the lower and upper bounds of stars which
        will form black holes.

    WD_mi : bounds
        The mass bounds defining the lower and upper bounds of stars which
        will form white dwarfs.

    WD_mi : bounds
        The mass bounds defining the lower and upper bounds of stars which
        will form neutron stars. This is defined as the space between the WD
        and BH bounds.

    BH_mf : bounds
        The mass bounds defining the possible (final) masses of black holes.
        Note that this is *not* necessarily the same as `IFMR.predict(BH_mi)`.

    WD_mf : bounds
        The mass bounds defining the possible (final) masses of White dwarfs.
        Note that this is *not* necessarily the same as `IFMR.predict(WD_mi)`.

    NS_mf : bounds
        The mass bounds defining the possible (final) masses of neutron stars.
        This is, by definition, simply (NS_mass, NS_mass).

    mBH_min : float
        Alias to BH_mf.lower, for backwards compatibility.

    mWD_max : float
        Alias to WD_mf.upper, for backwards compatibility.

    See Also
    --------
    _MIST18_WD_predictor : WD IFMR algorithm based on MIST 2018 models.
    _linear_WD_predictor : Linear WD IFMR algorithm.
    _F12_BH_predictor : BH IFMR algorithm based on Fryer+2012 prescription.
    _linear_BH_predictor : Linear BH IFMR algorithm.
    _powerlaw_BH_predictor : Single power law BH IFMR algorithm.
    _brokenpl_BH_predictor : Multiple power law BH IFMR algorithm.
    '''

    def __repr__(self):
        return f"IFMR(FeH={self.FeH})"

    def __init__(self, FeH, *, NS_mass=1.4,
                 WD_method='mist18', WD_kwargs=None,
                 BH_method='banerjee20', BH_kwargs=None):

        self.FeH = FeH

        # ------------------------------------------------------------------
        # Black Holes
        # ------------------------------------------------------------------

        if BH_kwargs is None:
            BH_kwargs = dict()

        match BH_method.casefold():

            case ('nbody7' | 'banerjee20' | 'ba20'
                  | 'nbody7-rapid' | 'banerjee20-rapid' | 'ba20-rapid'):
                BH_kwargs.setdefault('FeH', FeH)
                BH_func, BH_mi, BH_mf, = _Ba20_r_BH_predictor(**BH_kwargs)

            case 'nbody7-delayed' | 'banerjee20-delayed' | 'ba20-delayed':
                BH_kwargs.setdefault('FeH', FeH)
                BH_func, BH_mi, BH_mf, = _Ba20_d_BH_predictor(**BH_kwargs)

            case 'cosmic':
                BH_kwargs.setdefault('FeH', FeH)
                BH_func, BH_mi, BH_mf, = _COSMIC_full_BH_predictor(**BH_kwargs)

            case 'cosmic-rapid':
                BH_kwargs.setdefault('FeH', FeH)
                BH_func, BH_mi, BH_mf, = _COSMIC_r_BH_predictor(**BH_kwargs)

            case 'cosmic-delayed':
                BH_kwargs.setdefault('FeH', FeH)
                BH_func, BH_mi, BH_mf, = _COSMIC_d_BH_predictor(**BH_kwargs)

            case 'linear' | 'line':
                BH_func, BH_mi, BH_mf, = _linear_BH_predictor(**BH_kwargs)

            case 'power' | 'powerlaw' | 'pl':
                BH_func, BH_mi, BH_mf, = _powerlaw_BH_predictor(**BH_kwargs)

            case 'broken' | 'brokenpowerlaw' | 'bpl':
                BH_func, BH_mi, BH_mf, = _brokenpl_BH_predictor(**BH_kwargs)

            case _:
                raise ValueError(f"Invalid BH IFMR method: {BH_method}")

        self._BH_spline, self.BH_mi, self.BH_mf = BH_func, BH_mi, BH_mf

        # self.mBH_min, self.mBH_max = np.min(BH_mf), np.max(BH_mf)
        self.mBH_min = self.BH_mf.lower

        # ------------------------------------------------------------------
        # White Dwarfs
        # ------------------------------------------------------------------

        if WD_kwargs is None:
            WD_kwargs = dict()

        match WD_method.casefold():

            case 'mist18' | 'm18' | 'mist2018':
                WD_kwargs.setdefault('FeH', FeH)
                WD_func, WD_mi, WD_mf, = _MIST18_WD_predictor(**WD_kwargs)

            case 'linear' | 'line':
                WD_func, WD_mi, WD_mf, = _linear_WD_predictor(**WD_kwargs)

            case _:
                raise ValueError(f"Invalid WD IFMR method: {WD_method}")

        self._WD_spline, self.WD_mi, self.WD_mf = WD_func, WD_mi, WD_mf

        self.mWD_max = self.WD_mf.upper

        # ------------------------------------------------------------------
        # Neutron Stars
        # ------------------------------------------------------------------

        self._NS_mass = NS_mass

        self.NS_mi = bounds(self.WD_mi[1], self.BH_mi[0])
        self.NS_mf = bounds(self._NS_mass, self._NS_mass)

        if self.WD_mi.upper > self.BH_mi.lower:
            mssg = (f"Invalid initial mass bounds, "
                    f"WD upper bound ({self.WD_mi.upper}) cannot be higher "
                    f"than BH lower bound ({self.BH_mi.lower})")
            raise ValueError(mssg)

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
        '''Predict the final mass given the initial mass(es) `m_in`.'''

        final = np.where(
            m_in >= self.BH_mi[0], self._BH_spline(m_in),
            np.where(
                (self.WD_mi[1] < m_in) & (m_in <= self.BH_mi[0]), self._NS_mass,
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

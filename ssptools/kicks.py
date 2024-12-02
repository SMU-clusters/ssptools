#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ifmr import get_data

import numpy as np
from scipy.special import erf
import scipy.interpolate as interp


__all__ = ["natal_kicks"]


def _maxwellian_retention_frac(m, vesc, FeH, vdisp=265., vmax=1000, *,
                               SNe_method='rapid'):
    '''Retention fraction alg. based on a Maxwellian kick velocity distribution.

    This method is based on the assumption that the natal kick velocity is
    drawn from a Maxwellian distribution with a certain kick dispersion
    scaled down by a fallback fraction, as described by Fryer et al. (2012).

    The fraction of black holes retained in each mass bin is then found by
    integrating the kick velocity distribution from 0 to the estimated initial
    system escape velocity.

    Parameters
    ----------
    m : float
        The mean mass of a BH bin.

    vesc : float
        The initial escape velocity of the cluster.

    vdisp : float, optional
        The dispersion of the Maxwellian kick velocity distribution. Defaults
        to 265 km/s, as typically used for neutron stars.

    vmax : float, optional
        The maximum velocity bound of the Maxwellian distribution.
        As long as it's well above the escape velocity, this is not important.

    SNe_method : {'rapid', 'delayed'}, optional
        Whether to use the "rapid" (default) or "delayed" supernovae
        prescriptions described by Fryer+2012 to determine the fallback
        fraction as a function of the black hole mass.

    Returns
    -------
    float
        The retention fraction of BHs of this mass.

    Notes
    -----
    This function is *not* currently vectorized, and each mass must be
    given as an individual float, in contrast to other kick methods.
    '''

    def _maxwellian(x, a):
        norm = np.sqrt(2 / np.pi)
        exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
        return norm * exponent / a ** 3

    fb = _F12_fallback_frac(FeH, SNe_method=SNe_method)(m)

    if fb >= 1.0:  # TODO breaks on m is array
        return 1.0

    # Integrate over the Maxwellian up to the escape velocity
    v_space = np.linspace(0, vmax, 1000)

    # TODO might be a quicker way to numerically integrate than a spline
    retention = interp.UnivariateSpline(
        x=v_space,
        y=_maxwellian(v_space, vdisp * (1 - fb)),
        s=0,
        k=3,
    ).integral(0, vesc)

    return retention


def _F12_fallback_frac(FeH, *, SNe_method='rapid'):
    '''Get the fallback fraction for this mass, interpolated from SSE models
    based on the prescription from Fryer 2012.
    Note there are no checks on FeH here, so make sure it's within the grid.
    
    SNe_method must be one of rapid or delayed.
    '''

    # load in the ifmr data to interpolate fb from mr
    # feh_path = get_data(f"sse/MP_FEH{FeH:+.2f}.dat")  # .2f snaps to the grid
    feh_path = get_data(f"ifmr/uSSE_{SNe_method}/IFMR_FEH{FeH:+.2f}.dat")

    # load in the data (only final remnant mass and fbac)
    fb_grid = np.loadtxt(feh_path, usecols=(1, 3), unpack=True)

    # Interpolate the mr-fb grid
    return interp.interp1d(fb_grid[0], fb_grid[1], kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))


def _sigmoid_retention_frac(m, slope, scale):
    r'''Retention fraction alg. based on a paramatrized sigmoid function.

    This method is based on a simple parametrization of the relationship
    between the retention fraction and the BH mass as a sigmoid function,
    increasing smoothly between 0 and 1 around a scale mass.

   .. math::
        f_{\mathrm{ret}}(m) = \operatorname{erf}\left(
            e^{\mathrm{slope}\ (m - \mathrm{scale})}
        \right)

    Parameters
    ----------
    m : float
        The mean mass of a BH bin.

    slope : float
        The "slope" of the sigmoid function, defining the "sharpness" of the
        increase. A value of 0 is completely flat (at fret=erf(1)~0.85), an
        increasingly positive value approaches a step function at the scale
        mass, and a negative value will retain more low-mass bins than high.

    scale : float
        The scale-mass of the sigmoid function, defining the approximate mass
        of the turn-over from 0 to 1 (e.g. the position of the step function
        as the slope approaches infinity).

    Returns
    -------
    float
        The retention fraction of BHs of this mass.
    '''
    return erf(np.exp(slope * (m - scale)))


def _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs):
    '''
    Natal kicks without a modulating total mass to eject, just ejecting
    all mass in each bin based on the retention fraction
    M_BH,f = M_BH,i * fret(mj)
    '''
    natal_ejecta = 0.0

    for j in range(Mr_BH.size):

        # Skip the bin if its empty
        if Nr_BH[j] < 0.1:
            continue

        # Compute retention fraction
        retention = f_ret(Mr_BH[j] / Nr_BH[j], **ret_kwargs)

        # keep track of how much we eject
        natal_ejecta += Mr_BH[j] * (1 - retention)

        # eject the mass
        Mr_BH[j] *= retention
        Nr_BH[j] *= retention

    return Mr_BH, Nr_BH, natal_ejecta


def natal_kicks(Mr_BH, Nr_BH, f_kick=None, method='fryer2012', **ret_kwargs):
    r'''Computes the effects of BH natal-kicks on the mass and number of BHs

    Determines the effects of BH natal-kicks, and distributes said kicks
    throughout the different BH mass bins, based on the given natal kick
    algorithm. In general, BHs are preferentially lost in the low mass
    bins, with the lowest masses being entirely kicked, and the highest masses
    being entirely retained.

    Two natal kick algorithms are currently available. Both methods require
    different arguments, which can be passed to `ret_kwargs`.
    See the respective retention functions for details on these arguments.

    The first is based on the assumption that the kick velocity is drawn from
    a Maxwellian distribution with a certain kick dispersion (scaled down by
    a “fallback fraction” interpolated from a grid of SSE models). The
    fraction of black holes retained in each mass bin is then found by
    integrating the kick velocity distribution from 0 to the estimated initial
    system escape velocity. See Fryer et al. (2012) for more information.

    A second, more directly flexible method is not based on any modelled BH
    physics, but simply determines the retention fraction of BHs in each bin
    based on the simple sigmoid function
    :math:`f_{ret}(m)=\operatorname{erf}\left(e^{a(m-b)}\right)`.

    Both input BH arrays are modified *in place*, as well as returned.

    Parameters
    ----------
    Mr_BH : ndarray
        Array[nbin] of the total initial masses of black holes in each
        BH mass bin

    Nr_BH : ndarray
        Array[nbin] of the total initial numbers of black holes in each
        BH mass bin

    f_kick : float, optional
        Unused.

    method : {'sigmoid', 'maxwellian'}, optional
        Natal kick algorithm to use, defining the retention fraction as a
        function of mean bin mass. Defaults to the Maxwellian method.

    **ret_kwargs : dict, optional
        All other arguments are passed to the retention fraction function.

    Returns
    -------
    Mr_BH : ndarray
        Array[nbin] of the total final masses of black holes in each
        BH mass bin, after natal kicks

    Nr_BH : ndarray
        Array[nbin] of the total final numbers of black holes in each
        BH mass bin, after natal kicks

    natal_ejecta : float
        The total mass of BHs ejected through natal kicks

    See Also
    --------
    _maxwellian_retention_frac : Maxwellian retention fraction algorithm.
    _sigmoid_retention_frac : Sigmoid retention fraction algorithm.
    '''

    match method.casefold():

        case 'sigmoid':
            f_ret = _sigmoid_retention_frac

        case 'maxwellian' | 'f12' | 'fryer2012':
            f_ret = _maxwellian_retention_frac

        case _:
            raise ValueError(f"Invalid kick distribution method: {method}")

    # If no given total kick fraction, use old-style of directly using f_ret
    if f_kick is None:
        return _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs)

    else:
        # Not really possible to distribute cleanly, maybe just abandon
        raise NotImplementedError

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ifmr import get_data

import numpy as np
from scipy.special import erf
import scipy.interpolate as interp


__all__ = ["natal_kicks"]


def _maxwellian_retention_frac(m, vesc, FeH, vdisp=265., vmax=1000):

    def _maxwellian(x, a):
        norm = np.sqrt(2 / np.pi)
        exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
        return norm * exponent / a ** 3

    fb = _F12_fallback_frac(FeH)(m)

    if fb >= 1.0:
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


def _sigmoid_retention_frac(m, slope, scale):
    # TODO better coefficient names than a, b
    # b sets left-right transform, i.e. ~m of sigmoid turn
    # a sets "slope" of turn, a=0 is ~flat a<0 loses more high mass bin than low
    return erf(np.exp(slope * (m - scale)))


def _F12_fallback_frac(FeH):

    # load in the ifmr data to interpolate fb from mr
    feh_path = get_data(f"sse/MP_FEH{FeH:+.2f}.dat")  # .2f snaps to the grid

    # load in the data
    fb_grid = np.loadtxt(feh_path, usecols=(1, 3), unpack=True)

    # Interpolate the mr-fb grid
    return interp.interp1d(fb_grid[0], fb_grid[1], kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))


def _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs):
    '''natal kicks without a modulating total mass to eject, just ejecting
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


def natal_kicks(Mr_BH, Nr_BH, f_kick, method='sigmoid', **ret_kwargs):

    if method.lower() == 'sigmoid':
        f_ret = _sigmoid_retention_frac

    elif method.lower() == 'maxwellian':
        f_ret = _maxwellian_retention_frac

    else:
        raise ValueError(f"Invalid kick distribution method: {method}")

    # If no given total kick fraction, use old-style of directly using f_ret
    if f_kick is None:
        return _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs)

    else:
        raise NotImplementedError

    # M_tot = Mr_BH.sum()
    # # Total amount to kick
    # M_kick = f_kick * M_tot

    # try:
    #     retentions = np.array([
    #         f_ret(Mr_BH[j] / Nr_BH[j], **ret_kwargs) for j in range(Mr_BH.size)
    #     ])
    # except TypeError as err:
    #     mssg = f"missing required kwargs for kick method '{method}': {err}"
    #     raise TypeError(mssg)

    # # norm = np.sum(retentions * Mr_BH) / (M_tot * (1 - f_kick))

    # # Mr_BH *= retentions / norm
    # # Nr_BH *= retentions / norm

    # print(f'{retentions=}')
    # print(f'{(1 - f_kick)=}')
    # print(f'{retentions * (1 - f_kick)=}')
    # print(f'{Mr_BH * retentions * (1 - f_kick)=}')

    # Mr_BH *= retentions * (1 - f_kick)
    # Nr_BH *= retentions * (1 - f_kick)

    # return Mr_BH, Nr_BH, M_kick

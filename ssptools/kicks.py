#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ifmr import get_data

import numpy as np
import scipy.interpolate as interp


__all__ = ["maxwellian_natal_kicks"]


def _maxwellian_retention_frac(vesc, fb, vdisp=265., vmax=1000):

    def _maxwellian(x, a):
        norm = np.sqrt(2 / np.pi)
        exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
        return norm * exponent / a ** 3

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


def _fallback_frac(FeH):

    # load in the ifmr data to interpolate fb from mr
    feh_path = get_data(f"sse/MP_FEH{FeH:+.2f}.dat")  # .2f snaps to the grid

    # load in the data
    fb_grid = np.loadtxt(feh_path, usecols=(1, 3), unpack=True)

    # Interpolate the mr-fb grid
    return interp.interp1d(fb_grid[0], fb_grid[1], kind="linear",
                           bounds_error=False, fill_value=(0.0, 1.0))


def maxwellian_natal_kicks(Mr_BH, Nr_BH, vesc, FeH, *, vdisp=265.):
    '''Computes the effects of BH natal-kicks on the mass and number of BHs

    Determines the effects of BH natal-kicks based on the fallback fraction
    and escape velocity of this population.

    Both input arrays are modified *in place*, as well as returned.

    Parameters
    ----------
    Mr_BH : ndarray
        Array[nbin] of the total initial masses of black holes in each
        BH mass bin

    Nr_BH : ndarray
        Array[nbin] of the total initial numbers of black holes in each
        BH mass bin

    vesc : float
        Initial cluster escape velocity, in km/s, for use in the
        determining the magnitude of BH natal kick losses from the cluster.

    FeH : float
        Metallicity, in solar fraction [Fe/H].

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
    '''

    natal_ejecta = 0.0

    fb_interp = _fallback_frac(FeH)

    for j in range(Mr_BH.size):

        # Skip the bin if its empty
        if Nr_BH[j] < 0.1:
            continue

        # Interpolate fallback fraction at this mr
        fb = fb_interp(Mr_BH[j] / Nr_BH[j])

        if fb >= 1.0:  # bracket fb at 100%, meaningless value above that
            continue

        else:

            # Compute retention fraction
            retention = _maxwellian_retention_frac(vesc, fb=fb, vdisp=vdisp)

            # keep track of how much we eject
            natal_ejecta += Mr_BH[j] * (1 - retention)

            # eject the mass
            Mr_BH[j] *= retention
            Nr_BH[j] *= retention

    return Mr_BH, Nr_BH, natal_ejecta

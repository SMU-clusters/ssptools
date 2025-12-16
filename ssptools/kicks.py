#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ifmr import get_data

import dataclasses

import numpy as np
from scipy.special import erf
import scipy.interpolate as interp


__all__ = ["natal_kicks", "KickStats"]


@dataclasses.dataclass(eq=False, frozen=True)
class KickStats:
    retention: np.ndarray
    mass_kicked: np.ndarray
    parameters: dict

    @property
    def total_kicked(self) -> float:
        return self.mass_kicked.sum()

    @classmethod
    def no_kicks(cls, nmbin):
        return cls(
            retention=np.ones(nmbin),
            mass_kicked=np.zeros(nmbin),
            parameters=dict()
        )


# TODO there are currently no checks on input parameters to any fret function.
def _maxwellian_retention_frac(m, vesc, FeH, vdisp=265., *, SNe_method='rapid'):
    '''Retention fraction alg. based on a Maxwellian kick velocity distribution.

    This method is based on the assumption that the natal kick velocity is
    drawn from a Maxwellian distribution with a certain kick dispersion
    scaled down by a fallback fraction, as described by Fryer et al. (2012).

    The fraction of black holes retained in each mass bin is then found by
    integrating the kick velocity distribution from 0 to the estimated initial
    system escape velocity. In other words, by evaluating the CDF at the
    escape velocity.

    Parameters
    ----------
    m : float
        The mean mass of a BH bin.

    vesc : float
        The initial escape velocity of the cluster.

    vdisp : float, optional
        The dispersion of the Maxwellian kick velocity distribution. Defaults
        to 265 km/s, as typically used for neutron stars.

    SNe_method : {'rapid', 'delayed', 'NS', None}, optional
        Which method to use to determine the fallback fraction as a function of
        the black hole mass, which scales the dispersion as σ(1-fb).
        Available methods include the "rapid" (default) or "delayed" supernovae
        prescriptions described by Fryer+2012, or the ratio of the neutron star
        to black hole mass.
        If None, no fallback will be applied, and all masses will use `vdisp`.

    Returns
    -------
    float
        The retention fraction of BHs of this mass.

    '''

    def _maxwellian_cdf(x, a):
        norm = np.sqrt(2 / np.pi)
        err = erf(x / (np.sqrt(2) * a))
        exponent = np.exp(-(x**2) / (2 * (a**2)))
        return err - (norm * (x / a) * exponent)

    match SNe_method.casefold():

        case 'rapid' | 'delayed':

            # clip fb just below 1, to avoid divide by 0 errors
            fb = np.clip(
                _F12_fallback_frac(FeH, SNe_method=SNe_method)(m),
                0.0, 1 - 1e-16
            )

        case 'ns' | 'neutron' | 'neutron star':

            fb = 1. - _NS_reduced_kick(m_NS=1.4)(m)

        case 'neutrino' | 'neutrino-driven':

            fb = 1. - _neutrino_driven_kick(m_eff=7.0)(m)

        case None | 'none':

            fb = np.zeros_like(m)

        case _:

            raise ValueError(f"Invalid SNe method '{SNe_method}'.")

    scale = vdisp * (1. - fb)

    return _maxwellian_cdf(vesc, scale)


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


def _NS_reduced_kick(m_NS=1.4):
    '''Reduce σ by scaling the final BH mass based on the neutron star mass.'''
    return lambda m: m_NS / m


def _neutrino_driven_kick(m_eff=7.0):
    '''Kicks produced by asymmetric neutrino emission.'''
    return lambda m: np.min([np.full_like(m, m_eff), m], axis=0) / m


def _flat_fallback_frac(frac):
    '''Give a constant fallback fraction for all masses, at `frac`.'''
    return lambda m: frac


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


def _tanh_retention_frac(m, slope, scale):
    r'''Retention fraction alg. based on a paramatrized function of tanh.

    This method is based on a simple parametrization of the relationship
    between the retention fraction and the BH mass as a sigmoid function,
    namely the hyperbolic tangent, increasing smoothly between 0 and 1
    and reaching 50% at the given scale mass.

   .. math::
        f_{\mathrm{ret}}(m) = \frac{1}{2} \left(
            \tanh\left(\mathrm{slope}\ (m - \mathrm{scale})\right) + 1
        \right)

    Parameters
    ----------
    m : float
        The mean mass of a BH bin.

    slope : float
        The "slope" of the sigmoid function, defining the "sharpness" of the
        increase. A value of 0 is completely flat (at 50% for all masses), an
        increasingly positive value approaches a step function at the scale
        mass, and a negative value will retain more low-mass bins than high.

    scale : float
        The scale-mass of the sigmoid function, defining the approximate mass
        of the turn-over from 0 to 1. By definition, f_ret(m=scale)=0.5.

    Returns
    -------
    float
        The retention fraction of BHs of this mass.
    '''
    return 0.5 * (np.tanh(slope * (m - scale)) + 1)
    # return np.tanh(np.exp(slope * (m - scale)))  # alternative


def _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs):

    c = Nr_BH > 0.1
    mr_BH = Mr_BH[c] / Nr_BH[c]
    natal_ejecta = np.zeros_like(Mr_BH)
    retention = np.full_like(Mr_BH, np.nan)

    retention[c] = f_ret(mr_BH, **ret_kwargs)

    # keep track of how much we eject
    natal_ejecta[c] = Mr_BH[c] * (1 - retention[c])

    Mr_BH[c] *= retention[c]
    Nr_BH[c] *= retention[c]

    stats = KickStats(
        retention=retention, mass_kicked=natal_ejecta, parameters=ret_kwargs
    )

    return Mr_BH, Nr_BH, stats


def _determine_kick_params(Mr_BH, Nr_BH, f_ret, f_target, slope, scale=10.):
    '''
    Use a root finding algorithm to determine the value of `scale` needed in
    order to eject a total fraction of the given BHs `f_target`, using the
    given `f_ret` function to distribute the kicks among mass bins.
    Note that while it says "params", right now can only compute the scale.
    '''
    import scipy.optimize as opt

    c = Nr_BH > 0.1

    m_BH = Mr_BH[c] / Nr_BH[c]
    M_BH_tot = Mr_BH.sum()

    f_ini = Mr_BH[c] / M_BH_tot

    # f_ret = _tanh_retention_frac

    # Keep first guess on optionally given scale
    scale = scale if scale is not None else 10.0

    def target_fret(scl):

        retention = f_ret(m_BH, scale=scl, slope=slope)

        f_BH_final = (f_ini * (1 - retention)).sum(axis=0)

        return f_target - f_BH_final

    try:
        sol = opt.root_scalar(target_fret, x0=scale, bracket=(-25, 75))
    except ValueError as err:
        mssg = ("Root finder failed to find scale parameter matching target "
                f"{f_target}. 'f_target' or 'slope' need to be adjusted.")
        raise ValueError(mssg) from err

    root_scale = sol.root

    if not sol.converged:
        raise RuntimeError(f"root finder didn't converge on {f_target=}: {sol}")

    return slope, root_scale


def _get_kick_method(method):
    '''parse method to get the kick ret function (func to avoid repetition)'''

    match method.casefold():

        case 'sigmoid':
            f_ret = _sigmoid_retention_frac

        case 'tanh':
            f_ret = _tanh_retention_frac

        case 'maxwellian' | 'f12' | 'fryer2012':
            f_ret = _maxwellian_retention_frac

        case 'full' | 'everything' | 'all':
            f_ret = _flat_fallback_frac(0.0)

        case 'none':
            f_ret = _flat_fallback_frac(1.0)

        case _:
            raise ValueError(f"Invalid kick distribution method: {method}")

    return f_ret


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
        BH mass bin.

    Nr_BH : ndarray
        Array[nbin] of the total initial numbers of black holes in each
        BH mass bin.

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
        BH mass bin, after natal kicks.

    Nr_BH : ndarray
        Array[nbin] of the total final numbers of black holes in each
        BH mass bin, after natal kicks.

    stats : KickStats
        Statistics on how and how many BHs are kicked.

    See Also
    --------
    _maxwellian_retention_frac : Maxwellian retention fraction algorithm.
    _sigmoid_retention_frac : Sigmoid retention fraction algorithm.
    _tanh_retention_frac : Hyperbolic tangent retention fraction algorithm.
    '''

    f_ret = _get_kick_method(method)

    # If no given total kick fraction, use old-style of directly using f_ret
    if f_kick is None:
        return _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, **ret_kwargs)

    else:

        # Fit for the desired kick scale
        try:
            slp, scl = _determine_kick_params(Mr_BH, Nr_BH, f_ret, f_kick,
                                              **ret_kwargs)
        except TypeError as err:
            mssg = (f"Can only use `f_kick` with methods that use a `scale` and"
                    f" `slope` parameter, not '{method}'.")
            raise ValueError(mssg) from err

        # Compute the natal kicks based on fit scale
        return _unbound_natal_kicks(Mr_BH, Nr_BH, f_ret, slope=slp, scale=scl)

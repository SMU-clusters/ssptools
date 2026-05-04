#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .ifmr import get_data
from .evolve_mf import InitialBHPopulation

import dataclasses

import numpy as np
from scipy.special import erf, gammaincinv
import scipy.interpolate as interp


__all__ = ["kick_retention_fraction", "KickStats", "maxwellian_kick_v"]


@dataclasses.dataclass(eq=False, frozen=True)
class KickStats:
    retention: np.ndarray
    mass_kicked: np.ndarray
    num_kicked: np.ndarray
    parameters: dict

    @property
    def total_kicked(self) -> float:
        '''The total amount of BH mass kicked.'''
        return self.mass_kicked.sum()

    @classmethod
    def no_kicks(cls, nmbin):
        '''Kick stats when no kicks are applied.'''
        return cls(
            retention=np.ones(nmbin),
            mass_kicked=np.zeros(nmbin),
            num_kicked=np.zeros(nmbin),
            parameters=dict()
        )

    @classmethod
    def from_final(cls, Mr_BH, Nr_BH, ibh: InitialBHPopulation, parameters):
        '''Compute the kick stats based on final BH arrays.

        This constructor (the main one, likely) uses the final amounts of BHs,
        after all kicks have been applied (but no other ejections have taken
        place, e.g. no dynamical ejections), to compute the statistics based on
        a corresponding `InitialBHPopulation` where the kicks have *not* been
        applied.

        This is done by simply assuming that all mass that is in the initial
        BH population and not the given BH bins must have been natally kicked.

        This will, at best, provide an approximately correct view of the kicks.
        Any differences attributable to the binning effects in `Mr_BH` compared
        to `InitialBHPopulation` will be especially notable, and may lead to
        cases where, e.g., the kick amounts look very slightly negative.
        There is unfortunately no better way to determine the effects of
        natal kicks on the final BH bins themselves.

        Also note that this *will not* be valid for cases where the BH bins
        are created *before* the full amounts of BHs have been formed (i.e.
        younger than `ibh.age`).

        Parameters
        ----------
        Mr_BH : ndarray
            Array[nbin] of the total final masses of black holes in each
            BH mass bin, after natal kicks but before any other ejections.

        Nr_BH : ndarray
            Array[nbin] of the total final numbers of black holes in each
            BH mass bin, after natal kicks but before any other ejections.

        ibh : InitialBHPopulation
            The `InitialBHPopulation` instance representing the expected
            complete initial BH mass function. This must align exactly with
            the BH bins, and must have been created with `natal_kicks=False`.

        parameters : dict
            The kick retention function parameters used, for convenient storage.
        '''

        retention = np.ones_like(Mr_BH)
        c = ibh.M > 0
        retention[c] = Mr_BH[c] / ibh.M[c]

        M_kicked = ibh.M - Mr_BH
        N_kicked = ibh.N - Nr_BH

        return cls(
            retention=retention,
            mass_kicked=M_kicked,
            num_kicked=N_kicked,
            parameters=parameters
        )


# --------------------------------------------------------------------------
# Retention fraction functions
# --------------------------------------------------------------------------


# TODO there are currently no checks on input parameters to any fret function.
def _maxwellian_retention_frac(m, vesc, FeH, vdisp=265., *, SNe_method='rapid'):
    '''Retention fraction alg. based on a Maxwellian kick velocity distribution.

    This method is based on the assumption that the natal kick velocity is
    drawn from a Maxwellian distribution with a certain kick dispersion
    scaled down by a fallback fraction, as described by Fryer et al. (2012).

    The fraction of black holes retained is then found by
    integrating the kick velocity distribution from 0 to the estimated initial
    system escape velocity. In other words, by evaluating the CDF at the
    escape velocity.

    Parameters
    ----------
    m : float
        The mean initial mass of the progenitor star which will make a BH.
        Used to determine the fallback fraction.

    vesc : float
        The initial escape velocity of the cluster.

    vdisp : float, optional
        The dispersion of the Maxwellian kick velocity distribution. Defaults
        to 265 km/s, as typically used for neutron stars.

    SNe_method : {'rapid', 'delayed', 'NS', 'none'}, optional
        Which method to use to determine the fallback fraction as a function of
        the initial mass, which scales the dispersion as σ(1-fb).
        Available methods include the "rapid" (default) or "delayed" supernovae
        prescriptions described by Fryer+2012, or the ratio of the neutron star
        to black hole mass.
        If None, no fallback will be applied, and all masses will use `vdisp`.

    Returns
    -------
    float
        The retention fraction of BHs created by stars of this mass.

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
                _F12_fallback_frac(FeH, SNe_method=SNe_method.casefold())(m),
                0.0, 1 - 1e-16
            )

        case 'sevn-rapid' | 'sevn-delayed':

            meth = SNe_method.split('-')[-1].casefold()

            # clip fb just below 1, to avoid divide by 0 errors
            fb = np.clip(
                _F12_fallback_frac(FeH, model='SEVN', SNe_method=meth)(m),
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


def _F12_fallback_frac(FeH, *, model='uSSE', SNe_method='rapid'):
    '''Get the fallback fraction for this mass, interpolated from SSE models
    based on the prescription from Fryer 2012 (or from SEVN).
    Note there are no checks on FeH here, so make sure it's within the grid.

    model must be one of uSSE or SEVN
    SNe_method must be one of rapid or delayed.
    '''

    # load in the ifmr data to interpolate fb from mr
    # feh_path = get_data(f"sse/MP_FEH{FeH:+.2f}.dat")  # .2f snaps to the grid
    feh_path = get_data(f"ifmr/{model}_{SNe_method}/IFMR_FEH{FeH:+.2f}.dat")

    # load in the data (only initial star mass and fbac)
    fb_grid = np.loadtxt(feh_path, usecols=(0, 3), unpack=True)

    # Interpolate the mi-fb grid
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
    between the retention fraction and the initial mass as a sigmoid function,
    increasing smoothly between 0 and 1 around a scale mass.

   .. math::
        f_{\mathrm{ret}}(m) = \operatorname{erf}\left(
            e^{\mathrm{slope}\ (m - \mathrm{scale})}
        \right)

    Parameters
    ----------
    m : float
        The mean initial mass of the progenitor star which will make a BH.

    slope : float
        The "slope" of the sigmoid function, defining the "sharpness" of the
        increase. A value of 0 is completely flat (at fret=erf(1)~0.85), an
        increasingly positive value approaches a step function at the scale
        mass, and a negative value will retain more low-mass progenitor BHs
        than high.

    scale : float
        The scale-mass of the sigmoid function, defining the approximate mass
        of the turn-over from 0 to 1 (e.g. the position of the step function
        as the slope approaches infinity).

    Returns
    -------
    float
        The retention fraction of BHs created by stars of this mass.
    '''
    return erf(np.exp(slope * (m - scale)))


def _tanh_retention_frac(m, slope, scale):
    r'''Retention fraction alg. based on a paramatrized function of tanh.

    This method is based on a simple parametrization of the relationship
    between the retention fraction and the initial mass as a sigmoid function,
    namely the hyperbolic tangent, increasing smoothly between 0 and 1
    and reaching 50% at the given scale mass.

   .. math::
        f_{\mathrm{ret}}(m) = \frac{1}{2} \left(
            \tanh\left(\mathrm{slope}\ (m - \mathrm{scale})\right) + 1
        \right)

    Parameters
    ----------
    m : float
        The mean initial mass of the progenitor star which will make a BH.

    slope : float
        The "slope" of the sigmoid function, defining the "sharpness" of the
        increase. A value of 0 is completely flat (at 50% for all masses), an
        increasingly positive value approaches a step function at the scale
        mass, and a negative value will retain more low-mass progenitor BHs
        than high.

    scale : float
        The scale-mass of the sigmoid function, defining the approximate mass
        of the turn-over from 0 to 1. By definition, f_ret(m=scale)=0.5.

    Returns
    -------
    float
        The retention fraction of BHs created by stars of this mass.
    '''
    return 0.5 * (np.tanh(slope * (m - scale)) + 1)
    # return np.tanh(np.exp(slope * (m - scale)))  # alternative


def kick_retention_fraction(m_ini, method='fryer2012', **ret_kwargs):
    r'''Compute the retention fraction of BH natal-kicks from progenitor masses

    Determines the effects of BH natal-kicks, in the form of a retention
    fraction to be used when creating the BH masses, based on the given natal
    kick algorithm.

    Various natal kick algorithms are currently available. All methods require
    different arguments, which can be passed to `ret_kwargs`.
    See the respective retention functions for details on these arguments and
    the algorithms themselves.

    Parameters
    ----------
    m_ini : float or ndarray[float]
        The initial (ZAMS) mass of the progenitor star which will form a BH.
        Is passed to the requested retention fraction function.

    method : {'sigmoid', 'tanh', 'maxwellian'}, optional
        Natal kick algorithm to use, defining the retention fraction as a
        function of mean bin mass. Defaults to the Maxwellian method.

    **ret_kwargs : dict, optional
        All other arguments are passed to the retention fraction function.

    Returns
    -------
    f_rem : float
        The retention fraction of the BH created by this input progenitor mass.

    See Also
    --------
    _maxwellian_retention_frac : Maxwellian retention fraction algorithm.
    _sigmoid_retention_frac : Sigmoid retention fraction algorithm.
    _tanh_retention_frac : Hyperbolic tangent retention fraction algorithm.
    '''

    f_ret = _get_kick_method(method)

    return f_ret(m_ini, **ret_kwargs)


def _determine_kick_params(method, f_target, IMF, IFMR, slope, scale=10.):
    import scipy.optimize as opt

    f_ret = _get_kick_method(method)

    # Get domain of progenitor masses that make BHs
    mi = np.linspace(*IFMR.BH_mi, 50_000)
    dm = mi[1] - mi[0]

    # Determine the final BH masses
    mf = IFMR.predict(mi)

    M_BH_tot_ini = np.sum(mf * IMF.N(mi) * dm)

    # Keep first guess on optionally given scale
    scale = scale if scale is not None else 10.0

    def target_fret(scl):

        retention = f_ret(mi, scale=scl, slope=slope)

        M_BH_tot = np.sum(retention * mf * IMF.N(mi) * dm)

        f_BH_final = 1 - (M_BH_tot / M_BH_tot_ini)

        return f_target - f_BH_final

    try:
        sol = opt.root_scalar(target_fret, x0=scale,
                              bracket=(-25, IFMR.BH_mi.upper + 25))

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


# --------------------------------------------------------------------------
# Computing kick velocities directly
# --------------------------------------------------------------------------


def maxwellian_kick_v(m, FeH, vdisp=265., *, rng=None, SNe_method='rapid'):
    r'''Estimate the natal kick velocities for BHs of given masses.

    Computes the Maxwellian natal kick velocities for BHs of a certain (BH)
    mass, under the assumption that these kicks follow a Maxwellian velocity
    distribution, with a dispersion given by `vdisp` and scaled downwards
    by some fallback fraction, based on the chosen supernovae prescription.

    In contrast to the other natal kick methods provided, this function does
    not work on a population of BHs (e.g. within a certain mass bin) but
    instead randomly samples the velocities of individual BHs of a certain
    mass.

    Parameters
    ----------
    m : float or ndarray
        The (final) mass of the (individual) BHs to compute a kick velocity for.

    FeH : float
        The metallicity of the system, used to determine the fallback fraction
        under the 'rapid' or 'delayed' Fryer+2012 SNe methods.

    vdisp : float, optional
        The dispersion of the Maxwellian velocity distribution. Defaults to
        265 km/s, as typically used for neutron stars.

    rng : np.random.Generator, optional
        A random number generator instance for sampling the Maxwellian
        distribution. If None, a default RNG (`np.random.default_rng`) is used.

    SNe_method : {'rapid', 'delayed', 'NS', 'neutrino', 'none'}, optional
        Which method to use to determine the fallback fraction as a function of
        the black hole mass, which scales the dispersion as σ(1-fb).
        Available methods include the "rapid" (default) or "delayed" supernovae
        prescriptions described by Fryer+2012, or the ratio of the neutron star
        to black hole mass.
        If None, no fallback will be applied, and all masses will use `vdisp`.

    Returns
    -------
    ndarray
        The randomly sampled natal kick velocities for the given BH masses.
    '''

    # Get RNG sampler

    if rng is None:
        rng = np.random.default_rng()

    # Get fallback fraction for this mass

    match SNe_method.casefold():

        case 'rapid' | 'delayed':

            fb = np.clip(
                _F12_fallback_frac(FeH, SNe_method=SNe_method.casefold())(m),
                0.0, 1 - 1e-16
            )

        case 'sevn-rapid' | 'sevn-delayed':

            meth = SNe_method.split('-')[-1].casefold()

            # clip fb just below 1, to avoid divide by 0 errors
            fb = np.clip(
                _F12_fallback_frac(FeH, model='SEVN', SNe_method=meth)(m),
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

    # Sample the Maxwellian distribution (from it's CDF), and apply scaling

    scale = vdisp * (1. - fb)
    U = rng.uniform(size=np.shape(m))

    return np.sqrt(2 * gammaincinv(1.5, U)) * scale

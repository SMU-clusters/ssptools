#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

import numpy as np
from scipy.integrate import ode

from . import kicks
from .ifmr import IFMR, get_data
from .masses import PowerLawIMF, MassBins, Pk

# TODO optionally support units for some things


__all__ = ['EvolvedMF', 'EvolvedMFWithBH', 'InitialBHPopulation']


class EvolvedMF:
    r'''Evolve an IMF to a present-day mass function at a given age.

    Evolves an arbitrary N-component power law initial mass function (IMF)
    to a binned present-day mass function (PDMF) at a given set of ages, and
    computes the numbers and masses of stars and remnants in each mass bin.

    # TODO add more in-depth explanation or references to actual algorithm here

    Parameters
    ----------
    IMF : PowerLawIMF
        Initial Mass Function (IMF) object, to be used to generate the initial
        numbers of stars at time 0.

    nbins : int or list of int or dict
        Number of mass bins defining the mass function.
        See `masses.MassBins` for more information.
        If list of int, must match number of components provided by
        binning_breaks (which defaults to IMF.mb).

    FeH : float
        Metallicity, in solar fraction [Fe/H].

    tout : list of int
        Times, in years, at which to output PDMF. Defines the shape of many
        outputted attributes.

    esc_rate : float or callable
        Represents rate of change of stars over time due to tidal
        ejections (and other escape mechanisms). Regulates low-mass object
        depletion (ejection) due to dynamical evolution.
        If a float, will apply a constant escape rate across all time. If a
        callable, must take in a time `t` in Myr and output a (negative) float
        representing the rate at that time.
        Rates must be in units of stars per Myr if `esc_norm` is 'N' or solar
        masses per Myr if `esc_norm` is 'M'.

    N0 : int or None, optional
        Total initial number of stars, over all bins.
        If None, uses the N0 of the given IMF class. Defaults to 5e5 stars.

    tcc : float, optional
        Core collapse time, in years. Defaults to 0, effectively being ignored.

    NS_ret : float, optional
        Neutron star retention fraction (0 to 1). Defaults to 0.1 (10%).

    BH_ret_int : float, optional
        Initial black hole retention fraction (0 to 1). Defaults to 1 (100%).

    BH_ret_dyn : float, optional
        Dynamical black hole retention fraction (0 to 1), including both
        dynamical ejections and natal kicks. Defaults to 1 (100%).

    natal_kicks : bool, optional
        Whether to account for natal kicks in the BH dynamical retention.
        Defaults to False.

    kick_method : {'maxwellian', 'sigmoid'}, optional
        The BH natal kick algorithm to use, if `natal_kicks=True`.
        See `kicks.natal_kicks` for more details.

    SNe_method : {'rapid', 'delayed', None}, optional
        Which supernovae prescription to use to determine the fallback fraction.
        See `kicks._maxwellian_retention_frac` for more information.
        Note that the `BH_IFMR_method` should likely be changed in concert with
        this.

    vesc : float, optional
        Initial cluster escape velocity, in km/s, for use in the computation of
        BH natal kick effects. Defaults to 90 km/s.

    kick_vdisp : float, optional
        The dispersion of the Maxwellian natal kick velocity distribution.
        Only used if `kick_method='maxwellian'`.
        See `kicks._maxwellian_retention_frac` for more information.

    f_kick : float, optional
        Unused.

    kick_slope : float
        The "slope" of the sigmoid retention function, defining the
        "sharpness" of the increase. Only used if `kick_method='sigmoid'`.
        See `kicks._sigmoid_retention_frac` for more information.

    kick_scale : float
        The scale-mass of the sigmoid retention function, defining the
        approximate mass of the turn-over from 0 to 1.
        Only used if `kick_method='sigmoid'`.
        See `kicks._sigmoid_retention_frac` for more information.

    stellar_evolution : bool, optional
        Whether or not to include stellar evolution effects in the mass function
        evolution. There is not much real sense in turning this off, but it can
        be useful for testing escape rates on their own.

    md : float, optional
        Depletion mass, below which stars are preferentially disrupted during
        the stellar escape derivatives.
        The default 1.2 is based on Lamers et al. (2013) and shouldn't be
        changed unless you know what you're doing. Making this too large will
        break assumptions made internally about BH dynamics.

    esc_norm : {'N', 'M'}, optional
        Defines whether the given `esc_rate` is to be given in stars or mass
        per unit time. This affects the normalization of the escape derivatives.
        Note that a constant :math:`\dot{N}` will give rise to an increasing
        :math:`\dot{M}` over time, and vice versa, so the two options are not
        directly interchangeable. Defaults to 'N'.

    BH_IFMR_method : str, optional
        The BH IFMR algorithm to use when determining BH masses.
        See `ifmr` for more information.

    WD_IFMR_method : str, optional
        The WD IFMR algorithm to use when determining WD masses.
        See `ifmr` for more information.

    BH_IFMR_kwargs : dict, optional
        All arguments passed to the BH IFMR algorithm. See the specified
        functions in `ifmr` for information on all required methods.

    WD_IFMR_kwargs : dict, optional
        All arguments passed to the WD IFMR algorithm. See the specified
        functions in `ifmr` for information on all required methods.

    binning_breaks : list of float, optional
        The binning break masses to use when constructing the mass bins,
        including outer edges. See `masses.MassBins` for more information.
        If nbins is a list of int, this list must be one longer (i.e. bound
        all bins). By default, the break masses of the IMF will be used.

    binning_method : {'default', 'split_log', 'split_linear'}, optional
        The spacing method to use when constructing the mass bins.
        See `masses.MassBins` for more information.

    Attributes
    ----------
    nbin : int
        Total number of bins (sum of nbins parameter).

    nout : int
        Total number of output times (len of tout parameter).

    alphas : ndarray
        Array[nout, nbin] of  PDMF slopes. If esc_rate = 0, this is defined
        entirely by the IMF.

    Ns : ndarray
        Array[nout, nbin] of the total number of (main sequence) stars in each
        mass bin.

    Ms : ndarray
        Array[nout, nbin] of the total mass of (main sequence) stars in each
        mass bin.

    ms : ndarray
        Array[nout, nbin] of the individual mass of (main sequence) stars in
        each mass bin, as defined by Ms / Ns.

    mes : ndarray
        Array[nout, nbin + 1] representing the mass bin edges, defined by the
        m_breaks and nbins parameters. Note that ms *does not* necessarily fall
        in the middle of these edges.

    Nr : ndarray
        Array[nout, nbin] of the total number of remnants in each mass bin.

    Mr : ndarray
        Array[nout, nbin] of the total mass of remnants in each mass bin.

    mr : ndarray
        Array[nout, nbin] of the individual mass of remnants in
        each mass bin, as defined by Mr / Nr.

    rem_types : ndarray
        Array[nout, nbin] of 2-character strings representing the remnant types
        found in each mass bin of the remnant (Mr, Nr, mr) arrays. "WD" = white
        dwarfs, "NS" = neutron stars, "BH" = black holes.

    M : ndarray
        Array containing the total mass in all non-empty mass bins, including
        both stars and remnants, based on the final age given by tout.

    N : ndarray
        Array containing the total number in all non-empty mass bins, including
        both stars and remnants, based on the final age given by tout.

    m : ndarray
        Array containing the individual mass in all non-empty mass bins,
        including both stars and remnants, based on the final age given by tout.

    bin_widths : ndarray
        Array containing the width of all non-empty mass bins,
        including both stars and remnants, based on the final age given by tout.
        Equal to `np.diff(mes)`.

    types : ndarray
        Array of 2-character strings representing the object type of the
        corresponding bins of the M, N and m properties. "MS" = main sequence
        stars, "WD" = white dwarfs, "NS" = neutron stars, "BH" = black holes.

    Notes
    -----
        The BH kicks and ejections are computed post facto on the BH mass
        bins for each requested output age (`tout`). As such, parameters such
        as `BH_ret_dyn` are applied equally at each age, and not accounted
        for during the evolution, which may not be entirely realistic.
        Care should be taken when examining the BHs at multiple ages.

        If there is an evolution over time of `f_BH` you wish to match, see
        `EvolvedMFWithBH` instead.
    '''

    @property
    def M(self):
        # TODO why is it 10*Nmin
        cs = self.Ns[-1] > 10 * self.Nmin
        Ms = self.Ms[-1][cs]

        cr = np.c_[self.Nr][-1] > 10 * self.Nmin
        Mr = np.c_[self.Mr][-1][cr]

        return np.r_[Ms, Mr]

    @property
    def N(self):
        cs = self.Ns[-1] > 10 * self.Nmin
        Ns = self.Ns[-1][cs]

        cr = np.c_[self.Nr][-1] > 10 * self.Nmin
        Nr = np.c_[self.Nr][-1][cr]

        return np.r_[Ns, Nr]

    @property
    def m(self):
        return self.M / self.N

    @property
    def bin_widths(self):

        mto = self.compute_mto(self.tout[-1])
        mes = self.massbins.turned_off_bins(mto)

        cs = self.Ns[-1] > 10 * self.Nmin
        bws = (mes.upper - mes.lower)[cs]

        cr = np.c_[self.Nr][-1] > 10 * self.Nmin
        bwr = np.diff(np.c_[self.massbins.bins[1:]], axis=0)[-1][cr]

        return np.r_[bws, bwr]

    @property
    def types(self):
        cs = self.Ns[-1] > 10 * self.Nmin
        cr = np.c_[self.Nr][-1] > 10 * self.Nmin

        ts = ['MS'] * cs.sum()
        tr = self.rem_types[cr]

        return np.r_[ts, tr]

    @property
    def nms(self):
        return (self.Ns[-1] > 10 * self.Nmin).sum()

    @property
    def nmr(self):
        return (np.c_[self.Nr][-1] > 10 * self.Nmin).sum()

    def __init__(self, IMF, nbins, FeH, tout, esc_rate, N0=5e5,
                 tcc=0.0, NS_ret=0.1, BH_ret_int=1.0, BH_ret_dyn=1.0, *,
                 natal_kicks=False, kick_method='maxwellian',
                 SNe_method='rapid', vesc=90, kick_vdisp=265., f_kick=None,
                 kick_slope=1, kick_scale=20,
                 stellar_evolution=True, md=1.2, esc_norm='N',
                 BH_IFMR_method='banerjee20', WD_IFMR_method='mist18',
                 BH_IFMR_kwargs=None, WD_IFMR_kwargs=None,
                 binning_breaks=None, binning_method='default'):

        # ------------------------------------------------------------------
        # Set various other parameters
        # ------------------------------------------------------------------

        # Supplied parameters
        self.tcc = tcc
        self.tout = np.atleast_1d(tout)

        self.esc_rate = esc_rate
        self._esc_norm = esc_norm
        self._time_dep_esc = callable(esc_rate)

        if not self._time_dep_esc and esc_rate > 0:
            raise ValueError("'esc_rate' must be less than 0")

        if esc_norm not in ('N', 'M'):
            mssg = f"Invalid 'esc_norm' {esc_norm}, must be one of 'M', 'N'."
            raise ValueError(mssg)

        self._stellar_ev = stellar_evolution

        self.NS_ret = NS_ret
        self.BH_ret_int = BH_ret_int
        self.BH_ret_dyn = BH_ret_dyn
        self._frem = {'WD': 1., 'NS': NS_ret, 'BH': BH_ret_int}

        self.FeH = FeH

        # Initial-Final mass relations
        self.IFMR = IFMR(FeH,
                         WD_method=WD_IFMR_method, WD_kwargs=WD_IFMR_kwargs,
                         BH_method=BH_IFMR_method, BH_kwargs=BH_IFMR_kwargs)

        # Minimum of stars to call a bin "empty"
        self.Nmin = 0.1

        self.md = md

        # ------------------------------------------------------------------
        # Initialise the initial mass function and mass bins given the
        # power-law IMF slopes and bins
        # ------------------------------------------------------------------

        self.IMF = IMF
        self.N0 = N0

        binning_breaks = IMF.mb if binning_breaks is None else binning_breaks

        self.massbins = MassBins(binning_breaks, nbins, self.IMF, self.IFMR,
                                 binning_method=binning_method)

        # ------------------------------------------------------------------
        # Setup lifetime approximations and compute t_ms of all bin edges
        # ------------------------------------------------------------------

        # Load a_i coefficients derived from interpolated Dartmouth models
        mstogrid = np.loadtxt(get_data("sevtables/msto.dat"))
        nearest_FeH = np.argmin(np.abs(mstogrid[:, 0] - FeH))
        self._tms_constants = mstogrid[nearest_FeH, 1:]

        # Compute t_ms for all bin edges
        self.tms_l = self.compute_tms(self.massbins.bins.MS.lower)
        self.tms_u = self.compute_tms(self.massbins.bins.MS.upper)

        # ------------------------------------------------------------------
        # Generate times for integrator
        # ------------------------------------------------------------------

        t_end = np.max(tout)

        # Compute each time a new bin evolves out of MS, up till t_end
        self.t = np.sort(np.r_[self.tms_u[self.tms_u < t_end], self.tout])

        self.nt = self.t.size

        # ------------------------------------------------------------------
        # Setup BH natal kicks
        # ------------------------------------------------------------------

        self.natal_kicks = natal_kicks

        match kick_method.casefold():

            case 'maxwellian' | 'f12' | 'fryer2012':

                from .ifmr import _check_IFMR_FeH_bounds
                FeH_BH = _check_IFMR_FeH_bounds(FeH)

                self._kick_kw = dict(
                    method=kick_method, f_kick=f_kick, vesc=vesc,
                    FeH=FeH_BH, vdisp=kick_vdisp, SNe_method=SNe_method
                )

            case 'sigmoid' | 'tanh':
                self._kick_kw = dict(method=kick_method, f_kick=f_kick,
                                     slope=kick_slope, scale=kick_scale)

            case _:
                mssg = f"Invalid natal kick algorithm '{kick_method=}'"
                raise ValueError(mssg)

        # ------------------------------------------------------------------
        # Finally, evolve the population
        # ------------------------------------------------------------------

        # Initialize iteration counter
        self.nstep = 0

        self._evolve()

    @classmethod
    def from_powerlaw(cls, m_breaks, a_slopes, nbins, FeH, tout, esc_rate,
                      N0=5e5, *args, **kwargs):
        '''Construct class based on a power-law IMF breaks and slopes, directly.

        Alternative constructor to `EvolvedMF` based on explicitly providing the
        power-law IMF slopes and break masses, rather than an already created
        IMF object itself.
        Simply creates a `PowerLawIMF` class based on these arguments and passes
        it through to initialize the base class.

        Parameters
        ----------
        m_breaks : list of float
            Power law IMF break-masses (including outer bounds; size N+1).

        a_slopes : list of float
            Power law IMF slopes α (size N).

        nbins : int or list of int or dict
            Number of mass bins defining the mass function.
            See `masses.MassBins` for more information.
            If list of int, must match number of components provided by
            binning_breaks (which defaults to m_breaks; size N).

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        tout : list of int
            Times, in years, at which to output PDMF. Defines the shape of many
            outputted attributes.

        esc_rate : float or callable
            Represents rate of change of stars over time due to tidal
            ejections (and other escape mechanisms). Regulates low-mass object
            depletion (ejection) due to dynamical evolution.
            If a float, will apply a constant escape rate across all time. If a
            callable, must take in a time `t` in Myr and output a (negative)
            float representing the rate at that time.
            Rates must be in units of stars per Myr if `esc_norm` is 'N' or
            solar masses per Myr if `esc_norm` is 'M'.

        N0 : int or None, optional
            Total initial number of stars, over all bins.
            If None, uses the N0 of the given IMF class. Defaults to 5e5 stars.

        *args, **kwargs
            All other arguments are passed to EvolvedMF.

        Returns
        -------
        EvolvedMF
            The created evolved MF object, using the created IMF.
        '''
        imf = PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0, ext='zeros')
        return cls(imf, nbins, FeH, tout, esc_rate, N0=N0, *args, **kwargs)

    def compute_tms(self, mi):
        '''Compute main-sequence lifetime for a given mass `mi`'''
        a = self._tms_constants
        return a[0] * np.exp(a[1] * mi ** a[2])

    def compute_mto(self, t):
        '''Compute the turn-off mass for a given time `t` (inverse of tms)'''
        a0, a1, a2 = self._tms_constants

        out = np.empty_like(t, dtype=float)
        asympt = t > a0

        out[asympt] = (np.log(np.asanyarray(t)[asympt] / a0) / a1) ** (1 / a2)
        out[~asympt] = np.inf

        return out

    def _derivs(self, t, y):
        '''Main function for computing derivatives relevant to mass evolution

        Simply calls the two constituent mass evolution derivative methods;
        `_derivs_esc` and `_derivs_sev`. Designed to be solved using an ODE
        integrator, such as `scipy.integrate.ode`.

        Parameters
        ----------
        t : float
            Time step to compute derivatives at

        y : list of ndarray
            Equation system solution y. Size-4 array containing the arrays,
            for each mass bin, of the number of stars `Ns`, the mass function
            slopes `alpha`, the number of remnants `Nr` and the total bin mass
            of remnants `Mr`

        Returns
        -------
        list of ndarray
            Time derivatives of each of the four quantities described by `y`
        '''

        # Iterate step count
        self.nstep += 1

        # Compute stellar evolution derivatives
        if self._stellar_ev:
            derivs_sev = self._derivs_sev(t, y)
        else:
            derivs_sev = self.massbins.blanks(packed=True)

        # Only run the dynamical star losses `derivs_esc` if escape is not zero
        if self._time_dep_esc or self.esc_rate <= 0:
            derivs_esc = self._derivs_esc(t, y)
        else:
            derivs_esc = np.zeros_like(derivs_sev)

        # Combine mass loss derivatives
        return derivs_sev + derivs_esc

    def _derivs_sev(self, t, y):
        '''Derivatives relevant to mass changes due to stellar evolution'''

        # Setup derivative bins
        Ns, alpha, Nr, Mr = self.massbins.unpack_values(y, grouped_rem=True)
        dNs, dalpha, dNr, dMr = self.massbins.blanks(packed=False,
                                                     grouped_rem=True)

        # Apply only if this time is atleast later than the earliest tms
        if t > self.tms_u[-1]:

            # Find out which mass bin is the current turn-off bin
            isev = np.where(t > self.tms_u)[0][0]

            m1 = self.massbins.bins.MS.lower[isev]
            mto = self.compute_mto(t)
            Nj = Ns[isev]

            # Avoid "hitting" the bin edge
            if mto > m1 and Nj > self.Nmin:

                # Two parameters defining the bin
                alphaj = alpha[isev]

                # The normalization constant
                # TODO deal with the constant divide-by-zero warning here
                Aj = Nj / Pk(alphaj, 1, m1, mto)

                # Get the number of turn-off stars per unit of mass
                dNdm = Aj * mto ** alphaj

            else:
                dNdm = 0
                # TODO just break???

            # Compute the full dN/dt = dN/dm * dm/dt
            a = self._tms_constants
            dmdt = abs((1.0 / (a[1] * a[2] * t))
                       * (np.log(t / a[0]) / a[1]) ** (1 / a[2] - 1))

            dNdt = -dNdm * dmdt

            # Fill in star derivatives (alphaj remains constant for _derivs_sev)
            dNs[isev] = dNdt

            # Find remnant mass and which bin they go into
            m_rem, cls_rem = self.IFMR.predict(mto), self.IFMR.predict_type(mto)

            # Skip 0-mass remnants
            if m_rem > 0:

                # Find bin based on lower bin edge (must be careful later)
                irem = self.massbins.determine_index(m_rem, cls_rem)

                # Compute Remnant retention fractions based on remnant type
                frem = self._frem[cls_rem]

                # Fill in remnant derivatives
                getattr(dNr, cls_rem)[irem] = -dNdt * frem
                getattr(dMr, cls_rem)[irem] = -m_rem * dNdt * frem

        return self.massbins.pack_values(dNs, dalpha, *dNr, *dMr)

    def _derivs_esc(self, t, y):
        '''Derivatives relevant to mass loss due to escaping low-mass stars'''

        # Pull out individual arrays from y
        Ns, alpha, Nr, Mr = self.massbins.unpack_values(y, grouped_rem=True)
        dNs, dalpha, dNr, dMr = self.massbins.blanks(packed=False,
                                                     grouped_rem=True)

        # Compute various required quantities

        mto = self.compute_mto(t)

        bins_MS = self.massbins.turned_off_bins(mto)

        P1 = Pk(alpha, 1, *bins_MS)
        P15 = Pk(alpha, 1.5, *bins_MS)
        P2 = Pk(alpha, 2, *bins_MS)

        # floating point check for *very* thin bins (i.e. when mto ~ bin edge)
        finite_mask = ~np.isnan(P1)

        md = self.md
        ms = P2 / P1
        Ms = Ns * ms

        # Determine total loss rate (normalization)

        esc_rate = self.esc_rate(t) if self._time_dep_esc else self.esc_rate

        # If not core collapsed, use different, simpler, algorithm

        if t < self.tcc:

            if self._esc_norm == 'M':
                M_sum = np.sum(Ms[finite_mask]) + np.sum(np.r_[Mr])
                dNs += esc_rate * Ns / M_sum

                for c in range(len(Nr)):  # Each remnant class
                    sel = Nr[c] > 0
                    dNr[c][sel] += esc_rate * Nr[c][sel] / M_sum
                    dMr[c][sel] += esc_rate * Mr[c][sel] / M_sum

            elif self._esc_norm == 'N':
                N_sum = np.sum(Ns) + np.sum(np.r_[Nr])
                dNs += esc_rate * Ns / N_sum

                for c in range(len(Nr)):  # Each remnant class
                    sel = Nr[c] > 0
                    mr = Mr[c][sel] / Nr[c][sel]
                    dNr[c][sel] += esc_rate * Nr[c][sel] / N_sum
                    dMr[c][sel] += mr * esc_rate * Nr[c][sel] / N_sum

            return self.massbins.pack_values(dNs, dalpha, *dNr, *dMr)

        # Stellar Integrals

        depl_mask = (ms < md) & finite_mask

        Is = (Ns * (1 - md ** (-0.5) * (P15 / P1)))[depl_mask]

        # Remnant integrals

        *_, Ir, Jr = self.massbins.blanks(packed=False, grouped_rem=True)
        for c in range(len(Nr)):  # Each remnant class

            rem_mask = Nr[c] > 0

            mr = Mr[c][rem_mask] / Nr[c][rem_mask]

            Ir[c][rem_mask] = np.where(
                mr < md, Nr[c][rem_mask] * (1 - np.sqrt(mr / md)), 0
            )
            Jr[c][rem_mask] = np.where(
                mr < md, Mr[c][rem_mask] * (1 - np.sqrt(mr / md)), 0
            )

        # Normalization

        P25 = Pk(alpha, 2.5, *bins_MS)
        Js = (Ms * (1 - md ** (-0.5) * (P25 / P2)))[depl_mask]

        if self._esc_norm == 'M':
            P25 = Pk(alpha, 2.5, *bins_MS)
            Js = (Ms * (1 - md ** (-0.5) * (P25 / P2)))[depl_mask]
            B = esc_rate / (np.sum(Js) + np.sum(np.r_[Jr]))

        elif self._esc_norm == 'N':
            B = esc_rate / (np.sum(Is) + np.sum(np.r_[Ir]))

        else:
            raise ValueError(f"Invalid 'esc_norm' {self._esc_norm}")

        # Derivatives

        dNs[depl_mask] += B * Is

        dalpha[depl_mask] += (
            B * ((bins_MS.lower[depl_mask] / md) ** 0.5
                 - (bins_MS.upper[depl_mask] / md) ** 0.5)
            / np.log(bins_MS.upper[depl_mask] / bins_MS.lower[depl_mask])
        )

        for c in range(len(Nr)):  # Each remnant class

            rem_mask = Nr[c] > 0

            dNr[c][rem_mask] += B * Ir[c][rem_mask]
            dMr[c][rem_mask] += B * Jr[c][rem_mask]

        return self.massbins.pack_values(dNs, dalpha, *dNr, *dMr)

    def _dyn_eject_BH(self, Mr_BH, Nr_BH, *, M_eject=None):
        '''Determine and remove BHs, to represent dynamical ejections

        Determines and removes an amount of BHs from the BH mass bins, designed
        to reflect the effects of dynamical ejections of centrally concentrated
        BHs from a cluster.
        Proceeds from the heaviest BH bins to the lightest, removing all mass
        in each bin until the desired amount of ejected mass is reached.

        Both input arrays are modified *in place*, as well as returned.

        Parameters
        ----------
        Mr_BH : ndarray
            Array[nbin] of the total initial masses of black holes in each
            BH mass bin

        Nr_BH : ndarray
            Array[nbin] of the total initial numbers of black holes in each
            BH mass bin

        M_eject : float, optional
            Total amount of BH mass to remove through dynamical ejections.
            If None (default), will be computed based on the `BH_ret_dyn`
            parameter

        Returns
        -------
        Mr_BH : ndarray
            Array[nbin] of the total final masses of black holes in each
            BH mass bin, after dynamical ejections

        Nr_BH : ndarray
            Array[nbin] of the total final numbers of black holes in each
            BH mass bin, after dynamical ejections
        '''

        Mtot_0 = Mr_BH.sum()  # for error message

        # calculate total mass we want to eject
        if M_eject is None:
            M_eject = Mtot_0 * (1.0 - self.BH_ret_dyn)

        M_eject_0 = M_eject  # for error message

        # Remove BH starting from Heavy to Light
        j = Mr_BH.size

        while M_eject >= 0:
            j -= 1

            if j < 0:
                mssg = (f'Invalid `{M_eject_0=}`, '
                        f'must be less than total Mr_BH ({Mtot_0})')
                raise ValueError(mssg)

            # Removed entirety of this bin
            if Mr_BH[j] < M_eject:
                M_eject -= Mr_BH[j]
                Mr_BH[j] = 0
                Nr_BH[j] = 0
                continue

            # Remove required fraction of the last affected bin
            else:

                mr_BH_j = Mr_BH[j] / Nr_BH[j]

                Mr_BH[j] -= M_eject
                Nr_BH[j] -= M_eject / (mr_BH_j)

                break

        return Mr_BH, Nr_BH

    def _evolve(self):
        '''Main population evolution function, to be called on init'''

        # ------------------------------------------------------------------
        # Initialize output arrays
        # ------------------------------------------------------------------

        self.nout = len(self.tout)

        # Stars

        self.Ns, self.alpha, self.Nr, self.Mr = self.massbins.blanks(
            'empty', extra_dims=[self.nout], packed=False, grouped_rem=True
        )

        self.Ms, self.ms = np.empty_like(self.Ns), np.empty_like(self.Ns)
        self.mr = self.Nr._make(np.empty_like(x) for x in self.Nr)

        self.rem_types = np.repeat(self.massbins.nbin._fields[1:],
                                   self.massbins.nbin[1:])

        # ------------------------------------------------------------------
        # Initialise ODE solver
        # ------------------------------------------------------------------

        t0 = 0.0
        y0 = self.massbins.initial_values(N0=self.N0)

        sol = ode(self._derivs)
        sol.set_integrator("dopri5", max_step=1e12, atol=1e-5, rtol=1e-5)
        sol.set_initial_value(y0, t=t0)

        # ------------------------------------------------------------------
        # Evolve
        # ------------------------------------------------------------------

        for ti in self.t:

            # --------------------------------------------------------------
            # Integrate ODE solver
            # --------------------------------------------------------------

            sol.integrate(ti)

            # --------------------------------------------------------------
            # if this time is in the desired output times extract solutions
            # --------------------------------------------------------------

            if ti in self.tout:

                iout = np.where(self.tout == ti)[0][0]

                # ----------------------------------------------------------
                # Extract the N, M and alphas for stars and remnants
                # ----------------------------------------------------------

                Ns, alpha, Nr, Mr = self.massbins.unpack_values(
                    sol.y.copy(), grouped_rem=True
                )

                bins_MS = self.massbins.turned_off_bins(self.compute_mto(ti))

                As = Ns / Pk(alpha, 1, *bins_MS)
                Ms = As * Pk(alpha, 2, *bins_MS)

                # ----------------------------------------------------------
                # Eject BHs, first through natal kicks, then dynamically
                # ----------------------------------------------------------
                # TODO really feels like this should be done during the
                #   evolution/derivs? At least for the natal kicks.
                # TODO due to rem_classes tuple, ejections done in place, which
                #   is not ideal

                # Check if any BH have been created
                if ti > self.compute_tms(self.IFMR.BH_mi.upper):

                    # calculate total mass we want to eject
                    M_eject = Mr.BH.sum() * (1.0 - self.BH_ret_dyn)
                    M_ret = Mr.BH.sum() - M_eject

                    # If kicking basically all, skip ahead
                    if 0. <= M_ret / (Mr.BH[0] / Nr.BH[0]) < self.Nmin:
                        # TODO kick stats aren't stored when this is done
                        Mr.BH[:] = 0
                        Nr.BH[:] = 0

                    else:

                        # First remove mass from all bins by natal kicks
                        if self.natal_kicks:
                            *_, self._kick_stats = kicks.natal_kicks(
                                Mr.BH, Nr.BH, **self._kick_kw
                            )
                            M_eject -= self._kick_stats.total_kicked

                        if M_eject < 0:
                            mssg = (
                                f"Natal kicks already removed {-M_eject} Msun "
                                "more than total ejections desired by "
                                f"'BH_ret_dyn={self.BH_ret_dyn}'. "
                                "Increase BH_ret_dyn or alter natal kicks."
                            )
                            raise ValueError(mssg)

                        # Remove dynamical BH ejections
                        self._dyn_eject_BH(Mr.BH, Nr.BH, M_eject=M_eject)

                # ----------------------------------------------------------
                # save values into output arrays
                # ----------------------------------------------------------

                self.alpha[iout, :] = alpha

                # Stars
                self.Ns[iout, :] = Ns
                self.Ms[iout, :] = Ms

                self.ms[iout, :] = Ms / Ns

                # Remnants
                for c in range(len(Nr)):  # Each remnant class

                    self.Nr[c][iout, :] = Nr[c]
                    self.Mr[c][iout, :] = Mr[c]

                    # Precise mr only matters when Nr > 0
                    mr = 0.5 * np.sum(self.massbins.bins[c + 1], axis=0)

                    rem_mask = Nr[c] > 0
                    mr[rem_mask] = Mr[c][rem_mask] / Nr[c][rem_mask]

                    self.mr[c][iout, :] = mr

        Mtot = self.Ms.sum(axis=1) + np.c_[self.Mr].sum(axis=1)
        Ntot = self.Ns.sum(axis=1) + np.c_[self.Nr].sum(axis=1)
        self.mmean = Mtot / Ntot

        self.converged = sol.successful()
        if not self.converged:
            mssg = "ODE solver has *not* converged, this MF will not be valid."
            warnings.warn(mssg)


class EvolvedMFWithBH(EvolvedMF):
    r'''Evolve an IMF to a present-day mass function at a given age and f_BH.

    Subclass of `EvolvedMF`, evolving an arbitrary power law initial mass
    function (IMF) to a binned present-day mass function (PDMF) at a given set
    of ages, with an alternative BH prescription, allowing for a specified
    final BH mass fraction `f_BH`, rather than a BH retention fraction
    dependant on the amount of BHs created from the IMF.

    The given `f_BH` must be a valid fraction, between 0-1, and must be less
    than the mass fraction of BHs formed initially from the (and optionally
    after natal kicks), as BHs can only be removed through dynamical ejections.

    Parameters
    ----------
    IMF : PowerLawIMF
        Initial Mass Function (IMF) object, to be used to generate the initial
        numbers of stars at time 0.

    nbins : int or list of int or dict
        Number of mass bins defining the mass function.
        See `masses.MassBins` for more information.
        If list of int, must match number of components provided by
        binning_breaks (which defaults to IMF.mb).

    FeH : float
        Metallicity, in solar fraction [Fe/H].

    tout : list of int
        Times, in years, at which to output PDMF. Defines the shape of many
        outputted attributes.

    esc_rate : float or callable
        Represents rate of change of stars over time due to tidal
        ejections (and other escape mechanisms). Regulates low-mass object
        depletion (ejection) due to dynamical evolution.
        If a float, will apply a constant escape rate across all time. If a
        callable, must take in a time `t` in Myr and output a (negative) float
        representing the rate at that time.
        Rates must be in units of stars per Myr if `esc_norm` is 'N' or solar
        masses per Myr if `esc_norm` is 'M'.

    f_BH : float
        The desired final BH mass fraction (0 to 1).
        If more than one output time is specified, a list of floats with size
        `nout` is required.

    strict_BH_target : bool, optional
        If True, will error when the amount of BHs formed and then lost through
        natal kicks results in an f_BH less than the target final `f_BH`
        (this would make `f_BH` unreachable by dynamical BH losses). If False,
        the final `f_BH` may be smaller than requested.

    *args, **kwargs
        All remaining arguments are passed to `EvolvedMF`. Note that passing
        a `BH_ret_dyn` will have no effect on the BHs.

    Notes
    -----
    The final `f_BH` as determined by
    `self.M[self.types=='BH'].sum() / self.M.sum()` may be very slightly
    (ε < 1e-3) different from the requested `f_BH`, simply due to the removal
    of the nearly-empty bins in the creation of the `M` array.

    This can be remedied by setting `self.Nmin = 0`, though this is generally
    not recommended as bins with N < 1 are physically meaningless.
    '''

    def __init__(self, IMF, nbins, FeH, tout, esc_rate, f_BH,
                 *args, strict_BH_target=True, **kwargs):

        self._fBH_target = np.atleast_1d(f_BH)

        if self._fBH_target.size != np.atleast_1d(tout).size:
            mssg = "`f_BH` must be same size as desired output times `tout`"
            raise ValueError(mssg)

        if np.any(self._fBH_target < 0):
            mssg = f"`f_BH` ({self._fBH_target}) must be greater than 0"
            raise ValueError(mssg)

        self.strict_BH_target = strict_BH_target

        # leave BH_ret_dyn as default, will be ignored. BH_ret_int is fine
        super().__init__(IMF, nbins, FeH, tout, esc_rate,
                         *args, **kwargs)

    @classmethod
    def from_powerlaw(cls, m_breaks, a_slopes, nbins, FeH, tout, esc_rate, f_BH,
                      N0=5e5, *args, **kwargs):
        '''Construct class based on a power-law IMF breaks and slopes, directly.

        Alternative constructor to `EvolvedMFWithBH` based on explicitly
        providing the power-law IMF slopes and break masses, rather than an
        already created IMF object itself.
        Simply creates a `PowerLawIMF` class based on these arguments and passes
        it through to initialize the base class.

        Parameters
        ----------
        m_breaks : list of float
            Power law IMF break-masses (including outer bounds; size N+1).

        a_slopes : list of float
            Power law IMF slopes α (size N).

        nbins : int or list of int or dict
            Number of mass bins defining the mass function.
            See `masses.MassBins` for more information.
            If list of int, must match number of components provided by
            binning_breaks (which defaults to m_breaks; size N).

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        tout : list of int
            Times, in years, at which to output PDMF. Defines the shape of many
            outputted attributes.

        esc_rate : float or callable
            Represents rate of change of stars over time due to tidal
            ejections (and other escape mechanisms). Regulates low-mass object
            depletion (ejection) due to dynamical evolution.
            If a float, will apply a constant escape rate across all time. If a
            callable, must take in a time `t` in Myr and output a (negative)
            float representing the rate at that time.
            Rates must be in units of stars per Myr if `esc_norm` is 'N' or
            solar masses per Myr if `esc_norm` is 'M'.

        f_BH : float
            The desired final BH mass fraction (0 to 1).
            If more than one output time is specified, a list of floats with
            size `nout` is required.

        N0 : int or None, optional
            Total initial number of stars, over all bins.
            If None, uses the N0 of the given IMF class. Defaults to 5e5 stars.

        *args, **kwargs
            All other arguments are passed to EvolvedMF.

        Returns
        -------
        EvolvedMFwithBH
            The created evolved MF object, using the created IMF.
        '''
        imf = PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0, ext='zeros')
        return cls(imf, nbins, FeH, tout, esc_rate, f_BH, N0=N0,
                   *args, **kwargs)

    def _dyn_eject_BH(self, Mr_BH, Nr_BH, Mtot, fBH_target):
        '''Determine and remove BHs, to represent dynamical ejections.

        Determines and removes an amount of BHs from the BH mass bins, designed
        to reflect the effects of dynamical ejections of centrally concentrated
        BHs from a cluster.
        Proceeds from the heaviest BH bins to the lightest, removing all mass
        in each bin until the desired total BH mass fraction is reached.

        The target mass fraction *must* be less than the current
        (`Mr_BH.sum() / Mtot`) mass fraction, as mass will only be removed.

        Both input arrays are modified *in place*, as well as returned.

        Parameters
        ----------
        Mr_BH : ndarray
            Array[nbin] of the total initial masses of black holes in each
            BH mass bin.

        Nr_BH : ndarray
            Array[nbin] of the total initial numbers of black holes in each
            BH mass bin.

        Mtot : float
            The total cluster mass, before ejections. Needed to compute the
            current f_BH as BHs are removed.

        fBH_target : float
            The target BH mass fraction.

        Returns
        -------
        Mr_BH : ndarray
            Array[nbin] of the total final masses of black holes in each
            BH mass bin, after dynamical ejections

        Nr_BH : ndarray
            Array[nbin] of the total final numbers of black holes in each
            BH mass bin, after dynamical ejections
        '''

        def Mrem(Δfbh, Mb, Mt):
            '''Compute the amount of BH mass to remove required to reach Δfbh'''
            return (Mt**2 * Δfbh) / ((Mt * (1 + Δfbh)) - Mb)

        # Remove BH starting from Heavy to Light
        j = Mr_BH.size

        MBH = Mr_BH.sum()

        while (fBH_target < (fBH_current := MBH / Mtot)) and (j >= 0):
            j -= 1

            # Removed entirety of this bin
            if ((MBH - Mr_BH[j]) / (Mtot - Mr_BH[j])) >= fBH_target:

                MBH -= Mr_BH[j]
                Mtot -= Mr_BH[j]

                Mr_BH[j] = 0
                Nr_BH[j] = 0

                continue

            # Remove required fraction of the last affected bin
            else:

                Δfreq = fBH_current - fBH_target
                Mreq = Mrem(Δfreq, MBH, Mtot)

                mr_BH_j = Mr_BH[j] / Nr_BH[j]

                Mr_BH[j] -= Mreq
                Nr_BH[j] -= Mreq / (mr_BH_j)

                break

        return Mr_BH, Nr_BH

    def _evolve(self):
        '''Main population evolution function, to be called on init'''

        # ------------------------------------------------------------------
        # Initialize output arrays
        # ------------------------------------------------------------------

        self.nout = len(self.tout)

        # Stars

        self.Ns, self.alpha, self.Nr, self.Mr = self.massbins.blanks(
            'empty', extra_dims=[self.nout], packed=False, grouped_rem=True
        )

        self.Ms, self.ms = np.empty_like(self.Ns), np.empty_like(self.Ns)
        self.mr = self.Nr._make(np.empty_like(x) for x in self.Nr)

        self.rem_types = np.repeat(self.massbins.nbin._fields[1:],
                                   self.massbins.nbin[1:])

        # ------------------------------------------------------------------
        # Initialise ODE solver
        # ------------------------------------------------------------------

        t0 = 0.0
        y0 = self.massbins.initial_values(N0=self.N0)

        sol = ode(self._derivs)
        sol.set_integrator("dopri5", max_step=1e12, atol=1e-5, rtol=1e-5)
        sol.set_initial_value(y0, t=t0)

        # ------------------------------------------------------------------
        # Evolve
        # ------------------------------------------------------------------

        for ti in self.t:

            # --------------------------------------------------------------
            # Integrate ODE solver
            # --------------------------------------------------------------

            sol.integrate(ti)

            # --------------------------------------------------------------
            # if this time is in the desired output times extract solutions
            # --------------------------------------------------------------

            if ti in self.tout:

                iout = np.where(self.tout == ti)[0][0]

                # ----------------------------------------------------------
                # Extract the N, M and alphas for stars and remnants
                # ----------------------------------------------------------

                Ns, alpha, Nr, Mr = self.massbins.unpack_values(
                    sol.y.copy(), grouped_rem=True
                )

                bins_MS = self.massbins.turned_off_bins(self.compute_mto(ti))

                As = Ns / Pk(alpha, 1, *bins_MS)
                Ms = As * Pk(alpha, 2, *bins_MS)

                # ----------------------------------------------------------
                # Eject BHs, first through natal kicks, then dynamically
                # ----------------------------------------------------------

                # Check if any BH have been created
                if ti > self.compute_tms(self.IFMR.BH_mi.upper):

                    fBH_target = self._fBH_target[iout]

                    # First remove mass from all bins by natal kicks, if desired
                    #  Do it first because we dont control the exact amount so
                    #  this could make the target fBH invalid afterwards
                    if self.natal_kicks:
                        *_, self._kick_stats = kicks.natal_kicks(
                            Mr.BH, Nr.BH, **self._kick_kw
                        )

                    Mtot = np.r_[Mr].sum() + Ms.sum()
                    Mbhtot = Mr.BH.sum()
                    fBH_current = Mbhtot / Mtot

                    # can only make fBH go down by removing BHs
                    if fBH_target > fBH_current:
                        mssg = (
                            f"Target `f_BH` ({fBH_target}) is greater than "
                            f"f_BH formed at t={ti:.1f} ({fBH_current:.3f}; "
                            f"after natal kicks). Reduce `f_BH`, alter IMF "
                            f"slopes or turn off / alter natal kicks."
                        )
                        if self.strict_BH_target:
                            raise ValueError(mssg)
                        else:
                            # Will pass harmlessly through _dyn_eject_BH
                            mssg += (" Proceeding anyways due to "
                                     "`strict_BH_target=False`.")
                            warnings.warn(mssg)

                    # Remove dynamical BH ejections
                    #   (note different signature to EvolvedMF)
                    self._dyn_eject_BH(Mr.BH, Nr.BH, Mtot=Mtot,
                                       fBH_target=fBH_target)

                    # Reset, in case someone's curious
                    self.BH_ret_dyn = Mr.BH.sum() / Mbhtot

                # ----------------------------------------------------------
                # save values into output arrays
                # ----------------------------------------------------------

                self.alpha[iout, :] = alpha

                # Stars
                self.Ns[iout, :] = Ns
                self.Ms[iout, :] = Ms

                self.ms[iout, :] = Ms / Ns

                # Remnants
                for c in range(len(Nr)):  # Each remnant class

                    self.Nr[c][iout, :] = Nr[c]
                    self.Mr[c][iout, :] = Mr[c]

                    # Precise mr only matters when Nr > 0
                    mr = 0.5 * np.sum(self.massbins.bins[c + 1], axis=0)

                    rem_mask = Nr[c] > 0
                    mr[rem_mask] = Mr[c][rem_mask] / Nr[c][rem_mask]

                    self.mr[c][iout, :] = mr

        Mtot = self.Ms.sum(axis=1) + np.c_[self.Mr].sum(axis=1)
        Ntot = self.Ns.sum(axis=1) + np.c_[self.Nr].sum(axis=1)
        self.mmean = Mtot / Ntot

        self.converged = sol.successful()
        if not self.converged:
            mssg = "ODE solver has *not* converged, this MF will not be valid."
            warnings.warn(mssg)


class InitialBHPopulation:
    '''Generate an initial population of black holes from a mass function.

    Evolves and generates, from either a stellar IMF or an explicit BH mass
    function parametrization, the initial population of BHs which will form.
    This is similar, in general, to evolving a mass function as normal only
    up to the short age at which all of the most massive stars (which will
    make all of the BHs in a population) turn off and form remnants.

    As this class is *only* focused on the initial BH populations, the
    constituent DEs can be simplified, ignoring all other remnants (which will
    not form in time) and all stellar escapes (which are largely negligible
    by this time). This speeds up the computation considerably.

    This class should be initialized using one of the given classmethods,
    `from_IMF` (to evolve the cluster from a given IMF until all BHs are formed)
    or `from_BHMF` (to explicitly generate a population of BHs based on a given
    power-law BH mass function, without actually evolving anything).
    See the documentation on these methods for details on the available
    input parameters.

    Attributes
    ----------
    M, N, m : np.ndarray
        The total mass, total number and mean mass of BHs, in each mass bin.

    Mtot, Ntot : float
        The total mass and amount of BHs in the entire population.

    bins : masses.mbin
        The BH mass bin edges.

    age : float
        The final age, in Myr, of the evolution, i.e. when all BHs are formed.
        Only available if using the `from_IMF` init method.

    Ms_lost, Ns_lost : float
        The total mass and amount of stars lost, from the initial population
        based on the IMF, in the formation of the BHs.
        Only available if using the `from_IMF` init method.
    '''

    # TODO add unit tests

    age = None
    Ms_lost = None
    Ns_lost = None

    def f_BH(self, M_cluster):
        '''The total mass fraction in BHs, given a total population mass.'''
        return self.Mtot / M_cluster

    def __init__(self, M_BH, N_BH, BH_bins, FeH, *,
                 natal_kicks=False, kick_method='maxwellian', f_kick=None,
                 SNe_method='rapid', vesc=90, kick_vdisp=265.,
                 kick_slope=1, kick_scale=20):
        '''Should not init from this, use provided classmethods.'''

        # ------------------------------------------------------------------
        # Optionally perform all natal kicks on the input mass arrays
        # ------------------------------------------------------------------

        if natal_kicks:

            self.natal_kicks = natal_kicks

            match kick_method.casefold():

                case 'maxwellian' | 'f12' | 'fryer2012':

                    from .ifmr import _check_IFMR_FeH_bounds
                    FeH_BH = _check_IFMR_FeH_bounds(FeH)

                    kick_kw = dict(method=kick_method, f_kick=f_kick, vesc=vesc,
                                   FeH=FeH_BH, vdisp=kick_vdisp,
                                   SNe_method=SNe_method)

                case 'sigmoid' | 'tanh':
                    kick_kw = dict(method=kick_method, f_kick=f_kick,
                                   slope=kick_slope, scale=kick_scale)

                case _:
                    mssg = f"Invalid natal kick algorithm '{kick_method=}'"
                    raise ValueError(mssg)

            *_, self._kick_stats = kicks.natal_kicks(M_BH, N_BH, **kick_kw)

        # ------------------------------------------------------------------
        # Compute and store all final BH mass and number arrays and values
        # ------------------------------------------------------------------

        self.M = M_BH
        self.N = N_BH

        self.bins = BH_bins

        # get BH_mr
        zmsk = self.N > 0  # Precise mr only matters when Nr > 0
        self.m = 0.5 * np.sum(BH_bins, axis=0)
        self.m[zmsk] = self.M[zmsk] / self.N[zmsk]

        self.Mtot = self.M.sum()
        self.Ntot = self.N.sum()

    # ----------------------------------------------------------------------
    # Initializations
    # ----------------------------------------------------------------------

    @classmethod
    def from_IMF(cls, IMF, nbins, FeH, N0=5e5, *, natal_kicks=True,
                 binning_breaks=None, binning_method='default',
                 BH_IFMR_method='banerjee20', WD_IFMR_method='mist18',
                 BH_IFMR_kwargs=None, WD_IFMR_kwargs=None, **kwargs):
        '''Initialize a BH population by evolving from an IMF.

        Based on a given IMF, sharing many arguments with `EvolvedMF`, generate
        an initial stellar population and evolve it using simplified DEs to
        the age at which all BHs will have formed.

        As it is evolved from a stellar population, also stored is the age
        at which all BHs are formed, and the amount of stars which are lost
        to the formation of BHs (but not tidal escapes from a cluster).

        Parameters
        ----------
        IMF : PowerLawIMF
            Initial Mass Function (IMF) object, to be used to generate the
            initial numbers of stars at time 0.

        nbins : int or list of int or dict
            Number of mass bins defining the mass function.
            See `masses.MassBins` for more information.
            If list of int, must match number of components provided by
            binning_breaks (which defaults to IMF.mb).

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        N0 : int, optional
            Total initial number of stars, over all bins.
            If None, uses the N0 of the given IMF class. Defaults to 5e5 stars.

        natal_kicks : bool, optional
            Whether to account for natal kicks in the BH dynamical retention.
            Defaults to True (note this difference from `EvolvedMF`).

        vesc : float, optional
            Initial cluster escape velocity, in km/s, for use in the
            computation of BH natal kick effects. Defaults to 90 km/s.

        binning_breaks : list of float, optional
            The binning break masses to use when constructing the mass bins,
            including outer edges. See `masses.MassBins` for more information.
            If nbins is a list of int, this list must be one longer (i.e. bound
            all bins). By default, the break masses of the IMF will be used.

        binning_method : {'default', 'split_log', 'split_linear'}, optional
            The spacing method to use when constructing the mass bins.
            See `masses.MassBins` for more information.

        BH_IFMR_method : str, optional
            The BH IFMR algorithm to use when determining BH masses.
            See `ifmr` for more information.

        WD_IFMR_method : str, optional
            The WD IFMR algorithm to use when determining WD masses.
            See `ifmr` for more information.

        BH_IFMR_kwargs : dict, optional
            All arguments passed to the BH IFMR algorithm. See the specified
            functions in `ifmr` for information on all required methods.

        WD_IFMR_kwargs : dict, optional
            All arguments passed to the WD IFMR algorithm. See the specified
            functions in `ifmr` for information on all required methods.

        kwargs : dict, optional
            All other arguments are passed to the `InitialBHPopulation`.
        '''

        def compute_tms(mi):
            a = tms_constants
            return a[0] * np.exp(a[1] * mi ** a[2])

        def compute_mto(t):
            a0, a1, a2 = tms_constants
            if t > a0:
                return (np.log(t / a0) / a1) ** (1 / a2)
            else:
                return np.inf

        def _derivs_BHs(t, y):
            '''Derivatives relevant to mass changes due to stellar evolution.
            This is a massively simplified case of the DEs from `EvolvedMF`,
            and is only valid up to the age all BHs are formed.
            `y` is [MS N array, BH N array, BH M array].
            '''

            # Setup derivative bins
            Ns = y[:nbin_MS]  # only Ns is relevant right now
            dNs = np.zeros(nbin_MS)
            dNr, dMr = np.zeros(nbin_BH), np.zeros(nbin_BH)

            frem = 1.0  # Retain all BHs

            # Apply only if this time is atleast later than the earliest tms
            if t > tms_u[-1]:

                # Find out which mass bin is the current turn-off bin
                isev = np.where(t > tms_u)[0][0]

                mto, m1 = compute_mto(t), massbins.bins.MS.lower[isev]

                # Avoid "hitting" the bin edge
                if mto > m1 and (Nj := Ns[isev]) > 0.1:

                    # The normalization constant
                    # TODO deal with the constant divide-by-zero warning here
                    Aj = Nj / Pk(IMF.a[-1], 1, m1, mto)

                    # Get the number of turn-off stars per unit of mass
                    dNdm = Aj * mto**IMF.a[-1]

                else:
                    dNdm = 0
                    # TODO just break???

                # Compute the full dN/dt = dN/dm * dm/dt
                a = tms_constants
                dmdt = abs((1.0 / (a[1] * a[2] * t))
                           * (np.log(t / a[0]) / a[1]) ** (1 / a[2] - 1))

                dNdt = -dNdm * dmdt

                # Fill in star derivatives
                dNs[isev] = dNdt

                # Skip 0-mass remnants
                if t <= final_age and (m_rem := _ifmr.predict(mto)) > 0:

                    # If this happens its because IFMR is (numerically) broken!
                    if (cls_rem := _ifmr.predict_type(mto)) != 'BH':
                        mssg = f"Initial mass {mto} made a {cls_rem}, not a BH."
                        raise RuntimeError(mssg)

                    # Find bin based on lower bin edge (must be careful later)

                    irem = massbins.determine_index(m_rem, cls_rem)

                    # Fill in remnant derivatives
                    dNr[irem] = -dNdt * frem
                    dMr[irem] = -m_rem * dNdt * frem

            return np.r_[dNs, dNr, dMr]

        # ------------------------------------------------------------------
        # Initialise the initial mass function and mass bins given the
        # power-law IMF slopes and bins
        # ------------------------------------------------------------------

        _ifmr = IFMR(FeH, WD_method=WD_IFMR_method, WD_kwargs=WD_IFMR_kwargs,
                     BH_method=BH_IFMR_method, BH_kwargs=BH_IFMR_kwargs)

        binning_breaks = IMF.mb if binning_breaks is None else binning_breaks

        massbins = MassBins(binning_breaks, nbins, IMF, _ifmr,
                            binning_method=binning_method)

        nbin_MS, nbin_BH = massbins.nbin.MS, massbins.nbin.BH

        # Load a_i coefficients derived from interpolated Dartmouth models
        mstogrid = np.loadtxt(get_data("sevtables/msto.dat"))
        nearest_FeH = np.argmin(np.abs(mstogrid[:, 0] - FeH))
        tms_constants = mstogrid[nearest_FeH, 1:]

        # Compute t_ms for all bin edges
        tms_u = compute_tms(massbins.bins.MS.upper)

        init_N, init_M = massbins.initial_values(packed=False, N0=N0)

        # ------------------------------------------------------------------
        # Integrate and solve the derivatives (evolve)
        # ------------------------------------------------------------------

        y0 = np.r_[init_N.MS, init_N.BH, init_M.BH]

        sol = ode(_derivs_BHs)
        sol.set_integrator("dopri5", max_step=1e12, atol=1e-5, rtol=1e-5)
        sol.set_initial_value(y0, t=0.0)

        final_age = compute_tms(_ifmr.BH_mi.lower + 0.1)  # fltpnt bound errors

        # Integrate the solver at each bin bound, to match EvolvedMF.
        # The ODE solvers are suspiciously sensitive to this, and while this
        # may not be the most sane solution, this will recreate EvolvedMF.
        for ti in np.sort(tms_u[tms_u < final_age]):
            sol.integrate(ti)

        sol.integrate(final_age)

        N_BH, M_BH = sol.y[nbin_MS:nbin_MS + nbin_BH], sol.y[nbin_MS + nbin_BH:]

        # ------------------------------------------------------------------
        # Create the final class instance, and populate some extra values
        # specific to this init method, related to the evolution
        # ------------------------------------------------------------------

        out = cls(M_BH, N_BH, massbins.bins.BH,
                  FeH=FeH, natal_kicks=natal_kicks, **kwargs)

        out.IMF = IMF

        out.age = final_age

        Ns = sol.y[:nbin_MS]

        alphas = np.repeat(IMF.a, massbins._nbin_MS_each)
        As = Ns / Pk(alphas, 1, *massbins.bins.MS)
        Ms = As * Pk(alphas, 2, *massbins.bins.MS)

        # Stellar losses
        out.Ns_lost = init_N.MS.sum() - Ns.sum()
        out.Ms_lost = init_M.MS.sum() - Ms.sum()

        return out

    @classmethod
    def from_powerlaw(cls, m_breaks, a_slopes, nbins, FeH, N0=5e5, *,
                      natal_kicks=True,
                      binning_breaks=None, binning_method='default', **kwargs):
        '''Initialize a BH population by evolving from IMF breaks and slopes.

        Alternative constructor to `InitialBHPopulation.from_IMF` based on
        explicitly providing the power-law IMF slopes and break masses,
        rather than an already created IMF object itself.
        Simply creates a `PowerLawIMF` class based on these arguments and
        passes it through to `from_IMF`. See this method for more details.

        Parameters
        ----------
        m_breaks : list of float
            Power law IMF break-masses (including outer bounds; size N+1).

        a_slopes : list of float
            Power law IMF slopes α (size N).

        nbins : int or list of int or dict
            Number of mass bins defining the mass function.
            See `masses.MassBins` for more information.
            If list of int, must match number of components provided by
            binning_breaks (which defaults to m_breaks; size N).

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        N0 : int, optional
            Total initial number of stars, over all bins.
            If None, uses the N0 of the given IMF class. Defaults to 5e5 stars.

        natal_kicks : bool, optional
            Whether to account for natal kicks in the BH dynamical retention.
            Defaults to True (note this difference from `EvolvedMF`).

        vesc : float, optional
            Initial cluster escape velocity, in km/s, for use in the
            computation of BH natal kick effects. Defaults to 90 km/s.

        binning_breaks : list of float, optional
            The binning break masses to use when constructing the mass bins,
            including outer edges. See `masses.MassBins` for more information.
            If nbins is a list of int, this list must be one longer (i.e. bound
            all bins). By default, the break masses of the IMF will be used.

        binning_method : {'default', 'split_log', 'split_linear'}, optional
            The spacing method to use when constructing the mass bins.
            See `masses.MassBins` for more information.

        kwargs : dict, optional
            All other arguments are passed to the
            `InitialBHPopulation.from_IMF` constructor.
        '''

        imf = PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0, ext='zeros')

        return cls.from_IMF(imf, nbins, FeH, N0=N0, natal_kicks=natal_kicks,
                            binning_breaks=binning_breaks,
                            binning_method=binning_method, **kwargs)

    @classmethod
    def from_BHMF(cls, m_breaks, a_slopes, nbins, FeH, N0=1000, *,
                  natal_kicks=True, binning_method='default',
                  BH_IFMR_method='banerjee20', WD_IFMR_method='mist18',
                  BH_IFMR_kwargs=None, WD_IFMR_kwargs=None, **kwargs):
        '''Initialize a BH population based on a given power-law mass function.

        Based on an explicit given power-law parametrization of a BH mass
        function, generate a corresponding population of BH mass bins.
        This class *does not* simulate any evolution, and is not based on
        any sort of stellar mass function, but simply creates bins of mass
        (normalized to N0) for a given BH MF, and optionally provides
        natal kicks.

        Parameters
        ----------
        m_breaks : list of float
            IMF break-masses (including outer bounds; size N+1).

        a_slopes : list of float
            IMF slopes α. Supports any number of power law components (size N).

        nbins : int or list of int
            Number of mass bins in each regime of the given MF, as defined by
            a_slopes/m_breaks.

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        N0 : int, optional
            Total initial number of BHs, over all bins. Defaults to 1000 BHs.

        natal_kicks : bool, optional
            Whether to account for natal kicks in the BH dynamical retention.
            Defaults to True (note this difference from `EvolvedMF`).

        vesc : float, optional
            Initial cluster escape velocity, in km/s, for use in the
            computation of BH natal kick effects. Defaults to 90 km/s.

        binning_method : {'default', 'split_log', 'split_linear'}, optional
            The spacing method to use when constructing the mass bins.
            See `masses.MassBins` for more information.

        BH_IFMR_method : str, optional
            The BH IFMR algorithm to use when determining BH masses.
            See `ifmr` for more information.

        WD_IFMR_method : str, optional
            The WD IFMR algorithm to use when determining WD masses.
            See `ifmr` for more information.

        BH_IFMR_kwargs : dict, optional
            All arguments passed to the BH IFMR algorithm. See the specified
            functions in `ifmr` for information on all required methods.

        WD_IFMR_kwargs : dict, optional
            All arguments passed to the WD IFMR algorithm. See the specified
            functions in `ifmr` for information on all required methods.

        kwargs : dict, optional
            All other arguments are passed to the `InitialBHPopulation`.
        '''

        MF = PowerLawIMF(m_breaks, a_slopes, N0=N0)

        _ifmr = IFMR(FeH, WD_method=WD_IFMR_method, WD_kwargs=WD_IFMR_kwargs,
                     BH_method=BH_IFMR_method, BH_kwargs=BH_IFMR_kwargs)

        bins = MassBins(m_breaks, nbins, imf=MF, ifmr=_ifmr,
                        binning_method=binning_method).bins.MS

        N_BH, M_BH, _ = MF.binned_eval(bins)

        return cls(M_BH, N_BH, bins, FeH=FeH,
                   natal_kicks=natal_kicks, **kwargs)

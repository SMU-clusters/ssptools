#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d, UnivariateSpline

from .ifmr import IFMR, get_data

# TODO optionally support units for some things


__all__ = ['EvolvedMF']


class EvolvedMF:
    r'''Evolve an IMF to a present-day mass function at a given age

    Evolves an arbitrary N-component power law initial mass function (IMF)
    to a binned present-day mass function (PDMF) at a given set of ages, and
    computes the numbers and masses of stars and remnants in each mass bin.

    # TODO add more in-depth explanation or references to actual algorithm here

    Warning: Currently only supports N=3 IMF components

    Parameters
    ----------
    m_breaks : list of float
        IMF break-masses (including outer bounds; size N+1)

    a_slopes : list of float
        IMF slopes α (size N)

    nbins : list of int
        Number of mass bins in each regime of the IMF, as defined by m_breaks
        (size N)

    FeH : float
        Metallicity, in solar fraction [Fe/H]

    tout : list of int
        Times, in years, at which to output PDMF. Defines the shape of many
        outputted attributes

    Ndot : float
        Represents rate of change of the number of stars N over time, in stars
        per Myr. Regulates low-mass object depletion (ejection) due to dynamical
        evolution

    N0 : int, optional
        Total initial number of stars, over all bins. Defaults to 5e5 stars

    tcc : float, optional
        Core collapse time, in years. Defaults to 0, effectively being ignored

    NS_ret : float, optional
        Neutron star retention fraction (0 to 1). Defaults to 0.1 (10%)

    BH_ret_int : float, optional
        Initial black hole retention fraction (0 to 1). Defaults to 1 (100%)

    BH_ret_dyn : float, optional
        Dynamical black hole retention fraction (0 to 1). Defaults to 1 (100%)

    natal_kicks : bool, optional
        Whether to account for natal kicks in the BH dynamical retention.
        Defaults to False

    vesc : float, optional
        Initial cluster escape velocity, in km/s, for use in the computation of
        BH natal kick effects. Defaults to 90 km/s

    Attributes
    ----------

    nbin : int
        Total number of bins (sum of nbins parameter)

    nout : int
        Total number of output times (len of tout parameter)

    alphas : ndarray
        Array[nout, nbin] of  PDMF slopes. If Ndot = 0, this is defined entirely
        by the IMF

    Ns : ndarray
        Array[nout, nbin] of the total number of (main sequence) stars in each
        mass bin

    Ms : ndarray
        Array[nout, nbin] of the total mass of (main sequence) stars in each
        mass bin

    ms : ndarray
        Array[nout, nbin] of the individual mass of (main sequence) stars in
        each mass bin, as defined by Ms / Ns

    mes : ndarray
        Array[nout, nbin + 1] representing the mass bin edges, defined by the
        m_breaks and nbins parameters. Note that ms *does not* necessarily fall
        in the middle of these edges

    Nr : ndarray
        Array[nout, nbin] of the total number of remnants in each mass bin

    Mr : ndarray
        Array[nout, nbin] of the total mass of remnants in each mass bin

    mr : ndarray
        Array[nout, nbin] of the individual mass of remnants in
        each mass bin, as defined by Mr / Nr

    rem_types : ndarray
        Array[nout, nbin] of 2-character strings representing the remnant types
        found in each mass bin of the remnant (Mr, Nr, mr) arrays. "WD" = white
        dwarfs, "NS" = neutron stars, "BH" = black holes

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
        stars, "WD" = white dwarfs, "NS" = neutron stars, "BH" = black holes
    '''

    @property
    def M(self):
        # TODO why is it 10*Nmin
        cs = self.Ns[-1] > 10 * self.Nmin
        Ms = self.Ms[-1][cs]

        cr = self.Nr[-1] > 10 * self.Nmin
        Mr = self.Mr[-1][cr]

        return np.r_[Ms, Mr]

    @property
    def N(self):
        cs = self.Ns[-1] > 10 * self.Nmin
        Ns = self.Ns[-1][cs]

        cr = self.Nr[-1] > 10 * self.Nmin
        Nr = self.Nr[-1][cr]

        return np.r_[Ns, Nr]

    @property
    def m(self):
        return self.M / self.N

    @property
    def bin_widths(self):
        bw = np.diff(self.mes[-1])

        cs = self.Ns[-1] > 10 * self.Nmin
        bws = bw[cs]

        cr = self.Nr[-1] > 10 * self.Nmin
        bwr = bw[cr]

        return np.r_[bws, bwr]

    @property
    def types(self):
        cs = self.Ns[-1] > 10 * self.Nmin
        cr = self.Nr[-1] > 10 * self.Nmin

        ts = ['MS'] * cs.sum()
        tr = self.rem_types[cr]

        return np.r_[ts, tr]

    @property
    def nms(self):
        return (self.Ns[-1] > 10 * self.Nmin).sum()

    @property
    def nmr(self):
        return (self.Nr[-1] > 10 * self.Nmin).sum()

    def __init__(self, m_breaks, a_slopes, nbins, FeH, tout, Ndot,
                 N0=5e5, tcc=0.0, NS_ret=0.1, BH_ret_int=1.0, BH_ret_dyn=1.0,
                 natal_kicks=False, vesc=90):

        # ------------------------------------------------------------------
        # Initialise the mass bins given the power-law IMF slopes and bins
        # ------------------------------------------------------------------

        self._set_imf(m_breaks, a_slopes, nbins, N0)

        # ------------------------------------------------------------------
        # Set various other parameters
        # ------------------------------------------------------------------

        # Supplied parameters
        self.tcc = tcc
        self.tout = np.atleast_1d(tout)
        self.Ndot = Ndot
        self.NS_ret = NS_ret
        self.BH_ret_int = BH_ret_int
        self.BH_ret_dyn = BH_ret_dyn
        self.FeH = FeH

        if Ndot > 0:
            raise ValueError("'Ndot' must be less than 0")

        # Initial-Final mass relations
        self.IFMR = IFMR(FeH)

        # Minimum of stars to call a bin "empty"
        self.Nmin = 0.1

        # Depletion mass: stars below this mass are preferentially disrupted
        # Hardcoded for now, perhaps vary, fit on N-body?
        self.md = 1.2

        # ------------------------------------------------------------------
        # Setup lifetime approximations and compute t_ms of all bin edges
        # ------------------------------------------------------------------

        # Load a_i coefficients derived from interpolated Dartmouth models
        mstogrid = np.loadtxt(get_data("sevtables/msto.dat"))
        nearest_FeH = np.argmin(np.abs(mstogrid[:, 0] - FeH))
        self._tms_constants = mstogrid[nearest_FeH, 1:]

        # Compute t_ms for all bin edges
        self.tms_l = self.compute_tms(self.me[:-1])
        self.tms_u = self.compute_tms(self.me[1:])

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

        self.vesc = vesc
        self.natal_kicks = natal_kicks

        if self.natal_kicks:

            # load in the ifmr data to interpolate fb from mr
            feh_path = get_data(f"sse/MP_FEH{self.IFMR.FeH_BH:+.2f}.dat")

            # load in the data
            self.fb_grid = np.loadtxt(feh_path, usecols=(1, 3), unpack=True)

        # ------------------------------------------------------------------
        # Finally, evolve the population
        # ------------------------------------------------------------------

        # Initialize iteration counter
        self.nstep = 0

        self._evolve()

    @np.errstate(invalid='ignore')
    def _Pk(self, a, k, m1, m2):
        r'''Convenience function for computing quantities related to IMF

        ..math ::
            \begin{align}
                P_k(\alpha_j,\ m_{j,1},\ m_{j,2})
                    &= \int_{m_{j,1}}^{m_{j,2}} m^{\alpha_j + k - 1}
                        \ \mathrm{d}m \\
                    &= \begin{cases}
                        \frac{m_{j,2}^{\alpha_j+k} - m_{j,1}^{\alpha_j+k} }
                             {\alpha_j + k},
                             \quad \alpha_j + k \neq 0 \\
                        \ln{\left(\frac{m_{j,2}}{m_{j,1}}\right)},
                            \quad \alpha_j + k = 0
                    \end{cases}
            \end{align}

        Parameters
        ----------
        a : float
            Mass function power law slope effective between m1 and m2

        k : int
            k-index

        m1, m2 : float
            Upper and lower bound of given mass bin or range
        '''

        a = np.asarray(a, dtype=float)
        res = np.asarray((m2 ** (a + k) - m1 ** (a + k)) / (a + k))  # a != k

        if (casemask := np.asarray(-a == k)).any():
            res[casemask] = np.log(m2 / m1)[casemask]

        return res

    def _set_imf(self, m_break, a, nbin, N0):
        '''Initialize the mass bins based on the IMF and initial number of stars

        Modifies in place a number of attributes of this class, related to the
        initial setup of the output quantities, including nbin, me, mes0,
        alphas0, Ns0, Ms0, ms0, Nr0, Mr0, mr0

        Parameters
        ----------
        m_break : list of float
            Break-masses (including outer bounds; size N+1)

        a : list of float
            mass function slopes (size N)

        nbin : list of int
            Number of mass bins in each regime (size N)

        N0 : int
            Total initial number of stars
        '''

        # Total number of mass bins
        self.nbin = np.sum(nbin)

        # ------------------------------------------------------------------
        # Compute normalization factors A_j
        # ------------------------------------------------------------------

        A3 = (
            self._Pk(a[2], 1, m_break[2], m_break[3])
            + (m_break[1] ** (a[1] - a[0])
               * self._Pk(a[0], 1, m_break[0], m_break[1]))
            + (m_break[2] ** (a[2] - a[1])
               * self._Pk(a[1], 1, m_break[1], m_break[2]))
        ) ** (-1)

        A2 = A3 * m_break[2] ** (a[2] - a[1])
        A1 = A2 * m_break[1] ** (a[1] - a[0])

        A = N0 * np.repeat([A1, A2, A3], nbin)

        # ------------------------------------------------------------------
        # Set mass bin edges
        # Bins are logspaced evenly between the break masses, with the
        # number of bins specified by nbin. This spacing is thus *not*
        # consistent throughtout entire mass range.
        # ------------------------------------------------------------------

        # TODO equal-log-space between bins, would single logspace be better?
        me1 = np.geomspace(m_break[0], m_break[1], nbin[0] + 1)
        me2 = np.geomspace(m_break[1], m_break[2], nbin[1] + 1)
        me3 = np.geomspace(m_break[2], m_break[3], nbin[2] + 1)

        # Combine bin edges, avoiding repetition
        self.me = np.r_[me1, me2[1:], me3[1:]]

        # Set special edges for stars because stellar evolution affects this
        self.mes0 = np.copy(self.me)

        # ------------------------------------------------------------------
        # Set the initial Nj and mj for all bins (stars and remnants)
        # ------------------------------------------------------------------

        # Expand array of IMF slopes to all mass bins
        alpha = np.repeat(a, nbin)
        self.alphas0 = alpha

        # Set initial star bins based on IMF
        self.Ns0 = A * self._Pk(alpha, 1, self.me[0:-1], self.me[1:])
        self.Ms0 = A * self._Pk(alpha, 2, self.me[0:-1], self.me[1:])
        self.ms0 = self.Ms0 / self.Ns0

        # Set all initial remnant bins to zero
        self.Nr0 = np.zeros(self.nbin)
        self.Mr0 = np.zeros(self.nbin)
        self.mr0 = np.zeros(self.nbin)

    def compute_tms(self, mi):
        '''Compute main-sequence lifetime for a given mass `mi`'''
        a = self._tms_constants
        return a[0] * np.exp(a[1] * mi ** a[2])

    def compute_mto(self, t):
        '''Compute the turn-off mass for a given time `t` (inverse of tms)'''
        a = self._tms_constants
        return (np.log(t / a[0]) / a[1]) ** (1 / a[2])

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
        derivs_sev = self._derivs_sev(t, y)

        # Only run the dynamical star losses `derivs_esc` if Ndot is not zero
        if self.Ndot < 0:
            derivs_esc = self._derivs_esc(t, y)
        else:
            derivs_esc = np.zeros_like(derivs_sev)

        # Combine mass loss derivatives
        return derivs_sev + derivs_esc

    def _derivs_sev(self, t, y):
        '''Derivatives relevant to mass changes due to stellar evolution'''

        # Setup derivative bins
        Nj_dot_s, Nj_dot_r = np.zeros(self.nbin), np.zeros(self.nbin)
        Mj_dot_r = np.zeros(self.nbin)
        aj_dot_s = np.zeros(self.nbin)

        # Apply only if this time is atleast later than the earliest tms
        if t > self.tms_u[-1]:

            # Find out which mass bin is the current turn-off bin
            isev = np.where(t > self.tms_u)[0][0]

            # Find bin edges of turn-off bin
            m1 = self.me[isev]
            mto = self.compute_mto(t)
            Nj = y[isev]

            # Avoid "hitting" the bin edge
            # i.e. when evolving with tout: mto > m1, otherwise: mto == m1
            if mto > m1 and Nj > self.Nmin:

                # Two parameters defining the bin
                alphaj = y[self.nbin + isev]

                # The normalization constant
                # TODO deal with the constant divide-by-zero warning here
                Aj = Nj / self._Pk(alphaj, 1, m1, mto)

                # Get the number of turn-off stars per unit of mass
                dNdm = Aj * mto ** alphaj

            else:
                dNdm = 0

            # Compute the full dN/dt = dN/dm * dm/dt
            a = self._tms_constants
            dmdt = abs((1.0 / (a[1] * a[2] * t))
                       * (np.log(t / a[0]) / a[1]) ** (1 / a[2] - 1))

            dNdt = -dNdm * dmdt

            # Fill in star derivatives (alphaj remains constant for _derivs_sev)
            Nj_dot_s[isev] = dNdt

            # Find remnant mass and which bin they go into
            m_rem, cls_rem = self.IFMR.predict(mto), self.IFMR.predict_type(mto)

            # Skip 0-mass remnants
            if m_rem > 0:

                # Find bin based on lower bin edge (must be careful later)
                irem = np.where(m_rem > self.me)[0][-1]

                # Compute Remnant retention fractions based on remnant type

                if cls_rem == 'WD':
                    frem = 1.

                elif cls_rem == 'BH':
                    frem = self.BH_ret_int

                # elif cls_rem == 'NS':
                else:
                    frem = self.NS_ret

                # Fill in remnant derivatives
                Nj_dot_r[irem] = -dNdt * frem
                Mj_dot_r[irem] = -m_rem * dNdt * frem

        return np.r_[Nj_dot_s, aj_dot_s, Nj_dot_r, Mj_dot_r]

    def _derivs_esc(self, t, y):
        '''Derivatives relevant to mass loss due to escaping low-mass stars'''

        nb = self.nbin
        md = self.md
        Ndot = self.Ndot

        # Setup derivative bins
        Nj_dot_s, aj_dot_s = np.zeros(nb), np.zeros(nb)
        Nj_dot_r, Mj_dot_r = np.zeros(nb), np.zeros(nb)

        # Pull out individual arrays from y
        Ns = np.abs(y[0:nb])
        alphas = y[nb:2 * nb]
        Nr = np.abs(y[2 * nb:3 * nb])
        Mr = np.abs(y[3 * nb:4 * nb])

        # If core collapsed, use different, simpler, algorithm
        if t < self.tcc:
            N_sum = Ns.sum() + Nr.sum()
            Nj_dot_s += Ndot * Ns / N_sum
            sel = Nr > 0
            Nj_dot_r[sel] += Ndot * Nr[sel] / N_sum
            Mj_dot_r[sel] += (Ndot * Nr[sel] / N_sum) * (Mr[sel] / Nr[sel])
            return np.r_[Nj_dot_s, aj_dot_s, Nj_dot_r, Mj_dot_r]

        # Determine mass of remnants
        mr = 0.5 * (self.me[1:] + self.me[0:-1])
        c = Nr > 0
        mr[c] = Mr[c] / Nr[c]

        # Setup edges for stars accounting for mto
        mes = np.copy(self.me)

        # Set all bins above current turn-off to this mto
        if t > self.tms_u[-1]:
            isev = np.where(mes > self.compute_mto(t))[0][0]
            mes[isev] = self.compute_mto(t)

        # Helper mass and normalization quantities
        m1 = mes[0:-1]
        m2 = mes[1:]

        P1 = self._Pk(alphas, 1, m1, m2)
        P15 = self._Pk(alphas, 1.5, m1, m2)

        # Compute relevant rate-of-change integrals I_j, J_j
        c = (mr < self.md) & (m1 < m2)

        Is = Ns[c] * (1 - md ** (-0.5) * P15[c] / P1[c])
        Ir = Nr[c] * (1 - np.sqrt(mr[c] / md))
        Jr = Mr[c] * (1 - np.sqrt(mr[c] / md))

        # Compute normalization constant B
        B = Ndot / sum(Is + Ir)

        # Compute rates of change for all four quantities (cumulative per bin)
        Nj_dot_s[c] += B * Is
        aj_dot_s[c] += (B * ((m1[c] / md) ** 0.5 - (m2[c] / md) ** 0.5)
                        / np.log(m2[c] / m1[c]))
        Nj_dot_r[c] += B * Ir
        Mj_dot_r[c] += B * Jr

        return np.r_[Nj_dot_s, aj_dot_s, Nj_dot_r, Mj_dot_r]

    def _get_retention_frac(self, fb, vesc):
        '''Compute BH natal-kick retention fraction'''

        def _maxwellian(x, a):
            norm = np.sqrt(2 / np.pi)
            exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
            return norm * exponent / a ** 3

        if fb == 1.0:
            return 1.0

        # Integrate over the Maxwellian up to the escape velocity
        v_space = np.linspace(0, 1000, 1000)

        # TODO might be a quicker way to numerically integrate than a spline
        retention = UnivariateSpline(
            x=v_space,
            y=_maxwellian(v_space, 265 * (1 - fb)),
            s=0,
            k=3,
        ).integral(0, vesc)

        return retention

    def _natal_kick_BH(self, Mr_BH, Nr_BH):
        '''Computes the effects of BH natal-kicks on the mass and number of BHs

        Determines the effects of BH natal-kicks based on the fallback fraction
        and escape velocity of this population.

        Parameters
        ----------
        Mr_BH : ndarray
            Array[nbin] of the total initial masses of black holes in each
            BH mass bin

        Nr_BH : ndarray
            Array[nbin] of the total initial numbers of black holes in each
            BH mass bin

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

        # Interpolate the mr-fb grid
        fb_interp = interp1d(
            self.fb_grid[0],
            self.fb_grid[1],
            kind="linear",
            bounds_error=False,
            fill_value=(0.0, 1.0),
        )

        for j in range(Mr_BH.size):

            # Skip the bin if its empty
            if Nr_BH[j] < self.Nmin:
                continue

            # Interpolate fallback fraction at this mr
            fb = fb_interp(Mr_BH[j] / Nr_BH[j])

            if fb == 1.0:
                continue

            else:

                # Compute retention fraction
                retention = self._get_retention_frac(fb, self.vesc)

                # keep track of how much we eject
                natal_ejecta += Mr_BH[j] * (1 - retention)

                # eject the mass
                Mr_BH[j] *= retention
                Nr_BH[j] *= retention

        return Mr_BH, Nr_BH, natal_ejecta

    def _dyn_eject_BH(self, Mr_BH, Nr_BH, *, M_eject=None):
        '''Determine and remove BHs, to represent dynamical ejections

        Determines and removes an amount of BHs from the BH mass bins, designed
        to reflect the effects of dynamical ejections of centrally concentrated
        BHs from a cluster.
        Proceeds from the heaviest BH bins to the lightest, removing all mass
        in each bin until the desired amount of ejected mass is reached.

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

        # calculate total mass we want to eject
        if M_eject is None:
            M_eject = Mr_BH.sum() * (1.0 - self.BH_ret_dyn)

        # Remove BH starting from Heavy to Light
        j = Mr_BH.size

        while M_eject != 0:
            j -= 1

            if j < 0:
                mssg = 'Invalid `M_eject`, must be less than total Mr_BH'
                raise ValueError(mssg)

            # Skip empty bins
            if Nr_BH[j] < self.Nmin:
                continue

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

        nb, nout = self.nbin, len(self.tout)

        self.nout = nout

        self.alphas = np.empty((nout, nb))

        self.Ns = np.empty((nout, nb))
        self.Ms = np.empty((nout, nb))
        self.ms = np.empty((nout, nb))

        self.mes = np.empty((nout, nb + 1))

        self.Nr = np.empty((nout, nb))
        self.Mr = np.empty((nout, nb))
        self.mr = np.empty((nout, nb))

        # ------------------------------------------------------------------
        # Initialise ODE solver
        # ------------------------------------------------------------------

        t0 = 0.0
        y = np.r_[self.Ns0, self.alphas0, self.Nr0, self.Mr0]

        sol = ode(self._derivs)
        sol.set_integrator("dopri5", max_step=1e12, atol=1e-5, rtol=1e-5)
        sol.set_initial_value(y, t=t0)

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

                # Extract stars
                Ns = sol.y[0:nb]
                alphas = sol.y[nb:2 * nb]

                # Some special treatment to adjust current turn-off bin edge
                mes = np.copy(self.me)
                if ti > self.tms_u[-1]:
                    isev = np.where(self.me > self.compute_mto(ti))[0][0] - 1
                    mes[isev + 1] = self.compute_mto(ti)

                As = Ns / self._Pk(alphas, 1, mes[0:-1], mes[1:])
                Ms = As * self._Pk(alphas, 2, mes[0:-1], mes[1:])

                # Extract remnants (copies due to below ejections)
                Nr = sol.y[2 * nb:3 * nb].copy()
                Mr = sol.y[3 * nb:4 * nb].copy()

                # ----------------------------------------------------------
                # Determine types of remnants/stars in each bin
                # ----------------------------------------------------------
                # *not* the same as `IFMR.predict_type`, but the probable
                #   remnant in each final bin

                # TODO change how bins are done, this isn't guaranteed accurate
                rem_types = np.full(nb, 'NS')
                rem_types[self.me[:-1] < self.IFMR.WD_mf.upper] = 'WD'

                # Special handling to also include the bin containing mBH_min
                cutoff = self.me[:-1][self.me[:-1] < self.IFMR.BH_mf.lower][-1]
                rem_types[self.me[:-1] >= cutoff] = 'BH'

                # ----------------------------------------------------------
                # Eject BHs, first through natal kicks, then dynamically
                # ----------------------------------------------------------
                # TODO really feels like this should be done during the
                #   evolution/derivs? At least for the natal kicks.

                # Check if any BH have been created
                if ti > self.compute_tms(self.IFMR.BH_mi.lower):

                    BH_cut = (rem_types == 'BH')

                    Mr_BH, Nr_BH = Mr[BH_cut], Nr[BH_cut]

                    # calculate total mass we want to eject
                    M_eject = Mr_BH.sum() * (1.0 - self.BH_ret_dyn)

                    # First remove mass from all bins by natal kicks, if desired
                    if self.natal_kicks:
                        Mr_BH, Nr_BH, kicked = self._natal_kick_BH(Mr_BH, Nr_BH)
                        M_eject -= kicked

                    # Remove dynamical BH ejections
                    Mr[BH_cut], Nr[BH_cut] = self._dyn_eject_BH(Mr_BH, Nr_BH,
                                                                M_eject=M_eject)

                # ----------------------------------------------------------
                # save values into output arrays
                # ----------------------------------------------------------

                self.alphas[iout, :] = alphas

                # Stars
                self.Ns[iout, :] = Ns
                self.Ms[iout, :] = Ms

                self.ms[iout, :] = Ms / Ns

                # Edges of star mass bins
                self.mes[iout, :] = mes

                # Remnants
                self.Nr[iout, :] = Nr
                self.Mr[iout, :] = Mr

                # Precise mr only matters when Nr > 0
                mr = 0.5 * (self.me[1:] + self.me[0:-1])
                mr[Nr > 0] = Mr[Nr > 0] / Nr[Nr > 0]
                self.mr[iout, :] = mr

                # Remnant types
                self.rem_types = rem_types

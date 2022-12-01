#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d, UnivariateSpline

from .ifmr import IFMR, get_data


class evolve_mf:
    r'''
    Class to evolve the stellar mass function, to be included in EMACSS
    For nbin mass bins, the routine solves for an array with length 4nbin,
    containing:
    y = {N_stars_j, alpha_stars_j, N_remnants_j, M_remnants_j}

    based on \dot{y}

    Parameters
    ----------
    m123 : list of float
        Break-masses (including outer bounds; size N+1)

    a12 : list of float
        mass function slopes (size N)

    nbin12 : list of int
        Number of mass bins in each regime (size N)

    N0 : int
        Total initial number of stars

    tout : list of int
        Times to output masses at [years]

    Ndot : float
        Regulates low-mass object depletion due to dynamical evolution
        [stars / Myr]

    tcc : float
        Core collapse time

    NS_ret : float
        Neutron star retention fraction (0 to 1)

    BH_ret_int : float
        Initial black hole retention fraction (0 to 1)

    BH_ret_dyn : float
        Dynamical black hole retention fraction (0 to 1)

    FeH : float
        Metallicity, in solar fraction [Fe/H]

    natal_kicks : bool
        Whether to account for natal kicks in the BH dynamical retention

    vesc : float
        Cluster escape velocity, in km/s, for use in the computation of BH
        natal kick effects

    '''

    def __init__(self, m123, a12, nbin12, tout, N0, Ndot, tcc,
                 NS_ret, BH_ret_int, BH_ret_dyn,
                 FeH, natal_kicks=False, vesc=90, include_t0=False):

        # ------------------------------------------------------------------
        # Initialise the mass bins given the power-law IMF slopes and bins
        # ------------------------------------------------------------------

        self._set_imf(m123, a12, nbin12, N0)

        # ------------------------------------------------------------------
        # Set various other parameters
        # ------------------------------------------------------------------

        # Supplied parameters
        self.tcc = tcc
        self.tout = tout
        self.Ndot = Ndot
        self.NS_ret = NS_ret
        self.BH_ret_int = BH_ret_int
        self.BH_ret_dyn = BH_ret_dyn
        self.FeH = FeH

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

        self.evolve(include_t0=include_t0)

    def Pk(self, a, k, m1, m2):
        '''Useful function

        ..math ::

        '''
        try:
            return (m2 ** (a + k) - m1 ** (a + k)) / (a + k)

        except ZeroDivisionError:
            return np.log(m2 / m1)

    def _set_imf(self, m_break, a, nbin, N0):
        '''Initialize the mass bins based on the IMF and initial number of stars

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
            self.Pk(a[2], 1, m_break[2], m_break[3])
            + (m_break[1] ** (a[1] - a[0])
               * self.Pk(a[0], 1, m_break[0], m_break[1]))
            + (m_break[2] ** (a[2] - a[1])
               * self.Pk(a[1], 1, m_break[1], m_break[2]))
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
        self.Ns0 = A * self.Pk(alpha, 1, self.me[0:-1], self.me[1:])
        self.Ms0 = A * self.Pk(alpha, 2, self.me[0:-1], self.me[1:])
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

        # TODO why is this hard limit applied? if necessary, shouldnt recompute
        if t < self.compute_tms(100):
            mto = 100

        else:
            mto = (np.log(t / a[0]) / a[1]) ** (1 / a[2])

        return mto

    def _derivs(self, t, y):
        '''Main function for computing derivatives relevant to mass evolution

        Simply calls the two constituent mass evolution derivative methods;
        `_derivs_esc` and `_derivs_sev`. Designed to be solved using an ODE
        integrator, such as `scipy.integrate.ode`.

        Parameters
        ----------
        t : float
            Time step to compute derivatives at

        y : list of array
            Equation system solution y. Size-4 array containing the arrays,
            for each mass bin, of the number of stars `Ns`, the mass function
            slopes `alpha`, the number of remnants `Nr` and the total bin mass
            of remnants `Mr`

        Returns
        -------
        list of array
            Time derivatives of each of the four quantities described by `y`
        '''

        # Iterate step count
        self.nstep += 1

        # Compute stellar evolution derivatives
        derivs_sev = self._derivs_sev(t, y)

        # Only run the dynamical star losses `derivs_esc` if Ndot is not zero
        # TODO add initial check that Ndot is not > 0
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

        # Apply only to bins affected by stellar evolution at this time
        if t > self.tms_u[-1]:

            # Find out which mass bin is the current turn-off bin
            isev = np.where(t > self.tms_u)[0][0]

            # Find bin edges of turn-off bin
            m1 = self.me[isev]
            mto = self.compute_mto(t)
            Nj = y[isev]

            # Avoid "hitting" the bin edge
            if mto > m1 and Nj > self.Nmin:

                # Two parameters defining the bin
                alphaj = y[self.nbin + isev]

                # The normalization constant
                Aj = Nj / self.Pk(alphaj, 1, m1, mto)

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
            mrem = self.IFMR.predict(mto)

            # Skip 0-mass remnants
            if mrem > 0:

                irem = np.where(mrem > self.me)[0][-1]

                # frem = 1  # full retention for WD

                # if mrem >= 1.36:
                #     frem = self.NS_ret

                # if mrem >= self.IFMR.mBH_min:
                #     frem = self.BH_ret_int

                # Compute Remnant retention fractions based on remnant type
                # TODO maybe make predict return type directly to skip this
                if mto < self.IFMR.wd_m_max:
                    # White Dwarf retention (always 100%)
                    frem = 1.

                elif mrem >= self.IFMR.mBH_min:
                    # Black Hole initial retention
                    frem = self.BH_ret_int

                else:
                    # Neutron Star retention
                    frem = self.NS_ret

                # Fill in remnant derivatives
                Nj_dot_r[irem] = -dNdt * frem
                Mj_dot_r[irem] = -mrem * dNdt * frem

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

        P1 = self.Pk(alphas, 1, m1, m2)
        P15 = self.Pk(alphas, 1.5, m1, m2)

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

    def get_retention(self, fb, vesc):
        '''Compute BH natal-kick retention fraction

        ..math::

        based on fallback fraction and escape velocity
        '''

        def _maxwellian(x, a):
            norm = np.sqrt(2 / np.pi)
            exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
            return norm * exponent / a ** 3

        if fb == 1.0:
            return 1.0

        # Integrate over the Maxwellian up to the escape velocity
        v_space = np.linspace(0, 1000, 1001)

        # TODO might be a quicker way to numerically integrate than a spline
        retention = UnivariateSpline(
            x=v_space,
            y=_maxwellian(v_space, 265 * (1 - fb)),
            s=0,
            k=3,
        ).integral(0, vesc)

        return retention

    def extract_arrays(self, t, y):
        nb = self.nbin
        # Extract total N, M and split in Ns and Ms
        Ns = y[0:nb]
        alphas = y[nb:2 * nb]

        # Some special treatment to adjust edges to mto
        mes = np.copy(self.me)
        if t > self.tms_u[-1]:
            isev = np.where(self.me > self.compute_mto(t))[0][0] - 1
            mes[isev + 1] = self.compute_mto(t)

        As = Ns / self.Pk(alphas, 1, mes[0:-1], mes[1:])
        Ms = As * self.Pk(alphas, 2, mes[0:-1], mes[1:])

        Nr = y[2 * nb:3 * nb].copy()
        Mr = y[3 * nb:4 * nb].copy()

        # Do BH cut, if all BH where created
        if self.compute_tms(self.IFMR.m_min) < t:

            sel1 = self.me[:-1][self.me[:-1] < self.IFMR.mBH_min]
            sel_lim = sel1[-1]
            sel = self.me[:-1] >= sel_lim  # self.IFMR.mBH_min
            self.mBH_min = sel_lim  # export this to make counting BHs easier

            # calculate total mass we want to eject
            MBH = Mr[sel].sum() * (1.0 - self.BH_ret_dyn)
            # print("total mass we want to eject: " + str(MBH))

            natal_ejecta = 0.0
            if self.natal_kicks:
                fb_interp = interp1d(
                    self.fb_grid[0],
                    self.fb_grid[1],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(0.0, 1.0),
                )
                for i in range(len(Mr)):
                    # skip the bin if its empty
                    if Nr[i] < self.Nmin:
                        continue
                    else:
                        # get the mean mass
                        mr = Mr[i] / Nr[i]
                        # only eject the BHs
                        if mr < sel_lim:
                            continue
                        else:
                            # print("mr = " + str(mr))
                            fb = fb_interp(mr)
                            # print("fb = " + str(fb))

                            if fb == 1.0:
                                continue
                            else:
                                retention = self.get_retention(fb, self.vesc)
                                # keep track of how much we eject
                                natal_ejecta += Mr[i] * (1 - retention)
                                # eject the mass
                                Mr[i] *= retention
                                Nr[i] *= retention

            # adjust by the amount we've already ejected
            MBH -= natal_ejecta

            Mr, Nr = self._eject_BH_dyn(Mr, Nr, M_eject=MBH)

        return Ns, alphas, Ms, Nr, Mr, mes

    def _eject_BH_dyn(self, Mr, Nr, *, M_eject=None):
        '''Determine and remove an amount of BHs from the final BH mass bins

        M_eject is amount of total mass to remove. If not given, compute based
        on `BH_ret_dyn`.
        '''

        # Identify BH bins
        # TODO need a better way for identiyfing remnants all over the place
        BH_cut = self.me[:-1] >= self.mBH_min

        # calculate total mass we want to eject
        if M_eject is None:
            M_eject = Mr[BH_cut].sum() * (1.0 - self.BH_ret_dyn)

        i = self.nbin
        # Remove BH starting from Heavy to Light

        while M_eject != 0:
            i -= 1

            # Skip empty bins
            if Nr[i] < self.Nmin:
                continue

            # Removed entirety of this bin
            if Mr[i] < M_eject:
                M_eject -= Mr[i]
                Mr[i] = 0
                Nr[i] = 0
                continue

            # Remove required fraction of the last affected bin
            else:
                Mr[i] -= M_eject
                Nr[i] -= M_eject / (Mr[i] / Nr[i])
                break

        return Mr, Nr


    def evolve(self, *, include_t0=False):

        # ------------------------------------------------------------------
        # Initialize output arrays
        # ------------------------------------------------------------------

        self.alphas = self.alphas0

        self.Ns, self.Ms, self.ms = self.Ns0, self.Ms0, self.ms0

        self.Nr, self.Mr, self.mr = self.Nr0, self.Mr0, self.mr0

        self.mes = self.mes0

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

                # Extract values
                Ns, alphas, Ms, Nr, Mr, mes = self.extract_arrays(ti, sol.y)

                # Stack values onto output arrays
                self.alphas = np.stack((self.alphas, alphas), axis=0)

                self.Ns = np.stack((self.Ns, Ns), axis=0)
                self.Ms = np.stack((self.Ms, Ms), axis=0)
                self.ms = np.stack((self.ms, Ms / Ns), axis=0)

                self.Nr = np.stack((self.Nr, Nr), axis=0)
                self.Mr = np.stack((self.Mr, Mr), axis=0)
                mr = 0.5 * (self.me[1:] + self.me[0:-1])
                mr[Nr > 0] = Mr[Nr > 0] / Nr[Nr > 0]
                self.mr = np.stack((self.mr, mr), axis=0)

                self.mes = np.stack((self.mes, mes), axis=0)

        # ------------------------------------------------------------------
        # If desired, remove the initial 0th bin (probably not in `tout`)
        # ------------------------------------------------------------------

        if not include_t0:

            self.alphas = self.alphas[1:]

            self.Ns, self.Ms, self.ms = self.Ns[1:], self.Ms[1:], self.ms[1:]

            self.Nr, self.Mr, self.mr = self.Nr[1:], self.Mr[1:], self.mr[1:]

            self.mes = self.mes[1:]

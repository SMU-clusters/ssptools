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
                 FeH, natal_kicks=False, vesc=90):

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

        self.evolve()

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
        self.ms0 = A * self.Pk(alpha, 2, self.me[0:-1], self.me[1:]) / self.Ns0

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
        # Main function computing the various derivatives

        derivs_sev = self._derivs_sev(t, y)

        # Change here to stop divide by zero and to speed up,
        # only run derivs_esc if Ndot is not zero
        # Original implementation:
        # derivs_esc = self._derivs_esc(t, y)
        # New implementation: (see https://github.com/balbinot/ssptools/issues/3)
        if self.Ndot < 0:
            derivs_esc = self._derivs_esc(t, y)
        else:
            derivs_esc = np.zeros_like(derivs_sev)

        return derivs_sev + derivs_esc

    def _derivs_sev(self, t, y):
        self.nstep += 1

        nb = self.nbin
        Nj_dot_s, Nj_dot_r = np.zeros(nb), np.zeros(nb)
        Mj_dot_r = np.zeros(nb)

        # Apply only to bins affected by stellar evolution
        if t > self.tms_u[-1]:

            # Find out which bin we are
            isev = np.where(t > self.tms_u)[0][0]

            # bin edges of turn-off bin
            m1 = self.me[isev]
            mto = self.compute_mto(t)
            Nj = y[isev]

            # Avoid "hitting" the bin edge
            if mto > m1 and Nj > self.Nmin:
                # Two parameters defining the bin
                alphaj = y[nb + isev]

                # The constant
                Aj = Nj / self.Pk(alphaj, 1, m1, mto)

                # Get the number of turn-off stars per unit of mass from Aj and alphaj
                dNdm = Aj * mto ** alphaj
            else:
                dNdm = 0

            # Then find dNdt = dNdm * dmdt
            a = self._tms_constants
            dmdt = abs(
                (1.0 / (a[1] * a[2] * t)) * (np.log(t / a[0]) / a[1]) ** (1 / a[2] - 1)
            )
            dNdt = -dNdm * dmdt

            # Define derivatives, note that alphaj remains constant
            Nj_dot_s[isev] = dNdt

            # Find remnant mass and which bin they go
            mrem = self.IFMR.predict(mto)

            # Skip 0 mass remnants
            if mrem > 0:
                irem = np.where(mrem > self.me)[0][-1]
                frem = 1  # full retention for WD
                if mrem >= 1.36:
                    frem = self.NS_ret
                # print('mBH_min:', self.IFMR.mBH_min)
                if mrem >= self.IFMR.mBH_min:
                    frem = self.BH_ret_int
                Nj_dot_r[irem] = -dNdt * frem
                Mj_dot_r[irem] = -mrem * dNdt * frem

        self.niter += 1

        return np.r_[Nj_dot_s, np.zeros(nb), Nj_dot_r, Mj_dot_r]

    def _derivs_esc(self, t, y):
        nb = self.nbin
        md = self.md
        Ndot = self.Ndot

        Nj_dot_s, aj_dot_s = np.zeros(nb), np.zeros(nb)
        Nj_dot_r, Mj_dot_r = np.zeros(nb), np.zeros(nb)

        Ns = np.abs(y[0:nb])
        alphas = y[nb : 2 * nb]
        Nr = np.abs(y[2 * nb : 3 * nb])
        Mr = np.abs(y[3 * nb : 4 * nb])

        if t < self.tcc:
            N_sum = Ns.sum() + Nr.sum()
            Nj_dot_s += Ndot * Ns / N_sum
            sel = Nr > 0
            Nj_dot_r[sel] += Ndot * Nr[sel] / N_sum
            Mj_dot_r[sel] += (Ndot * Nr[sel] / N_sum) * (Mr[sel] / Nr[sel])
            return np.r_[Nj_dot_s, aj_dot_s, Nj_dot_r, Mj_dot_r]

        mr = 0.5 * (self.me[1:] + self.me[0:-1])
        c = Nr > 0
        mr[c] = Mr[c] / Nr[c]

        a1, a15, a2, a25 = alphas + 1, alphas + 1.5, alphas + 2, alphas + 2.5

        # Setup edges for stars accounting for mto
        mes = np.copy(self.me)

        if t > self.tms_u[-1]:
            isev = np.where(mes > self.compute_mto(t))[0][0] - 1
            mes[isev + 1] = self.compute_mto(t)

        m1 = mes[0:-1]
        m2 = mes[1:]

        P1 = self.Pk(alphas, 1, m1, m2)
        P15 = self.Pk(alphas, 1.5, m1, m2)
        As = Ns / P1

        c = (mr < self.md) & (m1 < m2)
        Is = Ns[c] * (1 - md ** (-0.5) * P15[c] / P1[c])
        # print('IS', Is)
        Ir = Nr[c] * (1 - np.sqrt(mr[c] / md))
        Jr = Mr[c] * (1 - np.sqrt(mr[c] / md))

        B = Ndot / sum(Is + Ir)

        Nj_dot_s[c] += B * Is
        aj_dot_s[c] += (
            B * ((m1[c] / md) ** 0.5 - (m2[c] / md) ** 0.5) / np.log(m2[c] / m1[c])
        )
        Nj_dot_r[c] += B * Ir
        Mj_dot_r[c] += B * Jr

        return np.r_[Nj_dot_s, aj_dot_s, Nj_dot_r, Mj_dot_r]

    def maxwellian(self, x, a):
        norm = np.sqrt(2 / np.pi)
        exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
        return norm * exponent / a ** 3

    def get_retention(self, fb, vesc):

        if fb == 1.0:
            return 1.0

        v_space = np.linspace(0, 1000, 1000)
        kick_spl_loop = UnivariateSpline(
            x=v_space,
            y=self.maxwellian(v_space, 265 * (1 - fb)),
            s=0,
            k=3,
        )
        retention = kick_spl_loop.integral(0, vesc)

        return retention

    def extract_arrays(self, t, y):
        nb = self.nbin
        # Extract total N, M and split in Ns and Ms
        Ns = y[0:nb]
        alphas = y[nb : 2 * nb]

        # Some special treatment to adjust edges to mto
        mes = np.copy(self.me)
        if t > self.tms_u[-1]:
            isev = np.where(self.me > self.compute_mto(t))[0][0] - 1
            mes[isev + 1] = self.compute_mto(t)

        As = Ns / self.Pk(alphas, 1, mes[0:-1], mes[1:])
        Ms = As * self.Pk(alphas, 2, mes[0:-1], mes[1:])

        Nr = y[2 * nb : 3 * nb].copy()
        Mr = y[3 * nb : 4 * nb].copy()

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

            i = nb
            # Remove BH starting from Heavy to Light

            while MBH != 0:
                i -= 1
                # print(i, Nr[i])
                if Nr[i] < self.Nmin:
                    continue
                # print('Mr[i], MBH', Mr[i], MBH)
                if Mr[i] < MBH:
                    MBH -= Mr[i]
                    Mr[i] = 0
                    Nr[i] = 0
                    continue
                mmr = Mr[i] / Nr[i]
                Mr[i] -= MBH
                Nr[i] -= MBH / mmr
                MBH = 0

        return Ns, alphas, Ms, Nr, Mr, mes

    def evolve(self):
        nb = self.nbin

        self.niter = 0

        # Initialise ODE solver
        y = np.r_[self.Ns0, self.alphas0, self.Nr0, self.Mr0]

        # Evolve
        i = 0

        sol = ode(self._derivs)
        sol.set_integrator("dopri5", max_step=1e12, atol=1e-5, rtol=1e-5)
        sol.set_initial_value(y, 0)

        iout = 0
        for i in range(len(self.t)):
            sol.integrate(self.t[i])
            # print(self.t[i])

            if self.t[i] >= self.tout[iout]:

                Ns, alphas, Ms, Nr, Mr, mes = self.extract_arrays(self.t[i], sol.y)
                if iout == 0:
                    self.Ns = [Ns]
                    self.alphas = [alphas]
                    self.Ms = [Ms]

                    self.Nr = [Nr]
                    self.Mr = [Mr]

                    self.ms = [Ms / Ns]
                    self.mr = np.copy(self.ms)  # avoid /0
                    self.mes = [mes]

                else:
                    self.Ns = np.vstack((self.Ns, Ns))
                    self.alphas = np.vstack((self.alphas, alphas))
                    self.Ms = np.vstack((self.Ms, Ms))

                    self.Nr = np.vstack((self.Nr, Nr))
                    self.Mr = np.vstack((self.Mr, Mr))

                    self.ms = np.vstack((self.ms, Ms / Ns))

                    mr = 0.5 * (self.me[1:] + self.me[0:-1])
                    c = Nr > 0
                    mr[c] = Mr[c] / Nr[c]
                    self.mr = np.vstack((self.mr, mr))
                    self.mes = np.vstack((self.mes, mes))
                iout += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Some integration settings
    Ndot = 0  # -20  # per Myr
    tcc = 0
    N = 2e5
    NS_ret = 0.1  # initial NS retention
    BH_ret_int = 1.0  # initial BH retention
    BH_ret_dyn = 0.5 / 100  # Dynamical BH retention
    FeH = -0.7  # Metallicity

    # tout = np.linspace(3e3, 3e3, 1)
    # tout = np.array([
    #    1, 2, 3, 4, 5, 6, 7, 8, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000,
    #    4000, 5000, 6000, 7000, 8000, 9000, 12000
    # ])
    tout = np.array([11000])
    # masses and slopes that define double power-law IMF
    m123 = [0.1, 0.5, 1.0, 100]
    a12 = [-0.5, -1.35, -2.5]

    nbin = [5, 5, 20]
    f = evolve_mf(
        m123, a12, nbin, tout, N, Ndot, tcc, NS_ret, BH_ret_int, BH_ret_dyn, FeH
    )

    plotmf = True
    plotm = False

    if plotmf:

        plt.ioff()
        plt.figure(1, figsize=(8, 8))
        plt.clf()

        print(" Nsteps = ", f.nstep)
        print(" Start plotting ...")
        for i in range(len(f.tout)):
            plt.clf()
            plt.axes([0.13, 0.13, 0.8, 0.8])
            plt.xlim(0.8e-1, 1.3e2)
            plt.ylim(3e-1, 3e7)
            plt.yscale("log")
            plt.xscale("log")

            plt.ylabel(r"$dN/dm\,{\rm [M}_\odot^{-1}{\rm ]}$")
            plt.xlabel(r"$m\,{\rm [M}_\odot{\rm ]}$")
            plt.title(r"$t = %5i\,{\rm Myr}$" % f.tout[i])
            plt.tick_params(axis="x", pad=10)
            print(" plot ", i, f.tout[i])
            cs = f.Ns[i] > 10 * f.Nmin
            cr = f.Nr[i] > 10 * f.Nmin

            dms = f.mes[i][1:] - f.mes[i][0:-1]
            dmr = f.me[1:] - f.me[0:-1]

            plt.plot(f.ms[i][cs], f.Ns[i][cs] / dms[cs], "go-")
            plt.plot(f.mr[i][cr], f.Mr[i][cr] / dmr[cr], "ko-")

            print(f.ms[i][cs])
            print(f.mto(f.tout[i]))

            for j in range(len(f.me)):
                plt.plot([f.me[j], f.me[j]], [1e-4, 1e9], "k--")
            mto = f.mto(f.tout[i])
            plt.plot([mto, mto], [1e-4, 1e9], "k-")
            plt.savefig("mf_%04d.png" % i)
            plt.show()
            plt.draw

            plt.pause(0.05)

    if plotm:
        plt.ioff()
        plt.clf()
        plt.axes([0.12, 0.12, 0.8, 0.8])
        #        plt.xscale('log')
        #        plt.yscale('log')
        plt.ylim(0.1, 2)
        #        plt.xlim(2,e4)
        plt.ylabel(r"$M(t)$")
        plt.xlabel(r"$t\,{\rm [Myr]}$")

        Nstot = np.zeros(len(f.tout))
        Mstot = np.zeros(len(f.tout))

        Nrtot = np.zeros(len(f.tout))
        Mrtot = np.zeros(len(f.tout))

        for i in range(len(f.tout)):
            Nstot[i] = sum(f.Ns[i])
            Mstot[i] = sum(f.Ms[i])

            Nrtot[i] = sum(f.Nr[i])
            Mrtot[i] = sum(f.Mr[i])

        mtot = (Mstot + Mrtot) / (Nstot + Nrtot)
        #        plt.plot(f.tout,Nstot+Nrtot,'bo-')
        plt.plot(f.tout, mtot, "b-")
        plt.savefig("mvt.png")
        plt.show()
#        frac = f.Nstar[-1]/f.Nstar0

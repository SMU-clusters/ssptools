#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections.abc
from collections import namedtuple

import numpy as np


mbin = namedtuple('mbin', ('lower', 'upper'))
rem_classes = namedtuple('rem_classes', ('WD', 'NS', 'BH'))
star_classes = namedtuple('star_classes', ('MS',) + rem_classes._fields)


@np.errstate(invalid='ignore')
def Pk(a, k, m1, m2):
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


def _divide_bin_sizes(N, Nsec):
    '''Split N into Nsec as equally as possible'''
    Neach, ext = divmod(N, Nsec)
    return ext * [Neach + 1] + (Nsec - ext) * [Neach]


class MassBins:
    '''
    Class constructing and holding the "blueprints" of the mass bins, both
    stellar and remnants, separated by types

    Able to, based on the IFMR and input params, setup the stellar and remnant
    mass bins, and create the "unpacked" `y` array which is meant to be used
    by ODE solvers, then, once passed to the ODEs, pack such a `y` array into
    the separate mass bins.

    Holds various helper functions for then dealing with mass bins, bin edges,
    determining bin indices, etc.

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

    ifmr : ifmr.IFMR
        Initial-final mass relation class, required to set remnant mass
        boundaries

    binning_method : {'default', 'split_log', 'split_linear'}, optional
        The spacing method to use when constructing the mass bins.
        The default method ('split_log') will logspace the mass bins between
        each break mass, restarting the spacing in each regime.
        'split_linear' will linearly space the bins between the break masses.
        Note that in both cases the spacing between the regimes will *not* be
        the same
    '''

    def __init__(self, m_break, a, nbins, N0, ifmr, *,
                 binning_method='default'):

        N_MS_breaks = len(m_break) - 1

        self.a, self.m_break, self.N0 = a, m_break, N0

        # ------------------------------------------------------------------
        # Unpack number of stellar bins
        # ------------------------------------------------------------------

        # Bins are specified in dict for each type
        if isinstance(nbins, collections.abc.Mapping):
            try:
                nbin_MS = nbins['MS']

            except KeyError as err:
                mssg = f"Missing required stellar type in `nbins`: {err}"
                raise KeyError(mssg)

        # Only the stellar bins are specified, implicitly
        else:
            nbin_MS = nbins

        # Single number divided equally between break masses
        if isinstance(nbin_MS, int):
            self._nbin_MS_each = _divide_bin_sizes(nbins, N_MS_breaks)

        # List of bins between each break mass
        else:
            self._nbin_MS_each = nbin_MS
            nbin_MS = np.sum(nbin_MS)

        # ------------------------------------------------------------------
        # Setup stellar mass bins (based entirely on edges)
        # ------------------------------------------------------------------

        # Equal-log-space between bins, started again at each break
        if binning_method in ('default', 'split_log', 'log_split'):

            # one-liner required for different Neach and no repeating breaks
            bin_sides = np.r_[tuple(
                np.geomspace(m_break[i], m_break[i + 1],
                             self._nbin_MS_each[i] + 1)[(i > 0):]
                for i in range(len(self._nbin_MS_each))
            )]

        elif binning_method in ('linear_split', 'split_linear'):

            bin_sides = np.r_[tuple(
                np.linspace(m_break[i], m_break[i + 1],
                            self._nbin_MS_each[i] + 1)[(i > 0):]
                for i in range(len(self._nbin_MS_each))
            )]

        # TODO unsure how to implement uniform spaces while still hitting breaks
        # elif binning_method in ('log', 'logged'):
        # elif binning_method in ('linear'):

        else:
            mssg = f"Unrecognized binning method '{binning_method}'"
            raise ValueError(binning_method)

        # Define a bin based on it's upper and lower bounds
        bins_MS = mbin(bin_sides[:-1], bin_sides[1:])

        # ------------------------------------------------------------------
        # Setup Remnant mass bins
        # ------------------------------------------------------------------

        # Nbins are specified in dict for each type, I guess log space them
        if isinstance(nbins, collections.abc.Mapping):

            if binning_method in ('default', 'split_log', 'log_split'):
                binfunc = np.geomspace
            else:
                binfunc = np.linspace

            # White Dwarfs

            nbin_WD = nbins['WD']

            WD_bl, WD_bu = ifmr.WD_mf
            WD_bl = m_break[0] if WD_bl < m_break[0] else WD_bl

            bin_sides = binfunc(WD_bl, WD_bu, nbin_WD)
            bins_WD = mbin(bin_sides[:-1], bin_sides[1:])

            # Black Holes

            nbin_BH = nbins['BH']

            BH_bl, BH_bu = ifmr.BH_mf
            BH_bu = m_break[-1] if BH_bu > m_break[-1] else BH_bu

            bin_sides = binfunc(BH_bl, BH_bu, nbin_BH)
            bins_BH = mbin(bin_sides[:-1], bin_sides[1:])

            # Neutron Stars

            nbin_NS = nbins.get('NS', 1)

            if nbin_NS != 1:
                mssg = "All NS have same mass, cannot have more than 1 bin"
                raise ValueError(mssg)

            # arbitrarily small width around 1.4
            bins_NS = mbin(*(np.array(ifmr.NS_mf) + [-0.01, 0.01]))

        # Divide out the bins as if they were cut out from the MS bins
        else:

            # White Dwarfs

            WD_mask = (bins_MS.lower <= ifmr.WD_mf.upper)

            nbin_WD = WD_mask.sum()

            bins_WD = mbin(bins_MS.lower[WD_mask].copy(),
                           bins_MS.upper[WD_mask].copy())
            bins_WD.upper[-1] = ifmr.WD_mf.upper

            # Black Holes

            BH_mask = (bins_MS.upper > ifmr.BH_mf.lower)

            nbin_BH = BH_mask.sum()

            bins_BH = mbin(bins_MS.lower[BH_mask].copy(),
                           bins_MS.upper[BH_mask].copy())
            bins_BH.upper[0] = ifmr.BH_mf.lower

            # Neutron Stars

            NS_mask = (bins_MS.lower < 1.4) & (1.4 < bins_MS.upper)

            nbin_NS = NS_mask.sum()  # Always = 1

            # Doesn't really matter, but keep the full bin width here
            bins_NS = mbin(bins_MS.lower[NS_mask].copy(),
                           bins_MS.upper[NS_mask].copy())

        # ------------------------------------------------------------------
        # Setup the "blueprint" for the packed values
        # The blueprint is an array of integers reflecting the slices in the
        # packed `y` representing each component, to be used in `np.split(y)`.
        # Splitting includes the last given index to the end, so the final
        # nbin_BH should not be included in `_blueprint`, but is part of the
        # total size of y
        # ------------------------------------------------------------------

        self._blueprint = np.cumsum([nbin_MS, nbin_MS,  # Ms, alphas
                                     nbin_WD, nbin_NS, nbin_BH,  # N rem
                                     nbin_WD, nbin_NS])  # , nbin_BH])  # M rem

        self._ysize = self._blueprint[-1] + nbin_BH

        # ------------------------------------------------------------------
        # Save collections of useful attributes
        # ------------------------------------------------------------------

        self.nbin = star_classes(MS=nbin_MS, WD=nbin_WD, NS=nbin_NS, BH=nbin_BH)
        self.nbin_tot = np.r_[self.nbin].sum()

        self.bins = star_classes(MS=bins_MS, WD=bins_WD, NS=bins_NS, BH=bins_BH)

    def initial_values(self, *, packed=True):
        '''return packed array of initial values

        if packed=False, will instead return the initial number and total
        masses in each clasee (not that thats super useful)
        '''

        # TODO this is where we restrict to 3comp IMF,
        #   but should be generalized to any number of comps really

        # ------------------------------------------------------------------
        # Compute normalization factors A_j
        # ------------------------------------------------------------------

        a, mb = self.a, self.m_break

        A3 = (
            Pk(a[2], 1, mb[2], mb[3])
            + (mb[1] ** (a[1] - a[0]) * Pk(a[0], 1, mb[0], mb[1]))
            + (mb[2] ** (a[2] - a[1]) * Pk(a[1], 1, mb[1], mb[2]))
        ) ** (-1)

        A2 = A3 * mb[2] ** (a[2] - a[1])
        A1 = A2 * mb[1] ** (a[1] - a[0])

        A = self.N0 * np.repeat([A1, A2, A3], self._nbin_MS_each)

        # ------------------------------------------------------------------
        # Set the initial Nj and mj for all bins (stars and remnants)
        # ------------------------------------------------------------------

        # Expand array of IMF slopes to all mass bins
        alpha = np.repeat(a, self._nbin_MS_each)

        # Set initial star bins based on IMF
        Ns = A * Pk(alpha, 1, *self.bins.MS)

        # Set all initial remnant bins to zero
        Nwd, Mwd = np.zeros(self.nbin.WD), np.zeros(self.nbin.WD)
        Nns, Mns = np.zeros(self.nbin.NS), np.zeros(self.nbin.NS)
        Nbh, Mbh = np.zeros(self.nbin.BH), np.zeros(self.nbin.BH)

        if packed:
            return self.pack_values(Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh)

        else:
            # this way so wouldn't have to recompute Ms from typical unpacked
            Ms = A * Pk(alpha, 2, *self.bins.MS)

            N = star_classes(MS=Ns, WD=Nwd, NS=Nns, BH=Nbh)
            M = star_classes(MS=Ms, WD=Mwd, NS=Mns, BH=Mbh)

            return N, M

    def blanks(self, value=0., extra_dims=None, *, packed=True, **kwargs):
        '''return arrays of correct shape for y or y' all set to zeros (or
        something else). Meant for use as like initial derivatives and stuff

        shape will be (*extra_dims, y size)
        '''

        shape = (*([] if extra_dims is None else extra_dims), self._ysize)

        if value == 0.:
            full = np.zeros(shape=shape)  # Much faster
        elif value == 'empty':
            full = np.empty(shape=shape)
        else:
            full = np.full(shape=shape, fill_value=value)

        return full if packed else self.unpack_values(full, **kwargs)

    def pack_values(self, Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh):
        '''Put a bunch of arrays into the correct packed format for y or y'
        Mostly a convenience function for packing derivatives, allowing to
        supply by keyword things out of order or from various lists using *
        '''
        return np.r_[Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh]

    def unpack_values(self, y, *, grouped_rem=False):
        '''Unpack a given y into the various arrays'''
        Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh = np.split(y, self._blueprint,
                                                           axis=-1)

        if grouped_rem:
            Nrem = rem_classes(WD=Nwd, NS=Nns, BH=Nbh)
            Mrem = rem_classes(WD=Mwd, NS=Mns, BH=Mbh)
            return Ns, alpha, Nrem, Mrem

        else:
            return Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh

    def determine_index(self, mass, massbins, *, allow_overflow=False):
        '''determine the mbin index in the a given mass falls into in
        a `mbin` type set of mass bins

        mass bins are left-inclusive, i.e. [lower, upper)

        currently only support one mass at a time, sorry
        '''

        # If a string labelling the class, get from stored bins
        # Otherwise assume it's already a massbins class
        if massbins in star_classes._fields:
            massbins = getattr(self.bins, massbins)

        # Since mass bins always increasing, can look at only the lower bound
        try:
            ind = np.flatnonzero(massbins.lower <= mass)[-1]
        except IndexError:
            mssg = f"mass {mass} is below lowest bound in {massbins}"
            raise ValueError(mssg)

        # If this is the last bin, maybe check if its actually overflowing
        if ind >= massbins.upper.size - 1:
            if (massbins.upper[-1] <= mass) and allow_overflow is False:
                mssg = f"mass {mass} is above highest bound in {massbins}"
                raise ValueError(mssg)

        return ind

    def turned_off_bins(self, mto):
        '''return a version of the MS bins with the upper bounds of the turn
        off mass bin adjusted to mto, which is required sometimes to basically
        spoof the fact that, in that bin, the mass above this mto will have
        actually been entirely converted to remnants. But, because of binning,
        we can't simulate that intra-bins and so need to functionally
        approximate it for use in some functions, where the mean mass of the
        bin would actually be lowered.
        '''

        low, up = self.bins.MS.lower.copy(), self.bins.MS.upper.copy()

        # Find the bin containing mto and adjust its upper bound only
        try:
            isev = self.determine_index(mto, 'MS')
            up[isev] = mto

        # If the mto is above or below the min/max bounds, just ignore it
        except ValueError:
            pass

        return mbin(low, up)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections.abc
from collections import namedtuple

import numpy as np


mbin = namedtuple('mbin', ('lower', 'upper'))
rem_classes = namedtuple('rem_classes', ('WD', 'NS', 'BH'))
star_classes = namedtuple('star_classes', ('MS',) + rem_classes._fields)


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Initial Mass Functions
# --------------------------------------------------------------------------

class PowerLawIMF:
    '''Representation of a N-component power law Initial Mass Function (IMF).

    Based on a set of power law exponent slopes and break masses, constructs a
    stellar initial mass function (IMF) with useful methods for computing the
    expected total number or mass of stars of a given individual mass, both
    for continuous mass distribution and sets of mass bins.

    Parameters
    ----------
    m_break : list of N+1 float
        IMF break-masses (including outer bounds) defining the mass ranges
        covered by each component of this IMF.

    a : list of N float
        IMF power law slopes defining the exponent (e.g. :math:`m^{a_i}`) of
        each component of this IMF.

    N0 : int, optional
        Total initial number of stars, over all bins. Used to determine the
        normalization factor. This value acts as the default when `N` is not
        passed to various methods.

    ext : int or str, optional
        Controls the behaviour of the IMF outside of the mass interval (defined
        by the edges of `m_break`).

        * if ext=0 or ‘extrapolate’, extrapolate the nearest component.
        * if ext=1 or ‘zeros’, return 0 (default).
        * if ext=2 or ‘raise’, raise a ValueError.

        If ext=0, the IMF will still be normalized to N0 within the mass bounds.
    '''

    def __repr__(self):
        return f"PowerLawIMF(m_break={self.mb}, a={self.a}, N0={self.N0})"

    @property
    def mmean(self):
        '''The overall mean individual stellar mass of this IMF'''
        return self.Mtot / self.N0

    @property
    def Mtot(self):
        '''Total mass of system under this IMF (assuming `self.N0` stars).'''
        from scipy.integrate import quad
        return quad(self.M, self.mb[0], self.mb[-1])[0]

    @classmethod
    def from_M0(cls, m_break, a, M0, *, ext='zeros'):
        '''Initialize an IMF with the N0 required to have a total mass of M0.'''

        imf = cls(m_break=m_break, a=a, N0=1, ext='zeros')

        N0 = M0 / imf.mmean
        imf.N0 = N0

        return imf

    def __init__(self, m_break, a, N0=1, *, ext='zeros'):

        Nc = len(a)
        mb = m_break

        if (Nmb := len(mb)) != Nc + 1:
            mssg = f"`m_break` must have size len(a) + 1 ({Nc + 1}), not {Nmb}"
            raise ValueError(mssg)
        elif Nmb < 2:
            mssg = "`m_break` must be at least size 2 (upper and lower bounds)"
            raise ValueError(mssg)

        if not np.all(np.diff(mb) > 0):
            mssg = "All break masses must be in increasing order"
            raise ValueError(mssg)

        self.mb = np.asarray(mb)
        self.a = np.asarray(a)
        self.N0 = N0
        self.Ncomp = Nc

        self._A_comps = np.empty(Nc)

        # Compute the final A_N normalization component
        # A_N^{-1} = \sum_{i=1}^{N} P(a_i) \prod_{j=i+1}^{N} m_{j}^{a_j-a_{j-1}}
        self._A_comps[Nc - 1] = (
            np.sum([
                Pk(a[i], 1, mb[i], mb[i + 1]) * np.prod([
                    mb[j]**(a[j] - a[j - 1])
                    for j in range(i + 1, Nc)
                ])
                for i in range(0, Nc)
            ])
        )**(-1)

        # Compute all the following A_i normalization components
        for i in range(self.Ncomp - 1, 0, -1):
            self._A_comps[i - 1] = self._A_comps[i] * mb[i]**(a[i] - a[i - 1])

        match ext:
            case 0 | 'ext' | 'extrapolate':
                self._ext = 0
            case 1 | 'zeros':
                self._ext = 1
            case 2 | 'raise':
                self._ext = 2
            case _:
                raise ValueError("ext must be one of ('ext', 'zeros', 'raise')")

    def __call__(self, mass, *, N=None):
        '''Return the number of stars at a given mass for this IMF, N(m)'''

        mass = np.float64(mass)  # mostly to avoid warnings from scipy.integrate

        N = N if N is not None else self.N0

        if self._ext == 0:

            if self.Ncomp == 1:
                bounds = [True, ] * self._A_comps.size

            else:
                # Don't cut on the outermost lower and upper bounds
                bounds = [
                    (mass <= self.mb[1]),
                    *((lw_bnd <= mass) & (mass <= up_bnd)
                      for lw_bnd, up_bnd in zip(self.mb[1:-2], self.mb[2:-1])),
                    (self.mb[-2] <= mass)
                ]

        else:
            bounds = [(lw_bnd <= mass) & (mass <= up_bnd)
                      for lw_bnd, up_bnd in zip(self.mb[:-1], self.mb[1:])]

        default = np.nan if self._ext == 2 else 0

        vals = (self._A_comps * mass[..., np.newaxis]**self.a).T

        out = N * np.select(bounds, vals, default=default)

        if self._ext == 2 and (~np.isfinite(out)).any():
            mssg = f"mass outside bounds ({self.mb[0]}, {self.mb[-1]})"
            raise ValueError(mssg)

        else:
            return out

    def N(self, m, *, N=None):
        '''Return the number of stars at a given mass for this IMF, N(m)'''
        return self(m, N=N)

    def M(self, m, *, N=None):
        '''Return the total mass of stars at a given mass for this IMF, M(m)'''
        return m * self(m, N=N)

    def binned_eval(self, bins, *, N=None):
        '''Evaluate this imf within a given set of mass bins.

        Computes the mass and numbers of stars dictated by this IMF within a
        set of given mass bins (i.e. bin edges) by evaluating the form of the
        IMF applicable to the area covered by each bin.

        The mean mass in each bin can then be computed as :math:`m=M/N`.

        Parameters
        ----------
        bins : mbin
            The mass bins (defining the upper and lower bounds of each bin) to
            evaluate the IMF over. These do not necessarily need to align with
            any IMF break masses.

        N : int, optional
            Total initial number of stars, over all bins. Used to determine the
            normalization factor. By default, uses the `N0` parameter set during
            creation of the class.

        Returns
        -------
        binned_N : np.ndarray[Nbins]
            The total number of stars in each bin. Normalized such that
            `binned_N.sum() = N`.

        binned_M : np.ndarray[Nbins]
            The total mass of stars in each bin.

        binned_alpha : np.ndarray[Nbins]
            The power law exponent corresponding to each mass bin.
        '''

        N = N if N is not None else self.N0

        # TODO broken when a bin falls on either side of a break mass... ouf

        if self._ext == 0:

            if self.Ncomp == 1:
                bin_masks = [True, ] * self._A_comps.size

            else:

                # Don't cut on the outermost lower and upper bounds
                bin_masks = [
                    (bins.upper <= self.mb[1]),
                    *((lw_bnd <= bins.lower) & (bins.upper <= up_bnd)
                      for lw_bnd, up_bnd in zip(self.mb[1:-2], self.mb[2:-1])),
                    (self.mb[-2] <= bins.upper)
                ]

        else:
            bin_masks = [(lw_bnd <= bins.lower) & (bins.upper <= up_bnd)
                         for lw_bnd, up_bnd in zip(self.mb[:-1], self.mb[1:])]

        default = np.nan if self._ext == 2 else 0

        A = N * np.select(bin_masks, self._A_comps, default=default)

        if self._ext == 2 and (~np.isfinite(A)).any():
            mssg = f"mass bins outside bounds ({self.mb[0]}, {self.mb[-1]})"
            raise ValueError(mssg)

        alpha = np.select(bin_masks, self.a, default=default)

        return A * Pk(alpha, 1, *bins), A * Pk(alpha, 2, *bins), alpha

    def binned_N(self, bins, *, N=None):
        '''Return the total number of stars within given mass bins.'''
        return self.binned_eval(bins=bins, N=N)[0]

    def binned_M(self, bins, *, N=None):
        '''Return the total mass of stars within given mass bins.'''
        return self.binned_eval(bins=bins, N=N)[1]


# --------------------------------------------------------------------------
# Mass Bins
# --------------------------------------------------------------------------


class MassBins:
    '''Representation and handling of mass bins, for both stars and remnants

    A class handling the construction, holding and handling of mass bins.
    Both stars and remnants mass bins are handled separately.

    Based on input parameters defining the binning scheme (break masses,
    number of bins, etc.) sets up the spacing and boundaries of mass bins.
    Mass bins are defined using the `mbin` named tuple, with upper and lower
    bounds for each bin.

    The spacing of the bins, currently, can be either linear or log spaced,
    between the break masses.

    Remnant bins are defined based on the possible remnant max/min bounds,
    dictated by the IFMR, and the stellar mass bins.
    Remnant bins are stored separately for each remnant class, avoiding any
    possible overlap between different kinds of remnants.

    Methods are provided to help with interacting with these mass bins,
    including separating the different types, determining which bin a mass falls
    into, etc.

    This class provides a number of methods to help and facilitate the use of
    numerical ODE solvers, such as `scipy.integrate.ode` in `EvolvedMF`, with
    the mass bins.
    Such solvers are based on a single array of `y` numerical values (such that
    :math:$y'(t)=f(t,y)$), which must be dissected within the derivative
    functions into their constituent values.
    In the case of the derivatives in `EvolvedMF`, this `y` array is made up of
    8 parameters (arrays); Ns, the number of stars in each stellar mass bin,
    alphas, the mass function slopes in each stellar mass bin, Nwd, Nns, Nbh,
    the number of remnants in each remnant mass bin, Mwd, Mns, Mbh, the total
    mass of remnants in each remnant mass bin. Each of these constituent arrays
    have the size of the corresponding number of mass bins.
    Methods are available here to pack these distinct arrays into a single `y`
    or unpack a single `y` into the separate arrays.

    Parameters
    ----------
    m_break : list of N+1 float
        Mass bin break-masses (including outer bounds) used to define (if N>1)
        portions of the mass interval to be divided separately.

    nbins : int or list of N int or dict
        Number of stellar mass bins in each regime. If a single integer, the
        number mass bins between each break mass will be equal, otherwise
        the number of bins between each can be specified directly.
        Remnant bins will be created between IFMR bounds by copying bins from
        the created stellar mass bins.
        Also allowed is a dict of {'MS', 'WD', 'NS', 'BH'}, each containing
        a number of bins, where the remnant bins can be specified directly.

    imf : PowerLawIMF
        Initial mass function class, required for determining the initial
        star amounts in the `initial_values` method. Any break masses used in
        the IMF construction are *not* used to setup mass bins.

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

    Attributes
    ----------
    nbin : star_classes of int
        The number of bins in each stellar object type

    nbin_tot : int
        The total number of bins, including stars and remnants

    bins : star_classes of mbin of ndarray
        Mass bins of each stellar object type, defined using the `mbin` named
        tuple class, with upper and lower bounds. Each will have a size
        corresponding to `nbin`.

    '''

    def __init__(self, m_break, nbins, imf, ifmr, *,
                 binning_method='default'):

        self.imf = imf

        N_MS_breaks = len(m_break) - 1

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

            # TODO fails if m_break[0] < ifmr.WD_mf.upper
            bins_WD = mbin(bins_MS.lower[WD_mask].copy(),
                           bins_MS.upper[WD_mask].copy())
            bins_WD.upper[-1] = ifmr.WD_mf.upper

            # Black Holes

            BH_mask = (bins_MS.upper > ifmr.BH_mf.lower)

            nbin_BH = BH_mask.sum()

            bins_BH = mbin(bins_MS.lower[BH_mask].copy(),
                           bins_MS.upper[BH_mask].copy())
            bins_BH.lower[0] = ifmr.BH_mf.lower

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

        self._blueprint = np.cumsum([0, nbin_MS, nbin_MS,  # Ms, alphas
                                     nbin_WD, nbin_NS, nbin_BH,  # N rem
                                     nbin_WD, nbin_NS, nbin_BH])  # M rem

        self._ysize = self._blueprint[-1]

        # ------------------------------------------------------------------
        # Save collections of useful attributes
        # ------------------------------------------------------------------

        self.nbin = star_classes(MS=nbin_MS, WD=nbin_WD, NS=nbin_NS, BH=nbin_BH)
        self.nbin_tot = np.r_[self.nbin].sum()

        self.bins = star_classes(MS=bins_MS, WD=bins_WD, NS=bins_NS, BH=bins_BH)

    def initial_values(self, *, packed=True, N0=None):
        '''Return an array corresponding to `y` populated based on the IMF

        Based on the input IMF , create an empty `y` array and populate it
        with "initial" values. These values will populate the number of stars
        and the alpha slopes, and all remnants will be left as 0.

        Parameters
        ----------
        packed : bool, optional
            If True (default) will return a single array, otherwise will return
            two named tuples of `star_classes` with the masses and numbers in
            each stellar class. Note that this is *not* the same as an unpacked
            array.

        Return
        ------
        ndarray or 2-tuple of named `star_classes` tuples
        '''

        Ns, Ms, alpha = self.imf.binned_eval(self.bins.MS, N=N0)

        # Set all initial remnant bins to zero
        Nwd, Mwd = np.zeros(self.nbin.WD), np.zeros(self.nbin.WD)
        Nns, Mns = np.zeros(self.nbin.NS), np.zeros(self.nbin.NS)
        Nbh, Mbh = np.zeros(self.nbin.BH), np.zeros(self.nbin.BH)

        if packed:
            return self.pack_values(Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh)

        else:
            N = star_classes(MS=Ns, WD=Nwd, NS=Nns, BH=Nbh)
            M = star_classes(MS=Ms, WD=Mwd, NS=Mns, BH=Mbh)

            return N, M

    def blanks(self, value=0., extra_dims=None, *, packed=True, **kwargs):
        '''Returns an array corresponding to `y` with a single (blank) value

        Returns a new array (or set of arrays) corresponding to the `y`
        array, with all values set to a single given value, meant to represent
        initial empty arrays to be filled with values in the course of a
        derivative function.

        Any value is supported, but 0 or an empty array will be the fastest.
        If desired, the `y` array can unpacked into a list of arrays, such
        as is done by the `unpack_values` method.

        Multidimensional arrays can be returned by supplying extra dimensions.
        The final dimension will always be the required size (nbin_tot * 2).
        Be aware that, while `unpack_values` will support such a
        multidimensional `y` array as input, `pack_values` will not accept any
        multidimensional arrays.

        Parameters
        ----------
        value : float or 'empty', optional
            The value to fill the new blank array with. If 'empty', will use the
            `np.empty` initilization. The default (0) and 'empty' will be the
            most efficient, however any value is supported.

        extra_dims : None or list of int, optional
            An optional list of integer dimensions. If given, the shape of the
            final array will be (*extra_dims, nbin_tot * 2).

        packed : bool, optional
            If True (default) will return a single blank array, otherwise this
            will be passed to `unpack_values` first. All other kwargs will
            also be passed to this call.

        Returns
        -------
        ndarray or list of ndarray
            The final, blank, array with the given value
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
        '''Pack various arrays into a single `y` array

        Given 8 separate arrays, combine them into a single unified "packed"
        array, such as those required for input/output to numerical DE solvers.

        Parameters
        ----------
        Ns : ndarray
            Array containing values corresponding to the number of stars,
            placed first in the packed array. Must have size `nbin.MS`.

        alpha : ndarray
            Array containing values corresponding to stellar alphas
            placed second in the packed array. Must have size `nbin.MS`.

        Nwd : ndarray
            Array containing values corresponding to number of WD
            placed third in the packed array. Must have size `nbin.WD`.

        Nns : ndarray
            Array containing values corresponding to number of NS
            placed fourth in the packed array. Must have size `nbin.NS`.

        Nbh : ndarray
            Array containing values corresponding to number of BH
            placed fifth in the packed array. Must have size `nbin.BH`.

        Mwd : ndarray
            Array containing values corresponding to mass of WD
            placed sixth in the packed array. Must have size `nbin.WD`.

        Mns : ndarray
            Array containing values corresponding to mass of NS
            placed seventh in the packed array. Must have size `nbin.NS`.

        Mbh : ndarray
            Array containing values corresponding to mass of BH
            placed last in the packed array. Must have size `nbin.BH`.

        Returns
        -------
        y : ndarray
            Single "packed" array of values with total size nbin_tot * 2.

        See also
        --------
        unpack_values : The inverse of this method
        '''

        # Manually pack the arrays instead of simply concatenating
        # (i.e. `np.r_`), just to speed up the packing slightly

        inp_arrays = (Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh)
        out = self.blanks('empty')
        bp = self._blueprint
        for inp, i, j in zip(inp_arrays, bp[:-1], bp[1:]):
            out[i:j] = inp

        return out

    def unpack_values(self, y, *, grouped_rem=False):
        '''Unpack a given `y` array into the various expected arrays

        Taking in a single unified "packed" array, such as those passed into
        derivative functions from numerical DE solvers, splits the array into
        the 8 expected arrays (Ns, alpha, Nwd, Nns, Nbh, Mwd, Mns, Mbh) based
        on the number of bins in each object type, returning a list of all
        arrays.

        Should work for multidimensional `y` arrays, as long as the *final*
        dimension is the size of the total arrays (nbin_tot * 2).

        Parameters
        ----------
        y : ndarray
            Single "packed" array of values to unpack. Final (-1) dimension
            must have correct size (nbin_tot * 2).

        grouped_rem : bool, optional
            Whether to pack the arrays for the remnant numbers and masses
            (i.e. the last six returned arrays) into a single named tuple
            (`rem_classes`) instead of returning a full separated list of all
            arrays (the default).

        Returns
        -------
        list of ndarray
            A list of 8 arrays representing each sliced out section of the input
            array, with sizes based on the bin sizes of the relevant classes.
            If `grouped_rem` is True, the list will instead contain 4 elements,
            two arrays, exactly as usual, and 2 `rem_classes` named tuples each
            containing half of the 6 other arrays.

        See also
        --------
        pack_values : The inverse of this method
        '''

        # Manually unpack each array, rather than using `np.split`, based on
        # the blueprint, just to speed up the unpacking slightly.

        bp = self._blueprint
        out = [y[..., i:j] for i, j in zip(bp[:-1], bp[1:])]

        # If desired, group the remnants (M and N) into `rem_classes`
        if grouped_rem:
            Ns, alpha = out[0:2]
            Nrem = rem_classes(*out[2:5])
            Mrem = rem_classes(*out[5:8])
            return [Ns, alpha, Nrem, Mrem]

        else:
            return out

    def determine_index(self, mass, massbins, *, allow_overflow=False):
        '''Determine the index of the bin in `massbins` containing `mass`

        Given a set of massbins, determine the index of the bin which the given
        mass falls into.

        Note that mass bins are considered "left-inclusive" here,
        i.e. [lower, upper), therefore masses falling on the bin edges will
        return the lower index.

        Currently this function only supports one input mass at a time.

        Parameters
        ----------
        mass : float
            The mass from which to determine the location/index within the mass
            bins.

        massbins : mbin or {'MS', 'WD', 'NS', 'BH'}
            The massbins to determine the mass index within. Either a mass bin
            named tuple with upper and lower bound fields, or a two character
            string indicating the stellar type, which will pull from the
            massbins stored in this instance.
        '''
        # TODO support multiple mass at once, by correctly arranging mass shape

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
        '''Return a copy of the MS bins with upper bounds adjusted up to `mto`

        Returns a copy of the stellar (MS) mass bins where the upper bounds of
        one bin, that holding the given turn-off mass (`mto`), are adjusted to
        fall on the `mto`. This may be required at times to, basically, spoof
        the fact that all mass within a given bin but above `mto` may have been
        lost (e.g. turned into remnants), and thus shouldn't be considered in
        some cases for computing quantities such as the mean bin mass.

        Parameters
        ----------
        mto : float
            The "turn-off" mass used to locate and truncate the turn-off bin.

        Returns
        -------
        mbin
            Near identical copy of the MS mass bin tuple, with the turn-off bin
            adjusted.
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


from . import ifmr
from .masses import PowerLawIMF

import numpy as np


class StellarEvMassLoss:
    '''Compute the impacts of stellar evolution over time.

    This class models the evolution of the total mass, number, and mean mass
    of the stellar population, starting from a given IMF, over time, by
    computing the derivatives (rate of change over time) of these quantities.

    This class uses the same general algorithms for stellar evolution as the
    classes within `evolve_mf`, but accounts *only* for stellar evolution, and
    is based on continuous functions of time and mass, rather than relying
    on discrete mass bins.

    Parameters
    ----------
    imf : PowerLawIMF
        The initial mass function (IMF) of the stellar population.

    FeH : float
        Metallicity, in solar fraction [Fe/H].

    '''

    def __call__(self, t, y):
        '''Evaluate the stellar evolution derivatives at time t.

        Computes and returns the derivatives (rate of change over time) of
        the mass, number and mean mass of the system, at the given time, and
        with the given current values (within `y`) of these quantities.

        Derivatives are returned with mass units of Msun and time units of Myr.

        Parameters
        ----------
        t : float or ndarray of float
            The time (or Array[nt] of times) to evaluate the derivatives at.

        y : ndarray
            Array[3, nt] containing the total mass, number and mean mass of the
            system at times `t`. If `t` contains multiple times, the shape of
            `y` must match. If `y` is invalid, the mass and number derivatives
            will remain valid, but the mean mass will not be.

        Returns
        -------
        dMdt :float or ndarray of float
            The rate of change of the total mass of stars over time over time,
            in units of Msun/Myr

        dNdt :float or ndarray of float
            The rate of change of the total number of stars over time, in units
            of 1/Myr

        dmavgdt :float or ndarray of float
            The rate of change of the mean mass of stars over time, in units
            of Msun/Myr
        '''
        return self._compute_derivs(t, y, dNdt_only=False, dMdt_only=False)

    def dNdt(self, t):
        '''Compute only the total number derivative dN/dt'''
        fake_y = [1., 1., 1.]
        return self._compute_derivs(t, fake_y, dNdt_only=True, dMdt_only=False)

    def dMdt(self, t):
        '''Compute only the total mass derivative dM/dt'''
        fake_y = [1., 1., 1.]
        return self._compute_derivs(t, fake_y, dNdt_only=False, dMdt_only=True)

    def __init__(self, imf, FeH):

        # imf = masses.PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0)
        self.imf = imf
        self.FeH = FeH

        # TODO should probably allow passing kwargs to this (for luminous v).
        self.ifmr = ifmr.IFMR(FeH)

        # Get tms constants (using nearest metallicity in grid)
        mstogrid = np.loadtxt(ifmr.get_data("sevtables/msto.dat"))
        self.a = mstogrid[np.argmin(np.abs(mstogrid[:, 0] - FeH)), 1:]

        # Assume no stars made above IMF upper limit
        self.tlim = self.a[0] * np.exp(self.a[1] * imf.mb[-1]**self.a[2])

        self.rets = {'WD': 1., 'NS': 0.1, 'BH': 0.0}  # ignore BHs anyways

    def _compute_derivs(self, t, y, *, dNdt_only=False, dMdt_only=False):

        M, N, mmean = y

        dmdt = abs(
            (1.0 / (self.a[1] * self.a[2] * t))
            * (np.log(t / self.a[0]) / self.a[1]) ** (1 / self.a[2] - 1)
        )

        mto = (np.log(t / self.a[0]) / self.a[1]) ** (1 / self.a[2])
        dNdm = self.imf(mto)

        dNdt = -dNdm * dmdt

        if dNdt_only:
            return dNdt

        # Avoid going above upper IMF bound
        dMdt = np.where(
            t <= self.tlim,
            np.zeros_like(dNdt),
            # dMdt
            dNdt * mto
        )

        if dMdt_only:
            return dMdt

        # dmavg/dt = d/dt(Mast/Nast), use quotient rule to handle all cases
        dmavgdt = ((N * dMdt) - (M * dNdt)) / (N**2)

        return dMdt, dNdt, dmavgdt

    @classmethod
    def from_powerlaw(cls, m_breaks, a_slopes, N0, FeH):
        '''Construct class based on a power-law IMF breaks and slopes, directly.

        Alternative constructor to `StellarEvMassLoss` based on explicitly
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

        FeH : float
            Metallicity, in solar fraction [Fe/H].

        Returns
        -------
        StellarEvMassLoss
            The created evolved stellar evolution object, using the created IMF.
        '''
        imf = PowerLawIMF(m_break=m_breaks, a=a_slopes, N0=N0, ext='zeros')
        return cls(imf, FeH)


class LuminousEvMassLoss(StellarEvMassLoss):
    '''Compute the impacts of stellar evolution over time on all non-BH objects.

    This subclass simply extends StellarEvMassLoss to consider the changes in
    all "luminous" objects (i.e. all MS stars and non black hole remnants),
    rather than only the living stars.
    This class is used in the same manner as StellarEvMassLoss, but returns
    this combined rate in each case.

    Parameters
    ----------
    imf : PowerLawIMF
        The initial mass function (IMF) of the stellar population.

    FeH : float
        Metallicity, in solar fraction [Fe/H].

    '''

    def _compute_derivs(self, t, y, *, dNdt_only=False, dMdt_only=False):

        Mast, Nast, mmast = y

        dmdt = abs(
            (1.0 / (self.a[1] * self.a[2] * t))
            * (np.log(t / self.a[0]) / self.a[1]) ** (1 / self.a[2] - 1)
        )

        mto = (np.log(t / self.a[0]) / self.a[1]) ** (1 / self.a[2])
        dNdm = self.imf(mto)

        dNdt = -dNdm * dmdt

        mr = self.ifmr.predict(mto)
        cls_rem = np.asarray(self.ifmr.predict_type(mto))

        fr = np.select([cls_rem == 'WD', cls_rem == 'NS'],
                       [self.rets['WD'], self.rets['NS']], default=0.0)

        # dNast/dt = 0 when making WD/NS and dN/dt when making BHs
        dNastdt = dNdt * (1 - fr)

        if dNdt_only:
            return dNastdt

        dMdt = dNdt * (mto - (fr * mr))  # works cause rets[BH]=0

        # Avoid going above upper IMF bound
        dMastdt = np.where(
            t <= self.tlim,
            np.zeros_like(dMdt),
            dMdt
        )

        if dMdt_only:
            return dMastdt

        # dmavg/dt = d/dt(Mast/Nast), use quotient rule to handle all cases
        dmavgdt = ((Nast * dMastdt) - (Mast * dNastdt)) / (Nast**2)

        return dMastdt, dNastdt, dmavgdt

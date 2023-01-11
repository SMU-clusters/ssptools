#!/usr/bin/env python

import warnings

import pytest
import numpy as np

from ssptools import evolve_mf


# Mixture of `ssptools` and `GCfit` defaults for `evolve_mf`
DEFAULT_KWARGS = dict(
    m_breaks=[0.1, 0.5, 1.0, 100], a_slopes=[-0.5, -1.3, -2.5],
    nbins=[5, 5, 20], FeH=-1.00, tout=[12_000], Ndot=0.,
    N0=5e5, tcc=0.0, NS_ret=0.1, BH_ret_int=1.0, BH_ret_dyn=1.0,
    natal_kicks=False, vesc=90
)


class TestClassMethods:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emf = evolve_mf.evolve_mf(**DEFAULT_KWARGS)

    # ----------------------------------------------------------------------
    # Testing computation of t_ms lifetime
    # ----------------------------------------------------------------------

    def test_tms_values(self):
        mi = [0.5, 1.0, 100.0]

        tms = self.emf.compute_tms(mi)

        expected = np.array([8.10364433e+04, 5.72063993e+03, 1.80376582e+00])

        assert tms == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
    @pytest.mark.parametrize(
        'mi, expected',
        [
            (-1.0, np.nan),
            (0, np.inf),
            (1e-15, np.inf),
            (np.finfo('float64').max, 0.18799593)  # FeH=-1 asympt. at ~0.187995
        ],
        ids=['negative', 'zero', 'near-zero', 'near-inf']
    )
    def test_tms_bounds(self, mi, expected):

        tms = self.emf.compute_tms(mi)

        assert tms == pytest.approx(expected, nan_ok=True)

    def test_tms_sort(self):
        mi = np.sort(np.random.random(100) * 100)

        tms = self.emf.compute_tms(mi)

        assert np.all(tms[:-1] >= tms[1:])

    # ----------------------------------------------------------------------
    # Testing computation of m_to turnoff mass
    # ----------------------------------------------------------------------

    def test_mto_values(self):
        ti = np.array([8.10364433e+04, 5.72063993e+03, 1.80376582e+00])

        mto = self.emf.compute_mto(ti)

        expected = [0.5, 1.0, 100.0]

        assert mto == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
    @pytest.mark.parametrize(
        'ti, expected',
        [
            (-1.0, np.nan),
            # (0, 0),  # Don't test this cause the eqns not physical here anyway
            (emf._tms_constants[0] - 1e-15, np.nan),
            (emf._tms_constants[0], np.inf),
            (np.finfo('float64').max, 0.0)
        ],
        ids=['negative', 't < a0', 't = a0', 'near-inf']
    )
    def test_mto_bounds(self, ti, expected):

        mto = self.emf.compute_mto(ti)

        assert mto == pytest.approx(expected, nan_ok=True)

    def test_mto_sort(self):
        ti = np.sort(np.random.random(100) * 100) + self.emf._tms_constants[0]

        mto = self.emf.compute_mto(ti)

        assert np.all(mto[:-1] >= mto[1:])

    def test_mto_tms_inverse(self):
        mi = [0.5, 1.0, 100.0]

        mf = self.emf.compute_mto(self.emf.compute_tms(mi))

        assert mf == pytest.approx(mi)

    # ----------------------------------------------------------------------
    # Testing computation of P_k helper integral solution
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('k', [1., 1.5, 2.])
    @pytest.mark.parametrize('a', [-1., -0.5, 1.0])
    def test_Pk(self, a, k):
        from scipy.integrate import quad

        m1, m2 = 0.5, 1.0
        expected, err = quad(lambda m: m**(a + k - 1), m1, m2)

        Pk = self.emf._Pk(a=a, k=k, m1=m1, m2=m2)

        assert Pk == pytest.approx(expected, abs=err)

# come up with different combination of initial params
# test the final value of all output attributes, for all these initials

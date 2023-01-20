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


class TestHelperMethods:

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emf = evolve_mf.EvolvedMF(**DEFAULT_KWARGS)

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
    # mto function asymptotes at a0, then is imaginary to the left, except at 0

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


class TestBHEvolution:

    emf_kw = DEFAULT_KWARGS.copy() | {'tout': [0.]}

    @pytest.fixture()
    def Mi(self):
        return np.array([10., 10., 10.])

    @pytest.fixture()
    def Ni(self):
        return np.array([20., 10., 5.])

    # ----------------------------------------------------------------------
    # Test BH natal kick routines
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'vesc, expected',
        [
            (25., np.stack((np.array([0.002227, 0.002227, 0.002779]),
                            np.array([0.004454, 0.002227, 0.001389])))),
            (100., np.stack((np.array([0.136963, 0.136963, 0.169875]),
                             np.array([0.273926, 0.136963, 0.084937])))),
            (200., np.stack((np.array([0.966443, 0.966443, 1.175480]),
                             np.array([1.932887, 0.966443, 0.587740])))),
        ],
        ids=[f'vesc = {v}' for v in (25, 100, 200)]
    )
    def test_natal_kick_quantities(self, Mi, Ni, vesc, expected):

        # Results not actually reliant on this evolution, just need the object
        kw = self.emf_kw | {'vesc': vesc, 'natal_kicks': True}
        emf = evolve_mf.EvolvedMF(**kw)

        Mf, Nf, _ = emf._natal_kick_BH(Mi, Ni)

        assert np.stack((Mf, Nf)) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize(
        'vesc, expected',
        [
            (25., 29.992765),
            (100., 29.556198),
            (200., 26.891631),
        ],
        ids=[f'vesc = {v}' for v in (25, 100, 200)]
    )
    def test_natal_kick_total(self, Mi, Ni, vesc, expected):

        kw = self.emf_kw | {'vesc': vesc, 'natal_kicks': True}
        emf = evolve_mf.EvolvedMF(**kw)

        _, _, ejected = emf._natal_kick_BH(Mi, Ni)

        assert ejected == pytest.approx(expected, rel=1e-3)

    # ----------------------------------------------------------------------
    # Test BH dynamical ejection routines
    # ----------------------------------------------------------------------

    base_emf = evolve_mf.EvolvedMF(**emf_kw)

    @pytest.mark.parametrize(
        'M_eject, expected',
        [
            (0., np.stack((np.array([10., 10, 10]), np.array([20., 10, 5])))),
            (15., np.stack((np.array([10., 5, 0]), np.array([20., 5, 0])))),
            (30., np.stack((np.array([0., 0, 0]), np.array([0., 0, 0])))),
        ],
    )
    def test_dyn_ej_quantities(self, Mi, Ni, M_eject, expected):

        Mf, Nf = self.base_emf._dyn_eject_BH(Mi, Ni, M_eject=M_eject)

        assert np.stack((Mf, Nf)) == pytest.approx(expected)

    def test_dyn_ej_overflow(self, Mi, Ni):

        M_eject_overflow = Mi.sum() + 0.1

        with pytest.raises(ValueError):
            self.base_emf._dyn_eject_BH(Mi, Ni, M_eject=M_eject_overflow)


# class TestMassEvolution:
class TestDerivatives:

    # TODO tests for the _set_imf and _evolve functions, the only ones which
    #   modify in-place, so I haven't touched them here

    emf_kw = DEFAULT_KWARGS.copy() | {'tout': [0.], 'nbins': [1, 1, 2]}

    # ----------------------------------------------------------------------
    # Derivative routines
    # ----------------------------------------------------------------------

    @pytest.fixture()
    def y(self):
        nb = sum(self.emf_kw['nbins'])
        return np.array([1000] * nb + [1] * nb + [10] * nb + [2] * nb)

    @pytest.mark.parametrize(
        't, expected',
        [
            (100, np.array([0., 0., -10.16057518, 0.,
                            0., 0., 0., 0.,
                            0., 0., 10.16057518, 0.,
                            0., 0., 11.28587621, 0.])),
            (12000, np.array([0., -0.07375997, 0., 0.,
                              0., 0., 0., 0.,
                              0., 0.07375997, 0., 0.,
                              0., 0.04325484, 0., 0.])),
        ],
    )
    def test_sev(self, y, t, expected):

        emf = evolve_mf.EvolvedMF(**self.emf_kw)

        ydot = emf._derivs_sev(t, y)
        assert ydot == pytest.approx(expected)

    @pytest.mark.parametrize(
        't, expected',
        [
            (100, np.array([7.588256e-01, 3.191671e-01, -9.576813e-01,
                            -1.015838e+01, -3.566531e-04, -6.205286e-04,
                            -1.096328e-03, -3.734391e-03, 9.519368e-03,
                            9.519368e-03, 9.519368e-03, 9.519368e-03,
                            1.903873e-03, 1.903873e-03, 1.903873e-03,
                            1.903873e-03])),
            (12000, np.array([6.814154e-01, 3.696330e-01, -1.907299e+00,
                              -9.177942e+00, -3.202699e-04, -5.273448e-04,
                              -1.187120e-03, -3.916009e-03, 8.548268e-03,
                              8.548268e-03, 8.548268e-03, 8.548268e-03,
                              1.709653e-03, 1.709653e-03, 1.709653e-03,
                              1.709653e-03])),
        ],
    )
    def test_esc(self, y, t, expected):

        kw = self.emf_kw | {'Ndot': -10}
        emf = evolve_mf.EvolvedMF(**kw)

        ydot = emf._derivs_esc(t, y)
        assert ydot == pytest.approx(expected)

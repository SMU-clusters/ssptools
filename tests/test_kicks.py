#!/usr/bin/env python

import pytest
import numpy as np
import scipy.special
import scipy.integrate as integ

from ssptools import kicks


class TestRetentionAlgorithms:

    masses = np.linspace(0.01, 150, 50)

    # def test_F12_fallback_frac(self, ):

    # Maxwellian not yet mass-vectorized, so need to check individual masses
    @pytest.mark.parametrize('vdisp', [200., 265., 300.])
    @pytest.mark.parametrize('FeH', [-2., -0.5, 0.3])
    @pytest.mark.parametrize('vesc', [25., 100., 200.])
    @pytest.mark.parametrize('m', [0.01, 0.5, 1.0, 10.0, 100., 150.])
    @pytest.mark.parametrize('SNe_method', ['rapid', 'delayed', 'none'])
    def test_maxwellian_retention_frac(self, m, vesc, FeH, vdisp, SNe_method):

        def maxwellian(x):
            a = vdisp * (1 - fb)
            exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
            return np.sqrt(2 / np.pi) * exponent / a ** 3

        if SNe_method == 'rapid':
            # TODO need test for fb func
            fb = kicks._F12_fallback_frac(FeH, SNe_method='rapid')(m)

        elif SNe_method == 'delayed':
            fb = kicks._F12_fallback_frac(FeH, SNe_method='delayed')(m)

        elif SNe_method == 'none':
            fb = 0.0

        expected = 1.0 if fb >= 1.0 else integ.quad(maxwellian, 0, vesc)[0]

        fret = kicks._maxwellian_retention_frac(m, vesc, FeH, vdisp,
                                                SNe_method=SNe_method)

        assert fret == pytest.approx(expected, rel=5e-3)

    @pytest.mark.filterwarnings("ignore:.*:RuntimeWarning  ")
    @pytest.mark.parametrize('scale', [-20, 0, 20, 150])
    @pytest.mark.parametrize('slope', [-1, 0, 0.5, 1, 10])
    def test_sigmoid_retention_frac(self, slope, scale):

        fret = kicks._sigmoid_retention_frac(self.masses, slope, scale)

        expected = scipy.special.erf(np.exp(slope * (self.masses - scale)))

        assert fret == pytest.approx(expected)


class TestNatalKicks:

    # TODO these aren't really representative m_BH (m=M/N=.5,1,2)
    @pytest.fixture()
    def Mi(self):
        return np.array([10., 10., 10.])

    @pytest.fixture()
    def Ni(self):
        return np.array([20., 10., 5.])

    # ----------------------------------------------------------------------
    # Test Maxwellian / Fryer+2012 natal kick algorithms
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'FeH, vesc, expected',
        [
            (-1., 25., np.stack((np.array([0.002227, 0.002227, 0.002810]),
                                 np.array([0.004454, 0.002227, 0.001405])))),
            (-1., 100., np.stack((np.array([0.136963, 0.136963, 0.171672]),
                                  np.array([0.273926, 0.136963, 0.085836])))),
            (-1., 200., np.stack((np.array([0.966443, 0.966443, 1.186696]),
                                  np.array([1.932887, 0.966443, 0.593348])))),
            (0.3, 25., np.stack((np.array([0.002227, 0.002227, 0.002727]),
                                 np.array([0.004454, 0.002227, 0.001363])))),
            (0.3, 100., np.stack((np.array([0.136963, 0.136963, 0.166788]),
                                  np.array([0.273926, 0.136963, 0.08339])))),
            (0.3, 200., np.stack((np.array([0.966443, 0.966443, 1.156175]),
                                  np.array([1.932887, 0.966443, 0.578087]))))
        ],
    )
    def test_F12_kicks_quantities(self, Mi, Ni, FeH, vesc, expected):

        Mf, Nf, _ = kicks.natal_kicks(Mi, Ni, method='F12', FeH=FeH, vesc=vesc)

        assert np.stack((Mf, Nf)) == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize(
        'FeH, vesc, expected',
        [
            (-1., 25., 29.992735),
            (-1., 100., 29.554400),
            (-1., 200., 26.880415),
            (0.3, 25., 29.992818),
            (0.3, 100., 29.559284),
            (0.3, 200., 26.910936),
        ],
    )
    def test_F12_kicks_total(self, Mi, Ni, FeH, vesc, expected):

        _, _, ejec = kicks.natal_kicks(Mi, Ni, method='F12', FeH=FeH, vesc=vesc)

        assert ejec == pytest.approx(expected, rel=1e-3)

    # ----------------------------------------------------------------------
    # Test sigmoid natal kick algorithm
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'slope, scale, expected',
        [
            (0.5, 0., np.stack((np.array([9.306121, 9.802805, 9.998790]),
                                np.array([18.612243, 9.802805, 4.999395])))),
            (0.5, 20., np.stack((np.array([0.000657, 0.000844, 0.001392]),
                                 np.array([0.001315, 0.000844, 0.000696])))),
            (0.5, 50., np.stack((np.array([0.0, 0.0, 0.0]),
                                 np.array([0.0, 0.0, 0.0])))),
            (10, 0., np.stack((np.array([10., 10., 10.]),
                               np.array([20., 10., 5.])))),
            (10, 20., np.stack((np.array([0.0, 0.0, 0.0]),
                                np.array([0.0, 0.0, 0.0])))),
            (10, 50., np.stack((np.array([0.0, 0.0, 0.0]),
                                np.array([0.0, 0.0, 0.0])))),
        ],
    )
    def test_sigmoid_kicks_quantities(self, Mi, Ni, slope, scale, expected):

        Mf, Nf, _ = kicks.natal_kicks(Mi, Ni, method='sigmoid',
                                      slope=slope, scale=scale)

        assert np.stack((Mf, Nf)) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize(
        'slope, scale, expected',
        [
            (0.5, 0, 0.892281),
            (0.5, 20, 29.997105),
            (0.5, 50, 29.999999),
            (10, 0, 0.0),
            (10, 20, 30.0),
            (10, 50, 30.0)
        ],
    )
    def test_sigmoid_kicks_total(self, Mi, Ni, slope, scale, expected):

        _, _, ejec = kicks.natal_kicks(Mi, Ni, method='sigmoid',
                                       slope=slope, scale=scale)

        assert ejec == pytest.approx(expected, rel=1e-3)

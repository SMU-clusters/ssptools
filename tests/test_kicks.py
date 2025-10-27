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
    # @pytest.mark.parametrize('m', [0.01, 0.5, 1.0, 10.0, 100., 150.])
    @pytest.mark.parametrize('SNe_method', ['rapid', 'delayed', 'ns',
                                            'neutrino', 'none'])
    def test_maxwellian_retention_frac(self, vesc, FeH, vdisp, SNe_method):

        def maxwellian(x, fbi):
            a = vdisp * (1 - fbi)
            exponent = (x ** 2) * np.exp((-1 * (x ** 2)) / (2 * (a ** 2)))
            return np.sqrt(2 / np.pi) * exponent / a ** 3

        if SNe_method == 'rapid':
            # TODO need test for fb func
            fb = kicks._F12_fallback_frac(FeH, SNe_method='rapid')(self.masses)

        elif SNe_method == 'delayed':
            fb = kicks._F12_fallback_frac(FeH, SNe_method='delayed')(self.masses)

        elif SNe_method == 'ns':
            fb = 1 - kicks._NS_reduced_kick(m_NS=1.4)(self.masses)

        elif SNe_method == 'neutrino':
            fb = 1 - kicks._neutrino_driven_kick(m_eff=7.0)(self.masses)

        elif SNe_method == 'none':
            fb = np.zeros_like(self.masses)

        # Might as well integrate everything to check CDF
        expected = np.full_like(self.masses, 1.0)
        for i, fbi, in enumerate(fb):
            if fbi < 1.0:
                expected[i] = integ.quad(maxwellian, 0, vesc, args=(fbi,))[0]

        fret = kicks._maxwellian_retention_frac(self.masses, vesc, FeH, vdisp,
                                                SNe_method=SNe_method)

        assert fret == pytest.approx(expected, rel=5e-3)

    @pytest.mark.filterwarnings("ignore:.*:RuntimeWarning  ")
    @pytest.mark.parametrize('scale', [-20, 0, 20, 150])
    @pytest.mark.parametrize('slope', [-1, 0, 0.5, 1, 10])
    def test_sigmoid_retention_frac(self, slope, scale):

        fret = kicks._sigmoid_retention_frac(self.masses, slope, scale)

        expected = scipy.special.erf(np.exp(slope * (self.masses - scale)))

        assert fret == pytest.approx(expected)

    @pytest.mark.parametrize('scale', [-20, 0, 20, 150])
    @pytest.mark.parametrize('slope', [-1, 0, 0.5, 1, 10])
    def test_tanh_retention_frac(self, slope, scale):

        fret = kicks._tanh_retention_frac(self.masses, slope, scale)

        expected = 0.5 * (np.tanh(slope * (self.masses - scale)) + 1)

        assert fret == pytest.approx(expected)

    @pytest.mark.parametrize('value', [0.0, 0.25, 0.5, 1.0])
    def test_flat_retention_frac(self, value):

        fret = kicks._flat_fallback_frac(value)(self.masses)

        assert fret == pytest.approx(value)


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

        _, _, ks = kicks.natal_kicks(Mi, Ni, method='F12', FeH=FeH, vesc=vesc)

        assert ks.total_kicked == pytest.approx(expected, rel=1e-3)

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

        _, _, ks = kicks.natal_kicks(Mi, Ni, method='sigmoid',
                                     slope=slope, scale=scale)

        assert ks.total_kicked == pytest.approx(expected, rel=1e-3)

    # ----------------------------------------------------------------------
    # Test tanh natal kick algorithm
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'slope, scale, expected',
        [
            (0.5, 0., np.stack((np.array([6.224593, 7.310585, 8.807970]),
                                np.array([12.449186, 7.310585, 4.403985])))),
            (0.5, 20., np.stack((np.array([3.3982e-08, 5.6027e-08, 1.5229e-07]),
                                np.array([6.7965e-08, 5.603e-08, 7.615e-08])))),
            (0.5, 50., np.stack((np.array([0., 0., 0.]),
                                np.array([0., 0., 0.])))),
            (10., 0., np.stack((np.array([9.999546, 9.999999, 10.]),
                                np.array([19.999092, 9.999999, 5.])))),
            (10., 20., np.stack((np.array([0., 0., 0.]),
                                np.array([0., 0., 0.])))),
            (10., 50., np.stack((np.array([0., 0., 0.]),
                                np.array([0., 0., 0.]))))
        ],
    )
    def test_tanh_kicks_quantities(self, Mi, Ni, slope, scale, expected):

        Mf, Nf, _ = kicks.natal_kicks(Mi, Ni, method='tanh',
                                      slope=slope, scale=scale)

        assert np.stack((Mf, Nf)) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize(
        'slope, scale, expected',
        [
            (0.5, 0, 7.65685),
            (0.5, 20, 29.99999),
            (0.5, 50, 30.0),
            (10, 0, 0.000454),
            (10, 20, 30.0),
            (10, 50, 30.0)
        ],
    )
    def test_tanh_kicks_total(self, Mi, Ni, slope, scale, expected):

        _, _, ks = kicks.natal_kicks(Mi, Ni, method='tanh',
                                     slope=slope, scale=scale)

        assert ks.total_kicked == pytest.approx(expected, rel=1e-3)

    # ----------------------------------------------------------------------
    # Test natal kick algorithm when explicitly specifying f_kick
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'slope, f_kick, expected',
        [
            (0.5, 0.1, np.stack((np.array([8.421878, 8.979450, 9.598670]),
                                 np.array([16.843757, 8.979450, 4.799335])))),
            (0.5, 0.5, np.stack((np.array([3.40972, 4.603431, 6.986839]),
                                 np.array([6.819459, 4.603431, 3.493419])))),
            (0.5, 0.9, np.stack((np.array([0.464521, 0.743462, 1.792015]),
                                 np.array([0.929043, 0.743462, 0.896007])))),
            (10., 0.1, np.stack((np.array([7.000194, 9.999805, 10.]),
                                 np.array([14.00038, 9.999805, 5.])))),
            (10., 0.5, np.stack((np.array([4.5389e-04, 4.999546, 9.999999]),
                                 np.array([9.0779e-04, 4.999546, 4.999999])))),
            (10., 0.9, np.stack((np.array([4.0134e-13, 8.8335e-09, 2.999999]),
                                 np.array([8.0269e-13, 8.8335e-09, 1.500000]))))
        ],
    )
    def test_expl_fkick_kicks_quantities(self, Mi, Ni, slope, f_kick, expected):

        Mf, Nf, _ = kicks.natal_kicks(Mi, Ni, method='tanh', f_kick=f_kick,
                                      slope=slope)

        assert np.stack((Mf, Nf)) == pytest.approx(expected, abs=1e-3)

    @pytest.mark.parametrize('slope', [0.5, 10.])
    @pytest.mark.parametrize('f_kick', [0.1, 0.5, 0.9])
    def test_expl_fkick_kicks_total(self, Mi, Ni, slope, f_kick):

        expected = f_kick * Mi.sum()

        _, _, ks = kicks.natal_kicks(Mi, Ni, method='tanh', f_kick=f_kick,
                                     slope=slope)

        assert ks.total_kicked == pytest.approx(expected, rel=1e-3)


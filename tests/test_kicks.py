#!/usr/bin/env python

import dataclasses

import pytest
import numpy as np
import scipy.special
import scipy.integrate as integ

from ssptools import kicks, evolve_mf, masses, ifmr


DEFAULT_M_BREAK = [0.1, 0.5, 1.0, 100]

DEFAULT_IMF = masses.PowerLawIMF(
    m_break=DEFAULT_M_BREAK, a=[-0.5, -1.3, -2.5], N0=5e5
)

DEFAULT_IFMR = ifmr.IFMR(FeH=-1)


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

    @pytest.mark.parametrize(
        'mi, rem_type, expected_frem',
        [
            (0.5, 'WD', 1.0),
            (1.0, 'WD', 1.0),
            (1.4, 'NS', 0.1),
            (5.0, 'BH', 0.01006),
            (10.0, 'BH', 0.01006,),
            (20.0, 'BH', 0.01229),
            (50.0, 'BH', 1.0),
            (100.0, 'BH', 1.0)
        ],
    )
    def test_frem(self, mi, rem_type, expected_frem):

        @dataclasses.dataclass
        class SpoofEMF:
            NS_ret = 0.1
            natal_kicks = True
            _kick_kw = dict(FeH=-1, method='fryer2012', vesc=90)

        frem = evolve_mf.EvolvedMF._frem(SpoofEMF(), mi, rem_type)

        assert frem == pytest.approx(expected_frem, abs=0.0001)

    @pytest.mark.parametrize(
        'f_target, expected_scale',
        [
            (0.001, 16.24579),
            (0.1, 21.66143),
            (0.5, 35.07088),
            (0.8, 62.84709),
            (0.99, 97.70552)
        ],
    )
    def test_determine_params(self, f_target, expected_scale):

        _, scale = kicks._determine_kick_params(
            method='tanh',
            f_target=f_target,
            IMF=DEFAULT_IMF,
            IFMR=DEFAULT_IFMR,
            slope=0.5
        )

        assert scale == pytest.approx(expected_scale, abs=0.0001)


class TestKickVelocities:

    @pytest.mark.parametrize('vdisp', [200., 265., 300.])
    def test_kick_velocity_distribution(self, vdisp):

        # test fb methods elsewhere, here scale=vdisp, no m-dep
        masses = np.ones(1_000_000)
        SNe_method = 'none'
        FeH = -1

        seed = 42

        rng_test = np.random.default_rng(seed=seed)

        vel_test = kicks.maxwellian_kick_v(
            m=masses, FeH=FeH, vdisp=vdisp,
            rng=rng_test, SNe_method=SNe_method
        )

        # Check mean

        exp_mean = 2 * vdisp * np.sqrt(2 / np.pi)
        assert np.mean(vel_test) == pytest.approx(exp_mean, rel=1e-3)

        # Check variance

        exp_var = vdisp**2 * ((3 * np.pi) - 8) / np.pi
        assert np.var(vel_test) == pytest.approx(exp_var, rel=1e-3)

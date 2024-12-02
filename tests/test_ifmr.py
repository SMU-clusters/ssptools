#!/usr/bin/env python

import pytest
import numpy as np

from ssptools import ifmr


metals = [-3.00, -2.50, -2.00, -1.00, -0.50, 0.40, 1.00]


class TestMetallicity:
    '''Tests about IFMR metallicity and bound checks'''

    WD_metals = [-2.00, -2.00, -2.00, -1.00, -0.50, -0.50, -0.50]
    BH_metals = [-2.50, -2.50, -2.00, -1.00, -0.50, 0.40, 0.40]

    @pytest.mark.parametrize('FeH', metals)
    def test_stored_FeH(self, FeH):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.FeH == FeH

    # WD FeH is no longer stored
    # @pytest.mark.parametrize("FeH, expected", zip(metals, WD_metals))
    # def test_WD_FeH(self, FeH, expected):
    #     IFMR = ifmr.IFMR(FeH)
    #     assert IFMR.FeH_WD == expected

    @pytest.mark.parametrize("FeH, expected", zip(metals, BH_metals))
    def test_BH_FeH(self, FeH, expected):
        assert ifmr._check_IFMR_FeH_bounds(FeH) == expected


class TestPredictors:

    # ----------------------------------------------------------------------
    # Base predictors
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('exponent', [-0.5, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.01, 0.05, 0.1])
    @pytest.mark.parametrize('scale', [0, 0.5, 0.9])
    def test_powerlaw_predictor(self, exponent, slope, scale):

        m_low, m_up = 1.0, 8.

        mi = np.linspace(m_low, m_up)

        expected = (slope * mi**exponent) + scale

        pred = ifmr._powerlaw_predictor(exponent, slope, scale, m_low, m_up)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize(
        'exponent, slope, scale, m_low, m_up',
        [
            (1, 0., 0., 1, 10),  # Always zero
            (1, 1, 1., 5, 2),  # Invalid bounds
            (1, -1, 5, 1, 10),  # root between bounds
            (2, -1, 10, 1, 10),  # root between bounds
            (2, 0.5, 0.0, 20, 100),  # goes above 1-1 line
        ]
    )
    def test_powerlaw_invalids(self, exponent, slope, scale, m_low, m_up):
        with pytest.raises(ValueError):
            ifmr._powerlaw_predictor(exponent, slope, scale, m_low, m_up)

    @pytest.mark.parametrize('exponents', [[-0.5, 1.0], [2.5, 2.0]])
    @pytest.mark.parametrize('slopes', [[0.01, 0.05], [0.05, 0.01]])
    @pytest.mark.parametrize('scales', [[0., 0.5], [0.5, 0.9]])
    def test_broken_powerlaw_predictor(self, exponents, slopes, scales):

        mb = [1.0, 5.0, 10.]

        m1 = np.linspace(mb[0], mb[1])
        m2 = np.linspace(mb[1], mb[2])[1:]  # algo prioritizes left lines

        expected = np.r_[((slopes[0] * m1**exponents[0]) + scales[0]),
                         ((slopes[1] * m2**exponents[1]) + scales[1])]

        pred = ifmr._broken_powerlaw_predictor(exponents, slopes, scales, mb)

        assert pred(np.r_[m1, m2]) == pytest.approx(expected)

    # ----------------------------------------------------------------------
    # White Dwarf predictors
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[0.6219, 0.7859, 1.015, 1.099, 1.1844],
                     [0.6219, 0.7859, 1.015, 1.099, 1.1844],
                     [0.6219, 0.7859, 1.015, 1.099, 1.1844],
                     [0.6176, 0.7219, 0.990, 1.088, 1.1445],
                     [0.6023, 0.6914, 0.894, 1.061, 1.1446],
                     [0.6023, 0.6914, 0.894, 1.061, 1.1446],
                     [0.6023, 0.6914, 0.894, 1.061, 1.1446]])
    )
    def test_MIST18_WD_predictor(self, FeH, expected):

        mi = np.linspace(1., 5., 5)

        pred, *_ = ifmr._MIST18_WD_predictor(FeH)

        assert pred(mi) == pytest.approx(expected, abs=1e-3)

    # ----------------------------------------------------------------------
    # Black Hole predictors
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[5.17365, 16.038835, 19.48679, 29.98945, 48.27369],
                     [5.17365, 16.038835, 19.48679, 29.98945, 48.27369],
                     [4.89109, 15.7271, 19.15276, 29.835705, 44.249],
                     [8.59656, 10.809675, 17.3977, 27.903175, 39.67126],
                     [11.26324, 8.875225, 14.38747, 22.30988, 27.53808],
                     [8.95447, 19.901125, 25.89775, 29.409525, 31.06735],
                     [8.95447, 19.901125, 25.89775, 29.409525, 31.06735]])
    )
    def test_Ba20_r_BH_predictor(self, FeH, expected):

        mi = np.linspace(5., 100., 5)

        pred, *_ = ifmr._Ba20_r_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize('exponent', [.1, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.01, 0.05, 0.1])
    @pytest.mark.parametrize('scale', [0, 0.5, 0.9])
    def test_powerlaw_BH_predictor(self, exponent, slope, scale):

        m_low, m_up = 1.0, 10.

        mi = np.linspace(m_low, m_up)

        expected = (slope * mi**exponent) + scale

        pred, *_ = ifmr._powerlaw_BH_predictor(exponent, slope, scale, m_low)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize('exponents', [[-0.5, 1.0], [2.5, 2.0]])
    @pytest.mark.parametrize('slopes', [[0.01, 0.05], [0.05, 0.01]])
    @pytest.mark.parametrize('scales', [[0., 0.5], [0.5, 0.9]])
    def test_brokenpl_BH_predictor(self, exponents, slopes, scales):

        mb = [1.0, 5.0, 10.]

        m1 = np.linspace(mb[0], mb[1])
        m2 = np.linspace(mb[1], mb[2])[1:]  # algo prioritizes left lines

        expected = np.r_[((slopes[0] * m1**exponents[0]) + scales[0]),
                         ((slopes[1] * m2**exponents[1]) + scales[1])]

        pred, *_ = ifmr._brokenpl_BH_predictor(exponents, slopes, scales, mb)

        assert pred(np.r_[m1, m2]) == pytest.approx(expected)


class TestBounds:
    '''Tests about IFMR remnant mass bounds'''

    # ----------------------------------------------------------------------
    # MIST 2018 WDs
    # ----------------------------------------------------------------------

    WD_mi = [(0.0, 5.318525), (0.0, 5.318525), (0.0, 5.318525),
             (0.0, 5.47216), (0.0, 5.941481), (0.0, 5.941481), (0.0, 5.941481)]

    WD_mf = [(0.0, 1.228837), (0.0, 1.228837), (0.0, 1.228837),
             (0.0, 1.228496), (0.0, 1.256412), (0.0, 1.256412), (0.0, 1.256412)]

    @pytest.mark.parametrize("FeH, expected_mi", zip(metals, WD_mi))
    def test_MIST18_WD_mi(self, FeH, expected_mi):
        _, mi, _ = ifmr._MIST18_WD_predictor(FeH)
        assert mi == pytest.approx(expected_mi)

    @pytest.mark.parametrize("FeH, expected_mf", zip(metals, WD_mf))
    def test_MIST18_WD_mf(self, FeH, expected_mf):
        _, _, mf = ifmr._MIST18_WD_predictor(FeH)
        assert mf == pytest.approx(expected_mf)

    # ----------------------------------------------------------------------
    # Fryer 2012 BHs
    # ----------------------------------------------------------------------

    # Initial and final mass bounds for all remnants, for each FeH in `metals`
    BH_mi = [(19.7, 250.0), (19.7, 250.0), (19.9, 250.0), (20.6, 250.0),
             (21.1, 250.0), (25.5, 250.0), (25.5, 250.0)]

    BH_mf = [(5.59413, np.inf), (5.59413, np.inf), (5.50497, np.inf),
             (5.51648, np.inf), (5.55204, np.inf), (5.50509, np.inf),
             (5.50509, np.inf)]

    @pytest.mark.parametrize("FeH, expected_mi", zip(metals, BH_mi))
    def test_Ba20_r_BH_mi(self, FeH, expected_mi):
        _, mi, _ = ifmr._Ba20_r_BH_predictor(FeH)
        assert mi == pytest.approx(expected_mi)

    @pytest.mark.parametrize("FeH, expected_mf", zip(metals, BH_mf))
    def test_Ba20_r_BH_mf(self, FeH, expected_mf):
        _, _, mf = ifmr._Ba20_r_BH_predictor(FeH)
        assert mf == pytest.approx(expected_mf)

    # ----------------------------------------------------------------------
    # Power law BHs
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('exponent', [.1, 1.0, 1.5])
    @pytest.mark.parametrize('slope', [0.01, 0.05, 0.1])
    @pytest.mark.parametrize('scale', [0, 0.5, 0.9])
    @pytest.mark.parametrize('ml', [1, 5, 15])
    def test_powerlaw_BH_mi(self, exponent, slope, scale, ml):
        _, mi, _ = ifmr._powerlaw_BH_predictor(exponent, slope, scale, ml)
        assert mi == pytest.approx((ml, np.inf))

    @pytest.mark.parametrize('exponent', [.1, 1.0, 1.5])
    @pytest.mark.parametrize('slope', [0.01, 0.05, 0.1])
    @pytest.mark.parametrize('scale', [0, 0.5, 0.9])
    def test_powerlaw_BH_mf(self, exponent, slope, scale):

        ml = 1.0

        _, _, mf = ifmr._powerlaw_BH_predictor(exponent, slope, scale, ml)

        expected = (slope * ml**exponent) + scale

        assert mf == pytest.approx((expected, np.inf), abs=1e-3)

    @pytest.mark.parametrize(
        'exponents, slopes, scales, mb',
        [
            ([1., 1.], [0.5, .01], [0.1, 0.], [2.1, 5., 10.]),
            ([1., -1.5], [.01, .01], [0.1, 0.1], [2.5, 3., 100.]),
            ([1., -1.5], [.01, -0.5], [0.1, 1.], [10., 20., 30.])
        ]
    )
    def test_brokenpl_BH_mi(self, exponents, slopes, scales, mb):

        _, mi, _ = ifmr._brokenpl_BH_predictor(exponents, slopes, scales, mb)

        assert mi == pytest.approx((mb[0], mb[-1]))

    @pytest.mark.parametrize(
        'exponents, slopes, scales, expected_mf',
        [
            ([1., 1.], [0., 1.], [1., 0.], (1., 10.)),
            ([1., 1.1], [1., .7], [0., 0.], (1., 8.812)),
            ([1., -1.5], [.01, -0.5], [0.1, 1.], (0.11, 0.9842))
        ]
    )
    def test_brokenpl_BH_mf(self, exponents, slopes, scales, expected_mf):

        mb = [1.0, 5.0, 10.]

        _, _, mf = ifmr._brokenpl_BH_predictor(exponents, slopes, scales, mb)

        assert mf == pytest.approx(expected_mf, abs=1e-3)


class TestPredictions:
    '''Tests about default IFMR remnant mass and type predictions'''

    IFMR = ifmr.IFMR(FeH=-1.00)

    ini_masses = np.geomspace(0.2, 100, 25)

    rem_masses = np.array([
        0.296115, 0.403242, 0.485982, 0.536884, 0.560416,  # White Dwarfs
        0.576329, 0.608092, 0.654573, 0.686416, 0.731103,
        0.897034, 1.05756359, 1.10926924,
        1.4, 1.4, 1.4, 1.4, 1.4,  # Neutron Stars
        16.560239, 9.382873, 15.226190, 14.71036, 20.419381,  # Black Holes
        28.369477, 39.67126
    ])

    rem_types = (['WD'] * 13) + (['NS'] * 5) + (['BH'] * 7)

    def test_predict_mass(self):
        mf = self.IFMR.predict(self.ini_masses)
        assert mf == pytest.approx(self.rem_masses, abs=1e-5)

    def test_predict_type(self):
        mt = self.IFMR.predict_type(self.ini_masses)
        assert mt == self.rem_types

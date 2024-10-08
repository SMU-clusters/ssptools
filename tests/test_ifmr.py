#!/usr/bin/env python

import pytest
import numpy as np

from ssptools import ifmr


metals = [-3.00, -2.50, -2.00, -1.00, -0.50, 0.50, 1.00]


class TestMetallicity:
    '''Tests about IFMR metallicity and bound checks'''

    WD_metals = [-2.00, -2.00, -2.00, -1.00, -0.50, -0.50, -0.50]
    BH_metals = [-2.50, -2.50, -2.00, -1.00, -0.50, 0.50, 0.50]

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
        assert ifmr._check_F12_BH_FeH_bounds(FeH) == expected


class TestPredictors:

    # ----------------------------------------------------------------------
    # Base predictors
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('exponent', [-2.0, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.5, 2, 10])
    @pytest.mark.parametrize('scale', [0, 5, 10])
    def test_powerlaw_predictor(self, exponent, slope, scale):

        m_low, m_up = 1.0, 10.

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
        ]
    )
    def test_powerlaw_invalids(self, exponent, slope, scale, m_low, m_up):
        with pytest.raises(ValueError):
            ifmr._powerlaw_predictor(exponent, slope, scale, m_low, m_up)

    @pytest.mark.parametrize('exponents', [[-2.0, 1.0], [3.0, 2.0]])
    @pytest.mark.parametrize('slopes', [[0.5, 2.], [2., 10.]])
    @pytest.mark.parametrize('scales', [[0., 5.0], [5.0, 10.0]])
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
        zip(metals, [[ 5.498, 15.6225, 19.069, 29.8365, 49.942],
                     [ 5.498, 15.6225, 19.069, 29.8365, 49.942],
                     [ 4.813, 15.9855, 19.078, 29.966, 45.441],
                     [ 7.266, 12.2005, 17.896, 28.841, 40.681],
                     [ 10.589,  9.8829, 15.54, 24.5635, 34.873],
                     [ 8.585, 12.98655, 28.132, 31.017, 33.916],
                     [ 8.585, 12.98655, 28.132, 31.017, 33.916]])
    )
    def test_F12_BH_predictor(self, FeH, expected):

        mi = np.linspace(5., 100., 5)

        pred, *_ = ifmr._F12_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize('exponent', [-2.0, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.5, 2, 10])
    @pytest.mark.parametrize('scale', [0, 5, 10])
    def test_powerlaw_BH_predictor(self, exponent, slope, scale):

        m_low, m_up = 1.0, 10.

        mi = np.linspace(m_low, m_up)

        expected = (slope * mi**exponent) + scale

        pred, *_ = ifmr._powerlaw_BH_predictor(exponent, slope, scale, m_low)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize('exponents', [[-2.0, 1.0], [3.0, 2.0]])
    @pytest.mark.parametrize('slopes', [[0.5, 2.], [2., 10.]])
    @pytest.mark.parametrize('scales', [[0., 5.0], [5.0, 10.0]])
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
    BH_mi = [(19.7, 150.0), (19.7, 150.0), (19.8, 150.0), (20.5, 150.0),
             (20.9, 150.0), (24.3, 150.0), (24.3, 150.0)]

    BH_mf = [(5.4977, np.inf), (5.4977, np.inf), (5.5952, np.inf),
             (5.503, np.inf), (5.5386, np.inf), (5.5081, np.inf),
             (5.5081, np.inf)]

    @pytest.mark.parametrize("FeH, expected_mi", zip(metals, BH_mi))
    def test_F12_BH_mi(self, FeH, expected_mi):
        _, mi, _ = ifmr._F12_BH_predictor(FeH)
        assert mi == pytest.approx(expected_mi)

    @pytest.mark.parametrize("FeH, expected_mf", zip(metals, BH_mf))
    def test_F12_BH_mf(self, FeH, expected_mf):
        _, _, mf = ifmr._F12_BH_predictor(FeH)
        assert mf == pytest.approx(expected_mf)

    # ----------------------------------------------------------------------
    # Power law BHs
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('exponent', [-2.0, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.5, 2, 10])
    @pytest.mark.parametrize('scale', [0, 5, 10])
    @pytest.mark.parametrize('ml', [0.1, 5, 10])
    def test_powerlaw_BH_mi(self, exponent, slope, scale, ml):
        _, mi, _ = ifmr._powerlaw_BH_predictor(exponent, slope, scale, ml)
        assert mi == pytest.approx((ml, np.inf))

    @pytest.mark.parametrize('exponent', [-2.0, 1.0, 2.0])
    @pytest.mark.parametrize('slope', [0.5, 2, 10])
    @pytest.mark.parametrize('scale', [0, 5, 10])
    def test_powerlaw_BH_mf(self, exponent, slope, scale):

        ml = 1.0

        _, _, mf = ifmr._powerlaw_BH_predictor(exponent, slope, scale, ml)

        expected = (slope * ml**exponent) + scale

        assert mf == pytest.approx((expected, np.inf), abs=1e-3)

    @pytest.mark.parametrize(
        'exponents, slopes, scales, mb',
        [
            ([1., 1.], [0., 1.], [1., 0.], [1., 5., 10.]),
            ([1., 2.], [1., 1.], [1., 1.], [2., 3., 100.]),
            ([1., 2], [1., -0.5], [1., 51.], [10., 20., 30.])
        ]
    )
    def test_brokenpl_BH_mi(self, exponents, slopes, scales, mb):

        _, mi, _ = ifmr._brokenpl_BH_predictor(exponents, slopes, scales, mb)

        assert mi == pytest.approx((mb[0], mb[-1]))

    @pytest.mark.parametrize(
        'exponents, slopes, scales, expected_mf',
        [
            ([1., 1.], [0., 1.], [1., 0.], (1., 10.)),
            ([1., 2.], [1., 1.], [1., 1.], (2., 101.)),
            ([1., 2], [1., -0.5], [1., 51.], (1., 38.5))
        ]
    )
    def test_brokenpl_BH_mf(self, exponents, slopes, scales, expected_mf):

        mb = [1.0, 5.0, 10.]

        _, _, mf = ifmr._brokenpl_BH_predictor(exponents, slopes, scales, mb)

        assert mf == pytest.approx(expected_mf, abs=1e-3)


class TestPredictions:
    '''Tests about IFMR remnant mass and type predictions'''

    IFMR = ifmr.IFMR(FeH=-1.00)

    ini_masses = np.geomspace(0.2, 100, 25)

    rem_masses = np.array([
        0.296115, 0.403242, 0.485982, 0.536884, 0.560416,  # White Dwarfs
        0.576329, 0.608092, 0.654573, 0.686416, 0.731103,
        0.897034, 1.05756359, 1.10926924,
        1.4, 1.4, 1.4, 1.4, 1.4,  # Neutron Stars
        17.413826, 10.313890, 19.816023, 15.147086, 21.023453,  # Black Holes
        29.188165, 40.681
    ])

    rem_types = (['WD'] * 13) + (['NS'] * 5) + (['BH'] * 7)

    def test_predict_mass(self):
        mf = self.IFMR.predict(self.ini_masses)
        assert mf == pytest.approx(self.rem_masses, abs=1e-5)

    def test_predict_type(self):
        mt = self.IFMR.predict_type(self.ini_masses)
        assert mt == self.rem_types

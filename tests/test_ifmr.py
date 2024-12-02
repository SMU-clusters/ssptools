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
        zip(metals, [[16.65645, 13.684405, 22.59955, 31.588195, 48.27369],
                     [16.65645, 13.684405, 22.59955, 31.588195, 48.27369],
                     [16.69869, 14.109375, 22.34194, 31.470415, 44.249],
                     [15.51396, 13.158875, 20.38378, 29.53558, 39.67126],
                     [14.66664, 13.642055, 16.65486, 23.562575, 27.53808],
                     [15.41827, 14.269495, 27.51753, 29.70356, 31.06735],
                     [15.41827, 14.269495, 27.51753, 29.70356, 31.06735]])
    )
    def test_Ba20_r_BH_predictor(self, FeH, expected):

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._Ba20_r_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[5.95672, 13.684405, 22.59955, 31.588195, 48.27369],
                     [5.95672, 13.684405, 22.59955, 31.588195, 48.27369],
                     [5.75891, 14.109375, 22.34194, 31.470415, 44.249],
                     [4.7856, 13.163405, 20.38378, 29.535580, 39.67126],
                     [4.22753, 13.642055, 16.65486, 23.562575, 27.53808],
                     [2.9936, 17.61182, 27.51753, 29.703560, 31.06735],
                     [2.9936, 17.61182, 27.51753, 29.703560, 31.06735]])
    )
    def test_Ba20_d_BH_predictor(self, FeH, expected):

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._Ba20_d_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[18.177670, 14.576215, 24.12029, 33.84658, 51.19238],
                     [18.177670, 14.576215, 24.12029, 33.84658, 51.19238],
                     [18.283899, 14.886669, 23.48641, 33.02413, 43.75668],
                     [16.703169, 14.467865, 20.61421, 29.30655, 38.48771],
                     [15.650480, 14.885835, 16.94441, 23.28699, 29.93602],
                     [17.25151, 19.166625, 35.70161, 39.736435, 42.42644],
                     [17.08081, 16.192965, 32.88968, 35.55251, 37.01797]])
    )
    def test_COSMIC_r_BH_predictor(self, FeH, expected):

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._COSMIC_r_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[6.47069, 14.576215, 24.12029, 33.846585, 51.19238],
                     [6.47069, 14.576215, 24.12029, 33.846585, 51.19238],
                     [6.23963, 14.886669, 23.48641, 33.02413, 43.75668],
                     [4.98317, 14.467865, 20.61421, 29.30655, 38.48771],
                     [4.26691, 14.885835, 16.94441, 23.286994, 29.93602],
                     [3.12858, 22.818915, 35.70161, 39.736435, 42.42644],
                     [2.94267, 20.37408, 32.88968, 35.55251, 37.01797]])
    )
    def test_COSMIC_d_BH_predictor(self, FeH, expected):

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._COSMIC_d_BH_predictor(FeH)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[18.456314, 38.749927, 58.999909, 79.249882, 99.499864],
                     [18.177642, 14.576217, 24.120291, 33.846583, 51.192379],
                     [18.283871, 14.886669, 23.486414, 33.024130, 43.756678],
                     [16.703152, 14.467867, 20.614208, 29.306547, 38.487708],
                     [15.650416, 14.885837, 16.944409, 23.286995, 29.936022],
                     [17.251590, 19.166625, 35.701608, 39.736438, 42.426442],
                     [16.727764, 36.509877, 44.5, 44.5, 44.5]])
    )
    def test_COSMIC_full_r_BH_predictor(self, FeH, expected):
        # Different from canned grid when FeH outside bounds.

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._COSMIC_full_BH_predictor(FeH, remnantflag=3)

        assert pred(mi) == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
    @pytest.mark.parametrize(
        'FeH, expected',
        zip(metals, [[6.778227, 38.749927, 58.999909, 79.249882, 99.499864],
                     [6.470690, 14.576217, 24.120291, 33.846583, 51.192379],
                     [6.239633, 14.886669, 23.486414, 33.024130, 43.756678],
                     [4.983174, 14.467867, 20.614208, 29.306547, 38.487708],
                     [4.266913, 14.885837, 16.944409, 23.286995, 29.936022],
                     [3.128582, 22.818911, 35.701608, 39.736438, 42.426442],
                     [7.775317, 36.509877, 44.5, 44.5, 44.5]])
    )
    def test_COSMIC_full_d_BH_predictor(self, FeH, expected):
        # Different from canned grid when FeH outside bounds.

        mi = np.linspace(19., 100., 5)

        pred, *_ = ifmr._COSMIC_full_BH_predictor(FeH, remnantflag=4)

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

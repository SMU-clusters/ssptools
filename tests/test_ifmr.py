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

    @pytest.mark.parametrize("FeH, expected", zip(metals, WD_metals))
    def test_WD_FeH(self, FeH, expected):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.FeH_WD == expected

    @pytest.mark.parametrize("FeH, expected", zip(metals, BH_metals))
    def test_BH_FeH(self, FeH, expected):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.FeH_BH == expected


class TestBounds:
    '''Tests about IFMR remnant mass bounds'''

    # Initial and final mass bounds for all remnants, for each FeH in `metals`
    BH_mi = [(19.7, 150.0), (19.7, 150.0), (19.8, 150.0), (20.5, 150.0),
             (20.9, 150.0), (24.3, 150.0), (24.3, 150.0)]

    BH_mf = [(5.4977, np.inf), (5.4977, np.inf), (5.5952, np.inf),
             (5.503, np.inf), (5.5386, np.inf), (5.5081, np.inf),
             (5.5081, np.inf)]

    WD_mi = [(0.0, 5.318525), (0.0, 5.318525), (0.0, 5.318525),
             (0.0, 5.47216), (0.0, 5.941481), (0.0, 5.941481), (0.0, 5.941481)]

    WD_mf = [(0.0, 1.228837), (0.0, 1.228837), (0.0, 1.228837),
             (0.0, 1.228496), (0.0, 1.256412), (0.0, 1.256412), (0.0, 1.256412)]

    NS_mi = [(5.318525, 19.7), (5.318525, 19.7), (5.318525, 19.8),
             (5.472164, 20.5), (5.941481, 20.9), (5.941481, 24.3),
             (5.941481, 24.3)]

    @pytest.mark.parametrize("FeH, mi", zip(metals, BH_mi))
    def test_BH_mi(self, FeH, mi):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.BH_mi == pytest.approx(mi)

    @pytest.mark.parametrize("FeH, mf", zip(metals, BH_mf))
    def test_BH_mf(self, FeH, mf):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.BH_mf == pytest.approx(mf)

    @pytest.mark.parametrize("FeH, mi", zip(metals, WD_mi))
    def test_WD_mi(self, FeH, mi):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.WD_mi == pytest.approx(mi)

    @pytest.mark.parametrize("FeH, mf", zip(metals, WD_mf))
    def test_WD_mf(self, FeH, mf):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.WD_mf == pytest.approx(mf)

    @pytest.mark.parametrize("FeH, mi", zip(metals, NS_mi))
    def test_NS_mi(self, FeH, mi):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.NS_mi == pytest.approx(mi)

    @pytest.mark.parametrize("FeH", metals)
    def test_NS_mf(self, FeH):
        IFMR = ifmr.IFMR(FeH)
        assert IFMR.NS_mf == pytest.approx((1.4, 1.4))


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

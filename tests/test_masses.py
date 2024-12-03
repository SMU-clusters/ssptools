#!/usr/bin/env python

from contextlib import nullcontext as does_not_raise

import pytest
import numpy as np

from ssptools import masses, ifmr


DEFAULT_M_BREAK = [0.1, 0.5, 1.0, 100]

DEFAULT_IMF = masses.PowerLawIMF(
    m_break=DEFAULT_M_BREAK, a=[-0.5, -1.3, -2.5], N0=5e5
)

DEFAULT_KWARGS = dict(
    m_break=DEFAULT_M_BREAK, nbins=[1, 1, 2],
    imf=DEFAULT_IMF, ifmr=ifmr.IFMR(-1)
)


# TODO IMF tests, `MassBins.initial_values` tests


class TestHelperMethods:

    # ----------------------------------------------------------------------
    # Testing division of bins helper function
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "N, Nsec, expected",
        [
            (10, 2, [5, 5]),
            (10, 3, [4, 3, 3]),
            (2, 5, [1, 1, 0, 0, 0]),
            (0, 5, [0, 0, 0, 0, 0]),
            (10, 0, pytest.raises(ZeroDivisionError)),
            (0, 0, pytest.raises(ZeroDivisionError))
        ]
    )
    def test_divide_bin_sizes(self, N, Nsec, expected):

        if isinstance(expected, list):
            assert masses._divide_bin_sizes(N, Nsec) == expected

        else:
            with expected:
                masses._divide_bin_sizes(N, Nsec)

    # ----------------------------------------------------------------------
    # Testing computation of P_k helper integral solution
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize('k', [1., 1.5, 2.])
    @pytest.mark.parametrize('a', [-2, -1., -0.5, 1.0])
    def test_Pk(self, a, k):
        from scipy.integrate import quad

        m1, m2 = 0.5, 1.0
        expected, err = quad(lambda m: m**(a + k - 1), m1, m2)

        Pk = masses.Pk(a=a, k=k, m1=m1, m2=m2)

        assert Pk == pytest.approx(expected, abs=err)


class TestArrayPacking:

    # Test the packing and unpacking of y

    mb = masses.MassBins(**DEFAULT_KWARGS)

    # ----------------------------------------------------------------------
    # Test the "packing" and "unpacking" functionality for ODE style y-arrays
    # ----------------------------------------------------------------------

    def test_packing(self):
        arrays = np.split(range(self.mb._ysize), self.mb._blueprint[1:-1])

        res = self.mb.pack_values(*arrays)

        expected = np.concatenate(arrays)

        np.testing.assert_equal(res, expected)

    def test_unpacking(self):

        y = np.arange(self.mb._ysize)

        res = self.mb.unpack_values(y)

        expected = np.split(range(self.mb._ysize), self.mb._blueprint[1:-1])

        assert len(res) == len(expected)

        for r, e in zip(res, expected):
            np.testing.assert_equal(r, e)


class TestArrayCreation:

    # Test the creation of both initial and blank arrays

    mb = masses.MassBins(**DEFAULT_KWARGS)

    # ----------------------------------------------------------------------
    # Testing creation of "blanks" of correct sizes
    # ----------------------------------------------------------------------
    # Parametrize with values each time because internal logic differs w/ value
    # Use numpy.testing because we want exact matches, not pytest.approx

    @pytest.mark.parametrize('value', [1., 0.])  # TODO how to test empty?
    def test_blanks_value(self, value):

        kw = {'extra_dims': None, 'packed': True}

        res = self.mb.blanks(value=value, **kw)

        expected = np.array([value] * self.mb._ysize)

        np.testing.assert_equal(res, expected)

    @pytest.mark.parametrize('value', [1., 0.])
    @pytest.mark.parametrize('dims', [[], [1], [1, 2, 3]])
    def test_blanks_shape(self, value, dims):

        kw = {'packed': True}

        res = self.mb.blanks(value=value, extra_dims=dims, **kw)

        expected = np.full(fill_value=value, shape=(*dims, self.mb._ysize))

        np.testing.assert_equal(res, expected)

    @pytest.mark.parametrize('value', [1., 0.])
    @pytest.mark.parametrize('packed', [True, False])
    def test_blanks_packing(self, value, packed):

        kw = {'extra_dims': None}

        res = self.mb.blanks(value=value, packed=packed, **kw)

        if packed is True:
            expected = np.array([value] * self.mb._ysize)

            np.testing.assert_equal(res, expected)

        else:
            nbin = self.mb.nbin
            shape = (nbin.MS, nbin.MS, nbin.WD, nbin.NS,
                     nbin.BH, nbin.WD, nbin.NS, nbin.BH)
            expected = [np.array([value] * i) for i in shape]

            assert len(res) == len(expected)

            for r, e in zip(res, expected):
                np.testing.assert_equal(r, e)


class TestMassIndices:

    # Test the finding of indices and of changing it based on mto

    mb = masses.MassBins(**DEFAULT_KWARGS)

    # ----------------------------------------------------------------------
    # Test determining the index of a given mass
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mass, massbin, expected",
        [
            (0.99, 'MS', 1),
            (1., 'MS', 2),
            (1., 'WD', 2),
            (1., 'NS', 0),
            (DEFAULT_KWARGS['ifmr'].BH_mf.lower, 'BH', 0),
            (99., 'BH', 1),
            (2., masses.mbin(*np.array([[1, 2, 3], [2, 3, 4]])), 1)
        ]
    )
    def test_determine_index(self, mass, massbin, expected):

        assert self.mb.determine_index(mass, massbin) == expected

    @pytest.mark.parametrize(
        "value, overflow, expected",
        [
            (0.01, True, pytest.raises(ValueError)),
            (0.1, True, does_not_raise()),
            (100, True, does_not_raise()),
            (101, True, does_not_raise()),
            #
            (0.01, False, pytest.raises(ValueError)),
            (0.1, False, does_not_raise()),
            (100, False, pytest.raises(ValueError)),
            (101, False, pytest.raises(ValueError)),
        ]
    )
    def test_determine_overflow(self, value, overflow, expected):

        with expected:
            self.mb.determine_index(value, 'MS', allow_overflow=overflow)

    # ----------------------------------------------------------------------
    # Test rearranging the mass bins based on a given mto
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mto, expected_upper",
        [
            (-1, [0.5, 1, 10, 100]),
            (0, [0.5, 1, 10, 100]),
            (0.3, [0.3, 1, 10, 100]),
            (0.5, [0.5, 0.5, 10, 100]),
            (101, [0.5, 1, 10, 100])
        ]
    )
    def test_to_bins(self, mto, expected_upper):

        res = self.mb.turned_off_bins(mto).upper

        np.testing.assert_equal(res, expected_upper)


# class TestMassBinInit:

    # Test the creation of mass bins with various methods


class TestIMFInit:

    # ----------------------------------------------------------------------
    # Test computation of normalization factors A, for different N comps
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        'mb, a, expected_A',
        [
            ([1, 100], [-1.5], [0.55555556]),
            ([1, 100], [+1.5], [2.500025e-05]),
            ([1, 50, 100], [-1.5, -2.5], [0.56239653, 28.11982643]),
            ([1, 50, 100], [-1.5, +2.5], [4.68626830e-01, 7.49802927e-08]),
            ([1, 25, 50, 75, 100], [-0.5, -1.5, -2.5, -3.5],
             [8.06597675e-02, 2.01649419e+00, 1.00824709e+02, 7.56185321e+03]),
            ([1, 25, 50, 75, 100], [-0.5, +1.5, +2.5, -3.5],
             [1.50104067e-02, 2.40166506e-05, 4.80333013e-07, 8.54889566e+04]),
        ]
    )
    def test_normalizations(self, mb, a, expected_A):

        imf = masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

        assert imf._A_comps == pytest.approx(expected_A)

    # ----------------------------------------------------------------------
    # Test failure when creating IMF with malformed m_break, a_slopes
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a, expected",
        [
            ([1, 100], [-1.5], does_not_raise()),
            ([1, 50, 100], [-1.5, -2.5], does_not_raise()),
            ([100, 1], [-1.5], pytest.raises(ValueError)),
            ([1, 50, 100], [-1.5], pytest.raises(ValueError)),
            ([1, 100], [-1.5, -2.5], pytest.raises(ValueError)),
            ([50], [], pytest.raises(ValueError)),
        ]
    )
    def test_invalid_init(self, mb, a, expected):

        with expected:
            masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

    # ----------------------------------------------------------------------
    # Test init to match a given total mass
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a", [([1, 100], [-1.5]), ([1, 50, 100], [-1.5, -1.5])]
    )
    @pytest.mark.parametrize('M0', (1, 5e5, 1e6))
    def test_init_from_M0(self, mb, a, M0):

        imf = masses.PowerLawIMF.from_M0(m_break=mb, a=a, M0=M0)

        assert imf.N0 * imf.mmean == pytest.approx(M0)


class TestIMFEval:

    # ----------------------------------------------------------------------
    # Test evaluating IMF (N, M) for continuous masses
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a, mass, expected_N",
        [
            ([1, 100], [-1.5], 50, 1.5713484),
            ([1, 100], [-1.5], [10, 20, 30], [17.568209, 6.211299, 3.381003]),
            ([1, 50, 100], [-1.5, -2.5], 25, 4.4991722),
            ([1, 50, 100], [-1.5, -2.5], 50, 1.5906975),
            ([1, 50, 100], [-1.5, -2.5], [25, 50, 75],
             [4.49917223, 1.5906976, 0.57724407]),
        ]
    )
    def test_call(self, mb, a, mass, expected_N):

        imf = masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

        assert imf(mass) == pytest.approx(expected_N)
        assert imf.N(mass) == pytest.approx(expected_N)

    @pytest.mark.parametrize(
        "mb, a, mass, expected_M",
        [
            ([1, 100], [-1.5], 50, 78.567420),
            ([1, 100], [-1.5], [10, 20, 30],
             [175.682092, 124.225998, 101.430103]),
            ([1, 50, 100], [-1.5, -2.5], 25, 112.479305),
            ([1, 50, 100], [-1.5, -2.5], 50, 79.534879),
            ([1, 50, 100], [-1.5, -2.5], [25, 50, 75],
             [112.479305, 79.534879, 43.293304]),
        ]
    )
    def test_call_M(self, mb, a, mass, expected_M):

        imf = masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

        assert imf.M(mass) == pytest.approx(expected_M)

    # ----------------------------------------------------------------------
    # Test calling outside of mass bounds
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a", [([1, 100], [-1.5]), ([1, 50, 100], [-1.5, -1.5])]
    )
    @pytest.mark.parametrize(
        "ext, mass, expected_N",
        [
            (0, 0.1, 17568.209223),
            (0, 101, 0.547325),
            ('extrapolate', 0.1, 17568.209223),
            ('extrapolate', 101, 0.547325),
            #
            (1, 0.1, 0.0),
            (1, 101, 0.0),
            ('zeros', 0.1, 0.0),
            ('zeros', 101, 0.0),
            #
            (2, 0.1, pytest.raises(ValueError)),
            (2, 101, pytest.raises(ValueError)),
            ('raise', 0.1, pytest.raises(ValueError)),
            ('raise', 101, pytest.raises(ValueError)),
        ]
    )
    def test_call_extrapolation(self, mb, a, ext, mass, expected_N):

        imf = masses.PowerLawIMF(m_break=mb, a=a, ext=ext, N0=1000)

        if isinstance(expected_N, float):
            assert imf(mass) == pytest.approx(expected_N)

        else:
            with expected_N:
                imf(mass)

    # ----------------------------------------------------------------------
    # Test computation of total mass, mean mass
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a, expected_Mtot",
        [
            ([1, 100], [-1.5], 10000.0),
            ([1, 50, 100], [-1.5, -2.5], 9158.217621),
            ([1, 50, 100], [+1.5, -2.5], 52440.52123),
        ]
    )
    def test_Mtot(self, mb, a, expected_Mtot):

        imf = masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

        assert imf.Mtot == pytest.approx(expected_Mtot)

    @pytest.mark.parametrize(
        "mb, a, expected_mmean",
        [
            ([1, 100], [-1.5], 10.0),
            ([1, 50, 100], [-1.5, -2.5], 9.158217),
            ([1, 50, 100], [+1.5, -2.5], 52.440521),
        ]
    )
    def test_mmean(self, mb, a, expected_mmean):

        imf = masses.PowerLawIMF(m_break=mb, a=a, N0=1000)

        assert imf.mmean == pytest.approx(expected_mmean)


class TestIMFBinnedEval:

    bins = masses.mbin(*np.array([[1, 25, 50], [25, 50, 100]]))

    # ----------------------------------------------------------------------
    # Test evaluating IMF (N, M) for binned masses
    # ----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "mb, a, expected_N",
        [
            ([1, 100], [-1.5], [888.888888, 65.087381, 46.023729]),
            ([1, 50, 100], [-1.5, -2.5], [899.834445, 65.888851, 34.276702]),
            ([10, 90], [-1.5], [0., 277.862904, 0.]),
        ]
    )
    def test_binned_N(self, mb, a, expected_N):

        imf = masses.PowerLawIMF(m_break=mb, a=a, ext='zeros', N0=1000)

        assert imf.binned_eval(self.bins)[0] == pytest.approx(expected_N)
        assert imf.binned_N(self.bins) == pytest.approx(expected_N)

    @pytest.mark.parametrize(
        "mb, a, expected_M",
        [
            ([1, 100], [-1.5], [4444.444444, 2301.186457, 3254.369097]),
            ([1, 50, 100], [-1.5, -2.5], [4499.17222, 2329.522696, 2329.52269]),
            ([10, 90], [-1.5], [0., 9823.937211, 0.]),
        ]
    )
    def test_binned_M(self, mb, a, expected_M):

        imf = masses.PowerLawIMF(m_break=mb, a=a, ext='zeros', N0=1000)

        assert imf.binned_eval(self.bins)[1] == pytest.approx(expected_M)
        assert imf.binned_M(self.bins) == pytest.approx(expected_M)

    @pytest.mark.parametrize(
        "mb, a, expected_alpha",
        [
            ([1, 100], [-1.5], [-1.5, -1.5, -1.5]),
            ([1, 50, 100], [-1.5, -2.5], [-1.5, -1.5, -2.5]),
            ([10, 90], [-1.5], [0., -1.5, 0.]),
        ]
    )
    def test_binned_alpha(self, mb, a, expected_alpha):

        imf = masses.PowerLawIMF(m_break=mb, a=a, ext='zeros', N0=1000)

        assert imf.binned_eval(self.bins)[-1] == pytest.approx(expected_alpha)

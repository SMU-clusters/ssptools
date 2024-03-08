#!/usr/bin/env python

import pytest
import numpy as np

from ssptools import masses, ifmr


DEFAULT_KWARGS = dict(
    m_break=[0.1, 0.5, 1.0, 100], a=[-0.5, -1.3, -2.5],
    nbins=[1, 1, 2], N0=5e5, ifmr=ifmr.IFMR(-1)
)


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
    from contextlib import nullcontext as does_not_raise

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

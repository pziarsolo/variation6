import unittest

import numpy as np
import dask.array as da
import math

from variation6 import (GT_FIELD, AO_FIELD, RO_FIELD, DP_FIELD,
                        EmptyVariationsError, FLT_VARS)
from variation6.in_out.zarr import load_zarr
from variation6.tests import TEST_DATA_DIR
from variation6.compute import compute
from variation6.variations import Variations
from variation6.stats import (calc_missing_gt, calc_maf_by_allele_count,
                              calc_maf_by_gt, calc_mac, count_alleles,
                              calc_obs_het)
from variation6.filters import remove_low_call_rate_vars


def _create_empty_dask_variations():
    variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
    return remove_low_call_rate_vars(variations, min_call_rate=1.1)[FLT_VARS]


def _create_dask_variations():
    variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
    return remove_low_call_rate_vars(variations, min_call_rate=0)[FLT_VARS]


class StatsTest(unittest.TestCase):

    def test_allele_count(self):
        gts = np.array([[[0, 2], [-1, -1]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        counts = count_alleles(gts, max_alleles=3)
        expected = np.array([[1, 0, 1, 2], [1, 1, 1, 1], [2, 2, 0, 0], [0, 0, 0, 4]])
        self.assertTrue(np.all(counts == expected))

    def test_allele_count_dask(self):
        variations = _create_dask_variations()
        gts = variations[GT_FIELD]
        counts = count_alleles(gts, max_alleles=3)
        expected = [[2, 2, 0, 2], [2, 2, 0, 2], [2, 2, 0, 2], [3, 1, 0, 2, ],
                    [0, 0, 0, 6], [1, 1, 0, 4], [1, 3, 0, 2]]
        self.assertTrue(np.all(expected == counts.compute()))

    def test_empty_gt_allele_count(self):
        gts = np.array([])
        with self.assertRaises(EmptyVariationsError):
            count_alleles(gts, max_alleles=3)

        variations = _create_empty_dask_variations()
        gts = variations[GT_FIELD]
        task = count_alleles(gts, max_alleles=3)
        counts = task.compute()
        self.assertEqual(counts.shape, (0, 4))

    def test_calc_missing(self):
        variations = _create_dask_variations()
        future_result = calc_missing_gt(variations, rates=False)
        result = compute(future_result)
        self.assertTrue(np.array_equal(result['num_missing_gts'],
                                       [1, 1, 1, 1, 3, 2, 1]))

        future_result = calc_missing_gt(variations, rates=True)
        result = compute(future_result)
        expected = [0.33, 0.33, 0.33, 0.333, 1, 0.666, 0.33]
        for a, b in zip(result['num_missing_gts'], expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_calc_missing_empty_vars(self):
        variations = _create_empty_dask_variations()

        task = calc_missing_gt(variations, rates=True)
        result = compute(task)
        self.assertEqual(result['num_missing_gts'].shape, (0,))
#         with self.assertRaises(EmptyVariationsError):

    def test_calc_maf_by_allele_count(self):
        variations = Variations(samples=da.array(['aa', 'bb']))
        variations[GT_FIELD] = da.from_array([[[-1, 1], [2, 1]],
                                              [[-1, -1], [-1, 2]],
                                              [[1, -1], [1, 1]]])
        variations[RO_FIELD] = da.from_array(np.array([[-1, 8], [-1, -1], [6, 4]]))
        variations[AO_FIELD] = da.from_array(np.array([[[1, 4], [2, 1]],
                                                  [[-1, -1], [3, 3]],
                                                  [[1, 4], [5, 1]]]))
        # with this step we create a  variation with dask arrays of unknown
        # shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        future_result = calc_maf_by_allele_count(variations,
                                                 min_num_genotypes=0)
        result = compute(future_result)

        expected = [0.5, 0.5, 0.47619048]
        for a, b in zip(result['mafs'], expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_calc_maf_by_allele_count_empty_vars(self):
        variations = _create_empty_dask_variations()
        task = calc_maf_by_allele_count(variations)
        result = compute(task)
        self.assertEqual(result['mafs'].shape, (0,))

    def test_calc_maf_by_gt(self):
        variations = Variations(samples=da.array(['aa', 'bb']))

        gts = np.array([[[0, 2], [-1, -1]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        variations[GT_FIELD] = da.from_array(gts)  # , chunks=(2, 1, 2))
        # with this step we create a  variation with dask arrays of unknown
        # shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        mafs = calc_maf_by_gt(variations, max_alleles=3,
                              min_num_genotypes=0)
        result = compute(mafs)

        expected = [0.5, 0.33333333, 0.5, math.nan]
        for a, b in zip(result['mafs'], expected):
            if math.isnan(a):
                self.assertTrue(math.isnan(b))
                continue
            self.assertAlmostEqual(a, b, places=2)

    def test_calc_mac(self):
        variations = Variations(samples=da.array(['aa', 'bb']))

        gts = np.array([[[0, 0], [0, 0]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        variations[GT_FIELD] = da.from_array(gts)
        # with this step we create a  variation with dask arrays of unknown
        # shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        macs = calc_mac(variations, max_alleles=3, min_num_genotypes=0)
        result = compute(macs)
        expected = [2, 1, 1, math.nan]
        for a, b in zip(result['macs'], expected):
            if math.isnan(a):
                self.assertTrue(math.isnan(b))
                continue
            self.assertAlmostEqual(a, b, places=2)


class ObsHetTest(unittest.TestCase):

    def test_calc_obs_het(self):
        variations = Variations(samples=da.array(['a', 'b', 'c', 'd']))
        gts = np.array([[[0, 0], [0, 1], [0, -1], [-1, -1]],
                           [[0, 0], [0, 0], [0, -1], [-1, -1]]])

        dps = np.array([[5, 12, 10, 10],
                           [10, 10, 10, 10]])
        variations[GT_FIELD] = da.from_array(gts)
        variations[DP_FIELD] = da.from_array(dps)
        # with this step we create a  variation with dask arrays of unknown shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        het = calc_obs_het(variations, min_num_genotypes=0)
        self.assertTrue(np.allclose(het['obs_het'].compute(), [0.5, 0]))

#         het = calc_obs_het(variations, min_num_genotypes=10)
#         assert np.allclose(het, [np.NaN, np.NaN], equal_nan=True)

        het = calc_obs_het(variations, min_num_genotypes=0, min_allowable_call_dp=10)
        self.assertTrue(np.allclose(het['obs_het'].compute(), [1, 0]))
        het = calc_obs_het(variations, min_num_genotypes=0, max_allowable_call_dp=11)
        self.assertTrue(np.allclose(het['obs_het'].compute(), [0, 0]))

        het = calc_obs_het(variations, min_num_genotypes=0, min_allowable_call_dp=5)
        self.assertTrue(np.allclose(het['obs_het'].compute(), [0.5, 0]))


if __name__ == '__main__':
#     import sys; sys.argv = ['', 'StatsTest.test_calc_maf_by_allele_count']
    unittest.main()

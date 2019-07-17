import unittest

import numpy as np
import dask.array as da
import math

from variation6 import GT_FIELD, AO_FIELD, AD_FIELD, RO_FIELD
from variation6.in_out.zarr import load_zarr
from variation6.tests import TEST_DATA_DIR
from variation6.compute import compute
from variation6.variations import Variations
from variation6.stats import (calc_missing_gt, calc_maf_by_allele_count,
                              calc_maf_by_gt, calc_mac, count_alleles)


class StatsTest(unittest.TestCase):

    def test_allele_count(self):
        gts = np.array([[[0, 2], [-1, -1]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        counts = count_alleles(gts)
        expected = np.array([[1, 0, 1, 2], [1, 1, 1, 1], [2, 2, 0, 0], [0, 0, 0, 4]])
        self.assertTrue(np.all(counts == expected))

    def test_calc_missing(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        future_result = calc_missing_gt(variations, rates=False)
        result = compute(future_result)
        self.assertTrue(np.array_equal(result['num_missing_gts'],
                                       [1, 1, 1, 1, 3, 2, 1]))

        future_result = calc_missing_gt(variations, rates=True)
        result = compute(future_result)
        expected = [0.33, 0.33, 0.33, 0.333, 1, 0.666, 0.33]
        for a, b in zip(result['num_missing_gts'], expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_calc_maf_by_allele_count(self):
        # variations with AD fields in the vcf
        variations = Variations(samples=da.array(['aa', 'bb']))
        variations[AD_FIELD] = da.from_array(np.array([[[-1, 1, 4], [8, 2, 1]],
                                                  [[-1, -1, -1], [-1, 3, 3]],
                                                  [[6, 1, 4], [4, 5, 1]]]))
        future_result = calc_maf_by_allele_count(variations)
        result = compute(future_result)
        expected = [0.5, 0.5, 0.47619048]
        for a, b in zip(result['mafs'], expected):
            self.assertAlmostEqual(a, b, places=2)

        variations = Variations(samples=da.array(['aa', 'bb']))
        variations[RO_FIELD] = da.from_array(np.array([[-1, 8], [-1, -1], [6, 4]]))
        variations[AO_FIELD] = da.from_array(np.array([[[1, 4], [2, 1]],
                                                  [[-1, -1], [3, 3]],
                                                  [[1, 4], [5, 1]]]))
        future_result = calc_maf_by_allele_count(variations)
        result = compute(future_result)

        expected = [0.5, 0.5, 0.47619048]
        for a, b in zip(result['mafs'], expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_calc_maf_by_gt(self):
        variations = Variations(samples=da.array(['aa', 'bb']))

        gts = np.array([[[0, 2], [-1, -1]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        variations[GT_FIELD] = da.from_array(gts)  # , chunks=(2, 1, 2))
        mafs = calc_maf_by_gt(variations)
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
        macs = calc_mac(variations)
        result = compute(macs)
        expected = np.array([ 2., 1., 1., math.nan])
        for a, b in zip(result['macs'], expected):
            if math.isnan(a):
                self.assertTrue(math.isnan(b))
                continue
            self.assertAlmostEqual(a, b, places=2)


if __name__ == '__main__':
    unittest.main()

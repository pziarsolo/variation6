import unittest
from pathlib import Path

import numpy as np

from variation6 import GT_FIELD
from variation6.in_out.zarr import load_zarr
from variation6.stats import calc_missing_gt

from variation6.tests import TEST_DATA_DIR


class StatsTest(unittest.TestCase):
    def test_calsc_missing(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        result = calc_missing_gt(variations, rates=False)
        result.compute()
        self.assertTrue(np.array_equal(result['num_missing_gts'],
                                       [1, 1, 1, 1, 3, 2, 1]))


        result = calc_missing_gt(variations, rates=True)
        result.compute()
        expected = [0.33, 0.33, 0.33, 0.333, 1, 0.666, 0.33]
        for a, b in zip(result['num_missing_gts'], expected):
            self.assertAlmostEqual(a, b, places=2)


if __name__ == '__main__':
    unittest.main()
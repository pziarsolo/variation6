import unittest

import numpy as np

from variation6.in_out.zarr import load_zarr
from variation6.stats import calc_missing_gt
from variation6.tests import TEST_DATA_DIR
from variation6.compute import compute


class StatsTest(unittest.TestCase):

    def test_calsc_missing(self):
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


if __name__ == '__main__':
    unittest.main()

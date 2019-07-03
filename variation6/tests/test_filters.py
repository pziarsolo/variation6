import unittest
import numpy as np

from variation6 import GT_FIELD, DP_FIELD, MISSING_INT
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr, save_zarr
from variation6.filters import (filter_by_min_call, FLT_VARS,
                                min_depth_gt_to_missing,
                                filter_samples)


class MinCallFilterTest(unittest.TestCase):

    def test_filter_by_call_rate(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        result = filter_by_min_call(variations, min_calls=0.5)

        result = filter_by_min_call(result[FLT_VARS], min_calls=0.5)
        gts = result[FLT_VARS][GT_FIELD].compute()
        self.assertEqual(gts.shape, (2, 3, 2))


class MinDepthGtToMissing(unittest.TestCase):

    def test_min_depth_gt_to_missing(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        depth = 9
        prev_gts = variations[GT_FIELD].compute()
        depths = variations[DP_FIELD].compute()
        result = min_depth_gt_to_missing(variations, min_depth=depth)
        post_gts = result[FLT_VARS][GT_FIELD].compute()

        for dp, prev_gt, post_gt in zip(depths, prev_gts, post_gts):
            for dp_, prev_gt_, post_gt_ in zip(dp, prev_gt, post_gt):
                if dp_ != -1 and dp_ < depth:
                    self.assertTrue(np.all(post_gt_ == [MISSING_INT, MISSING_INT]))
                    self.assertFalse(np.all(prev_gt_ == [MISSING_INT, MISSING_INT]))


class FilterSamplesTest(unittest.TestCase):

    def test_samples_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        samples = ['pepo', 'upv196']
        result = filter_samples(variations, samples=samples)
        dps = result[FLT_VARS][DP_FIELD].compute()
        expected = [[-1, 9], [-1, 8], [-1, 8], [14, 6], [-1, -1], [-1, -1],
                    [-1, 6]]
        self.assertTrue(np.all(dps == expected))

        save_zarr(result[FLT_VARS], TEST_DATA_DIR / 'prueba.zarr')


if __name__ == '__main__':
    unittest.main()

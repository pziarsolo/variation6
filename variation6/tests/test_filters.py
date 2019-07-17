import unittest

import numpy as np

from variation6 import (GT_FIELD, DP_FIELD, MISSING_INT, FLT_VARS, N_KEPT,
                        N_FILTERED_OUT)
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr
from variation6.filters import (remove_low_call_rate_vars,
                                min_depth_gt_to_missing,
                                filter_samples, filter_by_maf_by_allele_count, filter_by_mac,
    filter_by_maf)

from variation6.compute import compute


class MinCallFilterTest(unittest.TestCase):

    def test_filter_by_call_rate(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        pipeline_futures = {}
        future_result = remove_low_call_rate_vars(variations, min_call_rate=0.5)
        pipeline_futures.update(future_result)

        future_result2 = remove_low_call_rate_vars(future_result[FLT_VARS], min_call_rate=0.5,
                                           filter_id='call_rate2')
        pipeline_futures.update(future_result2)

        processed = compute(pipeline_futures, store_variation_to_memory=True)

        self.assertEqual(processed['call_rate'][N_KEPT], 2)
        self.assertEqual(processed['call_rate'][N_FILTERED_OUT], 5)

        gts = processed[FLT_VARS][GT_FIELD]
        self.assertEqual(gts.shape, (2, 3, 2))
        self.assertTrue(np.all(processed[FLT_VARS].samples == variations.samples.compute()))
        self.assertEqual(processed[FLT_VARS].metadata, variations.metadata)


class MinDepthGtToMissing(unittest.TestCase):

    def test_min_depth_gt_to_missing(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        depth = 9
        prev_gts = variations[GT_FIELD].compute()
        depths = variations[DP_FIELD].compute()
        task = min_depth_gt_to_missing(variations, min_depth=depth)
        processed = compute(task, store_variation_to_memory=True)
        post_gts = processed[FLT_VARS][GT_FIELD]

        for dp, prev_gt, post_gt in zip(depths, prev_gts, post_gts):
            for dp_, prev_gt_, post_gt_ in zip(dp, prev_gt, post_gt):
                if dp_ != -1 and dp_ < depth:
                    self.assertTrue(np.all(post_gt_ == [MISSING_INT, MISSING_INT]))
                    self.assertFalse(np.all(prev_gt_ == [MISSING_INT, MISSING_INT]))


class FilterSamplesTest(unittest.TestCase):

    def test_samples_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        samples = ['pepo', 'upv196']
        task = filter_samples(variations, samples=samples)
        processed = compute(task, store_variation_to_memory=True)
        dps = processed[FLT_VARS][DP_FIELD]
        self.assertTrue(np.all(samples == processed[FLT_VARS].samples))
        expected = [[-1, 9], [-1, 8], [-1, 8], [14, 6], [-1, -1], [-1, -1],
                    [-1, 6]]
        self.assertTrue(np.all(dps == expected))


class MafFilterTest(unittest.TestCase):

    def test_maf_by_allele_count_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        task = filter_by_maf_by_allele_count(variations, remove_above=0.6)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 4)
        self.assertEqual(result['filter_by_maf_by_allele_count'], {'n_kept': 4,
                                                   'n_filtered_out': 3})

    def test_maf_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        task = filter_by_maf(variations, remove_above=0.6)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 4)
        self.assertEqual(result['filter_by_maf'], {'n_kept': 4,
                                                   'n_filtered_out': 3})

    def test_mac_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        task = filter_by_mac(variations, remove_above=1)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 0)
        self.assertEqual(result['filter_by_mac'], {'n_kept': 0,
                                                   'n_filtered_out': 7})


if __name__ == '__main__':
    unittest.main()

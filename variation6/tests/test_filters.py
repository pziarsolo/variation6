import unittest
import dask.array as da
import numpy as np

from variation6 import (GT_FIELD, DP_FIELD, MISSING_INT, FLT_VARS, N_KEPT,
                        N_FILTERED_OUT, CHROM_FIELD, POS_FIELD)
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr
from variation6.filters import (remove_low_call_rate_vars,
                                min_depth_gt_to_missing,
                                keep_samples, filter_by_maf_by_allele_count,
                                filter_by_mac, filter_by_maf,
                                keep_variable_variations,
                                keep_variations_in_regions,
                                remove_variations_in_regions, remove_samples)

from variation6.compute import compute
from variation6.variations import Variations


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
        self.assertEqual(processed['call_rate'][N_KEPT], 5)
        self.assertEqual(processed['call_rate'][N_FILTERED_OUT], 2)

        gts = processed[FLT_VARS][GT_FIELD]
        self.assertEqual(gts.shape, (5, 3, 2))
        self.assertTrue(np.all(processed[FLT_VARS].samples == variations.samples.compute()))
        self.assertEqual(processed[FLT_VARS].metadata, variations.metadata)

    def test_filter_by_call_rate_twice(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        pipeline_futures = {}
        # this rate has no sense but I use to remove all calls
        future_result = remove_low_call_rate_vars(variations, min_call_rate=1.1)
        pipeline_futures.update(future_result)

        future_result2 = remove_low_call_rate_vars(future_result[FLT_VARS], min_call_rate=0.5,
                                           filter_id='call_rate2')
        pipeline_futures.update(future_result2)

        processed = compute(pipeline_futures, store_variation_to_memory=True)
        self.assertEqual(processed['call_rate2'], {'n_kept': 0, 'n_filtered_out': 0})


class MinDepthGtToMissing(unittest.TestCase):

    def test_min_depth_gt_to_missing(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr', chunk_size=2)
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]
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

    def test_keep_samples(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
#         print(variations.samples.compute())
#         print(variations[DP_FIELD].compute())

        samples = ['upv196', 'pepo']
        task = keep_samples(variations, samples=samples)
        processed = compute(task, store_variation_to_memory=True)
        dps = processed[FLT_VARS][DP_FIELD]

        self.assertTrue(np.all(processed[FLT_VARS].samples == ['pepo', 'upv196']))
        expected = [[-1, 9], [-1, 8], [-1, 8], [14, 6], [-1, -1], [-1, -1],
                    [-1, 6]]
        self.assertTrue(np.all(dps == expected))

    def test_remove_samples(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        samples = ['upv196', 'pepo']

        task = remove_samples(variations, samples=samples)
        processed = compute(task, store_variation_to_memory=True)
        dps = processed[FLT_VARS][DP_FIELD]
        self.assertTrue(np.all(['mu16'] == processed[FLT_VARS].samples))
        expected = [[10], [9], [9], [-1], [-1], [ 9], [10]]
        self.assertTrue(np.all(dps == expected))


class MafFilterTest(unittest.TestCase):

    def test_maf_by_allele_count_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        task = filter_by_maf_by_allele_count(variations, max_allowable_maf=0.6,
                                             min_num_genotypes=2)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 4)
        self.assertEqual(result['filter_by_maf_by_allele_count'], {'n_kept': 4,
                                                   'n_filtered_out': 3})

    def test_maf_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        task = filter_by_maf(variations, max_allowable_maf=0.6, max_alleles=3,
                             min_num_genotypes=2)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 3)
        self.assertEqual(result['filter_by_maf'], {'n_kept': 3,
                                                   'n_filtered_out': 4})

    def test_mac_filter(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr', chunk_size=2)
        task = filter_by_mac(variations, max_allowable_mac=1, max_alleles=3)
        result = compute(task, store_variation_to_memory=True)
        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 0)
        self.assertEqual(result['filter_by_mac'], {'n_kept': 0,
                                                   'n_filtered_out': 7})

    def test_filter_macs(self):
        # with some missing values
        gts = np.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [-1, -1], [0, 1], [0, 0], [1, 1]]])
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations = Variations(samples=da.array(samples))
        variations[GT_FIELD] = da.from_array(gts)
        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=5)
        result = compute(task, store_variation_to_memory=True)
        assert result['filter_by_mac'][N_KEPT] == 4
        assert result['filter_by_mac'][N_FILTERED_OUT] == 0

        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=5,
                             min_allowable_mac=0)
        result = compute(task, store_variation_to_memory=True)
        assert np.all(result[FLT_VARS][GT_FIELD] == gts[[0, 1, 2]])
        assert result['filter_by_mac'][N_KEPT] == 3
        assert result['filter_by_mac'][N_FILTERED_OUT] == 1

        # without missing values
        gts = np.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                           [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]],
                           [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations = Variations(samples=da.array(samples))
        variations[GT_FIELD] = da.from_array(gts)
        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=0,
                             max_allowable_mac=4)
        result = compute(task, store_variation_to_memory=True)
        expected = np.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]],
                               [[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])

        assert np.all(result[FLT_VARS][GT_FIELD] == expected)
        expected = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]],
                               [[0, 0], [0, 0], [0, 1], [0, 0], [1, 1]]])
        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=0,
                             min_allowable_mac=3.5, max_allowable_mac=4)
        result = compute(task, store_variation_to_memory=True)
        assert np.all(result[FLT_VARS][GT_FIELD] == expected)

        expected = np.array([[[0, 0], [1, 1], [0, 1], [1, 1], [0, 0]]])
        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=0,
                             max_allowable_mac=3)
        result = compute(task, store_variation_to_memory=True)
        assert np.all(result[FLT_VARS][GT_FIELD] == expected)

        task = filter_by_mac(variations, max_alleles=2, min_num_genotypes=0,
                             min_allowable_mac=2, max_allowable_mac=5)
        result = compute(task, store_variation_to_memory=True)

        assert np.all(result[FLT_VARS][GT_FIELD] == variations[GT_FIELD])


class NoVariableOrMissingTest(unittest.TestCase):

    def test_non_variable_filter(self):
        variations = Variations(samples=da.array(['aa', 'bb']))

        gts = np.array([[[0, 0], [0, 0]],
                        [[0, 2], [1, -1]],
                        [[0, 0], [1, 1]],
                        [[-1, -1], [-1, -1]]
                       ])
        variations[GT_FIELD] = da.from_array(gts)

        task = keep_variable_variations(variations, max_alleles=3)

        result = compute(task, store_variation_to_memory=True)

        filtered_vars = result[FLT_VARS]
        self.assertEqual(filtered_vars.num_variations, 2)
        self.assertEqual(result['variable_variations'], {'n_kept': 2,
                                                                   'n_filtered_out': 2})


class FilterByPositionTest(unittest.TestCase):

    def _create_fake_variations_and_regions(self):
        variations = Variations(samples=da.array(['aa', 'bb']))
        poss = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        chroms = np.array(['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1',
                           'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2'])
        variations[CHROM_FIELD] = da.from_array(chroms)
        variations[POS_FIELD] = da.from_array(poss)
        regions = [('chr1', 4, 6), ('chr2',)]
        return variations, regions

    def test_keep_variations_in_regions(self):
        variations, regions = self._create_fake_variations_and_regions()
        task = keep_variations_in_regions(variations, regions)
        result = compute(task, store_variation_to_memory=True)
        chroms = result[FLT_VARS][CHROM_FIELD]
        poss = result[FLT_VARS][POS_FIELD]
        self.assertTrue(np.all(poss == [4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
        self.assertTrue(np.all(chroms == ['chr1', 'chr1', 'chr2', 'chr2',
                                          'chr2', 'chr2', 'chr2', 'chr2',
                                          'chr2', 'chr2', 'chr2', 'chr2']))

    def test_remove_variations_in_regions(self):
        variations, regions = self._create_fake_variations_and_regions()
        task = remove_variations_in_regions(variations, regions)
        result = compute(task, store_variation_to_memory=True)
        chroms = result[FLT_VARS][CHROM_FIELD]
        poss = result[FLT_VARS][POS_FIELD]
        self.assertTrue(np.all(poss == [1, 2, 3, 6, 7, 8, 9, 0]))
        self.assertTrue(np.all(chroms == ['chr1', 'chr1', 'chr1', 'chr1',
                                          'chr1', 'chr1', 'chr1', 'chr1']))


if __name__ == '__main__':
#     import sys; sys.argv = ['.', 'FilterSamplesTest']
    unittest.main()

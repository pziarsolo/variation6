import unittest
import math

import numpy as np
import dask.array as da

from variation6.stats.distance import (calc_kosman_dist, _kosman,
                                       calc_pop_pairwise_unbiased_nei_dists,
                                       calc_dset_pop_distance)
from variation6.variations import Variations
from variation6 import GT_FIELD, FLT_VARS, DP_FIELD
from variation6.filters import keep_samples
from variation6.compute import compute


class PairwiseFilterTest(unittest.TestCase):

    def test_kosman_2_indis(self):
        a = np.array([[-1, -1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1],
                      [0, 1], [0, 0], [0, 0], [0, 1]])
        b = np.array([[1, 1], [-1, -1], [0, 0], [0, 0], [1, 1], [0, 1], [1, 0],
                      [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = np.stack((a, b), axis=1)
        variations = Variations()
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts

        vars1 = keep_samples(variations, ['0'])[FLT_VARS]
        vars2 = keep_samples(variations, ['1'])[FLT_VARS]
        snp_by_snp_compartion_array = _kosman(vars1, vars2)
        distance_ab = compute(snp_by_snp_compartion_array)
        distance = distance_ab.sum() / distance_ab.shape[0]

        assert distance == 1 / 3

        c = np.full(shape=(11, 2), fill_value=1, dtype=np.int16)
        d = np.full(shape=(11, 2), fill_value=1, dtype=np.int16)
        gts = np.stack((c, d), axis=1)
        variations = Variations()
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts

        vars1 = keep_samples(variations, ['0'])[FLT_VARS]
        vars2 = keep_samples(variations, ['1'])[FLT_VARS]
        snp_by_snp_compartion_array = _kosman(vars1, vars2)
        distance_ab = compute(snp_by_snp_compartion_array)
        distance = distance_ab.sum() / distance_ab.shape[0]
        assert distance == 0

        variations = Variations()
        gts = np.stack((b, d), axis=1)
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts

        vars1 = keep_samples(variations, ['0'])[FLT_VARS]
        vars2 = keep_samples(variations, ['1'])[FLT_VARS]
        snp_by_snp_compartion_array = _kosman(vars1, vars2)
        distance_ab = compute(snp_by_snp_compartion_array)
        distance = distance_ab.sum() / distance_ab.shape[0]
        assert distance == 0.45

    def test_kosman_missing(self):
        a = np.array([[-1, -1], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1],
                      [0, 1], [0, 0], [0, 0], [0, 1]])
        b = np.array([[1, 1], [-1, -1], [0, 0], [0, 0], [1, 1], [0, 1], [1, 0],
                      [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = np.stack((a, b), axis=1)
        variations = Variations()
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts

        vars1 = keep_samples(variations, ['0'])[FLT_VARS]
        vars2 = keep_samples(variations, ['1'])[FLT_VARS]

        snp_by_snp_compartion_array = _kosman(vars1, vars2)
        distance_ab = compute(snp_by_snp_compartion_array)

        c = np.array([[-1, -1], [-1, -1], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        d = np.array([[-1, -1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 1]])
        gts = np.stack((c, d), axis=1)
        variations = Variations()
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts

        vars1 = keep_samples(variations, ['0'])[FLT_VARS]
        vars2 = keep_samples(variations, ['1'])[FLT_VARS]

        snp_by_snp_compartion_array = _kosman(vars1, vars2)
        distance_cd = compute(snp_by_snp_compartion_array)

        assert np.all(distance_ab == distance_cd)

    def test_kosman_pairwise(self):
        a = np.array([[-1, -1], [0, 0], [0, 1],
                         [0, 0], [0, 0], [0, 1], [0, 1],
                         [0, 1], [0, 0], [0, 0], [0, 1]])
        b = np.array([[1, 1], [-1, -1], [0, 0],
                         [0, 0], [1, 1], [0, 1], [1, 0],
                         [1, 0], [1, 0], [0, 1], [1, 2]])
        c = np.full(shape=(11, 2), fill_value=1, dtype=np.int16)
        d = np.full(shape=(11, 2), fill_value=1, dtype=np.int16)
        gts = np.stack((a, b, c, d), axis=0)
        gts = np.transpose(gts, axes=(1, 0, 2)).astype(np.int16)

        variations = Variations()
        samples = np.array([str(i) for i in range(gts.shape[1])])
        variations.samples = da.from_array(samples)
        variations[GT_FIELD] = gts
        distances, samples = calc_kosman_dist(variations)
        expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.]
        assert np.allclose(distances, expected)


class NeiUnbiasedDistTest(unittest.TestCase):

    def test_nei_dist(self):

        gts = np.array([[[1, 1], [5, 2], [2, 2], [3, 2]],
                        [[1, 1], [1, 2], [2, 2], [2, 1]],
                        [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations = Variations()
        variations.samples = da.from_array(np.array([1, 2, 3, 4]))
        variations[GT_FIELD] = da.from_array(gts)

        pops = [[1, 2], [3, 4]]
        dists = calc_pop_pairwise_unbiased_nei_dists(variations,
                                                     max_alleles=6,
                                                     populations=pops,
                                                     silence_runtime_warnings=True,
                                                     min_num_genotypes=1)
        assert math.isclose(dists[0], 0.3726315908494797)

        # all missing
        gts = np.array([[[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])
        variations = Variations()
        variations.samples = da.from_array(np.array([1, 2, 3, 4]))
        variations[GT_FIELD] = da.from_array(gts)

        pops = [[1, 2], [3, 4]]
        dists = calc_pop_pairwise_unbiased_nei_dists(variations,
                                                     max_alleles=1,
                                                     populations=pops,
                                                     silence_runtime_warnings=True,
                                                     min_num_genotypes=1)
        assert math.isnan(dists[0])

        # min_num_genotypes
        gts = np.array([[[1, 1], [5, 2], [2, 2], [3, 2]],
                        [[1, 1], [1, 2], [2, 2], [2, 1]],
                        [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]])

        variations = Variations()
        variations.samples = da.from_array(np.array([1, 2, 3, 4]))
        variations[GT_FIELD] = da.from_array(gts)
        pops = [[1, 2], [3, 4]]
        dists = calc_pop_pairwise_unbiased_nei_dists(variations,
                                                     max_alleles=6,
                                                     populations=pops,
                                                     silence_runtime_warnings=True,
                                                     min_num_genotypes=1)
        assert math.isclose(dists[0], 0.3726315908494797)


class DsetDistTest(unittest.TestCase):

    def test_dest_jost_distance(self):
        gts = [[(1, 1), (1, 3), (1, 2), (1, 4), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [(1, 3), (1, 1), (1, 1), (1, 3), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)]]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
        dps = [[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
               [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]
        variations = Variations()
        variations.samples = da.from_array(np.array(samples))
        variations[GT_FIELD] = da.from_array(np.array(gts))
        variations[DP_FIELD] = da.from_array(np.array(dps))

        dists = calc_dset_pop_distance(variations, max_alleles=5,
                                       silence_runtime_warnings=True,
                                       populations=pops, min_num_genotypes=0)
        assert np.allclose(dists, [0.65490196])

        dists = calc_dset_pop_distance(variations, max_alleles=5,
                                       silence_runtime_warnings=True,
                                       populations=pops, min_num_genotypes=6)
        assert np.all(np.isnan(dists))

    def test_empty_pop(self):
        missing = (-1, -1)
        gts = [[(1, 1), (1, 3), (1, 2), (1, 4), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [(1, 3), (1, 1), (1, 1), (1, 3), (3, 3), (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               ]
        dps = [[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0],
               [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 0],
               [0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0]]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]

        variations = Variations()
        variations.samples = da.from_array(np.array(samples))
        variations[GT_FIELD] = da.from_array(np.array(gts))
        variations[DP_FIELD] = da.from_array(np.array(dps))

        dists = calc_dset_pop_distance(variations, max_alleles=5,
                                       silence_runtime_warnings=True,
                                       populations=pops, min_num_genotypes=0)
        assert np.allclose(dists, [0.65490196])
        gts = [[missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               [missing, missing, missing, missing, missing, (3, 2), (3, 4), (2, 2), (2, 4), (4, 4), (-1, -1)],
               ]
        dps = [[0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0],
               [0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0],
               [0, 0, 0, 0, 0, 20, 20, 20, 20, 20, 0]]
        samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        pops = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]

        variations = Variations()
        variations.samples = da.from_array(np.array(samples))
        variations[GT_FIELD] = da.from_array(np.array(gts))
        variations[DP_FIELD] = da.from_array(np.array(dps))

        dists = calc_dset_pop_distance(variations, max_alleles=5,
                                       silence_runtime_warnings=True,
                                       populations=pops, min_num_genotypes=0)
        assert np.isnan(dists[0])


if __name__ == '__main__':
    # import sys; sys.argv = ['.', 'DsetDistTest.test_empty_pop']
    unittest.main()

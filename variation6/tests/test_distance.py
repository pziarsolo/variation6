'''
Created on 2019(e)ko aza. 8(a)

@author: peio
'''
import unittest

import numpy as np
import dask.array as da

from variation6.distance import calc_kosman_dist, _kosman
from variation6.variations import Variations
from variation6 import GT_FIELD, FLT_VARS
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


if __name__ == '__main__':
#     import sys; sys.argv = ['.', 'ObsHetFiltterTest.test_obs_het_filter']
    unittest.main()

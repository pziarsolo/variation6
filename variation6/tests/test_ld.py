import numpy as np
import unittest
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr
from variation6.stats.ld import (iterate_chunk_pairs, _get_r,
                                 calc_rogers_huff_r, _calc_rogers_huff_r,
                                 calc_ld_along_genome,
                                 _calc_rogers_huff_r_for_snp_pair,
                                 calc_ld_random_pairs_from_different_chroms)
from variation6 import FLT_VARS, ALT_FIELD
from variation6.filters import remove_low_call_rate_vars, filter_by_maf
from variation6.compute import compute


class LDTest(unittest.TestCase):

    def test_iterate_chunk_pairs(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr', num_vars_per_chunk=1)
        variations = remove_low_call_rate_vars(variations, min_call_rate=0)[FLT_VARS]
        for p in iterate_chunk_pairs(variations, max_distance=100000):
            self.assertTrue(len(p), 2)

    def test_ld_calculation(self):
#         Y = [2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1]
#         Z = [2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1]

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1]
        yz_r = _get_r(Y, Z)
        yy_r = _get_r(Y, Y)
        zz_r = _get_r(Z, Z)
        # print('reference', yz_r, yy_r, zz_r)

        gt = np.array([Y, Z, Z])
        r = _calc_rogers_huff_r(gt)
        assert np.allclose(r, [yz_r, yz_r, zz_r])

        gts1 = np.array([Y, Z, Z])
        gts2 = np.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2)
        assert np.allclose(r, [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]])

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1]
        gts1 = np.array([Y, Z, Z])
        gts2 = np.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2, debug=False)
        assert np.allclose(r, [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]])

        Y = [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1,
                 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 0, 0, -1]

        Z = [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1,
                 2, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2,
                 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 0]
        gts1 = np.array([Y, Z, Z])
        gts2 = np.array([Z, Y])
        r = calc_rogers_huff_r(gts1, gts2, debug=True, min_num_gts=50)
        expected = [[yz_r, yy_r], [zz_r, yz_r], [zz_r, yz_r]]

        assert np.allclose(r, expected, atol=1e-3)

        r = calc_rogers_huff_r(gts1, gts2, debug=False, min_num_gts=51)
        expected = [[np.nan, np.nan], [zz_r, np.nan], [zz_r, np.nan]]
        assert np.allclose(r, expected, atol=1e-3, equal_nan=True)

    def xtest_calc_roger_huff_r_between_two_snps(self):
        gts_snp1 = [2, 2, 2, 2, 2, 2, 1]
        gts_snp2 = [1, 2, 2, 2, 2, 2, 2]
        result = _calc_rogers_huff_r_for_snp_pair(gts_snp1, gts_snp2, min_num_gts=2)
        print(result)

    def test_ld_genomewide(self):
        zarr_path = TEST_DATA_DIR / 'tomato.apeki_gbs.calmd.zarr'
        # vcf_to_zarr(vcf_path, zarr_path)
        variations = load_zarr(zarr_path, num_vars_per_chunk=200)

        # reduce vars to calculate
        variations = variations.get_vars(slice(0, 1000))

        max_alleles = variations[ALT_FIELD].shape[1]
        variations = filter_by_maf(variations, max_alleles=max_alleles,
                                   max_allowable_maf=0.98)[FLT_VARS]

        max_distance = 1000
        res = calc_ld_along_genome(variations, max_distance, min_num_gts=5,
                                   max_maf=0.98)
        self.assertEqual(len(list(res)), 6793)

#             print(i)
    def test_ld_random_pairs_from_different_chroms(self):
        variations = load_zarr(TEST_DATA_DIR / 'tomato.apeki_gbs.calmd.zarr',
                               num_vars_per_chunk=200)
        max_alleles = variations[ALT_FIELD].shape[1]
        variations = filter_by_maf(variations, max_alleles=max_alleles,
                                   max_allowable_maf=0.98)[FLT_VARS]

        lds = calc_ld_random_pairs_from_different_chroms(variations, 100,
                                                         max_maf=0.98,
                                                         silence_runtime_warnings=True)
        lds = list(lds)
        self.assertEqual(len(lds), 100)

    def test_ld_random_pairs_from_different_chroms_in_memory(self):
        variations = load_zarr(TEST_DATA_DIR / 'tomato.apeki_gbs.calmd.zarr',
                               num_vars_per_chunk=200)
        max_alleles = variations[ALT_FIELD].shape[1]
        variations = filter_by_maf(variations, max_alleles=max_alleles,
                                   max_allowable_maf=0.98)[FLT_VARS]
        variations = compute({'vars': variations},
                             store_variation_to_memory=True,
                             silence_runtime_warnings=True)['vars']
        lds = calc_ld_random_pairs_from_different_chroms(variations, 100,
                                                         max_maf=0.98,
                                                         silence_runtime_warnings=True)
        lds = list(lds)
        self.assertEqual(len(lds), 100)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'LDTest.test_ld_genomewide']
    unittest.main()

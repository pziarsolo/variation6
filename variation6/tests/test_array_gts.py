import unittest
import numpy as np
import variation6.array as va
from variation6.in_out.zarr import load_zarr
from variation6.tests import TEST_DATA_DIR
from variation6.filters import remove_low_call_rate_vars
from variation6 import FLT_VARS, GT_FIELD
from variation6.compute import compute


class MultivariateTest(unittest.TestCase):

    def test_gts_to_012mat(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        variations = remove_low_call_rate_vars(variations, min_call_rate=0)[FLT_VARS]
        gts012 = va.gts_as_mat012(variations[GT_FIELD])

        expected = [[-1, 0, 2], [-1, 0, 2], [-1, 0, 2],
                    [ 1, -1, 0], [-1, -1, -1], [-1, 1, -1], [-1, 1, 2]]
        self.assertTrue(np.allclose(expected, gts012.compute()))

        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        gts012 = va.gts_as_mat012(variations[GT_FIELD])
        self.assertTrue(np.allclose(expected, gts012.compute()))

        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        variations = compute({'vars': variations},
                             store_variation_to_memory=True)['vars']
        gts012 = va.gts_as_mat012(variations[GT_FIELD])
        self.assertTrue(np.allclose(expected, gts012))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

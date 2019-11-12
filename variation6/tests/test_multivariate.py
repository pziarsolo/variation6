import unittest

import numpy as np
import dask.array as da

from variation6 import FLT_VARS, GT_FIELD
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr
from variation6.stats.multivariate import gts_as_mat012, do_pca
from variation6.filters import remove_low_call_rate_vars
from variation6.variations import Variations


class MultivariateTest(unittest.TestCase):

    def test_gts_to_012mat(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        variations = remove_low_call_rate_vars(variations, min_call_rate=0)[FLT_VARS]
        gts012 = gts_as_mat012(variations)

        expected = [[-1, 0, 2], [-1, 0, 2], [-1, 0, 2],
                    [ 1, -1, 0], [-1, -1, -1], [-1, 1, -1], [-1, 1, 2]]
        self.assertTrue(np.allclose(expected, gts012.compute()))

        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        gts012 = gts_as_mat012(variations)
        self.assertTrue(np.allclose(expected, gts012.compute()))

    def test_do_pca(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        do_pca(variations)

        gts = np.array([[[0, 0], [0, 0], [1, 1]],
                        [[0, 0], [0, 0], [1, 1]],
                        [[0, 0], [0, 0], [1, 1]],
                        [[0, 0], [0, 0], [1, 1]]])
        variations = Variations()
        variations.samples = da.from_array(np.array(['a', 'b', 'c']))
        variations[GT_FIELD] = da.from_array(gts)

        res = do_pca(variations)
        projs = res['projections']
        assert projs.shape[0] == gts.shape[1]
        assert np.allclose(projs[0], projs[1])
        assert not np.allclose(projs[0], projs[2])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

import unittest

import numpy as np
import dask.array as da

from variation6 import GT_FIELD
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr
from variation6.stats.multivariate import do_pca
from variation6.variations import Variations
from variation6.compute import compute


class MultivariateTest(unittest.TestCase):

    def xtest_do_pca(self):
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

    def test_do_pca_in_memory(self):
        variations = load_zarr(TEST_DATA_DIR / 'test.zarr')
        variations = compute({'vars': variations},
                             store_variation_to_memory=True)['vars']
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

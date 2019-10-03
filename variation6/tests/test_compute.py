import numpy as np
import dask.array as da
import unittest
from variation6.tests import TEST_DATA_DIR
from variation6.in_out.zarr import load_zarr, prepare_zarr_storage
from tempfile import TemporaryDirectory
from pathlib import Path
from variation6.compute import compute
from variation6 import GT_FIELD


class ComputeTest(unittest.TestCase):

    def test_compute_vars_to_disk(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        da1 = da.from_array(np.array([1, 2, 3, 4, 5]))
        da2 = da.from_array(np.array([6, 7, 8, 9, 0]))
        da3 = da1 + da2

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            delayed_vars = prepare_zarr_storage(variations, tmp_path)
            initial = {'vars': delayed_vars,
                      'data': {'da1': da1, 'da2': da2, 'da3': da3}}
            processed = compute(initial)
            variations2 = load_zarr(tmp_path)
            self.assertTrue(np.all(variations.samples.compute() == variations2.samples.compute()))
            self.assertTrue(np.all(variations[GT_FIELD].compute() == variations2[GT_FIELD].compute()))
            self.assertTrue(np.all(processed['data']['da1'] == [1, 2, 3, 4, 5]))
            self.assertTrue(np.all(processed['data']['da3'] == [7, 9, 11, 13, 5]))

    def test_compute_vars_to_memory(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        da1 = da.from_array(np.array([1, 2, 3, 4, 5]))
        da2 = da.from_array(np.array([6, 7, 8, 9, 0]))
        da3 = da1 + da2

        initial = {'vars': variations,
                  'data': {'da1': da1, 'da2': da2, 'da3': da3}}
        processed = compute(initial, store_variation_to_memory=True)
        variations2 = processed['vars']
        self.assertTrue(np.all(variations.samples.compute() == variations2.samples))
        self.assertTrue(np.all(variations[GT_FIELD].compute() == variations2[GT_FIELD]))
        self.assertTrue(np.all(processed['data']['da1'] == [1, 2, 3, 4, 5]))
        self.assertTrue(np.all(processed['data']['da3'] == [7, 9, 11, 13, 5]))
        # if we are not storing or computing to memory, the variation
        # should be removed from the compute result
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        da1 = da.from_array(np.array([1, 2, 3, 4, 5]))
        da2 = da.from_array(np.array([6, 7, 8, 9, 0]))
        da3 = da1 + da2

        initial = {'vars': variations,
                  'data': {'da1': da1, 'da2': da2, 'da3': da3}}

        processed = compute(initial, store_variation_to_memory=False)

        self.assertNotIn('vars', processed)

        initial = {'vars': variations,
                  'data': {'da': {'da1':da1, 'da2': da2, 'da3': da3}}}

        # recursive
        processed = compute(initial, store_variation_to_memory=False)
        assert np.all(processed['data']['da']['da1'] == [1, 2, 3, 4, 5])
        assert np.all(processed['data']['da']['da2'] == [6, 7, 8, 9, 0])
        assert np.all(processed['data']['da']['da3'] == [ 7, 9, 11, 13, 5])
        self.assertNotIn('vars', processed)

        initial = {'vars': variations,
                  'data': {'d': {'da': {'da1':da1, 'da2': da2, 'da3': da3}}}}
        processed = compute(initial, store_variation_to_memory=False)
        assert np.all(processed['data']['d']['da']['da1'] == [1, 2, 3, 4, 5])
        assert np.all(processed['data']['d']['da']['da2'] == [6, 7, 8, 9, 0])
        assert np.all(processed['data']['d']['da']['da3'] == [ 7, 9, 11, 13, 5])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_compute']
    unittest.main()

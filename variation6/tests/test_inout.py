import unittest
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import dask
import numpy as np

from variation6 import GT_FIELD, QUAL_FIELD, FLT_VARS
from variation6.tests import TEST_DATA_DIR
from variation6.variations import ALLOWED_FIELDS
from variation6.filters import remove_low_call_rate_vars
from variation6.in_out.zarr import load_zarr, vcf_to_zarr, prepare_zarr_storage
from variation6.in_out.hdf5 import vcf_to_hdf5, load_hdf5, prepare_hdf5_storage


class TestVcfToZarr(unittest.TestCase):

    def test_vcf_to_zarr(self):
        with TemporaryDirectory() as tmpdir:
            vcf_path = TEST_DATA_DIR / 'freebayes5.vcf.gz'
            zarr_path = Path(tmpdir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vcf_to_zarr(vcf_path, zarr_path)
                variations = load_zarr(zarr_path)
                self.assertEqual(variations.samples.shape[0], 3)

    def test_zarr_to_variations(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        self.assertEqual(variations[GT_FIELD].shape, (7, 3, 2))


class TestZarrOut(unittest.TestCase):

    def test_save_to_zarr(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        # with this step we create a  variation with dask arrays of unknown shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            delayed_store = prepare_zarr_storage(variations, tmp_path)
            dask.compute(delayed_store)
            variations2 = load_zarr(tmp_path)
            self.assertTrue(np.all(variations.samples.compute() == variations2.samples.compute()))
            for field in ALLOWED_FIELDS:
                # dont chec
                if field == QUAL_FIELD:
                    continue
                original = variations[field]
                if original is None:
                    continue
                original = original.compute()
                new = variations2[field].compute()
                self.assertTrue(np.all(original == new))


class TestVcfTohHf5(unittest.TestCase):

    def test_vcf_to_hdf5(self):
        with NamedTemporaryFile() as tmp_fhand:
            vcf_path = TEST_DATA_DIR / 'freebayes5.vcf.gz'
            h5_path = Path(tmp_fhand.name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vcf_to_hdf5(vcf_path, h5_path)
                variations = load_hdf5(h5_path)
                self.assertEqual(variations.samples.shape[0], 3)

    def test_hdf5_to_variations(self):
        h5_path = TEST_DATA_DIR / 'test.h5'
        variations = load_hdf5(h5_path)
        self.assertEqual(variations[GT_FIELD].shape, (7, 3, 2))


class Testhdf5Out(unittest.TestCase):

    def test_save_to_hdf5(self):
        h5_path = TEST_DATA_DIR / 'test.h5'
        variations = load_hdf5(h5_path)
#         h5_path = TEST_DATA_DIR / 'test.zarr'
#         variations = load_zarr(h5_path)
        # with this step we create a  variation with dask arrays of unknown shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]

        with NamedTemporaryFile(suffix='.h5') as tmp_dir:
            tmp_path = Path(tmp_dir.name)
            delayed_store = prepare_hdf5_storage(variations, tmp_path)
            dask.compute(delayed_store)
            variations2 = load_hdf5(tmp_path)
            self.assertEqual(variations.metadata, variations2.metadata)
            self.assertTrue(np.all(variations.samples.compute() == variations2.samples.compute()))
            for field in ALLOWED_FIELDS:
                # dont chec
                if field == QUAL_FIELD:
                    continue
                original = variations[field]
                if original is None:
                    continue
                original = original.compute()
                new = variations2[field].compute()
                self.assertTrue(np.all(original == new))


if __name__ == '__main__':
    import sys; sys.argv = ['.', 'TestZarrOut']
    unittest.main()

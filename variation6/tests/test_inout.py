import unittest
import warnings

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from variation6 import GT_FIELD, QUAL_FIELD
from variation6.in_out.zarr import load_zarr, vcf_to_zarr, save_zarr

from variation6.tests import TEST_DATA_DIR
from variation6.variations import ALLOWED_FIELDS


class TestVcfToZarr(unittest.TestCase):

    def test_vcf_to_zarr(self):
        with TemporaryDirectory() as tmpdir:
            vcf_path = TEST_DATA_DIR / 'freebayes5.vcf.gz'
            zarr_path = Path(tmpdir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vcf_to_zarr(vcf_path, zarr_path)

    def test_zarr_to_variations(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)
        print(dir(variations[GT_FIELD]))
        self.assertEqual(variations[GT_FIELD].shape, (7, 3, 2))


class TestZarrOut(unittest.TestCase):

    def test_save_to_zarr(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        variations = load_zarr(zarr_path)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_zarr(variations, tmp_path)
            variations2 = load_zarr(tmp_path)
            print(variations.samples.compute(), variations2.samples.compute())
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
    unittest.main()

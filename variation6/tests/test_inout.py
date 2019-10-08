import unittest
import warnings
import re
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile

import zarr
import dask
import numpy as np
import dask.array as da

from variation6 import (GT_FIELD, QUAL_FIELD, FLT_VARS, VARIATION_FIELDS,
                        CALL_FIELDS)
from variation6.tests import TEST_DATA_DIR
from variation6.filters import remove_low_call_rate_vars
from variation6.in_out.zarr import load_zarr, vcf_to_zarr, prepare_zarr_storage
from variation6.in_out.hdf5 import vcf_to_hdf5, load_hdf5, prepare_hdf5_storage
from variation6.in_out.vcf import zarr_to_vcf


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
        variations = load_zarr(zarr_path, chunk_size=2)
        # with this step we create a  variation with dask arrays of unknown shapes
        variations = remove_low_call_rate_vars(variations, 0)[FLT_VARS]
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            delayed_store = prepare_zarr_storage(variations, tmp_path)
            dask.compute(delayed_store, scheduler='sync')
            variations2 = load_zarr(tmp_path)
            self.assertTrue(np.all(variations.samples.compute() == variations2.samples.compute()))
            for field in VARIATION_FIELDS + CALL_FIELDS:
                # dont chec
                if field == QUAL_FIELD:
                    continue
                original = variations[field]
                if original is None:
                    continue
                original = original.compute()
                new = variations2[field].compute()
                try:
                    self.assertTrue(np.all(original == new))
                except AssertionError:
                    for row in range(original.shape[0]):
                        print(row, original[row, ...], new[row, ...])
                    raise

    def test_zarr_functionament(self):
        # with shape
        np_array = np.random.randint(1, 10, size=1000)
        array = da.from_array(np_array)

        with TemporaryDirectory() as tmpdir:
            delayed = da.to_zarr(array, url=tmpdir,
                                 compute=False, component='/data')
            dask.compute(delayed)

            z_object = zarr.open_group(tmpdir, mode='r')

            assert np.all(np_array == z_object.data[:])

        # def without_shape():
        np_array = np.random.randint(1, 10, size=1000000)
        array = da.from_array(np_array)

        array = array[array > 5]

        with TemporaryDirectory() as tmpdir:
            array.compute_chunk_sizes()
            delayed = da.to_zarr(array, url=tmpdir,
                                 compute=False, component='/data')
            dask.compute(delayed)

            z_object = zarr.open_group(tmpdir, mode='r')

            assert np.all(np_array[np_array > 5] == z_object.data[:])

        # without_shape2
        np_array = np.random.randint(1, 10, size=10000000)
        array = da.from_array(np_array)

        array = array[array > 5]

        with TemporaryDirectory() as tmpdir:
            array.compute_chunk_sizes()
            delayed = da.to_zarr(array, url=tmpdir,
                                 compute=False, component='/data')
            dask.compute(delayed)

            z_object = zarr.open_group(tmpdir, mode='r')

            assert np.all(np_array[np_array > 5] == z_object.data[:])

        # write_chunks
        chunks = []

        sizes = (1, 2, 3)
        # total_size = sum(sizes)

        for i, n in enumerate(sizes):
            chunks.append(np.full(n, (i,)))
        with TemporaryDirectory() as tmpdir:
            store = zarr.DirectoryStore(tmpdir)
            root = zarr.group(store=store, overwrite=True)
            dataset = root.create_dataset('test', shape=(0,),
                                  chunks=chunks[0].shape,
                                  dtype=chunks[0].dtype)

            # offset = 0
            for chunk in chunks:
                dataset.append(chunk)


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
            for field in VARIATION_FIELDS + CALL_FIELDS:
                # dont chec
                if field == QUAL_FIELD:
                    continue
                original = variations[field]
                if original is None:
                    continue
                original = original.compute()
                new = variations2[field].compute()
                self.assertTrue(np.all(original == new))


class VcfTest(unittest.TestCase):

    def test_save_to_zarr(self):
        zarr_path = TEST_DATA_DIR / 'test.zarr'
        expected_vcf = '''##fileformat=VCFv4.2
##FORMAT=<ID=AO,Number=A,Type=Integer,Description="Alternate allele observation count">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Quality Read Depth of bases with Phred score >= 20">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=RO,Number=1,Type=Integer,Description="Reference allele observation count">
'''
        body = '''#CHROM    POS    ID    REF    ALT    QUAL    FILTER    INFO    FORMAT    pepo    mu16    upv196
CUUC00007_TC01    640    .    A    C    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    0,0:0:10:17:0,0:10    1,1:9:9:46:1,1:0
CUUC00007_TC01    656    .    A    C    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    0,0:0:9:18:0,0:9    1,1:8:8:41:1,1:0
CUUC00007_TC01    665    .    G    A    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    0,0:0:9:18:0,0:9    1,1:8:8:41:1,1:0
CUUC00025_TC01    285    .    C    G    .    .    .    GT:AO:DP:GQ:GT:RO    0,1:9:14:35:0,1:5    .:.:.:.:.:.    0,0:0:6:10:0,0:6
CUUC00027_TC01    238    .    A    G    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    .:.:.:.:.:.    .:.:.:.:.:.
CUUC00029_TC01    25    .    C    A    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    0,1:6:9:23:0,1:3    .:.:.:.:.:.
CUUC00029_TC01    34    .    A    G    .    .    .    GT:AO:DP:GQ:GT:RO    .:.:.:.:.:.    0,1:6:10:22:0,1:4    1,1:5:6:21:1,1:1
'''
        expected_vcf += re.sub(' +', '\t', body)
        with NamedTemporaryFile(mode='wb') as out_fhand:
            zarr_to_vcf(zarr_path, out_fhand, chunk_size=1)
            out_fhand.flush()
            with open(out_fhand.name, 'r') as in_fhand:
                result_vcf = in_fhand.read()
                assert expected_vcf in result_vcf


if __name__ == '__main__':
    # import sys; sys.argv = ['.', 'VcfTest']
    unittest.main()

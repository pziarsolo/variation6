import unittest
import numpy as np

from variation6.variations import Variations
from variation6 import GT_FIELD, CHROM_FIELD


class VariationsTest(unittest.TestCase):

    def test_basic_operations(self):
        variations = Variations()
        self.assertEqual(variations.num_variations, 0)
        self.assertEqual(variations.num_samples, 0)

        gts = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        # trying to add call data without samples fails
        with self.assertRaises(ValueError) as _:
            variations[GT_FIELD] = gts

        # set samples
        variations.samples = ['1', '2', '3']
        self.assertEqual(variations.num_samples, 3)

        # adding again samples fails
        with self.assertRaises(RuntimeError) as _:
            variations.samples = ['1', '2', '3']

        # add variationData
        chroms = np.array(['chr1', 'chr2', 'chr3'])
        variations[CHROM_FIELD] = chroms

        # add data with wrong shape
        with self.assertRaises(ValueError) as context:
            variations[GT_FIELD] = gts = np.array([[1, 2, 3]])
        self.assertIn('Introduced matrix shape', str(context.exception))

        with self.assertRaises(ValueError) as context:
            variations[GT_FIELD] = gts = np.array([[1, 2], [1, 2], [1, 2]])
        self.assertIn('not fit with num samples',
                       str(context.exception))

        # set gt array
        gts = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        variations[GT_FIELD] = gts
        self.assertTrue(np.array_equal(gts, variations[GT_FIELD]))
        self.assertEqual(variations.num_variations, 3)

    def test_iterate_chunks(self):
        variations = Variations()
        variations.samples = ['1', '2', '3']
        gts = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        variations[GT_FIELD] = gts
        for index, chunk in enumerate(variations.iterate_chunks(chunk_size=1)):
            assert np.all(chunk[GT_FIELD] == variations[GT_FIELD][index, :])
            assert np.all(chunk.samples == variations.samples)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import dask.array as da

import variation6.array as va


class ArrayCalculationTest(unittest.TestCase):

    def test_sum(self):
        np_array = np.array([1, 2, 3, 4, 5])
        self.assertEqual(va.sum(np_array), 15)

        da_array = da.from_array(np_array)
        task = va.sum(da_array)
        self.assertEqual(task.compute(), 15)

    def test_min(self):
        np_array = np.array([1, 2, 3, 4, 5])
        self.assertEqual(va.min(np_array), 1)

        da_array = da.from_array(np_array)
        task = va.min(da_array)
        self.assertEqual(task.compute(), 1)

    def test_max(self):
        np_array = np.array([1, 2, 3, 4, 5])
        self.assertEqual(va.max(np_array), 5)

        da_array = da.from_array(np_array)
        task = va.max(da_array)
        self.assertEqual(task.compute(), 5)

    def test_isnan(self):
        np_array = np.array([np.nan, 1, 3, 4])
        expected = np.array([True, False, False, False])
        self.assertTrue(np.all(va.isnan(np_array) == expected))

        da_array = da.from_array(np_array)
        task = va.isnan(da_array)
        self.assertTrue(np.all(task.compute() == expected))

    def test_amax(self):
        np_array = np.array([1, 2, 3, 4, 5])
        self.assertEqual(va.amax(np_array), 5)

        da_array = da.from_array(np_array)
        task = va.amax(da_array)
        self.assertEqual(task.compute(), 5)

    def test_full(self):
        self.assertTrue(np.all(va.create_full_array_in_memory((2,), 0) == np.array([0, 0])))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

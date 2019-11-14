import math
import warnings
from collections import OrderedDict

import numpy as np
import dask.array as da
from variation6 import (PUBLIC_CALL_GROUP, GT_FIELD, EmptyVariationsError,
                        DEF_CHUNK_SIZE)


class Variations:

    def __init__(self, samples=None, metadata=None):
        self._samples = None
        self.samples = samples
        self._arrays = {}

        self._metadata = {}

        self.metadata = metadata

    @property
    def ploidy(self):
        try:
            return self[GT_FIELD].shape[2]
        except (KeyError, IndexError):
            raise EmptyVariationsError('Variations without genotype data')

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if self._metadata:
            raise RuntimeError('Previous samples were present')

        if metadata is not None:
            self._metadata = metadata

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        if self._samples is not None:
            raise RuntimeError('Previous samples were present')

        if samples is None:
            self._samples = None
        else:
            if isinstance(samples, (list, tuple)):
                samples = np.array(samples)
            if not samples.size:
                raise ValueError('At least one sample is required')

            self._samples = samples

    @property
    def num_samples(self):
        if self.samples is None:
            return 0
        return self.samples.size

    @property
    def num_variations(self):
        one_path = next((x for x in self._arrays), None)
        if one_path is None:
            return 0

        one_mat = self[one_path]
        return one_mat.shape[0]

    def __setitem__(self, key, value):
        if key == 'samples':
            self.samples = value

        # we can not check by shape 0 if array is not computed.
        if (self.num_variations != 0 and not math.isnan(self.num_variations)
                and self.num_variations != value.shape[0]):
            msg = "Introduced matrix shape does not fit with already "
            msg += "addded matrices"
            raise ValueError(msg)

        if PUBLIC_CALL_GROUP in key:
            if self.num_samples == 0:
                msg = "Can not set call data if samples are not defined"
                raise ValueError(msg)
            if (not math.isnan(self.num_samples) and self.num_samples != 0
                    and value.ndim > 1 and self.num_samples != value.shape[1]):
                msg = 'Shape of the array does not fit with num samples'
                raise ValueError(msg)

        self._arrays[key] = value

    def __getitem__(self, key):
        return self._arrays.get(key)

    def __contains__(self, lookup):
        return lookup in self._arrays

    def get_vars(self, index):
        variations = Variations(samples=self.samples, metadata=self.metadata)
        for key, array in self._arrays.items():
            variations[key] = array[index, ...]
        return variations

    def keys(self):
        return self._arrays.keys()

    def items(self):
        return self._arrays.items()

    def iterate_chunks(self, chunk_size=None):
        gts = self._arrays[GT_FIELD]
        if isinstance(gts, da.Array) and np.any(np.isnan(gts.shape)):
            if chunk_size:
                msg = 'If variations is full of dask arrays with unknown '
                msg += 'shape, can not define chunk size. This is defined by '
                msg += 'chunks of each array'
                warnings.warn(msg)
            return self._iterate_chunks_of_unknown_shape_arrays()
        else:
            if chunk_size is None:
                chunk_size = DEF_CHUNK_SIZE
                chunk_size = gts.chunks[0][0]
            return self._iterate_chunks_of_known_shape_arrays(chunk_size)

    def _iterate_chunks_of_known_shape_arrays(self, chunk_size):
            chunk_indices = list(range(0, self.num_variations, chunk_size))
            for chunk_start in chunk_indices:
                index = slice(chunk_start, chunk_start + chunk_size)
                yield self.get_vars(index)

    def _iterate_chunks_of_unknown_shape_arrays(self):
        named_blocks = OrderedDict({key: array.blocks for key, array in self._arrays.items()})
        fields = named_blocks.keys()
        array_blocks = zip(*named_blocks.values())
        for array_block in array_blocks:
            variations = Variations(samples=self.samples, metadata=self.metadata)
            for field, block in zip(fields, array_block):
                variations[field] = block
            yield variations

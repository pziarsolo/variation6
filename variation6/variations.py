import math
import numpy as np

from variation6 import PUBLIC_CALL_GROUP, GT_FIELD, EmptyVariationsError

# ALLOWED_FIELDS = VARIATION_FIELDS + CALL_FIELDS


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
#         if value.shape == (0,):
#             print('data is empty')

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

    def items(self):
        return self._arrays.items()

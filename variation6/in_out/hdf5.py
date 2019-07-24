import allel
import dask.array as da

import h5py
from h5py._hl.group import Group

from variation6.variations import Variations
from variation6.in_out.zarr import (DEF_VCF_FIELDS,
                                    VARIATION_ZARR_FIELD_MAPPING,
                                    ZARR_VARIATION_FIELD_MAPPING)


def vcf_to_hdf5(vcf_path, zarr_path, fields=None):
    if fields is None:
        fields = DEF_VCF_FIELDS

    # convert our fields to allele zarr fields
    zarr_fields = [VARIATION_ZARR_FIELD_MAPPING[field] for field in fields]
    if 'samples' not in zarr_fields:
        zarr_fields.append('samples')

    allel.vcf_to_hdf5(str(vcf_path), str(zarr_path), fields=zarr_fields)


def load_hdf5(path):
    store = h5py.File(str(path), mode='r')
    samples = store['samples']
    variations = Variations(samples=da.from_array(samples,
                                                  chunks=samples.shape))
    metadata = {}
    for group_name, group in (store.items()):
        if isinstance(group, Group):
            for array_name, dataset in group.items():
                path = f'{group_name}/{array_name}'
                path = ZARR_VARIATION_FIELD_MAPPING[path]
                if dataset.attrs:
                    metadata[path] = dict(dataset.attrs.items())
                chunks = [600]
                if dataset.ndim > 1:
                    chunks.append(dataset.shape[1])
                if dataset.ndim > 2:
                    chunks.append(dataset.shape[2])
                variations[path] = da.from_array(dataset, chunks=tuple(chunks))

    variations.metadata = metadata
    return variations


def prepare_hdf5_storage(variations, out_path):
    store = h5py.File(str(out_path), mode='w')

    sources = []
    targets = []
    metadata = variations.metadata
    # samples
    # variants = root.create_group('samples', overwrite=True)
    sources.append(variations.samples)
    targets.append(LazyH5Dataset(store, path='/samples'))

    for path, array in variations.items():

        field_metadata = metadata.get(path, None)
        sources.append(array)

        path = VARIATION_ZARR_FIELD_MAPPING[path]
        targets.append(LazyH5Dataset(store, path=path, attrs=field_metadata))

    return da.store(sources, targets, compute=False)


class LazyH5Dataset():

    def __init__(self, h5, path, growing_dimension=(0,), mapper=None,
                 attrs=None):
        # create_dataset
        self._h5 = h5
        self._path = path
        self._growing_dimension = growing_dimension
        self._dataset = None
        self._mapper = mapper
        self._attrs = attrs

    def __setitem__(self, index, chunk):

        if self._dataset is None:
            if not chunk.size:
                return

            max_shape = list(chunk.shape)
            for dim in self._growing_dimension:
                max_shape[dim] = None
            max_shape = tuple(max_shape)
            # dtype = 'str' if chunk.dtype == object else chunk.dtype
            dtype = chunk.dtype
            self._dataset = self._h5.create_dataset(self._path, shape=chunk.shape,
                                                    dtype=dtype, maxshape=max_shape)
            if self._attrs is not None:
                for key, value in self._attrs.items():
                    self._dataset.attrs[key] = value

        # write to dataset
        if self._mapper is not None:
            chunk = self._mapper(chunk)

        filled_index = []
        for dim_slice, dim_size in zip(index, chunk.shape):
            start = dim_slice.start
            stop = start + dim_size
            assert dim_slice.step is None
            filled_index.append(slice(start, stop))
        self._dataset[tuple(filled_index)] = chunk

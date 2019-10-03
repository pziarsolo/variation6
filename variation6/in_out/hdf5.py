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


def _create_hdf5_dataset(h5, path, dask_array, field_metadata=None,
                      growing_dimension=(0,)):
    max_shape = list(dask_array.shape)
    for dim in growing_dimension:
        max_shape[dim] = None
        max_shape = tuple(max_shape)

    dataset = h5.create_dataset(path, shape=dask_array.shape,
                                dtype=dask_array.dtype,
                                maxshape=max_shape)
    if field_metadata is not None:
        for key, value in field_metadata.items():
            dataset.attrs[key] = value

    return dataset


def prepare_hdf5_storage(variations, out_path):
    store = h5py.File(str(out_path), mode='w')

    sources = []
    targets = []
    metadata = variations.metadata
    # samples
    samples_array = variations.samples
    samples_array.compute_chunk_sizes()

    sources.append(samples_array)
    dataset = _create_hdf5_dataset(store, '/samples', samples_array)
    targets.append(dataset)

    for path, array in variations.items():
        field_metadata = metadata.get(path, None)
        array.compute_chunk_sizes()
        sources.append(array)

        path = VARIATION_ZARR_FIELD_MAPPING[path]
        dataset = _create_hdf5_dataset(store, path, array, field_metadata)
        targets.append(dataset)

    return da.store(sources, targets, compute=False)


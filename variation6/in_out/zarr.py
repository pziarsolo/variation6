import allel
import dask.array as da
import zarr
import numcodecs
import os.path

from variation6 import (CHROM_FIELD, POS_FIELD, ID_FIELD, REF_FIELD, ALT_FIELD,
                        QUAL_FIELD, GT_FIELD, GQ_FIELD, DP_FIELD, AO_FIELD,
                        RO_FIELD, AD_FIELD, INDEX_FIELD)
from variation6.variations import Variations

DEFAULT_VARIATION_NUM_IN_CHUNK = 40000

ZARR_CHROM_FIELD_NAME = 'CHROM'
ZARR_POS_FIELD_NAME = 'POS'
ZARR_ID_FIELD_NAME = 'ID'
ZARR_REF_FIELD_NAME = 'REF'
ZARR_ALT_FIELD_NAME = 'ALT'
ZARR_QUAL_FIELD_NAME = 'QUAL'
ZARR_GT_FIELD_NAME = 'GT'
ZARR_GQ_FIELD_NAME = 'GQ'
ZARR_DP_FIELD_NAME = 'DP'
ZARR_AO_FIELD_NAME = 'AO'
ZARR_RO_FIELD_NAME = 'RO'
ZARR_AD_FIELD_NAME = 'AD'

ZARR_VARIANTS_GROUP_NAME = 'variants'
ZARR_CALL_GROUP_NAME = 'calldata'

ALLELE_ZARR_DEFINITION_MAPPINGS = {
    CHROM_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_CHROM_FIELD_NAME},
    POS_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_POS_FIELD_NAME},
    ID_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_ID_FIELD_NAME},
    REF_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_REF_FIELD_NAME},
    ALT_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_ALT_FIELD_NAME},
    QUAL_FIELD: {'group':ZARR_VARIANTS_GROUP_NAME, 'field': ZARR_QUAL_FIELD_NAME},
    GT_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_GT_FIELD_NAME},
    GQ_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_GQ_FIELD_NAME},
    DP_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_DP_FIELD_NAME},
    AO_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_AO_FIELD_NAME},
    RO_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_RO_FIELD_NAME},
    AD_FIELD: {'group':ZARR_CALL_GROUP_NAME, 'field': ZARR_AD_FIELD_NAME}
}

VARIATION_ZARR_FIELD_MAPPING = {key: f'{value["group"]}/{value["field"]}' for key, value in ALLELE_ZARR_DEFINITION_MAPPINGS.items()}
ZARR_VARIATION_FIELD_MAPPING = {value: key for key, value in VARIATION_ZARR_FIELD_MAPPING.items()}

DEF_VCF_FIELDS = list(VARIATION_ZARR_FIELD_MAPPING.keys())


def vcf_to_zarr(vcf_path, zarr_path, fields=None):
    if fields is None:
        fields = DEF_VCF_FIELDS

    # convert our fields to allele zarr fields
    zarr_fields = [VARIATION_ZARR_FIELD_MAPPING[field] for field in fields]
    if 'samples' not in zarr_fields:
        zarr_fields.append('samples')

    allel.vcf_to_zarr(str(vcf_path), str(zarr_path), fields=zarr_fields)


def load_zarr(path, chunk_size=DEFAULT_VARIATION_NUM_IN_CHUNK):
    z_object = zarr.open_group(str(path), mode='r')
    variations = Variations(samples=da.from_zarr(z_object.samples))
    metadata = {}
    for group_name, group in (z_object.groups()):
        for array_name, array in group.arrays():
            zarr_field = f'{group_name}/{array_name}'
            try:
                field = ZARR_VARIATION_FIELD_MAPPING[zarr_field]
            except KeyError:
                continue
            if array.attrs:
                metadata[field] = dict(array.attrs.items())

            chunks = (chunk_size,) + array.shape[1:]

            variations[field] = da.from_zarr(array, chunks=chunks)
    variations.metadata = metadata
    variations[INDEX_FIELD] = da.arange(0, variations.num_variations,
                                        dtype=int)

    return variations


def prepare_zarr_storage(variations, out_path):
    store = zarr.DirectoryStore(str(out_path))
    root = zarr.group(store=store, overwrite=True)
    metadata = variations.metadata
    sources = []
    targets = []

    samples_array = variations.samples
    samples_array.compute_chunk_sizes()
    sources.append(samples_array)

    object_codec = None
    if samples_array.dtype == object:
        object_codec = numcodecs.VLenUTF8()

    dataset = zarr.create(shape=samples_array.shape, path='samples', store=store,
                          dtype=samples_array.dtype, object_codec=object_codec)
    targets.append(dataset)

    variants = root.create_group(ZARR_VARIANTS_GROUP_NAME, overwrite=True)
    calls = root.create_group(ZARR_CALL_GROUP_NAME, overwrite=True)
    for field, array in variations.items():
        if field == INDEX_FIELD:
            definition = None
        else:
            definition = ALLELE_ZARR_DEFINITION_MAPPINGS[field]

        field_metadata = metadata.get(field, None)
        array = variations[field]
        if array is None:
            continue
        array.compute_chunk_sizes()
        sources.append(array)
        if definition is not None:
            group_name = definition['group']
            group = calls if group_name == ZARR_CALL_GROUP_NAME else variants
            path = os.path.sep + os.path.join(group.path, definition['field'])
        else:
            path = INDEX_FIELD

        object_codec = None
        if array.dtype == object:
            object_codec = numcodecs.VLenUTF8()
        dataset = zarr.create(shape=array.shape, path=path, store=store,
                              object_codec=object_codec, dtype=array.dtype)
        if field_metadata is not None:
            for key, value in field_metadata.items():
                dataset.attrs[key] = value

        targets.append(dataset)

    return da.store(sources, targets, compute=False)

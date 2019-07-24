import allel
import dask.array as da
import zarr
import numcodecs
import os.path

from variation6 import (CHROM_FIELD, POS_FIELD, ID_FIELD, REF_FIELD, ALT_FIELD,
                        QUAL_FIELD, GT_FIELD, GQ_FIELD, DP_FIELD, AO_FIELD,
                        RO_FIELD, AD_FIELD)
from variation6.variations import Variations

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


def load_zarr(path):
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

            variations[field] = da.from_zarr(array)
    variations.metadata = metadata
    return variations


def prepare_zarr_storage(variations, out_path):
    store = zarr.DirectoryStore(str(out_path))
    root = zarr.group(store=store, overwrite=True)
    delayed_datasets = []
    metadata = variations.metadata
    # samples
    # variants = root.create_group('samples', overwrite=True)
    dataset = da.to_zarr(variations.samples, url=str(out_path),
                         compute=False, component='/samples',
                         object_codec=numcodecs.VLenUTF8())
    delayed_datasets.append(dataset)

    variants = root.create_group(ZARR_VARIANTS_GROUP_NAME, overwrite=True)
    calls = root.create_group(ZARR_CALL_GROUP_NAME, overwrite=True)
    for field, definition in ALLELE_ZARR_DEFINITION_MAPPINGS.items():
        print(field)
        field_metadata = metadata.get(field, None)
        array = variations[field]
        if array is None:
            continue
        group_name = definition['group']
        group = calls if group_name == ZARR_CALL_GROUP_NAME else variants
        kwargs = {}
        if array.dtype == object:
            object_codec = numcodecs.VLenUTF8()
            kwargs = {'object_codec': object_codec}

        path = os.path.sep + os.path.join(group.path, definition['field'])
        dataset = da.to_zarr(array, url=str(out_path), compute=False,
                             component=path, **kwargs)

        if field_metadata:
            dataset.attrs.put(field_metadata)
        delayed_datasets.append(dataset)

    return delayed_datasets

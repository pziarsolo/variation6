import re

from functools import partial
from collections import OrderedDict

import numpy as np

from variation6.in_out.zarr import load_zarr
from variation6 import (GT_FIELD, CHROM_FIELD, POS_FIELD, ID_FIELD, REF_FIELD,
                        ALT_FIELD, QUAL_FIELD, MISSING_INT, MISSING_STR,
                        MISSING_FLOAT, DEF_CHUNK_SIZE)
from variation6.compute import compute

VCF_FORMAT = 'VCFv4.2'

GROUP_FIELD_MAPPING = {'/variations/info': 'INFO', '/calldata': 'FORMAT',
                       '/variations/filter': 'FILTER', '/other/alt': 'ALT',
                       '/other/contig': 'contig', '/other/sample': 'SAMPLE',
                       '/other/pedigree': 'PEDIGREE',
                       '/other/pedigreedb': 'pedigreeDB'}

VCF_FIELDS = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO',
              'FORMAT', 'CALLS']

str_type_regex = re.compile('\|S')
obj_type_regex = re.compile('\|O')
int_type_regex = re.compile('[\|<]i')
float_type_regex = re.compile('<f')
bool_type_regex = re.compile('\|b')
consecutive_digits_regex = re.compile('\d+')

_CACHE_WITH_SEP_MATRICES = {}


def zarr_to_vcf(zarr_path, out_fhand, vcf_format=VCF_FORMAT,
                chunk_size=DEF_CHUNK_SIZE):
    variations = load_zarr(zarr_path)
    _write_vcf_meta(variations, out_fhand, vcf_format)
    _write_vcf_header(variations, out_fhand)
    for chunk in variations.iterate_chunks(chunk_size=chunk_size):
        in_mem_chunk = compute({'vars': chunk}, store_variation_to_memory=True)['vars']
        _write_snvs(in_mem_chunk, out_fhand)


def _write_header_line(_id, record, group=None):
    required_fields = {'INFO': ['Number', 'Type', 'Description'],
                       'FILTER': ['Description'],
                       'FORMAT': ['Number', 'Type', 'Description'],
                       'ATL': ['Description']}
    if group is None:
        line = '##{}={}\n'.format(_id.strip('/'), record)
    else:
        line = '##{}=<ID={}'.format(group, _id.upper())
        for key in required_fields[group]:
            value = record[key]
            if key == 'Description':
                value = '"{}"'.format(value)
            line += ',{}={}'.format(key, value)
        line += '>\n'
    return line


def _parse_group_id(key):
    splitted_keys = key.split('/')
    group = '/'.join(splitted_keys[:-1])
    return GROUP_FIELD_MAPPING[group], splitted_keys[-1]


def _write_vcf_meta(variations, out_fhand, vcf_format):
    out_fhand.write('##fileformat={}\n'.format(vcf_format).encode())
    metadata = variations.metadata

    for key, value in metadata.items():
        if not isinstance(value, dict):
            line = _write_header_line(key, value, group=None)
            out_fhand.write(line.encode())

    for field, value in sorted(metadata.items()):
        if isinstance(value, dict) and field in variations:
            group, id_ = _parse_group_id(field)
            line = _write_header_line(id_, value, group=group)
            out_fhand.write(line.encode())


def _write_vcf_header(variations, out_fhand):
    header_items = VCF_FIELDS[:-1] + list(variations.samples.compute())
    header = '#' + '\t'.join(header_items) + '\n'
    out_fhand.write(header.encode())


def _write_snvs(variations, out_fhand):
    VCF_body_lines = _get_VCF_body_lines(variations)
    for line in VCF_body_lines:
        out_fhand.write(line + b"\n")


def _get_VCF_body_lines(variations):
    to_str_arrays = (
        (CHROM_FIELD, partial(_one_field_array_to_str_array, field_path=CHROM_FIELD)),
        (POS_FIELD, partial(_one_field_array_to_str_array, field_path=POS_FIELD)),
        (ID_FIELD, partial(_one_field_array_to_str_array, field_path=ID_FIELD)),
        (REF_FIELD, partial(_one_field_array_to_str_array, field_path=REF_FIELD)),
        (ALT_FIELD, _alt_array_to_str_array),
        (QUAL_FIELD, partial(_one_field_array_to_str_array, field_path=QUAL_FIELD)),
        ('/variations/filter', _filter_arrays_to_str_array),
        ('/variations/info', _info_arrays_to_str_array),
        ('/variations/format', _format_arrays_to_str_array),
        ('/variations/calls', _calls_arrays_to_str_array),
    )
    to_str_arrays = OrderedDict(to_str_arrays)

    VCF_body_stringified_fields = OrderedDict()
    for field_path in to_str_arrays.keys():
        VCF_body_stringified_fields[field_path] = to_str_arrays[field_path](variations)

    vcf_lines_array = _sum_str_arrays(list(VCF_body_stringified_fields.values()), sep=b'\t')

    return vcf_lines_array


def _format_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['format']:
        format_string = ':'.join(grouped_paths['format'])

        return np.full((variations.num_variations,), format_string.encode())
    else:
        return np.full((variations.num_variations,), b'.')


def _calls_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['calls']:

        call_2d_matrices = []
        for calls_path in grouped_paths['calls']:

            if variations[calls_path] is None:
                continue
            calls_data_uncollapsed = _stringify_array(variations[calls_path])
            data_id = calls_path.split('/')[-1]

            if calls_data_uncollapsed.ndim == 2:
                str_array_for_field = calls_data_uncollapsed
            elif calls_data_uncollapsed.ndim == 3:
                str_array_for_field = calls_data_uncollapsed[..., 0]
                sep = b'/' if data_id == 'GT' else b','
                for chromosome_set in range(1, calls_data_uncollapsed.shape[-1]):
                    str_array_for_field = _sum_str_arrays([str_array_for_field, calls_data_uncollapsed[..., chromosome_set]], sep)

            call_2d_matrices.append(str_array_for_field)

        field_str_array_by_sample = _sum_str_arrays(call_2d_matrices, b':')
        calls_field_data = _join_str_array_along_axis0(field_str_array_by_sample, sep=b'\t',
                                                 the_str_array_has_newlines=False)
        return np.char.replace(calls_field_data, b',.', b'')
    else:
        return np.full((variations.num_variations,), b'.')


def _info_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if not grouped_paths['info']:
        return np.full((variations.num_variations,), b'.')

    info_field_data = np.full((variations.num_variations,), b'')
    for i, info_path in enumerate(grouped_paths['info']):
        if variations[info_path] is not None:
            dtype = str(variations[info_path].dtype)
            field_data_holder = np.full((variations.num_variations,), b'.')
            field_key = info_path.split('/')[-1]

            # boolean info is translated to <id> by masking
            if 'bool' in dtype:
                if variations[info_path] is not None:
                    bool_mask = variations[info_path]
                    field_data_holder = np.full(bool_mask.shape, field_key.encode())
                    field_data_holder[~bool_mask] = '.'

            # non boolean info is preceded by <id>=
            elif 'bool' not in dtype:
                info_data = _stringify_array(variations[info_path])
                bool_mask = info_data == b'.'
                info_equals_string = field_key + '='
                field_data_holder = np.full((info_data.shape[0],), info_equals_string.encode())

                # info containing single values is retrieved
                if info_data.ndim == 1:
                    field_data_holder = _sum_str_arrays([field_data_holder, info_data])
                    field_data_holder[bool_mask] = '.'

                # info containing several values is collapsed into a coma separated string
                elif info_data.ndim == 2:
                    for index in range(0, info_data.shape[-1]):
                        info_data_slice = info_data[..., index]
                        bool_mask_slice = bool_mask[..., index]
                        info_data_slice[bool_mask_slice] = '.'
                        field_data_holder = _sum_str_arrays([field_data_holder, info_data_slice], sep=b',')
                    field_data_holder = np.char.replace(field_data_holder, b',.', b'')
                    field_data_holder = np.char.replace(field_data_holder, b'=,', b'=')

            # data is collapsed into a semicolon separated string
            if i == 0:
                info_field_data = _sum_str_arrays([info_field_data, field_data_holder])
            else:
                info_field_data = _sum_str_arrays([info_field_data, field_data_holder], sep=b';')

    # return numpy.char.replace(info_field_data, b';.', b'')
    return np.char.replace(info_field_data, b';.', b'')


def _get_group_variations_paths(variations):
    grouped_paths = {'filter': [], 'info': [], 'format': [], 'calls': []}

    if GT_FIELD in variations.keys():
        grouped_paths['format'] = ['GT']
        grouped_paths['calls'] = [GT_FIELD]
    for key in sorted(variations.keys()):
        if 'calldata' in key:
            if 'GT' not in key:
                grouped_paths['format'].append(key.split('/')[-1].upper())
                grouped_paths['calls'].append(key)
        elif 'info' in key:
            grouped_paths['info'].append(key)
        elif 'filter' in key:
            grouped_paths['filter'].append(key)
    return grouped_paths


def _filter_arrays_to_str_array(variations):
    grouped_paths = _get_group_variations_paths(variations)
    if grouped_paths['filter']:
        filter_str_arrays = []
        for filter_path in grouped_paths['filter']:
            if not variations[filter_path].any():
                continue
            filter_name = filter_path.split('/')[-1]
            filter_str_array = np.full((variations.num_variations,), b'.')
            filter_bool_array = variations[filter_path]
            filter_str_array[filter_bool_array] = filter_name
            filter_str_arrays.append(filter_str_array)

        filter_field_data = _sum_str_arrays(filter_str_arrays, sep=b';')
        filter_field_data = np.char.replace(filter_field_data, b';.', b'')
        return filter_field_data
    else:
        return np.full((variations.num_variations,), b'.')


def _sum_str_arrays(str_arrays, sep=None):
    concatenated_result = str_arrays[0]
    if sep is not None:
        sep_matrix = np.full(concatenated_result.shape, sep)
    for str_array in str_arrays[1:]:
        if sep is not None:
            concatenated_result = np.char.add(concatenated_result, sep_matrix)
        concatenated_result = np.char.add(concatenated_result, str_array)

    return concatenated_result


def _join_str_array_along_axis0(str_array, sep=None,
                                the_str_array_has_newlines=True):
    if the_str_array_has_newlines:
        raise NotImplementedError('If you want newlinex fix the implementation')

    num_snps = str_array.shape[0]
    if sep:
        shape = str_array.shape
        key = shape, sep
        if key in _CACHE_WITH_SEP_MATRICES:
            sep_matrix = _CACHE_WITH_SEP_MATRICES[key]
        else:
            sep_matrix = np.full(shape, sep)
            sep_matrix[:, -1] = b''
            _CACHE_WITH_SEP_MATRICES[key] = sep_matrix

        str_array = _sum_str_arrays([str_array, sep_matrix])

    new_line_column = np.full((str_array.shape[0], 1), b'\n')
    str_array = np.hstack((str_array, new_line_column))
    str_array_by_snp = np.array(str_array.tobytes().replace(b'\x00', b'').split(b'\n')[:-1])
    assert str_array_by_snp.shape[0] == num_snps

    return str_array_by_snp


def _alt_array_to_str_array(variations):
    if ALT_FIELD in variations:
        alt_array = variations[ALT_FIELD]
        alt_array['' == alt_array] = '.'
        alt_array = _stringify_array(alt_array)
        alt_field_data = _join_str_array_along_axis0(alt_array, sep=b',',
                                                     the_str_array_has_newlines=False)

        return np.char.replace(alt_field_data, b',.', b'')
    else:
        return np.full((variations.num_variations,), b'.')


def _one_field_array_to_str_array(variations, field_path):
    if field_path in variations.keys():
        one_field_data = _stringify_array(variations[field_path])
    else:
        one_field_data = np.full((variations.num_variations,), b'.')
    return one_field_data


def _get_str_mask_from_bool_array(bool_ndarray):
    int_mask = bool_ndarray.astype('int')
    str_mask = int_mask.astype('|S1')
    return str_mask


def _stringify_array(data):
    a = data.dtype
    data_type = data.dtype.str

    data = data[...].copy()

    # check type before casting
    if str_type_regex.match(data_type):
        stringified_data = data
        bool_mask = data == MISSING_STR.encode()
        stringified_data[bool_mask] = '.'
    elif obj_type_regex.match(data_type):
        longest = len(max(data, key=len))
        stringified_data = data.astype(f'|S{longest}')
        bool_mask = data == MISSING_STR.encode()
        stringified_data[bool_mask] = '.'
    elif int_type_regex.match(data_type):
        byte_depth = int(consecutive_digits_regex.search(data_type).group(0))
        number_length = len(str(2 ** (byte_depth * 8)))
        target_type = '|S' + str(number_length)
        int_data = data
        stringified_data = data.astype(target_type)
        stringified_data[int_data == MISSING_INT] = b'.'
    elif float_type_regex.match(data_type):
        # float data is rounded to make sure it fits a 16bytes string
        rounded_data = data.astype(np.float128).round(decimals=4)
        stringified_data = rounded_data[()].astype('|S16')
        if np.isnan(MISSING_FLOAT):
            stringified_data[np.isnan(data)] = b'.'
        else:
            raise RuntimeError('FIXME I used to work with nan as misssing float')
    elif bool_type_regex.match(data_type):
        stringified_data = _get_str_mask_from_bool_array(data)
    else:
        raise NotImplementedError(f'Could not match data type {data_type} {a}')

    return stringified_data


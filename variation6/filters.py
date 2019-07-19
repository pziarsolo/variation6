import dask.array as da
import numpy as np

from variation6 import (GT_FIELD, DP_FIELD, MISSING_INT, QUAL_FIELD,
                        PUBLIC_CALL_GROUP, N_KEPT, N_FILTERED_OUT,
                        FLT_VARS)
from variation6.variations import Variations
from variation6.stats import (calc_missing_gt, calc_maf_by_allele_count,
                              calc_mac, calc_maf_by_gt, count_alleles)


def remove_low_call_rate_vars(variations, min_call_rate, rates=True,
                              filter_id='call_rate'):
    num_missing_gts = calc_missing_gt(variations, rates=rates)['num_missing_gts']
    selected_vars = num_missing_gts >= min_call_rate
    variations = variations.get_vars(selected_vars)

    num_selected_vars = da.count_nonzero(selected_vars)
    num_filtered = da.count_nonzero(da.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}
    return {FLT_VARS: variations, filter_id: flt_stats}


def _gt_to_missing(variations, field, min_value):
    gts = variations[GT_FIELD]
    calls_setted_to_missing = variations[field] < min_value

    # as we can not slice using arrays of diferente dimensions, we need to
    # create one with same dimensions with stack
    p2 = da.stack([calls_setted_to_missing, calls_setted_to_missing], axis=2)
    gts[p2] = MISSING_INT

    variations[GT_FIELD] = gts

    return {FLT_VARS: variations}


def min_depth_gt_to_missing(variations, min_depth):
    return _gt_to_missing(variations, field=DP_FIELD, min_value=min_depth)


def min_qual_gt_to_missing(variations, min_qual):
    return _gt_to_missing(variations, field=QUAL_FIELD, min_value=min_qual)


def filter_samples(variations, samples):

    samples_in_variation = variations.samples.compute()
    sample_cols = np.array(sorted(list(samples_in_variation).index(sample) for sample in samples))

    new_variations = Variations(samples=da.from_array(samples),
                                metadata=variations.metadata)
    for field, array in variations._arrays.items():
        if PUBLIC_CALL_GROUP in field:
            array = array[:, sample_cols]
        new_variations[field] = array
    return {FLT_VARS: new_variations}


def _select_vars(variations, stats, min_allowable=None, max_allowable=None):
    selector_max = None if max_allowable is None else stats <= max_allowable
    selector_min = None if min_allowable is None else stats >= min_allowable

    if selector_max is None and selector_min is not None:
        selected_vars = selector_min
    elif selector_max is not None and selector_min is None:
        selected_vars = selector_max
    elif selector_max is not None and selector_min is not None:
        selected_vars = selector_min & selector_max
    else:
        selected_vars = _filter_no_row(variations)

    variations = variations.get_vars(selected_vars)

    num_selected_vars = da.count_nonzero(selected_vars)
    num_filtered = da.count_nonzero(da.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    return {FLT_VARS: variations, 'stats': flt_stats }


def _filter_no_row(variations):
    n_snps = variations.num_variations
    selector = da.ones((n_snps,), dtype=np.bool_)
    return selector


def filter_by_maf_by_allele_count(variations, max_allowable_maf=None, min_allowable_maf=None,
                                  filter_id='filter_by_maf_by_allele_count'):
    mafs = calc_maf_by_allele_count(variations)
    # print(compute(mafs))
    result = _select_vars(variations, mafs['mafs'], min_allowable_maf, max_allowable_maf)

    return {FLT_VARS: result[FLT_VARS], filter_id: result['stats'], 'maf': mafs}


def filter_by_maf(variations, max_allowable_maf=None, min_allowable_maf=None,
                                  filter_id='filter_by_maf'):
    mafs = calc_maf_by_gt(variations)

    result = _select_vars(variations, mafs['mafs'], min_allowable_maf,
                          max_allowable_maf)

    return {FLT_VARS: result[FLT_VARS], filter_id: result['stats'], 'maf': mafs}


def filter_by_mac(variations, max_allowable_mac=None, min_allowable_mac=None,
                  filter_id='filter_by_mac'):
    macs = calc_mac(variations)
    # print(compute(macs))

    result = _select_vars(variations, macs['macs'], min_allowable_mac, max_allowable_mac)

    return {FLT_VARS: result[FLT_VARS], filter_id: result['stats']}


def remove_non_variable_or_all_missing(variations, max_alleles,
                                   filter_id='remove_no_variable_or_all_missing'):
    gts = variations[GT_FIELD]
    some_not_missing_gts = da.any(gts != MISSING_INT, axis=2)
    selected_vars1 = da.any(some_not_missing_gts, axis=1)
    allele_counts = count_alleles(gts, max_alleles=max_alleles,
                                  count_missing=False)
    num_alleles_per_snp = da.sum(allele_counts > 0, axis=1)
    selected_vars2 = num_alleles_per_snp > 1

    selected_vars = da.logical_and(selected_vars1, selected_vars2)

    selected_variations = variations.get_vars(selected_vars)

    num_selected_vars = da.count_nonzero(selected_vars)
    num_filtered = da.count_nonzero(da.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    return {FLT_VARS: selected_variations, filter_id: flt_stats}

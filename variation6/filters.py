import sys
from collections import OrderedDict

import numpy as np

from variation6 import (GT_FIELD, DP_FIELD, MISSING_INT, QUAL_FIELD,
                        PUBLIC_CALL_GROUP, N_KEPT, N_FILTERED_OUT,
                        FLT_VARS, CHROM_FIELD, POS_FIELD,
                        MIN_NUM_GENOTYPES_FOR_POP_STAT, ALT_FIELD, FLT_STATS,
                        FLT_ID, COUNT, BIN_EDGES)
from variation6.variations import Variations
import variation6.array as va
from variation6.stats.diversity import (calc_missing_gt, calc_maf_by_allele_count,
                                        calc_mac, calc_maf_by_gt, count_alleles,
                                        calc_obs_het, DEF_NUM_BINS)
from variation6.in_out.zarr import load_zarr, prepare_zarr_storage
from variation6.compute import compute
from variation6.plot import plot_histogram


def remove_low_call_rate_vars(variations, min_call_rate, rates=True,
                              filter_id='call_rate', calc_histogram=False,
                              n_bins=DEF_NUM_BINS, limits=None):
    num_missing_gts = calc_missing_gt(variations, rates=rates)
    if rates:
        num_called = 1 - num_missing_gts
    else:
        num_called = variations.gt.shape[1] - num_missing_gts

    selected_vars = num_called >= min_call_rate
    variations = variations.get_vars(selected_vars)

    num_selected_vars = va.count_nonzero(selected_vars)
    num_filtered = va.count_nonzero(va.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    if calc_histogram:
        limits = (0, 1) if rates else (0, len(variations.num_samples))
        counts, bin_edges = va.histogram(num_called, n_bins=n_bins, limits=limits)
        flt_stats[COUNT] = counts
        flt_stats[BIN_EDGES] = bin_edges
        flt_stats['limits'] = [min_call_rate]

    return {FLT_VARS: variations, FLT_ID: filter_id, FLT_STATS:flt_stats}


def stack_in_memory(array, axis):
    return np.stack([array, array], axis)


def _gt_to_missing(variations, field, min_value):
    gts = variations[GT_FIELD]
    calls_setted_to_missing = variations[field] < min_value
    axis = 2

    def _stack_in_memory(array):
        return stack_in_memory(array, axis=axis)

    # as we can not slice using arrays of diferente dimensions, we need to
    # create one with same dimensions with stack
    p2 = va.map_blocks(_stack_in_memory, calls_setted_to_missing, dtype='i4',
                       new_axis=2)

    gts[p2] = MISSING_INT
    # va.assign_with_masking_value(gts, MISSING_INT, p2)

    variations[GT_FIELD] = gts

    return {FLT_VARS: variations}


def min_depth_gt_to_missing(variations, min_depth):
    return _gt_to_missing(variations, field=DP_FIELD, min_value=min_depth)


def min_qual_gt_to_missing(variations, min_qual):
    return _gt_to_missing(variations, field=QUAL_FIELD, min_value=min_qual)


def keep_samples(variations, samples):
    return _filter_samples(variations, samples, reverse=False)


def remove_samples(variations, samples):
    return _filter_samples(variations, samples, reverse=True)


def _filter_samples(variations, samples, reverse=False):
    samples_in_variation = va.make_sure_array_is_in_memory(variations.samples)
    sample_cols = np.array(sorted(list(samples_in_variation).index(sample) for sample in samples))
    samples = [samples_in_variation[index] for index in sample_cols]

    if reverse:
        sample_cols = [index for index in range(len(samples_in_variation)) if index not in sample_cols]
        samples = [sample for index, sample in enumerate(samples_in_variation) if index in sample_cols]

    new_variations = Variations(samples=np.array(samples),
                                metadata=variations.metadata)
    for field, array in variations._arrays.items():
        if PUBLIC_CALL_GROUP in field:
            array = array[:, sample_cols]
        new_variations[field] = array
    return {FLT_VARS: new_variations}


def _select_vars(variations, stats, min_allowable=None, max_allowable=None):
    with np.errstate(invalid='ignore'):
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

    num_selected_vars = va.count_nonzero(selected_vars)
    num_filtered = va.count_nonzero(va.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    return {FLT_VARS: variations, FLT_STATS: flt_stats }


def _filter_no_row(variations):
    n_snps = variations.num_variations
    selector = va.ones((n_snps,), dtype=np.bool_,
                       as_type_of=variations[GT_FIELD])
    return selector


def filter_by_maf_by_allele_count(variations, max_allowable_maf=None,
                                  min_allowable_maf=None,
                                  filter_id='filter_by_maf_by_allele_count',
                                  min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                  calc_histogram=False, n_bins=DEF_NUM_BINS,
                                  limits=None):
    mafs = calc_maf_by_allele_count(variations,
                                    min_num_genotypes=min_num_genotypes)
    # print(compute(mafs))
    result = _select_vars(variations, mafs, min_allowable_maf, max_allowable_maf)

    if calc_histogram:
        if limits is None:
            limits = (0, 1)
        counts, bin_edges = va.histogram(mafs, n_bins=n_bins, limits=limits)
        result[FLT_STATS][COUNT] = counts
        result[FLT_STATS][BIN_EDGES] = bin_edges
        limits = []
        if min_allowable_maf is not None:
            limits.append(min_allowable_maf)
        if max_allowable_maf is not None:
            limits.append(max_allowable_maf)
        result[FLT_STATS]['limits'] = limits

    return {FLT_VARS: result[FLT_VARS], FLT_ID: filter_id,
            FLT_STATS: result[FLT_STATS]}


def filter_by_maf(variations, max_alleles, max_allowable_maf=None,
                  min_allowable_maf=None, filter_id='filter_by_maf',
                  min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                  calc_histogram=False, n_bins=DEF_NUM_BINS, limits=None):
    mafs = calc_maf_by_gt(variations, max_alleles=max_alleles,
                          min_num_genotypes=min_num_genotypes)

    result = _select_vars(variations, mafs, min_allowable_maf,
                          max_allowable_maf)

    if calc_histogram:
        if limits is None:
            limits = (0, 1)
        counts, bin_edges = va.histogram(mafs, n_bins=n_bins, limits=limits)
        result[FLT_STATS][COUNT] = counts
        result[FLT_STATS][BIN_EDGES] = bin_edges
        limits = []
        if min_allowable_maf is not None:
            limits.append(min_allowable_maf)
        if max_allowable_maf is not None:
            limits.append(max_allowable_maf)
        result[FLT_STATS]['limits'] = limits

    return {FLT_VARS: result[FLT_VARS], FLT_ID: filter_id,
            FLT_STATS: result[FLT_STATS]}


def filter_by_mac(variations, max_alleles, max_allowable_mac=None,
                  min_allowable_mac=None, filter_id='filter_by_mac',
                  min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                  calc_histogram=False, n_bins=DEF_NUM_BINS, limits=None):
    macs = calc_mac(variations, max_alleles=max_alleles,
                    min_num_genotypes=min_num_genotypes)
    # print(compute(macs))

    result = _select_vars(variations, macs, min_allowable_mac, max_allowable_mac)

    if calc_histogram:
        if limits is None:
            limits = (0, variations.num_samples)
        counts, bin_edges = va.histogram(macs, n_bins=n_bins, limits=limits)
        result[FLT_STATS][COUNT] = counts
        result[FLT_STATS][BIN_EDGES] = bin_edges
        limits = []
        if min_allowable_mac is not None:
            limits.append(min_allowable_mac)
        if max_allowable_mac is not None:
            limits.append(max_allowable_mac)
        result[FLT_STATS]['limits'] = limits

    return {FLT_VARS: result[FLT_VARS], FLT_ID: filter_id,
            FLT_STATS: result[FLT_STATS]}


def keep_variable_variations(variations, max_alleles,
                             filter_id='variable_variations'):
    gts = variations[GT_FIELD]
    some_not_missing_gts = va.any(gts != MISSING_INT, axis=2)
    selected_vars1 = va.any(some_not_missing_gts, axis=1)
    allele_counts = count_alleles(gts, max_alleles=max_alleles,
                                  count_missing=False)
    num_alleles_per_snp = va.sum(allele_counts > 0, axis=1)
    selected_vars2 = num_alleles_per_snp > 1

    selected_vars = va.logical_and(selected_vars1, selected_vars2)

    selected_variations = variations.get_vars(selected_vars)

    num_selected_vars = va.count_nonzero(selected_vars)
    num_filtered = va.count_nonzero(va.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    return {FLT_VARS: selected_variations, FLT_ID: filter_id,
            FLT_STATS: flt_stats}


def keep_variations_in_regions(variations, regions,
                               filter_id='keep_variations_in_regions'):
    return _filter_by_snp_position(variations, regions, filter_id, reverse=False)


def remove_variations_in_regions(variations, regions,
                                 filter_id='remove_variations_in_regions'):
    return _filter_by_snp_position(variations, regions, filter_id, reverse=True)


def _select_variations_in_region(variations, regions):
    chroms = variations[CHROM_FIELD]
    poss = variations[POS_FIELD]

    in_any_region = None
    for region in regions:
        desired_chrom = region[0]
        if isinstance(desired_chrom, (tuple, list)):
            raise ValueError('Malformed region: ' + str(region))
        in_this_region = chroms[:] == desired_chrom
        if len(region) > 1:
            in_this_region = va.logical_and(in_this_region,
                                            va.logical_and(region[1] <= poss, poss < region[2]))
        if in_any_region is None:
            in_any_region = in_this_region
        else:
            in_any_region = va.logical_or(in_any_region, in_this_region)

    return in_any_region


def _filter_by_snp_position(variations, regions, filter_id, reverse=False):
    selected_vars = _select_variations_in_region(variations, regions)
    if reverse:
        selected_vars = va.logical_not(selected_vars)

    selected_variations = variations.get_vars(selected_vars)
#     print('sel', selected_variations.shape)
    num_selected_vars = va.count_nonzero(selected_vars)
    num_filtered = va.count_nonzero(va.logical_not(selected_vars))

    flt_stats = {N_KEPT: num_selected_vars, N_FILTERED_OUT: num_filtered}

    return {FLT_VARS: selected_variations, FLT_ID: filter_id,
            FLT_STATS: flt_stats}


def filter_by_obs_heterocigosis(variations, max_allowable_het=None,
                                min_allowable_het=None,
                                min_call_dp_for_het_call=None,
                                max_call_dp_for_het_call=None,
                                filter_id='obs_het',
                                min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                                calc_histogram=False, n_bins=DEF_NUM_BINS,
                                limits=None):

    obs_het = calc_obs_het(variations, min_num_genotypes=min_num_genotypes,
                           min_call_dp_for_het_call=min_call_dp_for_het_call,
                           max_call_dp_for_het_call=max_call_dp_for_het_call)

    result = _select_vars(variations, obs_het,
                          min_allowable=min_allowable_het,
                          max_allowable=max_allowable_het)
    if calc_histogram:
        if limits is None:
            limits = (0, 1)
        counts, bin_edges = va.histogram(obs_het, n_bins=n_bins, limits=limits)
        result[FLT_STATS][COUNT] = counts
        result[FLT_STATS][BIN_EDGES] = bin_edges
        limits = []
        if min_allowable_het is not None:
            limits.append(min_allowable_het)
        if max_allowable_het is not None:
            limits.append(max_allowable_het)
        result[FLT_STATS]['limits'] = limits

    return {FLT_VARS: result[FLT_VARS], FLT_ID: filter_id,
            FLT_STATS: result[FLT_STATS]}


def _reformat_task_dict(task):
#     task = {FLT_VARS: task[FLT_VARS], task[FLT_ID]: task[FLT_STATS]}
    task = {FLT_VARS: task[FLT_VARS], task[FLT_ID]: {FLT_STATS: task[FLT_STATS]}}
    task = {FLT_VARS: task[FLT_VARS], FLT_STATS: {task[FLT_ID]: task[FLT_STATS]}}
    return task


def _add_task_to_pipeline(pipeline_tasks, task):
    pipeline_tasks[FLT_VARS] = task[FLT_VARS]
    if FLT_STATS not in pipeline_tasks:
        pipeline_tasks[FLT_STATS] = OrderedDict()
    if FLT_STATS in task:
        pipeline_tasks[FLT_STATS][task[FLT_ID]] = task[FLT_STATS]


def filter_variations(in_zarr_path, out_zarr_path, samples_to_keep=None,
                      samples_to_remove=None, regions_to_remove=None,
                      regions_to_keep=None, min_call_rate=None,
                      min_dp_setter=None, remove_non_variable_snvs=None,
                      max_allowable_mac=None, max_allowable_het=None,
                      min_call_dp_for_het_call=None, verbose=True,
                      out_fhand=sys.stdout, calc_histogram=False):

    pipeline_tasks = {}
    variations = load_zarr(in_zarr_path)
    max_alleles = variations[ALT_FIELD].shape[1]
    task = {FLT_VARS: variations}

    if samples_to_keep is not None:
        task = keep_samples(task[FLT_VARS], samples_to_keep)
        _add_task_to_pipeline(pipeline_tasks, task)

    if samples_to_remove is not None:
        task = remove_samples(task[FLT_VARS], samples_to_remove)
        _add_task_to_pipeline(pipeline_tasks, task)

    if regions_to_remove is not None:
        task = remove_variations_in_regions(task[FLT_VARS], regions_to_remove)
        _add_task_to_pipeline(pipeline_tasks, task)

    if regions_to_keep is not None:
        task = keep_variations_in_regions(task[FLT_VARS], regions_to_keep)
        _add_task_to_pipeline(pipeline_tasks, task)

    if min_dp_setter is not None:
        task = min_depth_gt_to_missing(task[FLT_VARS], min_depth=min_dp_setter)
        _add_task_to_pipeline(pipeline_tasks, task)

    if remove_non_variable_snvs:
        task = keep_variable_variations(task[FLT_VARS],
                                        max_alleles=max_alleles)
        _add_task_to_pipeline(pipeline_tasks, task)

    if max_allowable_mac is not None:
        if samples_to_keep:
            max_allowable_mac = len(samples_to_keep) - max_allowable_mac
        elif samples_to_remove:
            max_allowable_mac = len(variations.samples) - \
                len(samples_to_remove) - max_allowable_mac
        else:
            max_allowable_mac = len(variations.samples) - max_allowable_mac
        task = filter_by_mac(task[FLT_VARS], max_allowable_mac=max_allowable_mac,
                             max_alleles=max_alleles,
                             calc_histogram=calc_histogram)
        _add_task_to_pipeline(pipeline_tasks, task)

    if min_call_rate:
        task = remove_low_call_rate_vars(task[FLT_VARS],
                                         min_call_rate=min_call_rate,
                                         calc_histogram=calc_histogram)
        _add_task_to_pipeline(pipeline_tasks, task)

    if max_allowable_het is not None and min_call_dp_for_het_call is not None:
        task = filter_by_obs_heterocigosis(task[FLT_VARS],
                                           max_allowable_het=max_allowable_het,
                                           min_call_dp_for_het_call=min_call_dp_for_het_call,
                                           calc_histogram=calc_histogram)
        _add_task_to_pipeline(pipeline_tasks, task)

    delayed_store = prepare_zarr_storage(task[FLT_VARS], out_zarr_path)
    pipeline_tasks[FLT_VARS] = delayed_store

    result = compute(pipeline_tasks, store_variation_to_memory=False,
                     silence_runtime_warnings=True)

    if verbose:
        for filter_id, task_result in result[FLT_STATS].items():
            if N_KEPT in task_result:
                total = task_result[N_FILTERED_OUT] + task_result[N_KEPT]
                out_fhand.write(f"Filter: {filter_id}\n")
                out_fhand.write("-" * (8 + len(filter_id)) + '\n')
                out_fhand.write(f"Processed: {total}\n")
                out_fhand.write(f"Kept vars: {task_result[N_KEPT]}\n")
                out_fhand.write(f"Filtered out: {task_result[N_FILTERED_OUT]}\n\n")

    return result


def write_histograms(result, hist_pathdir):

    for filter_id, res in result[FLT_STATS].items():
        if COUNT in res and BIN_EDGES in res:
            path = hist_pathdir / (filter_id + '.png')

            xlower, xupper = min(res[BIN_EDGES]), max(res[BIN_EDGES])
            plot_histogram(res[COUNT], res[BIN_EDGES],
                           fhand=path.open('wb'),
                           vlines=res.get('limits', None),
                           mpl_params={'set_yscale': {'args': ['log']},
                                       'set_xbound': {'kwargs': {'lower': xlower,
                                                                 'upper': xupper}}})


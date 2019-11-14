import math
import re
import numpy

import variation6.array as va
from variation6 import (GT_FIELD, MISSING_GT, AO_FIELD, MISSING_INT,
                        RO_FIELD, DP_FIELD, EmptyVariationsError,
                        MIN_NUM_GENOTYPES_FOR_POP_STAT, MISSING_VALUES,
                        ALT_FIELD, AD_FIELD)
from variation6.plot import plot_histogram
from variation6.compute import compute
from variation6.in_out.zarr import load_zarr

DEF_NUM_BINS = 40
MIN_DP_FOR_CALL_HET = 20


def _calc_histogram(vector, n_bins, limits, weights=None):
    try:
        dtype = vector.dtype
    except AttributeError:
        dtype = type(vector[0])

    missing_value = MISSING_VALUES[dtype]

    if weights is None:
        if math.isnan(missing_value):
            not_nan = ~va.isnan(vector)
        else:
            not_nan = vector != missing_value

        vector = vector[not_nan]

    if limits is None:
        limits = (va.min(vector), va.max(vector))

    try:
        result = va.histogram(vector, bins=n_bins, range=limits,
                              weights=weights)
    except ValueError as error:
        if ('parameter must be finite' in str(error) or
                re.search('autodetected range of .*finite', str(error))):
            isfinite = ~va.isinf(vector)
            vector = vector[isfinite]
            if weights is not None:
                weights = weights[isfinite]
            result = va.histogram(vector, bins=n_bins, range=limits,
                                  weights=weights)
        else:
            raise
    return result


def histogram(vector, n_bins=DEF_NUM_BINS, limits=None, weights=None):
    return _calc_histogram(vector, n_bins, limits=limits, weights=weights)


def calc_missing_gt(variations, rates=True):
    gts = variations[GT_FIELD]
    ploidy = variations.ploidy
    bool_gts = gts == MISSING_GT
    num_missing_gts = bool_gts.sum(axis=(1, 2)) / ploidy
    if rates:
        num_missing_gts = num_missing_gts / gts.shape[1]
    return num_missing_gts


def calc_maf_by_allele_count(variations,
                             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    ro = variations[RO_FIELD]
    ao = variations[AO_FIELD]

    ro[ro == MISSING_INT] = 0
    ao[ao == MISSING_INT] = 0

    ro_sum = va.sum(ro, axis=1)
    ao_sum = va.sum(ao, axis=1)

    max_ = va.sum(ao, axis=1).max(axis=1)

    sum_ = ao_sum.sum(axis=1) + ro_sum

    # we modify the max_ to update the values that are bigger in ro
    # here we have a setter that works different in numpy and dask
    va.assign_with_mask(array=max_, using=ro_sum, mask=ro_sum > max_)

    with numpy.errstate(invalid='ignore'):
        mafs = max_ / sum_

    return _mask_stats_with_few_samples(mafs, variations, min_num_genotypes)


def _count_alleles_in_memory(gts, max_alleles, count_missing=True):
    alleles = list(range(max_alleles))
    if count_missing:
        alleles += [MISSING_INT]
    counts = []
    for allele in alleles:
        gts_in_mem = allele == gts
        try:
            allele_count = va.count_nonzero(gts_in_mem, axis=(1, 2))
        except numpy.AxisError:
            raise EmptyVariationsError()
        # print(allele_count)
        counts.append(allele_count.reshape(allele_count.shape[0], 1))
    stacked = va.stack(counts, axis=2)
    return stacked.reshape(stacked.shape[0], stacked.shape[2])


def count_alleles(gts, max_alleles, count_missing=True):

    def _count_alleles(gts):
        return _count_alleles_in_memory(gts, max_alleles, count_missing=count_missing)

    chunks = va.calculate_chunks(gts)

    allele_counts_by_snp = va.map_blocks(_count_alleles, gts, chunks=chunks,
                                         drop_axis=(2,))

    return allele_counts_by_snp


def calc_maf_by_gt(variations, max_alleles,
                   min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]

    allele_counts_by_snp = count_alleles(gts, max_alleles, count_missing=False)
    max_ = va.max(allele_counts_by_snp, axis=1)
    sum_ = va.sum(allele_counts_by_snp, axis=1)

    with numpy.errstate(invalid='ignore'):
        mafs = max_ / sum_
    # return {'aa': allele_counts_by_snp}
    return _mask_stats_with_few_samples(mafs, variations, min_num_genotypes)


def _calc_mac(gts, max_alleles):
    gt_counts = count_alleles(gts, max_alleles=max_alleles)
    if gt_counts is None:
        return numpy.array([])

    missing_allele_idx = -1  # it's allways in the last position
    num_missing = numpy.copy(gt_counts[:, missing_allele_idx])
    gt_counts[:, missing_allele_idx] = 0

    max_ = va.amax(gt_counts, axis=1)

    num_samples = gts.shape[1]
    ploidy = gts.shape[2]
    num_chroms = num_samples * ploidy
    mac = num_samples - (num_chroms - num_missing - max_) / ploidy

    # we set the snps with no data to nan
    mac[max_ == 0] = numpy.nan
    return mac


def calc_mac(variations, max_alleles,
             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]
    # determine output chunks - preserve axis0; change axis1, axis2
#     chunks = (gts.chunks[0])
    chunks = None

    def _private_calc_mac(gts):
        return _calc_mac(gts, max_alleles=max_alleles)

    macs = va.map_blocks(_private_calc_mac, gts, chunks=chunks,
                                 drop_axis=(1, 2), dtype=numpy.float64)

    return _mask_stats_with_few_samples(macs, variations, min_num_genotypes)


def _call_is_hom_in_memory(gts):
    is_hom = va.create_full_array_in_memory(gts.shape[:-1], True,
                                            dtype=numpy.bool)
    for idx in range(1, gts.shape[2]):
        is_hom = va.logical_and(gts[:, :, idx] == gts[:, :, idx - 1], is_hom)
    return is_hom


def _call_is_hom(variations, is_missing=None):
    gts = variations[GT_FIELD]

    is_hom = va.map_blocks(_call_is_hom_in_memory, gts, drop_axis=2)
    if is_missing is not None:
        is_hom[is_missing] = False
    return is_hom


def _call_is_het(variations, is_missing=None):
    is_hom = _call_is_hom(variations, is_missing=is_missing)
#     if is_hom.shape[0] == 0:
#         return is_hom, is_missing
    is_het = va.logical_not(is_hom)
    if is_missing is not None:
        is_het[is_missing] = False
    return is_het


def _calc_obs_het_counts(variations, axis, min_call_dp_for_het_call,
                         max_call_dp_for_het_call=None):
    is_missing = va.any(variations[GT_FIELD] == MISSING_INT, axis=2)

    if min_call_dp_for_het_call is not None or max_call_dp_for_het_call is not None:
        dps = variations[DP_FIELD]
        if min_call_dp_for_het_call is not None:
            low_dp = dps < min_call_dp_for_het_call
            is_missing = va.logical_or(is_missing, low_dp)
        if max_call_dp_for_het_call is not None:
            high_dp = dps > max_call_dp_for_het_call
            is_missing = va.logical_or(is_missing, high_dp)
    is_het = _call_is_het(variations, is_missing=is_missing)

    return (va.sum(is_het, axis=axis),
            va.sum(va.logical_not(is_missing), axis=axis))


def calc_obs_het(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 min_call_dp_for_het_call=0, max_call_dp_for_het_call=None):

    het, called_gts = _calc_obs_het_counts(variations, axis=1,
                                           min_call_dp_for_het_call=min_call_dp_for_het_call,
                                           max_call_dp_for_het_call=max_call_dp_for_het_call)
    with numpy.errstate(invalid='ignore'):
        het = het / called_gts

    return _mask_stats_with_few_samples(het, variations, min_num_genotypes,
                                        num_called_gts=called_gts)


def calc_called_gt(variations, rates=True):

    if rates:
        missing = calc_missing_gt(variations, rates=rates)

        return 1 - missing
    else:
        ploidy = variations.ploidy
        bool_gts = variations[GT_FIELD] != MISSING_GT
        return bool_gts.sum(axis=(1, 2)) / ploidy


def calc_allele_freq_by_depth(variations):
    allele_counts = variations[AD_FIELD]
    allele_counts[allele_counts == -1] = 0
    allele_counts = va.sum(allele_counts, axis=1)
    total_counts = va.sum(allele_counts, axis=1)
    allele_freq = allele_counts / total_counts[:, None]
    return allele_freq


def calc_allele_freq(variations, max_alleles,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    gts = variations[GT_FIELD]
    if gts.shape[0] == 0:
        return va.empty_array(variations)
    allele_counts = count_alleles(gts, max_alleles, count_missing=False)

    if allele_counts is None:
        raise ValueError('No alleles, everything is missing data')
    total_counts = va.sum(allele_counts, axis=1)
    with numpy.errstate(invalid='ignore'):
        allele_freq = allele_counts / total_counts[:, None]
    allele_freq = _mask_stats_with_few_samples(
        allele_freq, variations, min_num_genotypes)
    return allele_freq


def calc_expected_het(variations, max_alleles,
                      min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    try:
        allele_freq = calc_allele_freq(variations, max_alleles=max_alleles,
                                       min_num_genotypes=min_num_genotypes)
    except ValueError:
        exp_het = va.create_not_initialized_array_in_memory((variations.num_variations,))
        exp_het[:] = numpy.nan
        return exp_het
    if allele_freq.shape[0] == 0:
        return va.empty_array(variations)

    gts = variations[GT_FIELD]
    ploidy = gts.shape[2]
    exp_het = 1 - va.sum(allele_freq ** ploidy, axis=1)

    return exp_het


def calc_unbias_expected_het(variations, max_alleles,
                             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    exp_het = calc_expected_het(variations, max_alleles=max_alleles,
                                min_num_genotypes=min_num_genotypes)

    num_called_gts = calc_called_gt(variations, rates=False)
    num_samples = num_called_gts.astype(float)
    num_samples[num_samples < min_num_genotypes] = numpy.nan

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return unbiased_exp_het


def _get_mask_for_masking_samples_with_few_gts(variations, min_num_genotypes,
                                               num_called_gts=None):
    if num_called_gts is None:
        num_called_gts = calc_called_gt(variations, rates=False)

    mask = num_called_gts < min_num_genotypes
    return mask


def _mask_stats_with_few_samples(stats, variations, min_num_genotypes,
                                 num_called_gts=None, masking_value=numpy.NaN):
    if min_num_genotypes is not None:
        mask = _get_mask_for_masking_samples_with_few_gts(variations,
                                                          min_num_genotypes,
                                                          num_called_gts=num_called_gts)

        va.assign_with_masking_value(stats, masking_value, mask)

    return stats


def calc_diversities(variations, max_alleles, min_num_genotypes,
                     min_call_dp_for_het_call=MIN_DP_FOR_CALL_HET,
                     polymorphic_threshold=0.95):
    diversities = {}

    mafs = calc_maf_by_gt(variations, max_alleles,
                          min_num_genotypes=min_num_genotypes)

    mafs_no_nan = mafs[va.logical_not(va.isnan(mafs))]

    num_variable_vars = va.sum(mafs_no_nan < 0.9999999999)

    diversities['num_variable_vars'] = num_variable_vars

    snp_is_poly = mafs_no_nan <= polymorphic_threshold
    num_poly = va.sum(snp_is_poly)
    diversities['num_polymorphic_vars'] = num_poly

    exp_het = calc_expected_het(variations, max_alleles=max_alleles,
                                min_num_genotypes=min_num_genotypes)
    diversities['exp_het'] = va.nanmean(exp_het)

    obs_het = calc_obs_het(variations,
                           min_call_dp_for_het_call=min_call_dp_for_het_call,
                           min_num_genotypes=min_num_genotypes)
    diversities['obs_het'] = va.nanmean(obs_het)
    diversities['num_total_variations'] = variations.num_variations
    return diversities


def summarize_variations(in_zarr_path, out_dir_path, draw_missin_rate=True,
                         draw_mac=True, draw_maf=True, draw_obs_het=True,
                         min_call_dp_for_het_call=MIN_DP_FOR_CALL_HET,
                         min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                         num_bins=DEF_NUM_BINS, silence_runtime_warnings=True):
    stats = {}
    variations = load_zarr(in_zarr_path)
    max_alleles = variations[ALT_FIELD].shape[1]
    num_variations = variations.num_variations
    num_samples = variations.num_samples

    if draw_missin_rate:
        _stats = calc_missing_gt(variations, rates=True)
        counts, edges = histogram(_stats, n_bins=num_bins)
        stats['missing'] = {'counts': counts, 'edges': edges}

    if draw_mac:
        _stats = calc_mac(variations, max_alleles, min_num_genotypes)
        counts, edges = histogram(_stats, n_bins=num_bins)
        stats['mac'] = {'counts': counts, 'edges': edges}

    if draw_maf:
        _stats = calc_maf_by_gt(variations, max_alleles, min_num_genotypes)
        counts, edges = histogram(_stats, n_bins=num_bins)
        stats['maf'] = {'counts': counts, 'edges': edges}

    if draw_obs_het:
        _stats = calc_obs_het(
            variations, min_num_genotypes=min_num_genotypes,
            min_call_dp_for_het_call=min_call_dp_for_het_call)
        counts, edges = histogram(_stats, n_bins=num_bins)
        stats['obs_heterocigosity'] = {'counts': counts, 'edges': edges}

    computed_stats = compute(stats,
                             silence_runtime_warnings=silence_runtime_warnings)

    for kind, stats in computed_stats.items():
        with (out_dir_path / f'{kind}.png').open('wb') as out_fhand:
            plot_histogram(stats['counts'], stats['edges'], out_fhand,
                           log_scale=True)

    with (out_dir_path / 'stats.txt').open('w') as fhand:
        fhand.write(f'STATS FOR: {in_zarr_path.name}\n')
        fhand.write('-----------' + '-' * len(in_zarr_path.name) + '\n')
        fhand.write(f'Num. variations: {num_variations}\n')
        fhand.write(f'Num. samples: {num_samples}\n')
        fhand.write('\n')

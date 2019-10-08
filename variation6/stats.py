import math
import re
import dask.array as da
import numpy as np

from variation6 import (GT_FIELD, MISSING_GT, AO_FIELD, MISSING_INT,
                        RO_FIELD, DP_FIELD, EmptyVariationsError,
                        MIN_NUM_GENOTYPES_FOR_POP_STAT, MISSING_VALUES)

DEF_NUM_BINS = 40


def _calc_histogram(vector, n_bins, limits, weights=None):
    try:
        dtype = vector.dtype
    except AttributeError:
        dtype = type(vector[0])

    missing_value = MISSING_VALUES[dtype]

    if weights is None:
        if math.isnan(missing_value):
            not_nan = ~da.isnan(vector)
        else:
            not_nan = vector != missing_value

        vector = vector[not_nan]

    if limits is None:
        limits = (da.min(vector), da.max(vector))

    try:
        result = da.histogram(vector, bins=n_bins, range=limits,
                              weights=weights)
    except ValueError as error:
        if ('parameter must be finite' in str(error) or
                re.search('autodetected range of .*finite', str(error))):
            isfinite = ~da.isinf(vector)
            vector = vector[isfinite]
            if weights is not None:
                weights = weights[isfinite]
            result = da.histogram(vector, bins=n_bins, range=limits,
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

    ro_sum = da.sum(ro, axis=1)
    ao_sum = da.sum(ao, axis=1)

    max_ = da.sum(ao, axis=1).max(axis=1)

    sum_ = ao_sum.sum(axis=1) + ro_sum

    # we modify the max_ to update the values that are bigger in ro
    max_[ro_sum > max_] = ro_sum

    mafs = max_ / sum_

    return _mask_stats_with_few_samples(mafs, variations, min_num_genotypes)


def _count_alleles_in_memory(gts, max_alleles, count_missing=True):
    alleles = list(range(max_alleles))
    if count_missing:
        alleles += [MISSING_INT]
    counts = []
    for allele in alleles:
        gts_in_mem = allele == gts
        allele_count = np.count_nonzero(gts_in_mem, axis=(1, 2))
        # print(allele_count)
        counts.append(allele_count.reshape(allele_count.shape[0], 1))
    stacked = np.stack(counts, axis=2)
    return stacked.reshape(stacked.shape[0], stacked.shape[2])


def count_alleles(gts, max_alleles, count_missing=True):

    def _count_alleles(gts):
        return _count_alleles_in_memory(gts, max_alleles, count_missing=count_missing)

    if isinstance(gts, np.ndarray):
        try:
            return _count_alleles_in_memory(gts, max_alleles, count_missing=count_missing)
        except np.AxisError:
            raise EmptyVariationsError()
    try:
        chunks = (gts.chunks[0], (1,) * len(gts.chunks[1]))
    except IndexError:
        raise EmptyVariationsError()

    allele_counts_by_snp = da.map_blocks(_count_alleles, gts, chunks=chunks,
                                         drop_axis=(2,))

    return allele_counts_by_snp


def calc_maf_by_gt(variations, max_alleles,
                   min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]

    allele_counts_by_snp = count_alleles(gts, max_alleles, count_missing=False)
    max_ = da.max(allele_counts_by_snp , axis=1)
    sum_ = da.sum(allele_counts_by_snp , axis=1)

    mafs = max_ / sum_
    # return {'aa': allele_counts_by_snp}
    return _mask_stats_with_few_samples(mafs, variations, min_num_genotypes)


def _calc_mac(gts, max_alleles):
    gt_counts = count_alleles(gts, max_alleles=max_alleles)
    if gt_counts is None:
        return np.array([])

    missing_allele_idx = -1  # it's allways in the last position
    num_missing = np.copy(gt_counts[:, missing_allele_idx])
    gt_counts[:, missing_allele_idx] = 0

    max_ = np.amax(gt_counts, axis=1)

    num_samples = gts.shape[1]
    ploidy = gts.shape[2]
    num_chroms = num_samples * ploidy
    mac = num_samples - (num_chroms - num_missing - max_) / ploidy

    # we set the snps with no data to nan
    mac[max_ == 0] = np.nan
    return mac


def calc_mac(variations, max_alleles,
             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]
    # determine output chunks - preserve axis0; change axis1, axis2
#     chunks = (gts.chunks[0])
    chunks = None

    def _private_calc_mac(gts):
        return _calc_mac(gts, max_alleles=max_alleles)

    macs = da.map_blocks(_private_calc_mac, gts, chunks=chunks,
                         drop_axis=(1, 2), dtype=np.float64)

    return _mask_stats_with_few_samples(macs, variations, min_num_genotypes)


def _call_is_hom_in_memory(gts):
    is_hom = da.full(gts.shape[:-1], True, dtype=np.bool)
    for idx in range(1, gts.shape[2]):
        is_hom = da.logical_and(gts[:, :, idx] == gts[:, :, idx - 1], is_hom)
    return is_hom


def _call_is_hom(variations, is_missing=None):
    gts = variations[GT_FIELD]

    is_hom = da.map_blocks(_call_is_hom_in_memory, gts, drop_axis=2)
    is_hom[is_missing] = False
    return is_hom


def _call_is_het(variations, is_missing=None):
    is_hom = _call_is_hom(variations, is_missing=is_missing)
#     if is_hom.shape[0] == 0:
#         return is_hom, is_missing
    is_het = da.logical_not(is_hom)
    is_het[is_missing] = False
    return is_het


def _calc_obs_het_counts(variations, axis, min_call_dp_for_het_call,
                         max_call_dp_for_het_call=None):
    is_missing = da.any(variations[GT_FIELD] == MISSING_INT, axis=2)

    if min_call_dp_for_het_call is not None or max_call_dp_for_het_call is not None:
        dps = variations[DP_FIELD]
        if min_call_dp_for_het_call is not None:
            low_dp = dps < min_call_dp_for_het_call
            is_missing = da.logical_or(is_missing, low_dp)
        if max_call_dp_for_het_call is not None:
            high_dp = dps > max_call_dp_for_het_call
            is_missing = da.logical_or(is_missing, high_dp)
    is_het = _call_is_het(variations, is_missing=is_missing)
#     if is_het.shape[0] == 0:
#         return is_het, is_missing
    return (da.sum(is_het, axis=axis),
            da.sum(da.logical_not(is_missing), axis=axis))


def calc_obs_het(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 min_call_dp_for_het_call=0, max_call_dp_for_het_call=None):

    het, called_gts = _calc_obs_het_counts(variations, axis=1,
                                           min_call_dp_for_het_call=min_call_dp_for_het_call,
                                           max_call_dp_for_het_call=max_call_dp_for_het_call)
    # To avoid problems with NaNs
    with np.errstate(invalid='ignore'):
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


def calc_allele_freq(variations, max_alleles,
                     min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    gts = variations[GT_FIELD]

    if gts.shape[0] == 0:
        return da.from_array(np.array([]))

    allele_counts = count_alleles(gts, max_alleles, count_missing=False)
    if allele_counts is None:
        raise ValueError('No alleles, everything is missing data')

    total_counts = da.sum(allele_counts, axis=1)

    allele_freq = allele_counts / total_counts[:, None]
    allele_freq = _mask_stats_with_few_samples(allele_freq, variations, min_num_genotypes)

    return allele_freq


def calc_expected_het(variations, max_alleles,
                      min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    try:
        allele_freq = calc_allele_freq(variations, max_alleles=max_alleles,
                                       min_num_genotypes=min_num_genotypes)
    except ValueError:
        exp_het = da.empty((variations.num_variations,))
        exp_het[:] = np.nan
        return exp_het
    if allele_freq.shape[0] == 0:
        return da.from_array(np.array([]))
    gts = variations[GT_FIELD]
    ploidy = gts.shape[2]
    exp_het = 1 - da.sum(allele_freq ** ploidy, axis=1)

    return exp_het


def calc_unbias_expected_het(variations, max_alleles,
                             min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):

    exp_het = calc_expected_het(variations, max_alleles=max_alleles,
                                min_num_genotypes=min_num_genotypes)

    num_called_gts = calc_called_gt(variations, rates=False)
    num_samples = num_called_gts.astype(float)
    num_samples[num_samples < min_num_genotypes] = np.nan

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return  unbiased_exp_het


def _get_mask_for_masking_samples_with_few_gts(variations, min_num_genotypes,
                                               num_called_gts=None):
    if num_called_gts is None:
        num_called_gts = calc_called_gt(variations, rates=False)
    mask = num_called_gts < min_num_genotypes
    return mask


def _mask_stats_with_few_samples(stats, variations, min_num_genotypes,
                                 num_called_gts=None, masking_value=np.NaN):
    if min_num_genotypes is not None:
        mask = _get_mask_for_masking_samples_with_few_gts(variations,
                                                          min_num_genotypes,
                                                          num_called_gts=num_called_gts)

        if (not np.any(np.isnan(stats.shape)) and
                len(stats.shape) != len(mask.shape)):
            mask = mask.reshape((stats.shape))

        stats[mask] = masking_value
    return stats


def _calc_percentaje(total, matrix_with_condition):
    return (total / matrix_with_condition.shape[0]) * 100


def calc_percentaje(*args):
    return da.map_blocks(_calc_percentaje, *args, chunks=None,
                         dtype=np.float64)


def calc_diversities(variations, max_alleles, min_num_genotypes,
                     polymorphic_threshold=0.95):
    diversities = {}

    mafs = calc_maf_by_gt(variations, max_alleles,
                          min_num_genotypes=min_num_genotypes)

    mafs_no_nan = mafs[da.logical_not(da.isnan(mafs))]

    num_variable_vars = da.sum(mafs_no_nan < 0.9999999999)

    diversities['num_variable_vars'] = num_variable_vars

    percent_variable_vars = calc_percentaje(num_variable_vars, mafs_no_nan)
    diversities['percent_variable_vars'] = percent_variable_vars

    snp_is_poly = mafs_no_nan <= polymorphic_threshold
    num_poly = da.sum(snp_is_poly)
    diversities['percent_polymorphic_vars'] = calc_percentaje(num_poly,
                                                              snp_is_poly)
    diversities['num_polymorphic_vars'] = num_poly

    exp_het = calc_expected_het(variations, max_alleles=max_alleles,
                                min_num_genotypes=min_num_genotypes)
    diversities['exp_het'] = da.nanmean(exp_het)

    obs_het = calc_obs_het(variations, min_num_genotypes=min_num_genotypes)
    diversities['obs_het'] = da.nanmean(obs_het)

    return diversities

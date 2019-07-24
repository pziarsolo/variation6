import dask.array as da
import numpy as np

from variation6 import (GT_FIELD, MISSING_GT, AO_FIELD, MISSING_INT,
                        RO_FIELD, DP_FIELD, EmptyVariationsError)

MIN_NUM_GENOTYPES_FOR_POP_STAT = 2


def calc_missing_gt(variations, rates=True):
    gts = variations[GT_FIELD]
    ploidy = variations.ploidy
    bool_gts = gts == MISSING_GT
    num_missing_gts = bool_gts.sum(axis=(1, 2)) / ploidy
    if rates:
        num_missing_gts = num_missing_gts / gts.shape[1]
    return {'num_missing_gts': num_missing_gts}


def calc_maf_by_allele_count(variations):
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

    return {'mafs': mafs}


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


def calc_maf_by_gt(variations, max_alleles):
    gts = variations[GT_FIELD]

    allele_counts_by_snp = count_alleles(gts, max_alleles, count_missing=False)
    max_ = da.max(allele_counts_by_snp , axis=1)
    sum_ = da.sum(allele_counts_by_snp , axis=1)

    mafs = max_ / sum_
    # return {'aa': allele_counts_by_snp}
    return {'mafs': mafs}  # , 'allele_counts': allele_counts_by_snp}


def _calc_mac(gts, max_alleles, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
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


def calc_mac(variations, max_alleles):
    gts = variations[GT_FIELD]
    # determine output chunks - preserve axis0; change axis1, axis2

#     chunks = (gts.chunks[0])
    chunks = None

    def _private_calc_mac(gts):
        return _calc_mac(gts, max_alleles=max_alleles)

    macs = da.map_blocks(_private_calc_mac, gts, chunks=chunks,
                         drop_axis=(1, 2), dtype=np.float64)

    return {'macs': macs}


def _call_is_hom_in_memory(gts):
    is_hom = da.full(gts.shape[:-1], True, dtype=np.bool)
    for idx in range(1, gts.shape[2]):
        is_hom = da.logical_and(gts[:, :, idx] == gts[:, :, idx - 1],
                                   is_hom)
    return is_hom


def _call_is_hom(variations, is_missing=None):
    gts = variations[GT_FIELD]

    is_hom = da.map_blocks(_call_is_hom_in_memory, gts, drop_axis=2)
    is_hom[is_missing] = False
    return is_hom


def _call_is_het(variations, min_call_dp, max_call_dp=None, is_missing=None):
    is_hom = _call_is_hom(variations, is_missing=is_missing)
#     if is_hom.shape[0] == 0:
#         return is_hom, is_missing
    is_het = da.logical_not(is_hom)
    is_het[is_missing] = False
    return is_het


def _calc_obs_het_counts(variations, axis, min_call_dp, max_call_dp=None):
    is_missing = da.any(variations[GT_FIELD] == MISSING_INT, axis=2)

    if min_call_dp or max_call_dp:
        dps = variations[DP_FIELD]
        if min_call_dp:
            low_dp = dps < min_call_dp
            is_missing = da.logical_or(is_missing, low_dp)
        if max_call_dp:
            high_dp = dps > max_call_dp
            is_missing = da.logical_or(is_missing, high_dp)
    is_het = _call_is_het(variations, min_call_dp, max_call_dp,
                          is_missing=is_missing)
#     if is_het.shape[0] == 0:
#         return is_het, is_missing
    return (da.sum(is_het, axis=axis),
            da.sum(da.logical_not(is_missing), axis=axis))


def calc_obs_het(variations, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                 min_allowable_call_dp=0, max_allowable_call_dp=None):
    het, called_gts = _calc_obs_het_counts(variations, axis=1,
                                           min_call_dp=min_allowable_call_dp,
                                           max_call_dp=max_allowable_call_dp)
    # To avoid problems with NaNs
    with np.errstate(invalid='ignore'):
        het = het / called_gts
    return {'obs_het': het}

import dask.array as da
import numpy as np

from variation6 import (GT_FIELD, MISSING_GT, AO_FIELD, MISSING_INT,
                        RO_FIELD, AD_FIELD)

MIN_NUM_GENOTYPES_FOR_POP_STAT = 2


def calc_missing_gt(variations, rates=True):
    gts = variations[GT_FIELD]
    ploidy = gts.shape[2]
    bool_gts = gts == MISSING_GT
    num_missing_gts = bool_gts.sum(axis=(1, 2)) / ploidy
    if rates:
        num_missing_gts = num_missing_gts / gts.shape[1]
    return {'num_missing_gts': num_missing_gts}


def calc_maf_by_allele_count(variations):
    if AD_FIELD in variations:
        allele_counts = variations[AD_FIELD]
    else:
        ro = variations[RO_FIELD]
        ro = ro.reshape(ro.shape[0], ro.shape[1], 1)
        allele_counts = da.concatenate([ro, variations[AO_FIELD]], axis=2)

    allele_counts[allele_counts == MISSING_INT ] = 0
    allele_counts_by_snp = da.sum(allele_counts, axis=1)

    max_ = da.max(allele_counts_by_snp, axis=1)
    sum_ = da.sum(allele_counts_by_snp, axis=1)

    mafs = max_ / sum_

    return {'mafs': mafs}


def _count_alleles_in_memory(gts, max_alleles=3, count_missing=True):
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


def count_alleles(gts, max_alleles=3, count_missing=True):

    def _count_alleles(gts):
        return _count_alleles_in_memory(gts, max_alleles, count_missing=count_missing)

    if isinstance(gts, np.ndarray):
        return _count_alleles_in_memory(gts, max_alleles, count_missing=count_missing)

    chunks = (gts.chunks[0], (1,) * len(gts.chunks[1]))

    allele_counts_by_snp = da.map_blocks(_count_alleles, gts, chunks=chunks,
                                         drop_axis=(2,))

    return allele_counts_by_snp


def calc_maf_by_gt(variations, max_alleles=3):
    gts = variations[GT_FIELD]

    allele_counts_by_snp = count_alleles(gts, max_alleles, count_missing=False)
    max_ = da.max(allele_counts_by_snp , axis=1)
    sum_ = da.sum(allele_counts_by_snp , axis=1)

    mafs = max_ / sum_
    # return {'aa': allele_counts_by_snp}
    return {'mafs': mafs}  # , 'allele_counts': allele_counts_by_snp}


def _calc_mac(gts, max_alleles=3, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
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


def calc_mac(variations, max_alleles=3):
    gts = variations[GT_FIELD]
    # determine output chunks - preserve axis0; change axis1, axis2

    chunks = (gts.chunks[0])

    def _private_calc_mac(gts):
        return _calc_mac(gts, max_alleles=max_alleles)

    macs = da.map_blocks(_private_calc_mac, gts, chunks=chunks,
                         drop_axis=(1, 2), dtype='i4')

    return {'macs': macs}

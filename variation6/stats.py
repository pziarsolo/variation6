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


def count_alleles(gts, max_alleles=3):
    alleles = list(range(max_alleles))
    counts = []
    for allele in alleles:
#         gts_in_mem = gts[:]
        gts_in_mem = allele == gts
        allele_count = np.count_nonzero(gts_in_mem, axis=(1, 2))
        counts.append(allele_count.reshape(allele_count.shape[0], 1))
    return np.stack(counts, axis=2)


def calc_maf_by_gt(variations, max_alleles=3):
    gts = variations[GT_FIELD]
    # determine output chunks - preserve axis0; change axis1, axis2
    chunks = (gts.chunks[0], (1,) * len(gts.chunks[1]), (max_alleles + 1,))

    def _count_alleles(gts):
        return count_alleles(gts, max_alleles)

#     count_alleles = partial(_count_alleles, max_alleles)
    # _count_alleles(gts, max_alleles)
    # map blocks and reduce
    allele_counts_by_snp = da.map_blocks(_count_alleles, gts, chunks=chunks).sum(axis=1, dtype='i4')

    max_ = da.max(allele_counts_by_snp, axis=1)
    sum_ = da.sum(allele_counts_by_snp, axis=1)

    mafs = max_ / sum_

    return {'mafs': mafs}

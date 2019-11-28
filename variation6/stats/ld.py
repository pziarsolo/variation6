import math
import itertools
import random

import numpy as np

import variation6.array as va
from variation6 import (CHROM_FIELD, POS_FIELD, GT_FIELD, MISSING_INT,
                        ALT_FIELD, DEF_CHUNK_SIZE)
from variation6.compute import compute
from variation6.stats.diversity import calc_maf_by_gt

DDOF = 1


def iterate_chunk_pairs(variations, max_distance, chunk_size=DEF_CHUNK_SIZE):
    chunks = list(variations.iterate_chunks(chunk_size))
    computed_chunks = {}
    for index1, chunk1 in enumerate(chunks):
        computed1 = compute({'vars': chunk1}, store_variation_to_memory=True,
                            silence_runtime_warnings=True)['vars']
        computed_chunks[index1] = computed1

        chunk1_end_pos = computed1[POS_FIELD][-1]
        chunk1_end_chrom = computed1[CHROM_FIELD][-1]

        for index, chunk2 in enumerate(chunks[index1:]):
            index2 = index1 + index
            if index2 in computed_chunks:
                computed2 = computed_chunks[index2]
            else:
                computed2 = compute({'vars': chunk2},
                                    store_variation_to_memory=True,
                                silence_runtime_warnings=True)['vars']
            if index2 != index1:
                chunk2_start_chrom = computed2[CHROM_FIELD][0]
                if chunk1_end_chrom != chunk2_start_chrom:
                    break
                chunk2_start_pos = computed2[POS_FIELD][0]
                if chunk2_start_pos - chunk1_end_pos > max_distance:
                    break

            yield computed1, computed2

        # remove from computed_chunks those with index minor than index1
        try:
            del computed_chunks[index1 - 1]
        except KeyError:
            pass


def calc_ld_along_genome(variations, max_distance, min_num_gts=10, max_maf=0.95):
    variation_pairs = iterate_chunk_pairs(variations, max_distance)
    results_by_pairs = (_calc_ld_between_variations(vars1, vars2, min_num_gts=min_num_gts, max_maf=max_maf)
                                                    for vars1, vars2 in variation_pairs)
    for result in itertools.chain.from_iterable(results_by_pairs):
        for ld, physical_dist, positions in result:
            if (np.isnan(ld) or np.isnan(physical_dist) or
                (positions[1] == positions[3] and positions[0] == positions[2]) or
                (positions[0] == positions[2] and abs(positions[3] - positions[1]) > max_distance)):
                continue
            yield ld, physical_dist, positions


def _calc_ld_between_variations(variations1, variations2, min_num_gts=10,
                                max_maf=0.95):
    max_alleles = variations1[ALT_FIELD].shape[1]
    maf1 = calc_maf_by_gt(variations1, max_alleles=max_alleles, min_num_genotypes=min_num_gts)
    maf2 = calc_maf_by_gt(variations2, max_alleles=max_alleles, min_num_genotypes=min_num_gts)

    if (np.any(np.isnan(maf1)) or np.any(maf1 > max_maf) or
        np.any(np.isnan(maf2)) or np.any(maf2 > max_maf)):
        msg = 'Not enough genotypes or MAF below allowed maximum, Rogers Huff calculations known to go wrong for very high maf'
        raise RuntimeError(msg)

    lds_for_pair = calc_rogers_huff_r(va.gts_as_mat012(variations1[GT_FIELD]),
                                      va.gts_as_mat012(variations2[GT_FIELD]),
                                      min_num_gts=min_num_gts)
    pos1 = variations1[POS_FIELD]
    pos2 = variations2[POS_FIELD]

    pos1_repeated = np.repeat(pos1, pos2.size).reshape((pos1.size, pos2.size))
    pos2_repeated = np.tile(pos2, pos1.size).reshape((pos1.size, pos2.size))
    physical_dist = np.abs(pos1_repeated - pos2_repeated).astype(float)
    assert lds_for_pair.shape == physical_dist.shape

    chrom1 = variations1[CHROM_FIELD]
    chrom2 = variations2[CHROM_FIELD]
    chrom1_repeated = np.repeat(chrom1, chrom2.size).reshape((chrom1.size, chrom2.size))
    chrom2_repeated = np.tile(chrom2, chrom1.size).reshape((chrom1.size, chrom2.size))

    physical_dist[chrom1_repeated != chrom2_repeated] = np.nan

    positions = list(zip(chrom1_repeated.flat, pos1_repeated.flat,
                         chrom2_repeated.flat, pos2_repeated.flat))
    yield zip(lds_for_pair.flat, physical_dist.flat, positions)


def calc_rogers_huff_r(gts1, gts2, min_num_gts=10, debug=False):
    if not (np.any(gts1 == MISSING_INT) or np.any(gts2 == MISSING_INT)):
        rogers_huff_r = _calc_rogers_huff_r2_no_nans(gts1, gts2, debug=debug)
    else:
        rogers_huff_r = np.empty((gts1.shape[0], gts2.shape[0]),
                                 dtype=np.float16)
        for idx1, gts1_snp_gts in enumerate(gts1):
            for idx2, gts2_snp_gts in enumerate(gts2):
                result = _calc_rogers_huff_r_for_snp_pair(gts1_snp_gts,
                                                          gts2_snp_gts,
                                                          min_num_gts=min_num_gts)

                rogers_huff_r[idx1, idx2] = result
    rogers_huff_r = np.abs(rogers_huff_r)
    return rogers_huff_r


def _calc_rogers_huff_r2_no_nans(gts1, gts2, debug=False):
    # means = numpy.nanmean(gts, axis=1)
    # var = numpy.nanvar(gts, axis=1)

    covars = np.cov(gts1, gts2, ddof=DDOF)
    n_vars1 = gts1.shape[0]
    n_vars2 = gts2.shape[0]
    if debug:
        print('nvars', n_vars1, n_vars2)
    variances = np.diag(covars)
    vars1 = variances[:n_vars1]
    vars2 = variances[n_vars1:]
    if debug:
        print('vars1', vars1)
        print('vars2', vars2)

    covars = covars[:n_vars1, n_vars1:]
    if debug:
        print('covars', covars)

    vars1 = np.repeat(vars1, n_vars2).reshape((n_vars1, n_vars2))
    vars2 = np.tile(vars2, n_vars1).reshape((n_vars1, n_vars2))
    with np.errstate(divide='ignore', invalid='ignore'):
        rogers_huff_r = covars / np.sqrt(vars1 * vars2)
    # print(vars1)
    # print(vars2)
    return rogers_huff_r


def _calc_rogers_huff_r_for_snp_pair(gts_snp1, gts_snp2, min_num_gts=10):
    with np.errstate(invalid='ignore', divide='ignore'):
        gts = np.array([gts_snp1, gts_snp2])

        rows_with_no_missing = np.logical_not((gts == MISSING_INT).any(axis=0))
        gts = gts[:, rows_with_no_missing]
        if gts.shape[1] < min_num_gts:
            result = np.nan
        else:
            covar = np.cov(gts, ddof=DDOF)
            variances = np.diag(covar)
            covar = covar[0, 1]
            denom = np.sqrt(variances[0] * variances[1])
            if math.isclose(denom, 0):
                result = np.nan
            else:
                result = covar / denom
        return result


def _bivmom(vec0, vec1):
    """
    Calculate means, variances, the covariance, from two data vectors.
    On entry, vec0 and vec1 should be vectors of numeric values and
    should have the same length.  Function returns m0, v0, m1, v1,
    cov, where m0 and m1 are the means of vec0 and vec1, v0 and v1 are
    the variances, and cov is the covariance.
    """
    m0 = m1 = v0 = v1 = cov = 0
    for x, y in zip(vec0, vec1):
        m0 += x
        m1 += y
        v0 += x * x
        v1 += y * y
        cov += x * y
    n = len(vec0)
    assert n == len(vec1)
    n = float(n)
    m0 /= n
    m1 /= n
    v0 /= n
    v1 /= n
    cov /= n

    cov -= m0 * m1
    v0 -= m0 * m0
    v1 -= m1 * m1

    return m0, v0, m1, v1, cov


def _get_r(Y, Z, debug=False):
    """
    Estimates r w/o info on gametic phase.  Also works with gametic
    data, in which case Y and Z should be vectors of 0/1 indicator
    variables.
    Uses the method of Rogers and Huff 2008.
    """
    _, vY, __, vZ, cov = _bivmom(Y, Z)  # _=mY, __=mZ
    if debug:
        print('cov', cov)
        print('vY', vY)
        print('vZ', vZ)
    return cov / math.sqrt(vY * vZ)


def _calc_rogers_huff_r(gts, debug=False):
    # means = numpy.nanmean(gts, axis=1)
    # var = numpy.nanvar(gts, axis=1)
    covar = np.cov(gts, ddof=DDOF)
    variances = np.diag(covar)
    covar_indices = np.tril_indices(covar.shape[0], -1)
    covars = covar[covar_indices]
    if debug:
        print(covar)
        print('vars:', variances)
        print(covar_indices)
        print('covars:', covars)
    vars1 = variances[covar_indices[0]]
    vars2 = variances[covar_indices[1]]
    rogers_huff_r = covars / np.sqrt(vars1 * vars2)
    if debug:
        print('r', rogers_huff_r)
    return rogers_huff_r


def calc_ld_random_pairs_from_different_chroms(variations, num_pairs,
                                               max_maf=0.95, min_num_gts=10,
                                               silence_runtime_warnings=False):
    chroms = va.make_sure_array_is_in_memory(variations[CHROM_FIELD],
        silence_runtime_warnings=silence_runtime_warnings)

    different_chroms = np.unique(chroms)
    if different_chroms.size < 2:
        raise ValueError('Only one chrom in variations')
    max_alleles = variations[ALT_FIELD].shape[1]

    mafs = calc_maf_by_gt(variations, max_alleles, min_num_gts)
    mafs = va.make_sure_array_is_in_memory(mafs,
        silence_runtime_warnings=silence_runtime_warnings)

    if va.any(va.isnan(mafs)) or va.any(mafs > max_maf):
        msg = 'Not enough genotypes or MAF below allowed maximum, Rogers Huff calculations known to go wrong for very high maf'
        raise RuntimeError(msg)

    gts = va.make_sure_array_is_in_memory(variations[GT_FIELD],
        silence_runtime_warnings=silence_runtime_warnings)

    num_variations = gts.shape[0]

    pairs_computed = 0
    while True:
        snp_idx1 = random.randrange(num_variations)
        snp_idx2 = random.randrange(num_variations)
        chrom1 = chroms[snp_idx1]
        chrom2 = chroms[snp_idx2]
        if chrom1 == chrom2:
            continue

        gts_snp1 = gts[snp_idx1]
        gts_snp2 = gts[snp_idx2]
        r2_ld = _calc_rogers_huff_r_for_snp_pair(gts_snp1, gts_snp2,
                                                 min_num_gts=min_num_gts)
        if not math.isnan(r2_ld):
            yield chrom1, snp_idx1, chrom2, snp_idx2, r2_ld
            pairs_computed += 1

        if pairs_computed >= num_pairs:
            break

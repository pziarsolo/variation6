import math
from itertools import combinations
from collections import OrderedDict

import dask.array as da
import numpy as np

from variation6 import GT_FIELD, FLT_VARS, MIN_NUM_GENOTYPES_FOR_POP_STAT
from variation6.filters import keep_samples
from variation6.stats import calc_missing_gt, calc_allele_freq
from variation6.compute import compute


def calc_kosman_dist(variations, min_num_snps=None):
    variations_by_sample = OrderedDict()
    samples = variations.samples.compute()
    for sample in samples:
        variations_by_sample[sample] = keep_samples(variations, [sample])[FLT_VARS]

    sample_combinations = combinations(samples, 2)

    distances_by_pair = OrderedDict()
    for sample1, sample2 in sample_combinations:
        vars1 = variations_by_sample[sample1]
        vars2 = variations_by_sample[sample2]

        snp_by_snp_comparation_array = _kosman(vars1, vars2)
        distances_by_pair[(sample1, sample2)] = snp_by_snp_comparation_array

    computed_distances_by_pair = compute(distances_by_pair)

    distances = []
    for sample_index, sample in enumerate(samples):
        starting_index2 = sample_index + 1
        if starting_index2 >= len(samples):
            break
        for sample2 in samples[starting_index2:]:
            result = computed_distances_by_pair[(sample, sample2)]
            n_snps = result.shape[0]

            if min_num_snps is not None and n_snps < min_num_snps:
                value = 0.0
            else:
                with np.errstate(invalid='ignore'):
                    value = np.sum(result) / result.shape[0]
            distances.append(value)
    return distances, samples


def _get_gts_non_missing_in_both(vars1, vars2):
    num_missing_gts1 = calc_missing_gt(vars1, rates=True)
    num_missing_gts2 = calc_missing_gt(vars2, rates=True)

    is_called = da.logical_not(da.logical_or(num_missing_gts1, num_missing_gts2))

    gts1 = vars1[GT_FIELD]
    gts2 = vars2[GT_FIELD]

    gts1 = gts1[is_called]
    gts2 = gts2[is_called]
    indi1 = gts1[:, 0]
    indi2 = gts2[:, 0]

    return indi1, indi2


def _kosman(vars1, vars2):
    indi1, indi2 = _get_gts_non_missing_in_both(vars1, vars2)

    if indi1.shape[1] != 2:
        raise ValueError('Only diploid are allowed')

    alleles_comparison1 = indi1 == indi2.transpose()[:, :, None]
    alleles_comparison2 = indi2 == indi1.transpose()[:, :, None]

    result = da.add(da.any(alleles_comparison2, axis=2).sum(axis=0),
                    da.any(alleles_comparison1, axis=2).sum(axis=0))
    result[result == 0] = 1
    result[result == 4] = 0
    result[da.logical_and(result != 1, result != 0)] = 0.5
    return result


def calc_pop_pairwise_unbiased_nei_dists(variations, max_alleles, populations,
                                         silence_runtime_warnings=False,
                                         min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    pop_ids = list(range(len(populations)))
    variations_per_pop = [keep_samples(variations, pop_samples)[FLT_VARS]
                          for pop_samples in populations]

    Jxy = {}
    uJx = {}
    uJy = {}
    for pop_id1, pop_id2 in combinations(pop_ids, 2):
        if pop_id1 not in Jxy:
            Jxy[pop_id1] = {}
        if pop_id1 not in uJx:
            uJx[pop_id1] = {}
        if pop_id1 not in uJy:
            uJy[pop_id1] = {}

        Jxy[pop_id1][pop_id2] = None
        uJx[pop_id1][pop_id2] = None
        uJy[pop_id1][pop_id2] = None

    for pop_id1, pop_id2 in combinations(pop_ids, 2):
        vars_for_pop1 = variations_per_pop[pop_id1]
        vars_for_pop2 = variations_per_pop[pop_id2]
        _accumulate_j_stats(vars_for_pop1, vars_for_pop2, max_alleles,
                            Jxy, uJx, uJy, pop_id1, pop_id2,
                            min_num_genotypes=min_num_genotypes)
    computed_result = compute({'Jxy': Jxy, 'uJx':uJx, 'uJy':uJy},
                              silence_runtime_warnings=silence_runtime_warnings)

    computedJxy = computed_result['Jxy']
    computeduJx = computed_result['uJx']
    computeduJy = computed_result['uJy']

    n_pops = len(populations)
    dists = np.empty(int((n_pops ** 2 - n_pops) / 2))
    dists[:] = np.nan
    for idx, (pop_id1, pop_id2) in enumerate(combinations(pop_ids, 2)):
        if Jxy[pop_id1][pop_id2] is None:
            unbiased_nei_identity = math.nan
        else:
            with np.errstate(invalid='ignore'):
                unbiased_nei_identity = computedJxy[pop_id1][pop_id2] / math.sqrt(computeduJx[pop_id1][pop_id2] * computeduJy[pop_id1][pop_id2])
        nei_unbiased_distance = -math.log(unbiased_nei_identity)
        if nei_unbiased_distance < 0:
            nei_unbiased_distance = 0
        dists[idx] = nei_unbiased_distance
    return dists


def _accumulate_j_stats(variations1, variations2, max_alleles,
                        Jxy, uJx, uJy, pop_name1,
                        pop_name2, min_num_genotypes=None):
    res = _calc_j_stats_per_locus(variations1, variations2, max_alleles,
                                  min_num_genotypes=min_num_genotypes)
    xUb_per_locus, yUb_per_locus, Jxy_per_locus = res
    # print('per locus')
    # print(xUb_per_locus, yUb_per_locus, Jxy_per_locus)

    if xUb_per_locus is None:
        return

    # sum over all loci
    if Jxy[pop_name1][pop_name2] is None:
        Jxy[pop_name1][pop_name2] = da.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] = da.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] = da.nansum(yUb_per_locus)
    else:
        Jxy[pop_name1][pop_name2] += da.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] += da.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] += da.nansum(yUb_per_locus)


def _calc_j_stats_per_locus(variations1, variations2, max_alleles,
                            min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT):
    res = _calc_allele_freq_and_unbiased_J_per_locus(variations1,
                                                     max_alleles=max_alleles,
                                                     min_num_genotypes=min_num_genotypes)
    allele_freq1, xUb_per_locus = res

    res = _calc_allele_freq_and_unbiased_J_per_locus(variations2,
                                                     max_alleles=max_alleles,
                                                     min_num_genotypes=min_num_genotypes)
    allele_freq2, yUb_per_locus = res

    if allele_freq2 is None or allele_freq1 is None:
        return None, None, None

    Jxy_per_locus = da.sum(allele_freq1 * allele_freq2, axis=1)

    return xUb_per_locus, yUb_per_locus, Jxy_per_locus


def _calc_allele_freq_and_unbiased_J_per_locus(variations, max_alleles,
                                               min_num_genotypes):
    try:
        allele_freq = calc_allele_freq(variations, max_alleles=max_alleles,
                                       min_num_genotypes=min_num_genotypes)
    except ValueError:
        allele_freq = None
        xUb_per_locus = None

    if allele_freq is not None:
        n_indi = variations[GT_FIELD].shape[1]
        xUb_per_locus = ((2 * n_indi * da.sum(allele_freq ** 2, axis=1)) - 1) / (2 * n_indi - 1)

    return allele_freq, xUb_per_locus

import math
from itertools import combinations
from collections import OrderedDict

import numpy as np
import variation6.array as va

from variation6 import GT_FIELD, FLT_VARS, MIN_NUM_GENOTYPES_FOR_POP_STAT
from variation6.filters import keep_samples
from variation6.stats.diversity import (calc_missing_gt, calc_allele_freq,
                                        calc_allele_freq_by_depth,
                                        _calc_obs_het_counts)
from variation6.compute import compute


def calc_kosman_dist(variations, min_num_snps=None,
                     silence_runtime_warning=False):
    variations_by_sample = OrderedDict()

    samples = va.make_sure_array_is_in_memory(variations.samples,
        silence_runtime_warnings=silence_runtime_warning)
    for sample in samples:
        variations_by_sample[sample] = keep_samples(variations, [sample])[FLT_VARS]

    sample_combinations = combinations(samples, 2)

    distances_by_pair = OrderedDict()
    for sample1, sample2 in sample_combinations:
        vars1 = variations_by_sample[sample1]
        vars2 = variations_by_sample[sample2]

        snp_by_snp_comparation_array = _kosman(vars1, vars2)
        distances_by_pair[(sample1, sample2)] = snp_by_snp_comparation_array

    computed_distances_by_pair = compute(distances_by_pair,
                                         silence_runtime_warnings=silence_runtime_warning)

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

    is_called = va.logical_not(va.logical_or(num_missing_gts1, num_missing_gts2))

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

    result = va.add(va.any(alleles_comparison2, axis=2).sum(axis=0),
                    va.any(alleles_comparison1, axis=2).sum(axis=0),
                    dtype=np.float64)

    result[result == 0] = 1
    result[result == 4] = 0

    mask = va.logical_and(result != 1, result != 0)
    result[mask] = 0.5

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
        Jxy[pop_name1][pop_name2] = va.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] = va.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] = va.nansum(yUb_per_locus)
    else:
        Jxy[pop_name1][pop_name2] += va.nansum(Jxy_per_locus)
        uJx[pop_name1][pop_name2] += va.nansum(xUb_per_locus)
        uJy[pop_name1][pop_name2] += va.nansum(yUb_per_locus)


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

    Jxy_per_locus = va.sum(allele_freq1 * allele_freq2, axis=1)

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
        xUb_per_locus = ((2 * n_indi * va.sum(allele_freq ** 2, axis=1)) - 1) / (2 * n_indi - 1)

    return allele_freq, xUb_per_locus


def calc_pop_pairwise_nei_dists_by_depth(variations, populations,
                                         silence_runtime_warnings=False):

    variations_per_pop = [keep_samples(variations, pop_samples)[FLT_VARS]
                          for pop_samples in populations]

    jxy = {}
    jxx = {}
    jyy = {}
    for pop_i, pop_j in combinations(range(len(populations)), 2):
        pop_i_vars = variations_per_pop[pop_i]
        pop_j_vars = variations_per_pop[pop_j]

        freq_al_i = calc_allele_freq_by_depth(pop_i_vars)
        freq_al_j = calc_allele_freq_by_depth(pop_j_vars)

        chunk_jxy = va.nansum(freq_al_i * freq_al_j)
        chunk_jxx = va.nansum(freq_al_i ** 2)
        chunk_jyy = va.nansum(freq_al_j ** 2)

        pop_idx = pop_i, pop_j
        if pop_idx not in jxy:
            jxy[pop_idx] = 0
            jxx[pop_idx] = 0
            jyy[pop_idx] = 0

        # The real Jxy is usually divided by num_snps, but it does not
        # not matter for the calculation
        jxy[pop_idx] += chunk_jxy
        jxx[pop_idx] += chunk_jxx
        jyy[pop_idx] += chunk_jyy

    computed_result = compute({'jxy': jxy, 'jxx':jxx, 'uJy':jyy},
                              silence_runtime_warnings=silence_runtime_warnings)

    computedjxy = computed_result['jxy']
    computedjxx = computed_result['jxx']
    computedjyy = computed_result['jyy']

    n_pops = len(populations)
    dists = np.zeros(int((n_pops ** 2 - n_pops) / 2))
    index = 0
    for pop_idx in combinations(range(len(populations)), 2):
        pjxy = computedjxy[pop_idx]
        pjxx = computedjxx[pop_idx]
        pjyy = computedjyy[pop_idx]

        try:
            nei = math.log(pjxy / math.sqrt(pjxx * pjyy))
            if nei != 0:
                nei = -nei
        except ValueError:
            nei = float('inf')

        dists[index] = nei
        index += 1

    return dists


def calc_dset_pop_distance(variations, max_alleles, populations,
                           min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
                           min_call_dp_for_het=0, silence_runtime_warnings=False):
    '''This is an implementation of the formulas proposed in GenAlex'''
    pop_ids = list(range(len(populations)))
    variations_per_pop = [keep_samples(variations, pop_samples)[FLT_VARS]
                          for pop_samples in populations]

    accumulated_dists = {}
    accumulated_hs = {}
    accumulated_ht = {}
    num_vars = {}

    for pop_id1, pop_id2 in combinations(pop_ids, 2):
        vars_for_pop1 = variations_per_pop[pop_id1]
        vars_for_pop2 = variations_per_pop[pop_id2]

        res = _calc_pairwise_dest(vars_for_pop1, vars_for_pop2,
                                  max_alleles=max_alleles,
                                  min_call_dp_for_het=min_call_dp_for_het,
                                  min_num_genotypes=min_num_genotypes)

        res['corrected_hs']
        res['corrected_ht']
        num_vars_in_chunk = va.count_nonzero(~va.isnan(res['corrected_hs']))

        hs_in_chunk = va.nansum(res['corrected_hs'])
        ht_in_chunk = va.nansum(res['corrected_ht'])

        key = (pop_id1, pop_id2)
        if key in accumulated_dists:
            accumulated_hs[key] += hs_in_chunk
            accumulated_ht[key] += ht_in_chunk
            num_vars[key] += num_vars_in_chunk
        else:
            accumulated_hs[key] = hs_in_chunk
            accumulated_ht[key] = ht_in_chunk
            num_vars[key] = num_vars_in_chunk

    task = {'accumulated_hs': accumulated_hs,
            'accumulated_ht':accumulated_ht,
            'num_vars': num_vars}

    result = compute(task, silence_runtime_warnings=silence_runtime_warnings)
    computed_accumulated_hs = result['accumulated_hs']
    computed_accumulated_ht = result['accumulated_ht']
    computed_num_vars = result['num_vars']

    tot_n_pops = len(populations)
    dists = np.empty(int((tot_n_pops ** 2 - tot_n_pops) / 2))
    dists[:] = np.nan
    num_pops = 2
    for idx, (pop_id1, pop_id2) in enumerate(combinations(pop_ids, 2)):
        key = pop_id1, pop_id2
        if key in accumulated_hs:
            with np.errstate(invalid='ignore'):
                corrected_hs = computed_accumulated_hs[key] / computed_num_vars[key]
                corrected_ht = computed_accumulated_ht[key] / computed_num_vars[key]
            dest = (num_pops / (num_pops - 1)) * ((corrected_ht - corrected_hs) / (1 - corrected_hs))
        else:
            dest = np.nan
        dists[idx] = dest
    return dists


def _calc_pairwise_dest(vars_for_pop1, vars_for_pop2, max_alleles,
                        min_call_dp_for_het, min_num_genotypes):
    num_pops = 2
    ploidy = vars_for_pop1.ploidy

    allele_freq1 = calc_allele_freq(vars_for_pop1, max_alleles=max_alleles,
                                    min_num_genotypes=0)
    allele_freq2 = calc_allele_freq(vars_for_pop2, max_alleles=max_alleles,
                                    min_num_genotypes=0)

    exp_het1 = 1 - va.sum(allele_freq1 ** ploidy, axis=1)
    exp_het2 = 1 - va.sum(allele_freq2 ** ploidy, axis=1)

    hs_per_var = (exp_het1 + exp_het2) / 2

    global_allele_freq = (allele_freq1 + allele_freq2) / 2
    global_exp_het = 1 - va.sum(global_allele_freq ** ploidy, axis=1)
    ht_per_var = global_exp_het

    obs_het1_counts, called_gts1 = _calc_obs_het_counts(vars_for_pop1,
                                                        axis=1,
                                                        min_call_dp_for_het_call=min_call_dp_for_het)
    obs_het1 = obs_het1_counts / called_gts1
    obs_het2_counts, called_gts2 = _calc_obs_het_counts(vars_for_pop2,
                                                        axis=1,
                                                        min_call_dp_for_het_call=min_call_dp_for_het)
    obs_het2 = obs_het2_counts / called_gts2

    called_gts = va.stack([called_gts1, called_gts2], as_type_of=called_gts1)

    try:
        called_gts_hmean = hmean(called_gts, axis=0)
    except ValueError:
        called_gts_hmean = None

    if called_gts_hmean is None:
        num_vars = vars_for_pop1.num_variations
        corrected_hs = va.full((num_vars,), np.nan, as_type_of=vars_for_pop1[GT_FIELD])
        corrected_ht = va.full((num_vars,), np.nan, as_type_of=vars_for_pop1[GT_FIELD])
    else:
        mean_obs_het_per_var = va.nanmean(va.stack([obs_het1, obs_het2],
                                                   as_type_of=obs_het1), axis=0)
        corrected_hs = (called_gts_hmean / (called_gts_hmean - 1)) * (hs_per_var - (mean_obs_het_per_var / (2 * called_gts_hmean)))

        corrected_ht = ht_per_var + (corrected_hs / (called_gts_hmean * num_pops)) - (mean_obs_het_per_var / (2 * called_gts_hmean * num_pops))

        not_enough_gts = va.logical_or(called_gts1 < min_num_genotypes,
                                          called_gts2 < min_num_genotypes)
        corrected_hs[not_enough_gts] = np.nan
        corrected_ht[not_enough_gts] = np.nan

    return {'corrected_hs': corrected_hs, 'corrected_ht': corrected_ht}


def hmean(array, axis=0, dtype=None):
    if axis is None:
        array = array.ravel()
        size = array.shape[0]
    else:
        size = array.shape[axis]
    with np.errstate(divide='ignore'):
        inverse_mean = va.sum(1.0 / array, axis=axis, dtype=dtype)
    is_inf = va.logical_not(va.isfinite(inverse_mean))
    hmean = size / inverse_mean
    hmean[is_inf] = np.nan

    return hmean


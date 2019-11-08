from itertools import combinations

import dask.array as da
import numpy as np

from variation6 import GT_FIELD, FLT_VARS
from variation6.filters import keep_samples
from variation6.stats import calc_missing_gt
from variation6.compute import compute
from collections import OrderedDict


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


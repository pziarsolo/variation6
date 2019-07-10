from variation6 import GT_FIELD, MISSING_GT


def calc_missing_gt(variations, rates=True):
    gts = variations[GT_FIELD]
    ploidy = gts.shape[2]
    bool_gts = gts == MISSING_GT
    num_missing_gts = bool_gts.sum(axis=(1, 2)) / ploidy
    if rates:
        num_missing_gts = num_missing_gts / gts.shape[1]
    return {'num_missing_gts': num_missing_gts}

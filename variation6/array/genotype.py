import variation6.array as va
from variation6 import MISSING_INT


def gts_as_mat012(gts):
    '''It transforms the GT matrix into 0 (major allele homo), 1 (het),
       2(other hom)'''
    gts012 = va.sum(gts, axis=2)
    gts012[va.any(gts == MISSING_INT, axis=2)] = MISSING_INT
    gts012[gts012 >= 1 ] = 2
    gts012[va.logical_and(gts012 == 2, va.any(gts == 0, axis=2))] = 1

    return gts012

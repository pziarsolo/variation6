from os.path import join

CHROM_FIELD_NAME = 'chrom'
POS_FIELD_NAME = 'pos'
ID_FIELD_NAME = 'id'
REF_FIELD_NAME = 'ref'
ALT_FIELD_NAME = 'alt'
QUAL_FIELD_NAME = 'qual'
INFO_FIELD_NAME = 'info'
INDEX_FIELD_NAME = '_index'
GT_FIELD_NAME = 'gt'
GQ_FIELD_NAME = 'gq'
DP_FIELD_NAME = 'dp'
AO_FIELD_NAME = 'ao'
RO_FIELD_NAME = 'ro'
AD_FIELD_NAME = 'ad'

PUBLIC_VARIATION_GROUP = '/variations'
PUBLIC_CALL_GROUP = '/calldata'

CHROM_FIELD = join(PUBLIC_VARIATION_GROUP, CHROM_FIELD_NAME)
POS_FIELD = join(PUBLIC_VARIATION_GROUP, POS_FIELD_NAME)
ID_FIELD = join(PUBLIC_VARIATION_GROUP, ID_FIELD_NAME)
REF_FIELD = join(PUBLIC_VARIATION_GROUP, REF_FIELD_NAME)
ALT_FIELD = join(PUBLIC_VARIATION_GROUP, ALT_FIELD_NAME)
QUAL_FIELD = join(PUBLIC_VARIATION_GROUP, QUAL_FIELD_NAME)
INFO_FIELD = join(PUBLIC_VARIATION_GROUP, INFO_FIELD_NAME)
INDEX_FIELD = join(PUBLIC_VARIATION_GROUP, INDEX_FIELD_NAME)

GT_FIELD = join(PUBLIC_CALL_GROUP, GT_FIELD_NAME)
GQ_FIELD = join(PUBLIC_CALL_GROUP, GQ_FIELD_NAME)
DP_FIELD = join(PUBLIC_CALL_GROUP, DP_FIELD_NAME)
AO_FIELD = join(PUBLIC_CALL_GROUP, AO_FIELD_NAME)
RO_FIELD = join(PUBLIC_CALL_GROUP, RO_FIELD_NAME)
AD_FIELD = join(PUBLIC_CALL_GROUP, AD_FIELD_NAME)

VARIATION_FIELDS = [CHROM_FIELD, POS_FIELD, ID_FIELD, REF_FIELD, ALT_FIELD,
                    QUAL_FIELD, INFO_FIELD ]

CALL_FIELDS = [GT_FIELD, GQ_FIELD, DP_FIELD, AO_FIELD, RO_FIELD, AD_FIELD]

MISSING_INT = -1
MISSING_GT = [MISSING_INT, MISSING_INT]

FLT_VARS = 'flt_vars'
FLT_STATS = 'flt_stats'
N_KEPT = 'n_kept'
N_FILTERED_OUT = 'n_filtered_out'
TOT = 'tot'


class EmptyVariationsError(Exception):
    pass

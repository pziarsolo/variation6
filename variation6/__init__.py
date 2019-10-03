from os.path import join
import numpy as np

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

FLT_VARS = 'flt_vars'
FLT_STATS = 'flt_stats'
FLT_ID = 'flt_id'
N_KEPT = 'n_kept'
N_FILTERED_OUT = 'n_filtered_out'
TOT = 'tot'
COUNT = 'counts'
BIN_EDGES = 'bin_edges'

MIN_NUM_GENOTYPES_FOR_POP_STAT = 10

MISSING_INT = -1
MISSING_FLOAT = float('nan')
MISSING_STR = ''
MISSING_BYTE = b''
MISSING_BOOL = False

MISSING_GT = [MISSING_INT, MISSING_INT]


class _MissingValues():

    def __init__(self):
        self._missing_values = {int: MISSING_INT,
                                'Integer': MISSING_INT,
                                float: MISSING_FLOAT,
                                'float64': MISSING_FLOAT,
                                'Float': MISSING_FLOAT,
                                str: MISSING_STR,
                                'String': MISSING_STR,
                                np.int8: MISSING_INT,
                                np.int16: MISSING_INT,
                                np.int32: MISSING_INT,
                                np.float16: MISSING_FLOAT,
                                np.float32: MISSING_FLOAT,
                                np.bool_: MISSING_BOOL,
                                np.bytes_: MISSING_BYTE,
                                bool: MISSING_BOOL}

    def __getitem__(self, dtype):
        str_dtype = str(dtype)
        if dtype in self._missing_values:
            return self._missing_values[dtype]
        elif isinstance(dtype, str):
            if 'str' in dtype:
                return MISSING_STR
            elif 'int' in dtype:
                return MISSING_INT
            elif 'float' in dtype:
                return MISSING_FLOAT
            elif dtype[0] == 'S':
                return MISSING_BYTE
            elif dtype[:2] == '|S':
                return MISSING_BYTE
        elif 'int' in str_dtype:
            return MISSING_INT
        elif 'float' in str_dtype:
            return MISSING_FLOAT
        elif 'bool' in str_dtype:
            return MISSING_BOOL
        elif str_dtype[:2] == '|S':
            return MISSING_BYTE
        elif str_dtype[:2] == '<U':
            return MISSING_STR
        else:
            raise ValueError('No missing type defined for type: ' + str(dtype))


MISSING_VALUES = _MissingValues()


class EmptyVariationsError(Exception):
    pass

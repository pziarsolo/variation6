import dask.array as da
import numpy as np
from variation6 import EmptyVariationsError


def _same_interface_funcs(funcname, array, *args, **kwargs):
    if isinstance(array, da.Array):
        module = da
    elif isinstance(array, np.ndarray):
        module = np
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)

    return getattr(module, funcname)(array, *args, **kwargs)


def sum(array, *args, **kwargs):
    return _same_interface_funcs('sum', array, *args, **kwargs)


def min(array, *args, **kwargs):
    return _same_interface_funcs('min', array, *args, **kwargs)


def max(array, *args, **kwargs):
    return _same_interface_funcs('max', array, *args, **kwargs)


def isnan(array, *args, **kwargs):
    return _same_interface_funcs('isnan', array, *args, **kwargs)


def isinf(array, *args, **kwargs):
    return _same_interface_funcs('isinf', array, *args, **kwargs)


def histogram(vector, *args, **kwargs):
    return _same_interface_funcs('histogram', vector, *args, **kwargs)


def count_nonzero(a, *args, **kwargs):
    return _same_interface_funcs('count_nonzero', a, *args, **kwargs)


def stack(a, *args, **kwargs):
    array_used_to_infere_type = kwargs.pop('as_type_of', None)
    if array_used_to_infere_type is None:
        array_used_to_infere_type = a
    if isinstance(array_used_to_infere_type, da.Array):
        module = da
    elif isinstance(array_used_to_infere_type, (np.ndarray, list, tuple)):
        module = np
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)

    return module.stack(a, *args, **kwargs)


def amax(array, *args, **kwargs):
    if isinstance(array, da.Array):
        function = da.max
    elif isinstance(array, np.ndarray):
        function = np.amax
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)
    return function(array, *args, **kwargs)


def create_full_array_in_memory(shape, fill_value, *args, **kwargs):
    return np.full(shape, fill_value, *args, **kwargs)


def logical_and(cond1, cond2, *args, **kwargs):
    if isinstance(cond1, da.Array) or isinstance(cond2, da.Array):
        function = da.logical_and
    elif isinstance(cond1, np.ndarray) or isinstance(cond2, np.ndarray):
        function = np.logical_and
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)
    return function(cond1, cond2, *args, **kwargs)


def logical_not(cond1, *args, **kwargs):
    if isinstance(cond1, da.Array):
        function = da.logical_not
    elif isinstance(cond1, np.ndarray):
        function = np.logical_not
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)
    return function(cond1, *args, **kwargs)


def logical_or(cond1, cond2, *args, **kwargs):
    if isinstance(cond1, da.Array) or isinstance(cond2, da.Array):
        function = da.logical_or
    elif isinstance(cond1, np.ndarray) or isinstance(cond2, np.ndarray):
        function = np.logical_or
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)
    return function(cond1, cond2, *args, **kwargs)


def any(array, *args, **kwargs):
    return _same_interface_funcs('any', array, *args, **kwargs)


def all(array, *args, **kwargs):
    return _same_interface_funcs('all', array, *args, **kwargs)


def add(array1, array2, *args, **kwargs):
    if isinstance(array1, da.Array) or isinstance(array2, da.Array):
        function = da.add
    elif isinstance(array1, np.ndarray) or isinstance(array2, np.ndarray):
        function = np.add
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)

    return function(array1, array2, *args, **kwargs)


def nanmean(array, *args, **kwargs):
    return _same_interface_funcs('nanmean', array, *args, **kwargs)


def isfinite(array, *args, **kwargs):
    return _same_interface_funcs('isfinite', array, *args, **kwargs)


def nansum(array, *args, **kwargs):
    return _same_interface_funcs('nansum', array, *args, **kwargs)


def create_not_initialized_array_in_memory(*args, **kwargs):
    return np.empty(*args, kwargs)


def full(shape, *args, **kwargs):
    try:
        array_used_to_infere_type = kwargs.pop('as_type_of')
    except KeyError:
        msg = 'as_type_of is mandatory:  This is an array to infer the type '
        msg += 'of the generated array'
        raise ValueError(msg)

    if isinstance(array_used_to_infere_type, da.Array):
        return da.full(shape, *args, **kwargs)
    elif isinstance(array_used_to_infere_type, np.ndarray):
        return np.full(shape, *args, **kwargs)
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)


def ones(shape, *args, **kwargs):
    try:
        array_used_to_infere_type = kwargs.pop('as_type_of')
    except KeyError:
        msg = 'as_type_of is mandatory:  This is an array to infer the type '
        msg += 'of the generated array'
        raise ValueError(msg)

    if isinstance(array_used_to_infere_type, da.Array):
        return da.ones(shape, *args, **kwargs)
    elif isinstance(array_used_to_infere_type, np.ndarray):
        return np.ones(shape, *args, **kwargs)
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)


def empty_array(same_type_of):
    if isinstance(same_type_of, da.Array):
        return da.from_array(np.array([]))
    elif isinstance(same_type_of, np.ndarray):
        return np.array([])
    else:
        msg = 'Not implemeted for type not in dask array or numpy ndarray'
        raise NotImplementedError(msg)


def map_blocks(func, *args, **kwargs):
    array = args[0]
    if isinstance(array, da.Array):
        return da.map_blocks(func, *args, **kwargs)
    else:
        return func(*args)


def make_sure_array_is_in_memory(array):
    if isinstance(array, da.Array):
        array = array.compute()
    return array


###############################################################################
# Bad or faulty implementations                                               #
###############################################################################
# rara
def assign_with_masking_value(array, masking_value, mask):
    # this reshap must be done with a daskarray without nan values in shape
    # and different shape for array and mask
    if (array.shape and isinstance(array, da.Array) and
        not np.any(np.isnan(array.shape)) and
        len(array.shape) != len(mask.shape)):

        mask = mask.reshape((array.shape))

    array[mask] = masking_value


# rara
def assign_with_mask(array, using, mask):
    if isinstance(array, da.Array):
        values_to_modify = using
    else:
        values_to_modify = using[mask]

    array[mask] = values_to_modify


# rara
def calculate_chunks(array):
    # print("[calculate_chunks] mira estafuncion. NO se como poner diferents formas de poner chunks")
    if isinstance(array, da.Array):
        try:
            return (array.chunks[0], (1,) * len(array.chunks[1]))
        except IndexError:
            raise EmptyVariationsError()

    return None
###############################################################################

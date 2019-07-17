import dask
import dask.array as da
from dask.delayed import Delayed

from variation6.variations import Variations


def compute(data, store_variation_to_memory=False):

    orig_dicts = []
    orig_keys = []
    darrays_to_compute = []
    metadata = None
    variation_key = None
    in_memory_variations = None

    for key_arg, cargo in data.items():
        if isinstance(cargo, (da.Array, Delayed)):
            orig_dicts.append(data)
            orig_keys.append(key_arg)
            darrays_to_compute.append(cargo)

        elif isinstance(cargo, dict):
            for local_key, local_cargo in cargo.items():
                if isinstance(local_cargo, (da.Array, Delayed)):
                    orig_dicts.append(cargo)
                    orig_keys.append(local_key)
                    darrays_to_compute.append(local_cargo)

        elif isinstance(cargo, Variations):
            variation_key = key_arg
            if store_variation_to_memory:
                metadata = cargo.metadata
                orig_dicts.append(cargo)
                orig_keys.append('samples')
                darrays_to_compute.append(cargo.samples)
                for local_key, local_cargo in cargo.items():
                    if isinstance(local_cargo, da.Array):
                        orig_dicts.append(cargo)
                        orig_keys.append(local_key)
                        darrays_to_compute.append(local_cargo)

    computed_darrays = dask.compute(*darrays_to_compute)

    for idx, computed_darray in enumerate(computed_darrays):
        key = orig_keys[idx]
        dict_in_which_the_result_was_stored = orig_dicts[idx]
        if (isinstance(dict_in_which_the_result_was_stored, Variations) and
                store_variation_to_memory):
            if in_memory_variations is None:
                in_memory_variations = Variations(metadata=metadata)
            if key == 'samples':
                in_memory_variations.samples = computed_darray
            else:
                in_memory_variations[key] = computed_darray
        else:
            dict_in_which_the_result_was_stored[key] = computed_darray

    if variation_key:
        if store_variation_to_memory:
            data[variation_key] = in_memory_variations
        else:
            del data[variation_key]

    return data
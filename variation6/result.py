
import dask.array as da
from variation6 import FLT_VARS, FLT_STATS


class Result(dict):

    def compute(self):
        das = []
        keys = []
        for key, value in self.items():
            if isinstance(value, da.Array):
                das.append(value)
                keys.append(key)
        if das:
            arrays = da.compute(*das)
            for key, array in zip(keys, arrays):
                self[key] = array

    def update(self, new_results):
        if FLT_VARS in new_results:
            self[FLT_VARS] = new_results[FLT_VARS]

        if FLT_STATS not in self:
            self[FLT_STATS] = {}

        if FLT_STATS in new_results:
            self[FLT_STATS].update(new_results[FLT_STATS])


import dask.array as da

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

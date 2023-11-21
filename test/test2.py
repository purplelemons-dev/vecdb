
import pickle
from json import dumps
from numpy import ndarray

with open("./testdb/d54390c9161b.pkl", "rb") as f:
    obj:dict = pickle.load(f)
    for k,v in obj.items():
        if isinstance(v, ndarray):
            obj[k] = v.tolist()
            continue
        obj[k] = v
    json_obj = dumps(obj)
    print(json_obj)

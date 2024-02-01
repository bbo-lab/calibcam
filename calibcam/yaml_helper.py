import numpy as np
from copy import deepcopy

def get_calib_numpy_fields():
    return ['rvec_cam', 'tvec_cam', 'A', 'xi', 'k', 'rvecs', 'tvecs', 'frames_mask', 'std_intrinsics',
            'std_extrinsics', 'per_view_errors']


def numpy_collection_to_list(collection):
    if isinstance(collection, np.ndarray):
        return collection.tolist()
    if not isinstance(collection, dict) and not isinstance(collection, list):
        return deepcopy(collection)
    collection = deepcopy(collection)
    for k in collection:
        if k in collection:
            if isinstance(collection[k], np.ndarray):
                collection[k] = collection[k].tolist()
            elif isinstance(collection[k], dict):
                collection[k] = numpy_collection_to_list(collection[k])
            elif isinstance(collection[k], list):
                collection[k] = [numpy_collection_to_list(le) for le in collection[k]]
    return collection


def load_calib(calib):
    calib = deepcopy(calib)
    for k in get_calib_numpy_fields():
        if k in calib:
            calib[k] = np.array(calib[k])
    return calib


def load_opts(opts):
    # TODO: Implement recursive conversion and limit fields to those that should actually be converted!
    opts = deepcopy(opts)
    for k in opts.keys():
        if k in opts:
            opts[k] = np.array(opts[k])
    return opts
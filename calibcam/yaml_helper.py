import numpy as np
from copy import deepcopy

def get_calib_numpy_fields():
    return ['rvec_cam', 'tvec_cam', 'A', 'xi', 'k', 'rvecs', 'tvecs', 'frames_mask', 'std_intrinsics',
            'std_extrinsics', 'per_view_errors']


def calib_to_list(calib):
    calib = deepcopy(calib)
    for k in get_calib_numpy_fields():
        if k in calib:
            calib[k] = calib[k].tolist()
    return calib

def calib_to_numpy(calib):
    calib = deepcopy(calib)
    for k in get_calib_numpy_fields():
        if k in calib:
            calib[k] = np.array(calib[k])
    return calib


def numpy_dict_to_list(opts):
    opts = deepcopy(opts)
    for k in opts.keys():
        if k in opts:
            if isinstance(opts[k], np.ndarray):
                opts[k] = opts[k].tolist()
            elif isinstance(opts[k], dict):
                opts[k] = numpy_dict_to_list(opts[k])
    return opts

def opts_to_numpy(opts):
    # TODO: Implement recursive conversion and limit fields to those that should actually be converted!
    opts = deepcopy(opts)
    for k in opts.keys():
        if k in opts:
            opts[k] = np.array(opts[k])
    return opts
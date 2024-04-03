import numpy as np
from copy import deepcopy
from calibcam import calibrator_opts
from calibcamlib.yaml_helper import collection_to_array, numpy_collection_to_list


def get_calib_numpy_fields():
    return ['rvec_cam', 'tvec_cam', 'A', 'xi', 'k', 'rvecs', 'tvecs', 'frames_mask', 'std_intrinsics',
            'std_extrinsics', 'per_view_errors']


def load_calib(calib):
    calib = deepcopy(calib)
    for k in get_calib_numpy_fields():
        if k in calib:
            calib[k] = np.array(calib[k])
    return calib


def load_opts(opts):
    opts = deepcopy(opts)
    default_opts = calibrator_opts.get_default_opts()
    for k in default_opts.keys():
        if k in opts:
            opts[k] = collection_to_array(opts[k])

    return opts

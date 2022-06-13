import cv2
import numpy as np


def get_default_opts():
    default_opts = {
        'coord_cam': 0,  # Reference camera that defines the multicam coordinate system
        'allow_unequal_n_frame': False,  # Sometimes last frame is cut, so this may be okay.
        'common_pose_r_err': 0.1,    # Iteratively exclude poses with higher rotation deviation from mean
        'color_convert': False,  # Set to cv2.COLOR_RGB2GRAY to convert rgb images to grayscale for corner detection
        'detect_cpu_divisor': 6,  # use N_CPU/detect_cpu_divisor threads for feature detection
        'optimize_only': False,  # Do not perform detection and single cam calibration. (Disable mostly for development.)
        'numerical_jacobian': False,  # Use autograd instead of 2-point numerical jacobian
        'free_vars': {
            'cam_pose': True,
            'board_poses': True,
            'A': np.asarray([[True,  False, True],   # a   c   u   (c is skew and should not be necessary)
                             [False, True,  True],   # 0   b   v
                             [False, False, False],  # 0   0   1
                             ]),
            'k': np.asarray([True, True, False, False, False]),  # This also determines the supported degree of
                                                                 # distortion. Others will be set to 0.
        },
        'detection': {
            'inter_frame_dist': 0.0,
            'aruco_calibration': {
                'flags': (cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3),
                'criteria': (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                             30,
                             1.1920928955078125e-07),  # float(np.finfo(np.float32).eps)
            },
            'aruco_detect': {
                'parameters': get_detector_parameters_opts(),
            },
            'aruco_refine': {
                'minRepDistance': 10.0,
                'errorCorrectionRate': 3.0,
                'checkAllOrders': True,
                'parameters': get_detector_parameters_opts(),
            },
            'aruco_interpolate': {
                'minMarkers': 2,
            },
        },
        'optimization': {
            'method': 'trf',
            'ftol': 1e-4,
            'xtol': 1e-8,
            'gtol': 1e-8,
            'x_scale': 'jac',
            'loss': 'linear',
            'tr_solver': 'exact',
            'max_nfev': 100,
            'verbose': 2,
        },
    }

    return default_opts


def get_detector_parameters_opts():
    detector_parameters = {  # SPOT for detector params
        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'cornerRefinementWinSize': 5,
        'cornerRefinementMaxIterations': 30,
        'cornerRefinementMinAccuracy': 0.1,
    }
    return detector_parameters


def finalize_aruco_detector_opts(aruco_detect_opts):
    # Separation is necessary as cv2.aruco.DetectorParameters cannot be pickled
    opts = aruco_detect_opts.copy()

    detector_parameters = cv2.aruco.DetectorParameters_create()
    detector_parameters.cornerRefinementMethod = aruco_detect_opts['parameters']['cornerRefinementMethod']
    detector_parameters.cornerRefinementWinSize = aruco_detect_opts['parameters']['cornerRefinementWinSize']
    detector_parameters.cornerRefinementMaxIterations = aruco_detect_opts['parameters']['cornerRefinementMaxIterations']
    detector_parameters.cornerRefinementMinAccuracy = aruco_detect_opts['parameters']['cornerRefinementMinAccuracy']

    opts['parameters'] = detector_parameters
    return opts

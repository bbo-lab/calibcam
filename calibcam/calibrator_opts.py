import cv2
import numpy as np


def get_default_opts(model="pinhole"):

    default_opts = {
        'debug': True,  # Will enable advanced tests and outputs
        'coord_cam': 0,  # Reference camera that defines the multicam coordinate system
        'frame_step': 1,  # Skip frames in recording
        'allow_unequal_n_frame': False,  # Sometimes last frame is cut, so this may be okay.
        'common_pose_r_err': 0.1,  # Iteratively exclude poses with higher rotation deviation from mean
        'color_convert': False,  # Set to cv2.COLOR_RGB2GRAY to convert rgb images to grayscale for corner detection
        'detect_cpu_divisor': 6,  # use N_CPU/detect_cpu_divisor threads for feature detection
        'optimize_only': False,
        # Do not perform detection and single cam calibration. (Disable mostly for development.)
        'numerical_jacobian': False,  # Use 2-point numerical jacobian instead of jax.jacobian
        'optimize_board_poses': False,  # Optimize individual board poses then all params again. In a test,
        #  optimality was already reached after a first general optimization

        'max_allowed_res': 1.0,  # In pixels. Reject the pose with higher error and insert 'nearby' pose with lower
        # error while optimizing individual board poses.

        'free_vars': get_free_vars(model),
        'detection': {
            'inter_frame_dist': 0.0,  # In pixels
            'aruco_calibration': {
                'flags': get_flags(model),
                'criteria': (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                             30,
                             float(np.finfo(np.float32).eps)),
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
            'tr_solver': 'lsmr',
            'max_nfev': 150,
            'verbose': 2,
        },
    }

    return default_opts


def get_free_vars(model: str):
    free_vars = {
        'cam_pose': True,
        'board_poses': True,
        'A': np.asarray([[True, False, True],  # a   c   u   (c is skew and should not be necessary)
                         [False, True, True],  # 0   b   v
                         [False, False, False],  # 0   0   1
                         ]),
        'xi': False,
        'k': np.asarray([1, 1, -1, -1, -1]),  # 1: optimize, 0: leave const, -1: force 0
    }

    if model == "omnidir":
        # 'A' or 'K' (opencv-omnidir notation) - camera matrix
        # 'k' or 'D' (opencv-omnidir notation) - distortion coeffs
        free_vars['xi'] = True
        free_vars['k'] = np.asarray([1, 1, 1, 1, 1])

    return free_vars


def get_flags(model: str):
    flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3

    if model == "omnidir":
        # "omnidir"
        # Should be set to None or 0, if no flags are to be used. Use 0, None is causing error with scipy_io_savemat
        # return (cv2.omnidir.CALIB_FIX_P1 + cv2.omnidir.CALIB_FIX_P2)
        flags = cv2.omnidir.CALIB_FIX_SKEW

    return flags


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

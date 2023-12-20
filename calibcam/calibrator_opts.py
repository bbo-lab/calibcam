import cv2
import numpy as np


def get_default_opts(models=["pinhole"]):

    default_opts = {
        'debug': True,  # Will enable advanced tests and outputs
        'jax_backend': 'cpu',  # GPU regularly runs out of memory for these problems
        'coord_cam': 0,  # Reference camera that defines the multicam coordinate system
        'frame_step': 1,  # Skip frames in recording
        'allow_unequal_n_frame': True,  # Sometimes last frame is cut, so this may be okay.
        'common_pose_r_err': 0.1,  # Iteratively exclude poses with higher rotation deviation from mean
        'color_convert': False,  # Set to cv2.COLOR_RGB2GRAY to convert rgb images to grayscale for corner detection
        'detect_cpu_divisor': 6,  # use N_CPU/detect_cpu_divisor threads for feature detection
        'optimize_only': False,
        # Do not perform detection and single cam calibration. (Disable mostly for development.)
        'numerical_jacobian': False,  # Use 2-point numerical jacobian instead of jax.jacobian
        # Optimise individual cameras immediately after performing opencv single calibration
        'optimize_ind_cams': False,

        # Optimize individual board poses then all params again.
        # In a test, optimality was already reached after a first general optimization
        'optimize_board_poses': False,
        'max_allowed_res': 2.0,  # In pixels. replace the pose with higher error and insert 'nearby' pose with lower
        # error while optimizing individual board poses.

        'reject_corners': False,  # Reject corners with high zscore, check rejection params below

        'free_vars': [get_free_vars(model) for model in models],
        'detection': {
            'inter_frame_dist': 1.0,  # In pixels
            'aruco_detect': {
                'parameters': get_detector_parameters_opts(),
            },
            'aruco_refine': {
                'minRepDistance': 3.0,
                'errorCorrectionRate': 1.0,
                'checkAllOrders': True,
                'parameters': get_detector_parameters_opts(),
            },
            'aruco_interpolate': {
                'minMarkers': 2,
            },
        },
        'aruco_calibration': [{
            'flags': get_flags(model),
            'criteria': (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                         30,
                         float(np.finfo(np.float32).eps)),
        } for model in models],
        'pose_estimation': {
            'use_required_corners': True,
        },
        'optimization': {
            'method': 'trf',
            'ftol': 1e-9,
            'xtol': 1e-9,
            'gtol': 1e-9,
            'x_scale': 'jac',
            'loss': 'linear',
            'tr_solver': 'exact',
            'max_nfev': 120,
            'verbose': 2,
        },
        'rejection': {
            'max_zscore': 3.0,
            'max_res': 2.0,  # in pixels, only reject corners with high residual (rep. error) and zscore
            'reject_poses': True  # Reject degenerate poses
        }
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
        free_vars['k'] = np.asarray([1, 1, 1, 1, 1])  # 1: optimize, 0: leave const, -1: force 0

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
        'adaptiveThreshWinSizeMin': 15,
        'adaptiveThreshWinSizeMax': 30,
        'adaptiveThreshWinSizeStep': 5,

        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'cornerRefinementWinSize': 3,
        'cornerRefinementMaxIterations': 60,
        'cornerRefinementMinAccuracy': 0.05,

        'errorCorrectionRate': 0.3,
        'perspectiveRemovePixelPerCell': 8
    }
    return detector_parameters


def finalize_aruco_detector_opts(aruco_detect_opts):
    # Separation is necessary as cv2.aruco.DetectorParameters cannot be pickled
    opts = aruco_detect_opts.copy()

    detector_parameters = cv2.aruco.DetectorParameters_create()
    for key, value in aruco_detect_opts['parameters'].items():
        detector_parameters.__setattr__(key, value)

    opts['parameters'] = detector_parameters
    return opts

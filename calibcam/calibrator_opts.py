import cv2
import numpy as np


def get_default_opts(ncams=0, do_fill=False):
    default_opts = {
        # === Program function control ===
        # If True, calibcam will perform detections. If list of yml files, these files will be used instead
        'detection': False,
        # If True, calibcam will perform single cam calibrations. If list of yml files, these files will be used instead
        'calibration_single': False,
        # If True, calibcam will perform multi cam calibration.
        'calibration_multi': False,

        # === Camera system description
        # Number of cams
        'n_cams': ncams,
        # Camera models. Will be filled with opts["n_cams"]*["pinhole"] if left False
        'models': False,
        # Free variables. Will be filled with get_free_vars() if left False
        'free_vars': False,

        # === Settings ===
        # Will enable advanced tests and outputs
        'debug': True,
        # GPU regularly runs out of memory for these problems
        'jax_backend': 'cpu',
        # Reference camera that defines the multicam coordinate system
        'coord_cam': 0,
        # Skip frames in recording
        'frame_step': 1,
        # Sometimes last frame is cut, so this may be okay.
        'allow_unequal_n_frame': True,
        # Iteratively exclude poses with higher rotation deviation from mean
        'common_pose_r_err': 0.1,
        # Set to cv2.COLOR_RGB2GRAY to convert rgb images to grayscale for corner detection
        'color_convert': False,
        # use N_CPU/detect_cpu_divisor threads for feature detection
        'detect_cpu_divisor': 6,
        # Use radial contrast value for rejecting corners, check rejection params below in detection_opts
        'RC_reject_corners': False,
        # DEPRECATED
        # Do not perform detection and single cam calibration. (Disable mostly for development.)
        'optimize_only': False,
        # Use 2-point numerical jacobian instead of jax.jacobian
        'numerical_jacobian': False,
        # Optimise individual cameras immediately after performing opencv single calibration
        'optimize_ind_cams': False,
        # Optimize individual board poses then all params again.
        # In a test, optimality was already reached after a first general optimization
        'optimize_board_poses': False,
        # In pixels. replace the pose with higher error and insert 'nearby' pose with lower
        # error while optimizing individual board poses.
        'max_allowed_res': 5.0,
        # Use these extrinsics for initialization dict('rvecs_cam': nx3, 'tvecs_cam': nx3)
        'init_extrinsics': {
            'rvecs_cam': -1,
            'tvecs_cam': -1,
        },

        'detection_opts': {
            'inter_frame_dist': 1.0,  # In pixels
            'min_corners': 5,  # Minimum number of corners to detect in a frame
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
            'radial_contrast_reject': {
                'options': {'lib': 'np'},
                'width': 20,
                'normalize': 50,
                'norm_mean': 0.311,
            }
        },
        'aruco_calibration': False,
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
            'tr_solver': 'lsmr',
            'max_nfev': 500,
            'verbose': 2,
        }
    }

    if do_fill:
        fill(default_opts)

    return default_opts


def fill(opts):
    if not opts["models"]:
        opts["models"] = opts["n_cams"] * ["pinhole"]
    if len(opts["models"]) == 1:
        opts["models"] *= opts["n_cams"]

    if not opts["free_vars"]:
        opts["free_vars"] = [get_free_vars(model) for model in opts["models"]]

    if not opts["aruco_calibration"]:
        opts['aruco_calibration'] = [{
            'flags': get_flags(model),
            'criteria': (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                         30,
                         float(np.finfo(np.float32).eps)),
        } for model in opts["models"]]

    return opts  # not necessarily required due to pass by reference


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
        'adaptiveThreshWinSizeMin': 3,
        'adaptiveThreshWinSizeMax': 23,
        'adaptiveThreshWinSizeStep': 10,
        'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        'cornerRefinementWinSize': 5,
        'cornerRefinementMaxIterations': 30,
        'cornerRefinementMinAccuracy': 0.01,
        # 'adaptiveThreshWinSizeMin': 15,
        # 'adaptiveThreshWinSizeMax': 45,
        # 'adaptiveThreshWinSizeStep': 5,
        #
        # 'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
        # 'cornerRefinementWinSize': 3,
        # 'cornerRefinementMaxIterations': 60,
        # 'cornerRefinementMinAccuracy': 0.05,
        #
        'errorCorrectionRate': 0.3,
        'perspectiveRemovePixelPerCell': 8
    }
    return detector_parameters


def finalize_aruco_detector_opts(aruco_detect_opts):
    # Separation is necessary as cv2.aruco.DetectorParameters cannot be pickled
    opts = aruco_detect_opts.copy()

    detector_parameters = cv2.aruco.DetectorParameters()
    for key, value in aruco_detect_opts['parameters'].items():
        detector_parameters.__setattr__(key, value)
        # print(f"{key}: {value}")

    opts['parameters'] = detector_parameters
    return opts

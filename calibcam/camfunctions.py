import numpy as np
from scipy.optimize import least_squares, OptimizeResult

import time

from calibcam import optimization, board
from .exceptions import *


def optimize_calib_parameters(corners_all, ids_all, calibs_multi, frame_masks, opts, board_params):
    start_time = time.time()

    board_coords_3d_0 = board.make_board_points(board_params)

    used_frame_mask = np.any(frame_masks, axis=0)
    used_frame_idxs = np.where(used_frame_mask)[0]
    n_used_frames = used_frame_mask.sum(dtype=int)
    n_cams = len(calibs_multi)
    n_corner = board_coords_3d_0.shape[0]

    # Generate vectors of all and of free variables
    vars_free, vars_full, mask_free = optimization.make_initialization(calibs_multi, frame_masks, opts)

    # Prepare ideal boards for each cam and frame. This costs almost 3x the necessary memory, but makes
    # further autograd compatible code much easier
    boards_coords_3d_0 = np.tile(board_coords_3d_0.T,
                                 (n_cams, n_used_frames, 1, 1))  # Note transpose for later linalg

    # Prepare array of corners (non-existing frames for individual cameras are filled with NaN)
    corners = np.empty(shape=(n_cams, n_used_frames, 2, n_corner))
    corners[:] = np.NaN
    for i_cam, frame_mask_cam in enumerate(frame_masks):
        frame_idxs_cam = np.where(frame_mask_cam)[0]

        for i_frame, f_idx in enumerate(used_frame_idxs):
            # print(ids_all[i_cam][i_frame].ravel())
            # print(corners[i_cam, f_idx].shape)
            # print(corners_all[i_cam][i_frame].shape)
            cam_fr_idx = np.where(frame_idxs_cam == f_idx)[0]
            if cam_fr_idx.size < 1:
                continue

            cam_fr_idx = int(cam_fr_idx)
            corners[i_cam, i_frame][:, ids_all[i_cam][cam_fr_idx].ravel()] = \
                corners_all[i_cam][cam_fr_idx][:, 0, :].T  # Note transpose for later linalg

    args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free,  # Mask of free vars within all vars
        'frame_masks': frame_masks,
        'opts_free_vars': opts['free_vars'],
        'precalc': {  # Stuff that can be precalculated
            'boards_coords_3d_0': boards_coords_3d_0,  # Board points in z plane
            'corners': corners,
            'jacobians': optimization.get_obj_fcn_jacobians(),
        },
        # Inapplicable tue to autograd slice limitations
        # 'memory': {  # References to memory that can be reused, avoiding cost of reallocation
        #     'residuals': np.zeros_like(corners),
        #     'boards_coords_3d': np.zeros_like(boards_coords_3d_0),
        #     'boards_coords_3d_cams': np.zeros_like(boards_coords_3d_0),
        #     'calibs': calibs_fit,
        # }
    }

    # For comparison with unraveled data, tested correct
    # print(calibs_multi[2]['rvec_cam'])
    # print(calibs_multi[2]['tvec_cam'])
    # print(calibs_multi[2]['A'])
    # print(calibs_multi[2]['k'])

    # Check quality of calibration, tested working (requires calibcamlib >=0.2.3 on path)
    test_objective_function(calibs_multi, vars_free, args, corners, board.make_board_points(board_params))

    print('Starting optimization procedure')

    if opts['use_autograd']:
        jac = optimization.obj_fcn_jacobian_wrapper
    else:
        jac = '2-point'

    min_result: OptimizeResult = least_squares(optimization.obj_fcn_wrapper,
                                               vars_free,
                                               jac=jac,
                                               bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                               args=[args],
                                               **opts['optimization'])
    current_time = time.time()
    print('Optimization algorithm converged:\t{:s}'.format(str(min_result.success)))
    print('Time needed:\t\t\t\t{:.0f} seconds'.format(current_time - start_time))

    calibs_fit, rvecs_boards, tvecs_boards = optimization.unravel_to_calibs(min_result.x, args)

    return calibs_fit, rvecs_boards, tvecs_boards, min_result, args


def get_n_frames_from_reader(reader):
    n_frames = len(reader)  # len() may be Inf for formats where counting frames can be expensive
    if 1000000000000000 < n_frames:
        try:
            n_frames = reader.count_frames()
        except ValueError:
            print('Could not determine number of frames')
            raise UnsupportedFormatException

    return n_frames


def get_header_from_reader(reader):
    header = reader.get_meta_data()
    # Add required headers that are not normally part of standard video formats but are required information
    # for a full calibration
    # TODO add option to supply this via options. Currently, compressed
    if "sensor" in header:
        header['offset'] = tuple(header['sensor']['offset'])
        header['sensorsize'] = tuple(header['sensor']['size'])
        del header['sensor']
    else:
        if 'offset' not in header:
            print("Setting offset to 0!")
            header['offset'] = tuple(np.asarray([0, 0]))

        if 'sensorsize' not in header:
            print("Inferring sensor size from image")
            header['sensorsize'] = tuple(reader.get_data(0).shape[0:2])

    return header


def test_objective_function(calibs, vars_free, args, corners_detection, board_points):
    from calibcamlib import Camerasystem
    from scipy.spatial.transform import Rotation as R  # noqa

    used_frame_mask = np.any(args['frame_masks'], axis=0)
    used_frame_idxs = np.where(used_frame_mask)[0]  # noqa

    residuals_objfun = np.abs(optimization.obj_fcn_wrapper(vars_free, args).reshape(corners_detection.shape))
    residuals_objfun[residuals_objfun == 0] = np.NaN

    pose_params = optimization.make_common_pose_params(calibs, args['frame_masks'])
    rvecs_board = pose_params[0].reshape(-1, 3)  # noqa
    tvecs_board = pose_params[1].reshape(-1, 3)  # noqa
    corners_cameralib = np.empty_like(residuals_objfun)
    corners_cameralib[:] = np.NaN
    cs = Camerasystem.from_calibs(calibs)
    for i_cam, calib in enumerate(calibs):
        # This calculates from individual board pose estimations
        # cam_frame_idxs = np.where(calibs[i_cam]['frame_mask'])[0]
        # b = board_points @ R.from_rotvec(calibs[i_cam]['rvecs'].reshape(-1, 3)).as_matrix().transpose((0, 2, 1)) + \
        #     calibs[i_cam]['tvecs'].reshape(-1, 1, 3)
        # corners_cameralib[i_cam, np.isin(used_frame_idxs, cam_frame_idxs)] = cs.project(b)[i_cam].transpose((0, 2, 1))
        # This calculates from common camera board estimations
        b = board_points @ R.from_rotvec(rvecs_board.reshape(-1, 3)).as_matrix().transpose((0, 2, 1)) + \
            tvecs_board.reshape(-1, 1, 3)
        corners_cameralib[i_cam, :] = cs.project(b)[i_cam].transpose((0, 2, 1))

    residuals_cameralib = np.abs(corners_detection - corners_cameralib)

    res_diff = residuals_cameralib - residuals_objfun

    print("Testing objective function vs cameralib (minor differences expected due to common board pose in objfun)")
    print("Cam | "
          "n objfun      | "
          "n cameralib   | "
          "max objfun    | "
          "med objfun    | "
          "max cameralib | "
          "med cameralib | "
          "diff"
          )
    for i_cam in range(len(calibs)):
        print(f"{i_cam:3} | "
              f"{(~np.isnan(residuals_objfun)).sum():13} | "
              f"{(~np.isnan(residuals_cameralib)).sum():13} | "
              f"{np.nanmax(residuals_objfun[i_cam]):13.2f} | "
              f"{np.nanmedian(residuals_objfun[i_cam]):13.2f} | "
              f"{np.nanmax(residuals_cameralib[i_cam]):13.2f} | "
              f"{np.nanmedian(residuals_cameralib[i_cam]):13.2f} | "
              f"{np.nanmax(res_diff[i_cam]):13.2f}"
              )

import numpy as np
from scipy.optimize import least_squares, OptimizeResult

import timeit

from calibcam import optimization, board, helper, calibrator_opts
from calibcam.exceptions import *


def optimize_calib_parameters(corners, calibs_multi, board_params, offsets, opts=None):
    if opts is None:
        opts = {}
    defaultopts = calibrator_opts.get_default_opts()
    opts = helper.deepmerge_dicts(opts, defaultopts)

    start_time = timeit.default_timer()

    board_coords_3d_0 = board.make_board_points(board_params)

    # Generate vectors of all and of free variables
    vars_free, vars_full, mask_free_input = optimization.make_initialization(calibs_multi, corners, board_params, offsets, opts)

    args = {
        'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
        'mask_opt': mask_free_input,  # Mask of free vars within all vars
        'opts_free_vars': opts['free_vars'],
        'coord_cam': opts['coord_cam'],  # This is currently only required due to unsolved jacobian issue
        'board_coords_3d_0': board_coords_3d_0,  # Board points in z plane
        'corners': corners,
        'precalc': optimization.get_precalc(),
        # Inapplicable tue to autograd slice limitations
        # 'memory': {  # References to memory that can be reused, avoiding cost of reallocation
        #     'residuals': np.zeros_like(corners),
        #     'boards_coords_3d': np.zeros_like(boards_coords_3d_0),
        #     'boards_coords_3d_cams': np.zeros_like(boards_coords_3d_0),
        #     'calibs': calibs_fit,
        # }
    }

    # This triggers JIT compilation
    optimization.obj_fcn_wrapper(vars_free, args)
    # This times
    tic = timeit.default_timer()
    result = optimization.obj_fcn_wrapper(vars_free, args)
    print(
        f"Objective function took {timeit.default_timer() - tic} s: squaresum {np.sum(result ** 2)} over {result.size} residuals.")

    if opts['numerical_jacobian']:
        jac = '2-point'
    else:
        jac = optimization.obj_fcn_jacobian_wrapper
        # This triggers JIT compilation
        jac(vars_free, args)
        # This times
        tic = timeit.default_timer()
        result = jac(vars_free, args)
        print(
            f"Jacobian took {timeit.default_timer() - tic} s: squaresum {np.sum(result ** 2)} over {result.size} residuals.")

    # Check quality of calibration, tested working (requires calibcamlib >=0.2.3 on path)
    test_objective_function(calibs_multi, vars_free, args, corners, board_params, offsets)

    print('Starting optimization procedure')
    # TODO Test if alternating optimization between camera parameters and poses with a breaking critierion on camera
    #  params could be more efficient ... I think often cam params are optimal quite quickly and the opimization runs on
    #  some rougue poses ...
    min_result: OptimizeResult = least_squares(optimization.obj_fcn_wrapper,
                                               vars_free,
                                               jac=jac,
                                               bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                               args=[args],
                                               **opts['optimization'])

    current_time = timeit.default_timer()
    print('Optimization algorithm converged:\t{:s}'.format(str(min_result.success)))
    print('Time needed:\t\t\t\t{:.0f} seconds'.format(current_time - start_time))

    calibs_fit, rvecs_boards, tvecs_boards = optimization.unravel_to_calibs(min_result.x, args)

    # We don't include poses in the calibs_fit dictionary, as the final calibration structure should be independent
    #  of the calibration process
    calibs_test = [
        calibs_fit[i_cam] | {
            'rvecs': rvecs_boards,
            'tvecs': tvecs_boards,
        }
        for i_cam in range(len(calibs_fit))
    ]
    test_objective_function(calibs_test, min_result.x, args, corners, board_params, offsets, individual_poses=True)

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
    # TODO add option to supply this via options. Currently, compressed videos may lack this info
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


def test_objective_function(calibs, vars_free, args, corners_detection, board_params, offsets, individual_poses=False):
    from calibcamlib import Camerasystem
    from scipy.spatial.transform import Rotation as R  # noqa

    residuals_objfun = np.abs(optimization.obj_fcn_wrapper(vars_free, args).reshape(corners_detection.shape))
    residuals_objfun[residuals_objfun == 0] = np.NaN

    corners_cameralib = np.empty_like(residuals_objfun)
    corners_cameralib[:] = np.NaN
    cs = Camerasystem.from_calibs(calibs)
    board_points = board.make_board_points(board_params)
    for i_cam, calib in enumerate(calibs):
        # This calculates from individual board pose estimations
        if individual_poses:
            rvecs_board = calibs[i_cam]['rvecs'].reshape(-1, 3)
            tvecs_board = calibs[i_cam]['tvecs'].reshape(-1, 3)
        # This calculates from common camera board estimations
        else:
            pose_params = optimization.make_common_pose_params(calibs, corners_detection, board_params, offsets)
            rvecs_board = pose_params[0].reshape(-1, 3)
            tvecs_board = pose_params[1].reshape(-1, 3)

        b = np.einsum('fij,bj->fbi', R.from_rotvec(rvecs_board.reshape(-1, 3)).as_matrix(), board_points) + \
            tvecs_board.reshape(-1, 1, 3)

        corners_cameralib[i_cam, :] = cs.project(b)[i_cam]

    residuals_cameralib = np.abs(corners_detection - corners_cameralib)

    res_diff = residuals_cameralib - residuals_objfun

    print("Testing objective function vs cameralib")
    if individual_poses:
        print("(Minor differences possible due to common board pose in objfun)")
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

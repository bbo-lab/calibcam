# import multiprocessing
# from joblib import Parallel, delayed
import warnings

import numpy as np

import sys

# This could also be done dynamically, based on opts ...
# from calibcam.opt_vmapgrad.optfunctions import obj_fcn_wrapper, obj_fcn_jacobian_wrapper, get_precalc  # noqa
# Calculating Jacobians would be much more straightforward, but seems to be prohibitively slow ...
import calibcamlib
from calibcam import board, helper
from calibcam.repro.optfunctions import obj_fcn_wrapper, obj_fcn_jacobian_wrapper, obj_fcn_jacobian_wrapper_sparse, \
    get_precalc  # noqa
from scipy.spatial.transform import Rotation as R  # noqa


def make_vars_full(vars_opt, args, verbose=False):
    n_cams = args['corners'].shape[0]
    n_boards = args['corners'].shape[1]

    # Update full set of vars with free vars
    vars_full = args['vars_full']
    if verbose:
        print(vars_full[0:7])
    mask_opt = args['mask_opt']
    vars_full[mask_opt] = vars_opt

    if verbose:
        print(mask_opt[0:7])
        print(vars_full[0:7])

    return vars_full, n_cams, n_boards


def get_cam_field_sizes(m):
    field_sizes = np.array([m * 3,  # cam extrinsic rotations
                            m * 3,  # cam extrinsic positions
                            m * 9,  # cam intrinsics A
                            m,  # cam intrinsics xi
                            m * 5,  # cam intrinsics k
                            ])
    return field_sizes


def get_board_field_sizes(k):
    field_sizes = np.array([k * 3,  # boards rotations
                            k * 3,  # boards extrinsic positions
                            ])
    return field_sizes


def get_var_edges(m, k):
    field_sizes = np.concatenate(([0], get_cam_field_sizes(m), get_board_field_sizes(k)))
    edges = np.cumsum(field_sizes)
    return edges


def unravel_vars_full(vars_full, n_cams, n_boards):
    edges = get_var_edges(n_cams, n_boards)
    edge_idx = 0

    rvecs_cams = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 3)
    edge_idx += 1
    tvecs_cams = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 3)
    edge_idx += 1

    cam_matrices = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 3, 3)
    edge_idx += 1
    xis = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 1)
    edge_idx += 1
    ks = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 5)
    edge_idx += 1

    rvecs_boards = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 3)
    edge_idx += 1
    tvecs_boards = vars_full[edges[edge_idx]:edges[edge_idx + 1]].reshape(-1, 3)
    edge_idx += 1

    return rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards


def make_initialization(calibs, corners, board_params, opts):
    opts_free_vars = opts['free_vars']

    # camera_params are raveled with first all rvecs, then tvecs, then A, then k
    camera_params = make_cam_params(calibs, opts_free_vars)
    # pose_params are raveled with first all rvecs and then all tvecs
    pose_params = make_common_pose_params(calibs, corners, board_params).ravel()

    vars_full = np.concatenate((camera_params, pose_params), axis=0)
    mask_free_input = make_free_parameter_mask(calibs, opts_free_vars, opts['coord_cam'])
    vars_free = vars_full[mask_free_input]

    return vars_free, vars_full, mask_free_input


def make_cam_params(calibs, opts_free_vars):
    # k_to_zero determines if non-free ks get set to 0 (for limited distortion model) or are kept (usually
    # when not optimizing distortion at all in the given step)

    rvecs_cam = np.zeros(shape=(len(calibs), 3))
    tvecs_cam = np.zeros(shape=(len(calibs), 3))
    cam_mats = np.zeros(shape=(len(calibs), 3, 3))
    xis = np.zeros(shape=(len(calibs), 1))
    ks = np.zeros(shape=(len(calibs), 5))

    for calib, cam_free_vars, r, t, cm, xi, k in zip(calibs, opts_free_vars, rvecs_cam, tvecs_cam, cam_mats, xis, ks):
        r[:] = calib['rvec_cam']
        t[:] = calib['tvec_cam']
        cm[:] = calib['A']

        xi[:] = calib.get('xi', 0.0)
        k[:] = calib['k']
        k[cam_free_vars['k'] == -1] = 0

    camera_params = np.concatenate((
        rvecs_cam.ravel(),
        tvecs_cam.ravel(),
        cam_mats.ravel(),
        xis.ravel(),
        ks.ravel(),
    ), axis=0)

    return camera_params


def make_common_pose_params(calibs, corners_array, board_params):
    pose_params = np.zeros(shape=(2, corners_array.shape[1], 3))

    # Decide which of the n_cam pose estimation from each frame to use based on reprojection error
    repro_errors = np.zeros(shape=len(calibs))
    # cs = calibcamlib.Camerasystem.from_calibs(calibs)
    cs = calibcamlib.Camerasystem.from_calibs(calibs)
    offsets = np.zeros(shape=(len(calibs), 2))  # Offsets were previously removed from corners

    pose_params_calcd = []
    for i_pose in range(pose_params.shape[1]):
        for i_cam, calib in enumerate(calibs):
            proj = cs.project(R.from_rotvec(calib['rvecs'][i_pose]).apply(board.make_board_points(board_params))
                              + calib['tvecs'][i_pose], offsets)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                repro_errors[i_cam] = np.nanmean(np.abs(proj - corners_array[:, i_pose]))

        # OpenCV omnidirectional camera calibration sometimes does not provide rvecs and tvecs for certain frames
        # when it fails to initialise.
        if np.any(~np.isnan(repro_errors)):
            pose_params[0, i_pose, :] = calibs[np.nanargmin(repro_errors)]['rvecs'][i_pose].ravel()
            pose_params[1, i_pose, :] = calibs[np.nanargmin(repro_errors)]['tvecs'][i_pose].ravel()
            pose_params_calcd.append(i_pose)

    # If the pose params are not calculated in the previous step,
    # they are taken from the nearest pose as initial estimates
    for i_pose in range(pose_params.shape[1]):
        if i_pose not in pose_params_calcd:
            nearest_i_pose = helper.nearest_element(i_pose, pose_params_calcd)

            pose_params[0, i_pose, :] = np.copy(pose_params[0, nearest_i_pose, :])
            pose_params[1, i_pose, :] = np.copy(pose_params[1, nearest_i_pose, :])

    return pose_params


def make_free_parameter_mask(calibs, opts_free_vars, coord_cam_idx):
    rvecs_cam_mask = np.zeros(shape=(len(calibs), 3), dtype=bool)
    tvecs_cam_mask = np.zeros(shape=(len(calibs), 3), dtype=bool)
    cam_mats_mask = np.zeros(shape=(len(calibs), 3, 3), dtype=bool)

    xis_mask = np.zeros(shape=(len(calibs), 1), dtype=bool)
    ks_mask = np.zeros(shape=(len(calibs), 5), dtype=bool)

    for i in range(len(calibs)):
        rvecs_cam_mask[i, :] = opts_free_vars[i]['cam_pose']
        tvecs_cam_mask[i, :] = opts_free_vars[i]['cam_pose']
        cam_mats_mask[i, :] = opts_free_vars[i]['A']

        xis_mask[i, :] = opts_free_vars[i]['xi']
        ks_mask[i, :] = opts_free_vars[i]['k'] == 1

    # Position of coord cam is not free
    rvecs_cam_mask[coord_cam_idx, :] = False
    tvecs_cam_mask[coord_cam_idx, :] = False

    pose_mask = np.ones(shape=(calibs[0]['tvecs'].shape[0], 2, 3), dtype=bool)
    pose_mask[:] = opts_free_vars[0]['board_poses']

    return np.concatenate((
        rvecs_cam_mask.ravel(),
        tvecs_cam_mask.ravel(),
        cam_mats_mask.ravel(),
        xis_mask.ravel(),
        ks_mask.ravel(),
        pose_mask.ravel()),
        axis=0)


def unravel_to_calibs(vars_opt, args):
    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams, n_boards = make_vars_full(vars_opt, args, verbose=False)

    # Unravel inputs.
    (rvecs_cams, tvecs_cams, cam_matrices, xis, ks,
     rvecs_boards, tvecs_boards) = unravel_vars_full(vars_full, n_cams, n_boards)

    calibs = [
        {
            'rvec_cam': rvecs_cams[i_cam],
            'tvec_cam': tvecs_cams[i_cam],
            'A': cam_matrices[i_cam],
            'xi': xis[i_cam],
            'k': ks[i_cam],
        }
        for i_cam in range(n_cams)
    ]

    return calibs, rvecs_boards, tvecs_boards

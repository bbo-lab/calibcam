# import multiprocessing
# from joblib import Parallel, delayed

import numpy as np

# This could also be done dynamically, based on opts ...
from calibcam.opt_vmapgrad.optfunctions import obj_fcn_wrapper, obj_fcn_jacobian_wrapper, get_obj_fcn_derivatives  # noqa
# from calibcam.opt_jacfwd.optfunctions import obj_fcn_wrapper, obj_fcn_jacobian_wrapper, get_obj_fcn_derivatives  # Calculating Jacobians would be much more straightforward, but seems to be prohibitively slow ...

def make_vars_full(vars_opt, args, verbose=False):
    n_cams = len(args['frames_masks'])

    # Update full set of vars with free wars
    vars_full = args['vars_full']
    if verbose:
        print(vars_full[0:7])
    mask_opt = args['mask_opt']
    vars_full[mask_opt] = vars_opt

    if verbose:
        print(mask_opt[0:7])
        print(vars_full[0:7])

    return vars_full, n_cams


def unravel_vars_full(vars_full, n_cams):
    n_cam_param_list = np.array([3, 3, 9, 5])  # r, t, A, k
    n_cam_params = n_cam_param_list.sum(dtype=int)

    p_idx = 0
    st_idx = n_cam_param_list[0:p_idx].sum(dtype=int) * n_cams
    rvecs_cams = vars_full[st_idx:st_idx + n_cam_param_list[p_idx] * n_cams].reshape(n_cam_param_list[p_idx], -1).T

    p_idx = 1
    st_idx = n_cam_param_list[0:p_idx].sum(dtype=int) * n_cams
    tvecs_cams = vars_full[st_idx:st_idx + n_cam_param_list[p_idx] * n_cams].reshape(n_cam_param_list[p_idx], -1).T

    p_idx = 2
    st_idx = n_cam_param_list[0:p_idx].sum(dtype=int) * n_cams
    cam_matrices = vars_full[st_idx:st_idx + n_cam_param_list[p_idx] * n_cams].reshape(3, 3, -1).transpose((2, 0, 1))

    p_idx = 3
    st_idx = n_cam_param_list[0:p_idx].sum(dtype=int) * n_cams
    ks = vars_full[st_idx:st_idx + n_cam_param_list[p_idx] * n_cams].reshape(n_cam_param_list[p_idx], -1).T

    board_pose_vars = vars_full[n_cams * n_cam_params:]
    rvecs_boards = board_pose_vars[0:int(board_pose_vars.size / 2)].reshape(-1, 3)
    tvecs_boards = board_pose_vars[int(board_pose_vars.size / 2):].reshape(-1, 3)

    return rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards


def make_initialization(calibs, frames_masks, opts, k_to_zero=True):
    # k_to_zero determines if non-free ks get set to 0 (for limited distortion model) or are kept (usually
    # when not optimizing distortion at all in the given step)
    opts_free_vars = opts['free_vars']

    camera_params = np.zeros(shape=(
        len(calibs),
        3 + 3 + 9 + 5  # r + t + A + k
    ))

    for calib, param in zip(calibs, camera_params):
        param[0:3] = calib['rvec_cam']
        param[3:6] = calib['tvec_cam']
        param[6:15] = calib['A'].ravel()
        if k_to_zero:
            idxs = (15 + np.where(opts_free_vars['k'])[0])
            param[idxs] = calib['k'][0][opts_free_vars['k']]
        else:
            param[15:20] = calib['k']

    # camera_params are raveled with one scalar parameter for all cams grouped
    # pose_params are raveled with first all rvecs and then all tvecs (for faster unraveling in obj_fun)
    camera_params = camera_params.T.ravel()
    pose_params = make_common_pose_params(calibs, frames_masks).ravel()

    vars_full = np.concatenate((camera_params, pose_params), axis=0)
    mask_free = make_free_parameter_mask(calibs, frames_masks, opts_free_vars, opts['coord_cam'])
    vars_free = vars_full[mask_free]

    return vars_free, vars_full, mask_free


def make_common_pose_params(calibs, frames_masks):
    pose_idxs = np.where(np.any(frames_masks, axis=0))[0]  # indexes into full frame range
    pose_params = np.zeros(shape=(2, pose_idxs.size, 3))
    # TODO Instead using pose from first available cam, it should be averaged over all available cams.
    # See pose_estimation.estimate_cam_poses for averaging poses
    # This might require fixing the other cam poses in calibration, see respective TODO in pose_estimation
    # calib = calibs[opts['coord_cam']]
    # frames_mask_cam = frames_masks[opts['coord_cam']]
    for i_pose, pose_idx in enumerate(pose_idxs):  # Loop through the poses (frames that have a board pose)
        for calib, frames_mask_cam in zip(calibs, frames_masks):  # Loop through cameras ...
            if np.all(pose_params[0, i_pose, :] == 0) and frames_mask_cam[pose_idx]:  # ... and check if frame present
                frame_idxs_cam = np.where(frames_mask_cam)[0]  # Frame indexes corresponding to available rvecs/tvecs
                pose_params[0, i_pose, :] = calib['rvecs'][frame_idxs_cam == pose_idx].ravel()
                pose_params[1, i_pose, :] = calib['tvecs'][frame_idxs_cam == pose_idx].ravel()

    return pose_params


def make_free_parameter_mask(calibs, frames_masks, opts_free_vars, coord_cam_idx):
    camera_mask = np.ones(shape=(
        len(calibs),
        3 + 3 + 9 + 5  # r + t + A + k
    ), dtype=bool)

    camera_mask[:, 0:3] = opts_free_vars['cam_pose']
    camera_mask[:, 3:6] = opts_free_vars['cam_pose']
    camera_mask[:, 6:15] = opts_free_vars['A'].ravel()
    camera_mask[:, 15:20] = opts_free_vars['k']

    # Position of coord cam is not free
    camera_mask[coord_cam_idx, 0:6] = False
    
    pose_idxs = np.where(np.any(frames_masks, axis=0))[0]  # indexes into full frame range
    pose_mask = np.ones(shape=(pose_idxs.size, 2, 3), dtype=bool)
    pose_mask[:] = opts_free_vars['board_poses']

    return np.concatenate((camera_mask.T.ravel(), pose_mask.ravel()), axis=0)


def unravel_to_calibs(vars_opt, args):
    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = make_vars_full(vars_opt, args, verbose=True)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = unravel_vars_full(vars_full, n_cams)

    calibs = [
        {
            'rvec_cam': rvecs_cams[i_cam],
            'tvec_cam': tvecs_cams[i_cam],
            'A': cam_matrices[i_cam],
            'k': ks[i_cam],
        }
        for i_cam in range(n_cams)
    ]

    return calibs, rvecs_boards, tvecs_boards

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa


def make_initialization(calibs, frame_masks, opts, k_to_zero=True):
    # k_to_zero determines if non-free ks get set to 0 (for limited distortion model) or are kept (usually
    # when not optimizing distortion at all in the given step)
    opts_free_vars = opts['free_vars']

    camera_params = np.zeros(shape=(
        len(calibs),
        3 + 3 + 9 + 5  # r + t + A + k
    ))

    for calib, param in zip(calibs, camera_params):
        print(R.from_rotvec(calib['rvec_cam']).as_matrix())
        print(calib['tvec_cam'])
        print()
        param[0:3] = calib['rvec_cam']
        param[3:6] = calib['tvec_cam']
        param[6:15] = calib['A'].ravel()
        if k_to_zero:
            idxs = (15 + np.where(opts_free_vars['k'])[0])
            param[idxs] = calib['k'][0][opts_free_vars['k']]
        else:
            param[15:20] = calib['k']

    pose_idxs = np.where(np.any(frame_masks, axis=0))[0]  # indexes into full frame range
    pose_params = np.empty(shape=(pose_idxs.size, 2, 3))
    for pose_idx, pose_frame in zip(pose_idxs, pose_params):
        for calib, frame_mask_cam in zip(calibs, frame_masks):
            if frame_mask_cam[pose_idx]:
                frame_idxs_cam = np.where(frame_mask_cam)[0]
                pose_frame[0, :] = calib['rvecs'][frame_idxs_cam == pose_idx].ravel()
                pose_frame[1, :] = calib['tvecs'][frame_idxs_cam == pose_idx].ravel()

    vars_full = np.concatenate((camera_params.ravel(), pose_params.ravel()), axis=0)
    mask_free = make_free_parameter_mask(calibs, frame_masks, opts_free_vars)
    vars_free = vars_full[mask_free]

    return vars_free, vars_full, mask_free


def make_free_parameter_mask(calibs, frame_masks, opts_free_vars):
    camera_mask = np.ones(shape=(
        len(calibs),
        3 + 3 + 9 + 5  # r + t + A + k
    ), dtype=bool)

    camera_mask[:, 0:3] = opts_free_vars['cam_pose']
    camera_mask[:, 3:6] = opts_free_vars['cam_pose']
    camera_mask[:, 6:15] = opts_free_vars['A'].ravel()
    camera_mask[:, 10:15] = opts_free_vars['k']

    pose_idxs = np.where(np.any(frame_masks, axis=0))[0]  # indexes into full frame range
    pose_mask = np.ones(shape=(pose_idxs.size, 2, 3), dtype=bool)
    pose_mask[:] = opts_free_vars['board_poses']

    return np.concatenate((camera_mask.ravel(), pose_mask.ravel()), axis=0)


def obj_fcn_free(vars_opt, args):
    vars_full = args['vars_full']
    mask_opt = args['mask_opt']

    vars_full[mask_opt] = vars_opt

    return obj_fcn(vars_full, args)


def obj_fcn(vars_full, args):
    pass

import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R  # noqa


def estimate_cam_poses(calibs_single, coord_cam):
    calibs = deepcopy(calibs_single)
    cams_oriented = np.zeros(len(calibs), dtype=bool)
    cams_oriented[coord_cam] = True

    frame_masks = np.asarray([cal['frame_mask'] for cal in calibs], dtype=bool)
    common_frame_mat = calc_common_frame_mat(frame_masks)

    # We allow some bonus to coord_cam as it might be beneficial to not have another cam as an inbetween step if the
    # difference in frame numbers is small. (Also good for testing if the propagation works.)
    common_frame_mat[:, coord_cam] = common_frame_mat[:, coord_cam]*1.1
    common_frame_mat[coord_cam, :] = common_frame_mat[:, coord_cam].T

    while not np.all(cams_oriented):
        # Find unoriented cam with the most overlaps with an oriented camera
        cams_oriented_idxs = np.where(cams_oriented)[0]
        cams_unoriented_idxs = np.where(~cams_oriented)[0]
        ori_nori_mat = common_frame_mat[cams_oriented]
        ori_nori_mat = ori_nori_mat[:, ~cams_oriented]
        ori_nori_mat = ori_nori_mat.reshape(cams_oriented.sum(), (~cams_oriented).sum())  # For masks with one entry
        max_row, max_col = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)

        refcam_idx = cams_oriented_idxs[max_row]
        oricam_idx = cams_unoriented_idxs[max_col]
        print(f"Orienting cam {oricam_idx} on cam {refcam_idx}")

        # Determine common frames
        common_frame_idxs = np.logical_and(frame_masks[refcam_idx], frame_masks[oricam_idx])
        refcam_common_mask = common_frame_idxs[frame_masks[refcam_idx]]
        oricam_common_mask = common_frame_idxs[frame_masks[oricam_idx]]

        # Calculate average transformation  from oricam to refcam coordinate system
        R_trans = (  # noqa
                R.from_rotvec(calibs[refcam_idx]['rvecs'][refcam_common_mask, :, 0]) *
                R.from_rotvec(calibs[oricam_idx]['rvecs'][oricam_common_mask, :, 0]).inv()
        ).mean()
        t_trans = (
                -R_trans.apply(
                    calibs[oricam_idx]['tvecs'][oricam_common_mask, :, 0]) +
                calibs[refcam_idx]['tvecs'][refcam_common_mask, :, 0]
        ).mean(axis=0).reshape((1, 3))

        calibs[oricam_idx]['rvecs'] = (
                R_trans *
                R.from_rotvec(calibs[oricam_idx]['rvecs'][:, :, 0])
        ).as_rotvec().reshape((-1, 3, 1))
        calibs[oricam_idx]['tvecs'] = (
                R_trans.apply(calibs[oricam_idx]['tvecs'][:, :, 0]) +
                t_trans
        ).reshape((-1, 3, 1))

        calibs[oricam_idx]['rvec_cam'] = (R_trans.inv() * R.from_rotvec(calibs[oricam_idx]['rvec_cam'])).as_rotvec()
        calibs[oricam_idx]['tvec_cam'] = R_trans.inv().apply(calibs[oricam_idx]['tvec_cam'] - t_trans)
        cams_oriented[oricam_idx] = True
    return calibs


def calc_common_frame_mat(frame_masks):
    n_cams = frame_masks.shape[0]
    common_frame_mat = np.zeros(shape=(n_cams, n_cams), dtype=int)

    for i in range(n_cams):
        for j in range(i, n_cams):
            common_frame_mat[i, j] = np.sum(frame_masks[i, :] & frame_masks[j, :])
            common_frame_mat[j, i] = common_frame_mat[i, j]

    return common_frame_mat


def make_initialization_from(calibs, frame_masks, opts, k_to_zero=True):
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

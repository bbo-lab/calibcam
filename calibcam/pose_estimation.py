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

        # TODO check if this is really correct. rvecs and tvecs should be really similar between cams in the same frames
        # (but not necessary idxs, as different cams can contain different frame idxs)
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

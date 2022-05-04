import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R  # noqa

def estimate_cam_poses(calibs_single, coord_cam):
    calibs = deepcopy(calibs_single)
    cams_oriented = np.zeros(len(calibs), dtype=bool)
    cams_oriented[coord_cam] = True

    frame_masks = np.asarray([cal['frame_mask'] for cal in calibs], dtype=bool)
    common_frame_mat = calc_common_frame_mat(frame_masks)

    while not np.all(cams_oriented):
        # Find cam with the most overlaps with an oriented camera
        cams_oriented_idxs = np.where(cams_oriented)
        cams_unoriented_idxs = np.where(~cams_oriented)
        ori_nori_mat = common_frame_mat[cams_oriented, ~cams_oriented]

        max_row, max_col = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)

        refcam_idx = cams_oriented_idxs[max_row]
        oricam_idx = cams_unoriented_idxs[max_col]
        common_frame_idxs = np.logical_and(frame_masks[refcam_idx, :], frame_masks[oricam_idx, :])
        refcam_vec_mask = common_frame_idxs[frame_masks[refcam_idx]]
        oricam_vec_mask = common_frame_idxs[frame_masks[oricam_idx]]

        R_trans = (R.from_rotvec(calibs[refcam_idx]['rvecs'])[refcam_vec_mask] * R.from_rotvec(calibs[oricam_idx]['rvecs'])[oricam_vec_mask].inv).mean()
        t_trans = (calibs[refcam_idx]['tvecs'][refcam_vec_mask] - calibs[oricam_idx]['tvecs'][oricam_vec_mask]).mean(axis=1)

        calibs[oricam_idx]['rvecs'] = calibs[coord_cam]['rvecs']  # TODO This is wrong, that cam has other poses than coord_cam
        calibs[oricam_idx]['tvecs'] = calibs[coord_cam]['tvecs']

        calibs[oricam_idx]['rvec_cam'] = R_trans.inv() * R.from_rotvec(calibs[refcam_idx]['rvec_cam'])
        calibs[oricam_idx]['tvec_cam'] = R_trans.inv() * (calibs[refcam_idx]['tvec_cam'] - t_trans)

    return calibs


def calc_common_frame_mat(frame_masks):
    n_cams = frame_masks.shape[0]
    common_frame_mat = np.zeros(shape=(n_cams, n_cams), dtype=int)

    for i in range(n_cams):
        for j in range(i, n_cams):
            common_frame_mat[i, j] = np.sum(frame_masks[i, :] & frame_masks[j, :])
            common_frame_mat[j, i] = common_frame_mat[i, j]

    return common_frame_mat

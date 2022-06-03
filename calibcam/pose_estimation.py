import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R  # noqa


def estimate_cam_poses(calibs_single, coord_cam, corner_ids=None, required_corners=None):
    calibs = deepcopy(calibs_single)
    cams_oriented = np.zeros(len(calibs), dtype=bool)
    cams_oriented[coord_cam] = True

    frames_masks = np.asarray([cal['frames_mask'] for cal in calibs], dtype=bool)
    frames_masks_req = get_required_corners_masks(frames_masks, corner_ids=corner_ids,
                                                  required_corners=required_corners)
    common_frame_mat = calc_common_frame_mat(frames_masks_req)

    rs = np.empty(shape=frames_masks.shape + (3,))
    rs[:] = np.nan
    ts = np.empty(shape=frames_masks.shape + (3,))
    ts[:] = np.nan
    for i_cam, calib in enumerate(calibs_single):
        rs[i_cam][calib['frames_mask']] = calib['rvecs'][..., 0]
        ts[i_cam][calib['frames_mask']] = calib['tvecs'][..., 0]

    # We allow some bonus to coord_cam as it might be beneficial to not have another cam as an inbetween step if the
    # difference in frame numbers is small. (Also good for testing if the propagation works.)
    common_frame_mat[:, coord_cam] = common_frame_mat[:, coord_cam] * 1.5
    common_frame_mat[coord_cam, :] = common_frame_mat[:, coord_cam].T

    while not np.all(cams_oriented):
        # Find unoriented cam with the most overlaps with an oriented camera
        ori_nori_mat = common_frame_mat.copy()
        ori_nori_mat[~cams_oriented] = -1
        ori_nori_mat[:, cams_oriented] = -1
        refcam_idx, oricam_idx = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)
        print(f"Orienting cam {oricam_idx} on cam {refcam_idx} on {ori_nori_mat[refcam_idx, oricam_idx]} poses")

        # Determine common frames
        common_frame_mask = frames_masks_req[refcam_idx] & frames_masks_req[oricam_idx]

        # Calculate average transformation from oricam to refcam coordinate system
        Rs_trans = (
                R.from_rotvec(rs[refcam_idx, common_frame_mask]) *
                R.from_rotvec(rs[oricam_idx, common_frame_mask]).inv()
        )
        R_trans = Rs_trans.mean()
        print(f"Mean rvec deviation: {np.mean(np.abs((R_trans.inv() * Rs_trans).as_rotvec()), axis=0)}")
        ts_trans = (
                ts[refcam_idx, common_frame_mask]
                - R_trans.apply(ts[oricam_idx, common_frame_mask])
        )
        t_trans = ts_trans.mean(axis=0).reshape((1, 3))
        print(f"Mean tvec deviation: {np.mean(np.abs(ts_trans - t_trans), axis=0)}")

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


def calc_common_frame_mat(frames_masks, corner_ids=None, required_corners=None):
    n_cams = frames_masks.shape[0]
    common_frame_mat = np.zeros(shape=(n_cams, n_cams), dtype=int)

    for i in range(n_cams):
        for j in range(i, n_cams):
            if required_corners is None or \
                    (np.all(corner_ids[i] in required_corners) and np.all(corner_ids[j] in required_corners)):
                common_frame_mat[i, j] = np.sum(frames_masks[i, :] & frames_masks[j, :])
                common_frame_mat[j, i] = common_frame_mat[i, j]

    return common_frame_mat


def get_required_corners_masks(frames_masks, corner_ids=None, required_corners=None):
    if corner_ids is None or required_corners is None:
        return frames_masks

    frames_masks_req = frames_masks.copy()
    for i_cam, frames_mask in enumerate(frames_masks_req):
        frames_idxs = np.where(frames_mask)[0]
        for i_cpose, frames_idx in enumerate(frames_idxs):
            frames_mask[frames_idx] = np.all(np.isin(required_corners, corner_ids[i_cam][i_cpose]))

    return frames_masks_req

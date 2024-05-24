import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R  # noqa
from bbo.geometry import RigidTransform

def build_initialized_calibs(calibs_single, opts, corners=None, required_corner_idxs=None):
    calibs = deepcopy(calibs_single)

    for i_calib, calib in enumerate(calibs):
        calib["rvec_cam"] = opts["init_extrinsics"]["rvecs_cam"][i_calib]
        calib["tvec_cam"] = opts["init_extrinsics"]["tvecs_cam"][i_calib]

        cam2camsystem = RigidTransform(rotation=calib["rvec_cam"], translation=calib["tvec_cam"],
                                       rotation_type="rotvec").inv()
        board2cam = RigidTransform(rotation=calib["rvecs"], translation=calib["tvecs"], rotation_type="rotvec")
        board2camsystem = cam2camsystem * board2cam

        calib["rvecs"] = board2camsystem.get_rotation().as_rotvec()
        calib["tvecs"] = board2camsystem.get_translation()

    return calibs


def estimate_cam_poses(calibs_single, opts, corners=None, required_corner_idxs=None):
    calibs = deepcopy(calibs_single)

    cams_oriented = np.zeros(len(calibs), dtype=bool)
    cams_oriented[opts['coord_cam']] = True

    # Only use frames that have these corners detected (usually "corner corners" for full boards)
    frames_masks_req = get_required_corners_masks(corners=corners,
                                                  required_corner_idxs=required_corner_idxs
                                                  if opts['pose_estimation']['use_required_corners']
                                                  else None)

    # Opencv omnidirectional camera calibration does not calculate extrisic paraemters for all the frames. In such case,
    # it is necessary to omit such frames from estimating camera poses.
    frames_rs_calcd = [calib["frames_mask"] for calib in calibs_single]
    # frames_rs_calcd = np.all(np.asarray(frames_rs_calcd), axis=0)
    frames_rs_calcd = np.asarray(frames_rs_calcd)
    frames_masks_req = frames_masks_req & frames_rs_calcd

    # n_cam x n_cam matrix of frames between two cams
    common_frame_mat = calc_common_frame_mat(frames_masks_req)

    # We allow some bonus to coord_cam as it might be beneficial to not have another cam as an inbetween step if the
    # difference in frame numbers is small. (Also good for testing if the propagation works.)
    common_frame_mat[:, opts['coord_cam']] = common_frame_mat[:, opts['coord_cam']] * 10
    common_frame_mat[opts['coord_cam'], :] = common_frame_mat[:, opts['coord_cam']].T

    rs = np.asarray([calib["rvecs"] for calib in calibs_single])
    ts = np.asarray([calib["tvecs"] for calib in calibs_single])

    while not np.all(cams_oriented):
        # Find unoriented cam with the most overlaps with an oriented camera
        ori_nori_mat = common_frame_mat.copy()
        ori_nori_mat[~cams_oriented] = -1
        ori_nori_mat[:, cams_oriented] = -1
        refcam_idx, oricam_idx = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)
        print(
            f"Orienting cam {oricam_idx} on cam {refcam_idx} on {ori_nori_mat[refcam_idx, oricam_idx]} potential poses")

        r_error = np.Inf
        R_trans = None
        Rs_trans = None
        # Copy, we will remove frames this
        frames_masks_req_ori = frames_masks_req[oricam_idx].copy()
        while r_error >= opts['common_pose_r_err']:
            # Remove frames with too high deviation from frames_mask
            # In single camera calibration misestimation of board pose may occur where the board is tilted around one of
            #  its axes relative  to the camera axis: c ----> / instead of c ----> \
            #  theses tilts do not yield a consistent alternative position and may thus be removed by iteratively
            #  removing the highest deviations.
            if R_trans is not None and Rs_trans is not None:
                # Remove frame with the highest error
                common_frame_idxs = np.where(common_frame_mask)[0]
                frames_masks_req_ori[
                    common_frame_idxs[
                        np.argmax(np.sum(np.abs((R_trans.inv() * Rs_trans).as_rotvec()), axis=1))
                    ]
                ] = False

            # Determine common frames
            common_frame_mask = frames_masks_req[refcam_idx] & frames_masks_req_ori

            # Calculate average transformation from oricam to refcam coordinate system
            Rs_trans = (
                    R.from_rotvec(rs[refcam_idx, common_frame_mask]) *
                    R.from_rotvec(rs[oricam_idx, common_frame_mask]).inv()
            )
            R_trans = Rs_trans.mean()
            r_error = np.max(np.sum(np.abs((R_trans.inv() * Rs_trans).as_rotvec()), axis=1))

            ts_trans = (
                    ts[refcam_idx, common_frame_mask]
                    - R_trans.apply(ts[oricam_idx, common_frame_mask])
            )

            t_trans = ts_trans.mean(axis=0).reshape((1, 3))

        print(f"Chose {np.sum(common_frame_mask)} poses.")
        print(f"Mean rvec deviation: {np.mean(np.abs((R_trans.inv() * Rs_trans).as_rotvec()), axis=0)}")
        print(f"Mean tvec deviation: {np.mean(np.abs(ts_trans - t_trans), axis=0)}")

        nanposemask = ~np.isnan(calibs[oricam_idx]['rvecs'][:, 0])
        calibs[oricam_idx]['rvecs'][nanposemask] = (
                R_trans *
                R.from_rotvec(calibs[oricam_idx]['rvecs'][nanposemask])
        ).as_rotvec().reshape((-1, 3))
        calibs[oricam_idx]['tvecs'] = (
                R_trans.apply(calibs[oricam_idx]['tvecs']) +
                t_trans
        ).reshape((-1, 3))

        calibs[oricam_idx]['rvec_cam'] = (R_trans.inv() * R.from_rotvec(calibs[oricam_idx]['rvec_cam'])).as_rotvec()
        calibs[oricam_idx]['tvec_cam'] = R_trans.inv().apply(calibs[oricam_idx]['tvec_cam'] - t_trans)
        cams_oriented[oricam_idx] = True

    return calibs


def calc_common_frame_mat(frames_masks):
    n_cams = frames_masks.shape[0]
    common_frame_mat = np.zeros(shape=(n_cams, n_cams), dtype=int)

    for i in range(n_cams):
        for j in range(i, n_cams):
            common_frame_mat[i, j] = np.sum(frames_masks[i, :] & frames_masks[j, :])
            common_frame_mat[j, i] = common_frame_mat[i, j]

    return common_frame_mat


def get_required_corners_masks(corners, required_corner_idxs=None):
    if required_corner_idxs is None:
        return np.sum(~np.isnan(corners[:, :, :, 1]), axis=2) > 0
    else:
        return np.sum(~np.isnan(corners[:, :, required_corner_idxs, 1]), axis=2) > 0

from copy import deepcopy

import numpy as np
from bbo.geometry import RigidTransform
from scipy.spatial.transform import Rotation as R  # noqa

from calibcamlib import Detections


def build_initialized_calibs(calibs_single, opts, detections: Detections):
    calibs = deepcopy(calibs_single)
    detection_idxs = detections.to_array()["detection_idxs"]

    for i_calib, calib in enumerate(calibs):
        calib["rvec_cam"] = opts["init_extrinsics"]["rvecs_cam"][i_calib]
        calib["tvec_cam"] = opts["init_extrinsics"]["tvecs_cam"][i_calib]

        cam2camsystem = RigidTransform(rotation=calib["rvec_cam"], translation=calib["tvec_cam"],
                                       rotation_type="rotvec").inv()
        board2cam = RigidTransform(rotation=calib["rvecs"], translation=calib["tvecs"], rotation_type="rotvec")
        board2camsystem = cam2camsystem * board2cam

        calib["rvecs"] = np.full((detections.get_n_frames(), 3), np.nan)
        calib["tvecs"] = np.full((detections.get_n_frames(), 3), np.nan)
        mask = np.isin(detection_idxs, calib["detection_idxs"])
        calib["rvecs"][mask] = board2camsystem.get_rotation().as_rotvec()
        calib["tvecs"][mask] = board2camsystem.get_translation()

        orig_frame_idxs = calib["frame_idxs"]
        calib["frame_idxs"] = np.full(detections.get_n_frames(), -1, dtype=int)
        calib["frame_idxs"][mask] = orig_frame_idxs
        calib["detection_idxs"] = detection_idxs

    return calibs


def estimate_cam_poses(calibs_single, opts, detections=None, required_corner_idxs=None):
    calibs = deepcopy(calibs_single)
    detections_array = detections.to_array()

    cams_oriented = np.zeros(len(calibs), dtype=bool)
    cams_oriented[opts['coord_cam']] = True

    n_cams = detections.get_n_cams()
    n_frames = detections.get_n_frames()

    assert n_cams == len(calibs), "Number of detections must match number of single calibrations"

    if len(opts['init_extrinsics_frames']) == 0:
        calibs = estimate_cam_poses_multiframe(calibs, cams_oriented, detections, detections_array, n_cams, n_frames,
                                               opts,
                                               required_corner_idxs)
    elif len(opts['init_extrinsics_frames']) > 1:
        raise ValueError("Multiple independent cameras are not supported yet")
    else:
        ie_fr_idx = opts['init_extrinsics_frames'][0]
        ie_fr_idx = np.where(detections_array['frame_idxs'][0] == ie_fr_idx)[0]
        if len(ie_fr_idx)==0:
            raise ValueError(f"init_extrinsics_frames {opts['init_extrinsics_frames'][0]} is not part of cam 0 detections")
        else:
            ie_fr_idx = ie_fr_idx[0]
        ie_ideal2camsys = RigidTransform(rotation=calibs[0]["rvecs"][ie_fr_idx],
                                         translation=calibs[0]["tvecs"][ie_fr_idx],
                                         rotation_type="rotvec")
        # calibs = estimate_cam_poses_singleframe(calibs, cams_oriented, detections, detections_array, n_cams, n_frames,
        #                                        opts,
        #                                        required_corner_idxs)
        for i_calib, calib in enumerate(calibs):
            ie_ideal2cam = RigidTransform(rotation=calib["rvecs"][ie_fr_idx],
                                          translation=calib["tvecs"][ie_fr_idx],
                                          rotation_type="rotvec")
            camsys2cam = ie_ideal2cam * ie_ideal2camsys.inv()
            calib["rvec_cam"] = camsys2cam.get_rotation().as_rotvec()
            calib["tvec_cam"] = camsys2cam.get_translation()

            ideal2cam = RigidTransform(rotation=calib["rvecs"],
                                       translation=calib["tvecs"],
                                       rotation_type="rotvec")
            ideal2camsys = camsys2cam.inv() * ideal2cam
            calib["rvecs"] = ideal2camsys.get_rotation().as_rotvec()
            calib["tvecs"] = ideal2camsys.get_translation()
    return calibs


def estimate_cam_poses_multiframe(calibs, cams_oriented, detections, detections_array, n_cams, n_frames, opts,
                                  required_corner_idxs):
    rs = np.full((n_cams, n_frames, 3), np.nan)
    ts = np.full((n_cams, n_frames, 3), np.nan)
    frames_masks_req = np.zeros((n_cams, n_frames), dtype=bool)
    for i_calib, calib in enumerate(calibs):
        mask = np.isin(detections_array["detection_idxs"], calib["detection_idxs"])
        rs[i_calib, mask] = calib["rvecs"]
        ts[i_calib, mask] = calib["tvecs"]
        frames_masks_req[i_calib, mask] = True
    # Only use frames that have these corners detected (usually "corner corners" for full boards)
    discard_detection_idxs = get_discard_detection_idxs(detections=detections,
                                                        required_corner_idxs=required_corner_idxs
                                                        if opts['pose_estimation']['use_required_corners']
                                                        else None)
    for i_cam, (fmr, dfi, rs_cam) in enumerate(zip(frames_masks_req, discard_detection_idxs, rs)):
        mask = np.isin(detections_array["detection_idxs"], dfi)
        fmr[mask] = False
        fmr[:] &= np.all(~np.isnan(rs_cam), axis=1)
        print(f"Found {np.sum(fmr):04d} frames pose estimation of for cam {i_cam:03d}")
    # n_cam x n_cam matrix of frames between two cams
    common_frame_mat = calc_common_frame_mat(frames_masks_req)
    # We allow some bonus to coord_cam as it might be beneficial to not have another cam as an inbetween step if the
    # difference in frame numbers is small. (Also good for testing if the propagation works.)
    common_frame_mat[:, opts['coord_cam']] = common_frame_mat[:, opts['coord_cam']] * 10
    common_frame_mat[opts['coord_cam'], :] = common_frame_mat[:, opts['coord_cam']].T
    while not np.all(cams_oriented):
        # Find unoriented cam with the most overlaps with an oriented camera
        ori_nori_mat = common_frame_mat.copy()
        ori_nori_mat[~cams_oriented] = -1
        ori_nori_mat[:, cams_oriented] = -1
        refcam_idx, oricam_idx = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)
        print(
            f"Orienting cam {oricam_idx} on cam {refcam_idx} on {ori_nori_mat[refcam_idx, oricam_idx]} potential poses")

        r_error = np.inf
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
                common_detection_idxs = np.where(common_frame_mask)[0]
                frames_masks_req_ori[
                    common_detection_idxs[
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


def get_discard_detection_idxs(detections, required_corner_idxs=None, min_marker_count=4):
    markers = detections.to_array()
    marker_coords = markers["marker_coords"]
    if required_corner_idxs is None:
        return [markers["detection_idxs"][m]
                for m in np.sum(~np.isnan(marker_coords[:, :, :, 1]), axis=2) < min_marker_count]
    else:
        return [markers["detection_idxs"][m]
                for m in np.any(np.isnan(marker_coords[:, :, required_corner_idxs, 1]), axis=2)]

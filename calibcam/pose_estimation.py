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
    rs_b02cw = np.full((n_cams, n_frames, 3), np.nan)
    ts_b02cw = np.full((n_cams, n_frames, 3), np.nan)
    frames_masks_req = np.zeros((n_cams, n_frames), dtype=bool)
    for i_calib, calib in enumerate(calibs):
        mask = np.isin(detections_array["detection_idxs"], calib["detection_idxs"])
        rs_b02cw[i_calib, mask] = calib["rvecs"]
        ts_b02cw[i_calib, mask] = calib["tvecs"]
        frames_masks_req[i_calib, mask] = True
    # Only use frames that have these corners detected (usually "corner corners" for full boards)
    discard_detection_idxs = get_discard_detection_idxs(detections=detections,
                                                        required_corner_idxs=required_corner_idxs
                                                        if opts['pose_estimation']['use_required_corners']
                                                        else None)
    for i_cam, (fmr, dfi, rs_cam) in enumerate(zip(frames_masks_req, discard_detection_idxs, rs_b02cw)):
        mask = np.isin(detections_array["detection_idxs"], dfi)
        fmr[mask] = False
        fmr[:] &= np.all(~np.isnan(rs_cam), axis=1)
        print(f"Found {np.sum(fmr):04d} frames pose estimation of for cam {i_cam:03d}")
    # n_cam x n_cam matrix of frames between two cams
    common_frame_mat = calc_common_frame_mat(frames_masks_req)
    # We allow some bonus to coord_cam as it might be beneficial to not have another cam as an inbetween step if the
    # difference in frame numbers is small. (Also good for testing if the propagation works.)
    common_frame_mat[:, opts['coord_cam']] *= 10
    common_frame_mat[opts['coord_cam'], :] *= 10
    while not np.all(cams_oriented):
        # Find unoriented cam with the most overlaps with an oriented camera
        ori_nori_mat = common_frame_mat.copy()
        ori_nori_mat[~cams_oriented] = -1
        ori_nori_mat[:, cams_oriented] = -1
        refcam_idx, oricam_idx = np.unravel_index(ori_nori_mat.argmax(), ori_nori_mat.shape)
        print(
            f"Orienting cam {oricam_idx} on cam {refcam_idx} on {ori_nori_mat[refcam_idx, oricam_idx]} potential poses")

        r_error = np.inf
        T_wo2wr = None
        Ts_wo2wr = None
        # Copy, we will remove frames this
        frames_masks_req_ori = frames_masks_req[oricam_idx].copy()
        while r_error >= opts['common_pose_r_err']:
            # Remove frames with too high deviation from frames_mask
            # In single camera calibration misestimation of board pose may occur where the board is tilted around one of
            #  its axes relative  to the camera axis: c ----> / instead of c ----> \
            #  these tilts do not yield a consistent alternative position and may thus be removed by iteratively
            #  removing the highest deviations.
            if T_wo2wr is not None and Ts_wo2wr is not None:
                # Remove frame with the highest error
                common_detection_idxs = np.where(common_frame_mask)[0]
                frames_masks_req_ori[
                    common_detection_idxs[
                        np.argmax(np.sum(np.abs((T_wo2wr.get_rotation().inv() *
                                                 Ts_wo2wr.get_rotation()).as_rotvec()), axis=1))
                    ]
                ] = False
            common_frame_mask = frames_masks_req[refcam_idx] & frames_masks_req_ori
            num_common = int(np.count_nonzero(common_frame_mask))
            if num_common == 0:
                raise RuntimeError(f"No common frames between cams {refcam_idx} and {oricam_idx}.")

            # Determine common frames
            common_frame_mask = frames_masks_req[refcam_idx] & frames_masks_req_ori

            # Transformations from ideal board space to reference world
            Ts_b02wr = RigidTransform(rotation=rs_b02cw[refcam_idx, common_frame_mask],
                                      translation=ts_b02cw[refcam_idx, common_frame_mask],
                                      rotation_type="rotvec")
            # Transformations from ideal board space to orientee world
            Ts_b02wo = RigidTransform(rotation=rs_b02cw[oricam_idx, common_frame_mask],
                                      translation=ts_b02cw[oricam_idx, common_frame_mask],
                                      rotation_type="rotvec")
            # Transformations from orientee world to reference world
            Ts_wo2wr = Ts_b02wr * Ts_b02wo.inv()

            if len(Ts_wo2wr) > 1:
                T_wo2wr = Ts_wo2wr.nanmean()
            else:
                T_wo2wr = Ts_wo2wr

            errs_R = np.linalg.norm(
                (T_wo2wr.get_rotation().inv() * Ts_wo2wr.get_rotation()).as_rotvec(), axis=1
            )
            r_error = float(np.max(errs_R))

        print(f"Chose {np.sum(common_frame_mask)} poses.")
        print(f"Mean rvec deviation: "
              f"{np.mean(np.abs((T_wo2wr.get_rotation().inv() * Ts_wo2wr.get_rotation()).as_rotvec()), axis=0)}")
        print(f"Mean tvec deviation: "
              f"{np.mean(np.abs(Ts_wo2wr.get_translation() - T_wo2wr.get_translation()), axis=0)}")

        nanposemask = ~np.isnan(calibs[oricam_idx]['rvecs'][:, 0])

        # Transformations from ideal board space to orientee world
        Ts_b02wo = RigidTransform(rotation=calibs[oricam_idx]['rvecs'][nanposemask],
                                 translation=calibs[oricam_idx]['tvecs'][nanposemask],
                                 rotation_type="rotvec")
        # Transformations from ideal board space to reference world
        Ts_b02wr = T_wo2wr * Ts_b02wo
        calibs[oricam_idx]['rvecs'][nanposemask] = Ts_b02wr.get_rotation().as_rotvec().reshape((-1, 3))
        calibs[oricam_idx]['tvecs'][nanposemask] = Ts_b02wr.get_translation().reshape((-1, 3))

        # Transformations from orientee world to orientee camera
        T_wo2co = RigidTransform(rotation=calibs[oricam_idx]['rvec_cam'],
                                 translation=calibs[oricam_idx]['tvec_cam'],
                                 rotation_type="rotvec")
        # Transformations from reference world to orientee camera
        T_wr2co =  T_wo2co * T_wo2wr.inv()
        calibs[oricam_idx]['rvec_cam'] = T_wr2co.get_rotation().as_rotvec()
        calibs[oricam_idx]['tvec_cam'] = T_wr2co.get_translation()
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

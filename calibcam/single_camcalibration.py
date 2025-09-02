import numpy as np
import cv2

from ccvtools import rawio  # noqa

from calibcam import helper, board
from calibcam.board import Board
from calibcam.detection import Detections


def calibrate_single_camera(detections_cam: Detections, sensor_size, board: Board, opts, mask=None, calib_init=None):
    if calib_init is not None:
        A = calib_init['A']
        k = calib_init['k']
        if "xi" in calib_init:
            xi = calib_init['xi'].reshape(1, -1)
        else:
            xi=0
    else:
        A = None
        xi = None
        k = None


    detections_cam_array = detections_cam.to_array()
    if mask is None:
        mask = np.sum(~np.isnan(detections_cam_array["marker_coords"][0, :, :, 1]),
                      axis=1) >= opts['corners_min_n']  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.sum(mask)

    if n_used_frames == 0:
        return {}

    detections_array_use = detections_cam_array["marker_coords"][:, mask]
    frame_idxs_use = detections_cam_array["frame_idxs"][mask]
    ids_use = detections_cam_array["marker_ids"]

    if opts['free_vars']['xi']:
        # Omnidir camera model
        if k is not None:
            k = k.reshape(1, -1)[:, :4]

        # Object points for each frame must match corners
        board_points = board.get_board_points()
        object_points = np.zeros((*detections_array_use[0].shape[0:2], 3))
        object_points[:] = board_points
        object_points[np.isnan(detections_array_use[0, :, :, 1])] = np.nan

        cal_res = cv2.omnidir.calibrate(object_points[0],  # noqa
                                        detections_array_use[0],
                                        sensor_size,
                                        A,
                                        xi,
                                        k,
                                        **opts['aruco_calibration'])

        retval, A, xi, k = cal_res[:4]
        others = cal_res[4:]

        k = np.concatenate((k.squeeze(), [0.0]))
        # Opencv Omnidir calibrate does not use all the given frames for calibration.
        # The extrisic paraemters are not calculated for these frames.
        mask_singlecam_calib = np.zeros_like(mask, dtype=bool)
        mask_singlecam_calib[np.where(mask)[0][others[2].flatten()]] = True

    else:
        detections_list_use = Detections.from_array({
            "marker_coords": detections_array_use,
            "frame_idxs": frame_idxs_use,
            "marker_ids": ids_use,
        }).to_list()

        # Pinhole camera model
        cal_res = cv2.aruco.calibrateCameraCharucoExtended(detections_list_use["marker_coords"][0],  # noqa
                                                           detections_list_use["marker_ids"][0],
                                                           board.get_cv2_board(),
                                                           sensor_size,
                                                           A,
                                                           k,
                                                           **opts['aruco_calibration'])

        retval, A, k = cal_res[:3]
        others = cal_res[3:]

        if xi is None:
            xi = [0.0]
        mask_singlecam_calib = np.copy(mask)

    rvecs = np.full(shape=(len(frame_idxs_use), 3), fill_value=np.nan)
    tvecs = np.full(shape=(len(frame_idxs_use), 3), fill_value=np.nan)

    rvecs[mask_singlecam_calib, :] = np.asarray(others[0])[..., 0]
    tvecs[mask_singlecam_calib, :] = np.asarray(others[1])[..., 0]

    cal = {
        'rvec_cam': np.asarray([0., 0., 0.]),
        'tvec_cam': np.asarray([0., 0., 0.]),
        'A': np.asarray(A),
        'xi': np.asarray(xi),
        'k': np.asarray(k).ravel(),
        'rvecs': np.asarray(rvecs),
        'tvecs': np.asarray(tvecs),
        'repro_error': retval,
        'frames_idxs': frame_idxs_use,
    }

    if not opts['free_vars']['xi']:
        # Note that from here on values are NOT expanded to full frames range, see frames_mask
        cal['std_intrinsics'] = others[2]
        cal['std_extrinsics'] = others[3]
        cal['per_view_errors'] = others[4]

    print('Finished single camera calibration.')
    return cal

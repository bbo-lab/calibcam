import cv2
import numpy as np
from ccvtools import rawio  # noqa

from calibcamlib import Board, Detections


def calibrate_single_camera(detections_cam: Detections, sensor_size, board: Board, opts, mask=None, calib_init=None):
    if calib_init is not None:
        A = calib_init['A']
        k = calib_init['k']
        if "xi" in calib_init:
            xi = calib_init['xi'].reshape(1, -1)
        else:
            xi = 0
    else:
        A = None
        xi = None
        k = None

    detections_cam_array = detections_cam.to_array()
    if mask is None:
        mask = np.sum(~np.isnan(detections_cam_array["marker_coords"][0, :, :, 1]),
                      axis=1) >= opts[
                   'corners_min_n']  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.sum(mask)

    if n_used_frames == 0:
        return {}

    detections_array_use = detections_cam_array["marker_coords"][:, mask]
    ids_use = detections_cam_array["marker_ids"]
    detection_idxs_use = detections_cam_array["detection_idxs"][mask]
    frame_idxs_use = detections_cam_array["frame_idxs"][0][mask]

    cal = {
        'rvec_cam': np.asarray([0., 0., 0.]),
        'tvec_cam': np.asarray([0., 0., 0.]),
        'A': None,
        'xi': np.asarray([0]),
        'k': None,
        'rvecs': None,
        'tvecs': None,
        'detection_idxs': detection_idxs_use,
        'frame_idxs': frame_idxs_use,
        'stdDeviationsIntrinsics': False,
        'stdDeviationsExtrinsics': False,
        'perViewErrors': False,
    }

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

        retval, A, xi, k, rvecs_used, tvecs_used, idxs_used = cal_res

        cal['A'] = np.asarray(A)
        cal['xi'] = np.asarray(xi)
        cal['k'] = np.concatenate((k.squeeze(), [0.0]))

        rvecs = np.full(shape=(len(detection_idxs_use), 3), fill_value=np.nan)
        rvecs[idxs_used] = np.asarray(rvecs_used).reshape((-1, 3))
        cal['rvecs'] = rvecs

        tvecs = np.full(shape=(len(detection_idxs_use), 3), fill_value=np.nan)
        tvecs[idxs_used] = np.asarray(tvecs_used).reshape((-1, 3))
        cal['tvecs'] = tvecs
    else:
        detections_list_use = Detections.from_array({
            "marker_coords": detections_array_use,
            "marker_ids": ids_use,
            "detection_idxs": detection_idxs_use,
            "frame_idxs": [frame_idxs_use],
        }).to_list()

        charuco_corners = detections_list_use["marker_coords"][0]
        charuco_ids = detections_list_use["marker_ids"][0]
        min_board_id = board.get_board_ids()[0]

        # Pinhole camera model [d.reshape((-1,2)) for d in detections_list_use["marker_coords"][0]]
        charuco_ids_zeroed = [ci-min_board_id for ci in charuco_ids]
        cal_res = cv2.aruco.calibrateCameraCharucoExtended(charuco_corners,
                                                           charuco_ids_zeroed,
                                                           board.get_cv2_board(zero_ids=True),
                                                           sensor_size,
                                                           A,
                                                           k,
                                                           **opts['aruco_calibration'])

        retval, A, k, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cal_res

        cal['A'] = np.asarray(A)
        cal['k'] = np.asarray(k).squeeze()
        cal['rvecs'] = np.asarray(rvecs).reshape((-1, 3))
        cal['tvecs'] = np.asarray(tvecs).reshape((-1, 3))

    print('Finished single camera calibration.')
    return cal

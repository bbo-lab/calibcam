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
    ids_use = detections_cam_array["marker_ids"]
    detection_idxs_use = detections_cam_array["detection_idxs"][mask]
    frame_idxs_use = [detections_cam_array["frame_idxs"][0][mask]]

    cal = {
        'rvec_cam': np.asarray([0., 0., 0.]),
        'tvec_cam': np.asarray([0., 0., 0.]),
        'A': None,
        'xi': np.asarray([0]),
        'k': None,
        'rvecs': None,
        'tvecs': None,
        'repro_error': None,
        'detection_idxs': None,
        'frame_idxs': None,
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
        rvecs[idxs_used] = np.asarray(rvecs_used)
        cal['rvecs'] = rvecs

        tvecs = np.full(shape=(len(detection_idxs_use), 3), fill_value=np.nan)
        tvecs[idxs_used] = np.asarray(tvecs_used)
        cal['tvecs'] = tvecs
    else:
        detections_list_use = Detections.from_array({
            "marker_coords": detections_array_use,
            "marker_ids": ids_use,
            "detection_idxs": detection_idxs_use,
            "frame_idxs": frame_idxs_use,
        }).to_list()

        # Pinhole camera model
        cal_res = cv2.aruco.calibrateCameraCharucoExtended(detections_list_use["marker_coords"][0],  # noqa
                                                           detections_list_use["marker_ids"][0],
                                                           board.get_cv2_board(),
                                                           sensor_size,
                                                           A,
                                                           k,
                                                           **opts['aruco_calibration'])

        retval, A, k, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cal_res

        cal['A'] = np.asarray(A)
        cal['k'] = np.concatenate((k.squeeze(), [0.0]))
        cal['rvecs'] = np.asarray(rvecs)
        cal['tvecs'] = np.asarray(tvecs)

    print('Finished single camera calibration.')
    return cal

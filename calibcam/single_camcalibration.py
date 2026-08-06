import cv2
import numpy as np
from ccvtools import rawio  # noqa
import logging

from calibcamlib import Board, Detections
logger = logging.getLogger(__name__)

def calibrate_single_camera(detections_cam: Detections, sensor_size, board: Board, opts, mask=None, calib_init=None, projection_model='perspective'):
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
        mask = (np.sum(~np.isnan(detections_cam_array["marker_coords"][0, :, :, 1]), axis=1)
                >= opts['corners_min_n'])  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.count_nonzero(mask)

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
    if opts['free_vars']['xi'] or projection_model == 'fisheye_equidistant':
        # Omnidir camera model
        if k is not None:
            k = k.reshape(1, -1)[:, :4]

        # Object points for each frame must match corners
        board_points = board.get_board_points()
        if detections_array_use.shape[-2] != len(board_points):
            logger.log(logging.WARNING, f"Number of detected corners {detections_array_use.shape[-2]} does not match number of board points {len(board_points)}, maybe you picked the wrong board?")
        detections_array_use = detections_array_use.reshape((n_used_frames, len(board_points), 2))

        object_points = np.zeros((*detections_array_use.shape[0:2], 3), dtype=detections_array_use.dtype)
        object_points[:] = board_points
        object_points[np.any(np.isnan(detections_array_use), axis=-1)] = np.nan

        object_points = np.ascontiguousarray(object_points, dtype=np.float64)
        detections_array_use = np.ascontiguousarray(detections_array_use, dtype=np.float64)

        object_points = [o for o in object_points]
        detections_array_use = [d for d in detections_array_use]

        object_points_list = []
        detections_array_list = []
        for o, d in zip(object_points, detections_array_use):
            mask = np.all(~np.isnan(d), axis=-1)
            if np.any(mask):
                object_points_list.append(np.expand_dims(o[mask], -2))
                detections_array_list.append(np.expand_dims(d[mask], -2))

        assert len(object_points_list) == len(detections_array_use), "Length of object points and detections must match!"
        print(len(object_points_list), object_points_list[0].shape, object_points_list[0].dtype)
        print(len(detections_array_list), detections_array_list[0].shape, detections_array_list[0].dtype)

        cal_res = cv2.omnidir.calibrate(object_points_list,  # noqa
                                        detections_array_list,
                                        sensor_size,
                                        A,
                                        xi,
                                        k,
                                        **opts['aruco_calibration'])

        retval, A, xi, k, rvecs_used, tvecs_used, idxs_used = cal_res

        if projection_model == "fisheye_equidistant":
            #Equidistant doesn't have a camera shift. We thus apply the shift as an equivalent scaling factor
            A[:, 0:2] = A[:, 0:2] / (xi + 1)
            xi = 0
        elif projection_model is None or projection_model == "perspective":
            pass
        else:
            raise ValueError(f"Unknown projection model: {projection_model}")

        cal['projection_model'] = projection_model
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

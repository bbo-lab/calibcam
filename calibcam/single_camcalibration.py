import numpy as np
import cv2

from ccvtools import rawio  # noqa

from calibcam import helper, board


def calibrate_single_camera(model: str):
    if model == "pinhole":
        return calibrate_pinhole_camera
    else:
        return calibrate_omnidir_camera


def calibrate_pinhole_camera(corners_cam, sensor_size, board_params, opts, mask=None, calib_init=None):
    if mask is None:
        mask = np.sum(~np.isnan(corners_cam[:, :, 1]),
                      axis=1) > 0  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.sum(mask)

    if n_used_frames == 0:
        return []

    corners_nn = corners_cam[mask]
    corners_use, ids_use = helper.corners_array_to_ragged(corners_nn)

    if calib_init is not None:
        A = calib_init['A']
        k = calib_init['k']
    else:
        A = None
        k = None

    cal_res = cv2.aruco.calibrateCameraCharucoExtended(corners_use,  # noqa
                                                       ids_use,
                                                       board.make_board(board_params),
                                                       sensor_size,
                                                       A,
                                                       k,
                                                       **opts['detection']['aruco_calibration'])

    rvecs = np.empty(shape=(mask.size, 3))
    rvecs[:] = np.NaN
    tvecs = np.empty(shape=(mask.size, 3))
    tvecs[:] = np.NaN
    retval, A, k, = cal_res[0:3]

    rvecs[mask, :] = np.asarray(cal_res[3])[..., 0]
    tvecs[mask, :] = np.asarray(cal_res[4])[..., 0]

    cal = {
        'rvec_cam': np.asarray([0., 0., 0.]),
        'tvec_cam': np.asarray([0., 0., 0.]),
        'A': np.asarray(A),
        'xi': np.asarray([0.0]),
        'k': np.asarray(k).ravel(),
        'rvecs': np.asarray(rvecs),
        'tvecs': np.asarray(tvecs),
        'repro_error': retval,
        # Not that from here on values are NOT expanded to full frames range, see frames_mask
        'std_intrinsics': cal_res[5],
        'std_extrinsics': cal_res[6],
        'per_view_errors': cal_res[7],
        'frames_mask': mask,
    }
    print('Finished single camera calibration.')
    return cal


def calibrate_omnidir_camera(corners_cam, sensor_size, board_params, opts, mask=None, calib_init=None):
    if mask is None:
        mask = np.sum(~np.isnan(corners_cam[:, :, 1]),
                      axis=1) > 0  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.sum(mask)

    if n_used_frames == 0:
        return []

    object_points_cam = np.zeros((*corners_cam.shape[0:2], 3))
    object_points_cam[:] = board.make_board_points(board_params)

    corners_nn = corners_cam[mask]

    mask_2 = np.isnan(corners_nn[:, :, 1])
    object_points_nn = object_points_cam[mask]
    object_points_nn[mask_2] = np.nan

    corners_use, _ = helper.corners_array_to_ragged(corners_nn)
    object_points_use, _ = helper.corners_array_to_ragged(object_points_nn)

    if calib_init is not None:
        A = calib_init['A']
        xi = calib_init['xi'].reshape(1, -1)
        k = calib_init['k'].reshape(1, -1)[:, :4]
    else:
        A = None
        xi = None
        k = None

    cal_res = cv2.omnidir.calibrate(object_points_use,  # noqa
                                    corners_use,
                                    sensor_size,
                                    A,
                                    xi,
                                    k,
                                    **opts['detection']['aruco_calibration'])

    mask_final = np.zeros_like(mask, dtype=bool)
    mask_final[np.where(mask)[0][cal_res[6].flatten()]] = True

    rvecs = np.empty(shape=(mask_final.size, 3))
    rvecs[:] = np.NaN
    tvecs = np.empty(shape=(mask_final.size, 3))
    tvecs[:] = np.NaN

    retval, A, xi = cal_res[0:3]

    k = np.asarray(cal_res[3]).squeeze()
    if k.size == 4:
        k = np.concatenate((k, [0.0]))

    rvecs[mask_final, :] = np.asarray(cal_res[4])[..., 0]
    tvecs[mask_final, :] = np.asarray(cal_res[5])[..., 0]

    cal = {
        'rvec_cam': np.asarray([0., 0., 0.]),
        'tvec_cam': np.asarray([0., 0., 0.]),
        'A': np.asarray(A),
        'xi': np.asarray(xi),
        'k': np.asarray(k).ravel(),
        'rvecs': np.asarray(rvecs),
        'tvecs': np.asarray(tvecs),
        'repro_error': retval,
        # Not that from here on values are NOT expanded to full frames range, see frames_mask
        'idx': cal_res[6],
        'frames_mask': mask_final,
    }
    print('Finished single camera calibration.')
    return cal

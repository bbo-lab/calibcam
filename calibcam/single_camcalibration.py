import numpy as np
import cv2

from ccvtools import rawio  # noqa

from calibcam import helper, board


def calibrate_single_camera(corners_cam, sensor_size, board_params, opts, mask=None, calib_init=None):
    if mask is None:
        mask = np.sum(~np.isnan(corners_cam[:, :, 1]),
                      axis=1) > 0  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

    n_used_frames = np.sum(mask)

    if n_used_frames == 0:
        return {}

    corners_nn = corners_cam[mask]
    corners_use, ids_use = helper.corners_array_to_ragged(corners_nn)

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

    if opts['free_vars']['xi']:
        # Omnidir camera model
        object_points_nn = np.zeros((*corners_nn.shape[0:2], 3))
        object_points_nn[:] = board.make_board_points(board_params)
        object_points_nn[np.isnan(corners_nn[:, :, 1])] = np.nan

        object_points_use, _ = helper.corners_array_to_ragged(object_points_nn)

        if k is not None:
            k = k.reshape(1, -1)[:, :4]

        cal_res = cv2.omnidir.calibrate(object_points_use,  # noqa
                                        corners_use,
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
        # Pinhole camera model
        cal_res = cv2.aruco.calibrateCameraCharucoExtended(corners_use,  # noqa
                                                           ids_use,
                                                           board.make_board(board_params),
                                                           sensor_size,
                                                           A,
                                                           k,
                                                           **opts['aruco_calibration'])

        retval, A, k = cal_res[:3]
        others = cal_res[3:]

        if xi is None:
            xi = [0.0]
        mask_singlecam_calib = np.copy(mask)

    rvecs = np.empty(shape=(mask.size, 3))
    rvecs[:] = np.NaN
    tvecs = np.empty(shape=(mask.size, 3))
    tvecs[:] = np.NaN

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
        'frames_mask': mask_singlecam_calib,
    }

    if not opts['free_vars']['xi']:
        # Not that from here on values are NOT expanded to full frames range, see frames_mask
        cal['std_intrinsics'] = others[2]
        cal['std_extrinsics'] = others[3]
        cal['per_view_errors'] = others[4]

    print('Finished single camera calibration.')
    return cal

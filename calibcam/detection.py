import multiprocessing
import cv2
import imageio
import numpy as np
from ccvtools import rawio  # noqa
from joblib import Parallel, delayed

from calibcam import camfunctions, board, helper
from calibcam.calibrator_opts import finalize_aruco_detector_opts


def detect_corners(rec_file_names, n_frames, board_params, opts):
    print('DETECTING FEATURES')
    frames_mask = np.zeros(shape=(len(rec_file_names), n_frames))
    corners_all = []
    ids_all = []

    # Empirically, detection seems to utilize about 6 cores
    detections = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / opts['detect_cpu_divisor'])))(
        delayed(detect_corners_cam)(rec_file_name, opts, board_params)
        for rec_file_name in rec_file_names)
    for i_cam, detection in enumerate(detections):
        corners_all.append(detection[0])
        ids_all.append(detection[1])
        frames_mask[i_cam, :] = detection[2]
        print(f'Detected features in {np.sum(frames_mask[i_cam]).astype(int):04d}  frames in camera {i_cam:02d}')

    return corners_all, ids_all, frames_mask


def detect_corners_cam(video, opts, board_params):
    reader = imageio.get_reader(video)

    corners_cam = []
    ids_cam = []
    frames_mask = np.zeros(camfunctions.get_n_frames_from_reader(reader), dtype=bool)

    # Detect corners over cams
    for (i_frame, frame) in enumerate(reader):
        # color management
        if not isinstance(opts['color_convert'], bool) and len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, opts['color_convert'])  # noqa

        # corner detection
        corners, ids, rejected_img_points = \
            cv2.aruco.detectMarkers(frame,  # noqa
                                    cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),  # noqa
                                    **finalize_aruco_detector_opts(opts['detection']['aruco_detect']))

        if len(corners) == 0:
            continue

        # corner refinement
        corners_ref, ids_ref = \
            cv2.aruco.refineDetectedMarkers(frame,  # noqa
                                            board.make_board(board_params),
                                            corners,
                                            ids,
                                            rejected_img_points,
                                            **finalize_aruco_detector_opts(opts['detection']['aruco_refine']))[0:2]

        # corner interpolation
        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                                ids_ref,
                                                frame,
                                                board.make_board(board_params),
                                                **opts['detection']['aruco_interpolate'])
        if charuco_corners is None:
            continue

        # check if the result is degenerated (all corners on a line)
        if not helper.check_detections_nondegenerate(board_params['boardWidth'], charuco_ids):
            continue

        # add offset
        # We take offset into consideration at corner detection level. This means that the calibration parameters always
        # refer to the offset-free pixel positions and offsets do NOT have to be taken into account anywhere in
        # this calibration procedure or when working with the 
        offset_x, offset_y = camfunctions.get_header_from_reader(reader)['offset']
        charuco_corners[:, :, 0] = charuco_corners[:, :, 0] + offset_x
        charuco_corners[:, :, 1] = charuco_corners[:, :, 1] + offset_y

        # check against last used frame
        # TODO check functionality of this code and determine actual value for maxdist
        used_frame_idxs = np.where(frames_mask)
        if not len(used_frame_idxs) > 0:
            last_used_frame_idx = used_frame_idxs[-1]

            ids_common = np.intersect1d(ids_cam[last_used_frame_idx], charuco_ids)

            if helper.check_detections_nondegenerate(board_params['boardWidth'], ids_common):
                prev_mask = np.isin(ids_cam[last_used_frame_idx], ids_common)
                curr_mask = np.isin(charuco_ids, ids_common)

                diff = corners_cam[last_used_frame_idx][prev_mask] - charuco_corners[curr_mask]
                dist = np.sqrt(np.sum(diff ** 2, 1))
                print(dist)
                dist_max = np.max(dist)
                if np.max(dist) < opts['detection']['inter_frame_dist']:
                    continue

        frames_mask[i_frame] = True
        corners_cam.append(charuco_corners)
        ids_cam.append(charuco_ids)

    return corners_cam, ids_cam, frames_mask

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
    frame_mask = np.zeros(shape=(len(rec_file_names), n_frames))
    corners_all = []
    ids_all = []

    detections = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / opts['detect_cpu_divisor'])))(
        delayed(detect_corners_cam)(rec_file_name, opts, board_params)
        for rec_file_name in rec_file_names)
    for i_cam, detection in enumerate(detections):
        corners_all.append(detection[0])
        ids_all.append(detection[1])
        frame_mask[i_cam, :] = detection[2]
        print(f'Detected features in {np.sum(frame_mask[i_cam]).astype(int):04d}  frames in camera {i_cam:02d}')

    return corners_all, ids_all, frame_mask


def detect_corners_cam(video, opts, board_params):
    reader = imageio.get_reader(video)

    corners_cam = []
    ids_cam = []
    frames_mask = np.zeros(camfunctions.get_n_frames_from_reader(reader), dtype=bool)

    # get offset
    offset_x, offset_y = camfunctions.get_header_from_reader(reader)['offset']

    # Detect corners over cams
    for (i_frame, frame) in enumerate(reader):
        # color management
        if opts['color_convert'] is not None and len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, opts['color_convert'])  # noqa

        # corner detection
        corners, ids, rejected_img_points = \
            cv2.aruco.detectMarkers(frame,  # noqa
                                    cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),  # noqa
                                    **finalize_aruco_detector_opts(opts['aruco_detect']))

        if len(corners) == 0:
            continue

        # corner refinement
        corners_ref, ids_ref = \
            cv2.aruco.refineDetectedMarkers(frame,  # noqa
                                            board.make_board(board_params),
                                            corners,
                                            ids,
                                            rejected_img_points,
                                            **finalize_aruco_detector_opts(opts['aruco_refine']))[0:2]

        # corner interpolation
        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                                ids_ref,
                                                frame,
                                                board.make_board(board_params),
                                                **opts['aruco_interpolate'])
        if charuco_corners is None:
            continue

        # check if the result is degenerated (all corners on a line)
        if not helper.check_detections_nondegenerate(board_params['boardWidth'], charuco_ids):
            continue

        # add offset
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
                if not (dist_max > 0.0):
                    continue

        frames_mask[i_frame] = True
        corners_cam.append(charuco_corners)
        ids_cam.append(charuco_ids)

    return corners_cam, ids_cam, frames_mask

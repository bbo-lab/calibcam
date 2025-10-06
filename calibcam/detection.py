import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import yaml
from ccvtools import rawio  # noqa
from joblib import Parallel, delayed
from svidreader import filtergraph

from calibcam import camfunctions, helper
from calibcam.calibrator_opts import finalize_aruco_detector_opts
from calibcamlib import Board, Detections


def detect_corners(rec_file_names, boards, opts, rec_pipelines=None):
    print('DETECTING FEATURES')
    if isinstance(opts['frames_offsets'], bool):
        frames_offsets = np.zeros(len(rec_file_names))
    else:
        frames_offsets = opts['frames_offsets']

    if rec_pipelines is None:
        rec_pipelines = [None] * len(rec_file_names)

    if isinstance(opts['frames_lists'], bool):
        frames_start = opts['frames_start']
        frames_end = opts['frames_end']
        frames_step = opts['frames_step']

        frames_lists = []
        for rec_file_name, rec_pipeline, offset in zip(rec_file_names, rec_pipelines, frames_offsets):
            reader = filtergraph.get_reader(rec_file_name, backend="iio", cache=False)
            if rec_pipeline is not None:
                fg = filtergraph.create_filtergraph_from_string([reader], rec_pipeline)
                reader = fg['out']

            frames_list = np.arange(
                frames_start + offset,
                min(frames_end + offset, camfunctions.get_n_frames_from_reader(reader)),
                frames_step,
                dtype=int
            )
            frames_lists.append(frames_list)
    else:
        frames_lists = opts['frames_lists']

        def load_frames_lists(frames_list):
            if isinstance(frames_list, str):
                frames_list = Path(frames_list)
            if frames_list.suffix == ".yml":
                with open(frames_list, "r") as stream:
                    frames_list = yaml.safe_load(stream)["frames_list"]
            elif frames_list.suffix == ".npy":
                frames_list = np.load(frames_list, allow_pickle=True)[()]["frames_list"]
            else:
                raise ValueError("Unknown file type for frames_list")
            return frames_list

        if isinstance(frames_lists, Path) or isinstance(frames_lists, str):
            frames_lists = [load_frames_lists(frames_lists)] * len(rec_file_names)
        else:
            frames_lists = [load_frames_lists(f) for f in frames_lists]

    if not opts["parallelize"]:
        detections_cams = []
        for rec_file_name, brd, frames_list, offset, rec_pipeline \
                in zip(rec_file_names, boards, frames_lists, frames_offsets, rec_pipelines):
            detections_cams.append(detect_corners_cam(
                rec_file_name, opts, brd, frames_list, rec_pipeline=rec_pipeline))
    else:
        detections_cams = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() // opts['detect_cpu_divisor'])))(
            delayed(detect_corners_cam)(rec_file_name, opts, brd, frames_list, rec_pipeline=rec_pipeline)
            for rec_file_name, brd, frames_list, offset, rec_pipeline
            in zip(rec_file_names, boards, frames_lists, frames_offsets, rec_pipelines))

    detections = sum(detections_cams, Detections())

    return detections


def detect_corners_cam(video, opts, board: Board, frames_list, rec_pipeline=None):
    board_params = board.get_board_params()
    board_start_id = board.get_board_ids()[0]

    reader = filtergraph.get_reader(video, backend="iio", cache=False)
    if rec_pipeline is not None:
        fg = filtergraph.create_filtergraph_from_string([reader], rec_pipeline)
        reader = fg['out']

    # We take offset into consideration at corner detection level. This means that the calibration parameters always
    # refer to the offset-free pixel positions and offsets do NOT have to be taken into account anywhere in
    # this calibration procedure or when working with the
    offset_x, offset_y = camfunctions.get_header_from_reader(reader)['offset']

    if opts['RC_reject_corners']:
        # Reject corners based on radial contrast value
        RC_params = opts['detection_opts']['radial_contrast_reject']
        RC_reader = helper.RadialContrast(reader, **RC_params)

    frames_list = np.asarray(frames_list)
    frames_list = frames_list[frames_list >= 0]
    frames_list = frames_list[frames_list < camfunctions.get_n_frames_from_reader(reader)]

    corners_cam = []
    ids_cam = []
    detection_idxs_cam = []

    # Detect corners over cams
    for i_fr, frame_idx in enumerate(frames_list):

        frame = reader.get_data(frame_idx)

        if opts.get("gamma_correction", None) is not None:  # TODO: Generalize this
            frame -= np.min(frame)
            frame = frame.astype(np.float64)
            frame /= np.max(frame)
            frame = np.sqrt(frame)
            frame = (frame * 255).astype(np.uint8)

        # color management
        if not isinstance(opts['color_convert'], bool) and len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, opts['color_convert'])  # noqa

        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),
                                           parameters)

        # corner detection
        corners, ids, rejected_img_points = detector.detectMarkers(frame)
        # corners, ids, rejected_img_points = \
        #     cv2.aruco.detectMarkers(frame,  # noqa
        #                             cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']),  # noqa
        #                             **finalize_aruco_detector_opts(opts['detection_opts']['aruco_detect']))

        if len(corners) == 0:
            continue

        board_obj = board.get_cv2_board()

        # corner refinement
        corners_ref, ids_ref = \
            cv2.aruco.refineDetectedMarkers(frame,  # noqa
                                            board_obj,
                                            corners,
                                            ids,
                                            rejected_img_points,
                                            **finalize_aruco_detector_opts(opts['detection_opts']['aruco_refine']))[0:2]

        # corner interpolation
        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                                ids_ref,
                                                frame,
                                                board_obj,
                                                **opts['detection_opts']['aruco_interpolate'])
        if charuco_corners is None:
            continue

        if opts['RC_reject_corners']:
            # Reject corners based on radial contrast value
            RC_frame = RC_reader.read(frame_idx)
            corners_frame = np.squeeze(charuco_corners).astype(int).T
            RC_bool = RC_frame[tuple(corners_frame[::-1, np.newaxis])] > 0
            charuco_ids = charuco_ids[RC_bool[0]]
            charuco_corners = charuco_corners[RC_bool[0]]

        # check if the result is degenerated (all corners on a line)
        if not helper.check_detections_nondegenerate(board_params['boardWidth'], charuco_ids,
                                                     opts['detection_opts']['min_corners']):
            continue

        # add offset
        charuco_corners[:, :, 0] = charuco_corners[:, :, 0] + offset_x
        charuco_corners[:, :, 1] = charuco_corners[:, :, 1] + offset_y

        # check against last used frame
        if len(detection_idxs_cam) > 0:
            ids_common = np.intersect1d(ids_cam[-1], charuco_ids)

            # TODO Check if replacement with current frame in case of more detections is feasible.
            # Should be a fringe problem, though
            if helper.check_detections_nondegenerate(board_params['boardWidth'], ids_common,
                                                     opts['detection_opts']['min_corners']):
                prev_mask = np.isin(ids_cam[-1], ids_common)
                curr_mask = np.isin(charuco_ids, ids_common)

                diff = corners_cam[-1][prev_mask] - charuco_corners[curr_mask]
                dist = np.sqrt(np.sum(diff ** 2, 1))

                if np.max(dist) < opts['detection_opts']['inter_frame_dist']:
                    continue

        corners_cam.append(charuco_corners)
        ids_cam.append(charuco_ids + board_start_id)
        detection_idxs_cam.append(i_fr)

    reader.close()

    markers_list = {
        "marker_coords": corners_cam,
        "marker_ids": ids_cam,
        "detection_idxs": detection_idxs_cam,
        "frame_idxs": frames_list[detection_idxs_cam],
    }

    return Detections.from_list(markers_list)  # corners_cam, ids_cam, fin_frames_mask

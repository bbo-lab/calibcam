import multiprocessing
import cv2
import numpy as np
from ccvtools import rawio  # noqa
from svidreader import filtergraph
from joblib import Parallel, delayed
from itertools import islice

from calibcam import camfunctions, board, helper
from calibcam.calibrator_opts import finalize_aruco_detector_opts


def detect_corners(rec_file_names, n_frames, board_params, opts, rec_pipelines=None, return_matrix=True):
    print('DETECTING FEATURES')

    start_frm_indexes = opts.get('start_frame_indexes', np.zeros(len(rec_file_names), dtype=int))
    stop_frm_indexes = opts.get('stop_frame_indexes', np.full(len(rec_file_names), fill_value=n_frames, dtype=int))
    if rec_pipelines is None:
        rec_pipelines = [None] * len(rec_file_names)

    init_frames_masks = opts.get('init_frames_masks', [None] * len(rec_file_names))
    if isinstance(init_frames_masks, str):
        init_frames_masks = np.load(init_frames_masks)

    fin_frames_masks = np.zeros(shape=(len(rec_file_names), np.min(stop_frm_indexes - start_frm_indexes)), dtype=bool)
    corners_all = []
    ids_all = []

    # Empirically, detection seems to utilize about 6 cores
    detections = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() // opts['detect_cpu_divisor'])))(
        delayed(detect_corners_cam)(rec_file_name, opts, board_params, start_frm_indexes[i_rec],
                                    stop_frm_indexes[i_rec], init_frames_masks[i_rec],
                                    rec_pipeline=rec_pipelines[i_rec])
        for i_rec, rec_file_name in enumerate(rec_file_names))

    for i_cam, detection in enumerate(detections):
        corners_all.append(detection[0])
        ids_all.append(detection[1])
        fin_frames_masks[i_cam, :] = detection[2][:fin_frames_masks.shape[1]]
        n_detections_cam = np.array([len(c) for c in corners_all[i_cam]])
        print(f'Detected features in {np.sum(fin_frames_masks[i_cam]).astype(int):04d}  frames in camera {i_cam:02d} - '
              f'({int(np.mean(n_detections_cam)):02d}±{int(np.std(n_detections_cam))})')

    if return_matrix:
        return helper.make_corners_array(corners_all, ids_all, (board_params["boardWidth"] - 1) * (
                board_params["boardHeight"] - 1), fin_frames_masks), np.where(np.any(fin_frames_masks, axis=0))[0]
    else:
        return corners_all, ids_all, fin_frames_masks


def detect_corners_cam(video, opts, board_params, start_frm_idx=0, stop_frm_idx=None, init_frames_mask=None, rec_pipeline=None):

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

    if stop_frm_idx is None:
        stop_frm_idx = camfunctions.get_n_frames_from_reader(reader)

    corners_cam = []
    ids_cam = []
    if init_frames_mask is None:
        init_frames_mask = np.ones(stop_frm_idx - start_frm_idx, dtype=bool)
    fin_frames_mask = np.zeros(stop_frm_idx - start_frm_idx, dtype=bool)

    # Detect corners over cams
    for (i_frame, frame) in enumerate(islice(reader, start_frm_idx, stop_frm_idx, opts["frame_step"])):
        i_frame = i_frame * opts["frame_step"]

        if opts.get("gamma_correction", None) is not None: # TODO: Generalize this
            frame -= np.min(frame)
            frame = frame.astype(np.float64)
            frame /= np.max(frame)
            frame = np.sqrt(frame)
            frame = (frame*255).astype(np.uint8)

        if not init_frames_mask[i_frame]:
            continue

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

        # corner refinement
        corners_ref, ids_ref = \
            cv2.aruco.refineDetectedMarkers(frame,  # noqa
                                            board.make_board(board_params),
                                            corners,
                                            ids,
                                            rejected_img_points,
                                            **finalize_aruco_detector_opts(opts['detection_opts']['aruco_refine']))[0:2]

        # corner interpolation
        retval, charuco_corners, charuco_ids = \
            cv2.aruco.interpolateCornersCharuco(corners_ref,  # noqa
                                                ids_ref,
                                                frame,
                                                board.make_board(board_params),
                                                **opts['detection_opts']['aruco_interpolate'])
        if charuco_corners is None:
            continue

        if opts['RC_reject_corners']:
            # Reject corners based on radial contrast value
            RC_frame = RC_reader.read(i_frame + start_frm_idx)
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
        # TODO check functionality of this code and determine actual value for maxdist
        #  Also, this bears the danger that different cams get detections in different frames and pose estimation
        #  becomes impossible. If this is ever required, it has to be made sure that cameras get detections on the same
        #  frames, e.g. by determining sufficient movement only on the first cam.
        #  Alternatively, in videos with a too high framerate, we could just use a frameskip.
        used_frame_ids = np.where(fin_frames_mask)[0]
        if len(used_frame_ids) > 0:
            ids_common = np.intersect1d(ids_cam[-1], charuco_ids)

            if helper.check_detections_nondegenerate(board_params['boardWidth'], ids_common,
                                                     opts['detection_opts']['min_corners']):
                prev_mask = np.isin(ids_cam[-1], ids_common)
                curr_mask = np.isin(charuco_ids, ids_common)

                diff = corners_cam[-1][prev_mask] - charuco_corners[curr_mask]
                dist = np.sqrt(np.sum(diff ** 2, 1))

                if np.max(dist) < opts['detection_opts']['inter_frame_dist']:
                    continue

        fin_frames_mask[i_frame] = True
        corners_cam.append(charuco_corners)
        ids_cam.append(charuco_ids)

    reader.close()

    return corners_cam, ids_cam, fin_frames_mask

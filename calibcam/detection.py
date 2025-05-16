import multiprocessing
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml
from ccvtools import rawio  # noqa
from svidreader import filtergraph
from joblib import Parallel, delayed
from itertools import islice

from calibcam import camfunctions, board, helper
from calibcam.board import Board
from calibcam.calibrator_opts import finalize_aruco_detector_opts


def detect_corners(rec_file_names, n_frames, boards, opts, rec_pipelines=None, data_path=None):
    print('DETECTING FEATURES')
    # if isinstance(opts['detect_use_single'], bool):
    #     opts['detect_use_single'] = [opts['detect_use_single'] for _ in rec_file_names]

    # calibration_paths = []
    # for i_cam, use_singe_opt in enumerate(opts['detect_use_single']):
    #     if isinstance(use_singe_opt, str):
    #         calibration_paths.append(use_singe_opt)
    #     elif isinstance(use_singe_opt, bool) and use_singe_opt is True:
    #         calibration_paths.append(Path(data_path) / f"calibration_single_{i_cam:03d}.yml")
    #     else:
    #         calibration_paths.append(None)
    # TODO finish implementation with refine markers function

    opts['start_frame_indexes'] = opts.get('start_frame_indexes', np.zeros(len(rec_file_names), dtype=int))
    start_frm_indexes = opts['start_frame_indexes']

    opts['stop_frame_indexes'] = opts.get('stop_frame_indexes',
                                          np.full(len(rec_file_names), fill_value=n_frames, dtype=int))
    stop_frm_indexes = opts['stop_frame_indexes']

    # Load frame masks
    opts["init_frames_masks"] = opts.get('init_frames_masks', [False] * len(rec_file_names))
    init_frames_masks = opts["init_frames_masks"]
    if isinstance(init_frames_masks, str):
        if Path(init_frames_masks).suffix == ".yml":
            with open(init_frames_masks, "r") as stream:
                init_frames_masks = yaml.safe_load(stream)["init_frames_masks"]
                init_frames_masks = [np.array(ifm).astype(np.uint32) for ifm in init_frames_masks]
        elif Path(init_frames_masks).suffix == ".npy":
            init_frames_masks = np.load(init_frames_masks)
    for i_mask, if_mask in enumerate(init_frames_masks):
        if not isinstance(if_mask, bool) and not if_mask.dtype == bool:
            init_frames_masks[i_mask] = np.zeros(n_frames, dtype=bool)
            init_frames_masks[i_mask][if_mask] = True

    if rec_pipelines is None:
        rec_pipelines = [None] * len(rec_file_names)

    fin_frames_masks = np.zeros(shape=(len(rec_file_names), np.min(stop_frm_indexes - start_frm_indexes)), dtype=bool)
    corners_all = []
    ids_all = []

    if not opts["parallelize"]:
        detections_cams = []
        for i_rec, rec_file_name in enumerate(rec_file_names):
            detections_cams.append(detect_corners_cam(rec_file_name, opts, boards[i_rec], start_frm_indexes[i_rec],
                                        stop_frm_indexes[i_rec], init_frames_masks[i_rec],
                                        rec_pipeline=rec_pipelines[i_rec]))
    else:
        # Empirically, detection seems to utilize about 6 cores
        detections_cams = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() // opts['detect_cpu_divisor'])))(
            delayed(detect_corners_cam)(rec_file_name, opts, boards[i_rec], start_frm_indexes[i_rec],
                                        stop_frm_indexes[i_rec], init_frames_masks[i_rec],
                                        rec_pipeline=rec_pipelines[i_rec])
            for i_rec, rec_file_name in enumerate(rec_file_names))

    detections = Detections()
    for i_cam, detection in enumerate(detections_cams):
        detections += detection
        n_detections_frames = detection.get_n_detections_frames()
        n_detections_markers = detection.get_n_detections_markers()
        print(f'Detected features in {n_detections_frames:04d} frames in camera {i_cam:02d} - '
              f'({int(np.mean(n_detections_markers)):02d}±{int(np.std(n_detections_markers))})')

    return detections


def detect_corners_cam(video, opts, board: Board, start_frm_idx=0, stop_frm_idx=None, init_frames_mask=None, rec_pipeline=None):
    board_params = board.get_board_params()

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
    if isinstance(init_frames_mask, bool):
        init_frames_mask = np.ones(stop_frm_idx - start_frm_idx, dtype=bool)
    fin_frames_mask = np.zeros(stop_frm_idx - start_frm_idx, dtype=bool)

    step_mask = np.zeros_like(init_frames_mask, dtype=bool)
    step_mask[::opts["frame_step"]] = True

    process_frame_idxs = start_frm_idx + np.where(step_mask & init_frames_mask)[0]

    # Detect corners over cams
    for frame_idx in process_frame_idxs:
        frame = reader.get_data(frame_idx)

        if opts.get("gamma_correction", None) is not None: # TODO: Generalize this
            frame -= np.min(frame)
            frame = frame.astype(np.float64)
            frame /= np.max(frame)
            frame = np.sqrt(frame)
            frame = (frame*255).astype(np.uint8)

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

        fin_frames_mask[frame_idx] = True
        corners_cam.append(charuco_corners)
        ids_cam.append(charuco_ids)

    reader.close()

    return corners_cam, ids_cam, fin_frames_mask


class Detections:
    def __init__(self, markers_array=None):
        self._markers_array = markers_array

    @staticmethod
    def from_list(markers_list, *args, **kwargs):
        if isinstance(markers_list, dict):
            marker_coords = markers_list["marker_coords"]
            frame_idxs = markers_list["frame_idxs"]
            marker_ids = markers_list["marker_ids"]
        else:
            marker_coords = markers_list
            frame_idxs = args[0]
            marker_ids = args[1]
        markers_array = helper.make_corners_array(marker_coords, frame_idxs, marker_ids)
        return Detections(markers_array)

    @staticmethod
    def from_array(markers_array):
        return Detections(markers_array)

    def to_array(self):
        return deepcopy(self._markers_array)

    def to_list(self):
        mis = self._markers_array["marker_ids"]
        fis = self._markers_array["frame_idxs"]
        marker_coords = []
        marker_ids = []
        frame_idxs = []
        for mc_c in self._markers_array["marker_coords"]:
            marker_coords_c = []
            marker_ids_c = []
            frame_idxs_c = []
            for frame_idx, mc_f in zip(fis, mc_c):
                mask = np.isnan(mc_f[:,0])
                if ~any(mask):
                    continue
                marker_coords_c.append(mc_f[mask])
                frame_idxs_c.append(frame_idx)
                marker_ids_c.append(mis[mask])
            marker_coords.append(marker_coords_c)
            marker_ids.append(marker_ids_c)
            frame_idxs.append(frame_idxs_c)

        return {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "frame_idxs": frame_idxs
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1) # Do not squeeze dimension

        return Detections({
            "marker_coords": self._markers_array["marker_coords"][key,],
            "frame_idxs": self._markers_array["frame_idxs"],
            "marker_ids": self._markers_array["marker_ids"],
        })

    def __add__(self, o):
        if self._markers_array is None:
            return o

        if not (
                len(self._markers_array["frame_idxs"]) == len(o.to_array()["frame_idxs"]) and
                np.all(self._markers_array["frame_idxs"] == o.to_array()["frame_idxs"]) and
                len(self._markers_array["marker_ids"]) == len(o.to_array()["marker_ids"]) and
                np.all(self._markers_array["marker_ids"] == o.to_array()["marker_ids"])
        ):
            frame_idxs = np.unique(np.concatenate((self._markers_array["frame_idxs"], o.to_array()[2])))
            marker_ids = np.unique(np.concatenate((self._markers_array["marker_ids"], o.to_array()[2])))
            marker_coords = np.full(
                (
                    len(self._markers_array["marker_coords"]) + len(o.to_array()["marker_coords"]),
                    len(frame_idxs),
                    len(marker_ids),
                    2
                ),
                fill_value=np.nan,
                dtype=self._markers_array["marker_coords"].dtype
            )
            marker_coords[:len(self._markers_array["marker_coords"]),
            np.isin(self._markers_array["frame_idxs"], frame_idxs),
            np.isin(self._markers_array["marker_ids"], marker_ids)] = self._markers_array["marker_coords"]
            marker_coords[len(self._markers_array["marker_coords"]):,
            np.isin(o.to_array()["frame_idxs"], frame_idxs),
            np.isin(o.to_array()["marker_ids"], marker_ids)] = o.to_array()["marker_coords"]
        else:
            frame_idxs = self._markers_array["frame_idxs"]
            marker_ids = self._markers_array["marker_ids"]
            marker_coords = np.concatenate((self._markers_array["marker_coords"], o.to_array()["marker_coords"]), axis=0)

        return Detections({
            "marker_coords": marker_coords,
            "frame_idxs": frame_idxs,
            "marker_ids": marker_ids,
        })

    def get_n_detections_markers(self):
        return np.isnan(self._markers_array["marker_coords"][..., 0]).sum(axis=2)

    def get_n_detections_frames(self):
        return np.any(~np.isnan(self._markers_array["marker_coords"][..., 0]), axis=2).sum(axis=1)

    @staticmethod
    def from_file(detection_files):
        if isinstance(detection_files, list):
            return sum([Detections.from_file(df) for df in detection_files], Detections())

        detection_files = Path(detection_files)
        if detection_files.suffix == ".yml":
            with open(detection_files, "r") as file:
                detection = yaml.safe_load(file)
        elif detection_files.suffix == ".npy":
            detection = np.load(detection_files, allow_pickle=True)[()]
        else:
            raise FileNotFoundError(f"{detection_files} is not supported")

        if "marker_coords" in detection:
            marker_coords = np.array(detection["marker_coords"])
        elif "corners" in detection:
            marker_coords = np.array([detection["corners"]])
        else:
            # TODO: write import code for multicamcal files
            raise ValueError("Unsupported dictionary content")

        if "marker_ids" in detection:
            marker_ids = np.array(detection["marker_ids"])
        elif "used_corner_ids" in detection:
            marker_ids = np.array(detection["used_corner_ids"])
        else:
            marker_ids = np.arange(np.array(marker_coords).shape[1])

        if "frame_idxs" in detection:
            frame_idxs = np.array(detection["frame_idxs"])
        elif "used_frames_ids" in detection:
            frame_idxs = np.array(detection["used_frames_ids"])
        else:
            frame_idxs = np.arange(np.array(marker_coords).shape[0])

        return Detections({
            "marker_coords": marker_coords,
            "frame_idxs": frame_idxs,
            "marker_ids": marker_ids,
        })

    def to_file(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = Path(file_paths)
            file_paths = [file_paths.parent / f"{file_paths.stem}_{i:03d}{file_paths.suffix}"
                          for i in range(len(self._markers_array))]

        assert len(file_paths) == len(self._markers_array), "Number of files must match number of detections"

        for file_path, marker_coords_cam in zip(file_paths, self._markers_array["marker_coords"]):
            markers_dict = {
                "marker_coords": marker_coords_cam,
                "frame_idxs": self._markers_array["frame_idxs"],
                "marker_ids": self._markers_array["marker_ids"],
            }
            if Path(file_path).suffix == ".yml":
                with open(file_path, "w") as file:
                    yaml.safe_dump(markers_dict, file)
            elif Path(file_path).suffix == ".npy":
                np.save(file_path, markers_dict)
            else:
                raise FileNotFoundError(f"{file_path} is not supported")

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
    if opts['frames_offsets'] is None:
        frames_offsets = np.zeros(len(rec_file_names))
    else:
        frames_offsets = opts['frames_offsets']

    if opts['frame_idx_lists'] is None:
        frames_start = opts['frames_start']
        frames_end = opts['frames_start']
        frames_step = opts['frame_step']

        frame_idx_lists = []
        for rec_file_name, rec_pipeline, offset in zip(rec_file_names, rec_pipelines, frames_offsets):
            reader = filtergraph.get_reader(rec_file_name, backend="iio", cache=False)
            if rec_pipeline is not None:
                fg = filtergraph.create_filtergraph_from_string([reader], rec_pipeline)
                reader = fg['out']

            frame_idx_list = np.arange(
                frames_start+offset,
                min(frames_end+offset, camfunctions.get_n_frames_from_reader(reader)),
                frames_step
            )
            frame_idx_lists.append(frame_idx_list)
    else:
        frame_idx_lists = opts['frame_idx_lists']
        if isinstance(frame_idx_lists, str):
            frame_idx_lists = Path(data_path)

        def load_frame_idx_lists(frame_idx_list):
            if isinstance(frame_idx_list, str):
                frame_idx_list = Path(frame_idx_list)
            if frame_idx_list.suffix == ".yml":
                with open(frame_idx_list, "r") as stream:
                    frame_idx_list = yaml.safe_load(stream)["frame_idx_list"]
            elif frame_idx_list.suffix == ".npy":
                frame_idx_list = np.load(frame_idx_list, allow_pickle=True)[()]["frame_idx_list"]
            else:
                raise ValueError("Unknown file type for frame_idx_list")
            return frame_idx_list

        if isinstance(frame_idx_lists, Path):
            frame_idx_lists = [load_frame_idx_lists(frame_idx_lists)] * len(rec_file_names)
        else:
            frame_idx_lists = [load_frame_idx_lists(f) for f in frame_idx_lists]


    if rec_pipelines is None:
        rec_pipelines = [None] * len(rec_file_names)

    if not opts["parallelize"]:
        detections_cams = []
        for rec_file_name, brd, frame_idx_list, offset, rec_pipeline \
                in zip(rec_file_names, boards, frame_idx_lists, frames_offsets, rec_pipelines):
            detections_cams.append(detect_corners_cam(
                rec_file_name, opts, boards, frame_idx_list, offset_from_real=offset, rec_pipeline=rec_pipeline))
    else:
        detections_cams = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() // opts['detect_cpu_divisor'])))(
            delayed(detect_corners_cam)(rec_file_name, opts, brd, frame_idx_list, offset_from_real=offset,
                                        rec_pipeline=rec_pipeline)
            for rec_file_name, brd, frame_idx_list, offset, rec_pipeline
            in zip(rec_file_names, boards, frame_idx_lists, frames_offsets, rec_pipelines))

    detections = sum(detections_cams, Detections())

    return detections


def detect_corners_cam(video, opts, board: Board, frame_idx_list, offset_from_real=0, rec_pipeline=None):
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

    frame_idx_list = np.asarray(frame_idx_list)
    frame_idx_list = frame_idx_list[frame_idx_list<camfunctions.get_n_frames_from_reader(reader)]

    corners_cam = []
    ids_cam = []
    detection_idxs_cam = []

    fin_frames_mask = np.zeros_like(frame_idx_list, dtype=bool)

    # Detect corners over cams
    for i_fr, frame_idx in enumerate(frame_idx_list):
        if frame_idx < 0 or frame_idx >= camfunctions.get_n_frames_from_reader(reader):
            corners_cam.append([])
            ids_cam.append([])
            continue

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
        ids_cam.append(charuco_ids)
        detection_idxs_cam.append(charuco_ids)

    reader.close()

    markers_list = {
        "marker_coords": corners_cam,
        "marker_ids": ids_cam,
        "detection_idxs": detection_idxs_cam,
        "frame_idxs": frame_idx_list[detection_idxs_cam],
    }

    return Detections.from_list(markers_list)  #corners_cam, ids_cam, fin_frames_mask


class Detections:
    def __init__(self, markers_array=None):
        if markers_array is not None:
            markers_array = deepcopy(markers_array)
            markers_array = self.strip_nans(markers_array)
        self._markers_array = markers_array

    @staticmethod
    def strip_nans(markers_array):
        frame_mask = np.any(~np.isnan(markers_array["marker_coords"][..., 0]), axis=(0, 2))
        markers_array["marker_coords"] = markers_array["marker_coords"][:, frame_mask]
        markers_array["detection_idxs"] = np.asarray(markers_array["detection_idxs"])[frame_mask]
        markers_array["frame_idxs"] = np.asarray(markers_array["frame_idxs"])[:, frame_mask]

        marker_mask = np.any(~np.isnan(markers_array["marker_coords"][..., 0]), axis=(0, 1))
        markers_array["marker_coords"] = markers_array["marker_coords"][:, :, marker_mask]
        markers_array["marker_ids"] = np.asarray(markers_array["marker_ids"])[marker_mask]

        assert markers_array["marker_coords"].shape[1] == len(markers_array["detection_idxs"])
        assert markers_array["marker_coords"].shape[1] == markers_array["marker_coords"].shape[1]
        assert markers_array["marker_coords"].shape[2] == len(markers_array["marker_ids"])
        return markers_array

    @staticmethod
    def from_list(markers_list, *args, **kwargs):
        if isinstance(markers_list, dict):
            marker_coords = markers_list["marker_coords"]
            detection_idxs = markers_list["detection_idxs"]
            frame_idxs = markers_list["frame_idxs"]
            marker_ids = markers_list["marker_ids"]
        else:
            marker_coords = markers_list
            marker_ids = args[0]
            detection_idxs = args[1]
            frame_idxs = args[2]
        markers_array = helper.make_corners_array(marker_coords, marker_ids, detection_idxs, frame_idxs)
        return Detections(markers_array)

    @staticmethod
    def from_array(markers_array):
        # TODO: CHeck content
        return Detections(markers_array)

    def to_array(self):
        return deepcopy(self._markers_array)

    def to_list(self):
        mis = self._markers_array["marker_ids"]
        dis = self._markers_array["detection_idxs"]
        fis = self._markers_array["frame_idxs"]
        marker_coords = []
        marker_ids = []
        detection_idxs = []
        frame_idxs = []
        for i_cam, mc_c in enumerate(self._markers_array["marker_coords"]):
            marker_coords_c = []
            marker_ids_c = []
            detection_idxs_c = []
            frame_idxs_c = []
            for detection_idx, frame_idx, mc_f in zip(dis, fis[i_cam], mc_c):
                mask = ~np.isnan(mc_f[:,0])
                if ~np.any(mask):
                    continue
                marker_coords_c.append(mc_f[mask].reshape(-1, 1, 2))
                frame_idxs_c.append(frame_idx.tolist())
                detection_idxs_c.append(frame_idx.tolist())
                marker_ids_c.append(mis[mask].reshape(-1, 1))
            marker_coords.append(marker_coords_c)
            marker_ids.append(marker_ids_c)
            detection_idxs.append(detection_idxs_c)
            frame_idxs.append(frame_idxs_c)

        return {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        }

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)

        markers_array = {
            "marker_coords": self._markers_array["marker_coords"][key,],
            "marker_ids": self._markers_array["marker_ids"],
            "detection_idxs": self._markers_array["detection_idxs"],
            "frame_idxs": self._markers_array["frame_idxs"][key,],
        }
        markers_array = self.strip_nans(markers_array)

        return Detections(markers_array)

    def __add__(self, o):
        if self._markers_array is None:
            return o

        o_array = o.to_array()
        if not (
                len(self._markers_array["detection_idxs"]) == len(o.to_array()["detection_idxs"]) and
                np.all(self._markers_array["detection_idxs"] == o.to_array()["detection_idxs"]) and
                len(self._markers_array["marker_ids"]) == len(o.to_array()["marker_ids"]) and
                np.all(self._markers_array["marker_ids"] == o.to_array()["marker_ids"])
        ):
            detection_idxs = np.unique(np.concatenate((self._markers_array["detection_idxs"], o_array["detection_idxs"])))
            marker_ids = np.unique(np.concatenate((self._markers_array["marker_ids"], o_array["marker_ids"])))
            marker_coords = np.full(
                (
                    len(self._markers_array["marker_coords"]) + len(o_array["marker_coords"]),
                    len(detection_idxs),
                    len(marker_ids),
                    2
                ),
                fill_value=np.nan,
                dtype=self._markers_array["marker_coords"].dtype
            )
            frame_idxs = np.full(marker_coords.shape[:2], fill_value=-1, dtype=int)

            frame_mask = np.isin(detection_idxs, self._markers_array["detection_idxs"])
            marker_mask = np.isin(marker_ids, self._markers_array["marker_ids"])
            len_o1 = len(self._markers_array["marker_coords"])
            for i_cam in range(len_o1):
                marker_coords[i_cam, np.flatnonzero(frame_mask)[:, None], np.flatnonzero(marker_mask)] = self._markers_array["marker_coords"][i_cam]
                frame_idxs[i_cam, frame_mask] = self._markers_array["frame_idxs"][i_cam]

            frame_mask = np.isin(detection_idxs, o_array["detection_idxs"])
            marker_mask = np.isin(marker_ids, o_array["marker_ids"])
            len_o2 = len(o_array["marker_coords"])
            for i_cam in range(len_o2):
                marker_coords[i_cam+len_o1, np.flatnonzero(frame_mask)[:, None], np.flatnonzero(marker_mask)] = o_array["marker_coords"][i_cam]
                frame_idxs[i_cam+len_o1, frame_mask] = o_array["frame_idxs"][i_cam]
        else:
            detection_idxs = self._markers_array["detection_idxs"]
            marker_ids = self._markers_array["marker_ids"]
            marker_coords = np.concatenate((self._markers_array["marker_coords"], o_array["marker_coords"]), axis=0)
            frame_idxs = np.concatenate((self._markers_array["frame_idxs"], o_array["frame_idxs"]), axis=0)

        markers_array = {
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        }
        markers_array = self.strip_nans(markers_array)
        return Detections(markers_array)

    def get_n_cams(self):
        """
        Returns camera dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[0]

    def get_n_frames(self):
        """
        Returns frames dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[1]

    def get_n_markers(self):
        """
        Returns marker dimension of contained array
        """
        return self._markers_array["marker_coords"].shape[2]

    def get_n_detections(self):
        """
        Returns number of overall detections
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:,:,:,1]))

    def get_n_detections_frames(self):
        """
        Returns number of detected frames per marker (shape cam x markers)
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:,:,:,1]), axis=1)

    def get_n_detections_markers(self):
        """
        Returns number of detected markers per frame (shape cam x frames)
        """
        return np.sum(~np.isnan(self._markers_array["marker_coords"][:,:,:,1]), axis=2)

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

        if "detection_idxs" in detection:
            detection_idxs = np.array(detection["detection_idxs"])
        elif "used_frames_ids" in detection:
            detection_idxs = np.array(detection["used_frames_ids"])
        else:
            detection_idxs = np.arange(np.array(marker_coords).shape[0])

        if "frame_idxs" in detection:
            frame_idxs = np.array(detection["frame_idxs"])
        elif "used_frames_ids" in detection:
            frame_idxs = np.array(detection["used_frames_ids"])
        else:
            frame_idxs = np.arange(np.array(marker_coords).shape[0])

        return Detections({
            "marker_coords": marker_coords,
            "marker_ids": marker_ids,
            "detection_idxs": detection_idxs,
            "frame_idxs": frame_idxs,
        })

    def to_file(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = Path(file_paths)

        if isinstance(file_paths, Path):
            file_paths = [file_paths.parent / f"{file_paths.stem}_{i:03d}{file_paths.suffix}"
                          for i in range(len(self._markers_array["marker_coords"]))]

        assert len(file_paths) == len(self._markers_array["marker_coords"]), "Number of files must match number of detections"

        for file_path, marker_coords_cam in zip(file_paths, self._markers_array["marker_coords"]):
            markers_dict = {
                "version": "2.0",
                "storage_method": "array",
                "marker_coords": marker_coords_cam[np.newaxis].tolist(),
                "marker_ids": self._markers_array["marker_ids"].tolist(),
                "detection_idxs": self._markers_array["detection_idxs"].tolist(),
                "frame_idxs": self._markers_array["frame_idxs"].tolist(),

            }
            if Path(file_path).suffix == ".yml":
                with open(file_path, "w") as file:
                    yaml.safe_dump(markers_dict, file)
            elif Path(file_path).suffix == ".npy":
                np.save(file_path, markers_dict)
            else:
                raise FileNotFoundError(f"{file_path} is not supported")

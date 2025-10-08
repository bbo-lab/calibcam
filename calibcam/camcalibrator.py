import logging
import multiprocessing
import os
from copy import deepcopy
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ccvtools import rawio  # noqa
from joblib import Parallel, delayed
from scipy.io import savemat as scipy_io_savemat
from svidreader import filtergraph

from calibcam import helper, camfunctions
from calibcam import yaml_helper
from calibcam.calibrator_opts import get_default_opts
from calibcam.camfunctions import test_objective_function, make_optim_input
from calibcam.detection import detect_corners
from calibcam.exceptions import *
from calibcam.pose_estimation import estimate_cam_poses, build_initialized_calibs
from calibcam.single_camcalibration import calibrate_single_camera
from calibcamlib import Board, Detections

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CamCalibrator:
    def __init__(self, recordings, pipelines=None, board_params=None, data_path=None, opts=None):
        if opts is None:
            opts = {}

        if data_path is not None:
            self.data_path = os.path.expanduser(data_path)
            os.makedirs(self.data_path, exist_ok=True)
        else:
            self.data_path = os.path.expanduser(os.path.dirname(recordings[0]))

        # Videos
        self.readers = None
        self.rec_file_names = None
        self.rec_pipelines = None
        self.n_frames = np.nan

        # Options
        self.opts = {}
        self.opts = self.load_opts(opts, self.data_path)

        # Recordings
        self.load_recordings(recordings, pipelines)

        self.boards = [Board(bp) for bp in self.resolve_board_params(board_params)]
        return

    def resolve_board_params(self, board_params):
        if board_params is None:
            board_params = Board.from_file(Path(self.rec_file_names[0]).parent).get_board_params()
        return board_params

    @staticmethod
    def load_opts(opts, data_path=None):
        if data_path is not None and os.path.isfile(data_path + "/opts.npy"):
            fileopts = np.load(data_path + "/opts.npy", allow_pickle=True).item()
            opts = helper.deepmerge_dicts(opts, fileopts)

        return helper.deepmerge_dicts(opts, get_default_opts(0))

    def load_recordings(self, recordings, pipelines=None):
        # TODO check if input files are valid files
        try:
            self.readers = []
            for irec, rec in enumerate(recordings):
                reader = filtergraph.get_reader(rec, backend="iio", cache=False)
                if pipelines is not None:
                    fg = filtergraph.create_filtergraph_from_string([reader], pipelines[irec])
                    reader = fg['out']
                self.readers.append(reader)
        except ValueError:
            print('At least one unsupported format supplied')
            raise UnsupportedFormatException

        self.rec_file_names = recordings
        self.rec_pipelines = pipelines

        # find frame numbers
        n_frames = [camfunctions.get_n_frames_from_reader(reader) for reader in self.readers]
        print(f'Found {n_frames} frames in cams')

    def close_readers(self):
        if not self.readers:
            return
        for reader in self.readers:
            reader.close()

    def perform_multi_calibration(self):
        required_corner_idxs = None
        # [[0,
        #                          bp["boardWidth"] - 2,
        #                          (bp["boardWidth"] - 1) * (bp["boardHeight"] - 2),
        #                          (bp["boardWidth"] - 1) * (bp["boardHeight"] - 1) - 1,
        #                          ] for bp in
        #                         self.board_params]  # Corners that we require to be detected for pose estimation

        if not self.opts["detection"] and (self.opts["calibration_single"] or self.opts["calibration_multi"]):
            self.opts["detection"] = sorted(glob(self.data_path + "/detection_*.yml"))

        # === Detection ===
        if isinstance(self.opts["detection"], list):
            # TODO: Support True in the list instead of strings to only detect individual cams
            assert len(self.opts["detection"]) == self.opts["n_cams"], ("Number of detection files must be equal "
                                                                        "to number of cameras")
            print("Loading detections from files")
            detections = Detections.from_file(self.opts["detection"])
        elif self.opts["detection"]:
            # detect corners
            # Corners are originally detected by cv2 as ragged lists with additional id lists (to determine which
            # corners the values refer to) and frame masks (to determine which frames the list elements refer to).
            # This saves memory, but significantly increases complexity of code as we might index into camera frames,
            # used frames or global frames. For simplification, corners are returned as a single matrix of shape
            #  n_cams x n_timepoints_with_used_detections x n_corners x 2
            # Memory footprint at this stage is not critical.
            print("Performing charuco detection")

            detections = detect_corners(self.rec_file_names, self.boards, self.opts,
                                        rec_pipelines=self.rec_pipelines)
            detections.to_file(Path(self.data_path) / f"detection.yml")
        else:
            print("Cannot proceed without detections. Exiting.")
            return

        for i_cam, detection in enumerate(detections):
            n_detections_frames = detection.get_n_detections_frames()
            n_detections_markers = detection.get_n_detections_markers()
            print(f'Detected features in {len(n_detections_markers[0]):04d} frames in camera {i_cam:02d} - '
                  f'({int(np.mean(n_detections_markers)):02d}±{int(np.std(n_detections_markers))})')

        # === Single cam calibration ===
        if not self.opts["calibration_single"] and self.opts["calibration_multi"]:
            self.opts["calibration_single"] = sorted(glob(self.data_path + "/calibration_single_*.yml"))

        if isinstance(self.opts["calibration_single"], list):
            # TODO: Support True in the list instead of strings to only detect individual cams
            assert len(self.opts["calibration_single"]) == self.opts["n_cams"], ("Number of calibration_single files "
                                                                                 "must be equal to number of cameras")

            # Import saved calibration; TODO refactor
            calibs_single = []
            for calibration_single_file in self.opts["calibration_single"]:
                calibration_single_file = Path(calibration_single_file)
                if calibration_single_file.suffix == ".yml":
                    with open(calibration_single_file, "r") as file:
                        calibs_dict = yaml_helper.load_calib(yaml.safe_load(file))
                        if "calibs" in calibs_dict:
                            calibs_dict = calibs_dict["calibs"][0]  # Use 0th calibration from multicam calibration file
                        calibs_dict = yaml_helper.collection_to_array(calibs_dict)
                        assert "A" in calibs_dict and "k" in calibs_dict, "File did not contain valid calibration"
                        calibs_single.append(calibs_dict)
                elif calibration_single_file.suffix == ".npy":
                    calib = np.load(calibration_single_file, allow_pickle=True)[()]
                    # For multicam_calibration files
                    if "calibs" in calib:
                        calib = calib["calibs"][0]
                    calibs_single.append(calib)
                else:
                    raise FileNotFoundError(f"{calibration_single_file} is not supported")

            # Fill with board positions for current detectionsö TODO: Refactor to separate functions
            calibs_single = self.obtain_single_cam_calibrations(self.readers, detections=detections, boards=self.boards,
                                                                opts=self.opts, calibs_single=calibs_single)
        elif self.opts["calibration_single"]:
            calibs_single = self.obtain_single_cam_calibrations(self.readers, detections=detections, boards=self.boards,
                                                                opts=self.opts)
            # TODO reimplement
            # if self.opts['optimize_ind_cams']:
            #     for i_cam, calib in enumerate(calibs_single):
            #         # analytically estimate initial camera poses
            #         # Although we don't have camera poses at this step, we use this function to correctly structure the
            #         # calibs_single to optimize poses.
            #         calibs_interim = estimate_cam_poses([calib], self.opts, corners=corners[[i_cam]],
            #                                             required_corner_idxs=required_corner_idxs[i_cam])
            #
            #         calibs_fit_single, rvecs_boards, tvecs_boards, _, _ = self.optimize_poses(corners[[i_cam]],
            #                                                                                   calibs_interim)
            #         calibs_single[i_cam].update(
            #             helper.combine_calib_with_board_params(calibs_fit_single, rvecs_boards, tvecs_boards)[0])
            #         # calibs_single[i_cam]['frames_idxs'] = np.where(np.sum(~np.isnan(corners[i_cam][:, :, 1]), axis=1) > 0)[0]
        else:
            print("Cannot proceed without single cam calbrations. Exiting.")
            return

        for i_cam, calib_single in enumerate(calibs_single):
            save_path = Path(self.data_path) / f"calibration_single_{i_cam:03d}.yml"
            with open(save_path, "w") as file:
                yaml.dump(yaml_helper.numpy_collection_to_list(calib_single), file, default_flow_style=True)
            # np.save(save_path.with_suffix(".npy"), calib_single, allow_pickle=True)

        # === Multi cam calibration ===
        if self.opts["calibration_multi"]:
            do_estimate_cam_poses = False
            if not (isinstance(self.opts["init_extrinsics"]["rvecs_cam"], np.ndarray) and
                    isinstance(self.opts["init_extrinsics"]["tvecs_cam"], np.ndarray)):
                do_estimate_cam_poses = True
                self.opts["init_extrinsics"]["rvecs_cam"] = np.zeros((len(calibs_single), 3))
                self.opts["init_extrinsics"]["tvecs_cam"] = np.zeros((len(calibs_single), 3))

            # Extend camera specific pose vecs to match length of detections
            calibs_multi = build_initialized_calibs(calibs_single, self.opts, detections=detections)

            if do_estimate_cam_poses:
                # analytically estimate initial camera poses
                calibs_multi = estimate_cam_poses(calibs_multi, self.opts, detections=detections,
                                                  required_corner_idxs=None)

            # At this point, we want to move away from thinking in boards. board_points_all is now a set of 3 points
            # that are rigidly transformed by the rvecs and tvecs, and their -2 axis corresponds to -2 axis of
            # marker_coords
            # TODO: Introduce an entry point here to optimize calibrations with arbitrary known marker points and
            #   an initialization for the calibration
            detections_array = detections.to_array()
            marker_coords = detections_array['marker_coords']
            marker_ids = detections_array['marker_ids']
            frame_idxs = detections.to_array()["frame_idxs"]

            result = self.build_result(calibs_multi, used_frames_ids=frame_idxs)
            print('SAVE MULTI CAMERA CALIBRATION')
            self.save_multicalibration(result, filename="joinedsingles_calibraton")

            board_points_all = helper.combine_boards_to_points(self.boards, marker_ids)

            if self.opts['debug']:
                args, vars_free = make_optim_input(
                    board_points_all, calibs_multi, marker_coords, self.opts)
                test_objective_function(calibs_multi, vars_free, args, marker_coords, board_points_all,
                                        individual_poses=True)

            print('OPTIMIZING ALL POSES')
            # self.plot(calibs_single, corners, used_frames_ids, self.board_params, 3, 35)
            calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_poses(
                marker_coords, calibs_multi, board_points_all)

            if self.opts['debug']:
                calibs_fit = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards)
                test_objective_function(calibs_fit, min_result.x, args, marker_coords, board_points_all,
                                        individual_poses=True)

            print('OPTIMIZING ALL PARAMETERS I')
            calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(
                marker_coords, calibs_fit, board_points_all)

            # According to tests with good calibration recordings, the following steps are unnecessary and optimality
            # was already reached in the previous step
            if self.opts["optimize_board_poses"]:
                if self.opts['debug']:
                    calibs_fit = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards)
                    test_objective_function(calibs_fit, min_result.x, args, marker_coords, board_points_all,
                                            individual_poses=True)

                print('OPTIMIZING BOARD POSES')
                calibs_fit, rvecs_boards, tvecs_boards, _, _ = self.optimize_board_poses(
                    marker_coords, calibs_fit, board_points_all, prev_fun=min_result.fun)
                calibs_fit = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards)

                print('OPTIMIZING ALL PARAMETERS II')
                calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(
                    marker_coords, calibs_fit, board_points_all)

            # No board poses in final calibration!
            calibs_test = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards, copy=True)
            test_objective_function(calibs_test, min_result.x, args, marker_coords, board_points_all,
                                    individual_poses=True)

            print('OPTIMIZING ALL PARAMETERS III - Removed high error frames')
            # At this point, there does not seem to be any hope to recover high error frames. We recalibrate without
            # them.
            errors = min_result.fun.copy().reshape(frame_idxs.shape + (-1, 2))
            errors[errors == 0] = np.nan
            errors = np.linalg.norm(errors, axis=-1)

            high_error_mask = errors > self.opts["max_allowed_res"]
            print(f"Discarded {100*(np.sum(high_error_mask)/np.sum(~np.isnan(errors))):.2f}% of detections due to high errors")
            marker_coords[high_error_mask] = np.nan

            marker_count = np.all(~np.isnan(marker_coords), axis=-1)
            marker_count = np.sum(marker_count, axis=-1)
            high_error_frame_mask = np.all(marker_count < self.opts["corners_min_n"], axis=0)
            good_frames = np.where(~high_error_frame_mask)[0]
            print(
                f"Discarded {100 * (1-len(good_frames)/frame_idxs.shape[1]):.2f}% of frames due to high errors")
            marker_coords = marker_coords[:, good_frames]
            rvecs_boards = rvecs_boards[good_frames]
            tvecs_boards = tvecs_boards[good_frames]
            frame_idxs = frame_idxs[:, good_frames]

            calibs_fit = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards)
            calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(
                 marker_coords, calibs_fit, board_points_all)

            calibs_test = helper.combine_calib_with_board_poses(calibs_fit, rvecs_boards, tvecs_boards, copy=True)
            test_objective_function(calibs_test, min_result.x, args, marker_coords, board_points_all,
                                    individual_poses=True)

            print('SAVE MULTI CAMERA CALIBRATION')
            result = self.build_result(calibs_fit, used_frames_ids=frame_idxs)
            board_result = self.build_board_result(rvecs_boards, tvecs_boards,
                                                   used_frames_ids=frame_idxs, min_result=min_result)
            self.save_multicalibration(result, board_result)
            # Builds a part of the v1 result that is necessary for other software
            # self.save_multicalibration(helper.build_v1_result(result), rvecs_boards, tvecs_boards, 'multicalibration_v1')

            print('SAVE FIUGRE WITH DETECTIONS')
            rep_err = min_result.fun.reshape(marker_coords.shape)
            for i_cam, (i_reader, c, err) in enumerate(zip(self.readers, marker_coords, rep_err)):
                fig_cam = self.get_corners_cam_fig(camfunctions.get_header_from_reader(i_reader)['sensorsize'],
                                                   c, err)
                fig_cam.savefig(self.data_path + f"/detections_cam_{i_cam:03d}.svg", dpi=300, bbox_inches='tight')
            print('FINISHED MULTI CAMERA CALIBRATION')
        else:
            return

        return

    @staticmethod
    def obtain_single_cam_calibrations(readers, detections, boards, opts, calibs_single=None):
        # Determine missing calibrations and sends off missing ones to a parallel job
        if calibs_single is None:
            calibs_single = detections.get_n_cams() * [None]

        cams_2calibrate = []
        for i_cam, cam_calib in enumerate(calibs_single):
            if cam_calib is not None:
                calibs_single[i_cam] = CamCalibrator.estimate_board_positions_in_single_cam(cam_calib,
                                                                                            detections[i_cam],
                                                                                            boards[i_cam],
                                                                                            opts)
            else:
                cams_2calibrate.append(i_cam)
        #  perform single calibration if needed
        cams_calibrated = CamCalibrator.perform_single_cam_calibrations(readers, detections, boards, opts,
                                                                        cams_2calibrate)

        for i, i_cam in enumerate(cams_2calibrate):
            calibs_single[i_cam] = cams_calibrated[i]

        return calibs_single

    @staticmethod
    def estimate_board_positions_in_single_cam(calib, detections_cam: Detections, board: Board, opts):
        calib = deepcopy(calib)

        board_points = board.get_board_points()
        min_board_id = board.get_board_ids()[0]

        markers = detections_cam.to_list()
        marker_coords = markers["marker_coords"][0]
        detection_idxs = markers["detection_idxs"][0]
        frame_idxs = markers["frame_idxs"][0]
        marker_ids = markers["marker_ids"][0]
        n_frames = detections_cam.get_n_frames()

        calib['rvecs'] = np.full((n_frames, 3), np.nan)
        calib['tvecs'] = np.full((n_frames, 3), np.nan)
        calib['detection_idxs'] = detection_idxs
        calib['frame_idxs'] = frame_idxs

        if opts["parallelize"]:
            board_positions = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count())))(
                delayed(CamCalibrator.estimate_single_board_position)(calib,
                                                                      marker_coords_fr,
                                                                      marker_ids_fr,
                                                                      board_points)
                for marker_coords_fr, marker_ids_fr in zip(marker_coords, marker_ids)
            )
        else:
            board_positions = []
            for marker_coords_fr, marker_ids_fr in zip(marker_coords, marker_ids):
                board_positions.append(
                    CamCalibrator.estimate_single_board_position(calib,
                                                                 marker_coords_fr,
                                                                 marker_ids_fr-min_board_id,
                                                                 board_points)
                )

        for i_pos, pos in enumerate(board_positions):
            if pos[0]:
                calib['rvecs'][i_pos] = pos[1][:, 0]
                calib['tvecs'][i_pos] = pos[2][:, 0]

        return calib

    @staticmethod
    def estimate_single_board_position(calib, marker_coords, ids, board_points):
        if len(ids) < 4:
            return 0, np.full((3,), np.nan), np.full((3,), np.nan)

        if calib.get("xi", 0) != 0:
            logger.log(logging.WARN, "xi is defined in calibration, but not used for board position estimation.")

        retval, rvec, tvec = cv2.solvePnP(board_points[ids].reshape((-1, 3)),
                                          marker_coords.reshape((-1, 2)),
                                          calib["A"], calib["k"],
                                          flags=cv2.SOLVEPNP_IPPE)
        return retval, rvec, tvec

    @staticmethod
    def perform_single_cam_calibrations(readers, detections: Detections, boards, opts, camera_indexes=None):
        print('PERFORM SINGLE CAMERA CALIBRATION')

        if camera_indexes is None:
            camera_indexes = range(len(readers))

        print(int(np.floor(multiprocessing.cpu_count())))

        if opts["parallelize"]:
            calibs_single = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count())))(
                delayed(calibrate_single_camera)(detections[i_cam],
                                                 camfunctions.get_header_from_reader(readers[i_cam])[
                                                     'sensorsize'],
                                                 boards[i_cam],
                                                 {'free_vars': opts['free_vars'][i_cam],
                                                  'aruco_calibration': opts['aruco_calibration'][i_cam],
                                                  'corners_min_n': opts['corners_min_n'],
                                                  })
                for i_cam in camera_indexes)
        else:
            calibs_single = [calibrate_single_camera(detections[i_cam],
                                                     camfunctions.get_header_from_reader(readers[i_cam])[
                                                         'sensorsize'],
                                                     boards[i_cam],
                                                     {'free_vars': opts['free_vars'][i_cam],
                                                      'aruco_calibration': opts['aruco_calibration'][i_cam],
                                                      'corners_min_n': opts['corners_min_n'],
                                                      }) for i_cam in camera_indexes]

        for i_cam, calib in enumerate(calibs_single):
            print(
                f'Used {(~np.isnan(calib["rvecs"][:, 1])).sum(dtype=int):03d} '
                f'frames for single cam calibration for cam {i_cam:02d}'
            )

        return calibs_single

    def optimize_poses(self, corners, calibs_multi, board_points_all, opts=None):
        if opts is None:
            opts = self.opts

        pose_opts = deepcopy(opts)
        free_vars = pose_opts['free_vars']
        for cam in free_vars:
            cam['A'][:] = False
            cam['k'][:] = False
            cam['xi'] = False

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_points_all, opts=pose_opts)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def optimize_board_poses(self, corners, calibs_multi, board_points_all, opts=None, prev_fun=None):
        if opts is None:
            opts = self.opts

        pose_opts = deepcopy(opts)
        pose_opts['optimization']['ftol'] = 1e-14
        pose_opts['optimization']['gtol'] = 1e-14
        pose_opts['optimization']['xtol'] = 1e-14
        free_vars = pose_opts['free_vars']
        for cam in free_vars:
            cam['cam_pose'] = False
            cam['A'][:] = False
            cam['k'][:] = False
            cam['xi'] = False

        calibs_multi_pose = deepcopy(calibs_multi)
        rvecs_boards = calibs_multi[0]["rvecs"]
        tvecs_boards = calibs_multi[0]["tvecs"]

        if prev_fun is not None:
            prev_fun = prev_fun.reshape(corners.shape)
            good_poses = set(np.arange(prev_fun.shape[1]))
            for i_cam in range(prev_fun.shape[0]):
                good_poses = good_poses - set(np.where(prev_fun[i_cam] > pose_opts['max_allowed_res'])[0])
            good_poses = list(good_poses)
        else:
            good_poses = list(range(len(rvecs_boards)))

        print("Number of bad_poses:", len(calibs_multi[0]['rvecs']) - len(good_poses))
        print(f"Optimizing {len(calibs_multi[0]['rvecs'])} poses: ", end='')
        for i_pose in range(len(calibs_multi[0]["rvecs"])):
            print(".", end='', flush=True)
            corners_pose = corners[:, [i_pose]]
            for calib, calib_orig in zip(calibs_multi_pose, calibs_multi):
                nearest_i_pose = helper.nearest_element(i_pose,
                                                        good_poses)  # nearest_i_pose = i_pose if i_pose in good_poses
                calib["rvecs"] = calib_orig["rvecs"][[nearest_i_pose]]
                calib["tvecs"] = calib_orig["tvecs"][[nearest_i_pose]]

            # print(i_pose, rvecs_boards[i_pose])
            calibs_fit_pose, rvecs_boards[i_pose], tvecs_boards[i_pose], min_result, args = \
                camfunctions.optimize_calib_parameters(corners_pose, calibs_multi_pose, board_points_all,
                                                       opts=pose_opts,
                                                       verbose=0)
            # print(i_pose, rvecs_boards[i_pose], min_result.cost)
        return calibs_fit_pose, rvecs_boards, tvecs_boards, None, None

    def optimize_calibration(self, corners, calibs_multi, board_points_all, opts=None):
        if opts is None:
            opts = self.opts

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_points_all, opts=opts)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def build_result(self, calibs, used_frames_ids, other=None):
        # Result should contain the calibration parameters plus all information to reproduce them
        # Siince a calibration can be
        # savemat cannot deal with None
        if other is None:
            other = dict()

        calibs = deepcopy(calibs)
        video_headers = [camfunctions.get_header_from_reader(r) for r in self.readers]
        for calib, header in zip(calibs, video_headers):
            calib['sensor_size'] = header['sensorsize']

        result = {
            'version': 4,  # Increase when this structure changes
            'calibs': calibs,

            # Headers. No content structure guaranteed
            'info': {  # Additional nonessential info from the calibration process
                'board_params': [brd.get_board_params() for brd in self.boards],  # All parameters to recreate the board
                'rec_file_names': self.rec_file_names,  # Recording filenames, may be used for cam names
                'vid_headers': video_headers,
                'used_frames_ids': used_frames_ids,
                'opts': self.opts,
                'other': other,  # Additional info without guaranteed structure
            }
        }

        if self.rec_pipelines is not None:
            result['info']['rec_pipelines'] = self.rec_pipelines

        return result

    def build_board_result(self, rvecs_boards, tvecs_boards, used_frames_ids, min_result=None):
        boards_dict = {
            'version': 4,
            'rvecs': rvecs_boards,
            'tvecs': tvecs_boards,
            'frame_idxs': used_frames_ids,
            'info': {},
        }

        if min_result is not None:
            boards_dict['info']['fun_final'] = min_result.fun
            boards_dict['info']['cost_val_final'] = min_result.cost
            boards_dict['info']['optimality_final'] = min_result.optimality

        return boards_dict

    def save_multicalibration(self, result, board_result=None, filename="multicam_calibration"):
        data_path = self.data_path
        result_path = Path(data_path + '/' + filename)
        return save_multicalibration(result_path, result, board_result)

    # Debug function
    def plot(self, calibs, corners, used_frames_ids, board_params, cidx, fidx):
        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation as R  # noqa
        import camfunctions_ag

        board_coords_3d_0 = Board(board_params).make_board_points()

        print(f"{cidx} - {fidx} - {used_frames_ids[fidx]} - {len(used_frames_ids)} - {len(corners[cidx])}")
        r = calibs[cidx]['rvecs'][fidx, :]
        t = calibs[cidx]['tvecs'][fidx, :]
        print(r)
        print(t)
        im = self.readers[cidx].get_data(used_frames_ids[fidx])

        corners_use, ids_use = helper.corners_array_to_ragged(corners[cidx])
        plt.imshow(cv2.aruco.drawDetectedCornersCharuco(im, corners_use[fidx], ids_use[fidx]))

        board_coords_3d = R.from_rotvec(r).apply(board_coords_3d_0) + t

        board_coords_3d = camfunctions_ag.board_to_unit_sphere(board_coords_3d)
        board_coords_3d = camfunctions_ag.shift_camera(board_coords_3d, calibs[cidx]['xi'].squeeze()[0])
        board_coords_3d = camfunctions_ag.to_ideal_plane(board_coords_3d)

        board_coords_3d_nd = camfunctions_ag.ideal_to_sensor(board_coords_3d, calibs[cidx]['A'])

        board_coords_3d_d = camfunctions_ag.distort(board_coords_3d, calibs[cidx]['k'])
        board_coords_3d_d = camfunctions_ag.ideal_to_sensor(board_coords_3d_d, calibs[cidx]['A'])

        plt.plot(board_coords_3d_d[(0, 4, 34), 0], board_coords_3d_d[(0, 4, 34), 1], 'r+')
        plt.plot(board_coords_3d_nd[(0, 4, 34), 0], board_coords_3d_nd[(0, 4, 34), 1], 'g+')

        plt.show()

    @staticmethod
    def get_corners_cam_fig(im_shape, corners_cam, repro_err_cam):

        im_w, im_h = im_shape
        corners_cam = corners_cam.reshape(-1, 2)
        repro_err_cam = repro_err_cam.reshape(-1, 2)

        fig, ax = plt.subplots()
        ax.errorbar(corners_cam[:, 0], corners_cam[:, 1],
                    fmt=".", ms=1.2,
                    xerr=np.absolute(repro_err_cam[:, 0]), yerr=np.absolute(repro_err_cam[:, 0]),
                    elinewidth=0.8, ecolor="red")
        ax.set_xlim(0, im_w)
        ax.set_ylim(0, im_h)
        ax.set_xlabel("Image x (pix.)")
        ax.set_ylabel("Image y (pix.)")
        ax.invert_yaxis()

        return fig


def save_multicalibration(result_path, result, board_result=None):
    if board_result is not None:
        with open(result_path.parent / f"{result_path.stem}_board_positions.yml", "w") as yml_file:
            yaml.dump(yaml_helper.numpy_collection_to_list(board_result), yml_file, default_flow_style=True)

    np.save(result_path.with_suffix('.npy'), result)
    scipy_io_savemat(result_path.with_suffix('.mat'), result)
    with open(result_path.with_suffix('.yml'), "w") as yml_file:
        yaml.dump(yaml_helper.numpy_collection_to_list(result), yml_file, default_flow_style=True)
    print(f'Saved multi camera calibration to file {result_path}')
    return

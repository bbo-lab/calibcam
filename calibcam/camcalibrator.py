import os

import numpy as np
from scipy.optimize import least_squares, OptimizeResult
from scipy.io import savemat as scipy_io_savemat
import cv2

from itertools import compress
from pathlib import Path

import imageio
from ccvtools import rawio  # noqa

import time
import multiprocessing
from joblib import Parallel, delayed

from .exceptions import *
from . import helper
from . import board
from . import optimization

from .calibrator_opts import get_default_opts, finalize_aruco_detector_opts
from .pose_estimation import estimate_cam_poses


class CamCalibrator:
    def __init__(self, board_name=None, opts=None):
        if opts is None:
            opts = {}

        self.board_name = board_name  # Currently, recordings are needed to determine the board path in most cases

        # Options
        self.opts = helper.deepmerge_dicts(opts, get_default_opts())

        # Board
        self.board_params = None

        # Videos
        self.readers = None
        self.rec_file_names = None
        self.recordingIsLoaded = False
        self.dataPath = None
        self.n_frames = np.NaN
        self.reset_recordings()  # SPOT principle, previous defaults are irrelevant and only there for initial type

        return

    def get_board_params_from_name(self, board_name):
        if board_name is not None:
            board_params = board.get_board_params(board_name)
        else:
            board_params = board.get_board_params(Path(self.rec_file_names[0]).parent)
        return board_params

    def reset_recordings(self):
        self.readers = None
        self.rec_file_names = None
        self.recordingIsLoaded = False
        self.dataPath = None
        self.n_frames = np.NaN

    def set_recordings(self, recordings):
        # check if input files are valid files
        try:
            self.readers = [imageio.get_reader(rec) for rec in recordings]
        except ValueError:
            print('At least one unsupported format supplied')
            raise UnsupportedFormatException

        self.dataPath = os.path.dirname(recordings[0])
        self.rec_file_names = recordings

        # find frame numbers
        n_frames = np.zeros(len(self.readers), dtype=np.int64)
        for (i_cam, reader) in enumerate(self.readers):
            n_frames[i_cam] = helper.get_n_frames_from_reader(reader)
            print(f'Found {n_frames[i_cam]} frames in cam {i_cam}')

        # check if frame number is consistent
        self.n_frames = n_frames[0]
        if not np.all(np.equal(n_frames[0], n_frames[1:])):
            print('WARNING: Number of frames is not identical for all cameras')
            print('Number of detected frames per camera:')
            for (i_cam, nF) in enumerate(n_frames):
                print(f'\tcamera {i_cam:03d}:\t{nF:04d}')

            if self.opts['allow_unequal_n_frame']:
                self.n_frames = np.int64(np.min(n_frames))
            else:
                # raise exception for outside confirmation
                raise UnequalFrameCountException

        self.board_params = self.get_board_params_from_name(self.board_name)

        self.recordingIsLoaded = True

    def perform_multi_calibration(self):
        # # detect corners
        # corners_all, ids_all, frames_masks = self.detect_corners()
        #
        # # # split into two frame sets
        # # # first set contains frames for single calibration
        # # # second set contains frames for multi calibration
        # # self.split_frame_sets()
        #
        # # perform single calibration
        # calibs_single = self.perform_single_cam_calibrations(corners_all, ids_all, frames_masks)
        #
        # # Save intermediate result, for dev purposes
        # np.savez(self.dataPath+'/preoptim.npz', calibs_single, corners_all, ids_all, frames_masks)
        preoptim = np.load(self.dataPath + '/preoptim.npz', allow_pickle=True)
        calibs_single = preoptim['arr_0']
        corners_all = preoptim['arr_1']
        ids_all = preoptim['arr_2']
        frames_masks = preoptim['arr_3']

        # analytically estimate initial camera poses
        calibs_multi = estimate_cam_poses(calibs_single, self.opts['coord_cam'])

        print('START MULTI CAMERA CALIBRATION')
        calibs_fit, _, _, min_result, args = \
            self.start_optimization(corners_all, ids_all, calibs_multi, frames_masks)

        result = self.build_result(calibs_fit, frames_masks=frames_masks, tvecs_boards=None, min_result=min_result, args=args)

        print('SAVE MULTI CAMERA CALIBRATION')
        self.save_multicalibration(result)

        print('FINISHED MULTI CAMERA CALIBRATION')
        return

    def detect_corners(self):
        print('DETECTING FEATURES')
        frame_mask = np.zeros(shape=(len(self.readers), self.n_frames))
        corners_all = []
        ids_all = []

        detections = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / self.opts['detect_cpu_divisor'])))(
            delayed(self.detect_corners_cam)(rec_file_name, self.opts, self.board_params)
            for rec_file_name in self.rec_file_names)
        for i_cam, detection in enumerate(detections):
            corners_all.append(detection[0])
            ids_all.append(detection[1])
            frame_mask[i_cam, :] = detection[2]
            print(f'Detected features in {np.sum(frame_mask[i_cam]).astype(int):04d}  frames in camera {i_cam:02d}')

        return corners_all, ids_all, frame_mask

    @staticmethod
    def detect_corners_cam(video, opts, board_params):
        reader = imageio.get_reader(video)

        corners_cam = []
        ids_cam = []
        frames_mask = np.zeros(helper.get_n_frames_from_reader(reader), dtype=bool)

        # get offset
        offset_x, offset_y = helper.get_header_from_reader(reader)['offset']

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

    @staticmethod
    def calibrate_single_camera(corners_cam, ids_cam, sensor_size, board_params, opts, mask=None):
        if mask is None:
            mask = np.asarray([len(c) > 0 for c in corners_cam], dtype=bool)

        n_used_frames = np.sum(mask)

        if n_used_frames > 0:  # do this to not run into indexing issues
            corners_use = list(compress(corners_cam,
                                        mask))

            ids_use = list(compress(ids_cam,
                                    mask))
            cal_res = cv2.aruco.calibrateCameraCharuco(corners_use,  # noqa
                                                       ids_use,
                                                       board.make_board(board_params),
                                                       sensor_size,
                                                       None,
                                                       None,
                                                       **opts['aruco_calibration'])

            retval, A, k, rvecs, tvecs = cal_res[0:5]

            cal = {
                'rvec_cam': np.asarray([0., 0., 0.]),
                'tvec_cam': np.asarray([0., 0., 0.]),
                'A': np.asarray(A),
                'k': np.asarray(k),
                'rvecs': np.asarray(rvecs),
                'tvecs': np.asarray(tvecs),
                'repro_error': retval,
                'frame_mask': mask,
            }
            print('Finished single camera calibration.')
            return cal
        else:
            return []

    def perform_single_cam_calibrations(self, corners_all, ids_all, frame_mask):
        print('PERFORM SINGLE CAMERA CALIBRATION')

        # calibs_single = [self.calibrate_single_camera(corners_all[i_cam],
        #                                               ids_all[i_cam],
        #                                               helper.get_header_from_reader(self.readers[i_cam])['sensorsize'],
        #                                               self.board_params,
        #                                               self.opts,
        #                                               mask)
        #                  for i_cam in range(len(self.readers))]
        calibs_single = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / 2)))(
            delayed(self.calibrate_single_camera)(corners_all[i_cam],
                                                  ids_all[i_cam],
                                                  helper.get_header_from_reader(self.readers[i_cam])['sensorsize'],
                                                  self.board_params,
                                                  self.opts)
            for i_cam in range(len(self.readers)))

        for i_cam, calib in enumerate(calibs_single):
            calib['frame_mask'] = frame_mask[i_cam].copy()
            assert calib['frame_mask'].sum(dtype=int) == calib['tvecs'].shape[0], "Sizes do not match, check masks."
            print(
                f'Used {calib["frame_mask"].sum(dtype=int):03d} frames for single cam calibration for cam {i_cam:02d}')

        return calibs_single

    def start_optimization(self, corners_all, ids_all, calibs_multi, frame_masks):
        print('Starting optimization procedure - This might take a while')
        print('The following lines are associated with the current state of the optimization procedure:')
        start_time = time.time()

        board_coords_3d_0 = board.make_board_points(self.board_params)
        calibs_fit = [dict((k, c[k].copy()) for k in ('A', 'k', 'rvec_cam', 'tvec_cam')) for c in calibs_multi]

        used_frame_mask = np.any(frame_masks, axis=0)
        used_frame_idxs = np.where(used_frame_mask)[0]
        n_used_frames = used_frame_mask.sum(dtype=int)
        n_cams = len(self.readers)
        n_corner = board_coords_3d_0.shape[0]

        vars_free, vars_full, mask_free = optimization.make_initialization(calibs_multi, frame_masks, self.opts)

        # Prepare ideal boards for each cam and frame. This costs almost 3x the necessary memory, but makes
        # further autograd compatible code much easier
        boards_coords_3d_0 = np.tile(board_coords_3d_0.T,
                                     (n_cams, n_used_frames, 1, 1))  # Note transpose for later linalg

        corners = np.empty(shape=(n_cams, n_used_frames, 2, n_corner))
        corners[:] = np.NaN
        for i_cam, frame_mask_cam in enumerate(frame_masks):
            frame_idxs_cam = np.where(frame_mask_cam)[0]

            for i_frame, f_idx in enumerate(used_frame_idxs):
                # print(ids_all[i_cam][i_frame].ravel())
                # print(corners[i_cam, f_idx].shape)
                # print(corners_all[i_cam][i_frame].shape)
                cam_fr_idx = np.where(frame_idxs_cam == f_idx)[0]
                if cam_fr_idx.size < 1:
                    continue

                cam_fr_idx = int(cam_fr_idx)
                corners[i_cam, i_frame][:, ids_all[i_cam][cam_fr_idx].ravel()] = \
                    corners_all[i_cam][cam_fr_idx][:, 0, :].T  # Note transpose for later linalg

        args = {
            'vars_full': vars_full,  # All possible vars, free vars will be substituted in _free wrapper functions
            'mask_opt': mask_free,  # Mask of free vars within all vars
            'frame_masks': frame_masks,
            'opts_free_vars': self.opts['free_vars'],
            'precalc': {  # Stuff that can be precalculated
                'boards_coords_3d_0': boards_coords_3d_0,  # Board points in z plane
                'corners': corners,
                'jacobians': optimization.get_obj_fcn_jacobians(),
            },
            'memory': {  # References to memory that can be reused, avoiding cost of reallocation
                'residuals': np.zeros_like(corners),
                'boards_coords_3d': np.zeros_like(boards_coords_3d_0),
                'boards_coords_3d_cams': np.zeros_like(boards_coords_3d_0),
                'calibs': calibs_fit,
            }
        }

        min_result: OptimizeResult = least_squares(optimization.obj_fcn_wrapper,
                                   vars_free,
                                   jac=optimization.obj_fcn_jacobian_wrapper,
                                   bounds=np.array([[-np.inf, np.inf]] * vars_free.size).T,
                                   args=[args],
                                   **self.opts['optimization'])
        current_time = time.time()
        print('Optimization algorithm converged:\t{:s}'.format(str(min_result.success)))
        print('Time needed:\t\t\t\t{:.0f} seconds'.format(current_time - start_time))

        calibs_fit, rvecs_boards, tvecs_boards = optimization.unravel_to_calibs(min_result.x, args)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def build_result(self, calibs,
                     frames_masks=None, rvecs_boards=None, tvecs_boards=None, min_result=None, args=None):  # noqa
        result = {
            'version': 2.0,
            'calibs': calibs,
            'board_params': self.board_params,
            'rec_file_names': self.rec_file_names,
            'vid_headers': [helper.get_header_from_reader(r) for r in self.readers],
            'info': {
                'cost_val_final': np.NaN,
                'optimality_final': np.NaN,
                'frames_masks': np.array([], dtype=bool),
                'opts': self.opts,
            }
        }

        if min_result is not None:
            result['info']['cost_val_final'] = min_result.cost
            result['info']['optimality_final'] = min_result.optimality

        if frames_masks is not None:
            result['info']['frames_masks'] = frames_masks

        return result

    def save_multicalibration(self, result):
        # save
        result_path = self.dataPath + '/multicam_calibration.'
        np.save(result_path+'npy', result)
        scipy_io_savemat(result_path + 'mat', result)
        helper.save_multicalibration_to_matlabcode(result, self.dataPath)
        print('Saved multi camera calibration to file {:s}'.format(result_path))
        return

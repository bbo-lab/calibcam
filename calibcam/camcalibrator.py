import copy
import os
from copy import deepcopy

import numpy as np
from scipy.io import savemat as scipy_io_savemat
import cv2

from pathlib import Path

from ccvtools import rawio  # noqa
from svidreader import filtergraph

import multiprocessing
from joblib import Parallel, delayed

from calibcam.camfunctions import test_objective_function, make_optim_input
from calibcam.detection import detect_corners
from calibcam.exceptions import *
from calibcam import helper, camfunctions, board, compatibility

from calibcam.calibrator_opts import get_default_opts
from calibcam.pose_estimation import estimate_cam_poses
from calibcam.single_camcalibration import calibrate_single_camera


class CamCalibrator:
    def __init__(self, recordings, pipelines=None, board_name=None, data_path=None, calibs_init=None, opts=None):
        if opts is None:
            opts = {}

        self.board_name = board_name  # Currently, recordings are needed to determine the board path in most cases

        if data_path is not None:
            self.data_path = data_path
        else:
            self.data_path = os.path.expanduser(os.path.dirname(recordings[0]))

        # Board
        self.board_params = None

        # Videos
        self.readers = None
        self.rec_file_names = None
        self.rec_pipelines = None
        self.n_frames = np.NaN

        # Options
        self.opts = {}
        self.opts = self.load_opts(opts, self.data_path)

        # Recordings
        self.load_recordings(recordings, pipelines)

        # Calibs initialization
        self.calibs_init = None
        self.calibs_init = self.load_calibs_init(calibs_init, self.data_path)

        return

    def get_board_params_from_name(self, board_name):
        if board_name is not None:
            board_params = board.get_board_params(board_name)
        else:
            board_params = board.get_board_params(Path(self.rec_file_names[0]).parent)
        return board_params

    @staticmethod
    def load_calibs_init(calibs_init, data_path=None):
        calib_init_path = data_path + "/multicam_calibration.npy"
        if calibs_init is None and data_path is not None and os.path.isfile(calib_init_path):
            print(f"Loading initialization from {calib_init_path}")
            calibs_init = np.load(calib_init_path, allow_pickle=True).item()["calibs"]

        return calibs_init

    @staticmethod
    def load_opts(opts, data_path=None):
        if data_path is not None and os.path.isfile(data_path + "/opts.npy"):
            fileopts = np.load(data_path + "/opts.npy", allow_pickle=True).item()
            opts = helper.deepmerge_dicts(opts, fileopts)

        return helper.deepmerge_dicts(opts, get_default_opts())

    def load_recordings(self, recordings, pipelines=None):
        # check if input files are valid files
        try:
            self.readers = []
            for irec, rec in enumerate(recordings):
                reader = filtergraph.get_reader(rec, backend="iio")
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
        n_frames = np.zeros(len(self.readers), dtype=np.int64)
        for (i_cam, reader) in enumerate(self.readers):
            n_frames[i_cam] = camfunctions.get_n_frames_from_reader(reader)
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

    def perform_multi_calibration(self):
        n_corners = (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 1)
        required_corner_idxs = [0,
                                self.board_params["boardWidth"] - 2,
                                (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 2),
                                (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 1) - 1,
                                ]  # Corners that we require to be detected for pose estimation
        if self.opts["optimize_only"]:  # We expect that detections and single cam calibs are already present
            preoptim = np.load(self.data_path + '/preoptim.npy', allow_pickle=True)[()]
            preoptim = compatibility.update_preoptim(preoptim, n_corners)

            del self.opts['internals']
            calibs_single = preoptim['info']['other']['calibs_single']
            used_frames_ids = preoptim['info']['used_frames_ids']
            corners = preoptim['info']['corners']

            # We just redo this since it is fast and the output may help
            calibs_multi = estimate_cam_poses(calibs_single, self.opts, corners=corners,
                                              required_corner_idxs=required_corner_idxs)
        else:
            # detect corners
            # Corners are originally detected by cv2 as ragged lists with additional id lists (to determine which
            # corners the values refer to) and frame masks (to determine which frames the list elements refer to).
            # This saves memory, but significantly increases complexity of code as we might index into camera frames,
            # used frames or global frames. For simplification, corners are returned as a single matrix of shape
            #  n_cams x n_timepoints_with_used_detections x n_corners x 2
            # Memory footprint at this stage is al but critical.
            corners, used_frames_ids = \
                detect_corners(self.rec_file_names, self.n_frames, self.board_params, self.opts,
                               rec_pipelines=self.rec_pipelines)

            calibs_single = self.obtain_single_cam_calibrations(calibs_single=self.opts.pop('internals'),
                                                                corners=corners)

            if self.opts['optimize_ind_cams']:
                for i_cam, calib in enumerate(calibs_single):
                    # analytically estimate initial camera poses
                    calibs_interim = estimate_cam_poses([calib], self.opts, corners=corners[[i_cam]],
                                                        required_corner_idxs=required_corner_idxs)

                    calibs_fit_single, rvecs_boards, tvecs_boards, _, _ = self.optimize_poses(corners[[i_cam]],
                                                                                              calibs_interim)
                    calibs_single[i_cam].update(
                        helper.combine_calib_with_board_params(calibs_fit_single, rvecs_boards, tvecs_boards)[0])
                    calibs_single[i_cam]['frames_mask'] = np.sum(~np.isnan(corners[i_cam][:, :, 1]), axis=1) > 0

            # analytically estimate initial camera poses
            calibs_multi = estimate_cam_poses(calibs_single, self.opts, corners=corners,
                                              required_corner_idxs=required_corner_idxs)

            # Save intermediate result, for dev purposes on optimization (quote code above and unquote code below)
            # pose_params = optimization.make_common_pose_params(calibs_multi, corners)
            result = self.build_result(calibs_multi,
                                       corners=corners, used_frames_ids=used_frames_ids,
                                       # rvecs_boards=pose_params[0], tvecs_boards=pose_params[1],
                                       other={'calibs_single': calibs_single})
            self.save_multicalibration(result, 'preoptim')

        if self.opts['debug']:
            args, vars_free = make_optim_input(self.board_params, calibs_multi, corners, self.opts)
            test_objective_function(calibs_multi, vars_free, args, corners, self.board_params,
                                    individual_poses=True)

        print('OPTIMIZING ALL POSES')
        # self.plot(calibs_single, corners, used_frames_ids, self.board_params, 3, 35)
        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_poses(corners, calibs_multi)

        if self.opts['debug']:
            calibs_fit = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards)
            test_objective_function(calibs_fit, min_result.x, args, corners, self.board_params,
                                    individual_poses=True)

        print('OPTIMIZING ALL PARAMETERS I')
        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(corners, calibs_fit)

        # According to tests with good calibration recordings, the following steps are unnecessary and optimality was
        #  already reached in the previous step
        if self.opts["optimize_board_poses"]:
            if self.opts['debug']:
                calibs_fit = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards)
                test_objective_function(calibs_fit, min_result.x, args, corners, self.board_params,
                                        individual_poses=True)

            print('OPTIMIZING BOARD POSES')
            calibs_fit, rvecs_boards, tvecs_boards, _, _ = self.optimize_board_poses(corners, calibs_fit,
                                                                                                 prev_fun=min_result.fun)
            calibs_fit = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards)

            print('OPTIMIZING ALL PARAMETERS II')
            calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(corners, calibs_fit)

        if self.opts["reject_corners"]:
            result = self.build_result(calibs_fit,
                                       corners=corners, used_frames_ids=used_frames_ids,
                                       min_result=min_result, args=args,
                                       rvecs_boards=rvecs_boards, tvecs_boards=tvecs_boards,
                                       other={'calibs_single': calibs_single, 'calibs_multi': calibs_multi,
                                                'board_coords_3d_0': board.make_board_points(self.board_params)})
            self.save_multicalibration(result, 'prereject')

            print('INSPECTING CORNERS')
            corners, rejected_poses, rejected_corners = helper.reject_corners(corners, min_result.fun,
                                                                              self.board_params,
                                                                              self.opts["rejection"])
            rvecs_boards = rvecs_boards[~rejected_poses]
            tvecs_boards = tvecs_boards[~rejected_poses]
            used_frames_ids = used_frames_ids[~rejected_poses]
            calibs_fit = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards)

            print('OPTIMIZING ALL PARAMETERS III')
            calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_calibration(corners, calibs_fit)

        # No board poses in final calibration!
        calibs_test = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards, copy=True)
        test_objective_function(calibs_test, min_result.x, args, corners, self.board_params,
                                individual_poses=True)

        result = self.build_result(calibs_fit,
                                   corners=corners, used_frames_ids=used_frames_ids,
                                   min_result=min_result, args=args,
                                   rvecs_boards=rvecs_boards, tvecs_boards=tvecs_boards,
                                   other={'calibs_single': calibs_single, 'calibs_multi': calibs_multi,
                                          'board_coords_3d_0': board.make_board_points(self.board_params)})

        print('SAVE MULTI CAMERA CALIBRATION')
        self.save_multicalibration(result)
        # Builds a part of the v1 result that is necessary for other software
        self.save_multicalibration(helper.build_v1_result(result), 'multicalibration_v1')

        print('FINISHED MULTI CAMERA CALIBRATION')
        return

    def obtain_single_cam_calibrations(self, calibs_single, corners):

        cams_2calibrate = []
        for i_cam, cam_calib in enumerate(calibs_single):
            if cam_calib is not None:
                calibs_single[i_cam] = self.estimate_board_positions_in_single_cam(cam_calib, corners[i_cam])
            else:
                cams_2calibrate.append(i_cam)

        #  perform single calibration if needed
        cams_calibrated = self.perform_single_cam_calibrations(corners,
                                                               camera_indexes=cams_2calibrate,
                                                               calibs_init=self.calibs_init)
        for i, i_cam in enumerate(cams_2calibrate):
            calibs_single[i_cam] = cams_calibrated[i]

        return calibs_single

    @DeprecationWarning
    def perform_board_position_estimation(self, calibs_single, corners):
        print('ESTIMATE BOARD POSITIONS')

        for i_cam, calib in enumerate(calibs_single):
            calibs_single[i_cam] = self.estimate_board_positions_in_single_cam(calib, corners[i_cam])

        return calibs_single

    def estimate_board_positions_in_single_cam(self, calib, corners_cam, mask=None):
        if mask is None:
            mask = np.sum(~np.isnan(corners_cam[:, :, 1]), axis=1) > 0

        corners_nn = corners_cam[mask]
        corners_use, ids_use = helper.corners_array_to_ragged(corners_nn)

        board_positions = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count())))(
            delayed(self.estimate_single_board_position)(calib,
                                                         corners_use[i_pose],
                                                         ids_use[i_pose],
                                                         self.board_params,
                                                         self.opts)
            for i_pose in range(len(corners_use)))

        if 'rvec_cam' in calib:
            calib['rvec_cam'] = np.zeros_like(calib['rvec_cam'])
            calib['tvec_cam'] = np.zeros_like(calib['tvec_cam'])

        calib['rvecs'] = np.full((corners_cam.shape[0], 3), np.nan)
        calib['tvecs'] = np.full((corners_cam.shape[0], 3), np.nan)
        calib['frames_mask'] = mask

        pose_idxs = np.where(mask)[0]
        for pose_idx, pos in zip(pose_idxs, board_positions):
            if pos[0]:
                calib['rvecs'][pose_idx] = pos[1][:,0]
                calib['tvecs'][pose_idx] = pos[2][:,0]
            else:
                calib['frames_mask'][pose_idx] = False

        return calib

    @staticmethod
    def estimate_single_board_position(calib, corners, ids, board_params, opts):
        if len(ids) < 4:
            return 0, np.full((3,), np.nan), np.full((3,), np.nan)

        retval, rvec, tvec = cv2.solvePnP(board.make_board_points(board_params)[ids].reshape((-1,3)),
                                          corners.reshape((-1, 2)),
                                          calib["A"], calib["k"],
                                          flags=cv2.SOLVEPNP_IPPE)
        return retval, rvec, tvec

    def perform_single_cam_calibrations(self, corners, camera_indexes=None, calibs_init=None):
        print('PERFORM SINGLE CAMERA CALIBRATION')

        if camera_indexes is None:
            camera_indexes = range(len(self.readers))

        if calibs_init is None:
            calibs_init = [None for _ in corners]

        print(int(np.floor(multiprocessing.cpu_count())))
        calibs_single = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count())))(
            delayed(calibrate_single_camera)(corners[i_cam],
                                             camfunctions.get_header_from_reader(self.readers[i_cam])[
                                                 'sensorsize'],
                                             self.board_params,
                                             {'free_vars': self.opts['free_vars'][i_cam],
                                              'aruco_calibration': self.opts['aruco_calibration'][i_cam]
                                              },
                                             calib_init=calibs_init[i_cam])
            for i_cam in camera_indexes)

        for i_cam, calib in enumerate(calibs_single):
            print(
                f'Used {(~np.isnan(calib["rvecs"][:, 1])).sum(dtype=int):03d} '
                f'frames for single cam calibration for cam {i_cam:02d}'
            )
            print(calib['rvecs'][0])
            print(calib['tvecs'][0])

        return calibs_single

    def optimize_poses(self, corners, calibs_multi, opts=None, board_params=None):
        if opts is None:
            opts = self.opts
        if board_params is None:
            board_params = self.board_params

        pose_opts = deepcopy(opts)
        free_vars = pose_opts['free_vars']
        for cam in free_vars:
            cam['A'][:] = False
            cam['k'][:] = False
            cam['xi'] = False

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_params, opts=pose_opts)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def optimize_board_poses(self, corners, calibs_multi, opts=None, board_params=None, prev_fun=None):
        if opts is None:
            opts = self.opts
        if board_params is None:
            board_params = self.board_params

        pose_opts = deepcopy(opts)
        pose_opts['optimization']['ftol'] = 1e-13
        pose_opts['optimization']['gtol'] = 1e-13
        pose_opts['optimization']['xtol'] = 1e-13
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
                nearest_i_pose = helper.nearest_element(i_pose, good_poses)  # nearest_i_pose = i_pose if i_pose in good_poses
                calib["rvecs"] = calib_orig["rvecs"][[nearest_i_pose]]
                calib["tvecs"] = calib_orig["tvecs"][[nearest_i_pose]]

            # print(i_pose, rvecs_boards[i_pose])
            calibs_fit_pose, rvecs_boards[i_pose], tvecs_boards[i_pose], min_result, args = \
                camfunctions.optimize_calib_parameters(corners_pose, calibs_multi_pose, board_params, opts=pose_opts,
                                                       verbose=0)
            # print(i_pose, rvecs_boards[i_pose], min_result.cost)
        return calibs_fit_pose, rvecs_boards, tvecs_boards, None, None

    def optimize_calibration(self, corners, calibs_multi, opts=None, board_params=None):
        if opts is None:
            opts = self.opts
        if board_params is None:
            board_params = self.board_params

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_params, opts=opts)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def build_result(self, calibs,
                     corners=None, used_frames_ids=None,
                     rvecs_boards=None, tvecs_boards=None, min_result=None, args=None,
                     other=None):

        # savemat cannot deal with None
        if other is None:
            other = dict()
        if tvecs_boards is None:
            tvecs_boards = []
        if rvecs_boards is None:
            rvecs_boards = []
        if used_frames_ids is None:
            used_frames_ids = []
        if corners is None:
            corners = []
        result = {
            'version': 2.2,  # Increase when this structure changes
            'calibs': calibs,
            # This field shall always hold all intrinsically necessary information to project and triangulate.
            'board_params': self.board_params,  # All parameters to recreate the board
            'rec_file_names': self.rec_file_names,  # Recording filenames, may be used for cam names
            'vid_headers': [camfunctions.get_header_from_reader(r) for r in self.readers],
            # Headers. No content structure guaranteed
            'info': {  # Additional nonessential info from the calibration process
                'cost_val_final': np.NaN,
                'optimality_final': np.NaN,
                'corners': corners,
                'used_frames_ids': used_frames_ids,
                'rvecs_boards': rvecs_boards,
                'tvecs_boards': tvecs_boards,
                'opts': self.opts,
                'other': other,  # Additional info without guaranteed structure
            }
        }

        if self.rec_pipelines is not None:
            result['rec_pipelines'] = self.rec_pipelines
        # savemat cannot deal with none!
        if min_result is not None:
            result['info']['fun_final'] = min_result.fun
            result['info']['cost_val_final'] = min_result.cost
            result['info']['optimality_final'] = min_result.optimality

        return result

    def save_multicalibration(self, result, filename="multicam_calibration"):
        # save
        result_path = self.data_path + '/' + filename
        np.save(result_path + '.npy', result)
        scipy_io_savemat(result_path + '.mat', result)
        print('Saved multi camera calibration to file {:s}'.format(result_path))
        return

    # Debug function
    def plot(self, calibs, corners, used_frames_ids, board_params, cidx, fidx):
        import matplotlib.pyplot as plt
        from scipy.spatial.transform import Rotation as R  # noqa
        import camfunctions_ag

        board_coords_3d_0 = board.make_board_points(board_params)

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

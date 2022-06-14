import os
from copy import deepcopy

import numpy as np
from scipy.io import savemat as scipy_io_savemat
import cv2

from pathlib import Path

import imageio
from ccvtools import rawio  # noqa

import multiprocessing
from joblib import Parallel, delayed

from calibcam.detection import detect_corners
from calibcam.exceptions import *
from calibcam import helper, camfunctions, board, optimization, compatibility

from calibcam.calibrator_opts import get_default_opts
from calibcam.pose_estimation import estimate_cam_poses


class CamCalibrator:
    def __init__(self, recordings, board_name=None, opts=None):
        if opts is None:
            opts = {}

        self.board_name = board_name  # Currently, recordings are needed to determine the board path in most cases

        # Board
        self.board_params = None

        # Videos
        self.readers = None
        self.rec_file_names = None
        self.data_path = None
        self.n_frames = np.NaN

        # Options
        self.opts = helper.deepmerge_dicts(opts, get_default_opts())

        self.set_recordings(recordings)

        return

    def get_board_params_from_name(self, board_name):
        if board_name is not None:
            board_params = board.get_board_params(board_name)
        else:
            board_params = board.get_board_params(Path(self.rec_file_names[0]).parent)
        return board_params

    def set_recordings(self, recordings):
        # check if input files are valid files
        try:
            self.readers = [imageio.get_reader(rec) for rec in recordings]
        except ValueError:
            print('At least one unsupported format supplied')
            raise UnsupportedFormatException

        self.data_path = os.path.expanduser(os.path.dirname(recordings[0]))
        self.rec_file_names = recordings

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
                            self.board_params["boardWidth"] - 1,
                            (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 2),
                            (self.board_params["boardWidth"] - 1) * (self.board_params["boardHeight"] - 1) - 1,
                            ]  # Corners that we require to be detected for pose estimation
        if self.opts["optimize_only"]:  # We expect that detections and single cam calibs are already present
            preoptim = np.load(self.data_path + '/preoptim.npy', allow_pickle=True)[()]
            preoptim = compatibility.update_preoptim(preoptim, n_corners)

            calibs_single = preoptim['info']['other']['calibs_single']
            if 'used_frames_ids' in preoptim['info']:
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
                detect_corners(self.rec_file_names, self.n_frames, self.board_params, self.opts)

            # perform single calibration
            calibs_single = self.perform_single_cam_calibrations(corners)

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

        print('OPTIMIZING POSES')

        # self.plot(calibs_single, corners, used_frames_ids, self.board_params, 3, 35)

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = self.optimize_poses(corners, calibs_multi)
        calibs_fit = helper.combine_calib_with_board_params(calibs_fit, rvecs_boards, tvecs_boards)

        print('OPTIMIZING ALL PARAMETERS')

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            self.optimize_calibration(corners, calibs_fit)

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

    @staticmethod
    def calibrate_single_camera(corners_cam, sensor_size, board_params, opts, mask=None):
        if mask is None:
            mask = np.sum(~np.isnan(corners_cam[:, :, 1]), axis=1) > 0  # Test for degeneration should be performed beforehand and respective frames excluded from corner array

        n_used_frames = np.sum(mask)

        if n_used_frames == 0:
            return []

        corners_nn = corners_cam[mask]
        corners_use, ids_use = helper.corners_array_to_ragged(corners_nn)

        cal_res = cv2.aruco.calibrateCameraCharucoExtended(corners_use,  # noqa
                                                           ids_use,
                                                           board.make_board(board_params),
                                                           sensor_size,
                                                           None,
                                                           None,
                                                           **opts['detection']['aruco_calibration'])

        rvecs = np.empty(shape=(mask.size, 3))
        rvecs[:] = np.NaN
        tvecs = np.empty(shape=(mask.size, 3))
        tvecs[:] = np.NaN
        retval, A, k,  = cal_res[0:3]

        print(cal_res[3][0])
        print(cal_res[4][0])

        rvecs[mask, :] = np.asarray(cal_res[3])[..., 0]
        tvecs[mask, :] = np.asarray(cal_res[4])[..., 0]

        cal = {
            'rvec_cam': np.asarray([0., 0., 0.]),
            'tvec_cam': np.asarray([0., 0., 0.]),
            'A': np.asarray(A),
            'k': np.asarray(k),
            'rvecs': np.asarray(rvecs),
            'tvecs': np.asarray(tvecs),
            'repro_error': retval,  # Not that from here on values are NOT expanded to full frames range, see frames_mask
            'std_intrinsics': cal_res[5],
            'std_extrinsics': cal_res[6],
            'per_view_errors': cal_res[7],
            'frames_mask': mask,
        }
        print('Finished single camera calibration.')
        return cal

    def perform_single_cam_calibrations(self, corners):
        print('PERFORM SINGLE CAMERA CALIBRATION')

        # calibs_single = [self.calibrate_single_camera(corners[i_cam],
        #                                               camfunctions.get_header_from_reader(self.readers[i_cam])['sensorsize'],
        #                                               self.board_params,
        #                                               self.opts)
        #                  for i_cam in range(len(self.readers))]
        print(int(np.floor(multiprocessing.cpu_count())))
        calibs_single = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count())))(
            delayed(self.calibrate_single_camera)(corners[i_cam],
                                                  camfunctions.get_header_from_reader(self.readers[i_cam])[
                                                      'sensorsize'],
                                                  self.board_params,
                                                  self.opts)
            for i_cam in range(len(self.readers)))

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
        pose_opts['free_vars']['A'][:] = False
        pose_opts['free_vars']['k'][:] = False

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_params,
                                                   [camfunctions.get_header_from_reader(r)["offset"] for r in self.readers],
                                                   opts=pose_opts)

        return calibs_fit, rvecs_boards, tvecs_boards, min_result, args

    def optimize_calibration(self, corners, calibs_multi, opts=None, board_params=None, readers=None):
        if opts is None:
            opts = self.opts
        if board_params is None:
            board_params = self.board_params
        if readers is None:
            readers = self.readers

        calibs_fit, rvecs_boards, tvecs_boards, min_result, args = \
            camfunctions.optimize_calib_parameters(corners, calibs_multi, board_params,
                                                   [camfunctions.get_header_from_reader(r)["offset"] for r in readers],
                                                   opts=opts)

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
            'version': 2.1,  # Increase when this structure changes
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

        # savemat cannot deal with none!
        if min_result is not None:
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
        board_coords_3d = camfunctions_ag.board_to_ideal_plane(board_coords_3d)

        board_coords_3d_nd = camfunctions_ag.ideal_to_sensor(board_coords_3d, calibs[cidx]['A'])

        board_coords_3d_d = camfunctions_ag.distort(board_coords_3d, calibs[cidx]['k'])
        board_coords_3d_d = camfunctions_ag.ideal_to_sensor(board_coords_3d_d, calibs[cidx]['A'])

        plt.plot(board_coords_3d_d[(0, 4, 34), 0], board_coords_3d_d[(0, 4, 34), 1], 'r+')
        plt.plot(board_coords_3d_nd[(0, 4, 34), 0], board_coords_3d_nd[(0, 4, 34), 1], 'g+')

        plt.show()

import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from itertools import compress
from pathlib import Path

import imageio
from ccvtools import rawio

import time

from autograd import elementwise_grad

from scipy.optimize import least_squares

from . import multical_func as func
from .helper import save_multicalibration_to_matlabcode, check_detections_nondegenerate
from .board import get_board_params

from pprint import pprint


class UnsupportedFormatException(Exception):
    pass


class UnequalFrameCountException(Exception):
    pass


# TODO: clean up mess of all these useless globals/properties inside methods
# TODO: reorganize output. class should print anything itself ...
class Calibrator:
    def __init__(self, board_name=None):
        self.board_name = board_name

        self.readers = None
        self.board_params = None
        self.board = None
        self.recFileNames = []
        self.dataPath = None
        self.nFrames = np.NaN
        self.recordingIsLoaded = False
        self.result = None

        self.x_all_fit = None

        self.x_single_fit = None
        self.r1_single_fit = None
        self.t1_single_fit = None
        self.R1_single_fit = None

        self.x_fit = None
        self.RX1_fit = None
        self.rX1_fit = None
        self.tX1_fit = None
        self.k_fit = None
        self.A_fit = None

        self.r1_fit = None
        self.t1_fit = None
        self.R1_fit = None

        self.tol = np.finfo(np.float64).eps  # machine epsilon
        self.min_result = None
        self.resultPath = None

        self.nFeatures = int(0)  # Number of overall corners of board
        self.allFramesMask = np.empty(0, dtype=bool)
        self.allCorners_list = []
        self.allIds_list = []
        self.mask_single = np.empty(0)  # later nC x nF, mask of frames that are only detected in one camera
        self.mask_multi = np.empty(0)  # later nF
        self.indexRefCam = np.nan
        self.nPoses_single = []  # later nC
        self.calib_single = dict()
        self.calib_multi = dict()
        self.nPoses = int(0)

        self.info = dict()  # Info on calibration for documentation purposes ONLY

        # flags and criteria for cv2.aruco.calibrateCameraCharuco
        self.flags = (cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3)
        self.criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                         30,
                         float(np.finfo(np.float32).eps))
        return

    def reset_recordings(self):
        self.recFileNames = []
        self.readers = []
        self.dataPath = None
        self.nFrames = np.NaN
        self.recordingIsLoaded = False
        self.result = None

    def set_board(self, board_name):
        if board_name is not None:
            self.board_params = get_board_params(board_name)
        else:
            self.board_params = get_board_params(Path(self.recFileNames[0]).parent)

        self.nFeatures = (self.board_params['boardWidth'] - 1) * (self.board_params['boardHeight'] - 1)
        self.board = cv2.aruco.CharucoBoard_create(self.board_params['boardWidth'],
                                                   self.board_params['boardHeight'],
                                                   self.board_params['square_size_real'],
                                                   self.board_params['marker_size']*self.board_params['square_size_real'],
                                                   self.board_params['dictionary'])
        return

    def set_recordings(self, recordings, allow_unequal=False):
        # check if input files are valid files:
        try:
            self.readers = [imageio.get_reader(rec) for rec in recordings]
        except ValueError:
            print('At least one unsupported format supplied')
            raise UnsupportedFormatException

        self.dataPath = os.path.dirname(recordings[0])
        self.recFileNames = recordings

        n_frames = np.zeros(len(self.readers), dtype=np.int64)
        for (i_cam, reader) in enumerate(self.readers):
            n_frames[i_cam] = self.get_frames_from_cam(reader)

        self.nFrames = n_frames[0]
        # check if frame number is consistent:
        if not np.all(np.equal(n_frames[0], n_frames[1:])):
            print('WARNING: Number of frames is not identical for all cameras')
            print('Number of detected frames per camera:')
            for (i_cam, nF) in enumerate(n_frames):
                print(f'\tcamera {i_cam:03d}:\t{nF:04d}')

            if allow_unequal:
                self.nFrames = np.int64(np.min(n_frames))
            else:
                # raise exception for outside confirmation
                raise UnequalFrameCountException
        self.recordingIsLoaded = True

        # Initialize quantities with camera and frame sizes
        self.allFramesMask = np.zeros((len(self.readers), self.nFrames),
                                      dtype=bool)
        self.mask_single = np.zeros((len(self.readers), self.nFrames),
                                    dtype=bool)
        self.mask_multi = np.zeros(self.nFrames, dtype=bool)
        self.nPoses_single = np.zeros(len(self.readers), dtype=np.int64)

        self.set_board(self.board_name)

    @staticmethod
    def get_frames_from_cam(reader):
        n_frames = len(reader)  # len() may be Inf for formats where counting frames can be expensive
        if 1000000000000000 < n_frames:
            try:
                n_frames = reader.count_frames()
            except ValueError:
                print('Could not determine number of frames')
                raise UnsupportedFormatException

        return n_frames

    @staticmethod
    def get_header(reader):
        header = reader.get_meta_data()
        # Add required headers that are not normally part of standard video formats but are required information for a full calibration
        # TODO add option to supply this via options. Currently, compressed
        if "sensor" in header:
            header['offset'] = tuple(header['sensor']['offset'])
            header['sensorsize'] = tuple(header['sensor']['size'])
            del header['sensor']
        else:
            if 'offset' not in header:
                print("Setting offset to 0!")
                header['offset'] = tuple(np.asarray([0, 0]))

            if 'sensorsize' not in header:
                print("Inferring sensor size from image")
                header['sensorsize'] = tuple(reader.get_data(0).shape[0:2])

        return header

    def perform_multi_calibration(self):
        # detect corners
        self.detect_corners()

        # split into two frame sets
        # first set contains frames for single calibration
        # second set contains frames for multi calibration
        self.split_frame_sets()

        # perform single calibration
        cal_single_list = self.perform_single_calibration()
        self.info['cal_single_list'] = cal_single_list

        # generate calib_single
        self.calib_single = self.generate_calib_single(cal_single_list)

        # perform multi calibration
        cal_multi_list = self.perform_single_calibration_for_multi()
        self.info['cal_multi_list'] = cal_multi_list

        # generate calib_multi
        self.calib_multi = self.generate_calib_multi(cal_single_list, cal_multi_list)

        # the following functions are based on multical_main.py
        print('PREPARE FOR MULTI CAMERA CALIBRATION')
        args, bounds = self.generate_args()
        self.info['args'] = args
        print('START MULTI CAMERA CALIBRATION')
        self.start_optimization(args, bounds)
        self.get_fitted_paras(args)
        print('SAVE MULTI CAMERA CALIBRATION')
        self.save_multicalibration()
        save_multicalibration_to_matlabcode(self.result, self.dataPath)
        print('FINISHED MULTI CAMERA CALIBRATION')
        return

    def detect_corners(self):
        print('DETECTING FEATURES')
        for i_cam, reader in enumerate(self.readers):
            print('Detecting features in camera {:02d}'.format(i_cam))
            all_corners, all_ids, frames_mask = self.detect_corners_cam(reader)
            self.allFramesMask[i_cam, :] = frames_mask
            self.allCorners_list.append(all_corners)
            self.allIds_list.append(all_ids)
            print(f'Detected features in {np.sum(self.allFramesMask[i_cam]):04d}  frames in camera {i_cam:02d}')
        return

    def detect_corners_cam(self, reader):
        all_corners = []
        all_ids = []
        previous_used_frame = np.nan
        previous_corners = np.zeros((self.nFeatures, 2), dtype=np.float64)
        current_corners = np.zeros((self.nFeatures, 2), dtype=np.float64)

        frames_mask = np.zeros(self.get_frames_from_cam(reader), dtype=bool)

        # calculate offset
        offset_x = self.get_header(reader)['offset'][0]
        offset_y = self.get_header(reader)['offset'][1]

        for (i_frame, frame) in enumerate(reader):
            mask_value2add = False
            corners2add = []
            ids2add = []
            # feature detection
            if len(frame.shape) > 2:
                frame = frame[:, :, 1]
            corners, ids, rejectedImgPoints = \
                cv2.aruco.detectMarkers(frame,
                                        self.board_params['dictionary'],
                                        parameters=self.set_detector_parameters())

            if len(corners) > 0:
                corners_ref, ids_ref = \
                    cv2.aruco.refineDetectedMarkers(frame,
                                                    self.board,
                                                    corners,
                                                    ids,
                                                    rejectedImgPoints,
                                                    minRepDistance=10.0,
                                                    errorCorrectionRate=3.0,
                                                    checkAllOrders=True,
                                                    parameters=self.set_detector_parameters())[0:2]
                retval, charucoCorners, charucoIds = \
                    cv2.aruco.interpolateCornersCharuco(corners_ref,
                                                        ids_ref,
                                                        frame,
                                                        self.board,
                                                        minMarkers=2)
                # checks if the result is degenerated
                if check_detections_nondegenerate(self.board_params['boardWidth'], charucoIds):
                    # add offset
                    charucoCorners[:, :, 0] = charucoCorners[:, :, 0] + offset_x
                    charucoCorners[:, :, 1] = charucoCorners[:, :, 1] + offset_y
                    mask_value2add = True
                    corners2add = np.copy(charucoCorners)
                    ids2add = np.copy(charucoIds)
                    # checks if consecutive frames are too similar
                    if not (np.isnan(previous_used_frame)):
                        # get current and previous features
                        previous_corners[:, :] = 0
                        previous_corners[all_ids[previous_used_frame].ravel()] = all_corners[
                            previous_used_frame].squeeze()
                        current_corners[:, :] = 0
                        current_corners[charucoIds.ravel()] = charucoCorners.squeeze()
                        # calculates euclidian distance between features
                        diff = current_corners - previous_corners
                        ids_use = np.intersect1d(all_ids[previous_used_frame].ravel(),
                                                 charucoIds.ravel())
                        diff = diff[ids_use]
                        dist = np.sqrt(np.sum(diff ** 2, 1))

                        # use frame when all ids are different
                        if np.size(dist) == 0:
                            dist_max = np.inf
                        else:
                            dist_max = np.max(dist)
                        # check if maximum distance is high enough
                        if not (dist_max > 0.0):
                            mask_value2add = False
                            corners2add = []
                            ids2add = []
            if mask_value2add:
                previous_used_frame = i_frame

            frames_mask[i_frame] = mask_value2add
            all_corners.append(corners2add)
            all_ids.append(ids2add)

        return all_corners, all_ids, frames_mask

    def split_frame_sets(self):
        for i_frame in range(self.nFrames):
            nFrames_used = np.sum(self.allFramesMask[:, i_frame])
            # if corners are detected in more than one camera use the frame for multi calibration
            if nFrames_used > 1:
                self.mask_multi[i_frame] = True
        # find reference cameras => camera which has seen the most frames within the multi calibration frame set
        multi_frame_count = np.sum(self.allFramesMask[:, self.mask_multi], 1)
        self.indexRefCam = np.where(multi_frame_count == np.max(multi_frame_count))[0][0]
        # only use frames where the reference camera has detected some features
        self.mask_multi = (self.mask_multi & self.allFramesMask[self.indexRefCam])
        # use all frames that do not belong to the multi calibration frame set for single calibration
        mask = np.logical_not(self.mask_multi)
        self.mask_single[:, mask] = self.allFramesMask[:, mask]
        return

    def calibrate_camera(self, cam_idx, mask=None):
        if mask is None:
            mask = np.asarray([len(c) > 0 for c in self.allCorners_list[cam_idx]], dtype=bool)

        n_used_frames = np.sum(mask)
        print(f'Using {n_used_frames:03d} frames to perform single camera calibration for camera {cam_idx:02d}')
        if n_used_frames > 0:  # do this to not run into indexing issues
            corners_use = list(compress(self.allCorners_list[cam_idx],
                                        mask))
            ids_use = list(compress(self.allIds_list[cam_idx],
                                    mask))
            cal = cv2.aruco.calibrateCameraCharuco(corners_use,
                                                   ids_use,
                                                   self.board,
                                                   self.get_header(self.readers[cam_idx])['sensorsize'],
                                                   None,
                                                   None,
                                                   flags=self.flags,
                                                   criteria=self.criteria)
            print('Completed single camera calibration')
            print(f'Reprojection error:\t{cal[0]:.08f}')

            return cal
        else:
            return []

    def perform_single_calibration(self):
        print('PERFORM SINGLE CAMERA CALIBRATION #1')
        cal_single_list = []
        for i_cam in range(len(self.readers)):
            mask = self.mask_single[i_cam]
            cal = self.calibrate_camera(i_cam, None)
            cal_single_list.append(cal)
        return cal_single_list

    def generate_calib_single(self, cal_single_list):
        calib_single = dict()
        self.nPoses_single = np.sum(self.mask_single, 1)
        for i_cam in range(0, len(self.readers), 1):
            key = 'cam{:01d}'.format(i_cam)
            calib_single[key] = dict()
            calib_single[key]['charuco_ids'] = []
            calib_single[key]['charuco_corners'] = []
            calib_single[key]['rotation_vectors'] = []
            calib_single[key]['translation_vectors'] = []
            index = 0
            for i_frame in range(self.nFrames):
                if self.mask_single[i_cam, i_frame]:
                    nFeats = np.size(self.allIds_list[i_cam][i_frame])
                    ids_use = np.array(self.allIds_list[i_cam][i_frame],
                                       dtype=np.int64).reshape(nFeats, 1)
                    calib_single[key]['charuco_ids'].append(ids_use)
                    corners_use = np.array(self.allCorners_list[i_cam][i_frame],
                                           dtype=np.float64).reshape(nFeats, 2)
                    calib_single[key]['charuco_corners'].append(corners_use)
                    # r and t
                    rotations_use = np.array(cal_single_list[i_cam][3][index],
                                             dtype=np.float64).reshape(3, 1)
                    calib_single[key]['rotation_vectors'].append(rotations_use)
                    translations_use = np.array(cal_single_list[i_cam][4][index],
                                                dtype=np.float64).reshape(3, 1)
                    calib_single[key]['translation_vectors'].append(translations_use)
                    index = index + 1
        return calib_single

    def perform_single_calibration_for_multi(self):
        print('PERFORM SINGLE CAMERA CALIBRATION #2')
        print('The following single camera calibrations will be used to initialize the multi camera calibration')
        cal_multi_list = []
        for i_cam in range(len(self.readers)):
            mask = self.mask_multi & self.allFramesMask[i_cam]
            cal = self.calibrate_camera(i_cam, mask)
            cal_multi_list.append(cal)
        return cal_multi_list

    def generate_calib_multi(self, cal_single_list, cal_multi_list):
        calib_multi = dict()
        self.nPoses = int(np.sum(self.mask_multi))
        A = np.zeros((len(self.readers), 3, 3), dtype=np.float64)
        k = np.zeros((len(self.readers), 5), dtype=np.float64)
        for i_cam in range(len(self.readers)):
            key = 'cam{:01d}'.format(i_cam)
            calib_multi[key] = dict()
            A[i_cam] = np.array(cal_multi_list[i_cam][1],
                                dtype=np.float64).reshape(3, 3)
            k[i_cam] = np.array(cal_multi_list[i_cam][2],
                                dtype=np.float64).reshape(1, 5)
            nUsedFrames = np.sum(self.mask_single[i_cam])
            if nUsedFrames > 0:
                if cal_single_list[i_cam][0] < cal_multi_list[i_cam][0]:
                    A[i_cam] = np.array(cal_single_list[i_cam][1],
                                        dtype=np.float64).reshape(3, 3)
                    k[i_cam] = np.array(cal_single_list[i_cam][2],
                                        dtype=np.float64).reshape(1, 5)
            #
            calib_multi[key]['camera_matrix'] = A[i_cam]
            calib_multi[key]['dist_coeffs'] = k[i_cam]
            # rest
            calib_multi[key]['charuco_ids'] = []
            calib_multi[key]['charuco_corners'] = []
            calib_multi[key]['rotation_vectors'] = []
            calib_multi[key]['translation_vectors'] = []
            index = np.int64(0)
            for i_frame in range(self.nFrames):
                if self.mask_multi[i_frame]:
                    nFeats = np.size(self.allIds_list[i_cam][i_frame])
                    ids_use = np.array(self.allIds_list[i_cam][i_frame],
                                       dtype=np.int64).reshape(nFeats, 1)
                    calib_multi[key]['charuco_ids'].append(ids_use)
                    corners_use = np.array(self.allCorners_list[i_cam][i_frame],
                                           dtype=np.float64).reshape(nFeats, 2)
                    calib_multi[key]['charuco_corners'].append(corners_use)
                    # r and t
                    # only append the list here when the camera has actually seen the pattern in the respective frame
                    # i.e. we have an estimate for r and t
                    if self.allFramesMask[i_cam, i_frame]:
                        rotations_use = np.array(cal_multi_list[i_cam][3][index],
                                                 dtype=np.float64).reshape(3, 1)
                        calib_multi[key]['rotation_vectors'].append(rotations_use)
                        translations_use = np.array(cal_multi_list[i_cam][4][index],
                                                    dtype=np.float64).reshape(3, 1)
                        calib_multi[key]['translation_vectors'].append(translations_use)
                        index = index + 1
        return calib_multi

    def generate_args(self):
        args = dict()

        # GENERAL
        kSize = 5
        tSize = 3
        rSize = 3
        ASize = 4
        nVars = rSize + tSize + kSize + ASize + rSize + tSize
        nAllVars = (len(self.readers) - 1) * (rSize + tSize) + len(self.readers) * kSize + len(
            self.readers) * ASize + self.nPoses * (rSize + tSize)
        nAllVars_single = np.sum((rSize + tSize) * self.nPoses_single)

        args_general = {
            'nCameras': len(self.readers),  # total number of cameras
            'nPoses': self.nPoses,  # total number of poses
            'nPoses_single': self.nPoses_single,  # total number of poses (single calibration)
            'boardWidth': self.board_params['boardWidth'],  # width of charuco board
            'boardHeight': self.board_params['boardHeight'],  # height of charuco board
            'nFeatures': self.nFeatures,  # total number of features on the charuco board
            'nRes': self.nFeatures * self.nPoses * len(self.readers),
            # total number of residuals for each direction (x, y)
            'nRes_single': np.sum(self.nFeatures * self.nPoses_single),
            # total number of residuals for each direction (x, y) (single calibration)
            'kSize': kSize,  # number of free distortion coefficients per residual
            'tSize': tSize,  # number of free translation parameters per residual
            'rSize': rSize,  # number of free rotation parameters per residual
            'ASize': ASize,  # number of free variables in camera matrix per residual
            'nVars': nVars,
            # number of free parameters per residual TODO Is this even correct with 2x rSize/tSize?
            'nAllVars': nAllVars,  # total number of free parameters
            'nAllVars_single': nAllVars_single,  # total number of free parameters (single calibration)
            'indexRefCam': self.indexRefCam,  # index of the reference camera
        }
        args = {**args, **args_general}

        # CONSTANTS
        M, m, delta = func.map_calib2consts(self.calib_multi, args_general)
        M_single, m_single, delta_single = func.map_calib2consts_single(self.calib_single, args_general)

        args_constants = {
            'M': M,
            'm': m,
            'delta': delta,
            'M_single': M_single,
            'm_single': m_single,
            'delta_single': delta_single,
        }
        args = {**args, **args_constants}

        # JACOBIAN
        args_jacobian = {
            'jac_x': np.asarray([elementwise_grad(func.calc_res_x, i_var) for i_var in range(nVars)]),
            'jac_y': np.asarray([elementwise_grad(func.calc_res_y, i_var) for i_var in range(nVars)]),
        }

        args = {**args, **args_jacobian}

        # OPTIMIZATION
        x0 = func.set_x0_objFunc(self.calib_multi, args)  # define initial x
        x0_single = func.set_x0_objFunc_single(self.calib_single, args)
        x0_all = np.concatenate([x0, x0_single], 0)

        free_para = np.ones(nAllVars, dtype=bool)
        index_free_k = (len(self.readers) - 1) * (rSize + tSize)
        free_para[index_free_k:index_free_k + len(self.readers) * kSize] = False
        free_para[index_free_k:index_free_k + len(self.readers) * kSize:kSize] = True
        free_para[index_free_k + 1:index_free_k + 1 + len(self.readers) * kSize:kSize] = True
        free_para_single = np.ones(nAllVars_single, dtype=bool)
        free_para_all = np.concatenate([free_para, free_para_single], 0)
        nAllVars_all = nAllVars + nAllVars_single

        args_optimization = {
            'x0': x0,
            'x0_single': x0_single,
            'x0_all': x0_all,
            'x0_all_free': x0_all[free_para_all],  # define correct initialization vector for x0_all
            'free_para': free_para,  # define free parameters
            'free_para_single': free_para_single,  # define free parameters (single calibration)
            'nAllVars_all': nAllVars_all,  # stack multi- and single-calibration
            'free_para_all': free_para_all
        }

        args = {**args, **args_optimization}

        # BOUNDARIES
        b_all = np.array([[-np.inf, np.inf]] * np.size(x0_all)).T
        bounds = {
            'all': b_all,
            'all_free': b_all[:, free_para_all]
        }

        return args, bounds

    def get_fitted_paras(self, args):
        self.x_all_fit = np.copy(args['x0_all'])
        self.x_all_fit[args['free_para_all']] = self.min_result.x
        self.x_single_fit = self.x_all_fit[args['nAllVars']:]
        self.x_fit = self.x_all_fit[:args['nAllVars']]

        self.rX1_fit, self.tX1_fit, self.k_fit, self.A_fit, self.r1_fit, self.t1_fit = func.calc_paras_from_x(
            self.x_fit, args)
        self.RX1_fit = func.map_r2R(self.rX1_fit)
        self.R1_fit = func.map_r2R(self.r1_fit)

        self.r1_single_fit, self.t1_single_fit = func.calc_paras_from_x_single2(self.x_single_fit, args)
        # do this since 1-dimensional arrays loose a dimension, e.g. shape (1, 3) --> (3)
        for i_cam in range(len(self.readers)):
            n_used_frames = np.sum(self.mask_single[i_cam])
            if n_used_frames == 1:
                self.r1_single_fit[i_cam][0] = self.r1_single_fit[i_cam][0][None, :]
                self.t1_single_fit[i_cam][0] = self.t1_single_fit[i_cam][0][None, :]
        self.R1_single_fit = []
        for i_cam in range(len(self.readers)):
            self.R1_single_fit.append([])
            for i_pose in range(len(self.r1_single_fit[i_cam])):
                self.R1_single_fit[i_cam].append(func.map_r2R(self.r1_single_fit[i_cam][i_pose]))
        return

    def start_optimization(self, args, bounds):
        print('Starting optimization procedure - This might take a while')
        print('The following lines are associated with the current state of the optimization procedure:')
        start_time = time.time()

        self.min_result = least_squares(func.obj_fcn_free,
                                        args['x0_all_free'],
                                        jac=func.obj_fcn_jac_free,
                                        bounds=bounds['all_free'],
                                        method='trf',
                                        ftol=self.tol,
                                        xtol=self.tol,
                                        gtol=self.tol,
                                        x_scale='jac',
                                        loss='linear',
                                        tr_solver='exact',
                                        max_nfev=np.inf,
                                        verbose=2,
                                        args=[args])
        current_time = time.time()
        print('Optimization algorithm converged:\t{:s}'.format(str(self.min_result.success)))
        print('Time needed:\t\t\t\t{:.0f} seconds'.format(current_time - start_time))
        self.info['optimization'] = {'message': self.min_result.message, 'success': self.min_result.success}
        return

    def save_multicalibration(self):
        self.result = dict()
        # general
        self.result['recFileNames'] = self.recFileNames
        self.result['headers'] = [self.get_header(reader) for reader in self.readers]
        self.result['nCameras'] = len(self.readers)
        self.result['nFrames'] = self.nFrames
        self.result['mask_multi'] = self.mask_multi
        self.result['indexRefCam'] = self.indexRefCam
        self.result['calib'] = self.calib_multi
        self.result['mask_single'] = self.mask_single
        self.result['calib_single'] = self.calib_single
        self.result['mask_all'] = self.allFramesMask
        # optimization input
        self.result['tolerance'] = self.tol
        # optimization variables (output)
        self.result['x_all_fit'] = self.x_all_fit
        self.result['rX1_fit'] = self.rX1_fit
        self.result['RX1_fit'] = self.RX1_fit
        self.result['tX1_fit'] = self.tX1_fit
        self.result['k_fit'] = self.k_fit
        self.result['A_fit'] = self.A_fit
        self.result['r1_fit'] = self.r1_fit
        self.result['R1_fit'] = self.R1_fit
        self.result['t1_fit'] = self.t1_fit
        self.result['r1_single_fit'] = self.r1_single_fit
        self.result['R1_single_fit'] = self.R1_single_fit
        self.result['t1_single_fit'] = np.asarray(self.t1_single_fit)
        # Historically, scale_factor=square_size_real, and not part of calibration.
        # New: square_size_real factored into spatial units

        self.result['board'] = self.board_params
        self.result['info'] = self.info

        # Deprecated fields
        self.result['square_size_real'] = self.board_params['square_size']
        self.result['marker_size_real'] = self.result['square_size_real']*self.board_params['marker_size']
        self.result['boardWidth'] = self.board_params['boardWidth']
        self.result['boardHeight'] = self.board_params['boardHeight']
        self.result['scale_factor'] = 1

        pprint(self.result)
        # save
        self.resultPath = self.dataPath + '/multicalibration.npy'
        np.save(self.resultPath, self.result)
        print('Saved multi camera calibration to file {:s}'.format(self.resultPath))
        return

    @staticmethod
    def set_detector_parameters():
        detector_parameters = cv2.aruco.DetectorParameters_create()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_parameters.cornerRefinementWinSize = 5  # default value
        detector_parameters.cornerRefinementMaxIterations = 30  # default value
        detector_parameters.cornerRefinementMinAccuracy = 0.1  # default value

        return detector_parameters

    # plot detection result only works if sensor sizes of all cameras are identical TODO finish refactoring into separate method
    def plot_detection(self):
        pass
        # if verbose:
        #     img_dummy = np.zeros_like(frame)
        #     fig = plt.figure(1,
        #                      figsize=(8, 8))
        #     nRowsCols = np.int64(np.ceil(np.sqrt(len(self.readers))))
        #     ax_list = []
        #     im_list = []
        #     for i_cam in range(len(self.readers)):
        #         ax = fig.add_subplot(nRowsCols, nRowsCols, i_cam + 1)
        #         ax.set_axis_off()
        #         ax_list.append(ax)
        #         im = ax_list[i_cam].imshow(img_dummy,
        #                                    aspect=1,
        #                                    cmap='gray',
        #                                    vmin=0,
        #                                    vmax=255)
        #         im_list.append(im)
        #     fig.tight_layout()
        #     fig.canvas.draw()
        #     plt.pause(1e-16)
        #     for i_frame in np.arange(0, self.nFrames, 1, dtype=np.int64):
        #         for i_cam in range(len(self.readers)):
        #             frame = reader.get_data(i_frame)
        #             if len(frame.shape) > 2:
        #                 frame = frame[:, :, 1]
        #             ax_list[i_cam].lines = []
        #             ax_list[i_cam].set_title('cam: {:01d}, frame: {:06d}'.format(i_cam, i_frame))
        #             im_list[i_cam].set_data(frame)
        #             if self.allFramesMask[i_cam, i_frame]:
        #                 corners_plot = np.array(self.allCorners_list[i_cam][i_frame])
        #                 ax_list[i_cam].plot(corners_plot[:, 0, 0],
        #                                     corners_plot[:, 0, 1],
        #                                     linestyle='',
        #                                     marker='x',
        #                                     color='red')
        #         fig.canvas.draw()
        #         plt.pause(5e-1)
        #     plt.close(1)

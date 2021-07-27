#!/usr/bin/env python3

import cv2
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import sys

import ccv

import multical_func_ccv as func
import multical_plot_ccv

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QAbstractItemView, \
                            QApplication, \
                            QComboBox, \
                            QFrame, \
                            QFileDialog, \
                            QGridLayout, \
                            QLabel, \
                            QLineEdit, \
                            QListWidget, \
                            QMainWindow, \
                            QMessageBox, \
                            QPushButton

from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

import time

from autograd import elementwise_grad
from scipy.optimize import least_squares



class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setGeometry(0, 0, 320, 96)

        # LAYOUT
        self.frame_main = QFrame()
        self.setStyleSheet("background-color: gray;")
        self.layoutGrid = QGridLayout()
        self.layoutGrid.setSpacing(10)
        self.frame_main.setMinimumSize(320, 96)
        self.frame_main.setLayout(self.layoutGrid)
        self.setCentralWidget(self.frame_main)

        self.set_control_buttons()

        self.setFocus()
        self.setWindowTitle('Multi Camera Calibration Tool')
        self.show()

        self.startDirectory = ''
        return

    def set_control_buttons(self):
        self.button_performCalibration = QPushButton()
        self.button_performCalibration.setText('Perform calibration')
        self.button_performCalibration.clicked.connect(self.button_performCalibration_press)
        self.layoutGrid.addWidget(self.button_performCalibration, 0, 0)

        self.button_loadCalibration = QPushButton()
        self.button_loadCalibration.setText('Load calibration')
        self.button_loadCalibration.clicked.connect(self.button_loadCalibration_press)
        self.layoutGrid.addWidget(self.button_loadCalibration, 1, 0)
        return
    
    def button_performCalibration_press(self):
        self.button_performCalibration.setEnabled(False)
        self.button_loadCalibration.setEnabled(False)

        self.recFileNames = list()
        self.dataPath = ''
        self.nCameras = 0
        self.nFrames = 0
        self.headers = list()
        self.recordingIsLoaded = False
        # read the dataset
        self.read_recording()
        if (self.recordingIsLoaded):
            # perform complete multi camera calibration
            self.perform_multi_calibration()
            self.plot_calibration()
        return

    def button_loadCalibration_press(self):
        self.button_performCalibration.setEnabled(False)
        self.button_loadCalibration.setEnabled(False)

        self.calibrationIsLoaded = False
        # read the calibration
        self.read_calibration()
        if (self.calibrationIsLoaded):
            self.plot_calibration()
        return

    def plot_calibration(self):
        self.PlotWindow = multical_plot_ccv.PlotWindow(self.result, self.dataPath)
        self.PlotWindow.show()
        self.button_performCalibration.setEnabled(True)
        self.button_loadCalibration.setEnabled(True)
        return

    def read_calibration(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        self.calFileName, _ = QFileDialog.getOpenFileNames(dialog,
                                                           "Choose calibration file",
                                                           self.startDirectory,
                                                           "npy files (*.npy)",
                                                           options=dialogOptions)
        if (len(self.calFileName) == 1):
            # check if input file is a npy-file:
            filesAreCorrect = True
            fileEnding = self.calFileName[0].split('/')[-1].split('.')[-1]
            if (fileEnding != 'npy'):
                filesAreCorrect = False
            if not(filesAreCorrect):
                print('WARNING: Input file is not correct (no npy-file)')
                self.button_performCalibration.setEnabled(True)
                self.button_loadCalibration.setEnabled(True)
            else:
                print('LOAD CALIBRATION')
                # if everything is fine keep on going with the plotting
                self.dataPath = '/'.join(self.calFileName[0].split('/')[:-1])
                self.result = np.load(self.calFileName[0], allow_pickle=True).item()
                self.calibrationIsLoaded = True
        else:
            print('WARNING: Provide exactly one input file')
            self.button_performCalibration.setEnabled(True)
            self.button_loadCalibration.setEnabled(True)
        return

    def read_recording(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        recFileNames_unsorted, _ = QFileDialog.getOpenFileNames(dialog,
                                                                "Choose files to calibrate",
                                                                self.startDirectory,
                                                                "ccv files (*.ccv)",
                                                                options=dialogOptions)
        if (len(recFileNames_unsorted) > 1):
            self.recFileNames = sorted(recFileNames_unsorted)
            # check if input files are ccv-files:
            filesAreCorrect = True
            for i_file in self.recFileNames:
                fileEnding = i_file.split('/')[-1].split('.')[-1]
                if (fileEnding != 'ccv'):
                    filesAreCorrect = False
            if not(filesAreCorrect):
                print('WARNING: At least on input file is not correct (no ccv-file)')
                self.button_performCalibration.setEnabled(True)
                self.button_loadCalibration.setEnabled(True)
            else:
                self.dataPath = '/'.join(self.recFileNames[0].split('/')[:-1])
                self.nCameras = int(np.size(self.recFileNames, 0))
                self.nFrames = np.zeros(self.nCameras, dtype=np.int64)
                print('LOAD RECORDING')
                for i_cam in range(self.nCameras):
                    print('Loading recording {:s}\t(camera {:02d})'.format(self.recFileNames[i_cam], i_cam))
                    header = ccv.get_header(self.recFileNames[i_cam])
                    self.nFrames[i_cam] = header['nframes']
                    self.headers.append(header)
                # check if frame number is consistent:
                if (np.all(self.nFrames[1:] == self.nFrames[:-1]) == True):
                    self.nFrames = self.nFrames[0]
                    # if everything is fine keep on going with the calibration
                    self.recordingIsLoaded = True
                else:
                    print('WARNING: Number of frames is not identical for all cameras')
                    print('Number of detected frames per camera:')
                    for i_cam in range(self.nCameras):
                        print('\tcamera {:03d}:\t{:04d}'.format(i_cam, self.nFrames[i_cam]))
                    print('Do you want the software to continue and ignore the last recorded frames in order to fix this issue?')
                    user_input = np.int64(0)
                    while not((user_input == np.int64(1024)) or
                              (user_input == np.int64(4194304))):
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Information)
                        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                        msg.setText('WARNING: Number of frames is not identical for all cameras')
                        msg.setInformativeText('Do you want the software to continue and leave out the last recorded frame in order to fix this issue?')
                        user_input = msg.exec_()
                        user_input = np.int64(user_input)
                    if (user_input == np.int64(1024)):
                        self.nFrames = np.int64(np.min(self.nFrames))
                        self.recordingIsLoaded = True
                    elif (user_input == np.int64(4194304)):
                        self.button_performCalibration.setEnabled(True)
                        self.button_loadCalibration.setEnabled(True)
        else:
            print('WARNING: Provide at least two input files')
            self.button_performCalibration.setEnabled(True)
            self.button_loadCalibration.setEnabled(True)
        return

    def perform_multi_calibration(self):
        # generate board
        self.generate_board()
        # detect corners
        self.allFramesMask = np.zeros((self.nCameras, self.nFrames),
                                      dtype=bool)
        self.allCorners_list = list()
        self.allIds_list = list()
        self.detect_corners()
        # split into two frame sets
        # first set contains frames for single calibration
        # second set contains frames for multi calibration
        self.mask_single = np.zeros((self.nCameras, self.nFrames),
                                    dtype=bool)
        self.mask_multi = np.zeros(self.nFrames, dtype=bool)
        self.indexRefCam = np.nan
        self.split_frame_sets()
        # flags and criteria for cv2.aruco.calibrateCameraCharuco
        self.flags = (cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3)
        self.criteria = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS,
                         30,
                         float(np.finfo(np.float32).eps))
        # perform single calibration
        self.cal_single_list = list()
        self.perform_single_calibration()
        # generate calib_single
        self.calib_single = dict()
        self.nPoses_single = np.zeros(self.nCameras, dtype=np.int64)
        self.generate_calib_single()
        # perform multi calibration
        self.cal_multi_list = list()
        self.perform_single_calibration_for_multi()
        # generate calib_multi
        self.calib_multi = dict()
        self.nPoses = int(0)
        self.generate_calib_mutli()

        # the following functions are based on multical_main.py
        print('PREPARE FOR MULTI CAMERA CALIBRATION')
        self.generate_args()
        print('START MULTI CAMERA CALIBRATION')
        self.start_optimization()
        self.get_fitted_paras()
        print('SAVE MULTI CAMERA CALIBRATION')
        self.save_multicalibration()
        self.save_multicalibration_to_text()
        print('FINISHED MULTI CAMERA CALIBRATION')
        return

    def generate_board(self):
        self.boardWidth = 4
        self.boardHeight = 6
        self.square_size = 1 # scalar
        self.marker_size = 0.6 # scalar
        self.square_size_real = 6.5 # cm
        self.marker_size_real = self.square_size_real * self.marker_size
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.nFeatures = (self.boardWidth - 1) * (self.boardHeight - 1)
        self.minDetectFeat = int(max(self.boardWidth, self.boardHeight))
        self.board = cv2.aruco.CharucoBoard_create(self.boardWidth,
                                                   self.boardHeight,
                                                   self.square_size,
                                                   self.marker_size,
                                                   self.dictionary)
        return

    def detect_corners(self, verbose=False):
        print('DETECTING FEATURES')
        detector_parameters = cv2.aruco.DetectorParameters_create()
        detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_parameters.cornerRefinementWinSize = 5 # default value
        detector_parameters.cornerRefinementMaxIterations = 30 # default value
        detector_parameters.cornerRefinementMinAccuracy = 0.1 # default value
        for i_cam in range(self.nCameras):
            print('Detecting features in camera {:02d}'.format(i_cam))
            allCorners = list()
            allIds = list()
            previousUsedFrame = np.nan
            previousCorners = np.zeros((self.nFeatures, 2), dtype=np.float64)
            currentCorners = np.zeros((self.nFeatures, 2), dtype=np.float64)
            # calculate offset
            offset_x = self.headers[i_cam]['offset'][0]
            offset_y = self.headers[i_cam]['offset'][1]
            for i_frame in np.arange(0, self.nFrames, 1, dtype=np.int64):
                maskValue2add = False
                corners2add = list()
                ids2add = list()
                # feature detection
                frame = ccv.get_frame(self.recFileNames[i_cam], i_frame + 1)
                res = cv2.aruco.detectMarkers(frame,
                                              self.dictionary,
                                              parameters=detector_parameters)

                if (len(res[0]) > 0):
                    res_ref = cv2.aruco.refineDetectedMarkers(frame,
                                                              self.board,
                                                              res[0],
                                                              res[1],
                                                              res[2],
                                                              minRepDistance=10.0,
                                                              errorCorrectionRate=3.0,
                                                              checkAllOrders=True,
                                                              parameters=detector_parameters)
                    res2 = cv2.aruco.interpolateCornersCharuco(res_ref[0],
                                                               res_ref[1],
                                                               frame,
                                                               self.board,
                                                               minMarkers=2)
                    
                    # checks if the requested minimum number of features are detected
                    if (res2[0] >= self.minDetectFeat):
                        # add offset
                        res2[1][:, :, 0] = res2[1][:, :, 0] + offset_x
                        res2[1][:, :, 1] = res2[1][:, :, 1] + offset_y
                        maskValue2add = True
                        corners2add = np.copy(res2[1])
                        ids2add = np.copy(res2[2])
                        # checks if consecutive frames are too similar
                        if not(np.isnan(previousUsedFrame)):
                            # get current and previous features
                            previousCorners[:, :] = 0
                            previousCorners[allIds[previousUsedFrame].ravel()] = allCorners[previousUsedFrame].squeeze()
                            currentCorners[:, :] = 0
                            currentCorners[res2[2].ravel()] = res2[1].squeeze()
                            # calculates euclidian distance between features
                            diff = currentCorners - previousCorners
                            ids_use = np.intersect1d(allIds[previousUsedFrame].ravel(),
                                                     res2[2].ravel())
                            diff = diff[ids_use]
                            dist = np.sqrt(np.sum(diff**2, 1))

                            # use frame when all ids are different
                            if (np.size(dist) == 0):
                                dist_max = np.inf
                            else:
                                dist_max = np.max(dist)
                            # check if maximum distance is high enough
                            if not(dist_max > 0.0):
                                maskValue2add = False
                                corners2add = list()
                                ids2add = list()
                if (maskValue2add):
                    previousUsedFrame = np.copy(i_frame)
                self.allFramesMask[i_cam, i_frame] = maskValue2add
                allCorners.append(corners2add)
                allIds.append(ids2add)
            self.allCorners_list.append(allCorners)
            self.allIds_list.append(allIds)
            print('Detected features in {:04d} frames in camera {:02d}'.format(np.sum(self.allFramesMask[i_cam]), i_cam))

        # only works if sensor sizes of all cameras are identical
        if (verbose):
            img_dummy = np.zeros_like(frame)
            fig = plt.figure(1,
                             figsize=(8, 8))
            nRowsCols = np.int64(np.ceil(np.sqrt(self.nCameras)))
            ax_list = list()
            im_list = list()
            for i_cam in range(self.nCameras):
                ax = fig.add_subplot(nRowsCols, nRowsCols, i_cam + 1)
                ax.set_axis_off()
                ax_list.append(ax)
                im = ax_list[i_cam].imshow(img_dummy,
                                           aspect=1,
                                           cmap='gray',
                                           vmin=0,
                                           vmax=255)
                im_list.append(im)
            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(1e-16)
            for i_frame in np.arange(0, self.nFrames, 1, dtype=np.int64):
                for i_cam in range(self.nCameras):
                    frame = ccv.get_frame(self.recFileNames[i_cam], i_frame + 1)
                    ax_list[i_cam].lines = list()
                    ax_list[i_cam].set_title('cam: {:01d}, frame: {:06d}'.format(i_cam, i_frame))
                    im_list[i_cam].set_data(frame)
                    if (self.allFramesMask[i_cam, i_frame]):
                        corners_plot = np.array(self.allCorners_list[i_cam][i_frame])
                        ax_list[i_cam].plot(corners_plot[:, 0, 0],
                                            corners_plot[:, 0, 1],
                                            linestyle='',
                                            marker='x',
                                            color='red')
                fig.canvas.draw()
                plt.pause(5e-1)
            plt.close(1)
            raise
        return

    def split_frame_sets(self):
        for i_frame in range(self.nFrames):
            nFrames_used = np.sum(self.allFramesMask[:, i_frame])
            # if corners are detected in more than one camera use the frame for multi calibration
            if (nFrames_used > 1):
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

    def perform_single_calibration(self):
        print('PERFORM SINGLE CAMERA CALIBRATION #1')
        for i_cam in range(self.nCameras):
            nUsedFrames = np.sum(self.mask_single[i_cam])
            print('Using {:03d} frames to perform single camera calibration for camera {:02d}'.format(nUsedFrames, i_cam))
            if (nUsedFrames > 0): # do this to not run into indexing issues
                corners_use = list(compress(self.allCorners_list[i_cam],
                                            self.mask_single[i_cam]))
                ids_use = list(compress(self.allIds_list[i_cam],
                                        self.mask_single[i_cam]))
                cal = cv2.aruco.calibrateCameraCharuco(corners_use,
                                                       ids_use,
                                                       self.board,
                                                       self.headers[i_cam]['sensorsize'],
                                                       None,
                                                       None,
                                                       flags=self.flags,
                                                       criteria=self.criteria)
                self.cal_single_list.append(cal)
                print('Completed single camera calibration')
                print('Reprojection error:\t{:.08f}'.format(cal[0]))
            else:
                self.cal_single_list.append(list()) 
        return

    def generate_calib_single(self):
        self.calib_single = dict()
        self.nPoses_single = np.sum(self.mask_single, 1)
        for i_cam in range(0, self.nCameras, 1):
            key = 'cam{:01d}'.format(i_cam)
            self.calib_single[key] = dict()
            self.calib_single[key]['charuco_ids'] = list()
            self.calib_single[key]['charuco_corners'] = list()
            self.calib_single[key]['rotation_vectors'] = list()
            self.calib_single[key]['translation_vectors'] = list()
            index = 0
            for i_frame in range(self.nFrames):
                if (self.mask_single[i_cam, i_frame]):
                    nFeats = np.size(self.allIds_list[i_cam][i_frame])
                    ids_use = np.array(self.allIds_list[i_cam][i_frame],
                                       dtype=np.int64).reshape(nFeats, 1)
                    self.calib_single[key]['charuco_ids'].append(ids_use)
                    corners_use = np.array(self.allCorners_list[i_cam][i_frame],
                                           dtype=np.float64).reshape(nFeats ,2)
                    self.calib_single[key]['charuco_corners'].append(corners_use)
                    # r and t
                    rotations_use = np.array(self.cal_single_list[i_cam][3][index],
                                             dtype=np.float64).reshape(3, 1)
                    self.calib_single[key]['rotation_vectors'].append(rotations_use)
                    translations_use = np.array(self.cal_single_list[i_cam][4][index],
                                                dtype=np.float64).reshape(3, 1)
                    self.calib_single[key]['translation_vectors'].append(translations_use)
                    index = index + 1
        return

    def perform_single_calibration_for_multi(self):
        print('PERFORM SINGLE CAMERA CALIBRATION #2')
        print('The following single camera calibrations will be used to initialize the multi camera calibration')
        for i_cam in range(self.nCameras):
            mask = self.mask_multi & self.allFramesMask[i_cam]
            nUsedFrames = np.sum(mask)
            print('Using {:03d} frames to perform single camera calibration for camera {:02d}'.format(nUsedFrames, i_cam))
            if (nUsedFrames > 0):
                corners_use = list(compress(self.allCorners_list[i_cam],
                                            mask))
                ids_use = list(compress(self.allIds_list[i_cam],
                                        mask))
                cal = cv2.aruco.calibrateCameraCharuco(corners_use,
                                                       ids_use,
                                                       self.board,
                                                       self.headers[i_cam]['sensorsize'],
                                                       None,
                                                       None,
                                                       flags=self.flags,
                                                       criteria=self.criteria)
                self.cal_multi_list.append(cal)
                print('Completed single camera calibration for camera {:02d}'.format(i_cam))
                print('Reprojection error:\t{:.08f}'.format(cal[0]))
        return

    def generate_calib_mutli(self):
        self.calib_multi = dict()
        self.nPoses = int(np.sum(self.mask_multi))
        A = np.zeros((self.nCameras, 3, 3), dtype=np.float64)
        k = np.zeros((self.nCameras, 5), dtype=np.float64)
        for i_cam in range(self.nCameras):
            key = 'cam{:01d}'.format(i_cam)
            self.calib_multi[key] = dict()
            A[i_cam] = np.array(self.cal_multi_list[i_cam][1],
                                        dtype=np.float64).reshape(3, 3)
            k[i_cam] = np.array(self.cal_multi_list[i_cam][2],
                                dtype=np.float64).reshape(1, 5)
            nUsedFrames = np.sum(self.mask_single[i_cam])
            if (nUsedFrames > 0):
                if (self.cal_single_list[i_cam][0] < self.cal_multi_list[i_cam][0]):
                    A[i_cam] = np.array(self.cal_single_list[i_cam][1],
                                        dtype=np.float64).reshape(3, 3)
                    k[i_cam] = np.array(self.cal_single_list[i_cam][2],
                                        dtype=np.float64).reshape(1, 5)
            #
            self.calib_multi[key]['camera_matrix'] = A[i_cam]
            self.calib_multi[key]['dist_coeffs'] = k[i_cam]
            # rest
            self.calib_multi[key]['charuco_ids'] = list()
            self.calib_multi[key]['charuco_corners'] = list()
            self.calib_multi[key]['rotation_vectors'] = list()
            self.calib_multi[key]['translation_vectors'] = list()
            index = np.int64(0)
            for i_frame in range(self.nFrames):
                if (self.mask_multi[i_frame]):
                    nFeats = np.size(self.allIds_list[i_cam][i_frame])
                    ids_use = np.array(self.allIds_list[i_cam][i_frame],
                                       dtype=np.int64).reshape(nFeats, 1)
                    self.calib_multi[key]['charuco_ids'].append(ids_use)
                    corners_use = np.array(self.allCorners_list[i_cam][i_frame],
                                           dtype=np.float64).reshape(nFeats ,2)
                    self.calib_multi[key]['charuco_corners'].append(corners_use)
                    # r and t
                    # only append the list here when the camera has actually seen the pattern in the respective frame
                    # i.e. we have an estimate for r and t
                    if (self.allFramesMask[i_cam, i_frame]):
                        rotations_use = np.array(self.cal_multi_list[i_cam][3][index],
                                                 dtype=np.float64).reshape(3, 1)
                        self.calib_multi[key]['rotation_vectors'].append(rotations_use)
                        translations_use = np.array(self.cal_multi_list[i_cam][4][index],
                                                    dtype=np.float64).reshape(3, 1)
                        self.calib_multi[key]['translation_vectors'].append(translations_use)
                        index = index + 1
        return

    def generate_args(self):
        print('Defining arguments for multi camera calibration')
        self.args = dict()

        # GENERAL
        print('\t - Defining general variables')
        # total number of cameras
        self.args['nCameras'] = self.nCameras
        # total number of poses
        self.args['nPoses'] = self.nPoses
        # total number of poses (single calibration)
        self.args['nPoses_single'] = self.nPoses_single
        # width of charuco board
        self.args['boardWidth'] = self.boardWidth
        # height of charuco board
        self.args['boardHeight'] = self.boardHeight
        # total number of features on the charuco board
        self.args['nFeatures'] = self.nFeatures
        # total number of residuals for each direction (x, y)
        self.nRes = self.nFeatures * self.nPoses * self.nCameras
        self.args['nRes'] = self.nRes
        # total number of residuals for each direction (x, y) (single calibration)
        self.nRes_single = np.sum(self.nFeatures * self.nPoses_single)
        self.args['nRes_single'] = self.nRes_single
        # number of free distortion coefficients per residual
        self.kSize = 5
        self.args['kSize'] = self.kSize
        # number of free variables in camera matrix per residual
        self.ASize = 4
        self.args['ASize'] = self.ASize
        # number of free rotation parameters per residual
        self.rSize = 3
        self.args['rSize'] = self.rSize
        # number of free translation parameters per residual
        self.tSize = 3
        self.args['tSize'] = self.tSize
        # number of free parameters per residual
        self.nVars = self.rSize + self.tSize + \
                     self.kSize + \
                     self.ASize + \
                     self.rSize + self.tSize
        self.args['nVars'] = self.nVars
        # total number of free parameters
        self.nAllVars = (self.nCameras - 1) * (self.rSize + self.tSize) + \
                        self.nCameras * self.kSize + \
                        self.nCameras * self.ASize + \
                        self.nPoses * (self.rSize + self.tSize)
        self.args['nAllVars'] = self.nAllVars
        # total number of free parameters (single calibration)
        self.nAllVars_single = np.sum((self.rSize + self.tSize) * self.nPoses_single)
        self.args['nAllVars_single'] = self.nAllVars_single
        # number of features which have to be detected (per pose) in order to include it for the calibration
        self.args['minDetectFeat'] = self.minDetectFeat
        # index of the reference camera
        self.args['indexRefCam'] = self.indexRefCam
        # boolean value which determines whether to use sparse jacobian
        self.use_sparse_jac = False # should always be False
        self.args['use_sparse_jac'] = self.use_sparse_jac

        # CONSTANTS
        print('\t - Defining constants')
        self.M, self.m, self.delta = func.map_calib2consts(self.calib_multi, self.args)
        self.args['M'] = self.M
        self.args['m'] = self.m
        self.args['delta'] = self.delta
        self.M_single, self.m_single, self.delta_single = func.map_calib2consts_single(self.calib_single, self.args)
        self.args['M_single'] = self.M_single
        self.args['m_single'] = self.m_single
        self.args['delta_single'] = self.delta_single

        # JACOBIAN
        print('\t - Defining jacobian')
        self.args['jac_x'] = np.zeros(self.nVars, dtype=object)
        self.args['jac_y'] = np.zeros(self.nVars, dtype=object)
        for i_var in range(self.nVars):
            self.args['jac_x'][i_var] = elementwise_grad(func.calc_res_x, i_var)
            self.args['jac_y'][i_var] = elementwise_grad(func.calc_res_y, i_var)

        # OPTIMIZATION
        print('\t - Defining optimization input variables')
        # definie inital x
        self.x0 = func.set_x0_objFunc(self.calib_multi, self.args)
        self.x0_single = func.set_x0_objFunc_single(self.calib_single, self.args)
        # define free parameters
        self.free_para = np.ones(self.nAllVars, dtype=bool)
        self.index_free_k = (self.nCameras - 1) * (self.rSize + self.tSize)
        self.free_para[self.index_free_k:self.index_free_k + self.nCameras * self.kSize] = False
        self.free_para[self.index_free_k:self.index_free_k + self.nCameras * self.kSize:self.kSize] = True
        self.free_para[self.index_free_k + 1:self.index_free_k + 1 + self.nCameras * self.kSize:self.kSize] = True
        self.args['free_para'] = self.free_para
        # define free parameters (single calibration)
        self.free_para_single = np.ones(self.nAllVars_single, dtype=bool)
        self.args['free_para_single'] = self.free_para_single
        # stack multi- and single-calibration
        self.nAllVars_all = self.nAllVars + self.nAllVars_single
        self.args['nAllVars_all'] = self.nAllVars_all
        self.x0_all = np.concatenate([self.x0, self.x0_single], 0)
        self.args['x0_all'] = self.x0_all
        self.free_para_all = np.concatenate([self.free_para, self.free_para_single], 0)
        self.args['free_para_all'] = self.free_para_all
        # define correct initialization vector for x0_all
        self.x0_all_free = self.x0_all[self.free_para_all]
        # define boundaries
        self.bounds_all = np.array([[-np.inf, np.inf]] * np.size(self.x0_all)).T
        self.bounds_all_free = self.bounds_all[:, self.free_para_all]
        return

    def get_initial_paras(self):
        self.x0 = func.set_x0_objFunc(self.calib_multi, self.args)
        self.rX1, self.tX1, self.k, self.A, self.r1, self.t1 = func.calc_paras_from_x(self.x0, self.args)
        self.RX1 = func.map_r2R(self.rX1)
        self.R1 = func.map_r2R(self.r1)
        self.r1_single, self.t1_single = func.calc_paras_from_x_single(self.x0_single, self.args)
        return

    def get_fitted_paras(self):
        self.x_all_fit = np.copy(self.x0_all)
        self.x_all_fit[self.free_para_all] = self.min_result.x
        self.x_fit = self.x_all_fit[:self.nAllVars]
        self.rX1_fit, self.tX1_fit, self.k_fit, self.A_fit, self.r1_fit, self.t1_fit = func.calc_paras_from_x(self.x_fit, self.args)
        self.RX1_fit = func.map_r2R(self.rX1_fit)
        self.R1_fit = func.map_r2R(self.r1_fit)
        self.x_single_fit = self.x_all_fit[self.nAllVars:]
        self.r1_single_fit, self.t1_single_fit = func.calc_paras_from_x_single2(self.x_single_fit, self.args)
        # do this since 1-dimensional arrays loose a dimension, e.g. shape (1, 3) --> (3)
        for i_cam in range(self.nCameras): 
            nUsedFrames = np.sum(self.mask_single[i_cam])
            if (nUsedFrames == 1):
                self.r1_single_fit[i_cam][0] = self.r1_single_fit[i_cam][0][None, :]
                self.t1_single_fit[i_cam][0] = self.t1_single_fit[i_cam][0][None, :]        
        self.R1_single_fit = list()
        for i_cam in range(self.nCameras):
            self.R1_single_fit.append(list())
            for i_pose in range(len(self.r1_single_fit[i_cam])):
                self.R1_single_fit[i_cam].append(func.map_r2R(self.r1_single_fit[i_cam][i_pose]))
        return

    def start_optimization(self):
        print('Starting optimization procedure - This might take a while')
        print('The following lines are associated with the current state of the optimization procedure:')
        start_time = time.time()
        # ATTENTION: use_sparse_jac is not implemented
        self.tol = np.finfo(np.float64).eps # machine epsilon
        if (self.use_sparse_jac):
            self.min_result = least_squares(func.obj_fcn_free,
                                            self.x0_all_free,
                                            jac=func.obj_fcn_jac_free,
                                            bounds=self.bounds_all_free,
                                            method='trf',
                                            ftol=self.tol,
                                            xtol=self.tol,
                                            gtol=self.tol,
                                            x_scale='jac',
                                            loss='soft_l1',
                                            tr_solver='exact',
                                            max_nfev=np.inf,
                                            verbose=2,
                                            args=[self.args])
        else:
            self.min_result = least_squares(func.obj_fcn_free,
                                            self.x0_all_free,
                                            jac=func.obj_fcn_jac_free,
                                            bounds=self.bounds_all_free,
                                            method='trf',
                                            ftol=self.tol,
                                            xtol=self.tol,
                                            gtol=self.tol,
                                            x_scale='jac',
                                            loss='linear',
                                            tr_solver='exact',
                                            max_nfev=np.inf,
                                            verbose=2,
                                            args=[self.args])
        current_time = time.time()
        print('Optimization algorithm converged:\t{:s}'.format(str(self.min_result.success)))
        print('Time needed:\t\t\t\t{:.0f} seconds'.format(current_time - start_time))
        self.message = self.min_result.message
        self.success = self.min_result.success
        return

    def save_multicalibration(self):
        self.result = dict()
        # general
        self.result['recFileNames'] = self.recFileNames
        self.result['headers'] = self.headers
        self.result['nCameras'] = self.nCameras
        self.result['nFrames'] = self.nFrames
        self.result['boardWidth'] = self.boardWidth
        self.result['boardHeight'] = self.boardHeight
        self.result['mask_multi'] = self.mask_multi
        self.result['indexRefCam'] = self.indexRefCam
        self.result['calib'] = self.calib_multi
        self.result['mask_single'] = self.mask_single
        self.result['calib_single'] = self.calib_single
        self.result['mask_all'] = self.allFramesMask
        # optimization input
        self.result['x0_all'] = self.x0_all
        self.result['free_para_all'] = self.free_para_all
        self.result['tolerance'] = self.tol
        # optimization output
        self.result['message'] = self.message
        self.result['convergence'] = self.success
        # optimization variables (output)
        self.result['x_all_fit'] = self.x_all_fit,
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
        self.result['t1_single_fit'] = self.t1_single_fit
        # save
        self.resultPath = self.dataPath + '/multicalibration.npy'
        np.save(self.resultPath, self.result)
        print('Saved multi camera calibration to file {:s}'.format(self.resultPath))
        return

    def save_multicalibration_to_text(self):
        self.resultPath_text = self.dataPath + '/multicalibration_matlab_mcl_gen.m'

        f = open(self.resultPath_text, 'w')

        f.write('% INFO:\n')
        f.write('% line break: \ n \n')
        f.write('% useful webpage: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html\n')
        f.write('\n\n\n')

        i = 'nCameras'
        nCams = self.result[i]
        f.write('% number of cameras:\n')
        f.write('nCams = {:01d}\n'.format(nCams))
        f.write('\n\n')

        i = 'recFileNames'
        f.write('% full path of used files (order equal to camera indexing):\n')
        f.write('f = { ...\n')
        for j in self.result[i]:
            f.write('\'' + str(j) + '\',...\n')
        f.write('}\n')
        f.write('\n\n')

        i = 'indexRefCam'
        f.write('% index of reference camera (starts at 0):\n')
        f.write('index_refCam = {:01d}\n'.format(self.result[i]))
        f.write('\n\n')

        i = 'A_fit'
        data_use = self.result[i]
        f.write('% camera matrices [f_x, c_x, f_y, c_y]:\n')
        f.write('A = [ ...\n')
        for i_cam in range(nCams):
            for i_row in range(4):
                f.write(str(data_use[i_cam, i_row]) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')

        i = 'k_fit'
        data_use = self.result[i]
        f.write('% distortion coefficients [k_1, k_2, p_1, p_2, k_3]:\n')
        f.write('k = [ ...\n')
        for i_cam in range(nCams):
            for i_row in range(5):
                f.write(str(data_use[i_cam, i_row]) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')

        i = 'RX1_fit'
        data_use = self.result[i]
        f.write('% rotation matrices to convert into coordinate system of the respective camera:\n')
        f.write('R = cat(3, ...\n')
        for i_cam in range(nCams):
            f.write('[')
            for i_row in range(3):
                for i_col in range(3):
                    f.write(str(data_use[i_cam, i_row, i_col]) + ' ')
                if (i_row != 2):
                    f.write('; ...\n')
                else:
                    if (i_cam != nCams - 1):
                        f.write('], ...\n')
                    else:
                        f.write('])\n')
        f.write('\n\n')

        i = 'tX1_fit'
        data_use = self.result[i]
        f.write('% translation vectors to convert into coordinate system of the respective camera (units in squares):\n')
        f.write('t = [ ...\n')
        for i_cam in range(nCams):
            for i_row in range(3):
                f.write(str(data_use[i_cam, i_row]) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\'\n')
        f.write('\n\n')

        i = 'headers'
        data_use = self.result[i]
        
        f.write('% sensor size in pixel:\n')
        f.write('sensorSize = [ ...\n')
        for i_cam in range(nCams):
            for i_row in range(2):
                f.write(str(data_use[i_cam]['sensorsize'][i_row]) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')
        
        f.write('% offset in pixel:\n')
        f.write('offset = [ ...\n')
        for i_cam in range(nCams):
            for i_row in range(2):
                f.write(str(data_use[i_cam]['offset'][i_row]) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')
        
        # optional:
        f.write('% used width in pixel:\n')
        f.write('width = [ ...\n')
        for i_cam in range(nCams):
            f.write(str(data_use[i_cam]['w']) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')
       
        f.write('% used height in pixel:\n')
        f.write('height = [ ...\n')
        for i_cam in range(nCams):
            f.write(str(data_use[i_cam]['h']) + ' ')
            if (i_cam != nCams - 1):
                f.write('; ...\n')
            else:
                f.write(']\n')
        f.write('\n\n')

        f.write('% square size in cm:\n')
        f.write('square_size = {:.8f}\n'.format(self.square_size_real))
        f.write('\n\n')

        f.write('% marker size in cm:\n')
        f.write('marker_size = {:.8f}\n'.format(self.marker_size_real))
        f.write('\n\n')
        
        f.write('[mc ,mcfn] = cameralib.helper.openCVToMCL(R,t,A,k,sensorSize,square_size,bbohelper.filesystem.filename(f))')

        f.close()
        print('Saved multi camera calibration to file {:s}'.format(self.resultPath_text))
        return

def main():  
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
    return

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import sys

import imageio
from ccvtools import rawio

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
                            QPushButton, \
                            QTextEdit 
                            
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

from . import multical_func as func



class PlotWindow(QMainWindow):
    def __init__(self, calibrator,
                 parent=None):
        super(PlotWindow, self).__init__(parent=parent)
        self.setGeometry(0, 160, 768, 768)
        
        # get calibration result
        self.result = calibrator.result
        self.result_directory = calibrator.dataPath
        self.readers = calibrator.readers
        self.recFileNames = self.result['recFileNames']
        self.mask_multi = self.result['mask_multi']
        self.calib_multi = self.result['calib']
        self.mask_single = self.result['mask_single']
        self.calib_single = self.result['calib_single']
        
        self.nCameras = self.result['nCameras']
        self.nFrames = self.result['nFrames']
        self.boardWidth = self.result['boardWidth']
        self.boardHeight = self.result['boardHeight']
        self.nFeatures = (self.boardWidth - 1) * (self.boardHeight - 1)
        
        self.set_layout()
        
        self.i_cam = 0
        self.i_pose = 0

        self.control_ini()
        self.plot_ini()
        
    def set_layout(self):
        # LAYOUT
        self.frame_main = QFrame()
        self.setStyleSheet("background-color: black;")
        self.layoutGrid = QGridLayout()
        self.layoutGrid.setSpacing(10)
        self.frame_main.setMinimumSize(768, 768)
        self.frame_main.setLayout(self.layoutGrid)
                
        # frame for controls
        self.frame_controls = QFrame()
        self.frame_controls.setStyleSheet("background-color: gray;")
        self.layoutGrid.setRowStretch(0, 1)
        self.layoutGrid.addWidget(self.frame_controls, 0, 0)
        # control grid
        self.layoutGrid_control = QGridLayout()
        self.layoutGrid_control.setSpacing(10)
        self.frame_controls.setLayout(self.layoutGrid_control)
        
        # frame for plots
        self.frame_plot = QFrame()
        self.frame_plot.setStyleSheet("background-color: gray;")
        self.layoutGrid.setRowStretch(1, 3)
        self.layoutGrid.addWidget(self.frame_plot, 1, 0)
        # plot grid
        self.layoutGrid_plot = QGridLayout()
        self.layoutGrid_plot.setSpacing(0)
        self.frame_plot.setLayout(self.layoutGrid_plot)
        
        # add to grid
        self.frame_main.setLayout(self.layoutGrid)
        self.setCentralWidget(self.frame_main)
                
        self.setFocus()
        self.setWindowTitle('Projection Plots')
        self.show()
    
    def control_ini(self):
        self.label_i_cam = QLabel()
        self.label_i_cam.setText('camera:')
        self.layoutGrid_control.addWidget(self.label_i_cam, 0, 0)
        
        self.list_i_cam = QComboBox()
        self.list_i_cam.addItems([str(i) for i in range(self.nCameras)])
        self.list_i_cam.currentIndexChanged.connect(self.list_i_cam_change)
        self.layoutGrid_control.addWidget(self.list_i_cam, 0, 1, 1, 1)
        
        self.label_i_frame = QLabel()
        self.label_i_frame.setText('frame:')
        self.layoutGrid_control.addWidget(self.label_i_frame, 1, 0)
        
        self.field_i_pose = QLineEdit()
        self.field_i_pose.setValidator(QIntValidator())
        self.field_i_pose.setText(str(self.i_pose))
        self.field_i_pose.returnPressed.connect(self.field_i_pose_change)
        self.layoutGrid_control.addWidget(self.field_i_pose, 1, 1, 1, 1)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.layoutGrid_control.addWidget(self.info_text, 2, 0, 1, 2)
        
    def list_i_cam_change(self):
        self.i_cam = int(self.list_i_cam.currentText())
        self.plot_draw()
        self.list_i_cam.clearFocus()
        
    def field_i_pose_change(self):
        try: 
            int(self.field_i_pose.text())
            fieldInputIsCorrect = True
        except ValueError:
            fieldInputIsCorrect = False  
        if (fieldInputIsCorrect):
            self.i_pose = int(np.median([0,
                                         int(self.field_i_pose.text()),
                                         self.nFrames - 1]))
            self.plot_draw()
        self.field_i_pose.setText(str(self.i_pose))
        self.field_i_pose.clearFocus()
    
    def plot_ini(self):        
        self.fig = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setParent(self.frame_plot)
        self.ax = self.fig.add_subplot('111')
        
        self.plot_draw()
                
        self.layoutGrid_plot.addWidget(self.canvas, 0, 0)
                        
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.hide()
        
    def plot_draw(self):
        self.img = self.readers[self.i_cam].get_data(self.i_pose)
        if len(self.img.shape)>2:
            self.img = self.img[:,:,1]
        self.ax.clear()
        self.ax.imshow(self.img,
                       aspect=1,
                       cmap='gray',
                       vmin=0,
                       vmax=255)
        self.ax.set_xticklabels('')
        self.ax.set_yticklabels('')
        
        self.status = 'none'
        self.max_error = np.nan
        # calculate offset
        offset_x = self.result['headers'][self.i_cam]['offset'][0]
        offset_y = self.result['headers'][self.i_cam]['offset'][1]
        if (self.mask_multi[self.i_pose]):
            self.status = 'multi'
            index = np.sum(self.mask_multi[:self.i_pose])
            feature_ids = self.calib_multi['cam{:01d}'.format(self.i_cam)]['charuco_ids'][index].ravel()
            if (len(feature_ids) > 0):
                feature_positions = self.calib_multi['cam{:01d}'.format(self.i_cam)]['charuco_corners'][index]
                # substract offset
                feature_positions[:, 0] = feature_positions[:, 0] - offset_x
                feature_positions[:, 1] = feature_positions[:, 1] - offset_y
                self.ax.plot(feature_positions[:, 0],
                             feature_positions[:, 1],
                             color='darkgreen',
                             linestyle='',
                             marker='x')
                A = self.result['A_fit'][self.i_cam]
                k = self.result['k_fit'][self.i_cam]
                r1 = self.result['r1_fit'][index]
                t1 = self.result['t1_fit'][index]
                rX1 = self.result['rX1_fit'][self.i_cam]
                tX1 = self.result['tX1_fit'][self.i_cam]
                proj = self.project_board(r1, t1, rX1, tX1, A, k)
                proj = proj[feature_ids]
                # substract offset
                proj[:, 0] = proj[:, 0] - offset_x
                proj[:, 1] = proj[:, 1] - offset_y
                self.max_error = np.max(np.sqrt(np.sum((proj - feature_positions)**2, 1)))
                self.ax.plot(proj[:, 0],
                             proj[:, 1],
                             color='blue',
                             linestyle='',
                             marker='+')
                self.ax.legend(['detection', 'projection'])            
        elif (self.mask_single[self.i_cam, self.i_pose]):
            self.status = 'single'
            index = np.sum(self.mask_single[self.i_cam, :self.i_pose])
            feature_ids = self.calib_single['cam{:01d}'.format(self.i_cam)]['charuco_ids'][index].ravel()
            if (len(feature_ids) > 0):
                feature_positions = self.calib_single['cam{:01d}'.format(self.i_cam)]['charuco_corners'][index]
                # substract offset
                feature_positions[:, 0] = feature_positions[:, 0] - offset_x
                feature_positions[:, 1] = feature_positions[:, 1] - offset_y
                self.ax.plot(feature_positions[:, 0],
                             feature_positions[:, 1],
                             color='darkgreen',
                             linestyle='',
                             marker='x')
                A = self.result['A_fit'][self.i_cam]
                k = self.result['k_fit'][self.i_cam]
                r1 = self.result['r1_single_fit'][self.i_cam][0][index]
                t1 = self.result['t1_single_fit'][self.i_cam][0][index]
                rX1 = np.empty(0)
                tX1 = np.empty(0)
                proj = self.project_board(r1, t1, rX1, tX1, A, k)
                proj = proj[feature_ids]
                # substract offset
                proj[:, 0] = proj[:, 0] - offset_x
                proj[:, 1] = proj[:, 1] - offset_y
                self.max_error = np.max(np.sqrt(np.sum((proj - feature_positions)**2, 1)))                
                self.ax.plot(proj[:, 0],
                             proj[:, 1],
                             color='blue',
                             linestyle='',
                             marker='x')
                self.ax.legend(['detection', 'projection'])            
        self.info_text_update()
        self.canvas.draw()
        
    def info_text_update(self):
        recFileNames_split = self.recFileNames[self.i_cam].split('/')
        dataPath = self.result_directory
        text = 'path:\t{:s}\n'.format(dataPath) + \
               'file:\t{:s}\n'.format(recFileNames_split[-1]) + \
               'status:\t{:s}\n'.format(self.status) + \
               'max. error:\t{:0.2f} pixel\n'.format(self.max_error)
        self.info_text.setPlainText(text)
        
    def project_board(self,
                      r1, t1,
                      rX1, tX1,
                      A, k):
        # M
        M_0 = np.repeat(np.arange(1, self.boardWidth).reshape(1, self.boardWidth-1), self.boardHeight-1, axis=0).ravel().reshape(self.nFeatures, 1)
        M_1 = np.repeat(np.arange(1, self.boardHeight), self.boardWidth-1, axis=0).reshape(self.nFeatures, 1)
        M_2 = np.zeros(self.nFeatures).reshape(self.nFeatures, 1)
        M = np.concatenate([M_0, M_1, M_2], 1)
        # R1 * M + t1
        R1 = func.rodrigues2rotMat_single(r1)    
        m = np.dot(R1, M.T).T + t1
        if (self.status == 'multi'):
            # RX1 * m_proj + tX1
            RX1 = func.rodrigues2rotMat_single(rX1)
            m = np.dot(RX1, m.T).T + tX1
        x_pre = m[:, 0] / m[:, 2]
        y_pre = m[:, 1] / m[:, 2]
        # distort
        r2 = x_pre**2 + y_pre**2
        k_1 = k[0]
        k_2 = k[1]
        k_3 = k[4]
        p_1 = k[2]
        p_2 = k[3]
        x = x_pre * (1 + k_1 * r2 + k_2 * r2**2 + k_3 * r2**3) + \
            2 * p_1 * x_pre * y_pre + \
            p_2 * (r2 + 2 * x_pre**2)
        y = y_pre * (1 + k_1 * r2 + k_2 * r2**2 + k_3 * r2**3) + \
            p_1 * (r2 + 2 * y_pre**2) + \
            2 * p_2 * x_pre * y_pre
        # A * m_proj
        fx = A[0]
        cx = A[1]
        fy = A[2]
        cy = A[3]
        x_post = x * fx + cx
        y_post = y * fy + cy
        proj = np.concatenate([x_post, y_post], 0).reshape(2, self.nFeatures).T
        return proj        
        
    # shortkeys
    def keyPressEvent(self, event):
        if not(event.isAutoRepeat()):
            if (event.key() == Qt.Key_D):
                self.i_pose = int(np.median([0,
                                             int(self.field_i_pose.text()) + 1,
                                             self.nFrames - 1]))
                self.plot_draw()
                self.field_i_pose.setText(str(self.i_pose))
            elif (event.key() == Qt.Key_A):
                self.i_pose = int(np.median([0,
                                             int(self.field_i_pose.text()) - 1,
                                             self.nFrames - 1]))
                self.plot_draw()
                self.field_i_pose.setText(str(self.i_pose))
            elif(event.key() == Qt.Key_H):
                self.toolbar.home()
            elif(event.key() == Qt.Key_Z):
                self.toolbar.zoom()
            elif((event.key() == Qt.Key_P) or
                 (event.key() == Qt.Key_W)):
                self.toolbar.pan()
        else:
            print('WARNING: Auto-repeat is not supported')

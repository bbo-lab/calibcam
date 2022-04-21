import numpy as np
import sys

from . import multical_plot
from .calibrator import Calibrator, UnsupportedFormatException, UnequalFrameCountException

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



class MainWindow(QMainWindow):
    calibrator = None
    def __init__(self, calibrator, parent=None):
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

        self.calibrator = calibrator;

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
        
        # read the dataset
        self.read_recording()
        
        if (self.calibrator.recordingIsLoaded):
            # perform complete multi camera calibration
            self.calibrator.perform_multi_calibration()
            self.plot_calibration()
        else:
            self.button_performCalibration.setEnabled(True)
            self.button_loadCalibration.setEnabled(True)
        return

    def button_loadCalibration_press(self):
        self.button_performCalibration.setEnabled(False)
        self.button_loadCalibration.setEnabled(False)

        # read the dataset
        self.read_recording()

        self.calibrationIsLoaded = False
        # read the calibration
        self.read_calibration()
        if (self.calibrationIsLoaded):
            self.plot_calibration()
        return

    def plot_calibration(self):
        self.PlotWindow = multical_plot.PlotWindow(self.calibrator)
        self.PlotWindow.show()
        self.button_performCalibration.setEnabled(True)
        self.button_loadCalibration.setEnabled(True)
        return

    def read_calibration(self):
        dialog = QFileDialog()
        dialog.setStyleSheet("background-color: white;")
        dialogOptions = dialog.Options()
        dialogOptions |= dialog.DontUseNativeDialog
        calFileName, _ = QFileDialog.getOpenFileNames(dialog,
                                                           "Choose calibration file",
                                                           self.startDirectory,
                                                           "npy files (*.npy)",
                                                           options=dialogOptions)
        if (len(calFileName) == 1):
            # check if input file is a npy-file:
            filesAreCorrect = True
            fileEnding = calFileName[0].split('/')[-1].split('.')[-1]
            if (fileEnding != 'npy'):
                filesAreCorrect = False
            if not(filesAreCorrect):
                print('WARNING: Input file is not correct (no npy-file)')
                self.button_performCalibration.setEnabled(True)
                self.button_loadCalibration.setEnabled(True)
            else:
                print('LOAD CALIBRATION')
                # if everything is fine keep on going with the plotting
                self.calibrator.dataPath = '/'.join(calFileName[0].split('/')[:-1])
                self.calibrator.result = np.load(calFileName[0], allow_pickle=True).item()
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
                                                                "Video files (*.*)",
                                                                options=dialogOptions)
        if (len(recFileNames_unsorted) > 1):
            recFileNames = sorted(recFileNames_unsorted)
            try:
                self.calibrator.set_recordings(recFileNames)
            except UnsupportedFormatException as err:
                pass
            except UnequalFrameCountException as err:
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
                    self.calibrator.recordingIsLoaded = True
        else:
            print('WARNING: Provide at least two input files')


        if self.calibrator.recordingIsLoaded:
            for (i_cam, recname) in enumerate(self.calibrator.recFileNames):
                print(f'Loading recording {recname}\t(camera {i_cam:02d})')
        else:
            self.calibrator.reset()
        return






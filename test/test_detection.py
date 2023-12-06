import matplotlib
import unittest
import numpy as np
import yaml
import cv2
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from svidreader import filtergraph

from calibcam.calibrator_opts import get_default_opts, finalize_aruco_detector_opts
from calibcam.board import get_board_params
from calibcam import board
from calibcam.detection import detect_corners_cam



class TestDetection(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_detect_corners_cam(self):
        video = "./test/sample_images_1"
        board_name = "../boards/board_small"

        board_params = get_board_params(board_name)
        opts = get_default_opts()
        detection = opts['detection']
        detection['aruco_refine']['errorCorrectionRate'] = 0.1
        detection['aruco_interpolate']['minMarkers'] = 1

        detector_parameters = {  # SPOT for detector params
            'adaptiveThreshWinSizeMin': 23,
            'adaptiveThreshWinSizeMax': 43,
            'adaptiveThreshWinSizeStep': 5,

            'polygonalApproxAccuracyRate': 0.01,

            'perspectiveRemovePixelPerCell': 8,
            'maxErroneousBitsInBorderRate': 0.1,
            'errorCorrectionRate': 0.1,

            'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
            'cornerRefinementWinSize': 3,
            'cornerRefinementMaxIterations': 100,
            'cornerRefinementMinAccuracy': 0.05,
        }
        detection['aruco_detect']['parameters'] = detector_parameters
        detection['aruco_refine']['parameters'] = detector_parameters

        corners, ids, fin_frames_mask = detect_corners_cam(video=video, opts=opts, board_params=board_params)

        assert fin_frames_mask[2]

        with open('./test/sample_images_1/detections.yml') as f:
            human_marked = yaml.safe_load(f)
            i = 0
            for frame_idx, value in enumerate(fin_frames_mask):
                if value and frame_idx in human_marked:
                    human_frame = dict(zip(human_marked[frame_idx]['ids'], human_marked[frame_idx]['corners']))
                    for id, corner in zip(np.squeeze(ids[i]), np.squeeze(corners[i])):
                        if id in human_frame:
                            assert np.allclose(corner, human_frame[id], atol=5)
                i += value

import unittest
import numpy as np
import yaml
import cv2
import os
import svidreader
import svidreader.local_radial

from calibcam.calibrator_opts import get_default_opts
from calibcam.detection import detect_corners_cam


class TestDetection(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_detect_corners_cam(self):
        video = "./test/sample_images_1/"
        board_path = "./boards/board_small.npy"

        board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()
        if board_params is not None:
            board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']

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

        plot = True
        images = svidreader.get_reader(video)

        if plot:
            with open('./test/sample_images_1/detections.yml') as f:
                human_marked = yaml.safe_load(f)
                frame2corneridx = np.full(len(fin_frames_mask), fill_value=-1)
                frame2corneridx[fin_frames_mask] = np.arange(0, len(corners))
                import imageio
                if not(os.path.exists('test/out/') and os.path.isdir('test/out/')):
                    os.mkdir('test/out/')
                for frame, img in enumerate(images):
                    if fin_frames_mask[frame]:
                        fr_corner = corners[frame2corneridx[frame]]
                        for m in fr_corner:
                            cv2.drawMarker(img, position=np.asarray(m[0],dtype=np.int32), markerType=1, thickness=2,
                                           color=(0,255,255))
                    if frame in human_marked:
                        for c in human_marked[frame]['corners']:
                            cv2.drawMarker(img, position=np.asarray(c, dtype=np.int32), color=(255, 0, 0))
                    imageio.imwrite(F'test/out/{frame}.png', img)

        assert fin_frames_mask[2]

        with open('./test/sample_images_1/detections.yml') as f:
            radcontrast = svidreader.local_radial.RadialContrast(images, options={'lib':'np'})
            human_marked = yaml.safe_load(f)
            detected_frames = np.where(fin_frames_mask)[0]
            count = 0
            for i, frame_idx in enumerate(detected_frames):
                img = radcontrast.read(frame_idx)
                imageio.imwrite(F'test/out/{frame_idx}_weight.png', img)
                if frame_idx in human_marked:
                    human_frame = dict(zip(human_marked[frame_idx]['ids'], human_marked[frame_idx]['corners']))
                    for id, corner in zip(np.squeeze(ids[i]), np.squeeze(corners[i])):
                        if id in human_frame and img[tuple(np.asarray(corner, dtype=int)[::-1,np.newaxis])] > 20:
                            count += 1
                            assert np.allclose(corner, human_frame[id], atol=5)
            print(F"accepted {count} of {0}")
            assert count > 4
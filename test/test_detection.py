import unittest
import numpy as np
import yaml
import cv2
import os
import svidreader

from calibcam import helper
from calibcam.calibrator_opts import get_default_opts
from calibcam.detection import detect_corners_cam


class TestDetection(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_detect_corners_cam(self):
        video = "./test/sample_images_1/"
        board_path = "./boards/bboboard-v4.npy"

        board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()
        if board_params is not None:
            board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']

        opts = get_default_opts(1, do_fill=True)
        detection = opts['detection_opts']
        detection['aruco_interpolate']['minMarkers'] = 1

        corners, ids, fin_frames_mask = detect_corners_cam(video=video, opts=opts, board_params=board_params)

        plot = True
        images = svidreader.get_reader(video)

        if plot:
            with open(video + '/detections.yml') as f:
                human_marked = yaml.safe_load(f)
                frame2corneridx = np.full(len(fin_frames_mask), fill_value=-1)
                frame2corneridx[fin_frames_mask] = np.arange(0, len(corners))
                import imageio
                os.makedirs('test/out/', exist_ok=True)
                for frame, img in enumerate(images):
                    if fin_frames_mask[frame]:
                        fr_corner = corners[frame2corneridx[frame]]
                        for m in fr_corner:
                            cv2.drawMarker(img, position=np.asarray(m[0], dtype=np.int32), markerType=1, thickness=2,
                                           color=(0, 255, 255))
                    if frame in human_marked:
                        for c in human_marked[frame]['corners']:
                            cv2.drawMarker(img, position=np.asarray(c, dtype=np.int32), color=(255, 0, 0))
                    imageio.imwrite(F'test/out/{frame}.png', img)

        assert fin_frames_mask[2], "No detections in frame 2"

        with open(video + '/detections.yml') as f:
            radcontrast = helper.RadialContrast(images, options={'lib': 'np'}, norm_mean=0.312)
            human_marked = yaml.safe_load(f)
            detected_frames = np.where(fin_frames_mask)[0]
            for i, frame_idx in enumerate(detected_frames):
                count = 0
                img = radcontrast.read(frame_idx)
                imageio.imwrite(F'test/out/{frame_idx}_weight.png', img)
                if frame_idx in human_marked:
                    human_frame = dict(zip(human_marked[frame_idx]['ids'], human_marked[frame_idx]['corners']))
                    for id, corner in zip(np.squeeze(ids[i]), np.squeeze(corners[i])):
                        if id in human_frame and img[tuple(np.asarray(corner, dtype=int)[::-1,np.newaxis])] > 0:
                            count += 1
                            assert np.allclose(corner, human_frame[id], atol=5)

                        # print('contrast_pixel:', img[tuple(np.asarray(corner, dtype=int)[::-1,np.newaxis])])
                print(F"accepted {count} of {len(np.squeeze(ids[i]))} in frame {frame_idx}")

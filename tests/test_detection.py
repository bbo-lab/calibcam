import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import svidreader
import yaml

from calibcam import helper
from calibcam.calibrator_opts import get_default_opts
from calibcam.detection import (
    _nanodeep_corners_to_charuco,
    _nanodeep_frame_prefix,
    _normalize_nanodeep_config_paths,
    detect_corners_cam,
)
from calibcamlib import Board, Detections


class TestDetection(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def atest_detect_corners_cam(self):
        video = "./test/sample_images_1/"
        board_path = "./boards/bboboard-v4.npy"

        board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()
        if board_params is not None:
            board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']

        board = Board(board_params)

        opts = get_default_opts(1, do_fill=True)
        detection = opts['detection_opts']
        detection['aruco_interpolate']['minMarkers'] = 1

        corners, ids, fin_frames_mask = detect_corners_cam(video=video, opts=opts, board=board,
                                                           load_frame_idx_list=None, offset_to_real=0)

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
                        if id in human_frame and img[tuple(np.asarray(corner, dtype=int)[::-1, np.newaxis])] > 0:
                            count += 1
                            assert np.allclose(corner, human_frame[id], atol=5)

                        # print('contrast_pixel:', img[tuple(np.asarray(corner, dtype=int)[::-1,np.newaxis])])
                print(F"accepted {count} of {len(np.squeeze(ids[i]))} in frame {frame_idx}")

    def test_nanodeep_corners_to_charuco(self):
        corners_by_id = {
            5: np.array([50.5, 51.5]),
            1: np.array([10.5, 11.5]),
            3: np.array([30.5, 31.5]),
        }

        corners, ids = _nanodeep_corners_to_charuco(corners_by_id)

        assert corners.shape == (3, 1, 2)
        assert ids.shape == (3, 1)

        assert corners.dtype == np.float32
        assert ids.dtype == np.int32

        np.testing.assert_array_equal(
            ids,
            np.array([[1], [3], [5]], dtype=np.int32),
        )

        np.testing.assert_allclose(
            corners[:, 0, :],
            np.array(
                [
                    [10.5, 11.5],
                    [30.5, 31.5],
                    [50.5, 51.5],
                ],
                dtype=np.float32,
            ),
        )

        empty_corners, empty_ids = _nanodeep_corners_to_charuco({})
        assert empty_corners is None
        assert empty_ids is None

    def test_nanodeep_frame_prefix_distinguishes_same_basenames(self):
        first = _nanodeep_frame_prefix(
            "/tmp/dataset_a/camera.mp4"
        )
        second = _nanodeep_frame_prefix(
            "/tmp/dataset_b/camera.mp4"
        )

        assert first.startswith("camera_")
        assert second.startswith("camera_")
        assert first != second

    def test_nanodeep_path_options_are_serializable(self):
        config = {
            "project_root": Path("/tmp/nanodeep"),
            "nano_executable": Path("/tmp/nanodeep/detect_batch"),
            "deep_checkpoint": Path("/tmp/nanodeep/detector.ckpt"),
            "refinenet_checkpoint": Path("/tmp/nanodeep/refinenet.ckpt"),
            "config_path": Path("/tmp/nanodeep/config.yaml"),
            "work_dir": Path("/tmp/nanodeep/work"),
            "device": "cpu",
            "keep_work": False,
        }

        normalized = _normalize_nanodeep_config_paths(config)

        assert normalized is config

        for key in (
            "project_root",
            "nano_executable",
            "deep_checkpoint",
            "refinenet_checkpoint",
            "config_path",
            "work_dir",
        ):
            assert isinstance(config[key], str)

        assert config["device"] == "cpu"
        assert config["keep_work"] is False

    def test_unsupported_detection_backend_raises(self):
        board_path = "./boards/bboboard-v4.npy"
        board_params = np.load(
            os.path.expanduser(board_path),
            allow_pickle=True,
        ).item()

        if "marker_size_real" not in board_params:
            board_params["marker_size_real"] = (
                board_params["square_size_real"]
                * board_params["marker_size"]
            )

        board = Board(board_params)

        opts = get_default_opts(1, do_fill=True)
        opts["detection_opts"]["backend"] = "does_not_exist"

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported detection backend",
        ):
            detect_corners_cam(
                video="fake_video.mp4",
                opts=opts,
                board=board,
                frames_list=[0],
            )

    def test_nanodeep_missing_required_options_raises(self):
        board_path = "./boards/bboboard-v4.npy"
        board_params = np.load(
            os.path.expanduser(board_path),
            allow_pickle=True,
        ).item()

        if "marker_size_real" not in board_params:
            board_params["marker_size_real"] = (
                board_params["square_size_real"]
                * board_params["marker_size"]
            )

        board = Board(board_params)

        opts = get_default_opts(1, do_fill=True)
        opts["detection_opts"]["backend"] = "nanodeepcharuco"

        with self.assertRaisesRegex(
            ValueError,
            "missing required options",
        ):
            detect_corners_cam(
                video="fake_video.mp4",
                opts=opts,
                board=board,
                frames_list=[0],
            )

    def test_nanodeep_backend_frame_path(self):
        board_path = "./boards/bboboard-v4.npy"
        board_params = np.load(
            os.path.expanduser(board_path),
            allow_pickle=True,
        ).item()

        if "marker_size_real" not in board_params:
            board_params["marker_size_real"] = (
                board_params["square_size_real"]
                * board_params["marker_size"]
            )

        board = Board(board_params)

        opts = get_default_opts(1, do_fill=True)
        opts["RC_reject_corners"] = False
        opts["detection_opts"]["backend"] = "nanodeepcharuco"

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        fake_reader = MagicMock()
        fake_reader.get_data.return_value = frame

        fake_detector = MagicMock()
        fake_detector.process_frame.return_value = SimpleNamespace(
            corners={
                0: np.array([10.0, 10.0], dtype=np.float32),
                1: np.array([20.0, 10.0], dtype=np.float32),
                4: np.array([10.0, 20.0], dtype=np.float32),
                5: np.array([20.0, 20.0], dtype=np.float32),
                8: np.array([10.0, 30.0], dtype=np.float32),
            }
        )

        with (
            patch(
                "calibcam.detection._build_nanodeep_detector",
                return_value=fake_detector,
            ) as build_detector,
            patch(
                "calibcam.detection.filtergraph.get_reader",
                return_value=fake_reader,
            ),
            patch(
                "calibcam.detection.camfunctions.get_header_from_reader",
                return_value={
                    "offset": (0, 0),
                },
            ),
            patch(
                "calibcam.detection.camfunctions.get_n_frames_from_reader",
                return_value=1,
            ),
        ):
            detections = detect_corners_cam(
                video="fake_video.mp4",
                opts=opts,
                board=board,
                frames_list=[0],
            )

        build_detector.assert_called_once()
        fake_detector.process_frame.assert_called_once()

        processed_frame, frame_tag = (
            fake_detector.process_frame.call_args.args
        )

        np.testing.assert_array_equal(processed_frame, frame)
        assert frame_tag.startswith("fake_video_")
        assert frame_tag.endswith("_00000000")

        assert detections.get_n_frames() == 1
        assert detections.get_n_markers() == 5

        fake_reader.close.assert_called_once()

    def test_detection_merge(self):
        detections_lists = [
            {
                'marker_coords': np.array([
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                ]),
                'marker_ids': [[10, 13, 14], [11, 13, 15], [11, 13, 14]],
                'detection_idxs': [0, 3, 4],
                'frame_idxs': [0, 3, 4],
            }, {
                'marker_coords': np.array([
                    [[13, 14], [15, 16]],
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                ]),
                'marker_ids': [[0, 3, 14], [1, 3, 15], [1, 3, 14]],
                'detection_idxs': [1, 3, 5],
                'frame_idxs': [11, 13, 15],
            }
        ]

        detections = [Detections.from_list(d) for d in detections_lists]

        detections_all = sum(detections, Detections())

        markers_all = [0, 1, 3, 10, 11, 13, 14, 15]
        frames_all = [0, 1, 3, 4, 5]

        d_marker_coords = detections_all.to_array()["marker_coords"]

        assert len(d_marker_coords) == 2
        assert len(markers_all) == d_marker_coords.shape[2]
        assert len(frames_all) == d_marker_coords.shape[1]

        for i_d, (d, d_list) in enumerate(zip(detections, detections_lists)):
            fr_mask = np.isin(frames_all, d_list["detection_idxs"])
            m_mask = np.isin(markers_all, d_list["marker_ids"])

            assert np.all(
                (d.to_array()["marker_coords"][0] == d_marker_coords[i_d][fr_mask][:, m_mask]) |
                np.isnan(d.to_array()["marker_coords"][0]) |
                np.isnan(d_marker_coords[i_d][fr_mask][:, m_mask])
            )

    def test_detection_sizes(self):
        detections_lists = [
            {
                'marker_coords': np.array([
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                ]),
                'marker_ids': [[10, 13, 14], [11, 13, 15], [11, 13, 14]],
                'detection_idxs': [0, 3, 4],
                'frame_idxs': [0, 3, 4],
            }, {
                'marker_coords': np.array([
                    [[13, 14], [15, 16]],
                    [[17, 18], [19, 20]],
                    [[21, 22], [23, 24]],
                ]),
                'marker_ids': [[0, 3, 14], [1, 3, 15], [1, 3, 14]],
                'detection_idxs': [1, 3, 5],
                'frame_idxs': [11, 13, 15],
            }
        ]

        detections = [Detections.from_list(d) for d in detections_lists]
        detections_all = sum(detections, Detections())

        assert detections_all.get_n_frames() == 5
        assert detections_all.get_n_markers() == 8
        assert np.all(detections_all.get_n_detections_frames() == np.array([[0, 0, 0, 1, 2, 3, 2, 1],
                                                                            [1, 2, 3, 0, 0, 0, 2, 1]]))
        assert np.all(detections_all.get_n_detections_markers() == np.array([[3, 0, 3, 3, 0],
                                                                             [0, 3, 3, 0, 3]]))
        assert np.all(detections_all.to_array()["frame_idxs"] == np.array([[0, -1, 3, 4, -1],
                                                                           [-1, 11, 13, -1, 15]]))

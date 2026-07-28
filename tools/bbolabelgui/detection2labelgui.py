import argparse
import os
from pathlib import Path

import numpy as np
from bbo import label_lib
from bbo.geometry import RigidTransform

from calibcamlib import Board
from calibcamlib.camerasystem import Camerasystem


def main(calibration_file, label_path=None, reference_path=None):
    calibration_file = Path(calibration_file)

    if label_path is None:
        label_path = os.getcwd()
    label_path = Path(label_path)
    if label_path.is_dir():
        label_path = label_path / f"labels-{calibration_file.parent.stem}"

    if reference_path is None:
        reference_path = os.getcwd()
    reference_path = Path(reference_path)
    if reference_path.is_dir():
        reference_path = reference_path / f"references-{calibration_file.parent.stem}"

    calibration_dict = Camerasystem.load_dict(calibration_file)

    cs = Camerasystem.load(calibration_file)
    board = get_board_from_calibration(calibration_dict)

    board_points = board.get_board_points()
    board_poses = get_board_poses_from_calibration(calibration_dict)
    used_frames_ids = calibration_dict["info"]["used_frames_ids"]

    used_aruco_ids = np.arange(len(board_points))  # For later more complex setups with multiple boards

    labeler = np.ones((len(cs.cameras),), dtype=int) * 2
    times = np.zeros((len(cs.cameras),))

    reference_labels_dict = get_empty_labels_dict(used_aruco_ids)
    for pose, frame_idx in zip(board_poses, used_frames_ids):
        points = pose.apply(board_points)
        pixel_coords = cs.project(points).transpose([1, 0, 2])

        for coords, id in zip(pixel_coords, used_aruco_ids):
            reference_labels_dict["labels"][f"corner_{id:04d}"][frame_idx] = {}
            reference_labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["coords"] = coords
            reference_labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["labeler"] = labeler
            reference_labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["point_times"] = times
    label_lib.save(reference_path, reference_labels_dict, yml_only=True)
    print(f"Saved {reference_path}")

    corners = np.asarray(calibration_dict["info"]["corners"]).transpose([1, 2, 0, 3])
    labels_dict = get_empty_labels_dict(used_aruco_ids)
    for corners_frame, frame_idx in zip(corners, used_frames_ids):
        for coords, id in zip(corners_frame, used_aruco_ids):
            labels_dict["labels"][f"corner_{id:04d}"][frame_idx] = {}
            labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["coords"] = coords
            labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["labeler"] = labeler
            labels_dict["labels"][f"corner_{id:04d}"][frame_idx]["point_times"] = times
    label_lib.save(label_path, labels_dict, yml_only=True)
    print(f"Saved {label_path}")


def get_board_from_calibration(calibration_dict):
    return Board(calibration_dict["board_params"])


def get_board_poses_from_calibration(calibration_dict):
    return RigidTransform(rotation=np.asarray(calibration_dict["info"]["rvecs_boards"]),
                          translation=np.asarray(calibration_dict["info"]["tvecs_boards"]),
                          rotation_type="rotvec")


def get_empty_labels_dict(used_aruco_ids):
    labels_dict = {
        "version": "1.0",
        "labeler_list": ["_unmarked", "_unknown", "OPENCV"],
        "action_list": ["create", "delete"],
        "labels": {}
    }

    for id in used_aruco_ids:
        labels_dict["labels"][f"corner_{id:04d}"] = {}

    return labels_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("CALIBRATION_FILE", type=str)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--reference_path", type=str, default=None)

    args = parser.parse_args()

    main(args.CALIBRATION_FILE, label_path=args.label_path, reference_path=args.reference_path)

#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from calibcamlib.board import Board
from calibcamlib.camerasystem import Camerasystem
from bbo.geometry import RigidTransform
from bbo import label_lib

import logging
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a labelled calibration-board sketch as a NumPy file."
    )
    parser.add_argument(
        "-c",
        "--calibration",
        type=Path,
        required=True,
        metavar="CALIBRATION_FILE",
        help="Input calibration yml file.",
    )
    parser.add_argument(
        "-p",
        "--poses",
        type=Path,
        required=True,
        metavar="POSES_FILE",
        help="Input board poses yml file.",
    )
    parser.add_argument(
        "-i",
        "--boardindex",
        type=int,
        required=False,
        default=0,
        metavar="BOARD_INDEX",
        help="Board index in calibration file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="REFERENCE_FILE",
        help="Output reference .yml file.",
    )
    return parser.parse_args()


def create_references(calibration_file: Path, poses_path: Path, boardindex: int = 0) -> dict:
    cs = Camerasystem.load(calibration_file)
    with open(poses_path, "r") as poses_file:
        poses = yaml.safe_load(poses_file)
    frame_idxs = np.array(poses['frame_idxs']).T

    board_params = Camerasystem.load_dict(calibration_file)["info"]["board_params"]
    if isinstance(board_params, list):
        board_params = board_params[boardindex]
    board = Board(board_params)
    ideal2cs = RigidTransform(rotation=np.array(poses["rvecs"]),
                              translation=np.array(poses["tvecs"]),
                              rotation_type="rotvec")

    board_points = board.get_board_points()  # kx3
    board_points_cs = ideal2cs.apply_broadcast(board_points)  # nxkx2
    board_point_coords = cs.project(board_points_cs)  # cxnxkx2
    board_point_coords_knc = board_point_coords.transpose([2, 1, 0, 3])  # kxnxcx2

    labels = {
        'version': 1.0,
        'labeler_list': ["_unmarked", "_unknown"],
        'action_list': ["create", "delete"],
        'labels': {},
    }

    for i_corn, (cid, board_point_coords_nc) in enumerate(zip(board.get_corner_ids(), board_point_coords_knc)):
        corner_name = f"corner_{cid:04d}"
        ll = labels["labels"].setdefault(corner_name, {})
        for fridxs, board_point_coords_c in zip(frame_idxs, board_point_coords_nc):
            for i_cam, cam_frame_idx in enumerate(fridxs):
                if cam_frame_idx == -1:
                    continue
                llc = ll.setdefault(cam_frame_idx, {})
                if not llc:
                     llc['coords'] = np.full((len(fridxs), 2), np.nan)
                     llc['labeler'] = [1, 1]
                     llc['point_times'] = [0, 0]
                llc["coords"][i_cam] = board_point_coords_c[i_cam]

    return labels


def main() -> None:
    args = parse_args()

    if not args.calibration.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {args.input}")

    if not args.poses.is_file():
        raise FileNotFoundError(f"Board poses file does not exist: {args.input}")

    labels = create_references(args.calibration, args.poses, args.boardindex)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_lib.save(args.output, labels)
    print(f"Saved sketch to: {args.output}")


if __name__ == "__main__":
    main()
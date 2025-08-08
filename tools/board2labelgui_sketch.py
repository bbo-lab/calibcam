import os
from pathlib import Path

import numpy as np
import argparse
from calibcamlib.camerasystem import Camerasystem
from calibcam.board import Board
from bbo.geometry import RigidTransform
from bbo import label_lib

def main(calibration_file, sketch_path = None):
    calibration_file = Path(calibration_file)

    if sketch_path is None:
        sketch_path = os.getcwd()
    sketch_path = Path(sketch_path)
    if sketch_path.is_dir():
        sketch_path = sketch_path / f"sketch-{calibration_file.parent.stem}"

    calibration_dict = Camerasystem.load_dict(calibration_file)

    board = get_board_from_calibration(calibration_dict)
    board_params = board.get_board_params()

    rows = int(board_params["boardWidth"])
    columns = int(board_params["boardHeight"])
    if "ids" in board_params:
        ids = board_params["ids"]
    else:
        ids = np.arange((rows-1) * (columns-1))

    board_img = board.get_board_img()
    pixel_size = board_img.shape
    row_px = pixel_size[0] / rows
    column_px = pixel_size[0] / columns

    sketch = {
        "sketch": board_img,
        "sketch_label_locations": {}
    }

    for i_id, id in enumerate(ids):
        i,j = np.unravel_index(i_id, (rows-1, columns-1))
        sketch["sketch_label_locations"][f"corner_{id:03d}"] = np.array([(j+1)*column_px-0.5, (i+1)*row_px-0.5])

    np.save(sketch_path, sketch)


def get_board_from_calibration(calibration_dict):
    return Board(calibration_dict["board_params"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("CALIBRATION_FILE", type=str)
    parser.add_argument("--path", type=str, default=None)

    args = parser.parse_args()

    main(args.CALIBRATION_FILE, sketch_path=args.path)

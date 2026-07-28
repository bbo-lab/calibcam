#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calibcamlib.board import Board

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a labelled calibration-board sketch as a NumPy file."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="BOARD_FILE",
        help="Input calibration-board .npy file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="SKETCH_FILE",
        help="Output sketch .npy file.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        metavar="IMAGE_FILE",
        help="Optionally save a visualization, for example as plot.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization interactively.",
    )
    return parser.parse_args()


def create_sketch(board_path: Path) -> dict:
    if board_path.suffix == ".npy":
        board = Board.from_file(board_path)
    elif board_path.suffix == ".yml":
        from calibcamlib.camerasystem import Camerasystem
        board = Board(Camerasystem.load_dict(board_path)["info"]["board_params"])
    else:
        raise ValueError(f"Unknown board format: {board_path.suffix}")

    board_params = board.get_board_params()
    image, pixel_size = board.get_board_img(return_pixel_size=True)

    board_dimensions = np.array(
        [
            board_params["boardWidth"],
            board_params["boardHeight"],
        ],
        dtype=float,
    )
    checker_pixels = np.asarray(pixel_size, dtype=float) / board_dimensions

    corner_ids = board.get_corner_ids()
    points = board.get_board_points_base()[:, :2]
    points = points * checker_pixels[np.newaxis, :]

    return {
        "sketch": image,
        "sketch_label_locations": {
            f"corner_{corner_id:04d}": point - 0.5
            for corner_id, point in zip(corner_ids, points)
        },
    }


def plot_sketch(
    sketch: dict,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    figure, axis = plt.subplots()
    axis.imshow(sketch["sketch"])

    for point in sketch["sketch_label_locations"].values():
        axis.plot(point[0], point[1], "b+")

    axis.set_title("Calibration-board sketch")
    axis.set_axis_off()
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close(figure)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Board file does not exist: {args.input}")

    sketch = create_sketch(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, sketch)
    print(f"Saved sketch to: {args.output}")

    if args.show or args.plot_output is not None:
        plot_sketch(
            sketch,
            output_path=args.plot_output,
            show=args.show,
        )


if __name__ == "__main__":
    main()
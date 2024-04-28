import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path

def generate_charuco_pattern(rows, columns, square_size_real, marker_ratio, aruco_size, output_file):
    output_file = Path(output_file)
    match aruco_size:
        case 4:
            aruco_dict = aruco.DICT_4X4_250
        case 5:
            aruco_dict = aruco.DICT_5X5_250

    # Create dictionary and board
    board = aruco.CharucoBoard_create(rows, columns, square_size_real, marker_ratio*square_size_real, aruco.Dictionary_get(aruco_dict))

    # Generate the Charuco board image
    board_image = board.draw(((aruco_size+4)*columns, (aruco_size+4)*rows))

    board = {
        "boardWidth": rows,
        "boardHeight": columns,
        "square_size_real": square_size_real,
        "marker_size": marker_ratio,
        "dictionary_type": aruco_dict,
        "opencv_version": cv2.__version__,
        "unit": "m",
        "board_name": output_file.stem,
        "board_format_version": "1.0.0",
    }

    # Save the Charuco board image as APNG
    cv2.imwrite(output_file.with_suffix(".png").as_posix(), board_image)
    np.save(output_file.with_suffix(".npy").as_posix(), board)

if __name__ == "__main__":
    # Parameters for the Charuco board
    rows = 7  # Number of checkerboard rows
    columns = 7  # Number of checkerboard columns
    aruco_size = 4  # Number of aruco rows and columns

    square_size_real = 0.14  # Size of each checker square in meters
    marker_ratio = 0.75  # Ratio of aruco to checker square
    output_file = f"board_pair_large_{rows}x{columns}_{rows*100*square_size_real}x{columns*100*square_size_real}.png"

    # Generate and save the Charuco pattern
    generate_charuco_pattern(rows, columns, square_size_real, marker_ratio, aruco_size, output_file)
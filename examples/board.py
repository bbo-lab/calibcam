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
    pixel_size = (round(((aruco_size+2)/marker_ratio)*rows), round(((aruco_size+2)/marker_ratio)*columns))
    print(pixel_size)
    board_image = board.draw(pixel_size)

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
    rows = 5  # Number of checkerboard rows
    columns = 7  # Number of checkerboard columns
    aruco_size = 4  # Number of aruco rows and columns

    square_size_real = 0.059  # Size of each checker square in meters
    marker_ratio = 0.75  # Ratio of aruco to checker square. Assert that (aruco_size+2)/marker_ratio is an integer

    square_pixel_width = (aruco_size+2)/marker_ratio
    assert square_pixel_width==int(square_pixel_width), "Marker ratio does not match"

    output_file = f"board_{rows}x{columns}_{rows*square_size_real:.5f}x{columns*square_size_real:.5f}.png"

    # Generate and save the Charuco pattern
    generate_charuco_pattern(rows, columns, square_size_real, marker_ratio, aruco_size, output_file)
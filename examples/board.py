import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path

def generate_charuco_pattern(rows, columns, square_size_real, marker_ratio, aruco_size, aruco_ids, output_file):
    output_file = Path(output_file)
    match aruco_size:
        case 4:
            aruco_dict = aruco.DICT_4X4_250
        case 5:
            aruco_dict = aruco.DICT_5X5_250

    aruco_ids = aruco_ids[:(rows * columns) // 2]  # This must probably be improved for boards with uneven rows/columns
    print(len(aruco_ids))

    # Create dictionary and board
    board = aruco.CharucoBoard((rows, columns), square_size_real, marker_ratio*square_size_real,
                                      aruco.getPredefinedDictionary(aruco_dict), ids=aruco_ids)

    # Generate the Charuco board image
    pixel_size = (round(((aruco_size+2)/marker_ratio)*rows), round(((aruco_size+2)/marker_ratio)*columns))
    print(pixel_size)
    board_image = board.generateImage(pixel_size)

    board = {
        "boardWidth": rows,
        "boardHeight": columns,
        "ids": aruco_ids,
        "square_size_real": square_size_real,
        "marker_size": marker_ratio,
        "dictionary_type": aruco_dict,
        "opencv_version": cv2.__version__,
        "unit": "m",
        "board_name": output_file.stem,
        "rotation": (0, 0, 0),
        "offset": (0, 0, 0),
        "legacy": False,
        "board_format_version": "1.1.0",
    }

    # Save the Charuco board image as APNG
    cv2.imwrite(output_file.with_suffix(".png").as_posix(), board_image)
    np.save(output_file.with_suffix(".npy").as_posix(), board)

if __name__ == "__main__":
    # Parameters for the Charuco board
    rows = 7  # Number of checkerboard rows
    columns = 7  # Number of checkerboard columns
    aruco_size = 4  # Number of aruco rows and columns

    square_size_real = 0.0035714286  # Size of each checker square in meters
    marker_ratio = 0.75  # Ratio of aruco to checker square. Assert that (aruco_size+2)/marker_ratio is an integer
    aruco_range = (25, 50)

    square_pixel_width = (aruco_size+2)/marker_ratio
    assert square_pixel_width==int(square_pixel_width), "Marker ratio does not match"

    output_file = f"board_{rows}x{columns}_{rows*square_size_real:.5f}x{columns*square_size_real:.5f}_{'-'.join([str(a) for a in aruco_range])}.png"

    # Generate and save the Charuco pattern
    generate_charuco_pattern(rows, columns, square_size_real, marker_ratio, aruco_size, np.arange(*aruco_range), output_file)

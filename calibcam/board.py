import os
import numpy as np
import pathlib
import cv2


def get_board_params(board_source):
    if isinstance(board_source, pathlib.Path):
        board_path = board_source / 'board.npy'
    else:
        board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../boards', board_source + '.npy')

    board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()

    if board_params is not None:
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']  # noqa

    return board_params


def make_board(board_params):
    board = cv2.aruco.CharucoBoard_create(board_params['boardWidth'],  # noqa
                                          board_params['boardHeight'],
                                          board_params['square_size_real'],
                                          board_params['marker_size'] * board_params['square_size_real'],
                                          cv2.aruco.getPredefinedDictionary(  # noqa
                                              board_params['dictionary_type']))

    return board


def make_board_points(board_params):
    board_width = board_params['boardWidth']
    board_height = board_params['boardHeight']
    square_size = board_params['square_size_real']

    n_corners = (board_width - 1) * (board_height - 1)

    board_0 = np.repeat(np.arange(1, board_width).reshape(1, board_width - 1), board_height - 1,
                        axis=0).ravel().reshape(n_corners, 1)
    board_1 = np.repeat(np.arange(1, board_height), board_width - 1, axis=0).reshape(n_corners, 1)
    board_2 = np.zeros(n_corners).reshape(n_corners, 1)
    board = np.concatenate([board_0, board_1, board_2], 1) * square_size

    return board  # n_corners x 3


def make_board_sketch(board_params, image_square_size=100, out_npy_file=None):
    """Draws a sketch of the board and saves it to a .npy file for bbo-labelui use"""
    square_size = board_params['square_size_real']
    board = make_board(board_params)
    board_img = board.draw((board_params['boardWidth'] * image_square_size,
                            board_params['boardHeight'] * image_square_size))
    output = {}
    if out_npy_file is None:
        out_npy_file = "board_sketch.npy"

    output['sketch'] = board_img
    board_points = make_board_points(board_params) / square_size
    board_points *= image_square_size
    board_points = board_points[:, :2]
    output['sketch_label_locations'] = {}
    for i, point in enumerate(board_points):
        output['sketch_label_locations'][f'corner_{i:03d}'] = point
    np.save(out_npy_file, output)

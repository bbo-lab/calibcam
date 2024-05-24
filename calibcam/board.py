import os
import numpy as np
import pathlib
import cv2
from pathlib import Path


def get_board_params(board_source):
    board_source = Path(board_source)
    if board_source.is_file():
        board_path = board_source.as_posix()
    elif board_source.is_dir():
        board_path = board_source / 'board.npy'
    else:
        board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../boards', board_source.as_posix() + '.npy')

    board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()

    if board_params is not None:
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']  # noqa

    return board_params


def make_board(board_params):
    board = cv2.aruco.CharucoBoard((board_params['boardWidth'],
                                    board_params['boardHeight']),
                                   board_params['square_size_real'],
                                   board_params['marker_size'] * board_params['square_size_real'],
                                   cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']))

    return board


def make_board_points(board_params, exact=False):
    board_width = board_params['boardWidth']
    board_height = board_params['boardHeight']
    if exact:
        square_size_x = board_params['square_size_real_y']
        square_size_y = board_params['square_size_real_x']
    else:
        square_size_x = board_params['square_size_real']
        square_size_y = board_params['square_size_real']

    n_corners = (board_width - 1) * (board_height - 1)

    board_0 = np.repeat(np.arange(1, board_width).reshape(1, board_width - 1), board_height - 1,
                        axis=0).ravel().reshape(n_corners, 1)
    board_1 = np.repeat(np.arange(1, board_height), board_width - 1, axis=0).reshape(n_corners, 1)
    board_2 = np.zeros(n_corners).reshape(n_corners, 1)
    board = np.concatenate([board_0 * square_size_x, board_1 * square_size_y,
                            board_2], 1)

    return board  # n_corners x 3

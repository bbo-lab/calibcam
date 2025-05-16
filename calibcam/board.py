import os
from collections.abc import Iterable

import numpy as np
import pathlib
import cv2
from pathlib import Path


def load_board_params(board_path, board_idx=None):
    if not isinstance(board_path, str) and isinstance(board_path, Iterable):
        board_params_list = [load_board_params(bp) for bp in board_path]
        if board_idx is not None:
            return board_params_list[board_idx]
        else:
            return board_params_list

    board_path = Path(board_path)
    if board_path.is_file():
        board_path = board_path.as_posix()
    elif board_path.is_dir():
        board_path = board_path / 'board.npy'
    else:
        board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../boards', board_path.as_posix() + '.npy')

    board_params = np.load(os.path.expanduser(board_path), allow_pickle=True).item()

    if board_params is not None:
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']  # noqa

    return board_params


def make_board(board_params):
    if isinstance(board_params, Iterable) and not isinstance(board_params, dict):
        return [make_board(bp) for bp in board_params]

    board = cv2.aruco.CharucoBoard((board_params['boardWidth'],
                                    board_params['boardHeight']),
                                   board_params['square_size_real'],
                                   board_params['marker_size'] * board_params['square_size_real'],
                                   cv2.aruco.getPredefinedDictionary(board_params['dictionary_type']))

    if "legacy" in board_params:
        board.setLegacyPattern(board_params["legacy"])
    elif "version" not in board_params: # TODO: This identification might still need some refinement
        board.setLegacyPattern(True)

    return board


def make_board_points(board_params, exact=False):
    if isinstance(board_params, Iterable):
        return [make_board_points(bp, exact) for bp in board_params]

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


class Board:
    def __init__(self, board_params):
        self.board_params = board_params

    def get_board_params(self):
        return self.board_params

    def get_cv2_board(self):
        return make_board(self.board_params)

    def get_board_points(self, exact=False):
        return make_board_points(self.board_params, exact=exact)
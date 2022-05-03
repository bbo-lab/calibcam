import os
import numpy as np
import pathlib


def get_board_params(board_source):
    if isinstance(board_source, pathlib.PosixPath) or isinstance(board_source, pathlib.WindowsPath):
        board_path = board_source / 'board.npy'
    else:
        board_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boards', board_source + '.npy')

    board_params = np.load(board_path, allow_pickle=True).item()

    if board_params is not None:
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']

    return board_params

import os
import cv2
import numpy as np
import pathlib

def get_board_params(board_source):
    if isinstance(board_source,pathlib.PosixPath) or isinstance(board_source,pathlib.WindowsPath):
        boardpath = board_source / 'board.npy'
    else:
        boardpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'boards',board_source+'.npy')
    
    board_params = np.load(boardpath, allow_pickle=True).item()
    
    if not board_params is None: 
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']
        board_params['dictionary'] = cv2.aruco.getPredefinedDictionary(board_params['dictionary_type'])
    
    return board_params

import cv2

def get_board_params(board_name):
    boards = { 
        'large_dual_led': {
            'boardWidth': 4,
            'boardHeight': 6,
            'square_size': 1, # scalar
            'marker_size': 0.6, # scalar
            'square_size_real': 6.5, # cm
            'dictionary': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            },
        'small_dual_led': { #TODO: Input params for this board
            'boardWidth': 0,
            'boardHeight': 0,
            'square_size': 0, # scalar
            'marker_size': 0, # scalar
            'square_size_real': 0, # cm
            'dictionary': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            },
        'npy': None,
        }
    
    board_params = boards[board_name]
    if not board_params is None: 
        board_params['marker_size_real'] = board_params['square_size_real'] * board_params['marker_size']
    
    return board_params

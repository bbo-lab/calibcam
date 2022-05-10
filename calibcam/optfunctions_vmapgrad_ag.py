# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

from . import camfunctions_jacfwd_ag as camfuncs_ag
from jax import numpy as np
from .helper_ag import rodrigues_as_rotmats


def obj_fcn(rvec_cams_1, rvec_cams_2, rvec_cams_3,
            tvec_cams_1, tvec_cams_2, tvec_cams_3,
            cam_matrices_1, cam_matrices_2, cam_matrices_3, cam_matrices_4, cam_matrices_5,
            cam_matrices_6, cam_matrices_7, cam_matrices_8, cam_matrices_9,
            ks_1, ks_2, ks_3, ks_4, ks_5,
            rvec_boards_1, rvec_boards_2, rvec_boards_3,
            tvec_boards_1, tvec_boards_2, tvec_boards_3,
            board_coord_3d_0_1, board_coord_3d_0_2, board_coord_3d_0_3,
            corners_1, corners_2):



    t_cam = np.array([tvec_cams_1, tvec_cams_2, tvec_cams_3])
    t_board = np.array([tvec_boards_1, tvec_boards_2, tvec_boards_3])

    cammat = np.array([
        [cam_matrices_1, cam_matrices_2, cam_matrices_3],
        [cam_matrices_4, cam_matrices_5, cam_matrices_6],
        [cam_matrices_7, cam_matrices_8, cam_matrices_9]
    ])

    board_coord = np.array([board_coord_3d_0_1, board_coord_3d_0_2, board_coord_3d_0_3])

    corner = np.array([corners_1, corners_2])

    # Make rotation mats
    R_cam = rodrigues_as_rotmats(np.array([rvec_cams_1, rvec_cams_2, rvec_cams_3]).T)
    R_board = rodrigues_as_rotmats(np.array([rvec_boards_1, rvec_boards_2, rvec_boards_3]).T)

    # To cam coordinates
    board_coord = board_coord@R_cam.T + t_cam

    board_coord = board_coord / board_coord[3]

    board_coord
    
    return boards_coords

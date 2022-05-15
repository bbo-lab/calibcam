# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

from jax import numpy as np
# import numpy as np
from calibcam import camfunctions_ag as camfuncs_ag
from calibcam.helper_ag import rodrigues_as_rotmats


def obj_fcn(rvec_cams_1, rvec_cams_2, rvec_cams_3,
            tvec_cams_1, tvec_cams_2, tvec_cams_3,
            cam_matrices_1, cam_matrices_2, cam_matrices_3, cam_matrices_4, cam_matrices_5,
            cam_matrices_6, cam_matrices_7, cam_matrices_8, cam_matrices_9,
            ks_1, ks_2, ks_3, ks_4, ks_5,
            rvec_boards_1, rvec_boards_2, rvec_boards_3,
            tvec_boards_1, tvec_boards_2, tvec_boards_3,
            board_coords_3d_0_1, board_coords_3d_0_2, board_coords_3d_0_3,
            corners_1, corners_2):

    rvecs_cams = np.moveaxis(np.array([rvec_cams_1, rvec_cams_2, rvec_cams_3]), 0, -1)
    tvecs_cams = np.moveaxis(np.array([tvec_cams_1, tvec_cams_2, tvec_cams_3]), 0, -1)

    rvecs_boards = np.moveaxis(np.array([rvec_boards_1, rvec_boards_2, rvec_boards_3]), 0, -1)
    tvecs_boards = np.moveaxis(np.array([tvec_boards_1, tvec_boards_2, tvec_boards_3]), 0, -1)

    board_coords_3d_0 = np.moveaxis(np.array([board_coords_3d_0_1, board_coords_3d_0_2, board_coords_3d_0_3]), 0, -1)

    corners = np.moveaxis(np.array([corners_1, corners_2]), 0, -1)

    cam_matrices = np.moveaxis(np.array([
        cam_matrices_1, cam_matrices_2, cam_matrices_3,
        cam_matrices_4, cam_matrices_5, cam_matrices_6,
        cam_matrices_7, cam_matrices_8, cam_matrices_9
    ]), 0, -1).reshape(corners.shape[0:-1] + (3, 3))

    ks = np.moveaxis(np.array([ks_1, ks_2, ks_3, ks_4, ks_5]), 0, -1)

    rotmats_cams = rodrigues_as_rotmats(rvecs_cams)

    rotmats_boards = rodrigues_as_rotmats(rvecs_boards)

    boards_coords = camfuncs_ag.map_ideal_board_to_world(board_coords_3d_0, rotmats_boards, tvecs_boards)
    boards_coords = camfuncs_ag.map_world_board_to_cams(boards_coords, rotmats_cams, tvecs_cams)
    boards_coords = camfuncs_ag.board_to_ideal_plane(boards_coords)
    boards_coords = camfuncs_ag.distort(boards_coords, ks)
    boards_coords = camfuncs_ag.ideal_to_sensor(boards_coords, cam_matrices)
    boards_coords = corners - boards_coords

    return boards_coords

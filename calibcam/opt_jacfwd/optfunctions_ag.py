# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

from calibcam import camfunctions_ag as camfuncs_ag
from calibcam.helper_ag import rodrigues_as_rotmats


def obj_fcn(rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards, boards_coords_3d_0, corners):
    rvecs_cams = rvecs_cams.reshape(-1, 3)
    tvecs_cams = tvecs_cams.reshape(-1, 3)
    cam_matrices = cam_matrices.reshape(-1, 3, 3)
    ks = ks.reshape(-1, 5)
    rvecs_boards = rvecs_boards.reshape(-1, 3)
    tvecs_boards = tvecs_boards.reshape(-1, 3)
    boards_coords_3d_0 = boards_coords_3d_0.reshape(rvecs_cams.shape[0], rvecs_boards.shape[0], 3, -1)
    corners = corners.reshape(rvecs_cams.shape[0], rvecs_boards.shape[0], 2, -1)

    rotmats_cams = rodrigues_as_rotmats(rvecs_cams)
    rotmats_boards = rodrigues_as_rotmats(rvecs_boards)

    boards_coords = camfuncs_ag.map_ideal_board_to_world(boards_coords_3d_0, rotmats_boards, tvecs_boards)
    boards_coords = camfuncs_ag.map_world_board_to_cams(boards_coords, rotmats_cams, tvecs_cams)
    boards_coords = camfuncs_ag.board_to_ideal_plane(boards_coords)
    boards_coords = camfuncs_ag.distort(boards_coords, ks)
    boards_coords = camfuncs_ag.ideal_to_sensor(boards_coords, cam_matrices)

    boards_coords = corners - boards_coords

    return boards_coords

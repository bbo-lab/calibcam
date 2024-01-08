# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

from calibcam import camfunctions_ag as camfuncs_ag
from calibcam.helper_ag import rodrigues_as_rotmats


def direction_distance(rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards, board_coords_3d_0, detected_boards_coords):
    rvecs_cams = rvecs_cams.reshape(-1, 3)
    tvecs_cams = tvecs_cams.reshape(-1, 3)
    cam_matrices = cam_matrices.reshape(-1, 3, 3)
    xis = xis.reshape(-1, 1)
    ks = ks.reshape(-1, 5)
    rvecs_boards = rvecs_boards.reshape(-1, 3)
    tvecs_boards = tvecs_boards.reshape(-1, 3)
    board_coords_3d_0 = board_coords_3d_0.reshape(-1, 3)
    corners = corners.reshape(rvecs_cams.shape[0], rvecs_boards.shape[0], -1, 2)

    rotmats_cams = rodrigues_as_rotmats(rvecs_cams)
    rotmats_boards = rodrigues_as_rotmats(rvecs_boards)

    optimized_boards_coords = camfuncs_ag.map_ideal_board_to_world(
        board_coords_3d_0.reshape((1, 1, corners.shape[2], 3)),
        rotmats_boards.reshape((1, corners.shape[1], 1, 3, 3)),
        tvecs_boards.reshape((1, corners.shape[1], 1, 3))
    )
    optimized_boards_coords = camfuncs_ag.map_world_board_to_cams(
        optimized_boards_coords,
        rotmats_cams.reshape((corners.shape[0], 1, 1, 3, 3)),
        tvecs_cams.reshape((corners.shape[0], 1, 1, 3))
    )
    optimized_boards_coords = camfuncs_ag.board_to_unit_sphere(optimized_boards_coords)

    detected_boards_coords = camfuncs_ag.unshift_camera(detected_boards_coords, xis.reshape((corners.shape[0], 1, 1, 1)))
    detected_boards_coords = camfuncs_ag.plane_to_ideal(detected_boards_coords)
    detected_boards_coords = camfuncs_ag.undistort(detected_boards_coords, ks.reshape((corners.shape[0], 1, 1, 5)))
    detected_boards_coords = camfuncs_ag.sensor_to_ideal(detected_boards_coords, cam_matrices.reshape(corners.shape[0], 1, 1, 3, 3))
    detected_boards_coords = camfuncs_ag.board_to_unit_sphere(detected_boards_coords)

    board_location_differences = detected_boards_coords - optimized_boards_coords
    #board_location_differences = np.sum(1 - np.dot(detected_boards_coords, optimized_boards_coords.T))

    return board_location_differences


def reprojection_distance(rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards, board_coords_3d_0, corners):
    rvecs_cams = rvecs_cams.reshape(-1, 3)
    tvecs_cams = tvecs_cams.reshape(-1, 3)
    cam_matrices = cam_matrices.reshape(-1, 3, 3)
    xis = xis.reshape(-1, 1)
    ks = ks.reshape(-1, 5)
    rvecs_boards = rvecs_boards.reshape(-1, 3)
    tvecs_boards = tvecs_boards.reshape(-1, 3)
    board_coords_3d_0 = board_coords_3d_0.reshape(-1, 3)
    corners = corners.reshape(rvecs_cams.shape[0], rvecs_boards.shape[0], -1, 2)

    rotmats_cams = rodrigues_as_rotmats(rvecs_cams)
    rotmats_boards = rodrigues_as_rotmats(rvecs_boards)

    boards_coords = camfuncs_ag.map_ideal_board_to_world(
        board_coords_3d_0.reshape((1, 1, corners.shape[2], 3)),
        rotmats_boards.reshape((1, corners.shape[1], 1, 3, 3)),
        tvecs_boards.reshape((1, corners.shape[1], 1, 3))
    )
    boards_coords = camfuncs_ag.map_world_board_to_cams(
        boards_coords,
        rotmats_cams.reshape((corners.shape[0], 1, 1, 3, 3)),
        tvecs_cams.reshape((corners.shape[0], 1, 1, 3))
    )
    boards_coords = camfuncs_ag.board_to_unit_sphere(boards_coords)
    boards_coords = camfuncs_ag.shift_camera(boards_coords, xis.reshape((corners.shape[0], 1, 1, 1)))

    boards_coords = camfuncs_ag.to_ideal_plane(boards_coords)
    boards_coords = camfuncs_ag.distort(boards_coords, ks.reshape((corners.shape[0], 1, 1, 5)))
    boards_coords = camfuncs_ag.ideal_to_sensor(
        boards_coords,
        cam_matrices.reshape(corners.shape[0], 1, 1, 3, 3)
    )

    boards_coords = corners - boards_coords

    return boards_coords


obj_fcn = reprojection_distance
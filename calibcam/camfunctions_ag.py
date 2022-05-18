# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

# These functions will properly broadcast as long as skipped dimensions are inserted as singular axes into input data.
#  E.g. rotmats_cams (n_cams, 3, 3) -> (n_cams, 1, 1, 3, 3)

import jax.numpy as np


def map_ideal_board_to_world(board_coords_3d_0, rotmats_boards, tvecs_boards):
    boards_coords_3d = np.einsum('...ij,...j->...i', rotmats_boards, board_coords_3d_0)  # TODO fix for other broadcasts
    boards_coords_3d = boards_coords_3d + tvecs_boards
    return boards_coords_3d


def map_world_board_to_cams(boards_coords_3d, rotmats_cams, tvecs_cams):
    boards_coords_3d = np.einsum('...ij,...j->...i', rotmats_cams, boards_coords_3d)  # TODO fix for other broadcasts
    boards_coords_3d = boards_coords_3d + tvecs_cams
    return boards_coords_3d


def board_to_ideal_plane(boards_coords_3d):
    # We add eps to the quotient to avoid division by 0 errors from non-tracked boards. Not exactly sure why
    # this happens?
    # TODO find more elegant way. This appears difficult since we cannot use a[a==0] = ... due to autograd limitations
    eps = np.finfo(np.float64).eps
    boards_coords_3d = np.concatenate((
        boards_coords_3d[..., (0,)] / (boards_coords_3d[..., (2,)] + eps),
        boards_coords_3d[..., (1,)] / (boards_coords_3d[..., (2,)] + eps),
        np.ones_like(boards_coords_3d[..., (2,)]),
    ), -1)
    return boards_coords_3d


def distort(boards_coords_ideal, ks):
    r2 = np.sum(boards_coords_ideal[..., 0:2] ** 2, axis=-1, keepdims=True)
    b = boards_coords_ideal

    def distort_dim(b_d):
        return (
                b_d * (1 + ks[..., (0,)] * r2 + ks[..., (1,)] * r2 ** 2 + ks[..., (2,)] * r2 ** 3) +
                2 * ks[..., (3,)] * b[..., (0,)] * b[..., (1,)] +
                ks[..., (4,)] * (r2 + 2 * b_d ** 2)
        )

    boards_coords_dist = np.concatenate((
        distort_dim(b[..., (0,)]),
        distort_dim(b[..., (1,)]),
        b[..., (2,)],
    ), -1)

    return boards_coords_dist


def ideal_to_sensor(boards_coords_dist, cam_matrices):
    boards_coords_dist = np.einsum('...ij,...j->...i', cam_matrices, boards_coords_dist)  # TODO fix for other broadcasts
    return boards_coords_dist[..., 0:2]

# import multiprocessing
# from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa
# from autograd import jacobian, elementwise_grad  # noqa
from jax import jacfwd as grad

from . import optimization
from . import optfunctions_vmapgrad_ag as opt_ag

import timeit


def obj_fcn_wrapper(vars_opt, args):
    corners = args['precalc']['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally  TODO find more efficient solution
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    boards_coords_3d_0 = args['precalc']['boards_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    n_cams, n_frames, _, n_bcorners = corners.shape

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    # Tested correct:
    # print(rvecs_cams[2])
    # print(tvecs_cams[2])
    # print(cam_matrices[2])
    # print(ks[2])

    rvecs_cams_tile = np.tile(rvecs_cams[:, np.newaxis, np.newaxis, np.newaxis, :],
                              (1, n_frames, 1, n_bcorners, 1)).reshape(-1, 3).T

    tvecs_cams_tile = np.tile(tvecs_cams[:, np.newaxis, np.newaxis, np.newaxis, :],
                              (1, n_frames, 1, n_bcorners, 1)).reshape(-1, 3).T
    cam_matrices_tile = np.tile(cam_matrices[:, np.newaxis, np.newaxis, np.newaxis, :],
                                (1, n_frames, 1, n_bcorners, 1)).reshape(-1, 9).T
    ks_tile = np.tile(ks[:, np.newaxis, np.newaxis, np.newaxis, :],
                      (1, n_frames, 1, n_bcorners, 1)).reshape(-1, 5).T

    rvecs_boards_tile = np.tile(rvecs_boards[np.newaxis, :, np.newaxis, np.newaxis, :],
                                (n_cams, 1, 1, n_bcorners, 1)).reshape(-1, 3).T
    tvecs_boards_tile = np.tile(tvecs_boards[np.newaxis, :, np.newaxis, np.newaxis, :],
                                (n_cams, 1, 1, n_bcorners, 1)).reshape(-1, 3).T

    residuals = np.array(opt_ag.obj_fcn(
        *rvecs_cams_tile,
        *tvecs_cams_tile,
        *cam_matrices_tile,
        *ks_tile,
        *rvecs_boards_tile,
        *tvecs_boards_tile,
        *boards_coords_3d_0[0, 0],
        corners.ravel()
    ))

    # Residuals of untracked corners are invalid
    residuals[corners_mask] = 0
    print(np.unravel_index(np.argmax(np.abs(residuals)), shape=residuals.shape))
    print(np.max(np.abs(residuals)))
    return residuals.ravel()


def obj_fcn_jacobian_wrapper(vars_opt, args):
    corners = args['precalc']['corners']
    corners_mask = np.isnan(corners)
    boards_coords_3d_0 = args['precalc']['boards_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    # All zero rotvec causes division by 0 problems. Usually, this usually does not matter since the ref cam
    # orientation is not part of the free variables, but we apply this fix to avoid misleading errors
    rvecs_cams = rvecs_cams.copy()
    for i_cam in range(corners.shape[0]):
        if np.all(rvecs_cams[i_cam] == 0):
            rvecs_cams[i_cam][:] = np.finfo(np.float16).eps

    derivatives = args['precalc']['derivatives']

    obj_fcn_jacobian = []
    return obj_fcn_jacobian[:, args['mask_opt']]


def get_obj_fcn_derivatives():
    return [grad(opt_ag.obj_fcn, i_var) for i_var in range(3 + 3 + 9 + 5)]

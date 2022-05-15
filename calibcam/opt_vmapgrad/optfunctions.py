# import multiprocessing
# from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa
# from autograd import jacobian, elementwise_grad  # noqa
from jax import grad, vmap, jit

from calibcam import optimization
from calibcam.opt_vmapgrad import optfunctions_ag as opt_ag


def obj_fcn_wrapper(vars_opt, args):
    corners = args['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally  TODO find more efficient solution
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    n_cams, n_frames, n_bcorners, _ = corners.shape

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    # Tested correct:
    # print(rvecs_cams[2])
    # print(tvecs_cams[2])
    # print(cam_matrices[2])
    # print(ks[2])

    rvecs_cams_tile = np.tile(rvecs_cams[:, np.newaxis, np.newaxis, :],
                              (1, n_frames, n_bcorners, 1))
    tvecs_cams_tile = np.tile(tvecs_cams[:, np.newaxis, np.newaxis, :],
                              (1, n_frames, n_bcorners, 1))
    cam_matrices_tile = np.tile(cam_matrices[:, np.newaxis, np.newaxis, :, :],
                                (1, n_frames, n_bcorners, 1, 1))
    cam_matrices_tile = cam_matrices_tile.reshape(cam_matrices_tile.shape[0:-2] + (-1,))
    ks_tile = np.tile(ks[:, np.newaxis, np.newaxis, :],
                      (1, n_frames, n_bcorners, 1))
    rvecs_boards_tile = np.tile(rvecs_boards[np.newaxis, :, np.newaxis, :],
                                (n_cams, 1, n_bcorners, 1))
    tvecs_boards_tile = np.tile(tvecs_boards[np.newaxis, :, np.newaxis, :],
                                (n_cams, 1, n_bcorners, 1))
    board_coords_3d_0_tile = np.tile(board_coords_3d_0[np.newaxis, np.newaxis, :, :],
                                     (n_cams, n_frames, 1, 1))

    #residuals = np.array(opt_ag.obj_fcn(  # This seems slower by a factor of ~30
    residuals = np.array(args['precalc']['obj_fun'](
        *np.moveaxis(rvecs_cams_tile, -1, 0).reshape(3, -1),
        *np.moveaxis(tvecs_cams_tile, -1, 0).reshape(3, -1),
        *np.moveaxis(cam_matrices_tile, -1, 0).reshape(9, -1),
        *np.moveaxis(ks_tile, -1, 0).reshape(5, -1),
        *np.moveaxis(rvecs_boards_tile, -1, 0).reshape(3, -1),
        *np.moveaxis(tvecs_boards_tile, -1, 0).reshape(3, -1),
        *np.moveaxis(board_coords_3d_0_tile, -1, 0).reshape(3, -1),  # TODO: I think it should be possible to do this without tiling this ...
        *np.moveaxis(corners, -1, 0).reshape(2, -1),
    ))

    residuals = residuals.reshape(corners.shape)

    # Residuals of untracked corners are invalid
    residuals[corners_mask] = 0

    if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
        print("Found nans or infs in objective function result, aborting")
        print(np.where(np.isnan(residuals)))
        print(np.where(np.isinf(residuals)))
        exit()

    return residuals.ravel()


def obj_fcn_jacobian_wrapper(vars_opt, args):
    corners = args['corners']
    corners_mask = np.isnan(corners)
    board_coords_3d_0 = args['boards_coords_3d_0']

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


def get_precalc():
    precalc = {
        'obj_fun': jit(vmap(opt_ag.obj_fcn, in_axes=(
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        ), out_axes=0)),
        'grads': [
            jit(grad(opt_ag.obj_fcn, i_var))
            for i_var in range(3 + 3 + 9 + 5 + 3 + 3)
        ]
    }
    return precalc

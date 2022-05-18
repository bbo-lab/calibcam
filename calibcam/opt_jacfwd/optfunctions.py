# import multiprocessing
# from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa
# from autograd import jacobian, elementwise_grad  # noqa
from jax import jit, jacfwd as jacobian  # jacfwd is recommended for 'tall' Jacobians, jacrev for 'wide'

from calibcam import optimization
from calibcam.opt_jacfwd import optfunctions_ag as opt_ag

import timeit


def get_precalc():
    return {
        'objfunc': jit(opt_ag.obj_fcn),
        'jacobians': [jit(jacobian(opt_ag.obj_fcn, i_var)) for i_var in range(6)]
    }


def obj_fcn_wrapper(vars_opt, args):
    corners = args['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    residuals = np.array(args['precalc']['objfunc'](
        rvecs_cams,
        tvecs_cams,
        cam_matrices,
        ks,
        rvecs_boards,
        tvecs_boards,
        board_coords_3d_0,
        corners
    ))

    # Residuals of untracked corners are invalid
    residuals[corners_mask] = 0
    return residuals.ravel()


def obj_fcn_jacobian_wrapper(vars_opt, args):
    obj_fcn_jacobian = obj_fcn_jacobian_wrapper_full(vars_opt, args)
    return obj_fcn_jacobian


def obj_fcn_jacobian_wrapper_full(vars_opt, args):
    corners = args['corners']
    corners_mask = np.isnan(corners)
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    obj_fcn_jacobian = [
        np.array(args['precalc']['jacobians'][i_var](
            rvecs_cams,
            tvecs_cams,
            cam_matrices,
            ks,
            rvecs_boards,
            tvecs_boards,
            board_coords_3d_0,
            corners
        ).reshape(corners.shape + (-1,))  # Ravel over input dimensions
                 ) for i_var in range(6)]

    # TODO: Find out why Jacobian of coord_cam pose values is nan ...
    obj_fcn_jacobian[0][args['coord_cam']] = 0

    # Concatenate along input dimensions
    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # Set undetected corners to 0
    obj_fcn_jacobian[corners_mask, :] = 0

    # Ravel along residual dimensions
    obj_fcn_jacobian = obj_fcn_jacobian.reshape(corners_mask.size, args['mask_opt'].size)

    # Dump non-free variables
    obj_fcn_jacobian = obj_fcn_jacobian[:, args['mask_opt']]

    return obj_fcn_jacobian


# Given that the jacobian is very sparse, an alternative solution would be to calculate the respective submatricies.
# Given the performance of obj_fcn_jacobian_wrapper_full and the very reasonable memory footprint, this seems utterly
# unnecessary at the moment.
def obj_fcn_jacobian_wrapper_sparse(vars_opt, args):
    pass

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
        'jacobians': [jit(jacobian(opt_ag.obj_fcn, i_var)) for i_var in range(6)]
    }


def obj_fcn_wrapper(vars_opt, args):
    corners = args['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally  TODO find more efficient solution
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    # Tested correct:
    # print(rvecs_cams[2])
    # print(tvecs_cams[2])
    # print(cam_matrices[2])
    # print(ks[2])

    residuals = np.array(opt_ag.obj_fcn(
        rvecs_cams.ravel(),
        tvecs_cams.ravel(),
        cam_matrices.ravel(),
        ks.ravel(),
        rvecs_boards.ravel(),
        tvecs_boards.ravel(),
        board_coords_3d_0.ravel(),
        corners.ravel()
    ))

    # Residuals of untracked corners are invalid
    residuals[corners_mask] = 0
    return residuals.ravel()


def obj_fcn_jacobian_wrapper(vars_opt, args):
    return obj_fcn_jacobian_wrapper_full(vars_opt, args)


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
            rvecs_cams.ravel(),
            tvecs_cams.ravel(),
            cam_matrices.ravel(),
            ks.ravel(),
            rvecs_boards.ravel(),
            tvecs_boards.ravel(),
            board_coords_3d_0.ravel(),
            corners.ravel()
        )) for i_var in range(6)]

    obj_fcn_jacobian = np.concatenate(
        obj_fcn_jacobian,
        axis=4)

    # TODO: Find out why Jacobian of coord_cam pose values is nan ...
    obj_fcn_jacobian[args['coord_cam']] = 0

    # Set undetected corners to 0
    obj_fcn_jacobian[corners_mask, :] = 0
    return obj_fcn_jacobian.reshape(corners_mask.size, args['mask_opt'].size)[:, args['mask_opt']]


def obj_fcn_jacobian_wrapper_sparse(vars_opt, args):
    corners = args['corners']
    corners_mask = np.isnan(corners)
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams)

    jacobians = args['jacobians']

    tic = timeit.default_timer()

    obj_fcn_jacobian_cam_pose, obj_fcn_jacobian_cam_mat, obj_fcn_jacobian_cam_k = \
        calc_cam_jacobian(jacobians,
                          rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards,
                          board_coords_3d_0, corners)
    obj_fcn_jacobian_pose = \
        calc_pose_jacobian(jacobians,
                           rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards,
                           board_coords_3d_0, corners)

    obj_fcn_jacobian = np.concatenate((
        obj_fcn_jacobian_cam_pose.reshape(corners.shape + (-1,)),
        obj_fcn_jacobian_cam_mat.reshape(corners.shape + (-1,)),
        obj_fcn_jacobian_cam_k.reshape(corners.shape + (-1,)),
        obj_fcn_jacobian_pose.reshape(corners.shape + (-1,)),
    ), corners.ndim)

    print(timeit.default_timer() - tic)

    # Residuals of untracked corners are invalid
    obj_fcn_jacobian[corners_mask] = 0

    # Return section of free variables
    obj_fcn_jacobian = obj_fcn_jacobian.reshape(np.prod(corners_mask.shape), -1)
    return obj_fcn_jacobian[:, args['mask_opt']]


def calc_cam_jacobian(jacobians, rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards,
                      board_coords_3d_0, corners):
    # n_cam_param_list = np.array([3, 3, 9, 5])

    obj_fcn_jacobian_cam_pose = np.zeros(corners.shape + (2, corners.shape[0], 3), dtype=np.float16)
    offset = 0
    for i_cam in range(corners.shape[0]):
        jacs = [calc_jacobian(jacobians[offset + i_param], (
            # jacs = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / 2) - 2))(
            #     delayed(calc_jacobian)(jacobians[i_param], (
            rvecs_cams[i_cam].ravel(),
            tvecs_cams[i_cam].ravel(),
            cam_matrices[i_cam].ravel(),
            ks[i_cam].ravel(),
            rvecs_boards[0:10].ravel(),
            tvecs_boards[0:10].ravel(),
            board_coords_3d_0[i_cam, 0:10].ravel(),
            corners[i_cam, 0:10].ravel()
        ))
                for i_param in range(2)]

        for i_param, j in enumerate(jacs):
            if np.any(np.isnan(j)):
                print("In cam")
                print(j)
                print(j.shape)
                print(i_cam)
                print(i_param)
                exit()
            obj_fcn_jacobian_cam_pose[i_cam, 0:10, :, :, i_param, i_cam, :] = j

    offset = offset + 2
    obj_fcn_jacobian_cam_mat = np.zeros(corners.shape + (1, corners.shape[0], 9), dtype=np.float16)
    for i_cam in range(corners.shape[0]):
        jacs = [calc_jacobian(jacobians[offset + i_param], (
            # jacs = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / 2) - 2))(
            #     delayed(calc_jacobian)(jacobians[i_param], (
            rvecs_cams[i_cam].ravel(),
            tvecs_cams[i_cam].ravel(),
            cam_matrices[i_cam].ravel(),
            ks[i_cam].ravel(),
            rvecs_boards[0:10].ravel(),
            tvecs_boards[0:10].ravel(),
            board_coords_3d_0[i_cam, 0:10].ravel(),
            corners[i_cam, 0:10].ravel()
        ))
                for i_param in range(1)]

        for i_param, j in enumerate(jacs):
            if np.any(np.isnan(j)):
                print("In cam")
                print(j)
                print(j.shape)
                print(i_cam)
                print(i_param)
                exit()
            obj_fcn_jacobian_cam_mat[i_cam, 0:10, :, :, i_param, i_cam, :] = j

    offset = offset + 1
    obj_fcn_jacobian_cam_k = np.zeros(corners.shape + (1, corners.shape[0], 5), dtype=np.float16)
    for i_cam in range(corners.shape[0]):
        jacs = [calc_jacobian(jacobians[offset + i_param], (
            # jacs = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / 2) - 2))(
            #     delayed(calc_jacobian)(jacobians[i_param], (
            rvecs_cams[i_cam].ravel(),
            tvecs_cams[i_cam].ravel(),
            cam_matrices[i_cam].ravel(),
            ks[i_cam].ravel(),
            rvecs_boards[0:10].ravel(),
            tvecs_boards[0:10].ravel(),
            board_coords_3d_0[i_cam, 0:10].ravel(),
            corners[i_cam, 0:10].ravel()
        ))
                for i_param in range(1)]

        for i_param, j in enumerate(jacs):
            if np.any(np.isnan(j)):
                print("In cam")
                print(j)
                print(j.shape)
                print(i_cam)
                print(i_param)
                exit()
            obj_fcn_jacobian_cam_k[i_cam, 0:10, :, :, i_param, i_cam, :] = j

    return obj_fcn_jacobian_cam_pose, obj_fcn_jacobian_cam_mat, obj_fcn_jacobian_cam_k


def calc_pose_jacobian(jacobians, rvecs_cams, tvecs_cams, cam_matrices, ks, rvecs_boards, tvecs_boards,
                       board_coords_3d_0, corners):
    n_cam_param_list = np.array([3, 3, 9, 5])
    n_pose_param_list = np.array([3, 3])

    obj_fcn_jacobian_pose = np.zeros(corners.shape + (n_pose_param_list.size, corners.shape[1], 3), dtype=np.float16)
    for i_pose in range(corners.shape[1]):
        jacs = [calc_jacobian(jacobians[i_param + n_cam_param_list.size], (
            # jacs = Parallel(n_jobs=int(np.floor(multiprocessing.cpu_count() / 2) - 2))(
            #     delayed(calc_jacobian)(jacobians[i_param + n_cam_param_list.size], (
            rvecs_cams.ravel(),
            tvecs_cams.ravel(),
            cam_matrices.ravel(),
            ks.ravel(),
            rvecs_boards[i_pose].ravel(),
            tvecs_boards[i_pose].ravel(),
            board_coords_3d_0[:, i_pose].ravel(),
            corners[:, i_pose].ravel()
        ))
                for i_param in range(len(n_pose_param_list))]

        for i_param, (j, len_param) in enumerate(zip(jacs, n_pose_param_list)):
            if np.any(np.isnan(j)):
                print("In pose")
                print(j)
                print(j.shape)
                print(i_pose)
                print(i_param)
                exit()
            obj_fcn_jacobian_pose[:, i_pose, :, :, i_param, i_pose, :] = j[:, 0]

    return obj_fcn_jacobian_pose


def calc_jacobian(jac, parameters):
    return jac(*parameters)

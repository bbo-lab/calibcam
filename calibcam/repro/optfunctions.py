# import multiprocessing
# from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.transform import Rotation as R  # noqa
from scipy.sparse import csr_matrix, hstack as sparse_hstack

from jax import jit, jacfwd as jacobian  # jacfwd is recommended for 'tall' Jacobians, jacrev for 'wide'


from calibcam import optimization
from calibcam.repro import optfunctions_ag as opt_ag

import timeit


def get_precalc(opts):
    return {
        'objfunc': jit(opt_ag.obj_fcn, backend=opts["jax_backend"]),
        'jacobians': [jit(jacobian(opt_ag.obj_fcn, i_var), backend=opts["jax_backend"]) for i_var in range(7)]
    }


def obj_fcn_wrapper(vars_opt, args):
    corners = args['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams, n_boards = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams, n_boards)

    residuals = np.array(args['precalc']['objfunc'](
        rvecs_cams,
        tvecs_cams,
        cam_matrices,
        xis,
        ks,
        rvecs_boards,
        tvecs_boards,
        board_coords_3d_0,
        corners
    ))  # Make np array since JAX arrays are immutable.

    # Residuals of untracked corners are invalid
    residuals[corners_mask] = 0
    return residuals.ravel()


def obj_fcn_jacobian_wrapper(vars_opt, args):
    obj_fcn_jacobian = obj_fcn_jacobian_wrapper_sparse(vars_opt, args)
    return obj_fcn_jacobian


def obj_fcn_jacobian_wrapper_full(vars_opt, args):
    corners = args['corners']
    corners_mask = np.isnan(corners)
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams, n_boards = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams, n_boards)
    rvecs_cams_mask, tvecs_cams_mask, cam_matrices_mask, xis_mask, ks_mask, rvecs_boards_mask, tvecs_boards_mask = \
        optimization.unravel_vars_full(args['mask_opt'], n_cams, n_boards)

    var_masks = [rvecs_cams_mask, tvecs_cams_mask, cam_matrices_mask, xis_mask, ks_mask, rvecs_boards_mask,
                 tvecs_boards_mask]

    obj_fcn_jacobian = [
        np.array(
            args['precalc']['jacobians'][i_var](
                rvecs_cams,
                tvecs_cams,
                cam_matrices,
                xis,
                ks,
                rvecs_boards,
                tvecs_boards,
                board_coords_3d_0,
                corners
            ).reshape(corners.shape + (-1,))  # Ravel over input dimensions
        ) if np.any(var_masks[i_var]) else np.zeros(shape=corners.shape + (var_masks[i_var].size,))
        for i_var in range(7)
    ]

    # TODO: Find out why Jacobian of coord_cam pose values is nan ...
    # obj_fcn_jacobian[0][args['coord_cam']] = 0

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


def obj_fcn_jacobian_wrapper_sparse(vars_opt, args) -> np.ndarray:
    corners = args['corners'].copy()  # copy is necessary since this is a reference, so further down, nans
    # will be replaced with 0 globally
    corners_mask = np.isnan(corners)
    corners[corners_mask] = 0
    board_coords_3d_0 = args['board_coords_3d_0']

    # Fill vars_full from initialization with vars_opts
    vars_full, n_cams, n_boards = optimization.make_vars_full(vars_opt, args)

    # Unravel inputs. Note that calibs, board_coords_3d and their representations in args are changed in this function
    # and the return is in fact unnecessary!
    rvecs_cams, tvecs_cams, cam_matrices, xis, ks, rvecs_boards, tvecs_boards = \
        optimization.unravel_vars_full(vars_full, n_cams, n_boards)

    jacs = args['precalc']['jacobians']

    sm, sk, sn, sd = corners.shape[:4]

    rm = np.arange(sm)
    rk = np.arange(sk)
    rn = np.arange(sn)
    rd = np.arange(sd)

    rm_b = rm[:, None, None, None]
    # rk_b = rk[None, :, None, None]
    rn_b = rn[None, None, :, None]
    rd_b = rd[None, None, None, :]

    result_size = (sm * sk * sn * sd)

    edges = optimization.get_var_edges(sm, sk)
    jacs_sparse = []

    # Camera parameters
    # cam_rotvecs, cam_ts, A, xi, k
    # TODO: This can further be sparsified by a factor of n_cams
    for jac_idx in range(5):
        j = np.array(jacs[jac_idx](
            rvecs_cams,
            tvecs_cams,
            cam_matrices,
            xis,
            ks,
            rvecs_boards,
            tvecs_boards,
            board_coords_3d_0,
            corners
        ))
        j[np.isnan(j)] = 0  # Necessary?
        # Set undetected corners to 0
        j[corners_mask, :] = 0
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
        jacs_sparse.append(csr_matrix(j.reshape(result_size, len(mask))[:, mask]))

    # board positions, orientations
    for jac_idx in range(5, 7):
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]

        if not np.any(mask):
            jacs_sparse.append(csr_matrix(([], ([], [])), shape=(result_size, 0)))
        else:
            element_length = 3
            n_cols = np.sum(mask)
            next_col = 0
            rows = np.empty((sm * sn * sd * n_cols,))
            cols = np.empty((sm * sn * sd * n_cols,))
            data = np.empty((sm * sn * sd * n_cols,))
            data_ptr = 0

            row_indices_base = (rm_b * (sk * sn * sd) +
                                rn_b * sd +
                                rd_b).reshape(-1)

            for i_k in rk:
                start_idx = i_k * element_length
                element_mask = mask[start_idx:(start_idx + element_length)]

                j = np.array(jacs[jac_idx](
                    rvecs_cams,
                    tvecs_cams,
                    cam_matrices,
                    xis,
                    ks,
                    rvecs_boards[(i_k,),],
                    tvecs_boards[(i_k,),],
                    board_coords_3d_0,
                    corners[:, (i_k,),]
                ))
                # Set undetected corners to 0
                j[corners_mask[:, (i_k,),], :] = 0
                j = j[..., element_mask]

                en_cols = np.sum(element_mask)
                row_indices = row_indices_base + i_k * (sn * sd)
                column_indices = np.tile(np.arange(en_cols) + next_col, row_indices.size)
                row_indices = np.repeat(row_indices, np.sum(element_mask))

                data_size = column_indices.size
                rows[data_ptr:data_ptr + data_size] = row_indices
                cols[data_ptr:data_ptr + data_size] = column_indices
                data[data_ptr:data_ptr + data_size] = j.ravel()
                data_ptr += data_size

                next_col += np.sum(element_mask)

            data[np.isnan(data)] = 0  # Necessary?
            jacs_sparse.append(csr_matrix((data, (rows, cols)), shape=(result_size, n_cols)))

    # Others
    for jac_idx in range(8, len(jacs)):
        j = np.array(jacs[jac_idx](
            rvecs_cams,
            tvecs_cams,
            cam_matrices,
            xis,
            ks,
            rvecs_boards(),
            tvecs_boards,
            board_coords_3d_0,
            corners
        ).reshape(result_size, -1))
        j[np.isnan(j)] = 0  # Necessary?
        # Set undetected corners to 0
        j[corners_mask, :] = 0
        mask = args['mask_opt'][edges[jac_idx]:edges[jac_idx + 1]]
        jacs_sparse.append(csr_matrix(j[:, mask]))

    jac_sparse = sparse_hstack(jacs_sparse)
    return jac_sparse

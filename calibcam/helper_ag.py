# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

import jax.numpy as np

# Converts array of rotation vectors to array of rotation matrices
def rodrigues_as_rotmats(r):
    r_shape = r.shape
    r = r.reshape(-1, 3)

    theta = np.power(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2, 0.5)
    u = r / (theta + -np.abs(np.sign(theta)) + 1).reshape(-1, 1)

    # row 1
    rotmat_00 = np.cos(theta) + u[:, 0] ** 2 * (1 - np.cos(theta))
    rotmat_01 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) - u[:, 2] * np.sin(theta)
    rotmat_02 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) + u[:, 1] * np.sin(theta)
    rotmat_0 = np.concatenate([rotmat_00.reshape(-1, 1, 1),
                               rotmat_01.reshape(-1, 1, 1),
                               rotmat_02.reshape(-1, 1, 1)], 2)

    # row 2
    rotmat_10 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) + u[:, 2] * np.sin(theta)
    rotmat_11 = np.cos(theta) + u[:, 1] ** 2 * (1 - np.cos(theta))
    rotmat_12 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) - u[:, 0] * np.sin(theta)
    rotmat_1 = np.concatenate([rotmat_10.reshape(-1, 1, 1),
                               rotmat_11.reshape(-1, 1, 1),
                               rotmat_12.reshape(-1, 1, 1)], 2)

    # row 3
    rotmat_20 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) - u[:, 1] * np.sin(theta)
    rotmat_21 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) + u[:, 0] * np.sin(theta)
    rotmat_22 = np.cos(theta) + u[:, 2] ** 2 * (1 - np.cos(theta))
    rotmat_2 = np.concatenate([rotmat_20.reshape(-1, 1, 1),
                               rotmat_21.reshape(-1, 1, 1),
                               rotmat_22.reshape(-1, 1, 1)], 2)

    rotmat = np.concatenate([rotmat_0,
                             rotmat_1,
                             rotmat_2], 1)

    rotmat = rotmat.reshape(r_shape[0:-1] + (3, 3))

    return rotmat

# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

# import autograd.numpy as np
import jax.numpy as np

# Converts array of rotation vectors to array of rotation matrices
def rodrigues_as_rotmats(r):
    # noinspection PyUnresolvedReferences
    theta = np.power(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2, 0.5)
    # noinspection PyUnresolvedReferences
    u = r / (theta + -np.abs(np.sign(theta)) + 1).reshape(-1, 1)

    # row 1
    # noinspection PyUnresolvedReferences
    rotmat_00 = np.cos(theta) + u[:, 0] ** 2 * (1 - np.cos(theta))
    # noinspection PyUnresolvedReferences
    rotmat_01 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) - u[:, 2] * np.sin(theta)
    # noinspection PyUnresolvedReferences
    rotmat_02 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) + u[:, 1] * np.sin(theta)
    rotmat_0 = np.concatenate([rotmat_00.reshape(-1, 1, 1),
                               rotmat_01.reshape(-1, 1, 1),
                               rotmat_02.reshape(-1, 1, 1)], 2)

    # row 2
    # noinspection PyUnresolvedReferences
    rotmat_10 = u[:, 0] * u[:, 1] * (1 - np.cos(theta)) + u[:, 2] * np.sin(theta)
    # noinspection PyUnresolvedReferences
    rotmat_11 = np.cos(theta) + u[:, 1] ** 2 * (1 - np.cos(theta))
    # noinspection PyUnresolvedReferences
    rotmat_12 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) - u[:, 0] * np.sin(theta)
    rotmat_1 = np.concatenate([rotmat_10.reshape(-1, 1, 1),
                               rotmat_11.reshape(-1, 1, 1),
                               rotmat_12.reshape(-1, 1, 1)], 2)

    # row 3
    # noinspection PyUnresolvedReferences
    rotmat_20 = u[:, 0] * u[:, 2] * (1 - np.cos(theta)) - u[:, 1] * np.sin(theta)
    # noinspection PyUnresolvedReferences
    rotmat_21 = u[:, 1] * u[:, 2] * (1 - np.cos(theta)) + u[:, 0] * np.sin(theta)
    # noinspection PyUnresolvedReferences
    rotmat_22 = np.cos(theta) + u[:, 2] ** 2 * (1 - np.cos(theta))
    rotmat_2 = np.concatenate([rotmat_20.reshape(-1, 1, 1),
                               rotmat_21.reshape(-1, 1, 1),
                               rotmat_22.reshape(-1, 1, 1)], 2)

    rotmat = np.concatenate([rotmat_0,
                             rotmat_1,
                             rotmat_2], 1)

    return rotmat

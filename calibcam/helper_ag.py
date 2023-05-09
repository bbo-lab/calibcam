# Functions in this file will be subject to autograd and need to be written accordingly
# - Do not import functions that are not compatible with autograd
# - Autograd numpy used here
# - Do not use asarray, as it does not seem to differentiable
# - Do not use for loops
# - Do not use array assignment, e.g. A[i,j] = x

import jax
import jax.numpy as jnp

from numpy import errstate

# Converts array of rotation vectors to array of rotation matrices
def rodrigues_as_rotmats(rot_vecs):
    rot_vecs_shape = rot_vecs.shape
    rot_vecs = rot_vecs.reshape(-1, 3)

    rot_mats = jax.vmap(rodrigues_as_rotmat)(rot_vecs)

    rot_mats = rot_mats.reshape(rot_vecs_shape[0:-1] + (3, 3))
    return rot_mats

    # rot_vecs_shape = rot_vecs.shape
    # rot_vecs = rot_vecs.reshape(-1, 3)
    #
    # norms = jnp.linalg.norm(rot_vecs, axis=1)
    # identity_mask = (norms == 0)
    # identity_matrices = jnp.tile(jnp.eye(3), (len(rot_vecs), 1, 1))
    # k = rot_vecs / norms.reshape((-1, 1))
    # K = jnp.stack((jnp.zeros_like(k[:, 0]), -k[:, 2], k[:, 1], k[:, 2], jnp.zeros_like(k[:, 0]), -k[:, 0], -k[:, 1],
    #                k[:, 0], jnp.zeros_like(k[:, 0])), axis=1).reshape((-1, 3, 3))
    # R = jnp.tile(jnp.eye(3), (len(rot_vecs), 1, 1)) + jnp.sin(norms).reshape((-1, 1, 1)) * K + (
    #             1 - jnp.cos(norms)).reshape((-1, 1, 1)) * K @ K
    #
    # rotmat = jnp.where(identity_mask.reshape((-1, 1, 1)), identity_matrices, R)
    #
    # rotmat = rotmat.reshape(rot_vecs_shape[0:-1] + (3, 3))
    #
    # return rotmat

def rodrigues_as_rotmat(rotvec):
    theta = jnp.linalg.norm(rotvec)

    def rodrigues_as_rotmat_nonzero(rotvec):
        theta = jnp.linalg.norm(rotvec)
        theta = jnp.where(theta == 0, 1, theta)
        k = rotvec / theta
        K = jnp.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * jnp.dot(K, K)
        return R

    def rodrigues_as_rotmat_zero(rotvec):
        return jnp.full((3,3), 1.23456)

    return jax.lax.cond(theta == 0, rodrigues_as_rotmat_nonzero, rodrigues_as_rotmat_nonzero, rotvec)
# modified from: https://github.com/bentaps/strupnet/blob/main/strupnet/utils.py
import jax
import jax.numpy as jnp

from functools import partial

def get_pq(x, dim):
    """Returns p, q from x of shape (nbatch, 2*dim)"""
    return x[..., :dim], x[..., dim:]


def get_x(p, q):
    """Returns x from p, q of shape (nbatch, dim)"""
    return jnp.concat([p, q], axis=-1)

@partial(jax.jit, static_argnames=['dim'])
def canonical_symplectic_matrix(dim):
    """Returns the canonical symplectic matrix of dimension 2*dim."""
    J = jnp.zeros((2 * dim, 2 * dim))
    J.at[:dim, dim:].set(-jnp.eye(dim))
    J.at[dim:, :dim].set(jnp.eye(dim))
    return J

@partial(jax.jit, static_argnames=['dim'])
def symplectic_matrix_transformation_2d(dim, w, i):
    """Used for the 2D symplectic volume-preserving substeps. Takes an n-dim vector w and an index i, and returns [0, ..., w[i+1], -w[i], ..., 0]"""
    out = jnp.zeros(dim)
    out[i] = w[i + 1]
    out[i + 1] = -w[i]
    return out
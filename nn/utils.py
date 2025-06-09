import jax.numpy as jnp


def get_pq(x, dim):
    """Returns p, q from x of shape (nbatch, 2*dim)"""
    return x[..., :dim], x[..., dim:]


def get_x(p, q):
    """Returns x from p, q of shape (nbatch, dim)"""
    return jnp.concat([p, q], axis=-1)
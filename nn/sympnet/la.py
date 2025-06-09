# Modified from: https://github.com/bentaps/strupnet/blob/main/strupnet/layers/LA.py
import jax
import jax.numpy as jnp
import jax.random as jnr
from flax import nnx

from ..utils import get_pq, get_x

class LinearLayer(nnx.Module):
    """Linear symplectic layer."""

    def __init__(self, dim, sublayers, *, rngs: nnx.Rngs):
        self.dim, self.sublayers = dim, sublayers

        key = rngs.params()

        self.S_ = nnx.Param(jnr.uniform(key, (sublayers, dim)) * 0.01)
        self.bp = nnx.Param(jnp.zeros(dim))
        self.bq = nnx.Param(jnp.zeros(dim))

    def __call__(self, x, h):
        x = jax.lax.scan(lambda carry, s: self.apply_sublayer(carry, s, h), (x, 0), self.S_)

        return x

    def apply_sublayer(self, carry, s, h):
        x, i = carry
        p, q = get_pq(x, self.dim)

        odd_i = lambda p, q: (p + q @ (s+s.T) * h, q)
        even_i = lambda p, q: (p, q + p @ (s+s.T) * h)

        p, q = jax.lax.cond(i % 2, odd_i, even_i, p, q)

        return (get_x(p, q), i + 1), None

class ActivationLayer(nnx.Module):
    """Activation symplectic layer."""

    def __init__(self, dim, mode, *, rngs: nnx.Rngs):
        self.dim, self.mode = dim, mode

        self.activation = nnx.tanh

        key = rngs.params()
        self.a = nnx.Param(jnr.uniform(key, (dim)) * 0.01)
    
    def __call__(self, x, h):
        p, q = get_pq(x, self.dim)

        odd_mode = lambda p, q: (p + self.activation(q) * self.a * h, q)
        even_mode = lambda p, q: (p, q + self.activation(p) * self.a * h)

        p, q = jax.lax.cond(self.mode == "odd", odd_mode, even_mode, p, q)

        return get_x(p, q)

class LA_Layer(nnx.Module):
    """LA layer """

    def __init__(self, dim, sublayers, mode="odd", *, rngs: nnx.Rngs):
        self.dim, self.sublayers, self.mode = dim, sublayers, mode

        self.linear_layer = LinearLayer(self.dim, self.sublayers, rngs=rngs)
        self.activation_layer = ActivationLayer(self.dim, self.mode, rngs=rngs)

    def __call__(self, x, h, reverse=False):
        normal_apply = lambda x: self.activation_layer(self.linear_layer(x, h), h)
        reverse_apply = lambda x: self.linear_layer(self.activation_layer(x, h), h)

        x = jax.lax.cond(reverse, reverse_apply, normal_apply, x)

        return x

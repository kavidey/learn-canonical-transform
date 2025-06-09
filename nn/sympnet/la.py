# Modified from: https://github.com/bentaps/strupnet/blob/main/strupnet/layers/LA.py
import jax
import jax.numpy as jnp
import jax.random as jnr
from flax import nnx

from ..utils import get_pq, get_x

class LinearLayer(nnx.Module):
    """Linear symplectic layer."""

    def __init__(self, dim, *, rngs: nnx.Rngs):
        self.dim = dim

        self.S_ = nnx.Param(jnr.uniform(rngs.params(), (2, dim)) * 0.01)
        self.bp = nnx.Param(jnp.zeros(dim))
        self.bq = nnx.Param(jnp.zeros(dim))

    def __call__(self, x, h):
        p, q = get_pq(x, self.dim)
        p = p + (q @ (self.S_[0]+self.S_[0].T))[..., None] * h
        q = q + (p @ (self.S_[1]+self.S_[1].T))[..., None] * h

        p = p + self.bp * h
        q = q + self.bq * h

        return x

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

        p, q = jax.lax.cond(self.mode % 2, odd_mode, even_mode, p, q)

        return get_x(p, q)

class LA_Layer(nnx.Module):
    """LA layer """

    def __init__(self, dim, *, rngs: nnx.Rngs):
        self.dim = dim

        self.linear1 = LinearLayer(self.dim, rngs=rngs)
        self.linear2 = LinearLayer(self.dim, rngs=rngs)

        self.a = nnx.Param(jnr.uniform(rngs.params(), (2, dim)) * 0.01)

    def __call__(self, x, h):
        x = self.linear1(x, h)

        p, q = get_pq(x, self.dim)
        p = p + nnx.tanh(q) * self.a[0] * h
        x = get_x(p, q)

        x = self.linear1(x, h)

        p, q = get_pq(x, self.dim)
        q = q + nnx.tanh(p) * self.a[1] * h
        x = get_x(p, q)

        return x

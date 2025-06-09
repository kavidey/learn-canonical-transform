# Modified from: https://github.com/bentaps/strupnet/blob/main/strupnet/layers/G.py
import jax
import jax.numpy as jnp
import jax.random as jnr
from flax import nnx

from ..utils import get_pq, get_x

class G_Layer(nnx.Module):
    """Gradient symplectic module."""

    def __init__(self, dim, width, mode, *, rngs: nnx.Rngs):
        self.dim, self.width, self.mode = dim, width, mode

        self.activation = nnx.tanh

        self.K = nnx.Param(jnr.uniform(rngs.params(), (dim, width)) * 0.01)
        self.a = nnx.Param(jnr.uniform(rngs.params(), (width)) * 0.01)
        self.b = nnx.Param(jnr.uniform(rngs.params(), (width)) * 0.01)

    def __call__(self, x, h):
        p, q = get_pq(x, self.dim)

        odd_mode = lambda p, q: (p - h * (self.activation(q @ self.K + self.b * self.a)) @ self.K.T, q)
        even_mode = lambda p, q: (p, q - h * (self.activation(p @ self.K + self.b * self.a)) @ self.K.T)

        p, q = jax.lax.cond(self.mode % 2, odd_mode, even_mode, p, q)

        return get_x(p, q)

    def hamiltonian(self, x):
        p, q = get_pq(x, self.dim)

        integral_tanh = lambda x: jnp.log(jnp.cosh(x ** 2))
        odd_mode = lambda p, q: integral_tanh(q @ self.K + self.b) @ self.a
        even_mode = lambda p, q: integral_tanh(p @ self.K + self.b) @ self.a

        return jax.lax.cond(self.mode % 2, odd_mode, even_mode, p, q)
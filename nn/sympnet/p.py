# Modified from: https://github.com/bentaps/strupnet/blob/main/strupnet/layers/P.py
import jax
import jax.numpy as jnp
import jax.random as jnr
from flax import nnx

from ..utils import get_pq, get_x, canonical_symplectic_matrix, symplectic_matrix_transformation_2d

class P_Layer(nnx.Module):
    def __init__(self, dim, min_degree=None, max_degree=4, keepdim=False, *, rngs: nnx.Rngs):
        self.dim, self.max_degree = dim, max_degree
        self.min_degree = min_degree or 2

        self.a = nnx.Param(jnr.uniform(rngs.params(), (self.max_degree - self.min_degree + 1,)))
        self.w_dim = dim if keepdim else 2*dim
        self.w = nnx.Param(jnr.uniform(rngs.param(), self.w_dim))
    
    def __call__(self, x, h, i=None):
        monomial = (x @ self.w)[..., None]
        
        polynomial_derivative = jnp.sum(jax.vmap(
            lambda i: i * self.a[i-self.min_degree] * jnp.pow(monomial, i-1))(
                jnp.arange(self.min_degree, self.max_degree+1
        )), axis=0)
        # symp_weight = jax.lax.cond(i is None, canonical_symplectic_matrix, lambda w: symplectic_matrix_transformation_2d(self.w_dim, w, i), self.w)
        symp_weight = canonical_symplectic_matrix(self.dim)

        print(monomial.shape, polynomial_derivative.shape, symp_weight.shape, x.shape)

        x = x + h * polynomial_derivative * symp_weight
        return x
    
    def hamiltonian(self, x):
        """Returns the sub-hamiltonian of the layer"""
        monomial = jnp.sum(x * self.w)
        polynomial = jnp.sum(jax.vmap(
            lambda i: self.a[i-self.min_degree] * jnp.pow(monomial, i))(
                jnp.arange(self.min_degree, self.max_degree+1
        )))

        return polynomial
# %%
import matplotlib.pyplot as plt
from sklearn import datasets, manifold
import numpy as np
rng = np.random.default_rng(12345)

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
# %%
# Diffusion map implementation modified from: https://www.kaggle.com/code/rahulrajpl/diffusion-map-for-manifold-learning and https://github.com/sgh14/diffusion-maps-with-nystrom/blob/master/src/diffusionmaps/diffusion_maps.py

def pairwise_distance(x, y, dist_fn=lambda a, b: jnp.linalg.norm(a-b)):
    dist_vec = jax.vmap(lambda a: jax.vmap(lambda b: dist_fn(a, b))(y))(x)
    return dist_vec

def normalize(M, d_i, d_j, alpha=0.0):
    '''
    Performs alpha-normalization on a matrix M using degree vectors.

    M_norm[i, j] = M[i, j] / (d_i[i]**alpha * d_j[j]**alpha)
    '''
    d_i_alpha = jnp.power(d_i, alpha)
    d_j_alpha = jnp.power(d_j, alpha)
    M_alpha = M / jnp.outer(d_i_alpha, d_j_alpha)

    return M_alpha

def spectral_decomposition(A):
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    # eigh returns eigenvalues in ascending order, so we reverse the order
    order = jnp.argsort(jnp.abs(eigenvalues))[::-1]

    # Sort eigenvalues and eigenvectors in descending order
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors

    def fix_vector(v):
        first_nonzero = jnp.nonzero(v)[0]
        return v * jax.lax.cond(v[first_nonzero] < 0, lambda: -1, lambda: 1)
    return eigenvalues, jax.vmap(fix_vector)(eigenvectors)

def fit_diffusion(X, sigma=1.0, alpha=0.0):
    '''
    sigma : float
        The scale parameter for the RBF (Gaussian) kernel. It controls the
        locality of the connections in the graph. Corresponds to the `sigma`
        in `exp(-||x-y||^2 / (2 * sigma^2))`.
    alpha : float, default=0.0
        Normalization parameter for the kernel matrix.
        - `alpha = 0`: Standard diffusion maps (row-stochastic P).
        - `alpha = 0.5`: Fokker-Planck diffusion.
        - `alpha = 1`: Laplace-Beltrami diffusion (requires manifold assumption).
        Controls the influence of data density on the diffusion process.
    '''
    gamma = 1 / (2 * sigma**2)

    # Compute RBF Kernel
    dists = pairwise_distance(X, X)
    K = jnp.exp(-gamma * dists**2)
    
    # Compute degree vector
    d_K = jnp.sum(K, axis=1)
    
    # Compute normalized kernel
    W = normalize(K, d_K, d_K, alpha)

    # Compute the degree vector
    d_W = jnp.sum(W, axis=1)

    # Compute stationary distribution
    pi = d_W / jnp.sum(d_W)

    # Compute the matrix A
    A = normalize(W, d_W, d_W, 0.5)

    # Get the eigenvectors and eigenvalues
    lambdas, phis = spectral_decomposition(A)

    return lambdas, phis, pi

def reduce_dim_diffusion(steps, lambdas_red, phis_red, pi):
    # Compute P right eigenvectors (psis) from A eigenvectors (phis)
    psis = phis_red / jnp.sqrt(pi[:, None])
    
    # Compute the final embedding coordinates by scaling psis
    embedding = psis * jnp.power(lambdas_red[None, :], steps)

    return embedding

# def fit_new_data_diffusion(X_new, lambdas, phis, pi, A):
# %%
sr_points, sr_color = datasets.make_swiss_roll(n_samples=1000, random_state=0)
# Add a known amount of noise
sr_dirs = sr_points / jnp.linalg.norm(sr_points, axis=1, keepdims=True)
radial_noise = rng.normal(0, 0.5, (sr_points.shape[0],1))
# sr_points += sr_dirs * radial_noise

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=50, alpha=0.8
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
_ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)
# %%
lambdas, phis, pi = fit_diffusion(jnp.array(sr_points), 2, 0)
plt.scatter(lambdas, np.arange(lambdas.shape[0]))
# %%
n_components = 2
sr_red = reduce_dim_diffusion(1, lambdas[1:n_components+1], phis[:, 1:n_components+1], pi)

plt.scatter(*sr_red.T, c=sr_color, alpha=0.8)
# %%

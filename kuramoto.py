# %%
import jax; jax.config.update("jax_enable_x64", True)
from jax import random
import jax.numpy as jnp
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import integrate_leapfrog, get_eqns_motion

from sklearn.decomposition import PCA
from sklearn import manifold
# %%
@jax.jit
def to_aa(X):
    N = X.shape[0]//2
    q = X[:N]
    p = X[N:]

    return jnp.concat(((q**2 + p**2)/2, jnp.arctan(q/p)))

@jax.jit
def from_aa(X):
    N = X.shape[0]//2
    I = jnp.sqrt(X[:N]*2)
    phi = X[N:]

    return jnp.concat((I*jnp.sin(phi), I*jnp.cos(phi)))

def action_angle_hamiltonian(X, t, omega, K0, N):
    I = X[:N]
    phi = X[N:]

    H = 0

    # sum_l I_l * omega_l
    H += jnp.average(I, weights=omega) * omega.sum()

    combs = jnp.array(list(product(range(N), range(N))))

    # (K0/N) * sum_l,m sqrt(I_l*I_m) * (I_m - I_l) * sin(phi_m - phi_l)
    I_m = I[combs[:,0]]
    I_l = I[combs[:,1]]
    phi_m = phi[combs[:,0]]
    phi_l = phi[combs[:,1]]
    H += K0 * jnp.average(jnp.sqrt(I_m * I_l) * (I_m - I_l) * jnp.sin(phi_m - phi_l))

    return H

def hamiltonian(X, t, omega, K0, N):
    q = X[:N]
    p = X[N:]

    H = 0.0

    # sum_l (omega_l/2) * (q_l^2 + p_l^2)
    H += jnp.sum((omega/2) * (q**2 + p**2))

    # (K0/4N) * sum_l,m (q_l*p_m - q_m*p_l) * (q_m^2 + p_m^2 - q_l^2 - p_m^2)
    # combs = jnp.array(list(product(range(N), range(N))))
    # q_m = q[combs[:,0]]
    # q_l = q[combs[:,1]]
    # p_m = p[combs[:,0]]
    # p_l = p[combs[:,1]]
    # H += -(K0/(4*N)) * jnp.sum((q_l*p_m - q_m*p_l) * (q_m**2 + p_m**2 - q_l**2 - p_l**2))

    return H

motion_func = jax.jit(get_eqns_motion(hamiltonian), static_argnums=4) # N needs to be a static argnum
# %%
K0 = 0.0
# omega = jnp.array([-2., -1., 4.]) # sum cannot be zero
omega = jnp.array([3., 4., 5.])
I0 = jnp.array([1., 1., 1.])
phi0 = jnp.array([1., 2., 3.])

# omega = jnp.array([4.])
# I0 = jnp.array([1.])
# phi0 = jnp.array([1.])

N = I0.shape[0]
t = jnp.linspace(0, 10, 1000)

y_c = integrate_leapfrog(motion_func, from_aa(jnp.concat((I0, phi0))), t, dt=0.001, args=(omega, K0, N))
# from scipy.integrate import odeint
# y_c = odeint(motion_func, from_aa(jnp.concat((I0, phi0))), t, args=(omega, K0, N,))
y = jax.vmap(to_aa)(y_c)
# y = y_c

E = jax.vmap(hamiltonian, in_axes=(0, None, None, None, None))(y_c, 0.0, omega, K0, N)

fig, axs = plt.subplots(1,3, figsize=(10,5))
axs[0].plot(t, y[:,:N])
axs[1].plot(t, y[:, N:])
axs[2].plot(t, abs((E[0] - E)/E[0]))
axs[2].set_yscale('log')
# %%
X = jnp.concat((sol.observables, jnp.gradient(sol.observables, axis=0)), axis=1)
X /= X.mean(axis=0)
X = X[::10]
X += random.uniform(random.PRNGKey(1), X.shape) * 0.0005
t = sol.ts[::10]
# %%
pca = PCA().fit(X[:2000])

found_combs = []
for i in reversed(range(1, 4)):
    # gamma_n = pca.components_[-i] / np.median(np.abs(pca.components_[-i]))
    # gamma_n = pca.components_[-i] / (jnp.max(jnp.abs(pca.components_[-i])) / 2)
    # gamma_n = gamma_n.round()

    gamma_n = pca.components_[-i]

    C_opt = gamma_n @ X.T
    C_opt_hat = C_opt / jnp.linalg.norm(gamma_n)
    plt.plot(t, C_opt_hat - C_opt_hat.mean(), label=r"$\hat C_{opt,"+str(i)+"}$")
plt.legend()
plt.ylim(-0.05, 0.05)
plt.axvline(x=t[2000], c='grey', linestyle='--')
# %%
params = {
    "n_neighbors": 30,
    "n_components": 6,
    "eigen_solver": "dense",
    "random_state": 0,
}

lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
S_standard = lle_standard.fit_transform(X[:2000])

lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
S_ltsa = lle_ltsa.fit_transform(X[:2000])

lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
S_hessian = lle_hessian.fit_transform(X[:2000])

lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
S_mod = lle_mod.fit_transform(X[:2000])

isomap = manifold.Isomap(n_neighbors=30, n_components=6, p=1)
S_isomap = isomap.fit_transform(X[:2000])

# t_sne = manifold.TSNE(
#     n_components=6,
#     perplexity=30,
#     init="random",
#     max_iter=250,
#     random_state=0,
# )
# S_t_sne = t_sne.fit_transform(X[:2000])

# spectral = manifold.SpectralEmbedding(n_components=6, n_neighbors=30, random_state=42)
# S_spectral = spectral.fit_transform(X[:2000])
# %%
fig, axs = plt.subplots(2, 3)
axs = axs.flatten()
for i in reversed(range(1, 4)):
    axs[0].plot(t, lle_standard.transform(X)[:,-i])
    axs[0].axvline(x=t[2000], c='grey', linestyle='--')
    axs[0].set_title("standard")

    axs[0].plot(t, lle_ltsa.transform(X)[:,-i])
    axs[1].axvline(x=t[2000], c='grey', linestyle='--')
    axs[1].set_title("ltsa")

    axs[2].plot(t, lle_hessian.transform(X)[:,-i])
    axs[2].set_title("hessian")
    axs[2].axvline(x=t[2000], c='grey', linestyle='--')

    axs[3].plot(t, lle_mod.transform(X)[:, -i])
    axs[3].set_title("mod")
    axs[3].axvline(x=t[2000], c='grey', linestyle='--')

    axs[4].plot(t, isomap.transform(X)[:, -i])
    axs[4].set_title("isomap")
    axs[4].axvline(x=t[2000], c='grey', linestyle='--')

    # axs[5].plot(t, S_spectral[:, -i])
    # axs[5].set_title("spectral")
    # axs[5].axvline(x=t[2000], c='grey', linestyle='--')
plt.tight_layout()
# %%
fig, axs = plt.subplots(1,2, figsize=(8,5))
axs[0].plot(t, S_spectral[:, 3:])
axs[1].plot(t, S_spectral[:, :3])
# %%

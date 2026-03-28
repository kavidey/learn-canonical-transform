# %%
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib.pyplot as plt

from utils import integrate_leapfrog, get_eqns_motion

from scipy.integrate import odeint
# %%
def hamiltonian(X, t, N):
    q = X[:N]
    p = X[N:]

    H = 0

    # sum_l (omega_l/2) * (q_l^2 + p_l^2)
    H += (1/2) * jnp.average(q**2 + p**2) * N

    return H

motion_func = jax.jit(get_eqns_motion(hamiltonian), static_argnums=2)
# %%
N = 2
t = jnp.linspace(0, 10, 1000)

y = integrate_leapfrog(motion_func, jnp.array([2,3,-4,-5]), t, dt=0.001, args=(N,))
# y = odeint(motion_func, jnp.array([2,3,-4,-5]), t, args=(N,))

E = jax.vmap(hamiltonian, in_axes=(0, None, None))(y, 0.0, N)

fig, axs = plt.subplots(1,3, figsize=(10,5))
axs[0].plot(t, y[:,:N])
axs[1].plot(t, y[:, N:])
axs[2].plot(t, abs((E[0] - E)/E[0]))
axs[2].set_yscale('log')
# %%
plt.plot(y[:, 0]**2 + y[:, 2]**2)
plt.plot(y[:, 1]**2 + y[:, 3]**2)
# dp/dt = f(q)
# dq/dt = p
# %%

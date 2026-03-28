# %%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.experimental.ode import odeint
# %%
# https://ybarmaz.github.io/blog/posts/2021-12-04-hamiltonian-mechanics-with-jax.html
def poisson_bracket(f, g):
    return lambda q, p: (jnp.dot(grad(f, argnums=0)(q, p),
                                 grad(g, argnums=1)(q, p))
                        - jnp.dot(grad(f, argnums=1)(q, p),
                                  grad(g, argnums=0)(q, p)))
def position(q, p):
    return q

def momentum(q, p):
    return p
# %%
omega1 = 2
omega2 = 3
epsilon = 0.2

def hamiltonian(q, p):
    return omega1 * p[0] + omega2 * p[1] + epsilon * (jnp.cos(q[0] - q[1]))# + jnp.cos(2*q[0] - 3*q[1]))
# %%
n = 2
t = jnp.linspace(0., 10., 1001)
y0 = jnp.concatenate([jnp.array([0., 0.]), jnp.array([4., 5.])])

def make_hamiltonian_vector_field(n, H):
    """
    Returns a function suitable for odeint: dy/dt = f(y, t)
    where y is a flat array [q0, q1, ..., p0, p1, ...]
    """
    def vector_field(y, t):
        q = y[:n]
        p = y[n:]
        # Compute gradients with respect to q and p
        dH_dq = grad(H, argnums=0)(q, p)  # shape (n,)
        dH_dp = grad(H, argnums=1)(q, p)  # shape (n,)
        dqdt = dH_dp
        dpdt = -dH_dq
        return jnp.concatenate([dqdt, dpdt])  # shape (2n,)
    return vector_field

odefun = make_hamiltonian_vector_field(n, hamiltonian)

y = odeint(odefun, y0, t)
q_sol = y[:, :n]
p_sol = y[:, n:]

complex_sol = p_sol * jnp.exp(1j * q_sol)
# %%
fig, axs = plt.subplots(2,2)
axs[0,0].plot(jnp.real(complex_sol[:,0]), jnp.imag(complex_sol[:,0]))
axs[0,0].set_aspect('equal')
axs[1,0].plot(t, p_sol[:,0])

axs[0,1].plot(jnp.real(complex_sol[:,1]), jnp.imag(complex_sol[:,1]))
axs[0,1].set_aspect('equal')
axs[1,1].plot(t, p_sol[:,1])
# %%
plt.plot(t, p_sol[:,0], label="$I_1$")
plt.plot(t, p_sol[:,1], label="$I_2$")
# plt.plot(t, jnp.sin(t * (omega1-omega2) - jnp.pi/2))
plt.plot(t, p_sol[:,0]+p_sol[:,1], label="$J_1$")
plt.plot(t, 3*p_sol[:,0]+2*p_sol[:,1]-15, label="$J_2$")
plt.legend()
# %%

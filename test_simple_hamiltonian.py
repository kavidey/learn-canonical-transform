# %%
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.experimental.ode import odeint

import action_angle_tools

%config InlineBackend.figure_format = 'retina'
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
t = jnp.linspace(0., 100., 5001)
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

y = odeint(odefun, y0, t, atol=1e-10)
q_sol = y[:, :n]
p_sol = y[:, n:]

complex_sol = p_sol * jnp.exp(1j * q_sol)
# %%
fig, axs = plt.subplots(2,2)
axs[0,0].plot(jnp.real(complex_sol[:,0]), jnp.imag(complex_sol[:,0]))
axs[0,0].set_aspect('equal')
axs[0,0].set_xlim(-6,6)
axs[0,0].set_ylim(-6,6)
axs[1,0].plot(t, p_sol[:,0])
axs[1,0].set_ylim(0,6)

axs[0,1].plot(jnp.real(complex_sol[:,1]), jnp.imag(complex_sol[:,1]))
axs[0,1].set_aspect('equal')
axs[0,1].set_xlim(-6,6)
axs[0,1].set_ylim(-6,6)
axs[1,1].plot(t, p_sol[:,1])
axs[1,1].set_ylim(0,6)
# %%
# %%
fig, axs = plt.subplots(1,2, figsize=(8,3))
axs[0].plot(jnp.real(complex_sol[:,0]), jnp.imag(complex_sol[:,0]), linewidth=0.5)
axs[0].plot(jnp.real(complex_sol[:,1]), jnp.imag(complex_sol[:,1]), linewidth=0.5)
axs[0].set_aspect('equal')
axs[0].set_xlim(-6,6)
axs[0].set_ylim(-6,6)
axs[0].set_ylabel(r"$Im(X_n)$")
axs[0].set_xlabel(r"$Re(X_n)$")

axs[1].plot(t, p_sol[:,1], label=r"$(I_1,\theta_1)$")
axs[1].plot(t, p_sol[:,0], label=r"$(I_2,\theta_2)$")
axs[1].set_ylim(-0.2,6)
axs[1].set_ylabel("$J_n$")
axs[1].set_xlabel("$t$")
axs[1].legend()

plt.tight_layout()
plt.savefig("figs/simple_example.eps")
# %%
a = 2
b = 3
J1 = (b*p_sol[:,0] - a*p_sol[:,1])/(a+b)
J2 = (p_sol[:,0] + p_sol[:,1])/(a+b)
phi1 = q_sol[:,0] - q_sol[:,1]
phi2 = a*q_sol[:,0] + b*q_sol[:,1]
complex_sol_trans = jnp.array([J1*jnp.exp(1j*phi1), J2*jnp.exp(1j*phi2)]).T
# %%
fig, axs = plt.subplots(1,2, figsize=(8,3))
t_plot = t<10
axs[0].plot(jnp.real(complex_sol[t_plot,0]), jnp.imag(complex_sol[t_plot,0]), linewidth=0.5, c='grey', linestyle="--")
axs[0].plot(jnp.real(complex_sol[t_plot,1]), jnp.imag(complex_sol[t_plot,1]), linewidth=0.5, c='black', linestyle=":")
axs[0].plot(jnp.real(complex_sol_trans[t_plot,0]), jnp.imag(complex_sol_trans[t_plot,0]), linewidth=0.5)
axs[0].plot(jnp.real(complex_sol_trans[t_plot,1]), jnp.imag(complex_sol_trans[t_plot,1]), linewidth=0.5)
axs[0].set_aspect('equal')
axs[0].set_xlim(-6,6)
axs[0].set_ylim(-6,6)
axs[0].set_ylabel(r"$Im(X_n)$")
axs[0].set_xlabel(r"$Re(X_n)$")

axs[1].plot(t, p_sol[:,1], label=r"$(I_1,\theta_1)$", c='grey', linestyle="--")
axs[1].plot(t, p_sol[:,0], label=r"$(I_2,\theta_2)$", c='black', linestyle=":")
axs[1].plot(t, J1, label=r"$(J_1,\phi_1)$")
axs[1].plot(t, J2, label=r"$(J_2,\phi_2)$")
axs[1].set_ylim(-0.2,6)
axs[1].set_xlim(0,50)
axs[1].set_ylabel("$J_n$")
axs[1].set_xlabel("$t$")
axs[1].legend()

plt.savefig("figs/simple_example_analytical_sol.eps")
# %%
plt.plot(t, p_sol[:,0], label="$I_1$")
plt.plot(t, p_sol[:,1], label="$I_2$")
# plt.plot(t, jnp.sin(t * (omega1-omega2) - jnp.pi/2))
plt.plot(t, p_sol[:,0]+p_sol[:,1], label="$J_1$")
plt.plot(t, 3*p_sol[:,0]+2*p_sol[:,1]-15, label="$J_2$")
plt.legend()
# %%
action_fmft_calc = action_angle_tools.get_planet_fmft(["p_"+str(i) for i in range(2)], t, complex_sol.T, N=5, fmft_alg="default", display=True, scalar=1)
omega_vec_calc = np.array([list(action_fmft_calc["p_0"].items())[0][0], list(action_fmft_calc["p_1"].items())[0][0]])
omega_amp_calc = np.array([action_fmft_calc["p_0"][omega_vec_calc[0]], action_fmft_calc["p_1"][omega_vec_calc[1]]])
# %%
action_fmft = {
    "p_0": {
        2.0: y0[2] - epsilon,
        3.0: epsilon
    },
    "p_1": {
        3.0: y0[3] + epsilon,
        4.0: -epsilon
    }
}
omega_vec = np.array([2.0, 3.0])
omega_amp = [3.8, 5.2]

fig, axs = plt.subplots(1, 2)
for i in range(2):
    axs[i].plot(np.real(complex_sol.T[i]), np.imag(complex_sol.T[i]), linewidth=0.1, label="GT")
    
    fmft_recon_calc = np.sum([amp * np.exp(1j*freq*t) for freq,amp in list(action_fmft_calc["p_"+str(i)].items())],axis=0)
    axs[i].plot(np.real(fmft_recon_calc), np.imag(fmft_recon_calc), c='red', linewidth=0.1, label="Generated")

    fmft_recon = np.sum([amp * np.exp(1j*freq*t) for freq,amp in list(action_fmft["p_"+str(i)].items())],axis=0)
    axs[i].plot(np.real(fmft_recon), np.imag(fmft_recon), c='black', linewidth=0.1, label="Manual")

    axs[i].set_aspect(1)
axs[0].legend()
plt.show()
# %%
fig, axs = plt.subplots(1, 2, figsize=(20,5))
for i in range(2):
    fmft_recon = np.sum([amp * np.exp(1j*freq*t) for freq,amp in list(action_fmft["p_"+str(i)].items())],axis=0)
    axs[i].plot(t, np.abs(fmft_recon), label='FMFT Recon', alpha=0.5, c='grey', linewidth=5)
    
    axs[i].plot(t, np.abs(complex_sol.T[i]), label='Action')
axs[0].legend()
plt.show()
# %%
psi_cancelled, trans_fns, combs = action_angle_tools.cancel_frequencies(
    complex_sol.T, t,
    omega_vec, omega_amp,
    iterations=[1,3], psi_planet_list=["p_0", "p_1"],
    # initial_fmft=action_fmft,
    n_fmft=5,
    omega_abs_thresh=1e-1, omega_pct_thresh=1e-1,
    fmft_scalar=1, debug=True, fmft_alg="default")
# %%
# action_fmft = action_angle_tools.get_planet_fmft(["p_"+str(i) for i in range(2)], t, psi_cancelled, N=5, fmft_alg="default", display=True, scalar=1)
fig, axs = plt.subplots(1, 2, figsize=(20,5))
for i in range(2):
    # fmft_recon = np.sum([amp * np.exp(1j*freq*t) for freq,amp in list(action_fmft["p_"+str(i)].items())[:3]],axis=0)
    # axs[i].plot(t, np.abs(fmft_recon), label='FMFT Recon', alpha=0.5, c='grey', linewidth=5)
    
    axs[i].plot(t, np.abs(complex_sol.T[i]), label='Original')
    
    axs[i].plot(t, np.abs(psi_cancelled[i]), label='Cancelled')
axs[0].legend()
plt.show()
# %%
plt.plot(np.abs(complex_sol.T[0]))
# plt.plot(complex_sol.T[0])
plt.plot(np.abs(complex_sol.T[0] - 0.053*3.8 * (complex_sol.T[1]/5.2)))
plt.plot(np.abs(psi_cancelled[0]))
# %%
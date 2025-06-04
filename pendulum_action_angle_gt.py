# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe, ellipkinc
from scipy.integrate import solve_ivp
# %%
F = 2
G = 3
# %%
def hamiltonian(Y):
    phi, p = Y
    return 0.5 * G * p**2 - F * np.cos(phi)

def pendulum(t, Y):
    phi, p = Y

    p_dot = -F * np.sin(phi)
    phi_dot = G * p

    return [phi_dot, p_dot]
# %%
sx = np.arange(0.1, 2.7, 0.5)
teval = np.linspace(0, 10, 1000)

sols = []

fig, ax = plt.subplots(1,1)
for x in sx:
    sol = solve_ivp(pendulum, [np.min(teval), np.max(teval)], [x,0.1], t_eval=teval, atol=1e-8, rtol=1e-8)
    sols.append(sol.y)
    ax.plot(*sol.y)

ax.set_aspect('equal')
sols = np.array(sols)
# %%
R = np.sqrt(F/G)

fig = plt.figure()

for sol in sols:
    phi, p = sol
    E = hamiltonian(sol)
    kappa = np.sqrt((1+E/F)/2)
    eta = np.arcsin(np.sin(0.5 * phi) / kappa)
    J = R * (8/np.pi) * (ellipe(kappa**2) - (1-kappa**2)*ellipk(kappa**2))
    theta = (np.pi/2) * (1/ellipk(kappa**2) * ellipkinc(eta, kappa**2))
    
    plt.subplot(2, 2, 1)
    plt.plot(teval, J)
    plt.xlabel("$t$")
    plt.ylabel(r"$J$")

    plt.subplot(2, 2, 3)
    plt.plot(teval, theta)
    plt.xlabel("$t$")
    plt.ylabel(r"$\theta$")

    plt.subplot(1, 2, 2)
    x = J * np.exp(1j * theta)
    plt.plot(np.real(x), np.imag(x))
    plt.gca().set_aspect('equal')
fig.tight_layout()
# %%
